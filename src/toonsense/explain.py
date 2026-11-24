"""Explainability: modality-level IG, token-level IG through the text encoder, Grad-CAM(++) on CLIP ViT."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .encoders import ImageEncoder, TextEncoder, pad_to_square
from .heads import Head


# ------------------------------------------------------------------ modality-level IG
def modality_ig(head: Head, Xt: np.ndarray, Xv: np.ndarray, target: np.ndarray, steps: int = 32) -> dict[str, np.ndarray]:
    """Integrated Gradients w.r.t. the raw fused input [t ; v] with the training-mean baseline.

    Returns per-sample L1 attribution mass of each modality and the text share in [0, 1].
    """
    t = torch.tensor(Xt, dtype=torch.float32)
    v = torch.tensor(Xv, dtype=torch.float32)
    bt = torch.tensor(head.st.mean_, dtype=torch.float32).expand_as(t)
    bv = torch.tensor(head.sv.mean_, dtype=torch.float32).expand_as(v)
    gt, gv = torch.zeros_like(t), torch.zeros_like(v)
    tgt = torch.tensor(target).long()
    if hasattr(head.model, "eval") and not hasattr(head.model, "coef_"):
        head.model.eval()
    for a in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:  # midpoint Riemann sum
        ti = (bt + a * (t - bt)).requires_grad_(True)
        vi = (bv + a * (v - bv)).requires_grad_(True)
        logit = head.torch_logits(ti, vi).gather(1, tgt[:, None]).sum()
        dt, dv = torch.autograd.grad(logit, [ti, vi])
        gt += dt / steps
        gv += dv / steps
    at = ((t - bt) * gt).abs().sum(1).numpy()
    av = ((v - bv) * gv).abs().sum(1).numpy()
    return {"attr_text": at, "attr_image": av, "text_share": at / np.maximum(at + av, 1e-12)}


# ------------------------------------------------------------------ token-level IG
def token_ig(enc: TextEncoder, head: Head, text: str, v_raw: np.ndarray | None, target: int, steps: int = 32):
    """IG over input token embeddings (pad-token baseline, special tokens kept). Returns [(word, attribution)]."""
    batch = enc.tokenize([text])
    ids, attn = batch["input_ids"], batch["attention_mask"]
    emb_layer = enc.model.get_input_embeddings()
    with torch.no_grad():
        x = emb_layer(ids)
        pad_id = enc.tok.pad_token_id if enc.tok.pad_token_id is not None else 0
        special = torch.tensor(enc.tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True)).bool()
        base_ids = ids.clone()
        base_ids[0, ~special] = pad_id
        b = emb_layer(base_ids)
    v = None if v_raw is None else torch.tensor(v_raw[None], dtype=torch.float32)
    grads = torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    chunk = 8
    for k in range(0, steps, chunk):
        a = alphas[k:k + chunk].view(-1, 1, 1)
        xi = (b + a * (x - b)).requires_grad_(True)
        feats = enc.forward_embeds(xi, attn.expand(len(a), -1))
        logits = head.torch_logits(feats, None if v is None else v.expand(len(a), -1))
        g, = torch.autograd.grad(logits[:, target].sum(), xi)
        grads += g.sum(0, keepdim=True) / steps
    attr = ((x - b) * grads).sum(-1)[0].detach().numpy()
    toks = enc.tok.convert_ids_to_tokens(ids[0])
    return merge_subwords(toks, attr, special.numpy())


def merge_subwords(tokens: list[str], attr: np.ndarray, special: np.ndarray) -> list[tuple[str, float]]:
    """Sum sub-word attributions into words for SentencePiece/BPE ('▁'/'Ġ' word starts) or WordPiece ('##')."""
    wordpiece = tokens[0] == "[CLS]"
    words: list[list] = []
    for tok, a, sp in zip(tokens, attr, special):
        if sp:
            continue
        cont = tok.startswith("##") if wordpiece else not tok.startswith(("▁", "Ġ"))
        if cont and words:
            words[-1][0] += tok[2:] if wordpiece else tok
            words[-1][1] += float(a)
        else:
            words.append([tok if wordpiece else tok.lstrip("▁Ġ"), float(a)])
    return [(w, a) for w, a in words if w]


# ------------------------------------------------------------------ Grad-CAM on CLIP ViT
def gradcam(enc: ImageEncoder, head: Head, image: Image.Image, t_raw: np.ndarray | None, target: int | None = None,
            method: str = "gradcam") -> tuple[np.ndarray, int]:
    """Grad-CAM / Grad-CAM++ over ViT patch tokens at the input of the last transformer block.

    Returns (cam on the 224x224 padded-square model input, normalised to [0,1], predicted/target class).
    """
    if enc.kind != "clip":
        raise NotImplementedError("Grad-CAM implemented for CLIP ViT encoders")
    layer = enc.model.vision_model.encoder.layers[-1]
    store = {}

    def hook(_m, inp):
        store["act"] = inp[0]
        inp[0].retain_grad()

    h = layer.register_forward_pre_hook(hook)
    try:
        pv = enc.pixel_values([image]).requires_grad_(False)
        feats = enc.model(pixel_values=pv).image_embeds
        t = None if t_raw is None else torch.tensor(t_raw[None], dtype=torch.float32)
        logits = head.torch_logits(t, feats)
        if target is None:
            target = int(logits.argmax(1))
        enc.model.zero_grad()
        logits[0, target].backward()
        A = store["act"][0, 1:].detach()        # (P, D) patch tokens
        G = store["act"].grad[0, 1:].detach()   # (P, D)
    finally:
        h.remove()
    if method == "gradcam++":
        g2, g3 = G.pow(2), G.pow(3)
        alpha = g2 / (2 * g2 + A.sum(0, keepdim=True) * g3 + 1e-8)
        w = (alpha * F.relu(G)).sum(0)
    else:
        w = G.mean(0)
    cam = F.relu((A * w).sum(-1))
    g = int(round(cam.numel() ** 0.5))
    cam = cam.view(1, 1, g, g)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
    return cam, target


def square_geometry(size: tuple[int, int]) -> tuple[int, int, int]:
    w, h = size
    s = max(w, h)
    return s, (s - w) // 2, (s - h) // 2


def boxes_mask_224(size: tuple[int, int], boxes: list[list[float]]) -> np.ndarray:
    """Binary mask of OCR text boxes in the 224x224 padded-square model frame."""
    s, ox, oy = square_geometry(size)
    k = 224 / s
    m = np.zeros((224, 224), bool)
    for x0, y0, x1, y1 in boxes:
        m[int((y0 + oy) * k): int(np.ceil((y1 + oy) * k)), int((x0 + ox) * k): int(np.ceil((x1 + ox) * k))] = True
    return m


def content_mask_224(size: tuple[int, int]) -> np.ndarray:
    s, ox, oy = square_geometry(size)
    k = 224 / s
    m = np.zeros((224, 224), bool)
    w, h = size
    m[int(oy * k): int(np.ceil((oy + h) * k)), int(ox * k): int(np.ceil((ox + w) * k))] = True
    return m


def text_attention_ratio(cam: np.ndarray, size, boxes) -> dict[str, float]:
    """CAM mass inside text boxes vs. the boxes' share of the image area (ratio > 1: over-attends text)."""
    tm, cm = boxes_mask_224(size, boxes), content_mask_224(size)
    mass = cam[cm].sum()
    in_text = cam[tm & cm].sum() / mass if mass > 0 else np.nan
    area = (tm & cm).sum() / max(cm.sum(), 1)
    return {"cam_text_mass": float(in_text), "text_area": float(area), "text_ratio": float(in_text / area) if area > 0 else np.nan}


def cam_overlay(image: Image.Image, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (padded-square RGB image at 224, cam) for plotting."""
    sq = pad_to_square(image.convert("RGB")).resize((224, 224), Image.BICUBIC)
    return np.asarray(sq), cam
