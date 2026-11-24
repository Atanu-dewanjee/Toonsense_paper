"""Frozen pretrained encoders and a per-(encoder, variant) embedding cache."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image

from .config import Config, get_device

torch.set_num_threads(max(1, torch.get_num_threads()))


class TextEncoder:
    def __init__(self, name: str, spec: dict, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.name, self.spec, self.device = name, spec, device
        self.tok = AutoTokenizer.from_pretrained(spec["hf_id"])
        self.model = AutoModel.from_pretrained(spec["hf_id"]).eval().to(device)
        self.max_len = int(spec.get("max_len", 256))
        self.prefix = spec.get("prefix", "")

    def tokenize(self, texts: list[str]):
        return self.tok(
            [self.prefix + (t if t.strip() else "<empty>") for t in texts],
            padding=True, truncation=True, max_length=self.max_len, return_tensors="pt",
        )

    def pool(self, out, attn: torch.Tensor) -> torch.Tensor:
        pooling = self.spec.get("pooling", "mean")
        if pooling == "cls":
            z = out.last_hidden_state[:, 0]
        elif pooling == "pooler" and getattr(out, "pooler_output", None) is not None:
            z = out.pooler_output
        else:
            m = attn.unsqueeze(-1).to(out.last_hidden_state.dtype)
            z = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
        if self.spec.get("normalize"):
            z = torch.nn.functional.normalize(z, dim=-1)
        return z

    def forward_embeds(self, inputs_embeds: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Differentiable path from input embeddings to pooled features (for Integrated Gradients)."""
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn)
        return self.pool(out, attn)

    @torch.inference_mode()
    def encode(self, texts: list[str], bs: int = 16) -> np.ndarray:
        feats = []
        for i in range(0, len(texts), bs):
            enc = {k: v.to(self.device) for k, v in self.tokenize(texts[i: i + bs]).items()}
            out = self.model(**enc)
            feats.append(self.pool(out, enc["attention_mask"]).float().cpu().numpy())
        return np.concatenate(feats).astype(np.float32)


def pad_to_square(im: Image.Image) -> Image.Image:
    """Letterbox to a square with the median border colour. CLIP-style processors otherwise centre-crop,
    which would cut the top/bottom panels of tall multi-panel comics."""
    w, h = im.size
    if w == h:
        return im
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(c) for c in np.median(border, axis=0))
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


class ImageEncoder:
    def __init__(self, name: str, spec: dict, device: str = "cpu"):
        from transformers import AutoImageProcessor

        self.name, self.spec, self.kind, self.device = name, spec, spec.get("kind", "clip"), device
        self.proc = AutoImageProcessor.from_pretrained(spec["hf_id"])
        if self.kind == "clip":
            from transformers import CLIPVisionModelWithProjection

            self.model = CLIPVisionModelWithProjection.from_pretrained(spec["hf_id"]).eval().to(device)
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(spec["hf_id"]).eval().to(device)

    def prepare(self, im: Image.Image) -> Image.Image:
        im = im.convert("RGB")
        return pad_to_square(im) if self.spec.get("pad_square", True) else im

    def pixel_values(self, images: list[Image.Image]) -> torch.Tensor:
        return self.proc(images=[self.prepare(im) for im in images], return_tensors="pt")["pixel_values"]

    def forward_pixels(self, pv: torch.Tensor) -> torch.Tensor:
        if self.kind == "clip":
            return self.model(pixel_values=pv).image_embeds
        h = self.model(pixel_values=pv).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)

    @torch.inference_mode()
    def encode(self, images: Iterable[Image.Image], bs: int = 16) -> np.ndarray:
        images = list(images)
        feats = []
        for i in range(0, len(images), bs):
            feats.append(self.forward_pixels(self.pixel_values(images[i: i + bs]).to(self.device)).float().cpu().numpy())
        return np.concatenate(feats).astype(np.float32)


def load_encoder(cfg: Config, modality: str, name: str, device: str | None = None):
    spec = dict(cfg.encoders[modality][name])
    device = device or get_device(cfg)
    return TextEncoder(name, spec, device) if modality == "text" else ImageEncoder(name, spec, device)


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ cache
def feature_path(cfg: Config, modality: str, encoder: str, variant: str) -> Path:
    return cfg.path("features") / modality / encoder / f"{variant}.npz"


def has_features(cfg: Config, modality: str, encoder: str, variant: str) -> bool:
    return feature_path(cfg, modality, encoder, variant).exists()


def save_features(cfg: Config, modality: str, encoder: str, variant: str, ids: list[str], X: np.ndarray) -> None:
    p = feature_path(cfg, modality, encoder, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, ids=np.array(ids), X=X)


def load_features(cfg: Config, modality: str, encoder: str, variant: str, ids: list[str]) -> np.ndarray:
    d = np.load(feature_path(cfg, modality, encoder, variant), allow_pickle=False)
    pos = {s: i for i, s in enumerate(d["ids"].tolist())}
    missing = [s for s in ids if s not in pos]
    if missing:
        raise KeyError(f"{modality}/{encoder}/{variant}: missing {len(missing)} ids, e.g. {missing[:3]}")
    return d["X"][[pos[s] for s in ids]]


def ensure_features(
    cfg: Config, modality: str, encoder, variant: str, ids: list[str], make_inputs: Callable[[], list]
) -> None:
    """Compute and cache features for `variant` if absent. `encoder` may be a loaded encoder or a name."""
    if has_features(cfg, modality, encoder if isinstance(encoder, str) else encoder.name, variant):
        return
    enc = load_encoder(cfg, modality, encoder) if isinstance(encoder, str) else encoder
    X = enc.encode(make_inputs())
    save_features(cfg, modality, enc.name, variant, ids, X)
