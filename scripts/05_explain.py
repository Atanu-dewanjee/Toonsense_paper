"""Step 5: explainability before/after perturbation (run after 04_robustness.py).

  A. Modality reliance  - IG on fusion-head inputs -> text share of attribution, every sample, clean vs each
                          perturbation; GMU text-gate activation.
  B. Visual attention   - Grad-CAM on CLIP for every sample: share of CAM mass on rendered meme text, and
                          CAM stability (Pearson r) under spatially aligned visual perturbations.
  C. Token attribution  - IG through the text encoder for selected memes whose prediction flips under a
                          linguistic perturbation (clean vs perturbed text); HTML visualisation.
  D. Figures            - Grad-CAM(++) overlays clean vs perturbed.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.cv import run_cv  # noqa: E402
from toonsense.data import TASKS, classes_for, load_dataset, primary_granularity  # noqa: E402
from toonsense.encoders import free, has_features, load_encoder, load_features  # noqa: E402
from toonsense.explain import cam_overlay, gradcam, modality_ig, text_attention_ratio, token_ig  # noqa: E402
from toonsense.ocr import load_boxes, load_texts  # noqa: E402
from toonsense.perturb_image import perturb_image  # noqa: E402

FUSION = ["concat_lr", "concat_mlp", "gmu", "bilinear"]
ALIGNED = ["blur", "jpeg", "resize", "text_occlusion", "region_occlusion"]  # spatially aligned with the clean image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--granularity", default=None, help="default: the dataset's primary granularity")
    ap.add_argument("--cam-model", default="concat_mlp")
    ap.add_argument("--cam-severity", type=int, default=2)
    ap.add_argument("--skip-cam", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    torch.manual_seed(0)
    ds = load_dataset(cfg)
    ids = ds.ids
    paths = ds.frame["image_path"].tolist()
    texts = load_texts(cfg, ids, ds.gold)["text"].tolist()
    boxes = {sid: load_boxes(cfg, sid) for sid in ids}
    t_name, i_name = cfg.primary["text"], cfg.primary["image"]
    gran = args.granularity or primary_granularity(cfg)
    task_list = list(TASKS)
    fig_task = "sentiment" if "sentiment" in TASKS else task_list[0]
    steps = int(cfg.explain["ig_steps"])
    out = cfg.path("results") / "explain"
    out.mkdir(parents=True, exist_ok=True)
    ptexts = pd.read_csv(cfg.path("results") / "robustness" / "perturbed_texts.csv", dtype={"id": str})

    Xt = load_features(cfg, "text", t_name, "clean", ids)
    Xv = load_features(cfg, "image", i_name, "clean", ids)
    sevs = cfg.robustness["severities"]
    variants = {}
    for fam in cfg.robustness["text"]:
        for s in sevs:
            if has_features(cfg, "text", t_name, f"{fam}_s{s}"):
                variants[f"{fam}_s{s}"] = (load_features(cfg, "text", t_name, f"{fam}_s{s}", ids), None)
    for fam in cfg.robustness["image"]:
        for s in sevs:
            if has_features(cfg, "image", i_name, f"{fam}_s{s}"):
                variants[f"{fam}_s{s}"] = (None, load_features(cfg, "image", i_name, f"{fam}_s{s}", ids))
    print(f"{len(variants)} perturbation variants available")

    # Train fold heads once per task (seed 0) and keep them for all analyses.
    fold_heads = {}
    for task in TASKS:
        y = ds.labels(task, gran)
        C = len(classes_for(task, gran))
        oof, heads = run_cv(Xt, Xv, y, C, FUSION + ["text_mlp"], cfg.benchmark, cfg.head, 0, variants=variants, return_heads=True)
        fold_heads[task] = (y, C, oof, heads)
        print(f"fold heads trained: {task}", flush=True)

    # ---------------- A. modality reliance
    rows = []
    for task, (y, C, oof, heads) in fold_heads.items():
        for tr, te, hd in heads:
            for v in ["clean"] + list(variants):
                xt = Xt if v == "clean" or variants[v][0] is None else variants[v][0]
                xv = Xv if v == "clean" or variants[v][1] is None else variants[v][1]
                for m in FUSION:
                    h = hd[m]
                    pred = h.predict_proba(xt[te], xv[te]).argmax(1)
                    r = modality_ig(h, xt[te], xv[te], pred, steps=steps)
                    gate = np.full(len(te), np.nan)
                    if m == "gmu":
                        t_s, v_s = h._scale(xt[te], xv[te])
                        with torch.no_grad():
                            h.model(torch.from_numpy(t_s), torch.from_numpy(v_s))
                        gate = h.model.last_gate.mean(1).numpy()
                    for k, idx in enumerate(te):
                        rows.append({"task": task, "model": m, "variant": v, "id": ids[idx], "pred": int(pred[k]),
                                     "gold": int(y[idx]), "text_share": float(r["text_share"][k]), "gmu_text_gate": float(gate[k])})
    ms = pd.DataFrame(rows)
    ms.to_csv(out / "modality_share_samples.csv", index=False)
    agg = ms.groupby(["task", "model", "variant"]).agg(text_share=("text_share", "mean"), text_share_sd=("text_share", "std"),
                                                       gmu_text_gate=("gmu_text_gate", "mean")).reset_index()
    agg.to_csv(out / "modality_share.csv", index=False)
    print(agg[agg.variant == "clean"].pivot_table(index="model", columns="task", values="text_share").round(3).to_string())

    # ---------------- B. Grad-CAM statistics + D. figures
    if not args.skip_cam:
        enc_i = load_encoder(cfg, "image", i_name, device="cpu")
        cam_cap = int(cfg.explain.get("cam_max_per_task", 0) or 0)
        cam_rows, fig_items = [], []
        for task, (y, C, oof, heads) in fold_heads.items():
            # stratified subsample (per fold, proportional) to bound Grad-CAM cost on large datasets
            rng = np.random.default_rng(0)
            for tr, te, hd in heads:
                h = hd[args.cam_model]
                te_cam = te
                if cam_cap and len(y) > cam_cap:
                    k = max(1, round(len(te) * cam_cap / len(y)))
                    te_cam = np.sort(rng.choice(te, size=min(k, len(te)), replace=False))
                for idx in te_cam:
                    sid = ids[idx]
                    img = Image.open(paths[idx]).convert("RGB")
                    cam0, c0 = gradcam(enc_i, h, img, Xt[idx])
                    rec = {"task": task, "id": sid, "pred": c0, "gold": int(y[idx]), **text_attention_ratio(cam0, img.size, boxes[sid])}
                    for fam in ALIGNED:
                        pimg = perturb_image(img, fam, args.cam_severity, sid, boxes[sid])
                        cam1, c1 = gradcam(enc_i, h, pimg, Xt[idx], target=c0)
                        rec[f"r_{fam}"] = float(np.corrcoef(cam0.ravel(), cam1.ravel())[0, 1])
                        if task == fig_task and len(fig_items) < 4 and fam == ALIGNED[-1] and len(boxes[sid]) >= 3:
                            fig_items.append((sid, img, h, Xt[idx], c0, classes_for(task, gran)[c0]))
                    cam_rows.append(rec)
            print(f"grad-cam {task}: done", flush=True)
        cams = pd.DataFrame(cam_rows)
        cams.to_csv(out / "gradcam_samples.csv", index=False)
        summ = cams.drop(columns=["id", "pred", "gold"]).groupby("task").mean(numeric_only=True)
        summ.to_csv(out / "gradcam_summary.csv")
        print(summ.round(3).to_string())

        cols = ["clean"] + ALIGNED
        fig, axes = plt.subplots(len(fig_items), len(cols), figsize=(2.3 * len(cols), 2.5 * len(fig_items)), squeeze=False)
        for r, (sid, img, h, xt, c0, cname) in enumerate(fig_items):
            for c, fam in enumerate(cols):
                im = img if fam == "clean" else perturb_image(img, fam, args.cam_severity, sid, boxes[sid])
                cam, _ = gradcam(enc_i, h, im, xt, target=c0, method="gradcam++")
                base, cam = cam_overlay(im, cam)
                ax = axes[r, c]
                ax.imshow(base)
                ax.imshow(cam, cmap="jet", alpha=0.45)
                ax.set_xticks([]), ax.set_yticks([])
                if r == 0:
                    ax.set_title(fam.replace("_", " "), fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"#{sid}\n→ {cname}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "gradcam_examples.png", dpi=200)
        plt.close(fig)
        free(enc_i)

    # ---------------- C. token-level IG on prediction flips
    enc_t = load_encoder(cfg, "text", t_name, device="cpu")
    n_per_task = max(1, int(cfg.explain["n_examples"]) // min(3, len(task_list)))
    examples = []
    for task in task_list[:3]:
        y, C, oof, heads = fold_heads[task]
        chosen = 0
        for fam in ("translit_s3", "codemix_s3", "roman_noise_s3", "bangla_noise_s3"):
            if fam not in variants:
                continue
            pc, pp = oof["concat_mlp"]["clean"].argmax(1), oof["concat_mlp"][fam].argmax(1)
            for idx in np.where((pc != pp) & (pc == y))[0]:
                if chosen >= n_per_task or len(texts[idx].split()) < 3:
                    continue
                h = next(hd["concat_mlp"] for tr, te, hd in heads if idx in te)
                pert = ptexts[(ptexts.id == ids[idx]) & (ptexts.variant == fam)].text.iloc[0]
                a0 = token_ig(enc_t, h, texts[idx], Xv[idx], int(pc[idx]), steps)
                a1 = token_ig(enc_t, h, pert, Xv[idx], int(pc[idx]), steps)
                cls = classes_for(task, gran)
                examples.append({"task": task, "id": ids[idx], "perturbation": fam, "gold": cls[y[idx]], "pred_clean": cls[pc[idx]],
                                 "pred_pert": cls[pp[idx]], "clean": a0, "perturbed": a1})
                chosen += 1
    (out / "token_ig_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    write_token_html(examples, out / "token_ig_examples.html")
    print(f"token IG: {len(examples)} examples")


def _spans(attrs):
    m = max([abs(a) for _, a in attrs] + [1e-9])
    out = []
    for w, a in attrs:
        x = a / m
        col = f"rgba(214,39,40,{abs(x):.2f})" if x > 0 else f"rgba(31,119,180,{abs(x):.2f})"
        out.append(f'<span style="background:{col};padding:1px 3px;border-radius:3px" title="{a:.4f}">{html.escape(w)}</span>')
    return " ".join(out)


def write_token_html(examples, path: Path) -> None:
    parts = ["<!doctype html><meta charset='utf-8'><title>Token attributions</title>",
             "<style>body{font-family:'Noto Sans Bengali','Nirmala UI',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;line-height:2}"
             "h3{margin-bottom:0}.meta{color:#555;font-size:13px}</style>",
             "<h1>Integrated Gradients: clean vs perturbed text</h1><p>Red = supports the clean prediction, blue = opposes it.</p>"]
    for e in examples:
        parts.append(f"<h3>#{e['id']} · {e['task']} · {e['perturbation']}</h3>"
                     f"<div class='meta'>gold={e['gold']} · clean→{e['pred_clean']} · perturbed→{e['pred_pert']}</div>"
                     f"<p><b>clean:</b> {_spans(e['clean'])}</p><p><b>perturbed:</b> {_spans(e['perturbed'])}</p>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
