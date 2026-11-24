"""Vertical methodology flowchart in the style of the project proposal (dark background, centred stages).

    python scripts/12_flow_vertical.py
writes paper/figures/methodology_flow_vertical.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
BG, FG, SUB, LINE = "#1c1c1c", "#ffffff", "#d6d6d6", "#bdbdbd"
TITLE_FS, SUB_FS = 12.5, 9.2
H = 12.9                     # figure height (in); all vertical spacings below are scaled by 11.6 / H
K = 11.6 / H

# (title, subtitle lines) for the single-column stages, top to bottom
TOP = [
    ("Datasets", ["CMBan (133 labelled memes)  ·  BanglaAbuseMeme (4,043)  ·  MemoSen (4,368)",
                  "Bangla + English + Code-mixed"]),
    ("Data Preparation", ["Text: EasyOCR (Bangla + English); MemoSen: human captions",
                          "Image: letterbox to square (no centre crop)"]),
]
ENCODERS = [
    ("Text Encoder (frozen)", ["XLM-R-base  (primary)", "Benchmark: XLM-R-large, mBERT,", "BanglaBERT, LaBSE, mE5-base"]),
    ("Image Encoder (frozen)", ["CLIP ViT-B/32  (primary)", "Benchmark: CLIP ViT-B/16,", "CLIP ViT-L/14, DINOv2-base"]),
]
BOTTOM = [
    ("Feature Fusion", ["Concat-LR  /  Concat-MLP  /  GMU  /  Bilinear  /  Late fusion"]),
    ("Classifier", ["One head per task  (CMBan 5  ·  BanglaAbuseMeme 4  ·  MemoSen 1)"]),
    ("Original Evaluation", ["Accuracy / Precision / Recall / Macro-F1",
                             "5-fold CV × 3 seeds  ·  bootstrap 95% CI  ·  Holm-corrected tests"]),
    ("Linguistic Perturbations", ["Transliteration / Romanised & Bangla spelling noise / code-mixing / emoji  (s = 1–3)"]),
    ("Visual Perturbations", ["Blur / crop / down-sampling / JPEG / text & region occlusion  (s = 1–3)"]),
    ("Robustness Analysis", ["Original vs. perturbed: Robustness Drop (abs / rel), flip rate, by language group"]),
    ("Explainability", ["Integrated Gradients (modality & token)  ·  Grad-CAM / Grad-CAM++"]),
    ("Error Analysis", ["Linguistic / visual / cross-modal / OCR-limited / contextual-cultural"]),
    ("Mitigation", ["Romanisation augmentation of the training data"]),
]


def stage(ax, x, y, title, lines):
    """Draw a stage centred at (x, y = top of title). Returns the y of the bottom of its last line."""
    ax.text(x, y, title, ha="center", va="top", color=FG, fontsize=TITLE_FS, fontweight="bold")
    yy = y - 0.024 * K
    for ln in lines:
        ax.text(x, yy, ln, ha="center", va="top", color=SUB, fontsize=SUB_FS)
        yy -= 0.0175 * K
    return yy


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=LINE, linewidth=1.2,
                                 shrinkA=0, shrinkB=0))


def main() -> None:
    fig = plt.figure(figsize=(7.6, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.03, 0.012), 0.94, 0.976, fill=False, edgecolor=LINE, linewidth=0.9))

    gap, alen = 0.010 * K, 0.022 * K  # space around arrows, arrow length
    y = 0.965
    for k, (t, lines) in enumerate(TOP):
        bottom = stage(ax, 0.5, y, t, lines)
        if k < len(TOP) - 1:
            arrow(ax, (0.5, bottom - gap + 0.006 * K), (0.5, bottom - gap - alen + 0.006 * K))
            y = bottom - 2 * gap - alen + 0.004 * K

    # split into the two encoder branches
    xl, xr = 0.27, 0.73
    top_split = bottom - gap + 0.006 * K
    arrow(ax, (0.44, top_split), (xl + 0.06, top_split - alen - 0.004 * K))
    arrow(ax, (0.56, top_split), (xr - 0.06, top_split - alen - 0.004 * K))
    y = top_split - alen - 2 * gap
    bl = stage(ax, xl, y, *ENCODERS[0])
    br = stage(ax, xr, y, *ENCODERS[1])
    bottom = min(bl, br)

    # merge into fusion
    merge_top = bottom - gap + 0.006 * K
    arrow(ax, (xl + 0.06, merge_top), (0.44, merge_top - alen - 0.004 * K))
    arrow(ax, (xr - 0.06, merge_top), (0.56, merge_top - alen - 0.004 * K))
    y = merge_top - alen - 2 * gap
    for k, (t, lines) in enumerate(BOTTOM):
        bottom = stage(ax, 0.5, y, t, lines)
        if k < len(BOTTOM) - 1:
            arrow(ax, (0.5, bottom - gap + 0.006 * K), (0.5, bottom - gap - alen + 0.006 * K))
            y = bottom - 2 * gap - alen + 0.004 * K
    if bottom < 0.02:
        raise SystemExit(f"layout overflow: bottom at {bottom:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "methodology_flow_vertical.png", dpi=300, facecolor=BG)
    fig.savefig(OUT / "methodology_flow_vertical.pdf", facecolor=BG)
    print(OUT / "methodology_flow_vertical.png", f"(bottom margin {bottom:.3f})")


if __name__ == "__main__":
    main()
