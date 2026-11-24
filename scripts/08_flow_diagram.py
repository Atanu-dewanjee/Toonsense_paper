"""Methodology flow diagram for the paper (paper/figures/methodology_flow.{pdf,png})."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
INK, INK2 = "#0b0b0b", "#52514e"
# light tints of the reference palette: blue = text, orange = image, aqua = fusion, grey = evaluation
TEXT_F, TEXT_E = "#e3eefb", "#2a78d6"
IMG_F, IMG_E = "#fde9e0", "#eb6834"
FUS_F, FUS_E = "#dcf3ea", "#1baf7a"
EVA_F, EVA_E = "#f0efec", "#52514e"
DATA_F, DATA_E = "#ffffff", "#0b0b0b"


def box(ax, x, y, w, h, title, sub="", fc="#fff", ec=INK, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=fs, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.062, sub, ha="center", va="top", fontsize=fs - 1.3, color=INK2, linespacing=1.3)


def arrow(ax, a, b, ls="-", color=INK2, rad=0.0, head=True):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if head else "-", mutation_scale=10, lw=1.0, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "Benchmark encoders  —  text: XLM-R-base, XLM-R-large, mBERT, BanglaBERT, LaBSE, mE5-base"
            "   |   image: CLIP ViT-B/32, CLIP ViT-B/16, CLIP ViT-L/14, DINOv2-base",
            fontsize=7.6, color=INK2, ha="center", va="center")

    h = 0.17
    yt, yi = 0.73, 0.29
    # input
    box(ax, 0.005, 0.44, 0.105, 0.24, "Meme", "CMBan (133)\nBanglaAbuseMeme\n(4,043)\nMemoSen (4,368)\n\nBangla, English\nor code-mixed", DATA_F, DATA_E)
    # text lane
    box(ax, 0.145, yt, 0.13, h, "OCR", "EasyOCR (bn + en)\nconfidence ≥ 0.10\n(MemoSen: captions)", TEXT_F, TEXT_E)
    box(ax, 0.305, yt, 0.175, h, "Linguistic perturbation",
        "transliteration · Roman. noise\nBangla noise · code-mixing\nemoji add / remove  (s = 1–3)", TEXT_F, TEXT_E)
    box(ax, 0.51, yt, 0.17, h, "Text encoder (frozen)", "XLM-RoBERTa-base\n278M params · mean pooling\n" r"$t \in \mathbb{R}^{768}$",
        TEXT_F, TEXT_E)
    # image lane
    box(ax, 0.145, yi, 0.13, h, "Letterbox", "pad to square with\nmedian border colour\n(no centre crop)", IMG_F, IMG_E)
    box(ax, 0.305, yi, 0.175, h, "Visual perturbation",
        "blur · crop · down-sampling\nJPEG · text-region occlusion\nregion occlusion  (s = 1–3)", IMG_F, IMG_E)
    box(ax, 0.51, yi, 0.17, h, "Image encoder (frozen)", "CLIP ViT-B/32\n87M params · projected emb.\n" r"$v \in \mathbb{R}^{512}$",
        IMG_F, IMG_E)
    # fusion + output
    box(ax, 0.715, 0.44, 0.13, 0.24, "Fusion head",
        "z-score (train fold)\n\nConcat-LR\nConcat-MLP · GMU\nBilinear · Late fusion", FUS_F, FUS_E)
    box(ax, 0.87, 0.44, 0.125, 0.24, "Predictions", "one head per task\n\nCMBan: 5 tasks\nBAM: 4 tasks\nMemoSen: 1 task", FUS_F, FUS_E)

    # main arrows
    arrow(ax, (0.11, 0.60), (0.145, yt + h / 2), rad=-0.25)
    arrow(ax, (0.11, 0.52), (0.145, yi + h / 2), rad=0.25)
    for x0, x1 in [(0.275, 0.305), (0.48, 0.51)]:
        arrow(ax, (x0, yt + h / 2), (x1, yt + h / 2))
        arrow(ax, (x0, yi + h / 2), (x1, yi + h / 2))
    arrow(ax, (0.68, yt + h / 2), (0.715, 0.63), rad=-0.2)
    arrow(ax, (0.68, yi + h / 2), (0.715, 0.49), rad=0.2)
    arrow(ax, (0.845, 0.56), (0.87, 0.56))
    # OCR boxes reused for text-region occlusion
    arrow(ax, (0.21, yt), (0.36, yi + h), ls=(0, (3, 2)), color=TEXT_E)
    ax.text(0.33, 0.60, "OCR boxes →\ntext-region occlusion", fontsize=6.8, color=TEXT_E, ha="left", va="center")
    ax.text(0.595, 0.60, "embeddings cached per\n(encoder, clean / perturbed)", fontsize=6.8, color=INK2,
            ha="center", va="center", style="italic")

    # evaluation band, fed by a distribution bus from the predictions
    ye, he = 0.02, 0.15
    w, gap, x0 = 0.183, 0.0175, 0.0125
    bus = 0.215
    arrow(ax, (0.9325, 0.44), (0.9325, bus), head=False)
    arrow(ax, (x0 + w / 2, bus), (0.9325, bus), head=False)
    items = [("Clean evaluation", "Acc · macro P / R / F1\n5-fold CV × 3 seeds\nbootstrap 95% CI"),
             ("Robustness analysis", "Robustness Drop (abs / rel)\nflip rate\nby language group"),
             ("Explainability", "Integrated Gradients\n(modality & token)\nGrad-CAM / Grad-CAM++"),
             ("Error analysis", "linguistic · visual\ncross-modal · OCR-limited\ncontextual / cultural"),
             ("Mitigation", "Romanisation augmentation\ntransliterated training\ncopies, same labels")]
    for k, (t, sub) in enumerate(items):
        x = x0 + k * (w + gap)
        box(ax, x, ye, w, he, t, sub, EVA_F, EVA_E)
        arrow(ax, (x + w / 2, bus), (x + w / 2, ye + he))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "methodology_flow.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "methodology_flow.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(OUT / "methodology_flow.png")


if __name__ == "__main__":
    main()
