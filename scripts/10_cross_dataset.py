"""Cross-dataset summary tables and figure for the paper.

Reads the per-dataset outputs of steps 03-06 (``results/<dataset>/...``) for every dataset that has them and writes
compact comparison tables to ``paper/tables/cross/`` and a figure to ``paper/figures/cross_robustness.{pdf,png}``.
Each dataset is evaluated at its primary label granularity.

    python scripts/10_cross_dataset.py [--datasets cmban banglaabusememe memosen]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import primary_granularity  # noqa: E402

NAMES = {"cmban": "CMBan", "banglaabusememe": "BanglaAbuseMeme", "memosen": "MemoSen"}
SHORT = {"cmban": "CMBan", "banglaabusememe": "BAM", "memosen": "MemoSen"}
COLORS = {"cmban": "#2a78d6", "banglaabusememe": "#eb6834", "memosen": "#1baf7a"}  # validated categorical order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ENC = {"xlmr-base": "XLM-R", "xlmr-large": "XLM-R-L", "mbert": "mBERT", "banglabert": "BanglaBERT", "labse": "LaBSE",
       "me5-base": "mE5", "clip-b32": "CLIP-B/32", "clip-b16": "CLIP-B/16", "clip-l14": "CLIP-L/14", "dinov2-base": "DINOv2"}
HEAD = {"text_logreg": "LR", "text_mlp": "MLP", "image_logreg": "LR", "image_mlp": "MLP", "concat_lr": "Concat-LR",
        "concat_mlp": "Concat-MLP", "gmu": "GMU", "bilinear": "Bilinear", "late": "Late", "tfidf_lr": "LR", "majority": ""}
TEXT_FAMS = ["translit", "roman_noise", "bangla_noise", "codemix", "emoji_add", "emoji_remove"]
IMAGE_FAMS = ["blur", "crop", "resize", "jpeg", "text_occlusion", "region_occlusion"]
FAM = {"translit": "Transliteration", "roman_noise": "Roman. spelling noise", "bangla_noise": "Bangla spelling noise",
       "codemix": "Code-mixing", "emoji_add": "Emoji insertion", "emoji_remove": "Emoji/symbol removal", "blur": "Blur",
       "crop": "Crop", "resize": "Down-sampling", "jpeg": "JPEG", "text_occlusion": "Text occlusion",
       "region_occlusion": "Region occlusion"}
TEXT_HEADS, IMAGE_HEADS = ["text_logreg", "text_mlp"], ["image_logreg", "image_mlp"]
FUSION_HEADS = ["concat_lr", "concat_mlp", "gmu", "bilinear", "late"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


MD: list[str] = []  # markdown sections collected for results/RESULTS.md
PLAIN = [("\\%", "%"), ("$\\Delta$", "Δ"), ("$r$", "r"), ("\\,+\\,", " + "), ("\\,", " "), ("$\\geq 2$", "≥ 2"),
         ("$\\geq$", "≥"), ("$\\alpha=0.05$", "α = 0.05"), ("$\\pm$", "±"), ("\\ ", " "), ("$>1$", "> 1"),
         ("\\textbackslash{}", "/"), ("$", ""), ("--", "–")]


def plain(x) -> str:
    """LaTeX fragment -> plain text for the markdown report."""
    if isinstance(x, float):
        s = f"{x:.2f}" if abs(x) < 2 else f"{x:.1f}"
    else:
        s = str(x)
    for a, b in PLAIN:
        s = s.replace(a, b)
    s = re.sub(r"\\(?:emph|textbf)\{([^}]*)\}", r"\1", s)
    return s.replace("|", "/")


def md_table(df: pd.DataFrame, title: str, caption: str) -> None:
    cols = [plain(c) for c in df.columns]
    lines = [f"### {title}", "", plain(caption), "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(plain(v) for v in r) + " |" for r in df.itertuples(index=False)]
    MD.append("\n".join(lines) + "\n")


def table(df: pd.DataFrame, path: Path, caption: str, label: str, wide: bool = False, colfmt: str | None = None,
          floatfmt: str = "%.1f") -> None:
    body = df.to_latex(index=False, escape=False, column_format=colfmt or "l" + "c" * (df.shape[1] - 1), float_format=floatfmt)
    width = "\\textwidth" if wide else "\\columnwidth"
    env = "table*" if wide else "table"
    tex = (f"\\begin{{{env}}}[t]\n\\centering\n\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{adjustbox}}{{max width={width}}}\n{body}\\end{{adjustbox}}\n\\end{{{env}}}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    md_table(df, TITLES.get(path.stem, path.stem), caption)


TITLES = {"benchmark": "1. Clean-data benchmark", "significance": "2. Fusion vs. best unimodal (significance)",
          "robustness": "3. Robustness Drop", "flip_by_language": "4. Flip rate by language group",
          "augmentation": "5. Mitigation: Romanisation augmentation", "explain": "6. Explainability",
          "errors_clean": "7. Clean error taxonomy", "fragility": "8. Robustness fragility"}


def enc_name(exp: str) -> str:
    parts = exp.split(":", 1)[1].split("+")
    return "+".join(ENC.get(p, p) for p in parts)


def load(ds: str) -> dict | None:
    cfg = load_config(dataset=ds)
    res = cfg.path("results")
    if not (res / "benchmark" / "summary.csv").exists():
        return None
    gran = primary_granularity(cfg)
    out = {"cfg": cfg, "res": res, "gran": gran, "pair": f"{cfg.primary['text']}+{cfg.primary['image']}"}
    b = pd.read_csv(res / "benchmark" / "summary.csv")
    out["bench"] = b[b.granularity == gran]
    for key, rel in [("sig", "benchmark/significance.csv"), ("rob", "robustness/summary.csv"),
                     ("share", "explain/modality_share.csv"), ("cam", "explain/gradcam_summary.csv"),
                     ("err", "error_analysis/clean_error_table.csv"), ("frag", "error_analysis/fragility_table.csv")]:
        p = res / rel
        out[key] = pd.read_csv(p) if p.exists() else None
    for key in ("sig", "rob"):
        if out[key] is not None and "granularity" in out[key]:
            out[key] = out[key][out[key].granularity == gran]
    return out


def benchmark_rows(name: str, d: dict) -> dict:
    b = d["bench"]
    avg = b.groupby(["exp", "family", "model"]).macro_f1_mean.mean().mul(100).reset_index()

    def best(mask) -> pd.Series:
        return avg[mask].sort_values("macro_f1_mean", ascending=False).iloc[0]

    fmt = lambda r, enc=True: f"{r.macro_f1_mean:.1f} ({enc_name(r.exp) + ' ' if enc else ''}{HEAD[r.model]})".replace(" )", ")")
    maj = avg[avg.model == "majority"].macro_f1_mean.iloc[0]
    tfidf = avg[avg.model == "tfidf_lr"].macro_f1_mean.iloc[0]
    bt, bi = best(avg.family == "text"), best(avg.family == "image")
    prim = best(avg.exp == f"F:{d['pair']}")
    fus = avg[avg.family == "fusion"].sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    bf = fus.iloc[0]
    rank = int(fus.index[(fus.exp == prim.exp) & (fus.model == prim.model)][0]) + 1
    return {"Dataset": name, "Tasks": b.task.nunique(), "Majority": maj, "Char TF-IDF": tfidf,
            "Best text": fmt(bt), "Best image": fmt(bi), "XLM-R+CLIP-B/32": fmt(prim, enc=False),
            "Best fusion": fmt(bf), "Rank": f"{rank}/{len(fus)}"}


def significance_rows(name: str, d: dict) -> dict | None:
    s = d["sig"]
    if s is None or not len(s):
        return None
    row = {"Dataset": name}
    for comp, lab in [("fusion_vs_text", "text"), ("fusion_vs_image", "image")]:
        c = s[s.comparison == comp]
        row[f"$\\Delta$ vs {lab}"] = f"{100 * c['diff'].mean():+.1f}"
        row[f"Wins vs {lab}"] = f"{(c['diff'] > 0).sum()}/{len(c)}"
        row[f"Sig. vs {lab}"] = f"{(c.p_holm < 0.05).sum()}/{len(c)}"
    return row


def robustness_frame(d: dict) -> pd.DataFrame | None:
    r = d["rob"]
    if r is None:
        return None
    r = r[(r.group == "all") & (r.pair == d["pair"]) & (r.setting == "decoupled")]
    rows = []
    for fam in TEXT_FAMS + IMAGE_FAMS:
        g = r[r.family == fam]
        if not len(g):
            continue
        uni = TEXT_HEADS if fam in TEXT_FAMS else IMAGE_HEADS
        fu = g[g.model.isin(FUSION_HEADS)]
        rows.append({"family": fam, "uni_rd": g[g.model.isin(uni)].rd_rel_mean.mean(), "fus_rd": fu.rd_rel_mean.mean(),
                     "fus_flip": 100 * fu.flip_rate_mean.mean(), "fus_abs": 100 * fu.rd_abs_mean.mean()})
    return pd.DataFrame(rows).set_index("family")


def explain_row(name: str, d: dict) -> dict | None:
    e, c = d["share"], d["cam"]
    if e is None:
        return None
    e = e[e.model == "concat_mlp"]
    sh = e.groupby("variant").text_share.mean()
    row = {"Dataset": name, "Clean": sh.get("clean", np.nan)}
    for v, lab in [("translit_s3", "Translit."), ("roman_noise_s3", "Roman. noise"), ("emoji_add_s3", "Emoji ins."),
                   ("blur_s3", "Blur"), ("resize_s3", "Down-samp."), ("text_occlusion_s3", "Text occ.")]:
        row[lab] = sh.get(v, np.nan)
    if c is not None:
        row["CAM ratio"] = c.text_ratio.mean()
        row["$r$ blur"] = c.r_blur.mean()
        row["$r$ JPEG"] = c.r_jpeg.mean()
    return row


def category_row(name: str, t: pd.DataFrame | None) -> dict | None:
    if t is None:
        return None
    m = t.drop(columns="task").mean()
    return {"Dataset": name, **{k.replace("-", " ").capitalize().replace("Ocr", "OCR"): v for k, v in m.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(NAMES))
    ap.add_argument("--out", default=str(ROOT / "paper"))
    args = ap.parse_args()
    tdir, fdir = Path(args.out) / "tables" / "cross", Path(args.out) / "figures"
    data = {ds: d for ds in args.datasets if (d := load(ds)) is not None}
    print("datasets with results:", list(data))
    if not data:
        return

    # 1. benchmark summary
    rows = [benchmark_rows(NAMES[k], d) for k, d in data.items()]
    table(pd.DataFrame(rows), tdir / "benchmark.tex",
          "Clean-data benchmark across datasets: task-averaged macro-F1 (\\%) at each dataset's primary label "
          "granularity. Parentheses give the selected encoder(s) and head. Rank: position of the best primary-pair head "
          "among all fusion configurations.", "tab:cross_bench", wide=True)

    # 2. fusion vs unimodal significance
    rows = [r for k, d in data.items() if (r := significance_rows(NAMES[k], d))]
    if rows:
        table(pd.DataFrame(rows), tdir / "significance.tex",
              "Best fusion vs.\\ best unimodal model per task: mean macro-F1 difference (points), tasks where fusion is "
              "better, and tasks significant after Holm correction (paired bootstrap, $\\alpha=0.05$).", "tab:cross_sig")

    # 3. robustness
    rob = {k: f for k, d in data.items() if (f := robustness_frame(d)) is not None}
    if rob:
        fams = [f for f in TEXT_FAMS + IMAGE_FAMS if any(f in v.index for v in rob.values())]
        out = []
        for f in fams:
            row = {"Perturbation": FAM[f]}
            for k, v in rob.items():
                ok = f in v.index
                row[f"{SHORT[k]} uni."] = f"{v.loc[f, 'uni_rd']:.1f}" if ok else "--"
                row[f"{SHORT[k]} fus."] = f"{v.loc[f, 'fus_rd']:.1f} ({v.loc[f, 'fus_flip']:.0f})" if ok else "--"
            out.append(row)
        df = pd.DataFrame(out)
        table(df, tdir / "robustness.tex",
              "Relative Robustness Drop (\\%) across datasets, averaged over tasks and severities, for the matching "
              "unimodal heads (uni.) and the five XLM-R + CLIP-B/32 fusion heads (fus.; flip rate \\% in parentheses). "
              "The top six rows are linguistic, the bottom six visual perturbations.", "tab:cross_rob", wide=True)

        # figure: fusion RD per family, grouped by dataset
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        x = np.arange(len(fams))
        n = len(rob)
        w = 0.8 / n
        for j, (k, v) in enumerate(rob.items()):
            vals = [v.loc[f, "fus_rd"] if f in v.index else np.nan for f in fams]
            ax.bar(x + (j - (n - 1) / 2) * w, vals, w * 0.92, color=COLORS[k], label=NAMES[k], edgecolor="white",
                   linewidth=0.8)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvline(len([f for f in fams if f in TEXT_FAMS]) - 0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xticks(x, [FAM[f] for f in fams], rotation=35, ha="right")
        ax.set_ylabel("Relative drop of macro-F1 (%)")
        ax.grid(axis="x", visible=False)
        n_text = len([f for f in fams if f in TEXT_FAMS])
        tr = ax.get_xaxis_transform()
        ax.text((n_text - 1) / 2, 1.02, "linguistic perturbations", fontsize=8, color=INK2, ha="center", transform=tr)
        ax.text(n_text + (len(fams) - n_text - 1) / 2, 1.02, "visual perturbations", fontsize=8, color=INK2, ha="center",
                transform=tr)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fdir / "cross_robustness.png", dpi=300)
        fig.savefig(fdir / "cross_robustness.pdf")
        plt.close(fig)

        # language-group flip rates (fusion heads) for the two largest perturbations and blur
        lg = []
        for k, d in data.items():
            r = d["rob"]
            r = r[(r.pair == d["pair"]) & (r.setting == "decoupled") & r.model.isin(FUSION_HEADS) & ~r.group.isin(["all", "Unknown"])]
            for grp, g in r.groupby("group"):
                row = {"Dataset": SHORT[k], "Group": grp}
                for f in ("translit", "roman_noise", "codemix", "blur", "text_occlusion"):
                    row[FAM[f]] = 100 * g[g.family == f].flip_rate_mean.mean()
                lg.append(row)
        if lg:
            table(pd.DataFrame(lg), tdir / "flip_by_language.tex",
                  "Flip rate (\\%) of the fusion heads by script/language group of the meme text.", "tab:cross_lang")

    # 3b. mitigation: Romanisation augmentation (scripts/11_augmentation.py)
    rows = []
    for k, d in data.items():
        p = d["res"] / "augmentation" / "summary.csv"
        if not p.exists():
            continue
        a = pd.read_csv(p)
        a["grp"] = np.where(a.model.str.startswith("text"), "Text", "Fusion")
        a = a.groupby(["grp", "setting", "family"]).agg(f1=("f1", "mean"), rd=("rd_rel", "mean"))
        for setting, lab in [("none", "none"), ("translit", "+translit."), ("translit+roman", "+translit.\\,+\\,noise")]:
            row = {"Dataset": SHORT[k] if setting == "none" else "", "Augmentation": lab}
            for grp in ("Text", "Fusion"):
                row[f"{grp} clean"] = f"{100 * a.loc[(grp, setting, 'clean'), 'f1']:.1f}"
                for fam, fl in [("translit", "translit."), ("roman_noise", "noise")]:
                    r = a.loc[(grp, setting, fam)]
                    row[f"{grp} {fl}"] = f"{100 * r.f1:.1f} ({r.rd:.0f})"
            rows.append(row)
    if rows:
        table(pd.DataFrame(rows), tdir / "augmentation.tex",
              "Train-time Romanisation augmentation (XLM-R + CLIP-B/32). Macro-F1 (\\%) on clean test folds and under "
              "transliteration and Romanised spelling noise (mean over heads, tasks and severities); relative Robustness "
              "Drop (\\%) in parentheses. Spelling noise is unseen during training in the \\emph{+translit.} setting.",
              "tab:cross_aug", wide=True)

    # 4. explainability
    rows = [r for k, d in data.items() if (r := explain_row(NAMES[k], d))]
    if rows:
        table(pd.DataFrame(rows), tdir / "explain.tex",
              "Explainability across datasets. Columns 2--8: text share of the Integrated-Gradients attribution of "
              "Concat-MLP on clean inputs and at severity 3. CAM ratio: Grad-CAM mass on OCR text boxes divided by their "
              "area share ($>1$: over-attention to rendered text). $r$: Pearson correlation of clean vs.\\ perturbed CAMs "
              "(severity 2).", "tab:cross_xai", wide=True, floatfmt="%.2f")
        pd.DataFrame(rows).to_csv(tdir / "explain.csv", index=False)

    # 5. error taxonomy
    for key, fname, cap, lab in [
            ("err", "errors_clean", "Clean-data error categories of the primary fusion model (\\% of memes, averaged over "
             "tasks).", "tab:cross_err"),
            ("frag", "fragility", "Robustness fragility categories (\\% of memes, averaged over tasks; severity $\\geq 2$).",
             "tab:cross_frag")]:
        rows = [r for k, d in data.items() if (r := category_row(NAMES[k], d[key]))]
        if rows:
            df = pd.DataFrame(rows)
            table(df, tdir / f"{fname}.tex", cap, lab, wide=df.shape[1] > 6)
    head = ["# ToonSense results", "",
            "Generated by `scripts/10_cross_dataset.py` from `results/<dataset>/` (do not edit by hand). Primary pair: "
            "XLM-R-base + CLIP ViT-B/32. Macro-F1 in %, stratified 5-fold CV x 3 seeds. CMBan uses its coarse labels "
            "(133 publicly labelled memes); BanglaAbuseMeme (BAM, 4,043 memes) and MemoSen (4,368 memes) use their "
            "released labels.", "",
            "Figure: `paper/figures/cross_robustness.png`. Per-dataset tables and figures: "
            "`results/<dataset>/tables/`, `results/<dataset>/figures/`.", ""]
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(head + MD), encoding="utf-8")
    print(f"tables -> {tdir}\nfigure -> {fdir / 'cross_robustness.pdf'}\nreport -> {ROOT / 'results' / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
