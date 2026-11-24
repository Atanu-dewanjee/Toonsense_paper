"""Step 7: paper-ready LaTeX tables and figures from results/*.

Tables (results/tables/): dataset statistics, main benchmark, encoder grid, robustness, error analysis.
Figures (results/figures/): benchmark bars with 95% CI, robustness heatmap, severity curves,
modality-reliance shift. PNG (300 dpi) + PDF.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import TASKS, describe, load_dataset, primary_granularity  # noqa: E402
from toonsense.ocr import load_texts  # noqa: E402

# Reference palette (validated): first three categorical slots + blue<->red diverging with grey midpoint.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"])
class _TaskLabels(dict):
    def __missing__(self, k):
        return str(k).replace("_", " ").title()


TASK_LABEL = _TaskLabels({"humour": "Humour", "sarcasm": "Sarcasm", "offensive": "Offensive", "motivational": "Motivational",
                          "sentiment": "Sentiment", "abuse": "Abuse", "vulgar": "Vulgarity"})
MODEL_LABEL = {"text_logreg": "Text LR", "text_mlp": "Text MLP", "image_logreg": "Image LR", "image_mlp": "Image MLP",
               "concat_lr": "Concat LR", "concat_mlp": "Concat MLP", "gmu": "GMU", "bilinear": "Bilinear", "late": "Late fusion",
               "majority": "Majority", "tfidf_lr": "Char TF-IDF LR"}
FAM_LABEL = {"translit": "Transliteration", "roman_noise": "Roman. spelling noise", "bangla_noise": "Bangla spelling noise",
             "codemix": "Code-mixing", "emoji_add": "Emoji insertion", "emoji_remove": "Emoji/symbol removal",
             "blur": "Blur", "crop": "Crop", "resize": "Down-sampling", "jpeg": "JPEG", "text_occlusion": "Text occlusion",
             "region_occlusion": "Region occlusion"}

DATASET_LABEL = {"cmban": "the labelled CMBan subset", "banglaabusememe": "BanglaAbuseMeme", "memosen": "MemoSen"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


DS = {"key": "data", "name": "Data"}  # set in main(); used to make labels/captions unique per dataset
SHORT_NAME = {"cmban": "CMBan", "banglaabusememe": "BanglaAbuseMeme", "memosen": "MemoSen"}


def latex(df: pd.DataFrame, path: Path, caption: str, label: str, wide: bool = False, fit: bool = False,
          floatfmt: str = "%.1f") -> None:
    """Write a booktabs table. Labels get a dataset suffix and captions a dataset prefix so that tables of
    several datasets can coexist in one paper. `wide` -> table*, `fit` -> scale to the available width."""
    body = df.to_latex(index=False, escape=False, column_format="l" + "c" * (df.shape[1] - 1), float_format=floatfmt)
    if fit:
        width = "\\textwidth" if wide else "\\columnwidth"
        body = "\\begin{adjustbox}{max width=" + width + "}\n" + body + "\\end{adjustbox}\n"
    env = "table*" if wide else "table"
    tex = (f"\\begin{{{env}}}[t]\n\\centering\n\\small\n\\caption{{{DS['name']}: {caption}}}\n"
           f"\\label{{{label}_{DS['key']}}}\n{body}\\end{{{env}}}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def pm(mean, std) -> str:
    return f"{100 * mean:.1f}\\,$\\pm$\\,{100 * std:.1f}"


def model_name(r) -> str:
    if r.family == "baseline":
        return MODEL_LABEL[r.model]
    enc = r.text_enc if r.family == "text" else r.image_enc if r.family == "image" else f"{r.text_enc} + {r.image_enc}"
    return f"{enc} / {MODEL_LABEL.get(r.model, r.model)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--granularity", default=None, help="default: the dataset's primary granularity")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    res = cfg.path("results")
    tdir, fdir = res / "tables", res / "figures"
    gran = args.granularity or primary_granularity(cfg)
    pt, pi = cfg.primary["text"], cfg.primary["image"]
    pair = f"{pt}+{pi}"
    ds = load_dataset(cfg)
    DS.update(key=cfg.dataset_name, name=SHORT_NAME.get(cfg.dataset_name, cfg.dataset_name))

    # ---------------- dataset statistics
    d = describe(ds)
    d = d[d.granularity == "fine"].copy()
    d["n (\\%)"] = d.apply(lambda r: f"{r.n} ({r.pct:.1f})", axis=1)
    latex(d[["task", "class", "n (\\%)"]].rename(columns={"task": "Task", "class": "Class"}), tdir / "dataset_stats.tex",
          f"Label distribution of {DATASET_LABEL.get(cfg.dataset_name, cfg.dataset_name)} (N={len(ds)}).", "tab:data")
    tx = load_texts(cfg, ds.ids, ds.gold)
    ds.fill_lang_groups(tx.text.tolist())
    stats = pd.DataFrame([{"N": len(ds), **ds.frame.lang_group.value_counts().to_dict(),
                           "Text source": "/".join(sorted(tx.source.unique())),
                           "Mean words": round(tx.text.str.split().str.len().mean(), 1),
                           "Mean OCR conf.": round(tx.ocr_conf.mean(), 3)}])
    tdir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(tdir / "dataset_overview.csv", index=False)

    # ---------------- benchmark
    bpath = res / "benchmark" / "summary.csv"
    if bpath.exists():
        s = pd.read_csv(bpath)
        s = s[s.granularity == gran].copy()
        s["name"] = s.apply(model_name, axis=1)
        # main table: baselines, every unimodal encoder (best head), primary fusion heads, best grid pair
        keep = []
        for fam in ("baseline", "text", "image"):
            sub = s[s.family == fam]
            for exp, g in sub.groupby("exp", sort=False):
                best = g.groupby("model").macro_f1_mean.mean().idxmax()
                keep.append(g[g.model == best])
        keep.append(s[s.exp == f"F:{pair}"])
        grid = s[(s.family == "fusion") & (s.exp != f"F:{pair}")]
        if len(grid):
            bp = grid.groupby(["exp", "model"]).macro_f1_mean.mean().idxmax()
            keep.append(grid[(grid.exp == bp[0]) & (grid.model == bp[1])])
        main = pd.concat(keep)
        tab = main.pivot_table(index=["family", "name"], columns="task", values="macro_f1_mean", sort=False)
        sd = main.pivot_table(index=["family", "name"], columns="task", values="macro_f1_std", sort=False)
        tab = tab[[t for t in TASKS if t in tab.columns]]
        rows = []
        for (fam, name), r in tab.iterrows():
            row = {"Family": fam.capitalize(), "Model": name.replace("_", "\\_")}
            for t in tab.columns:
                row[TASK_LABEL[t]] = pm(r[t], sd.loc[(fam, name), t])
            row["Avg."] = f"{100 * r.mean():.1f}"
            rows.append(row)
        latex(pd.DataFrame(rows), tdir / f"benchmark_{gran}.tex",
              f"Clean-data benchmark ({gran} labels): macro-F1 (\\%), mean $\\pm$ std over {len(cfg.benchmark['seeds'])} seeds "
              f"of {cfg.benchmark['folds']}-fold stratified CV.", f"tab:bench_{gran}", wide=True, fit=True)
        pd.DataFrame(rows).to_csv(tdir / f"benchmark_{gran}.csv", index=False)

        # full metric table for the primary pair
        prim = s[s.exp.isin([f"T:{pt}", f"I:{pi}", f"F:{pair}"])]
        full = []
        for _, r in prim.iterrows():
            full.append({"Task": TASK_LABEL[r.task], "Model": r["name"], "Acc": pm(r.acc_mean, r.acc_std), "P": pm(r.precision_mean, r.precision_std),
                         "R": pm(r.recall_mean, r.recall_std), "Macro-F1": pm(r.macro_f1_mean, r.macro_f1_std),
                         "95\\% CI": f"[{100 * r.ens_ci_lo:.1f}, {100 * r.ens_ci_hi:.1f}]"})
        latex(pd.DataFrame(full), tdir / f"primary_full_{gran}.tex",
              f"Accuracy, macro precision/recall/F1 of the primary XLM-R + CLIP configuration ({gran}).", f"tab:primary_{gran}",
              wide=True, fit=True)

        # encoder grid heatmap table (best grid head per pair, averaged over tasks)
        fg = s[s.family == "fusion"].groupby(["text_enc", "image_enc", "task"]).macro_f1_mean.max().groupby(["text_enc", "image_enc"]).mean()
        if len(fg):
            gt = (100 * fg.unstack()).round(1)
            gt.to_csv(tdir / f"fusion_grid_{gran}.csv")
            latex(gt.reset_index().rename(columns={"text_enc": "Text \\textbackslash{} Image"}), tdir / f"fusion_grid_{gran}.tex",
                  "Fusion grid: task-averaged macro-F1 (\\%) of the best fusion head for each encoder pair.", f"tab:grid_{gran}", fit=True)

        # figure: best text / image / fusion per task with 95% CI
        fig, ax = plt.subplots(figsize=(6.8, 2.8))
        tasks = [t for t in TASKS if t in s.task.unique()]
        x = np.arange(len(tasks))
        for k, (fam, col, lab) in enumerate([("text", C1, "Text-only"), ("image", C2, "Image-only"), ("fusion", C3, "Multimodal")]):
            vals, lo, hi = [], [], []
            for t in tasks:
                r = s[(s.family == fam) & (s.task == t)].sort_values("macro_f1_mean", ascending=False).iloc[0]
                vals.append(r.ens_macro_f1), lo.append(r.ens_macro_f1 - r.ens_ci_lo), hi.append(r.ens_ci_hi - r.ens_macro_f1)
            ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=col, label=lab, edgecolor="white", linewidth=1)
            ax.errorbar(x + (k - 1) * 0.26, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for k, t in enumerate(tasks):
            mj = s[(s.model == "majority") & (s.task == t)].macro_f1_mean
            if len(mj):
                ax.hlines(mj.iloc[0], k - 0.42, k + 0.42, colors=INK2, linestyles="--", linewidth=0.9)
        ax.set_xticks(x, [TASK_LABEL[t] for t in tasks])
        ax.set_ylabel("Macro-F1 (seed ensemble)")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=3, loc="upper left", fontsize=8)
        ax.text(len(tasks) - 0.5, 0.97, "dashed: majority baseline", ha="right", va="top", fontsize=7, color=INK2)
        save(fig, fdir / f"benchmark_{gran}")

    # ---------------- robustness
    rpath = res / "robustness" / "summary.csv"
    if rpath.exists():
        r = pd.read_csv(rpath)
        r = r[(r.granularity == gran) & (r.group == "all") & (r.pair == pair) & (r.setting == "decoupled")]
        fams = [f for f in list(cfg.robustness["text"]) + list(cfg.robustness["image"]) if f in r.family.unique()]
        models = [m for m in MODEL_LABEL if m in r.model.unique()]
        # heatmap: relative robustness drop averaged over tasks and severities
        hm = r.pivot_table(index="model", columns="family", values="rd_rel_mean", aggfunc="mean").loc[models, fams]
        fl = r.pivot_table(index="model", columns="family", values="flip_rate_mean", aggfunc="mean").loc[models, fams]
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        lim = max(5.0, np.nanmax(np.abs(hm.values)))
        im = ax.imshow(hm.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for a in range(hm.shape[0]):
            for b in range(hm.shape[1]):
                v = hm.values[a, b]
                ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(len(fams)), [FAM_LABEL[f] for f in fams], rotation=40, ha="right")
        ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
        ax.axvline(len(cfg.robustness["text"]) - 0.5, color=INK, linewidth=1)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Relative robustness drop (%)")
        ax.set_title("Linguistic  |  Visual", fontsize=9, color=INK2)
        save(fig, fdir / f"robustness_heatmap_{gran}")

        out = []
        for f in fams:
            row = {"Perturbation": FAM_LABEL[f]}
            for m in models:
                rel = [x for x in ("text", "image") if m.startswith(x)]
                if rel and f not in cfg.robustness[rel[0]]:
                    row[MODEL_LABEL[m]] = "--"  # unimodal model is unaffected by the other modality
                else:
                    row[MODEL_LABEL[m]] = f"{hm.loc[m, f]:.1f} ({100 * fl.loc[m, f]:.0f})"
            out.append(row)
        latex(pd.DataFrame(out), tdir / f"robustness_{gran}.tex",
              f"Relative Robustness Drop (\\%) of macro-F1, averaged over tasks and severities; flip rate (\\%) in parentheses "
              f"({pt} + {pi}, {gran}).", f"tab:rob_{gran}", wide=True, fit=True)

        # per-task RD for the best fusion head
        fus = [m for m in ("concat_mlp", "gmu", "bilinear", "concat_lr", "late") if m in models]
        best_f = r[r.model.isin(fus)].groupby("model").rd_rel_mean.mean().idxmin() if fus else None
        if best_f:
            pt_tab = r[r.model == best_f].pivot_table(index="family", columns="task", values="rd_abs_mean", aggfunc="mean").loc[fams]
            pt_tab = (100 * pt_tab).round(1)
            pt_tab.index = [FAM_LABEL[f] for f in pt_tab.index]
            pt_tab.columns = [TASK_LABEL[c] for c in pt_tab.columns]
            latex(pt_tab.reset_index().rename(columns={"index": "Perturbation"}), tdir / f"robustness_by_task_{gran}.tex",
                  f"Absolute macro-F1 drop (points) per task for the most robust fusion head ({MODEL_LABEL[best_f]}).", f"tab:rob_task_{gran}", fit=True)

        # severity curves (small multiples, 3 series)
        show = [("text_mlp", C1, "Text MLP"), ("image_mlp", C2, "Image MLP"), (best_f or "concat_mlp", C3, MODEL_LABEL.get(best_f, "Fusion"))]
        n = len(fams)
        cols = 6
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(9.5, 3.4), sharey=True, squeeze=False)
        for k, f in enumerate(fams):
            ax = axes[k // cols, k % cols]
            for m, col, lab in show:
                g = r[(r.model == m) & (r.family == f)].groupby("severity")
                f1c = r[r.model == m].f1_clean_mean.mean()
                ys = [f1c] + g.f1_pert_mean.mean().tolist()
                ax.plot(range(len(ys)), ys, color=col, linewidth=2, marker="o", markersize=4, label=lab)
            ax.set_title(FAM_LABEL[f], fontsize=8)
            ax.set_xticks(range(len(cfg.robustness["severities"]) + 1), ["0"] + [str(v) for v in cfg.robustness["severities"]])
            ax.grid(axis="x", visible=False)
        for k in range(n, axes.size):
            axes.flat[k].axis("off")
        for a in axes[:, 0]:
            a.set_ylabel("Macro-F1")
        fig.supxlabel("Severity (0 = clean)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
        save(fig, fdir / f"severity_curves_{gran}")

        # language-group flip rates
        rg = pd.read_csv(rpath)
        rg = rg[(rg.granularity == gran) & (rg.pair == pair) & (~rg.group.isin(["all", "Unknown"])) & (rg.model == (best_f or "concat_mlp"))]
        if len(rg):
            lg = (100 * rg.pivot_table(index="family", columns="group", values="flip_rate_mean", aggfunc="mean")).round(1)
            lg = lg.loc[[f for f in fams if f in lg.index]]
            lg.index = [FAM_LABEL[f] for f in lg.index]
            lg.to_csv(tdir / f"flip_by_language_{gran}.csv")

    # ---------------- explainability: modality reliance shift
    epath = res / "explain" / "modality_share.csv"
    if epath.exists():
        e = pd.read_csv(epath)
        m = "concat_mlp"
        e = e[e.model == m]
        clean = e[e.variant == "clean"].text_share.mean()
        e["family"] = e.variant.str.rsplit("_s", n=1).str[0]
        e["severity"] = pd.to_numeric(e.variant.str.rsplit("_s", n=1).str[1], errors="coerce")
        sh = e[e.severity == e.severity.max()].groupby("family").text_share.mean()
        fams = [f for f in list(cfg.robustness["text"]) + list(cfg.robustness["image"]) if f in sh.index]
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        yy = np.arange(len(fams))
        ax.axvline(clean, color=INK2, linestyle="--", linewidth=1)
        ax.text(clean, 1.01, f"clean {clean:.2f}", fontsize=7, color=INK2, va="bottom", ha="center", transform=ax.get_xaxis_transform())
        for k, f in enumerate(fams):
            col = C1 if f in cfg.robustness["text"] else C2
            ax.plot([clean, sh[f]], [k, k], color=col, linewidth=2)
            ax.plot(sh[f], k, "o", color=col, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
        ax.set_yticks(yy, [FAM_LABEL[f] for f in fams])
        ax.invert_yaxis()
        ax.set_xlabel("Text share of IG attribution (Concat MLP, max severity)")
        ax.grid(axis="y", visible=False)
        save(fig, fdir / "modality_share_shift")
        tab = e.groupby(["task", "variant"]).text_share.mean().unstack(0).round(3)
        tab.to_csv(tdir / "modality_share.csv")

    gpath = res / "explain" / "gradcam_summary.csv"
    if gpath.exists():
        g = pd.read_csv(gpath).set_index("task")
        g.index = [TASK_LABEL.get(t, t) for t in g.index]
        g = g.round(3).reset_index().rename(columns={"index": "Task"})
        g = g.rename(columns={"cam_text_mass": "CAM on text", "text_area": "Text area", "text_ratio": "Ratio"})
        g.columns = [c.replace("r_text_occlusion", "$r$ text occ.").replace("r_region_occlusion", "$r$ region occ.")
                     .replace("r_", "$r$ ").replace("_", " ") for c in g.columns]
        latex(g, tdir / "gradcam.tex", "Grad-CAM analysis: CAM mass on OCR text regions vs. their area share (ratio $>1$: over-attention "
              "to rendered text) and Pearson $r$ between clean and perturbed CAMs.", "tab:gradcam", fit=True, floatfmt="%.2f")

    # ---------------- error analysis
    for name, cap in [("clean_error_table", "Clean-data error categories of the primary fusion model (\\% of memes)."),
                      ("fragility_table", "Robustness fragility categories (\\% of memes; severity $\\geq$ 2).")]:
        p = res / "error_analysis" / f"{name}.csv"
        if p.exists():
            t = pd.read_csv(p)
            t["task"] = t["task"].map(TASK_LABEL)
            t.columns = ["Task" if c == "task" else c.replace("-", " ").capitalize().replace("Ocr", "OCR") for c in t.columns]
            latex(t, tdir / f"{name}.tex", cap, f"tab:{name}", fit=True)
    print(f"tables -> {tdir}\nfigures -> {fdir}")


if __name__ == "__main__":
    main()
