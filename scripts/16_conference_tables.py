"""Compact single-column tables for the 8-page conference version of the paper.

Inputs: ``results/<dataset>/benchmark/summary.csv`` (step 03) and the cross-dataset tables written by step 10
(``paper/tables/cross/{explain,errors_clean,fragility}.tex``), so the numbers are identical to the extended paper.
Outputs: ``paper/tables/conf/{text_encoders,explain,errors}.tex``.

    python scripts/16_conference_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import primary_granularity  # noqa: E402

BS = chr(92)
CROSS = ROOT / "paper" / "tables" / "cross"
OUT = ROOT / "paper" / "tables" / "conf"
DATASETS = [("cmban", "CMBan"), ("banglaabusememe", "BAM"), ("memosen", "MemoSen")]
ENCODERS = [("xlmr-base", "XLM-R base"), ("xlmr-large", "XLM-R large"), ("mbert", "mBERT"),
            ("banglabert", "BanglaBERT"), ("labse", "LaBSE"), ("me5-base", "mE5 base")]
HEAD = {"text_logreg": "LR", "text_mlp": "MLP"}


def rows_of(tex_path: Path) -> list[list[str]]:
    """Body rows of a booktabs table written by step 10, split into cells."""
    lines = tex_path.read_text(encoding="utf-8").splitlines()
    start = lines.index(BS + "midrule") + 1
    end = lines.index(BS + "bottomrule")
    return [[c.strip() for c in l.rstrip(BS).rstrip().rstrip(BS).split("&")] for l in lines[start:end]]


def header_of(tex_path: Path) -> list[str]:
    lines = tex_path.read_text(encoding="utf-8").splitlines()
    h = lines[lines.index(BS + "toprule") + 1]
    return [c.strip() for c in h.replace(BS + BS, "").split("&")]


def table(caption: str, label: str, colfmt: str, header: str, body: list[str], resize: bool = True) -> str:
    tab = [BS + "begin{tabular}{" + colfmt + "}", BS + "toprule", header + " " + BS + BS, BS + "midrule"]
    tab += [r + " " + BS + BS if not r.startswith(BS) else r for r in body]
    tab += [BS + "bottomrule", BS + "end{tabular}"]
    inner = "\n".join(tab)
    if resize:
        inner = BS + "begin{adjustbox}{max width=" + BS + "columnwidth}\n" + inner + "\n" + BS + "end{adjustbox}"
    return "\n".join([BS + "begin{table}[t]", BS + "centering", BS + "small", BS + "caption{" + caption + "}",
                      BS + "label{" + label + "}", inner, BS + "end{table}", ""])


def text_encoders() -> str:
    best = {}
    for ds, _ in DATASETS:
        cfg = load_config(dataset=ds)
        s = pd.read_csv(cfg.path("results") / "benchmark" / "summary.csv")
        s = s[s.granularity == primary_granularity(cfg)]
        best[ds] = s.groupby(["exp", "model"])[["acc_mean", "macro_f1_mean"]].mean().mul(100)
    body, vals = [], {}
    for enc, name in ENCODERS:
        row = [name]
        for ds, _ in DATASETS:
            g = best[ds].loc["T:" + enc]
            m = g.macro_f1_mean.idxmax()  # head chosen by macro-F1
            vals[(enc, ds)] = (g.loc[m, "acc_mean"], g.loc[m, "macro_f1_mean"], HEAD[m])
    top = {ds: max(round(vals[(e, ds)][1], 1) for e, _ in ENCODERS) for ds, _ in DATASETS}
    for enc, name in ENCODERS:
        cells = [name]
        for ds, _ in DATASETS:
            a, f, h = vals[(enc, ds)]
            fs = "%.1f" % f
            if round(f, 1) == top[ds]:
                fs = BS + "textbf{" + fs + "}"
            cells += ["%.1f" % a, fs + "$^{" + h[0] + "}$"]
        body.append(" & ".join(cells))
    body.append(BS + "midrule")
    for exp, model, name in [("B:majority", "majority", "Majority class"), ("B:tfidf-char", "tfidf_lr", "Char TF-IDF")]:
        cells = [name]
        for ds, _ in DATASETS:
            a, f = best[ds].loc[(exp, model)]
            cells += ["%.1f" % a, "%.1f" % f]
        body.append(" & ".join(cells))
    header = ("Encoder & " + " & ".join(BS + "multicolumn{2}{c}{" + n + "}" for _, n in DATASETS) + " " + BS + BS
              + "\n" + " ".join(BS + "cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(3))
              + "\n & " + " & ".join(["Acc. & F1"] * 3))
    cap = ("Frozen text encoders: task-averaged accuracy and macro-F1 (\\%) with the better head per encoder "
           "($^{L}$: LR, $^{M}$: MLP). Bold: best text encoder. Both heads are listed in the supplementary material.")
    return table(cap, "tab:conf_text", "l" + "cc" * 3, header, body)


def explain() -> str:
    h = header_of(CROSS / "explain.tex")
    keep = ["Dataset", "Clean", "Translit.", "Blur", "Text occ.", "CAM ratio", "$r$ blur", "$r$ JPEG"]
    idx = [h.index(k) for k in keep]
    short = {"BanglaAbuseMeme": "BAM"}
    body = []
    for r in rows_of(CROSS / "explain.tex"):
        cells = [r[i] for i in idx]
        cells[0] = short.get(cells[0], cells[0])
        body.append(" & ".join(cells))
    header = (" & " + BS + "multicolumn{4}{c}{IG text share} & " + BS + "multicolumn{3}{c}{Grad-CAM} " + BS + BS
              + "\n" + BS + "cmidrule(lr){2-5} " + BS + "cmidrule(lr){6-8}\n"
              + "Dataset & Clean & Translit. & Blur & Text occ. & Ratio & $r$ blur & $r$ JPEG")
    cap = ("Explainability. IG text share of Concat-MLP on clean inputs and at severity 3. Ratio: Grad-CAM mass on "
           "OCR text boxes over their area share ($>1$: over-attention to rendered text). $r$: correlation of clean "
           "and perturbed CAMs (severity 2).")
    return table(cap, "tab:conf_xai", "lccccccc", header, body)


def errors() -> str:
    he, hf = header_of(CROSS / "errors_clean.tex"), header_of(CROSS / "fragility.tex")
    e = {r[0]: dict(zip(he, r)) for r in rows_of(CROSS / "errors_clean.tex")}
    f = {r[0]: dict(zip(hf, r)) for r in rows_of(CROSS / "fragility.tex")}
    short = {"BanglaAbuseMeme": "BAM"}
    body = []
    for ds in e:
        body.append(" & ".join([short.get(ds, ds), e[ds]["Contextual/cultural"], e[ds]["Linguistic"],
                                e[ds]["Visual"], e[ds]["OCR limited"], e[ds]["Cross modal"], f[ds]["Stable"],
                                f[ds]["Linguistic fragile"], f[ds]["Visual fragile"], f[ds]["Cross modal fragile"]]))
    header = (" & " + BS + "multicolumn{5}{c}{Clean errors} & " + BS + "multicolumn{4}{c}{Under perturbation} "
              + BS + BS + "\n" + BS + "cmidrule(lr){2-6} " + BS + "cmidrule(lr){7-10}\n"
              + "Dataset & Ctx. & Ling. & Vis. & OCR & X-mod. & Stable & Ling. & Vis. & X-mod.")
    cap = ("Error taxonomy of the best learned XLM-R + CLIP-B/32 fusion head (\\% of memes, mean over tasks). "
           "Clean errors: contextual/cultural, linguistic, visual, OCR-limited, cross-modal. Under perturbation "
           "(severity $\\geq2$): stable memes and memes broken only by linguistic, only by visual, or by both "
           "(X-mod.) perturbation types.")
    return table(cap, "tab:conf_err", "l" + "c" * 9, header, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("text_encoders", text_encoders), ("explain", explain), ("errors", errors)]:
        (OUT / (name + ".tex")).write_text(fn(), encoding="utf-8")
        print("wrote", (OUT / (name + ".tex")).relative_to(ROOT))


if __name__ == "__main__":
    main()
