"""Text-encoder comparison table (accuracy and macro-F1) across the three datasets.

Reads ``results/<dataset>/benchmark/summary.csv`` (written by step 03) and writes
``paper/tables/cross/text_encoders.tex``: every frozen text encoder with both heads (LR, MLP), plus the majority and
character TF-IDF baselines, as task-averaged accuracy and macro-F1 at each dataset's primary label granularity.
The best text encoder per column is set in bold.

    python scripts/13_encoder_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import primary_granularity  # noqa: E402

DATASETS = [("cmban", "CMBan"), ("banglaabusememe", "BanglaAbuseMeme"), ("memosen", "MemoSen")]
ENCODERS = [("xlmr-base", "XLM-R base"), ("xlmr-large", "XLM-R large"), ("mbert", "mBERT"),
            ("banglabert", "BanglaBERT"), ("labse", "LaBSE"), ("me5-base", "mE5 base")]
HEADS = [("text_logreg", "LR"), ("text_mlp", "MLP")]
BASELINES = [("B:majority", "majority", "Majority class", "--"), ("B:tfidf-char", "tfidf_lr", "Char TF-IDF", "LR")]


def scores(ds: str) -> pd.DataFrame:
    cfg = load_config(dataset=ds)
    gran = primary_granularity(cfg)
    s = pd.read_csv(cfg.path("results") / "benchmark" / "summary.csv")
    s = s[s.granularity == gran]
    return s.groupby(["exp", "model"])[["acc_mean", "macro_f1_mean"]].mean().mul(100)


def main() -> None:
    data = {ds: scores(ds) for ds, _ in DATASETS}
    rows = []  # (encoder label, head label, {ds: (acc, f1)}, is_text)
    for enc, name in ENCODERS:
        for i, (model, head) in enumerate(HEADS):
            vals = {ds: tuple(data[ds].loc[("T:" + enc, model)]) for ds, _ in DATASETS}
            rows.append((name if i == 0 else "", head, vals, True))
    for exp, model, name, head in BASELINES:
        vals = {ds: tuple(data[ds].loc[(exp, model)]) for ds, _ in DATASETS}
        rows.append((name, head, vals, False))

    best = {}  # (ds, metric index) -> best value among text encoders
    for ds, _ in DATASETS:
        for k in (0, 1):
            best[(ds, k)] = max(round(r[2][ds][k], 1) for r in rows if r[3])

    def cell(v: float, ds: str, k: int, is_text: bool) -> str:
        s = "%.1f" % v
        return "\\textbf{" + s + "}" if is_text and round(v, 1) == best[(ds, k)] else s

    n = len(DATASETS)
    head1 = "Encoder & Head & " + " & ".join("\\multicolumn{2}{c}{" + nm + "}" for _, nm in DATASETS) + " \\\\"
    rules = " ".join("\\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i) for i in range(n))
    head2 = " & & " + " & ".join(["Acc. & Macro-F1"] * n) + " \\\\"
    body = []
    for idx, (name, head, vals, is_text) in enumerate(rows):
        if not is_text and rows[idx - 1][3]:
            body.append("\\midrule")
        elif is_text and name and idx > 0:
            body.append("\\addlinespace[2pt]")
        cells = []
        for ds, _ in DATASETS:
            cells += [cell(vals[ds][0], ds, 0, is_text), cell(vals[ds][1], ds, 1, is_text)]
        body.append(name + " & " + head + " & " + " & ".join(cells) + " \\\\")

    caption = ("Frozen text encoders on clean data: accuracy and macro-F1 (\\%), averaged over each dataset's tasks "
               "(CMBan: 5 coarse-label tasks; BanglaAbuseMeme: 4; MemoSen: 1), repeated stratified 5-fold CV $\\times$ 3 "
               "seeds. Bold: best text encoder per column. Because the labels are imbalanced, the majority-class "
               "predictor reaches high accuracy but low macro-F1, so models are ranked by macro-F1.")
    tex = "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\small", "\\caption{" + caption + "}", "\\label{tab:text_encoders}",
        "\\begin{adjustbox}{max width=\\textwidth}", "\\begin{tabular}{ll" + "cc" * n + "}", "\\toprule", head1, rules, head2,
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""])
    out = ROOT / "paper" / "tables" / "cross" / "text_encoders.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
