"""Step 6: error analysis — categorise clean-data errors and robustness failures.

Clean errors of the primary fusion model (seed-ensembled OOF) are attributed by contrasting the
unimodal models trained on the same folds:
  linguistic          text-only wrong, image-only right   (text branch misleads)
  visual              image-only wrong, text-only right   (image branch misleads)
  cross-modal         both unimodal right, fusion wrong   (fusion failure)
  ocr-limited         all wrong and OCR text empty / low confidence
  contextual/cultural all wrong with usable text — candidate for implicit/cultural meaning (verify manually)
Robustness fragility (a correct clean prediction becomes wrong at severity >= 2):
  linguistic-fragile / visual-fragile / cross-modal-fragile (both) / stable
A manual annotation sheet is exported so that 'cultural' can be confirmed by annotators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import TASKS, classes_for, load_dataset, primary_granularity  # noqa: E402
from toonsense.ocr import load_texts  # noqa: E402

LOW_CONF = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--granularity", default=None, help="default: the dataset's primary granularity")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    ds = load_dataset(cfg)
    ids = ds.ids
    gran = args.granularity or primary_granularity(cfg)
    tx = load_texts(cfg, ids, ds.gold)
    ds.fill_lang_groups(tx.text.tolist())
    pt, pi = cfg.primary["text"], cfg.primary["image"]
    seeds = cfg.benchmark["seeds"]
    res = cfg.path("results")
    out = res / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(res / "benchmark" / "summary.csv")
    rob_dir = res / "robustness" / "preds"

    rows, frag_rows, sheet = [], [], []
    for task in TASKS:
        y = ds.labels(task, gran)
        cls = classes_for(task, gran)
        o = np.load(res / "benchmark" / "oof" / f"{gran}_{task}.npz")
        fsum = summ[(summ.task == task) & (summ.granularity == gran) & (summ.exp == f"F:{pt}+{pi}")]
        # Late fusion is excluded: it averages the two unimodal models, so (for binary tasks) it cannot be wrong when
        # both are right and the cross-modal category would be empty by construction.
        fsum = fsum[fsum.model != "late"]
        best_head = fsum.sort_values("macro_f1_mean", ascending=False).model.iloc[0]

        def ens(key):
            return np.mean([o[f"{key}|s{s}".replace(":", "~")] for s in seeds], 0).argmax(1)

        p_t, p_i, p_f = ens(f"T:{pt}|text_mlp"), ens(f"I:{pi}|image_mlp"), ens(f"F:{pt}+{pi}|{best_head}")
        for k, sid in enumerate(ids):
            ok_t, ok_i, ok_f = p_t[k] == y[k], p_i[k] == y[k], p_f[k] == y[k]
            if ok_f:
                cat = "correct"
            elif ok_t and ok_i:
                cat = "cross-modal"
            elif not ok_t and ok_i:
                cat = "linguistic"
            elif ok_t and not ok_i:
                cat = "visual"
            elif tx.n_frags[k] == 0 or tx.ocr_conf[k] < LOW_CONF:
                cat = "ocr-limited"
            else:
                cat = "contextual/cultural"
            rows.append({"task": task, "id": sid, "lang_group": ds.frame.lang_group[k], "category": cat, "fusion_head": best_head})
            if cat != "correct":
                sheet.append({"id": sid, "task": task, "gold": cls[y[k]], "fusion_pred": cls[p_f[k]], "text_pred": cls[p_t[k]],
                              "image_pred": cls[p_i[k]], "auto_category": cat, "ocr_conf": round(tx.ocr_conf[k], 3),
                              "language": ds.frame.language[k], "text": tx.text[k],
                              "manual_category": "", "cultural_reference": "", "notes": ""})

        # robustness fragility (same fusion head, majority vote over seeds)
        rp = rob_dir / f"{gran}_{task}.npz"
        if rp.exists():
            r = np.load(rp)
            pair = f"{pt}+{pi}"

            def vote(variant):
                P = np.stack([r[f"{pair}|{best_head}|{variant}|s{s}"] for s in seeds])
                return np.array([np.bincount(col, minlength=len(cls)).argmax() for col in P.T])

            clean = vote("clean")
            breaks = {"text": np.zeros(len(ids), bool), "image": np.zeros(len(ids), bool)}
            fam_break = {}
            for mod in ("text", "image"):
                for fam in cfg.robustness[mod]:
                    for s in cfg.robustness["severities"]:
                        if s < 2 or f"{pair}|{best_head}|{fam}_s{s}|s{seeds[0]}" not in r.files:
                            continue
                        b = (clean == y) & (vote(f"{fam}_s{s}") != y)
                        breaks[mod] |= b
                        fam_break[f"{fam}_s{s}"] = b
            for k, sid in enumerate(ids):
                if clean[k] != y[k]:
                    cat = "clean-error"
                elif breaks["text"][k] and breaks["image"][k]:
                    cat = "cross-modal-fragile"
                elif breaks["text"][k]:
                    cat = "linguistic-fragile"
                elif breaks["image"][k]:
                    cat = "visual-fragile"
                else:
                    cat = "stable"
                frag_rows.append({"task": task, "id": sid, "lang_group": ds.frame.lang_group[k], "fragility": cat,
                                  "ocr_conf": tx.ocr_conf[k], "n_words": len(tx.text[k].split()),
                                  **{f"break_{v}": bool(b[k]) for v, b in fam_break.items()}})

    err = pd.DataFrame(rows)
    err.to_csv(out / "clean_error_categories.csv", index=False)
    tab = pd.crosstab(err.task, err.category, normalize="index").mul(100).round(1)
    tab.to_csv(out / "clean_error_table.csv")
    print("Clean error categories (% of samples):\n", tab.to_string())
    pd.crosstab([err.lang_group], err.category, normalize="index").mul(100).round(1).to_csv(out / "clean_error_by_language.csv")
    pd.DataFrame(sheet).to_csv(out / "manual_annotation_sheet.csv", index=False, encoding="utf-8-sig")

    if frag_rows:
        fr = pd.DataFrame(frag_rows)
        fr.to_csv(out / "fragility_samples.csv", index=False)
        ft = pd.crosstab(fr.task, fr.fragility, normalize="index").mul(100).round(1)
        ft.to_csv(out / "fragility_table.csv")
        print("\nRobustness fragility (% of samples):\n", ft.to_string())
        pd.crosstab(fr.lang_group, fr.fragility, normalize="index").mul(100).round(1).to_csv(out / "fragility_by_language.csv")
        # which perturbations break correct predictions most often
        bcols = [c for c in fr.columns if c.startswith("break_")]
        fr.groupby("task")[bcols].mean().mul(100).round(1).T.to_csv(out / "break_rate_by_perturbation.csv")
        # do fragile memes have worse OCR / shorter text?
        print("\n", fr.groupby("fragility")[["ocr_conf", "n_words"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
