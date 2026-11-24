"""Step 3: clean-data model benchmark (baselines, unimodal, fusion) with repeated stratified CV.

Outputs (results/benchmark/):
  raw_scores.csv     one row per (experiment, model, task, granularity, seed)
  summary.csv        mean/std over seeds + seed-ensemble macro-F1 with 95% bootstrap CI
  significance.csv   paired bootstrap: best fusion vs best text-only / best image-only (Holm-adjusted)
  oof/*.npz          out-of-fold probabilities for every run (used by error analysis)
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import os  # noqa: E402

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)
os.environ["PYTHONPATH"] = _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")  # for joblib workers

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import TASKS, classes_for, load_dataset  # noqa: E402
from toonsense.encoders import has_features, load_features  # noqa: E402
from toonsense.metrics import bootstrap_ci, paired_bootstrap, scores  # noqa: E402
from toonsense.ocr import load_texts  # noqa: E402


def baseline_oof(kind: str, texts: list[str], y: np.ndarray, C: int, folds: int, seed: int) -> np.ndarray:
    P = np.zeros((len(y), C))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        if kind == "majority":
            P[te, np.bincount(y[tr], minlength=C).argmax()] = 1.0
        else:  # char n-gram TF-IDF + logistic regression (script-agnostic, robust to code-mixing)
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
            Xtr = vec.fit_transform([texts[i] for i in tr])
            m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
            P[np.ix_(te, m.classes_)] = m.predict_proba(vec.transform([texts[i] for i in te]))
    return P


def build_experiments(cfg, avail_t, avail_i, primary_only=False):
    b = cfg.benchmark
    exps = [("B:majority", "baseline", None, None, ["majority"]), ("B:tfidf-char", "baseline", None, None, ["tfidf_lr"])]
    for t in avail_t:
        exps.append((f"T:{t}", "text", t, None, [f"text_{h}" for h in b["unimodal_heads"]]))
    for i in avail_i:
        exps.append((f"I:{i}", "image", None, i, [f"image_{h}" for h in b["unimodal_heads"]]))
    pt, pi = cfg.primary["text"], cfg.primary["image"]
    if pt in avail_t and pi in avail_i:
        exps.append((f"F:{pt}+{pi}", "fusion", pt, pi, list(b["fusion_heads"])))
    if not primary_only:
        for t, i in itertools.product(avail_t, avail_i):
            if (t, i) != (pt, pi):
                exps.append((f"F:{t}+{i}", "fusion", t, i, list(b["grid_heads"])))
    return exps


def run_task(cfg_path, task, gran, exps, ids, texts, y, C):
    import torch

    torch.set_num_threads(1)
    from toonsense.config import load_config as _lc
    from toonsense.cv import run_cv

    cfg = _lc(cfg_path)
    b, hp = cfg.benchmark, cfg.head
    feats_t = {t: load_features(cfg, "text", t, "clean", ids) for t in {e[2] for e in exps if e[2]}}
    feats_i = {i: load_features(cfg, "image", i, "clean", ids) for i in {e[3] for e in exps if e[3]}}
    rows, oofs = [], {}
    for exp, fam, t, i, models in exps:
        for seed in b["seeds"]:
            if fam == "baseline":
                oof = {models[0]: {"clean": baseline_oof(models[0].split("_")[0], texts, y, C, b["folds"], seed)}}
            else:
                oof = run_cv(feats_t.get(t), feats_i.get(i), y, C, models, b, hp, seed)
            for m in models:
                P = oof[m]["clean"]
                oofs[f"{exp}|{m}|s{seed}"] = P.astype(np.float32)
                rows.append({"exp": exp, "family": fam, "text_enc": t, "image_enc": i, "model": m, "task": task,
                             "granularity": gran, "seed": seed, **scores(y, P.argmax(1), C)})
    return task, gran, rows, oofs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--primary-only", action="store_true", help="skip the encoder-pair fusion grid")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    ds = load_dataset(cfg)
    ids = ds.ids
    texts = load_texts(cfg, ids, ds.gold)["text"].tolist()
    avail_t = [t for t in cfg.encoders["text"] if has_features(cfg, "text", t, "clean")]
    avail_i = [i for i in cfg.encoders["image"] if has_features(cfg, "image", i, "clean")]
    print(f"text encoders: {avail_t}\nimage encoders: {avail_i}")
    exps = build_experiments(cfg, avail_t, avail_i, args.primary_only)
    out = cfg.path("results") / "benchmark"
    (out / "oof").mkdir(parents=True, exist_ok=True)

    jobs = []
    for gran in cfg.data["granularities"]:
        for task in TASKS:
            C = len(classes_for(task, gran))
            jobs.append((task, gran, ds.labels(task, gran), C))
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(args.config, task, gran, exps, ids, texts, y, C) for task, gran, y, C in jobs
    )
    print(f"benchmark finished in {(time.time() - t0) / 60:.1f} min")

    rows = []
    for task, gran, r, oofs in res:
        rows += r
        np.savez_compressed(out / "oof" / f"{gran}_{task}.npz", ids=np.array(ids), **{k.replace(":", "~"): v for k, v in oofs.items()})
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw_scores.csv", index=False)

    # ---- summary: mean/std across seeds + seed-ensemble macro-F1 with bootstrap CI
    B = int(cfg.benchmark["bootstrap"])
    oof_all = {(t, g): o for t, g, _, o in res}
    labels = {(t, g): y for t, g, y, _ in jobs}
    ncls = {(t, g): C for t, g, _, C in jobs}
    summ = []
    for (exp, m, task, gran), grp in raw.groupby(["exp", "model", "task", "granularity"], sort=False):
        o = oof_all[(task, gran)]
        Pm = np.mean([o[f"{exp}|{m}|s{s}"] for s in cfg.benchmark["seeds"]], axis=0)
        y, C = labels[(task, gran)], ncls[(task, gran)]
        ens = scores(y, Pm.argmax(1), C)
        lo, hi = bootstrap_ci(y, Pm.argmax(1), C, B=B)
        rec = {"exp": exp, "family": grp.family.iloc[0], "text_enc": grp.text_enc.iloc[0], "image_enc": grp.image_enc.iloc[0],
               "model": m, "task": task, "granularity": gran}
        for k in ("acc", "precision", "recall", "macro_f1", "weighted_f1"):
            rec[f"{k}_mean"], rec[f"{k}_std"] = grp[k].mean(), grp[k].std(ddof=1) if len(grp) > 1 else 0.0
        rec.update({"ens_macro_f1": ens["macro_f1"], "ens_ci_lo": lo, "ens_ci_hi": hi})
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)

    # ---- significance: best fusion vs best text-only / image-only (selected by mean macro-F1)
    sig = []
    for (task, gran), grp in summ.groupby(["task", "granularity"], sort=False):
        best = {f: grp[grp.family == f].sort_values("macro_f1_mean", ascending=False).head(1) for f in ("fusion", "text", "image")}
        if any(len(v) == 0 for v in best.values()):
            continue
        o, y, C = oof_all[(task, gran)], labels[(task, gran)], ncls[(task, gran)]

        def ens_pred(r):
            return np.mean([o[f"{r.exp.iloc[0]}|{r.model.iloc[0]}|s{s}"] for s in cfg.benchmark["seeds"]], 0).argmax(1)

        for other in ("text", "image"):
            pb = paired_bootstrap(y, ens_pred(best["fusion"]), ens_pred(best[other]), C, B=B)
            sig.append({"task": task, "granularity": gran, "comparison": f"fusion_vs_{other}",
                        "fusion": f"{best['fusion'].exp.iloc[0]}|{best['fusion'].model.iloc[0]}",
                        "other": f"{best[other].exp.iloc[0]}|{best[other].model.iloc[0]}", **pb})
    sig = pd.DataFrame(sig)
    if len(sig):
        for (gran, comp), idx in sig.groupby(["granularity", "comparison"]).groups.items():
            p = sig.loc[idx, "p_one_sided"].to_numpy()
            order = np.argsort(p)
            adj = np.empty_like(p)
            running = 0.0
            for rank, k in enumerate(order):  # Holm step-down
                running = max(running, min(1.0, (len(p) - rank) * p[k]))
                adj[k] = running
            sig.loc[idx, "p_holm"] = adj
    sig.to_csv(out / "significance.csv", index=False)

    # ---- console overview
    view = summ.sort_values("macro_f1_mean", ascending=False).groupby(["granularity", "task", "family"]).head(1)
    print(view[["granularity", "task", "family", "exp", "model", "macro_f1_mean", "macro_f1_std", "ens_ci_lo", "ens_ci_hi"]]
          .sort_values(["granularity", "task", "family"]).to_string(index=False))


if __name__ == "__main__":
    main()
