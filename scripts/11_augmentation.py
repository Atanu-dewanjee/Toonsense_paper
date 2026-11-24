"""Mitigation experiment: train-time Romanisation augmentation.

For the primary encoder pair, every training fold is augmented with transliterated copies of its training memes
(text features of the `translit_s2` variant, clean image features, original labels). The heads are then evaluated on
clean and on all linguistic perturbations of the test fold, exactly as in step 04. Romanised spelling noise is *not*
used for augmentation in the main setting, so it measures generalisation to an unseen Romanisation style.

Settings
  none            clean training data only (= step 04)
  translit        + translit_s2 copies
  translit+roman  + translit_s2 and roman_noise_s2 copies (upper bound: noise style seen in training)

Requires the cached text features of the perturbed variants (written by step 04).

    python scripts/11_augmentation.py --dataset cmban
Outputs results/<dataset>/augmentation/{raw,summary}.csv and tables/augmentation.tex
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import TASKS, classes_for, load_dataset, primary_granularity  # noqa: E402
from toonsense.encoders import load_features  # noqa: E402

MODELS = ["text_logreg", "text_mlp", "concat_lr", "concat_mlp", "gmu", "bilinear", "late"]
SETTINGS = {"none": [], "translit": ["translit_s2"], "translit+roman": ["translit_s2", "roman_noise_s2"]}
FAM = {"translit": "Translit.", "roman_noise": "Roman. noise", "bangla_noise": "Bangla noise", "codemix": "Code-mix",
       "emoji_add": "Emoji ins.", "emoji_remove": "Emoji rem."}
SHORT_NAME = {"cmban": "CMBan", "banglaabusememe": "BanglaAbuseMeme", "memosen": "MemoSen"}


def eval_task(task, y, C, Xt, Xv, var, setting, seed, bench, head, part):
    """One (task, augmentation setting, seed) job. Feature arrays come from the parent process; joblib passes large
    arrays to the workers as shared read-only memory maps, so the 19 feature sets are held in memory only once.
    The rows are written to `part` so that an interrupted run resumes without recomputing finished jobs."""
    import torch

    torch.set_num_threads(1)
    from sklearn.metrics import f1_score

    from toonsense.cv import run_cv

    augment = [(var[a], None) for a in SETTINGS[setting]]
    oof = run_cv(Xt, Xv, y, C, MODELS, bench, head, seed, variants={v: (x, None) for v, x in var.items()},
                 augment=augment)
    rows = []
    for m in MODELS:
        pc = oof[m]["clean"].argmax(1)
        rows.append({"task": task, "setting": setting, "model": m, "seed": seed, "variant": "clean", "family": "clean",
                     "severity": 0, "f1": f1_score(y, pc, average="macro", labels=range(C), zero_division=0), "flip": 0.0})
        for v in var:
            pp = oof[m][v].argmax(1)
            fam, sev = v.rsplit("_s", 1)
            rows.append({"task": task, "setting": setting, "model": m, "seed": seed, "variant": v, "family": fam,
                         "severity": int(sev), "f1": f1_score(y, pp, average="macro", labels=range(C), zero_division=0),
                         "flip": float((pp != pc).mean())})
    pd.DataFrame(rows).to_csv(part, index=False)
    return part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--jobs", type=int, default=4, help="parallel workers (keep low on machines with little RAM)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    ds = load_dataset(cfg)
    ids = ds.ids
    gran = primary_granularity(cfg)
    variants = [f"{f}_s{s}" for f in cfg.robustness["text"] for s in cfg.robustness["severities"]]
    out = cfg.path("results") / "augmentation"
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    t, i = cfg.primary["text"], cfg.primary["image"]
    Xt = load_features(cfg, "text", t, "clean", ids)
    Xv = load_features(cfg, "image", i, "clean", ids)
    var = {v: load_features(cfg, "text", t, v, ids) for v in variants}
    todo, done = [], []
    for task in TASKS:
        y, C = ds.labels(task, gran), len(classes_for(task, gran))
        for setting in SETTINGS:
            for seed in cfg.benchmark["seeds"]:
                part = parts / f"{task}__{setting}__s{seed}.csv"
                (done if part.exists() else todo).append((task, y, C, setting, seed, part))
    print(f"{len(done)} jobs already finished, {len(todo)} to run with {args.jobs} workers", flush=True)
    t0 = time.time()
    Parallel(n_jobs=args.jobs, verbose=10, max_nbytes="1M", mmap_mode="r")(
        delayed(eval_task)(task, y, C, Xt, Xv, var, setting, seed, cfg.benchmark, cfg.head, part)
        for task, y, C, setting, seed, part in todo)
    res = [pd.read_csv(j[-1]) for j in done + todo]
    print(f"augmentation experiment finished in {(time.time() - t0) / 60:.1f} min")
    raw = pd.concat(res, ignore_index=True)
    raw.to_csv(out / "raw.csv", index=False)

    # per (setting, model, family): mean over tasks, severities and seeds; RD relative to the same setting's clean F1
    clean = raw[raw.family == "clean"].groupby(["setting", "model", "task", "seed"]).f1.mean().rename("f1_clean")
    pert = raw[raw.family != "clean"].join(clean, on=["setting", "model", "task", "seed"])
    pert["rd_rel"] = 100 * (pert.f1_clean - pert.f1) / pert.f1_clean.clip(lower=1e-9)
    summ = pert.groupby(["setting", "model", "family"]).agg(f1=("f1", "mean"), rd_rel=("rd_rel", "mean"),
                                                             flip=("flip", "mean")).reset_index()
    cl = raw[raw.family == "clean"].groupby(["setting", "model"]).f1.mean().reset_index().assign(family="clean", rd_rel=0.0, flip=0.0)
    summ = pd.concat([cl, summ], ignore_index=True)
    summ.to_csv(out / "summary.csv", index=False)

    # table: model groups (text heads / fusion heads), macro-F1 on clean and perturbed data
    summ["group"] = np.where(summ.model.str.startswith("text"), "Text", "Fusion")
    g = summ.groupby(["group", "setting", "family"]).agg(f1=("f1", "mean"), rd=("rd_rel", "mean")).reset_index()
    fams = ["clean"] + [f for f in cfg.robustness["text"] if f in g.family.unique()]
    rows = []
    for grp in ("Text", "Fusion"):
        for setting in SETTINGS:
            sub = g[(g.group == grp) & (g.setting == setting)].set_index("family")
            row = {"Heads": grp, "Augmentation": {"none": "none", "translit": "+translit.",
                                                   "translit+roman": "+translit.\\,+\\,noise"}[setting]}
            for f in fams:
                row["Clean" if f == "clean" else FAM[f]] = (f"{100 * sub.loc[f, 'f1']:.1f}" if f == "clean" else
                                                            f"{100 * sub.loc[f, 'f1']:.1f} ({sub.loc[f, 'rd']:.0f})")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table.csv", index=False)
    body = df.to_latex(index=False, escape=False, column_format="ll" + "c" * (df.shape[1] - 2))
    name = SHORT_NAME.get(cfg.dataset_name, cfg.dataset_name)
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n"
           f"\\caption{{{name}: effect of train-time Romanisation augmentation. Macro-F1 (\\%) on clean and perturbed "
           "test folds, averaged over heads, tasks and severities; relative Robustness Drop (\\%) in parentheses. "
           "Romanised spelling noise is unseen in the \\emph{+translit.} setting.}}\n"
           f"\\label{{tab:aug_{cfg.dataset_name}}}\n\\begin{{adjustbox}}{{max width=\\textwidth}}\n{body}\\end{{adjustbox}}\n\\end{{table*}}\n")
    (cfg.path("results") / "tables").mkdir(parents=True, exist_ok=True)
    (cfg.path("results") / "tables" / "augmentation.tex").write_text(tex, encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
