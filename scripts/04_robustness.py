"""Step 4: robustness evaluation under controlled linguistic and visual perturbations.

For each encoder pair, heads are trained on clean data (same repeated CV as the benchmark) and the
identical fold models are evaluated on perturbed copies of the test folds. Decoupled setting:
text perturbations change only the text input, visual perturbations only the image input.
Optional coupled setting (--coupled FAMILY ...): the text of a visually perturbed meme is re-OCRed
from the perturbed image, as it would be in deployment.

Outputs (results/robustness/):
  perturbed_texts.csv   every text variant + token-level coverage
  raw.csv               (pair, model, task, granularity, seed, variant, group) robustness metrics
  summary.csv           mean over seeds
  preds/*.npz           OOF predictions for every variant (used by error analysis / explainability)
  examples/             example perturbed images for figures
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)
os.environ["PYTHONPATH"] = _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import TASKS, classes_for, load_dataset, primary_granularity  # noqa: E402
from toonsense.encoders import feature_path, free, has_features, load_encoder, load_features, save_features  # noqa: E402
from toonsense.ocr import fragments_to_text, load_boxes, load_texts, run_ocr  # noqa: E402
from toonsense.perturb_image import perturb_image  # noqa: E402
from toonsense.perturb_text import change_rate, perturb_text  # noqa: E402

MODELS = ["text_logreg", "text_mlp", "image_logreg", "image_mlp", "concat_lr", "concat_mlp", "gmu", "bilinear", "late"]


def variant_name(fam: str, sev: int) -> str:
    return f"{fam}_s{sev}"


def make_text_variants(cfg, ids, texts) -> pd.DataFrame:
    rows = []
    for fam in cfg.robustness["text"]:
        for sev in cfg.robustness["severities"]:
            for sid, t in zip(ids, texts):
                pt = perturb_text(t, fam, sev, sid)
                rows.append({"id": sid, "variant": variant_name(fam, sev), "family": fam, "severity": sev,
                             "text": pt, "coverage": change_rate(t, pt)})
    return pd.DataFrame(rows)


def encode_images_streaming(enc, paths, ids, fam, sev, boxes, bs=16) -> np.ndarray:
    out = []
    for k in range(0, len(paths), bs):
        imgs = [perturb_image(Image.open(p), fam, sev, sid, boxes[sid]) for p, sid in zip(paths[k:k + bs], ids[k:k + bs])]
        out.append(enc.encode(imgs, bs=bs))
    return np.concatenate(out)


def coupled_texts(cfg, ids, paths, boxes, fam, sev) -> list[str]:
    """Re-OCR perturbed images (cached per variant)."""
    import json

    d = cfg.path("ocr_dir") / "coupled" / variant_name(fam, sev)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for sid, p in zip(ids, paths):
        f = d / f"{sid}.json"
        if f.exists():
            frags = json.loads(f.read_text(encoding="utf-8"))
        else:
            frags = run_ocr(cfg, perturb_image(Image.open(p), fam, sev, sid, boxes[sid]))
            f.write_text(json.dumps(frags, ensure_ascii=False), encoding="utf-8")
        out.append(fragments_to_text(frags, float(cfg.ocr["min_confidence"])))
    return out


def eval_task(cfg_path, task, gran, pairs, ids, groups, y, C, variant_meta):
    import torch

    torch.set_num_threads(1)
    from toonsense.config import load_config as _lc
    from toonsense.cv import run_cv
    from toonsense.metrics import robustness

    cfg = _lc(cfg_path)
    rows, preds = [], {}
    for t, i in pairs:
        Xt = load_features(cfg, "text", t, "clean", ids)
        Xv = load_features(cfg, "image", i, "clean", ids)
        variants = {}
        for v, (mod, fam, sev, setting) in variant_meta.items():
            if mod == "text":
                variants[v] = (load_features(cfg, "text", t, v, ids), None)
            elif setting == "coupled":
                variants[v] = (load_features(cfg, "text", t, v, ids), load_features(cfg, "image", i, v.replace("coupled_", ""), ids))
            else:
                variants[v] = (None, load_features(cfg, "image", i, v, ids))
        for seed in cfg.benchmark["seeds"]:
            oof = run_cv(Xt, Xv, y, C, MODELS, cfg.benchmark, cfg.head, seed, variants=variants)
            for m in MODELS:
                pc = oof[m]["clean"].argmax(1)
                preds[f"{t}+{i}|{m}|clean|s{seed}"] = pc.astype(np.int8)
                for v, (mod, fam, sev, setting) in variant_meta.items():
                    pp = oof[m][v].argmax(1)
                    preds[f"{t}+{i}|{m}|{v}|s{seed}"] = pp.astype(np.int8)
                    for g in ["all"] + sorted(set(groups)):
                        mask = np.ones(len(y), bool) if g == "all" else np.array(groups) == g
                        if mask.sum() < 10:
                            continue
                        rows.append({"pair": f"{t}+{i}", "text_enc": t, "image_enc": i, "model": m, "task": task,
                                     "granularity": gran, "seed": seed, "variant": v, "modality": mod, "family": fam,
                                     "severity": sev, "setting": setting, "group": g, "n": int(mask.sum()),
                                     **robustness(y[mask], pc[mask], pp[mask], C)})
    return task, gran, rows, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--text", nargs="*", default=None, help="text encoders (default: primary)")
    ap.add_argument("--image", nargs="*", default=None, help="image encoders (default: primary)")
    ap.add_argument("--coupled", nargs="*", default=None, help="visual families to re-OCR (slow), e.g. blur jpeg")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    ds = load_dataset(cfg)
    ids = ds.ids
    paths = ds.frame["image_path"].tolist()
    texts = load_texts(cfg, ids, ds.gold)["text"].tolist()
    ds.fill_lang_groups(texts)
    groups = ds.frame["lang_group"].tolist()
    boxes = {sid: load_boxes(cfg, sid) for sid in ids}
    t_encs = args.text or [cfg.primary["text"]]
    i_encs = args.image or [cfg.primary["image"]]
    coupled = args.coupled if args.coupled is not None else (list(cfg.robustness["image"]) if cfg.robustness.get("coupled_ocr") else [])
    sevs = list(cfg.robustness["severities"])
    out = cfg.path("results") / "robustness"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(parents=True, exist_ok=True)

    # ---------------- 1. perturbed texts
    tv = make_text_variants(cfg, ids, texts)
    tv.to_csv(out / "perturbed_texts.csv", index=False, encoding="utf-8")
    print(tv.groupby("variant").coverage.mean().round(3).to_string())

    variant_meta: dict[str, tuple] = {}
    for fam in cfg.robustness["text"]:
        for s in sevs:
            variant_meta[variant_name(fam, s)] = ("text", fam, s, "decoupled")
    for fam in cfg.robustness["image"]:
        for s in sevs:
            variant_meta[variant_name(fam, s)] = ("image", fam, s, "decoupled")
    coupled_txt = {}
    for fam in coupled:
        for s in sevs:
            v = "coupled_" + variant_name(fam, s)
            variant_meta[v] = ("image+ocr", fam, s, "coupled")
            t0 = time.time()
            coupled_txt[v] = coupled_texts(cfg, ids, paths, boxes, fam, s)
            print(f"coupled OCR {v}: {time.time() - t0:.0f}s", flush=True)

    # ---------------- 2. features for every variant
    for t in t_encs:
        todo = [v for v, m in variant_meta.items() if m[0] != "image" and not has_features(cfg, "text", t, v)]
        if todo:
            enc = load_encoder(cfg, "text", t)
            for v in todo:
                src = coupled_txt[v] if v.startswith("coupled_") else tv[tv.variant == v].set_index("id").loc[ids, "text"].tolist()
                save_features(cfg, "text", t, v, ids, enc.encode(src))
            free(enc)
        print(f"[text] {t}: {len(todo)} variants encoded", flush=True)
    for i in i_encs:
        todo = [v for v, m in variant_meta.items() if m[0] == "image" and not has_features(cfg, "image", i, v)]
        if todo:
            enc = load_encoder(cfg, "image", i)
            for v in todo:
                _, fam, s, _ = variant_meta[v]
                t0 = time.time()
                save_features(cfg, "image", i, v, ids, encode_images_streaming(enc, paths, ids, fam, s, boxes))
                print(f"  [image] {i} {v}: {time.time() - t0:.0f}s", flush=True)
            free(enc)
        print(f"[image] {i}: {len(todo)} variants encoded", flush=True)

    # example images for figures (first meme with text boxes)
    ex = next((k for k, sid in enumerate(ids) if len(boxes[sid]) >= 4), 0)
    for fam in cfg.robustness["image"]:
        for s in sevs:
            perturb_image(Image.open(paths[ex]), fam, s, ids[ex], boxes[ids[ex]]).save(out / "examples" / f"{ids[ex]}_{fam}_s{s}.jpg", quality=92)
    Image.open(paths[ex]).convert("RGB").save(out / "examples" / f"{ids[ex]}_clean.jpg", quality=92)

    # ---------------- 3. evaluation
    pairs = list(itertools.product(t_encs, i_encs))
    jobs = [(task, gran, ds.labels(task, gran), len(classes_for(task, gran))) for gran in cfg.data["granularities"] for task in TASKS]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(eval_task)(args.config, task, gran, pairs, ids, groups, y, C, variant_meta) for task, gran, y, C in jobs
    )
    print(f"robustness evaluation finished in {(time.time() - t0) / 60:.1f} min")
    rows = []
    for task, gran, r, preds in res:
        rows += r
        np.savez_compressed(out / "preds" / f"{gran}_{task}.npz", ids=np.array(ids), **preds)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw.csv", index=False)
    keys = ["pair", "model", "task", "granularity", "variant", "modality", "family", "severity", "setting", "group"]
    summ = raw.groupby(keys, sort=False)[["f1_clean", "f1_pert", "rd_abs", "rd_rel", "acc_drop", "flip_rate"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    view = summ[(summ.group == "all") & (summ.granularity == primary_granularity(cfg))]
    print(view.pivot_table(index="model", columns="family", values="rd_rel_mean", aggfunc="mean").round(1).to_string())


if __name__ == "__main__":
    main()
