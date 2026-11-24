"""Step 1: OCR every labelled meme (or every image with --all) and cache text + boxes."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toonsense.config import ROOT, load_config  # noqa: E402
from toonsense.data import IMG_EXTS, load_dataset  # noqa: E402
from toonsense.ocr import cache_path, load_texts, ocr_cached  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--all", action="store_true", help="also OCR unlabelled images")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    ds = load_dataset(cfg)
    todo = list(zip(ds.frame["id"], ds.frame["image_path"]))
    if args.all:
        seen = {i for i, _ in todo}
        for d in cfg.paths["image_dirs"]:
            for p in sorted((ROOT / d).iterdir()):
                if p.suffix.lower() in IMG_EXTS and p.stem not in seen:
                    seen.add(p.stem)
                    todo.append((p.stem, str(p)))
    pending = [(i, p) for i, p in todo if not cache_path(cfg, i).exists()]
    print(f"{len(todo)} images, {len(pending)} need OCR", flush=True)
    t0 = time.time()
    for k, (sid, path) in enumerate(pending, 1):
        frags = ocr_cached(cfg, sid, path)
        el = time.time() - t0
        print(f"[{k}/{len(pending)}] {sid}: {len(frags)} fragments  (eta {el / k * (len(pending) - k) / 60:.1f} min)", flush=True)
    texts = load_texts(cfg, ds.ids, ds.gold)
    out = cfg.path("ocr_dir") / "texts.csv"
    texts.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out}; empty texts: {(texts.text.str.len() == 0).sum()}, mean conf {texts.ocr_conf.mean():.3f}")


if __name__ == "__main__":
    main()
