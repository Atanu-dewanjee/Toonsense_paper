"""Step 2: extract clean embeddings for every configured text and image encoder (cached)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from toonsense.config import load_config  # noqa: E402
from toonsense.data import load_dataset  # noqa: E402
from toonsense.encoders import ensure_features, free, has_features, load_encoder  # noqa: E402
from toonsense.ocr import load_texts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="dataset name from configs (default: config / $TOONSENSE_DATASET)")
    ap.add_argument("--text", nargs="*", default=None, help="text encoders (default: all in config)")
    ap.add_argument("--image", nargs="*", default=None, help="image encoders (default: all in config)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    ds = load_dataset(cfg)
    ids = ds.ids
    paths = ds.frame["image_path"].tolist()

    for name in args.text if args.text is not None else list(cfg.encoders["text"]):
        if has_features(cfg, "text", name, "clean"):
            print(f"[text] {name}: cached")
            continue
        t0 = time.time()
        texts = load_texts(cfg, ids, ds.gold)["text"].tolist()
        enc = load_encoder(cfg, "text", name)
        ensure_features(cfg, "text", enc, "clean", ids, lambda: texts)
        print(f"[text] {name}: done in {time.time() - t0:.0f}s", flush=True)
        free(enc)

    for name in args.image if args.image is not None else list(cfg.encoders["image"]):
        if has_features(cfg, "image", name, "clean"):
            print(f"[image] {name}: cached")
            continue
        t0 = time.time()
        enc = load_encoder(cfg, "image", name)
        ensure_features(cfg, "image", enc, "clean", ids, lambda: [Image.open(p).convert("RGB") for p in paths])
        print(f"[image] {name}: done in {time.time() - t0:.0f}s", flush=True)
        free(enc)


if __name__ == "__main__":
    main()
