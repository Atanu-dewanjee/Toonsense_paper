"""Bangla/English OCR for meme images (EasyOCR) with on-disk caching.

The meme text in CMBan is rendered inside the image, so OCR is the text modality's source.
Each image's OCR output (text fragments, boxes, confidences) is cached as JSON so that
(i) text features are reproducible and (ii) text-region occlusion can reuse the boxes.
A manual transcript file, when provided, overrides OCR text for the text modality.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config, get_device

_READER = None


def get_reader(cfg: Config):
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(list(cfg.ocr["languages"]), gpu=get_device(cfg) == "cuda", verbose=False)
    return _READER


def _reading_order(frags: list[dict]) -> list[dict]:
    """Sort fragments top-to-bottom, then left-to-right within a line band."""
    if not frags:
        return frags
    heights = [f["box"][3] - f["box"][1] for f in frags]
    band = max(8.0, float(np.median(heights)) * 0.6)
    return sorted(frags, key=lambda f: (round(f["box"][1] / band), f["box"][0]))


def run_ocr(cfg: Config, image: Image.Image | str | Path) -> list[dict]:
    """OCR one image -> fragments [{'box': [x0,y0,x1,y1] (original px), 'text', 'conf'}]."""
    img = Image.open(image) if not isinstance(image, Image.Image) else image
    img = img.convert("RGB")
    scale = 1.0
    max_side = int(cfg.ocr.get("max_side", 0) or 0)
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    res = get_reader(cfg).readtext(np.asarray(img))
    frags = []
    for pts, text, conf in res:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = [min(xs) / scale, min(ys) / scale, max(xs) / scale, max(ys) / scale]
        frags.append({"box": [float(v) for v in box], "text": str(text), "conf": float(conf)})
    return _reading_order(frags)


def fragments_to_text(frags: list[dict], min_conf: float) -> str:
    return " ".join(f["text"].strip() for f in frags if f["conf"] >= min_conf and f["text"].strip())


def cache_path(cfg: Config, sample_id: str) -> Path:
    return cfg.path("ocr_dir") / "json" / f"{sample_id}.json"


def ocr_cached(cfg: Config, sample_id: str, image_path: str) -> list[dict]:
    p = cache_path(cfg, sample_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    frags = run_ocr(cfg, image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(frags, ensure_ascii=False, indent=1), encoding="utf-8")
    return frags


def load_boxes(cfg: Config, sample_id: str) -> list[list[float]]:
    p = cache_path(cfg, sample_id)
    if not p.exists():
        return []
    return [f["box"] for f in json.loads(p.read_text(encoding="utf-8"))]


def load_texts(cfg: Config, ids: list[str], gold: dict[str, str] | None = None) -> pd.DataFrame:
    """Return DataFrame[id, text, source, ocr_conf, n_frags].

    Text priority: manual transcript file > dataset-provided gold text (`gold`) > OCR. OCR statistics are
    reported whenever an OCR cache exists, even if the text itself comes from a gold source.
    """
    gold = gold or {}
    min_conf = float(cfg.ocr["min_confidence"])
    manual: dict[str, str] = {}
    mp = cfg.path("manual_transcripts")
    if mp.exists():
        m = pd.read_csv(mp, dtype={"id": str})
        manual = {str(r.id): str(r.text) for r in m.itertuples() if isinstance(r.text, str) and r.text.strip()}
    # Fallback when the per-image JSON cache is unavailable (e.g. only texts.csv was copied back from a remote run).
    summary: dict[str, dict] = {}
    sp = cfg.path("ocr_dir") / "texts.csv"
    if sp.exists():
        s = pd.read_csv(sp, dtype={"id": str}, keep_default_na=False)
        summary = {str(r["id"]): r for r in s.to_dict("records")}
    rows = []
    for sid in ids:
        p = cache_path(cfg, sid)
        frags = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        r = summary.get(sid)
        if frags is None and r is not None and (sid in manual or sid in gold or r.get("source", "ocr") == "ocr"):
            src = "manual" if sid in manual else "gold" if sid in gold else "ocr"
            rows.append({"id": sid, "text": manual[sid] if src == "manual" else gold[sid] if src == "gold" else str(r["text"]),
                         "source": src, "ocr_conf": float(r["ocr_conf"] or 0.0), "n_frags": int(r["n_frags"] or 0)})
            continue
        if frags is None and sid not in manual and sid not in gold:
            raise FileNotFoundError(f"No OCR cache for {sid}; run scripts/01_ocr.py first")
        frags = frags or []
        confs = [f["conf"] for f in frags if f["conf"] >= min_conf]
        rows.append(
            {
                "id": sid,
                "text": manual[sid] if sid in manual else gold[sid] if sid in gold else fragments_to_text(frags, min_conf),
                "source": "manual" if sid in manual else "gold" if sid in gold else "ocr",
                "ocr_conf": float(np.mean(confs)) if confs else 0.0,
                "n_frags": len(confs),
            }
        )
    return pd.DataFrame(rows)
