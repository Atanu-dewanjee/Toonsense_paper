"""Controlled visual perturbations for meme images (severities 1..3).

  blur              Gaussian blur; sigma defined in 224-px-equivalent units so it is resolution independent
  crop              random crop keeping a fraction of the image area
  resize            down-sample to a low resolution then up-sample back (detail loss)
  jpeg              JPEG re-compression at decreasing quality
  text_occlusion    fill a fraction of OCR-detected text boxes with their median colour (removes rendered text)
  region_occlusion  grey patches over a fraction of the *non-text* area (removes pictorial evidence)
"""
from __future__ import annotations

import io
import random
import zlib

import numpy as np
from PIL import Image, ImageFilter

IMAGE_PERTURBATIONS = ["blur", "crop", "resize", "jpeg", "text_occlusion", "region_occlusion"]

PARAMS = {
    "blur": {1: 1.0, 2: 2.0, 3: 4.0},             # sigma at 224-px scale
    "crop": {1: 0.85, 2: 0.70, 3: 0.55},          # kept area fraction
    "resize": {1: 160, 2: 96, 3: 48},             # intermediate max side (px)
    "jpeg": {1: 30, 2: 15, 3: 5},                 # JPEG quality
    "text_occlusion": {1: 0.33, 2: 0.66, 3: 1.0}, # fraction of text boxes covered
    "region_occlusion": {1: 0.10, 2: 0.25, 3: 0.40},  # fraction of non-text area covered
}


def _rng(sample_id: str, family: str, severity: int) -> random.Random:
    return random.Random(zlib.crc32(f"{sample_id}|{family}|{severity}|img".encode()))


def _text_mask(size: tuple[int, int], boxes: list[list[float]], pad: int = 2) -> np.ndarray:
    w, h = size
    m = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(y0) - pad): min(h, int(y1) + pad), max(0, int(x0) - pad): min(w, int(x1) + pad)] = True
    return m


def perturb_image(
    img: Image.Image, family: str, severity: int, sample_id: str = "", boxes: list[list[float]] | None = None
) -> Image.Image:
    img = img.convert("RGB")
    rng = _rng(sample_id, family, severity)
    p = PARAMS[family][severity]
    w, h = img.size

    if family == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=p * max(w, h) / 224))

    if family == "crop":
        side = np.sqrt(p)
        cw, ch = max(1, round(w * side)), max(1, round(h * side))
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        return img.crop((x0, y0, x0 + cw, y0 + ch))

    if family == "resize":
        s = p / max(w, h)
        if s >= 1:
            return img
        small = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    if family == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=p)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if family == "text_occlusion":
        boxes = list(boxes or [])
        if not boxes:
            return img
        k = max(1, round(p * len(boxes)))
        chosen = rng.sample(boxes, k)
        arr = np.asarray(img).copy()
        for x0, y0, x1, y1 in chosen:
            xa, ya = max(0, int(x0) - 2), max(0, int(y0) - 2)
            xb, yb = min(w, int(x1) + 2), min(h, int(y1) + 2)
            if xb <= xa or yb <= ya:
                continue
            patch = arr[ya:yb, xa:xb].reshape(-1, 3)
            arr[ya:yb, xa:xb] = np.median(patch, axis=0).astype(np.uint8)
        return Image.fromarray(arr)

    if family == "region_occlusion":
        grid = 12
        tmask = _text_mask((w, h), boxes or [])
        cells = []
        for gy in range(grid):
            for gx in range(grid):
                xa, xb = gx * w // grid, (gx + 1) * w // grid
                ya, yb = gy * h // grid, (gy + 1) * h // grid
                if tmask[ya:yb, xa:xb].mean() < 0.2:
                    cells.append((xa, ya, xb, yb))
        if not cells:
            return img
        rng.shuffle(cells)
        target = p * len(cells)
        arr = np.asarray(img).copy()
        for xa, ya, xb, yb in cells[: max(1, round(target))]:
            arr[ya:yb, xa:xb] = 127
        return Image.fromarray(arr)

    raise ValueError(f"unknown image perturbation {family}")
