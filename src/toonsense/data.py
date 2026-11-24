"""Dataset loading for any configured meme dataset (CMBan, BanglaAbuseMeme, MemoSen, ...).

Task taxonomies come from the active dataset's config entry (see `datasets:` in configs/default.yaml).
`TASKS` is a module-level dict that `activate()` refreshes *in place*, so modules that did
`from toonsense.data import TASKS` always see the active dataset's tasks.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Config

# task -> (label column, ordered fine-grained classes, fine->coarse mapping)
TASKS: dict[str, tuple[str, list[str], dict[str, str]]] = {}
GRANULARITIES: list[str] = []
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def activate(spec: dict) -> None:
    """Install the task taxonomy of a dataset spec (called by load_config)."""
    TASKS.clear()
    for task, t in spec["tasks"].items():
        fine = [str(c) for c in t["classes"]]
        over = {str(k): str(v) for k, v in (t.get("coarse") or {}).items()}
        TASKS[task] = (t["column"], fine, {c: over.get(c, c) for c in fine})
    GRANULARITIES[:] = list(spec.get("granularities", ["fine"]))


def classes_for(task: str, granularity: str) -> list[str]:
    _, fine, to_coarse = TASKS[task]
    if granularity == "fine":
        return fine
    seen: list[str] = []
    for c in fine:
        if to_coarse[c] not in seen:
            seen.append(to_coarse[c])
    return seen


def primary_granularity(cfg: Config) -> str:
    return cfg.data.get("primary_granularity", cfg.data.get("granularities", ["fine"])[-1])


def _norm(s: object) -> str:
    return " ".join(str(s).strip().split()).lower()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_image(name: str, dirs: list[Path]) -> Path | None:
    """`name` may be a bare id ('6578') or a file name ('image_1.jpg')."""
    for d in dirs:
        if Path(name).suffix.lower() in IMG_EXTS and (d / name).exists():
            return d / name
        for ext in IMG_EXTS:
            p = d / f"{name}{ext}"
            if p.exists():
                return p
    return None


def language_group(lang: str) -> str:
    """Collapse CMBan language tags into script/mixing groups used for stratified analysis."""
    parts = {p.strip().lower() for p in str(lang).split(",") if p.strip()}
    if parts == {"bangla"}:
        return "Bangla"
    if parts == {"english"}:
        return "English"
    return "Code-mixed"


_BN = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")


def language_group_from_text(text: str) -> str:
    """Script-based group for datasets without language tags.

    'Latin-script' covers both English and Romanised Bangla, which script alone cannot separate.
    """
    bn, la = len(_BN.findall(str(text))), len(_LATIN.findall(str(text)))
    if bn + la == 0:
        return "Unknown"
    r = bn / (bn + la)
    return "Bangla" if r >= 0.85 else "Latin-script" if r <= 0.15 else "Code-mixed"


@dataclass
class Dataset:
    frame: pd.DataFrame  # id, image_path, language, lang_group, [gold_text], <task>_fine, <task>_coarse

    @property
    def ids(self) -> list[str]:
        return self.frame["id"].tolist()

    @property
    def gold(self) -> dict[str, str] | None:
        """Human-written meme text, if the dataset provides it."""
        if "gold_text" not in self.frame:
            return None
        return dict(zip(self.frame["id"], self.frame["gold_text"]))

    def labels(self, task: str, granularity: str) -> np.ndarray:
        classes = classes_for(task, granularity)
        idx = {c: i for i, c in enumerate(classes)}
        return self.frame[f"{task}_{granularity}"].map(idx).to_numpy()

    def fill_lang_groups(self, texts: list[str]) -> None:
        """Derive language groups from text for datasets without a language column."""
        if (self.frame["lang_group"] == "Unknown").all():
            self.frame["lang_group"] = [language_group_from_text(t) for t in texts]

    def __len__(self) -> int:
        return len(self.frame)


def load_dataset(cfg: Config) -> Dataset:
    spec = cfg.data
    raw = _read_table(cfg.path("labels"))
    raw.columns = [str(c).strip() for c in raw.columns]
    id_col = spec["id_column"]
    lang_col = spec.get("language_column")
    text_col = spec.get("text_column")
    dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in cfg.paths["image_dirs"]]

    rows, missing, bad = [], [], 0
    for _, r in raw.iterrows():
        v = r[id_col]
        if pd.isna(v):
            continue
        name = str(int(v)) if isinstance(v, (int, float, np.integer)) else str(v).strip()
        img = find_image(name, dirs)
        if img is None:
            missing.append(name)
            continue
        # Keep the full file name when the label file gives one: some datasets (MemoSen) contain different
        # memes that share a stem, e.g. 'KAM (1).jpg' and 'KAM (1).png'.
        sid = name
        row = {"id": sid, "image_path": str(img)}
        if lang_col:
            row["language"] = str(r.get(lang_col, ""))
            row["lang_group"] = language_group(row["language"])
        else:
            row["language"], row["lang_group"] = "", "Unknown"
        if text_col:
            t = r.get(text_col, "")
            row["gold_text"] = "" if pd.isna(t) else str(t)
        ok = True
        for task, (col, fine, to_coarse) in TASKS.items():
            lookup = {_norm(c): c for c in fine}
            lab = lookup.get(_norm(r[col]))
            if lab is None:
                ok = False
                break
            row[f"{task}_fine"] = lab
            row[f"{task}_coarse"] = to_coarse[lab]
        if not ok:
            bad += 1
            continue
        rows.append(row)
    if missing:
        warnings.warn(f"{cfg.dataset_name}: {len(missing)} labelled rows have no image (e.g. {missing[:3]}); skipped")
    if bad:
        warnings.warn(f"{cfg.dataset_name}: {bad} rows with missing/unknown labels skipped")
    frame = pd.DataFrame(rows)
    n0 = len(frame)
    frame = frame.drop_duplicates("id").reset_index(drop=True)
    if len(frame) < n0:
        warnings.warn(f"{cfg.dataset_name}: {n0 - len(frame)} duplicate ids dropped")
    limit = int(os.environ.get("TOONSENSE_LIMIT", "0") or 0)  # smoke tests only
    if 0 < limit < len(frame):
        frame = frame.sample(n=limit, random_state=0).sort_index().reset_index(drop=True)
    return Dataset(frame)


def describe(ds: Dataset) -> pd.DataFrame:
    """Label distribution table (counts and %) for every task and granularity."""
    out = []
    for task in TASKS:
        for g in GRANULARITIES:
            vc = ds.frame[f"{task}_{g}"].value_counts()
            for c in classes_for(task, g):
                n = int(vc.get(c, 0))
                out.append({"task": task, "granularity": g, "class": c, "n": n, "pct": 100 * n / len(ds)})
    return pd.DataFrame(out)
