"""Configuration loading, active-dataset resolution and global environment setup."""
from __future__ import annotations

import copy
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASET_ENV = "TOONSENSE_DATASET"


class Config(dict):
    """dict with attribute access and project-root-aware path resolution."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p

    @property
    def dataset_name(self) -> str:
        return self["dataset"]


def load_config(path: str | os.PathLike | None = None, dataset: str | None = None) -> Config:
    """Load the YAML config and activate one dataset.

    Dataset precedence: explicit argument > $TOONSENSE_DATASET > `dataset:` in the YAML. The choice is exported
    to the environment so that joblib worker processes (which re-load the config) see the same dataset.
    """
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    name = dataset or os.environ.get(DATASET_ENV) or raw.get("dataset", "cmban")
    if name not in raw["datasets"]:
        raise KeyError(f"unknown dataset {name!r}; available: {list(raw['datasets'])}")
    os.environ[DATASET_ENV] = name
    raw["dataset"] = name
    spec = copy.deepcopy(raw["datasets"][name])
    raw["paths"] = {k: (v.format(dataset=name) if isinstance(v, str) else v) for k, v in raw["paths"].items()}
    raw["paths"]["labels"] = spec["labels"]
    raw["paths"]["image_dirs"] = spec["image_dirs"]
    raw["data"] = spec
    cfg = Config(raw)
    setup_env(cfg)
    from . import data as _data  # local import: data depends on config

    _data.activate(spec)
    return cfg


def setup_env(cfg: Config) -> None:
    os.environ.setdefault("HF_HOME", str(cfg.path("hf_cache")))
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(cfg.path("easyocr_cache")))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:  # Windows consoles default to cp1252 and choke on Bangla
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_device(cfg: Config | None = None) -> str:
    want = (cfg or {}).get("device", "auto") if cfg is not None else "auto"
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if want == "auto":
        return "cuda" if has_cuda else "cpu"
    return want


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
