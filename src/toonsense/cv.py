"""Repeated stratified k-fold training with out-of-fold (OOF) evaluation on clean and perturbed inputs.

For each seed: StratifiedKFold(k) -> for each fold a stratified inner validation split is carved out of
the training portion (early stopping / C selection). Heads are trained on clean features only and
evaluated on the clean test fold and on every perturbed variant of the same test fold, so each
perturbation is compared against the *same* trained model. OOF probabilities cover every sample once
per seed, which makes all metrics, bootstrap CIs and robustness drops computable on the full dataset.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

from .heads import MODEL_SPECS, Head

warnings.filterwarnings("ignore", message=".*least populated class.*")
warnings.filterwarnings("ignore", category=FutureWarning)


def _inner_val(y_tr: np.ndarray, frac: float, seed: int) -> np.ndarray:
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        _, val = next(sss.split(np.zeros(len(y_tr)), y_tr))
    except ValueError:
        _, val = train_test_split(np.arange(len(y_tr)), test_size=frac, random_state=seed)
    return np.sort(val)


def run_cv(
    Xt: np.ndarray | None,
    Xv: np.ndarray | None,
    y: np.ndarray,
    n_classes: int,
    models: list[str],
    cfg_bench: dict,
    hp: dict,
    seed: int,
    variants: dict[str, tuple[np.ndarray | None, np.ndarray | None]] | None = None,
    return_heads: bool = False,
    augment: list[tuple[np.ndarray | None, np.ndarray | None]] | None = None,
):
    """Returns oof[model][variant] -> (N, C) probabilities ('clean' always present).

    `variants` maps a variant name to (Xt_variant, Xv_variant); None means 'use the clean features'.
    'late' fusion is derived from text_mlp and image_mlp (which are trained implicitly if needed).
    `augment` is a list of (Xt_aug, Xv_aug) feature sets aligned with `y` (None = clean features). In every fold, the
    augmented copies of the *training* memes (never the inner-validation or test memes) are appended to the
    training data with their original labels (train-time data augmentation).
    """
    variants = dict(variants or {})
    need = set(m for m in models if m != "late")
    if "late" in models:
        need |= {"text_mlp", "image_mlp"}
    N = len(y)
    names = ["clean"] + list(variants)
    oof = {m: {v: np.zeros((N, n_classes)) for v in names} for m in need | ({"late"} if "late" in models else set())}
    heads_out = []
    skf = StratifiedKFold(n_splits=int(cfg_bench["folds"]), shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        val = _inner_val(y[tr], float(cfg_bench["inner_val_fraction"]), seed + fold)
        fold_heads = {}
        Xt_fit = None if Xt is None else Xt[tr]
        Xv_fit = None if Xv is None else Xv[tr]
        y_fit = y[tr]
        if augment:
            fit_rows = tr[np.setdiff1d(np.arange(len(tr)), val)]
            for at, av in augment:
                at = Xt if at is None else at
                av = Xv if av is None else av
                Xt_fit = None if Xt is None else np.vstack([Xt_fit, at[fit_rows]])
                Xv_fit = None if Xv is None else np.vstack([Xv_fit, av[fit_rows]])
                y_fit = np.concatenate([y_fit, y[fit_rows]])
        for m in sorted(need):
            kind, modality = MODEL_SPECS[m]
            h = Head(kind, modality, n_classes, hp, seed=seed * 100 + fold)
            h.fit(Xt_fit, Xv_fit, y_fit, val)
            fold_heads[m] = h
            for v in names:
                xt, xv = (Xt, Xv) if v == "clean" else variants[v]
                xt = Xt if xt is None else xt
                xv = Xv if xv is None else xv
                oof[m][v][te] = h.predict_proba(
                    None if (xt is None or modality == "image") else xt[te],
                    None if (xv is None or modality == "text") else xv[te],
                )
        if return_heads:
            heads_out.append((tr, te, fold_heads))
    if "late" in models:
        for v in names:
            oof["late"][v] = 0.5 * (oof["text_mlp"][v] + oof["image_mlp"][v])
    keep = set(models)
    oof = {m: d for m, d in oof.items() if m in keep or m in ("text_mlp", "image_mlp") and "late" in keep}
    return (oof, heads_out) if return_heads else oof
