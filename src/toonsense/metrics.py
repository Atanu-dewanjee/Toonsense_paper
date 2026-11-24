"""Classification metrics, bootstrap confidence intervals, significance tests and robustness measures."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def scores(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[str, float]:
    labels = list(range(n_classes))
    kw = dict(labels=labels, average="macro", zero_division=0)
    return {
        "acc": accuracy_score(y, pred),
        "precision": precision_score(y, pred, **kw),
        "recall": recall_score(y, pred, **kw),
        "macro_f1": f1_score(y, pred, **kw),
        "weighted_f1": f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
    }


def macro_f1(y, pred, n_classes):
    return f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def bootstrap_ci(y, pred, n_classes, B: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = [macro_f1(y[i], pred[i], n_classes) for i in (rng.integers(0, n, n) for _ in range(B))]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(y, pred_a, pred_b, n_classes, B: int = 1000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of macro-F1(a) - macro-F1(b). Returns observed diff, 95% CI and one-sided p (H1: a > b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = macro_f1(y, pred_a, n_classes) - macro_f1(y, pred_b, n_classes)
    diffs = np.empty(B)
    for k in range(B):
        i = rng.integers(0, n, n)
        diffs[k] = macro_f1(y[i], pred_a[i], n_classes) - macro_f1(y[i], pred_b[i], n_classes)
    return {
        "diff": float(obs),
        "ci_lo": float(np.quantile(diffs, 0.025)),
        "ci_hi": float(np.quantile(diffs, 0.975)),
        "p_one_sided": float((np.sum(diffs <= 0) + 1) / (B + 1)),
    }


def robustness(y, pred_clean, pred_pert, n_classes) -> dict[str, float]:
    """Robustness Drop (absolute and relative macro-F1 drop), accuracy drop, and prediction flip rate."""
    f_c, f_p = macro_f1(y, pred_clean, n_classes), macro_f1(y, pred_pert, n_classes)
    a_c, a_p = float(np.mean(y == pred_clean)), float(np.mean(y == pred_pert))
    return {
        "f1_clean": f_c,
        "f1_pert": f_p,
        "rd_abs": f_c - f_p,
        "rd_rel": 100 * (f_c - f_p) / f_c if f_c > 0 else np.nan,
        "acc_drop": a_c - a_p,
        "flip_rate": float(np.mean(pred_clean != pred_pert)),
    }
