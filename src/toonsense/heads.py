"""Lightweight classification heads over frozen, pre-computed embeddings.

Unimodal : logreg, mlp                     (applied to text-only or image-only features)
Fusion   : concat_lr   early fusion, L2-regularised logistic regression on [t ; v]
           concat_mlp  early fusion MLP on [t ; v]
           gmu         Gated Multimodal Unit (Arevalo et al., 2017)
           bilinear    projected interaction features [p_t ; p_v ; p_t*p_v ; |p_t-p_v|]
           late        decision-level fusion: mean of text-MLP and image-MLP probabilities
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return w


# ------------------------------------------------------------------ torch modules
class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout / 2), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, t=None, v=None):
        x = t if v is None else (v if t is None else torch.cat([t, v], -1))
        return self.net(x)


class GMU(nn.Module):
    def __init__(self, d_t: int, d_v: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.drop_in = nn.Dropout(dropout / 2)
        self.ht = nn.Linear(d_t, hidden)
        self.hv = nn.Linear(d_v, hidden)
        self.z = nn.Linear(d_t + d_v, hidden)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        self.last_gate: torch.Tensor | None = None

    def forward(self, t, v):
        t, v = self.drop_in(t), self.drop_in(v)
        z = torch.sigmoid(self.z(torch.cat([t, v], -1)))
        self.last_gate = z.detach()
        h = z * torch.tanh(self.ht(t)) + (1 - z) * torch.tanh(self.hv(v))
        return self.out(h)


class Bilinear(nn.Module):
    def __init__(self, d_t: int, d_v: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.drop_in = nn.Dropout(dropout / 2)
        self.pt = nn.Linear(d_t, hidden)
        self.pv = nn.Linear(d_v, hidden)
        self.out = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, t, v):
        a, b = torch.tanh(self.pt(self.drop_in(t))), torch.tanh(self.pv(self.drop_in(v)))
        return self.out(torch.cat([a, b, a * b, (a - b).abs()], -1))


# ------------------------------------------------------------------ head wrapper
class Head:
    """Uniform fit / predict_proba interface. Inputs are raw (unscaled) features; scaling is internal."""

    def __init__(self, kind: str, modality: str, n_classes: int, hp: dict, seed: int):
        self.kind, self.modality, self.C, self.hp, self.seed = kind, modality, n_classes, hp, seed
        self.st: StandardScaler | None = None
        self.sv: StandardScaler | None = None

    # -- helpers
    def _uses_t(self) -> bool:
        return self.modality in ("text", "fusion")

    def _uses_v(self) -> bool:
        return self.modality in ("image", "fusion")

    def _scale(self, Xt, Xv, fit: bool = False):
        if fit:
            self.st = StandardScaler().fit(Xt) if self._uses_t() else None
            self.sv = StandardScaler().fit(Xv) if self._uses_v() else None
        t = self.st.transform(Xt).astype(np.float32) if self._uses_t() else None
        v = self.sv.transform(Xv).astype(np.float32) if self._uses_v() else None
        return t, v

    @staticmethod
    def _cat(t, v):
        return np.concatenate([x for x in (t, v) if x is not None], axis=1)

    def _build(self, d_t: int, d_v: int) -> nn.Module:
        h, p = int(self.hp["hidden"]), float(self.hp["dropout"])
        if self.kind == "gmu":
            return GMU(d_t, d_v, self.C, h, p)
        if self.kind == "bilinear":
            return Bilinear(d_t, d_v, self.C, h, p)
        return MLP(d_t + d_v, self.C, h, p)

    # -- api
    def fit(self, Xt, Xv, y, val_idx: np.ndarray):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        t, v = self._scale(None if Xt is None else Xt[tr_idx], None if Xv is None else Xv[tr_idx], fit=True)
        tv, vv = self._scale(None if Xt is None else Xt[val_idx], None if Xv is None else Xv[val_idx])
        ytr, yval = y[tr_idx], y[val_idx]

        if self.kind in ("logreg", "concat_lr"):
            best = (-1.0, None)
            for c in (0.01, 0.1, 1.0):
                m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced" if self.hp.get("class_weighting") else None)
                m.fit(self._cat(t, v), ytr)
                f1 = f1_score(yval, m.predict(self._cat(tv, vv)), average="macro", zero_division=0) if len(yval) else 0
                if f1 > best[0]:
                    best = (f1, c)
            self.model = LogisticRegression(
                C=best[1], max_iter=3000, class_weight="balanced" if self.hp.get("class_weighting") else None
            ).fit(self._cat(t, v), ytr)
            return self

        torch.manual_seed(self.seed)
        d_t = t.shape[1] if t is not None else 0
        d_v = v.shape[1] if v is not None else 0
        self.model = self._build(d_t, d_v)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.hp["lr"]), weight_decay=float(self.hp["weight_decay"]))
        w = torch.tensor(class_weights(ytr, self.C), dtype=torch.float32) if self.hp.get("class_weighting") else None
        T = lambda a: None if a is None else torch.from_numpy(a)  # noqa: E731
        tt, vt, yt = T(t), T(v), torch.from_numpy(ytr).long()
        tvt, vvt, yvt = T(tv), T(vv), torch.from_numpy(yval).long()
        best_loss, best_state, bad = float("inf"), None, 0
        for _ in range(int(self.hp["max_epochs"])):
            self.model.train()
            opt.zero_grad()
            loss = F.cross_entropy(self.model(tt, vt), yt, weight=w)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(self.model(tvt, vvt), yvt, weight=w).item() if len(yval) else loss.item()
            if vl < best_loss - 1e-4:
                best_loss, best_state, bad = vl, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= int(self.hp["patience"]):
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, Xt, Xv) -> np.ndarray:
        t, v = self._scale(Xt, Xv)
        if self.kind in ("logreg", "concat_lr"):
            P = np.zeros((len(t if t is not None else v), self.C))
            P[:, self.model.classes_] = self.model.predict_proba(self._cat(t, v))
            return P
        with torch.no_grad():
            T = lambda a: None if a is None else torch.from_numpy(a)  # noqa: E731
            return torch.softmax(self.model(T(t), T(v)), -1).numpy()

    # -- differentiable access for explainability (inputs already in raw feature space)
    def torch_logits(self, t_raw: torch.Tensor | None, v_raw: torch.Tensor | None) -> torch.Tensor:
        def sc(x, s):
            if x is None or s is None:
                return None
            return (x - torch.tensor(s.mean_, dtype=x.dtype)) / torch.tensor(s.scale_, dtype=x.dtype)

        t, v = sc(t_raw, self.st), sc(v_raw, self.sv)
        if self.kind in ("logreg", "concat_lr"):
            x = torch.cat([a for a in (t, v) if a is not None], -1)
            W = torch.tensor(self.model.coef_, dtype=x.dtype)
            b = torch.tensor(self.model.intercept_, dtype=x.dtype)
            if W.shape[0] == 1:  # binary sklearn LR
                z = x @ W.T + b
                return torch.cat([-z / 2, z / 2], -1)
            return x @ W.T + b
        return self.model(t, v)


MODEL_SPECS = {
    # name: (head kind, modality)
    "text_logreg": ("logreg", "text"),
    "text_mlp": ("mlp", "text"),
    "image_logreg": ("logreg", "image"),
    "image_mlp": ("mlp", "image"),
    "concat_lr": ("concat_lr", "fusion"),
    "concat_mlp": ("mlp", "fusion"),
    "gmu": ("gmu", "fusion"),
    "bilinear": ("bilinear", "fusion"),
}
