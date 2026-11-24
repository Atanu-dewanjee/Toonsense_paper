"""Fast unit tests for perturbations, metrics and the CV engine (no model downloads)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toonsense.cv import run_cv  # noqa: E402
from toonsense.metrics import paired_bootstrap, robustness  # noqa: E402
from toonsense.perturb_image import IMAGE_PERTURBATIONS, perturb_image  # noqa: E402
from toonsense.perturb_text import TEXT_PERTURBATIONS, bn_to_roman, change_rate, perturb_text  # noqa: E402

TEXT = "ভাই আমার দুদিন ধরে পেটে আগুন! এটা লালবাজার control room? আমি তোমাকে ভালোবাসি 😂"


@pytest.mark.parametrize("word,expected", [
    ("লালবাজার", "lalbajar"), ("কলকাতা", "kolkata"), ("সরকার", "sorkar"), ("সময়", "somoy"),
    ("ভালোবাসি", "bhalobasi"), ("হয়েছে", "hoyeche"), ("আমরা", "amra"), ("জনগণ", "jonogon"),
])
def test_transliteration(word, expected):
    assert bn_to_roman(word) == expected


@pytest.mark.parametrize("family", TEXT_PERTURBATIONS)
def test_text_perturbations_deterministic_and_monotone(family):
    outs = [perturb_text(TEXT, family, s, "42") for s in (1, 2, 3)]
    assert outs == [perturb_text(TEXT, family, s, "42") for s in (1, 2, 3)]
    if family not in ("emoji_remove", "bangla_noise"):
        assert change_rate(TEXT, outs[2]) >= change_rate(TEXT, outs[0])
    assert change_rate(TEXT, outs[2]) > 0


def test_translit_full_removes_bangla_script():
    out = perturb_text(TEXT, "translit", 3, "1")
    assert not any("ঀ" <= ch <= "৿" for ch in out)
    assert "control room?" in out


def test_emoji_remove():
    assert "😂" not in perturb_text(TEXT, "emoji_remove", 1, "1")


@pytest.mark.parametrize("family", IMAGE_PERTURBATIONS)
def test_image_perturbations(family):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 200, 3), dtype=np.uint8))
    boxes = [[10, 10, 100, 40], [20, 200, 180, 240]]
    for s in (1, 2, 3):
        out = perturb_image(img, family, s, "7", boxes)
        assert out.mode == "RGB"
        if family != "crop":
            assert out.size == img.size
        assert np.asarray(out).tolist() == np.asarray(perturb_image(img, family, s, "7", boxes)).tolist()


def test_metrics():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    r = robustness(y, y.copy(), 1 - y, 2)
    assert r["f1_clean"] == 1 and r["flip_rate"] == 1 and r["rd_rel"] == 100
    pb = paired_bootstrap(y, y, 1 - y, 2, B=200)
    assert pb["diff"] == 1 and pb["p_one_sided"] < 0.05


def test_cv_engine_isolates_modalities():
    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 2, n)
    Xt = rng.normal(size=(n, 16)).astype("f4")
    Xt[:, 0] += 3 * y
    Xv = rng.normal(size=(n, 8)).astype("f4")
    noisy = Xt.copy()
    noisy[:, 0] = rng.normal(size=n)
    bench = {"folds": 4, "inner_val_fraction": 0.2}
    hp = {"hidden": 32, "dropout": 0.1, "lr": 1e-2, "weight_decay": 1e-2, "max_epochs": 100, "patience": 20, "class_weighting": True}
    oof = run_cv(Xt, Xv, y, 2, ["text_logreg", "image_mlp", "concat_mlp", "late"], bench, hp, 0, variants={"tn": (noisy, None)})
    acc = lambda P: (P.argmax(1) == y).mean()  # noqa: E731
    assert acc(oof["text_logreg"]["clean"]) > 0.8
    assert acc(oof["text_logreg"]["tn"]) < 0.75
    np.testing.assert_allclose(oof["image_mlp"]["clean"], oof["image_mlp"]["tn"])  # image model unaffected by text noise


def _toy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    Xt = rng.normal(size=(n, 16)).astype("f4")
    Xt[:, 0] += 3 * y
    Xv = rng.normal(size=(n, 8)).astype("f4")
    bench = {"folds": 4, "inner_val_fraction": 0.2}
    hp = {"hidden": 32, "dropout": 0.1, "lr": 1e-2, "weight_decay": 1e-2, "max_epochs": 100, "patience": 20, "class_weighting": True}
    return rng, y, Xt, Xv, bench, hp


def test_augmentation_uses_only_training_rows(monkeypatch):
    """Augmented copies must come from the fold's training rows only (never inner-validation or test rows)."""
    from toonsense import cv

    _, y, Xt, Xv, bench, hp = _toy()
    n = len(y)
    Xt[:, 1] = np.arange(n)                     # clean rows carry their id in column 1
    aug = Xt.copy()
    aug[:, 1] = np.arange(n) + 1000             # augmented rows carry id + 1000
    calls = []
    orig = cv.Head.fit

    def spy(self, t, v, yy, val_idx):
        calls.append((t[:, 1].copy(), np.asarray(val_idx)))
        return orig(self, t, v, yy, val_idx)

    monkeypatch.setattr(cv.Head, "fit", spy)
    run_cv(Xt, Xv, y, 2, ["text_logreg"], bench, hp, 0, augment=[(aug, None)])
    assert len(calls) == bench["folds"]
    for ids, val in calls:
        clean = ids[ids < 1000].astype(int)
        added = (ids[ids >= 1000] - 1000).astype(int)
        val_ids = ids[val].astype(int)
        assert np.all(val < len(clean))                          # validation indices point at clean rows
        assert set(added) == set(clean) - set(val_ids)           # copies of training rows, minus validation
        assert len(clean) < n                                    # test rows were held out of training


def test_augmentation_teaches_new_view():
    """A model trained with augmented copies of a shifted view becomes robust to that view."""
    rng, y, Xt, Xv, bench, hp = _toy(n=120, seed=1)
    shifted = Xt.copy()
    shifted[:, 0] = rng.normal(size=len(y))
    shifted[:, 2] = Xt[:, 0]                    # the signal moves to another dimension
    acc = lambda P: (P.argmax(1) == y).mean()  # noqa: E731
    base = run_cv(Xt, Xv, y, 2, ["text_logreg"], bench, hp, 0, variants={"sh": (shifted, None)})
    augd = run_cv(Xt, Xv, y, 2, ["text_logreg"], bench, hp, 0, variants={"sh": (shifted, None)}, augment=[(shifted, None)])
    assert acc(base["text_logreg"]["sh"]) < 0.75
    assert acc(augd["text_logreg"]["sh"]) > 0.8
    assert acc(augd["text_logreg"]["clean"]) > 0.8


def test_load_texts_falls_back_to_summary_csv(tmp_path):
    """Without per-image OCR JSONs, load_texts uses texts.csv; gold text keeps priority but OCR stats are kept."""
    import pandas as pd

    from toonsense.ocr import load_texts

    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    pd.DataFrame([{"id": "a", "text": "ocr a", "source": "ocr", "ocr_conf": 0.7, "n_frags": 3},
                  {"id": "b", "text": "gold b", "source": "gold", "ocr_conf": 0.4, "n_frags": 1}]).to_csv(ocr_dir / "texts.csv", index=False)

    class Cfg:
        ocr = {"min_confidence": 0.2}

        def path(self, key):
            return {"ocr_dir": ocr_dir, "manual_transcripts": tmp_path / "none.csv"}[key]

    t = load_texts(Cfg(), ["a", "b"], gold={"b": "caption b"}).set_index("id")
    assert t.loc["a", "text"] == "ocr a" and t.loc["a", "source"] == "ocr" and t.loc["a", "n_frags"] == 3
    assert t.loc["b", "text"] == "caption b" and t.loc["b", "source"] == "gold" and abs(t.loc["b", "ocr_conf"] - 0.4) < 1e-9
    with pytest.raises(FileNotFoundError):
        load_texts(Cfg(), ["missing"])
