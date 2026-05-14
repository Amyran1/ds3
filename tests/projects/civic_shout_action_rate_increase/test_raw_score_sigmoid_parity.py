from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


def _make_binary_lgbm() -> tuple[LGBMClassifier, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    y = (rng.standard_normal(200) + X[:, 0] > 0).astype(int)
    model = LGBMClassifier(n_estimators=20, random_state=42, verbose=-1)
    model.fit(X, y)
    return model, X


def test_sigmoid_raw_matches_predict_proba() -> None:
    model, X = _make_binary_lgbm()
    proba_direct = model.predict_proba(X)[:, 1]
    raw = model.predict(X, raw_score=True)
    proba_derived = 1.0 / (1.0 + np.exp(-raw))
    np.testing.assert_allclose(proba_derived, proba_direct, atol=1e-6)


def test_auc_rank_invariant_under_sigmoid() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    raw_scores = rng.standard_normal(300) * 3.0
    sig_scores = 1.0 / (1.0 + np.exp(-raw_scores))
    auc_raw = roc_auc_score(y, raw_scores)
    auc_sig = roc_auc_score(y, sig_scores)
    assert abs(auc_raw - auc_sig) < 1e-12


def test_psi_bins_use_proba_not_raw() -> None:
    rng = np.random.default_rng(7)
    raw = rng.uniform(-10.0, 10.0, size=500)
    proba = 1.0 / (1.0 + np.exp(-raw))

    bins = np.linspace(0, 1, 11)
    bins[0], bins[-1] = -np.inf, np.inf
    counts_proba, _ = np.histogram(proba, bins=bins)

    assert np.all(counts_proba > 0), "all PSI bins should be populated when using proba"

    bins_unit = np.linspace(0, 1, 11)
    counts_raw, _ = np.histogram(raw, bins=bins_unit)
    assert np.sum(counts_raw == 0) > 0, "raw scores outside [0,1] produce empty PSI bins"
