from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from projects.civic_shout_action_rate_increase.harness import _rank_auc_numpy


def _random_pair(rng: np.random.Generator, n: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=n).astype(np.float64)
    scores = rng.random(size=n)
    return y, scores


def test_rank_auc_matches_sklearn_random() -> None:
    for seed in (42, 137, 2024):
        rng = np.random.default_rng(seed)
        y, scores = _random_pair(rng)
        assert abs(_rank_auc_numpy(y, scores) - roc_auc_score(y, scores)) < 1e-12


def test_rank_auc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000).astype(np.float64)
    scores = np.random.default_rng(1).integers(0, 5, size=1000).astype(np.float64)
    assert abs(_rank_auc_numpy(y, scores) - roc_auc_score(y, scores)) < 1e-12


def test_rank_auc_handles_all_positive() -> None:
    y = np.ones(50, dtype=np.float64)
    scores = np.random.default_rng(7).random(size=50)
    result = _rank_auc_numpy(y, scores)
    assert math.isnan(result)


def test_rank_auc_handles_all_negative() -> None:
    y = np.zeros(50, dtype=np.float64)
    scores = np.random.default_rng(7).random(size=50)
    result = _rank_auc_numpy(y, scores)
    assert math.isnan(result)


def test_rank_auc_raises_on_nan_in_scores() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    scores = np.array([0.1, float("nan"), 0.3, 0.9])
    with pytest.raises(ValueError, match="NaN"):
        _rank_auc_numpy(y, scores)


def test_rank_auc_raises_on_nan_in_labels() -> None:
    y = np.array([0.0, float("nan"), 0.0, 1.0])
    scores = np.array([0.1, 0.5, 0.3, 0.9])
    with pytest.raises(ValueError, match="NaN"):
        _rank_auc_numpy(y, scores)
