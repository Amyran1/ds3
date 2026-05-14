from __future__ import annotations

import numpy as np

from projects.civic_shout_action_rate_increase.harness import (
    _weighted_rank_auc,
    _weighted_rank_auc_numba,
)


def _make_fixture(seed: int, n: int = 10_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.random(n).astype(np.float64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    sort_idx = np.argsort(s, kind="stable")
    y_sorted = y[sort_idx]
    s_sorted = s[sort_idx]
    w = np.ones(n, dtype=np.int64)
    return y_sorted, s_sorted, w


def test_numba_auc_matches_numpy_unit_weights():
    for seed in (42, 137, 2024):
        y, s, w = _make_fixture(seed)
        expected = _weighted_rank_auc(y, s, w)
        got = _weighted_rank_auc_numba(y, s, w)
        assert abs(got - expected) < 1e-9, f"seed={seed}: got={got} expected={expected}"


def test_numba_auc_matches_numpy_with_ties():
    rng = np.random.default_rng(42)
    n = 10_000
    s_raw = rng.integers(0, 5, size=n).astype(np.float64)
    y_raw = rng.integers(0, 2, size=n).astype(np.int64)
    sort_idx = np.argsort(s_raw, kind="stable")
    y = y_raw[sort_idx]
    s = s_raw[sort_idx]
    w = np.ones(n, dtype=np.int64)
    expected = _weighted_rank_auc(y, s, w)
    got = _weighted_rank_auc_numba(y, s, w)
    assert abs(got - expected) < 1e-9, f"got={got} expected={expected}"


def test_numba_auc_matches_numpy_random_weights():
    rng = np.random.default_rng(99)
    n = 10_000
    s_raw = rng.random(n).astype(np.float64)
    y_raw = rng.integers(0, 2, size=n).astype(np.int64)
    sort_idx = np.argsort(s_raw, kind="stable")
    y = y_raw[sort_idx]
    s = s_raw[sort_idx]
    w = rng.integers(0, 5, size=n).astype(np.int64)
    expected = _weighted_rank_auc(y, s, w)
    got = _weighted_rank_auc_numba(y, s, w)
    assert abs(got - expected) < 1e-9, f"got={got} expected={expected}"


def test_numba_auc_degenerate_returns_nan():
    n = 100
    s = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = np.ones(n, dtype=np.int64)

    y_all_pos = np.ones(n, dtype=np.int64)
    result_pos = _weighted_rank_auc_numba(y_all_pos, s, w)
    assert np.isnan(result_pos), f"all-positive should return nan, got {result_pos}"

    y_all_neg = np.zeros(n, dtype=np.int64)
    result_neg = _weighted_rank_auc_numba(y_all_neg, s, w)
    assert np.isnan(result_neg), f"all-negative should return nan, got {result_neg}"
