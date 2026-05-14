from __future__ import annotations

import numpy as np

from projects.civic_shout_action_rate_increase.harness import (
    _user_block_bootstrap,
    _weighted_rank_auc_numba,
)


def _make_sorted_fixture(seed: int, n: int = 1_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.random(n).astype(np.float64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    sort_idx = np.argsort(s, kind="stable")
    return y[sort_idx], s[sort_idx]


def test_int32_kernel_matches_int64_kernel():
    y, s = _make_sorted_fixture(42)
    n = len(y)
    rng = np.random.default_rng(7)
    w_int32 = rng.integers(1, 5, size=n).astype(np.int32)
    w_int64 = w_int32.astype(np.int64)

    result_int32 = _weighted_rank_auc_numba(y, s, w_int32)
    result_int64 = _weighted_rank_auc_numba(y, s, w_int64)

    assert abs(result_int32 - result_int64) < 1e-12, (
        f"int32 result {result_int32} != int64 result {result_int64}"
    )


def test_int32_no_overflow_at_5pct_scale():
    n_users = 100_000
    rng = np.random.default_rng(13)
    y = rng.integers(0, 2, size=n_users).astype(np.int64)
    s = rng.random(n_users).astype(np.float64)
    sort_idx = np.argsort(s, kind="stable")
    y_sorted = y[sort_idx]
    s_sorted = s[sort_idx]
    weights = np.ones(n_users, dtype=np.int32)

    result = _weighted_rank_auc_numba(y_sorted, s_sorted, weights)
    assert np.isfinite(result), f"Expected finite AUC, got {result}"
    assert 0.0 < result < 1.0, f"AUC {result} out of [0, 1]"


def test_existing_kernel_parity_preserved():
    rng = np.random.default_rng(99)
    n_rows = 2_000
    n_users = 200

    s = rng.random(n_rows).astype(np.float64)
    y = rng.integers(0, 2, size=n_rows).astype(np.int64)
    user_ids = rng.integers(0, n_users, size=n_rows).astype(np.int64)

    result = _user_block_bootstrap(
        y=y,
        s_resid=s,
        user_ids=user_ids,
        n_boot=50,
        seed=42,
        n_jobs=1,
        bootstrap_n_jobs=1,
    )

    assert len(result) == 50
    assert np.all(np.isfinite(result))
    mean_auc = float(np.mean(result))
    assert 0.4 < mean_auc < 0.6, f"Bootstrap mean AUC {mean_auc} outside expected range"
