from __future__ import annotations

import math

import numpy as np

from projects.civic_shout_action_rate_increase.harness import (
    _rank_auc_numpy,
    _user_block_bootstrap,
    _weighted_rank_auc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_frame(
    rng: np.random.Generator,
    n_rows: int = 5000,
    n_users: int = 200,
    pos_rate: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, scores, user_ids) for a synthetic frame."""
    user_ids = rng.integers(0, n_users, size=n_rows)
    y = (rng.random(size=n_rows) < pos_rate).astype(np.float64)
    scores = rng.random(size=n_rows)
    return y, scores, user_ids


def _make_tied_frame(
    rng: np.random.Generator,
    n_rows: int = 2000,
    n_users: int = 100,
    pos_rate: float = 0.15,
    n_score_levels: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = rng.integers(0, n_users, size=n_rows)
    y = (rng.random(size=n_rows) < pos_rate).astype(np.float64)
    scores = rng.integers(0, n_score_levels, size=n_rows).astype(np.float64)
    return y, scores, user_ids


def _loop_bootstrap(
    y: np.ndarray,
    s_resid: np.ndarray,
    user_ids: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Reference T1 loop-based bootstrap for parity comparison."""
    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)

    sort_idx = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[sort_idx]
    boundaries = np.searchsorted(sorted_users, unique_users, side="left")
    boundaries_end = np.searchsorted(sorted_users, unique_users, side="right")
    user_to_rows = {
        u: sort_idx[boundaries[i] : boundaries_end[i]] for i, u in enumerate(unique_users)
    }

    child_seeds = rng.integers(0, 2**32 - 1, size=n_boot)
    results = []
    for rep_seed in child_seeds:
        rep_rng = np.random.default_rng(int(rep_seed))
        sampled_users = rep_rng.choice(unique_users, size=len(unique_users), replace=True)
        row_indices = np.concatenate([user_to_rows[u] for u in sampled_users])
        y_b = y[row_indices]
        s_b = s_resid[row_indices]
        if len(np.unique(y_b)) < 2:
            results.append(float("nan"))
        else:
            results.append(_rank_auc_numpy(y_b, s_b))
    return np.array(results, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_weighted_auc_matches_rank_auc_unit_weights() -> None:
    """_weighted_rank_auc with all-ones weights must match _rank_auc_numpy on the same data."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 500
        y = rng.integers(0, 2, size=n).astype(np.float64)
        scores = rng.random(size=n)
        if np.sum(y) == 0 or np.sum(y) == n:
            continue

        sort_idx = np.argsort(scores, kind="stable")
        y_sorted = y[sort_idx]
        s_sorted = scores[sort_idx]
        weights = np.ones(n, dtype=np.int64)

        result_weighted = _weighted_rank_auc(y_sorted, s_sorted, weights)
        result_loop = _rank_auc_numpy(y, scores)

        assert abs(result_weighted - result_loop) < 1e-12, (
            f"weighted={result_weighted} vs loop={result_loop}"
        )


def test_weighted_auc_unit_weights_with_ties() -> None:
    """Unit-weight _weighted_rank_auc matches _rank_auc_numpy when scores have many ties."""
    rng = np.random.default_rng(7)
    n = 1000
    y = rng.integers(0, 2, size=n).astype(np.float64)
    scores = rng.integers(0, 5, size=n).astype(np.float64)

    sort_idx = np.argsort(scores, kind="stable")
    y_sorted = y[sort_idx]
    s_sorted = scores[sort_idx]
    weights = np.ones(n, dtype=np.int64)

    result_weighted = _weighted_rank_auc(y_sorted, s_sorted, weights)
    result_loop = _rank_auc_numpy(y, scores)

    assert abs(result_weighted - result_loop) < 1e-12, (
        f"weighted={result_weighted} vs loop={result_loop}"
    )


def test_batched_bootstrap_matches_loop_seed42() -> None:
    """T2.1 batched bootstrap CI must match reference T1 loop CI to 1e-9 on seed=42."""
    rng = np.random.default_rng(0)
    y, scores, user_ids = _make_toy_frame(rng)

    n_boot = 500
    seed = 42

    loop_aucs = _loop_bootstrap(y, scores, user_ids, n_boot=n_boot, seed=seed)
    batched_aucs = _user_block_bootstrap(
        y, scores, user_ids, n_boot=n_boot, seed=seed, n_jobs=1, bootstrap_n_jobs=1
    )

    lo_loop = float(np.nanquantile(loop_aucs, 0.025))
    hi_loop = float(np.nanquantile(loop_aucs, 0.975))
    lo_batched = float(np.nanquantile(batched_aucs, 0.025))
    hi_batched = float(np.nanquantile(batched_aucs, 0.975))

    assert abs(lo_batched - lo_loop) < 1e-9, f"CI low mismatch: {lo_batched} vs {lo_loop}"
    assert abs(hi_batched - hi_loop) < 1e-9, f"CI high mismatch: {hi_batched} vs {hi_loop}"


def test_batched_bootstrap_matches_loop_seed137() -> None:
    """Regression hardening: seed=137 must also match to 1e-9."""
    rng = np.random.default_rng(0)
    y, scores, user_ids = _make_toy_frame(rng)

    n_boot = 500
    seed = 137

    loop_aucs = _loop_bootstrap(y, scores, user_ids, n_boot=n_boot, seed=seed)
    batched_aucs = _user_block_bootstrap(
        y, scores, user_ids, n_boot=n_boot, seed=seed, n_jobs=1, bootstrap_n_jobs=1
    )

    lo_loop = float(np.nanquantile(loop_aucs, 0.025))
    hi_loop = float(np.nanquantile(loop_aucs, 0.975))
    lo_batched = float(np.nanquantile(batched_aucs, 0.025))
    hi_batched = float(np.nanquantile(batched_aucs, 0.975))

    assert abs(lo_batched - lo_loop) < 1e-9, f"CI low mismatch: {lo_batched} vs {lo_loop}"
    assert abs(hi_batched - hi_loop) < 1e-9, f"CI high mismatch: {hi_batched} vs {hi_loop}"


def test_batched_bootstrap_matches_loop_ties() -> None:
    """Toy frame with many tied scores: CI endpoints must match to 1e-9."""
    rng = np.random.default_rng(0)
    y, scores, user_ids = _make_tied_frame(rng)

    n_boot = 500
    seed = 42

    loop_aucs = _loop_bootstrap(y, scores, user_ids, n_boot=n_boot, seed=seed)
    batched_aucs = _user_block_bootstrap(
        y, scores, user_ids, n_boot=n_boot, seed=seed, n_jobs=1, bootstrap_n_jobs=1
    )

    lo_loop = float(np.nanquantile(loop_aucs, 0.025))
    hi_loop = float(np.nanquantile(loop_aucs, 0.975))
    lo_batched = float(np.nanquantile(batched_aucs, 0.025))
    hi_batched = float(np.nanquantile(batched_aucs, 0.975))

    assert abs(lo_batched - lo_loop) < 1e-9, f"Ties CI low mismatch: {lo_batched} vs {lo_loop}"
    assert abs(hi_batched - hi_loop) < 1e-9, f"Ties CI high mismatch: {hi_batched} vs {hi_loop}"


def test_batched_bootstrap_degenerate_resample_returns_nan() -> None:
    """A weights vector with W_pos=0 must produce nan from _weighted_rank_auc."""
    y_sorted = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    s_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    weights_all_neg = np.array([0, 0, 1, 3, 2], dtype=np.int64)

    result = _weighted_rank_auc(y_sorted, s_sorted, weights_all_neg)
    assert math.isnan(result), f"Expected nan for W_pos=0, got {result}"

    weights_all_pos = np.array([2, 3, 0, 0, 0], dtype=np.int64)
    result2 = _weighted_rank_auc(y_sorted, s_sorted, weights_all_pos)
    assert math.isnan(result2), f"Expected nan for W_neg=0, got {result2}"
