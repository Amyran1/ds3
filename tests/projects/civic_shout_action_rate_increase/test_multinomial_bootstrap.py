from __future__ import annotations

import numpy as np
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    _pooled_user_block_bootstrap_mean_of_folds,
    _user_block_bootstrap,
)


def _make_toy_fold(n_users: int = 30, n_rows: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(0, n_users, size=n_rows)
    y = rng.integers(0, 2, size=n_rows)
    s_resid = rng.standard_normal(n_rows)
    return {"user_ids": user_ids, "y_test": y, "s_resid": s_resid}


def test_multinomial_distribution_unbiased() -> None:
    n_users = 20
    n_rows = 200
    n_boot = 10_000
    seed = 42

    rng_outer = np.random.default_rng(seed)
    unique_users = np.arange(n_users)
    child_seeds = rng_outer.integers(0, 2**32 - 1, size=n_boot)

    totals = np.zeros(n_users, dtype=np.float64)
    for s in child_seeds:
        rep_rng = np.random.default_rng(int(s))
        mult = rep_rng.multinomial(n_users, np.full(n_users, 1.0 / n_users))
        totals += mult

    empirical_mean = totals / n_boot
    # Each user should appear exactly once on average (n_users draws, n_users slots).
    np.testing.assert_allclose(empirical_mean, np.ones(n_users), atol=0.05)


def test_multinomial_vs_choice_ci_within_mc_tolerance() -> None:
    n_boot = 500
    toy = _make_toy_fold(n_users=40, n_rows=400, seed=7)
    user_ids = toy["user_ids"]
    y = toy["y_test"]
    s = toy["s_resid"]

    # New multinomial-based implementation (current harness code).
    new_aucs = _user_block_bootstrap(
        y, s, user_ids, n_boot=n_boot, seed=42, n_jobs=1, bootstrap_n_jobs=1
    )
    new_ci_low = float(np.nanquantile(new_aucs, 0.025))
    new_ci_high = float(np.nanquantile(new_aucs, 0.975))

    # Reference: re-implement old rng.choice + bincount path inline for comparison.
    rng_ref = np.random.default_rng(42)
    unique_users = np.unique(user_ids)
    n_users = len(unique_users)
    child_seeds_ref = rng_ref.integers(0, 2**32 - 1, size=n_boot)

    from projects.civic_shout_action_rate_increase.harness import (
        _build_sorted_user_map,
        _weighted_rank_auc_numba,
    )

    y_sorted, s_sorted, row_to_user_idx = _build_sorted_user_map(y, s, user_ids, unique_users)
    ref_aucs: list[float] = []
    for seed_val in child_seeds_ref:
        rep_rng = np.random.default_rng(int(seed_val))
        sampled_users = rep_rng.choice(unique_users, size=n_users, replace=True)
        sampled_idx = np.searchsorted(unique_users, sampled_users).astype(np.int32)
        mult = np.bincount(sampled_idx, minlength=n_users).astype(np.int32)
        weights = mult[row_to_user_idx].astype(np.int64)
        ref_aucs.append(
            _weighted_rank_auc_numba(
                y_sorted.astype(np.int64),
                s_sorted.astype(np.float64),
                weights,
            )
        )
    ref_ci_low = float(np.nanquantile(ref_aucs, 0.025))
    ref_ci_high = float(np.nanquantile(ref_aucs, 0.975))

    # MC std error for a quantile near 0.5 at B=500: sqrt(0.25/500) ≈ 0.022; tolerance = 2x.
    tol = 0.045
    assert abs(new_ci_low - ref_ci_low) < tol, (
        f"CI low shifted by {abs(new_ci_low - ref_ci_low):.4f}, tolerance {tol}"
    )
    assert abs(new_ci_high - ref_ci_high) < tol, (
        f"CI high shifted by {abs(new_ci_high - ref_ci_high):.4f}, tolerance {tol}"
    )
