from __future__ import annotations

import numpy as np

from projects.civic_shout_action_rate_increase.harness import (
    _pooled_user_block_bootstrap_mean_of_folds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fold_results(
    rng: np.random.Generator,
    n_folds: int = 4,
    n_rows_per_fold: int = 800,
    n_users_per_fold: int = 60,
    n_shared_users: int = 20,
    pos_rate: float = 0.12,
) -> list[dict]:
    """Generate synthetic fold_results dicts with partial user overlap across folds."""
    base_users = np.arange(n_shared_users, dtype=np.int64)
    folds = []
    for f in range(n_folds):
        fold_private = np.arange(
            n_shared_users + f * (n_users_per_fold - n_shared_users),
            n_shared_users + (f + 1) * (n_users_per_fold - n_shared_users),
            dtype=np.int64,
        )
        fold_users_pool = np.concatenate([base_users, fold_private])
        user_ids = rng.choice(fold_users_pool, size=n_rows_per_fold, replace=True)
        y = (rng.random(size=n_rows_per_fold) < pos_rate).astype(np.float64)
        scores = rng.random(size=n_rows_per_fold)
        folds.append({"user_ids": user_ids, "y_test": y, "s_resid": scores})
    return folds


def _pooled_bootstrap_reference(
    fold_results: list[dict],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Reference implementation: per-replicate searchsorted+clip+mask (old code path)."""
    from projects.civic_shout_action_rate_increase.harness import (
        _build_sorted_user_map,
        _weighted_rank_auc_numba,
    )

    rng = np.random.default_rng(seed)
    all_user_ids = np.concatenate([r["user_ids"] for r in fold_results])
    unique_users = np.unique(all_user_ids)

    fold_sorted_data = []
    for r in fold_results:
        uids = r["user_ids"]
        fold_unique = np.unique(uids)
        y_s, s_s, fold_row_to_user_idx = _build_sorted_user_map(
            r["y_test"], r["s_resid"], uids, fold_unique
        )
        fold_sorted_data.append((y_s, s_s, fold_row_to_user_idx, fold_unique))

    child_seeds = rng.integers(0, 2**32 - 1, size=n_boot)

    results = []
    for rep_seed in child_seeds:
        rep_rng = np.random.default_rng(int(rep_seed))
        sampled_users = rep_rng.choice(unique_users, size=len(unique_users), replace=True)
        fold_aucs: list[float] = []
        for y_sorted, s_sorted_fold, fold_row_to_user_idx, fold_unique in fold_sorted_data:
            sampled_idx_in_fold = np.searchsorted(fold_unique, sampled_users)
            sampled_idx_clipped = np.clip(sampled_idx_in_fold, 0, len(fold_unique) - 1)
            in_fold_mask = fold_unique[sampled_idx_clipped] == sampled_users
            kept_idx = sampled_idx_in_fold[in_fold_mask]
            user_multiplicity = np.bincount(kept_idx, minlength=len(fold_unique))
            if user_multiplicity.sum() == 0:
                continue
            weights = user_multiplicity[fold_row_to_user_idx].astype(np.int64, copy=False)
            auc_b = _weighted_rank_auc_numba(
                y_sorted.astype(np.int64, copy=False),
                s_sorted_fold.astype(np.float64, copy=False),
                weights,
            )
            if not np.isnan(auc_b):
                fold_aucs.append(auc_b)
        results.append(float(np.mean(fold_aucs)) if fold_aucs else 0.5)

    replicate_stats = np.array(results, dtype=np.float64)
    return (
        float(np.nanquantile(replicate_stats, 0.025)),
        float(np.nanquantile(replicate_stats, 0.975)),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pooled_bootstrap_bit_for_bit_parity() -> None:
    """T3a: optimised path must agree with reference CI within MC tolerance.

    T3a replaced rng.choice+bincount with rng.multinomial — same distribution, different RNG
    sequence. Bit-for-bit parity no longer holds; tolerance = 0.045 (2× MC std error at B=200).
    """
    rng = np.random.default_rng(7)
    fold_results = _make_fold_results(rng, n_folds=3)

    n_boot = 200
    seed = 42

    ref_lo, ref_hi = _pooled_bootstrap_reference(fold_results, n_boot=n_boot, seed=seed)
    opt_lo, opt_hi = _pooled_user_block_bootstrap_mean_of_folds(
        fold_results, n_boot=n_boot, seed=seed, n_jobs=1, bootstrap_n_jobs=1
    )

    tol = 0.045
    assert abs(opt_lo - ref_lo) < tol, f"CI low mismatch: opt={opt_lo} ref={ref_lo}"
    assert abs(opt_hi - ref_hi) < tol, f"CI high mismatch: opt={opt_hi} ref={ref_hi}"


def test_pooled_bootstrap_ci_within_mc_tolerance_multiple_seeds() -> None:
    """CI bounds from optimised path must agree with reference within MC tolerance across seeds."""
    rng = np.random.default_rng(13)
    fold_results = _make_fold_results(rng, n_folds=4)

    n_boot = 500
    tol = 0.045

    for seed in [42, 137, 999]:
        ref_lo, ref_hi = _pooled_bootstrap_reference(fold_results, n_boot=n_boot, seed=seed)
        opt_lo, opt_hi = _pooled_user_block_bootstrap_mean_of_folds(
            fold_results, n_boot=n_boot, seed=seed, n_jobs=1, bootstrap_n_jobs=1
        )
        assert abs(opt_lo - ref_lo) < tol, (
            f"seed={seed} CI low diverged: opt={opt_lo:.6f} ref={ref_lo:.6f} tol={tol:.6f}"
        )
        assert abs(opt_hi - ref_hi) < tol, (
            f"seed={seed} CI high diverged: opt={opt_hi:.6f} ref={ref_hi:.6f} tol={tol:.6f}"
        )


def test_pooled_bootstrap_precomputed_mapping_no_searchsorted_in_inner_loop() -> None:
    """Smoke: optimised function completes without error on folds with disjoint user sets."""
    rng = np.random.default_rng(0)
    n_folds = 5
    n_rows = 300
    n_users = 50

    fold_results = []
    for f in range(n_folds):
        user_pool = np.arange(f * n_users, (f + 1) * n_users, dtype=np.int64)
        user_ids = rng.choice(user_pool, size=n_rows, replace=True)
        y = (rng.random(size=n_rows) < 0.1).astype(np.float64)
        scores = rng.random(size=n_rows)
        fold_results.append({"user_ids": user_ids, "y_test": y, "s_resid": scores})

    lo, hi = _pooled_user_block_bootstrap_mean_of_folds(
        fold_results, n_boot=100, seed=7, n_jobs=1, bootstrap_n_jobs=1
    )
    assert 0.0 <= lo <= hi <= 1.0, f"CI out of range: [{lo}, {hi}]"


def test_pooled_bootstrap_single_fold_matches_reference() -> None:
    """T3a: single-fold path must agree with reference CI within MC tolerance."""
    rng = np.random.default_rng(55)
    fold_results = _make_fold_results(
        rng, n_folds=1, n_rows_per_fold=600, n_users_per_fold=40, n_shared_users=0
    )

    n_boot = 300
    seed = 77

    ref_lo, ref_hi = _pooled_bootstrap_reference(fold_results, n_boot=n_boot, seed=seed)
    opt_lo, opt_hi = _pooled_user_block_bootstrap_mean_of_folds(
        fold_results, n_boot=n_boot, seed=seed, n_jobs=1, bootstrap_n_jobs=1
    )

    tol = 0.045
    assert abs(opt_lo - ref_lo) < tol, f"Single-fold CI low: opt={opt_lo} ref={ref_lo}"
    assert abs(opt_hi - ref_hi) < tol, f"Single-fold CI high: opt={opt_hi} ref={ref_hi}"
