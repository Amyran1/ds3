from __future__ import annotations

import mlx.core as mx
import numpy as np


def gpu_auc_single(scores: mx.array, labels: mx.array) -> float:
    """Sort-based ROC AUC for one sample.

    Equivalent to sklearn.metrics.roc_auc_score — O(N log N) via Wilcoxon rank sum.
    Ascending sort: rank 1 = lowest score. AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    """
    n = scores.shape[0]
    # Ascending sort — rank 1 is the lowest score (same convention as scipy wilcoxon).
    sorted_idx = mx.argsort(scores)
    sorted_labels = mx.take(labels.astype(mx.float32), sorted_idx)

    n_pos = float(sorted_labels.sum().item())
    n_neg = float(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = mx.arange(1, n + 1, dtype=mx.float32)
    rank_sum_pos = float((ranks * sorted_labels).sum().item())
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def gpu_auc_batch(scores: mx.array, labels: mx.array) -> mx.array:
    """Batched ROC AUC via sort + rank sum.

    scores: (B, N) float32 — B batches of N scores
    labels: (B, N) int8 or float32 — B batches of N binary labels (0/1)
    Returns: (B,) float32 AUC per batch.

    Ascending sort: rank 1 = lowest score. Same Wilcoxon rank-sum convention as sklearn.
    """
    sorted_idx = mx.argsort(scores, axis=-1)
    sorted_labels = mx.take_along_axis(labels.astype(mx.float32), sorted_idx, axis=-1)

    n = scores.shape[-1]
    n_pos = sorted_labels.sum(axis=-1)
    n_neg = n - n_pos

    ranks = mx.arange(1, n + 1, dtype=mx.float32).reshape(1, -1)
    rank_sum_pos = (ranks * sorted_labels).sum(axis=-1)

    safe_denom = mx.maximum(n_pos * n_neg, mx.array(1e-12, dtype=mx.float32))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / safe_denom
    auc = mx.where(n_pos * n_neg > 0, auc, mx.array(0.5, dtype=mx.float32))
    mx.eval(auc)
    return auc


def gpu_user_block_bootstrap(
    y: np.ndarray,
    s: np.ndarray,
    user_ids: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """GPU-accelerated user-block bootstrap AUC.

    Same contract as harness.py:_user_block_bootstrap — returns array of n_boot
    bootstrap AUC values. Unique users are resampled with replacement; all rows
    for sampled users are concatenated. AUC computed via sort-based rank-sum on GPU.

    Uses numpy to pre-spawn child seeds (matching the CPU pattern at harness.py line 740).
    """
    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)
    n_users = len(unique_users)

    sort_idx = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[sort_idx]
    boundaries = np.searchsorted(sorted_users, unique_users, side="left")
    boundaries_end = np.searchsorted(sorted_users, unique_users, side="right")
    user_to_rows: dict[int, np.ndarray] = {
        int(u): sort_idx[boundaries[i] : boundaries_end[i]] for i, u in enumerate(unique_users)
    }

    child_seeds = rng.integers(0, 2**32 - 1, size=n_boot)

    results = np.empty(n_boot, dtype=np.float64)
    for b, rep_seed in enumerate(child_seeds):
        rep_rng = np.random.default_rng(int(rep_seed))
        sampled_users = rep_rng.choice(unique_users, size=n_users, replace=True)
        row_indices = np.concatenate([user_to_rows[int(u)] for u in sampled_users])
        y_b = y[row_indices]
        s_b = s[row_indices]
        if len(np.unique(y_b)) < 2:
            results[b] = 0.5
            continue
        y_mx = mx.array(y_b.astype(np.float32))
        s_mx = mx.array(s_b.astype(np.float32))
        results[b] = gpu_auc_single(s_mx, y_mx)

    return results


def gpu_pooled_bootstrap(
    fold_results: list[dict],
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """GPU-accelerated cluster-bootstrap of the mean-across-folds AUC estimator.

    Same contract as harness.py:_pooled_user_block_bootstrap_mean_of_folds —
    returns (ci_low_2.5%, ci_high_97.5%). Each replicate resamples users globally
    with replacement and recomputes the mean fold AUC via GPU sort-based rank-sum AUC.
    """
    rng = np.random.default_rng(seed)

    all_user_ids = np.concatenate([r["user_ids"] for r in fold_results])
    unique_users = np.unique(all_user_ids)
    n_users = len(unique_users)

    fold_user_to_rows: list[dict[int, np.ndarray]] = []
    for r in fold_results:
        uids = r["user_ids"]
        sort_idx = np.argsort(uids, kind="stable")
        sorted_uids = uids[sort_idx]
        fold_unique = np.unique(uids)
        starts = np.searchsorted(sorted_uids, fold_unique, side="left")
        ends = np.searchsorted(sorted_uids, fold_unique, side="right")
        fold_user_to_rows.append(
            {int(u): sort_idx[starts[i] : ends[i]] for i, u in enumerate(fold_unique)}
        )

    child_seeds = rng.integers(0, 2**32 - 1, size=n_boot)

    fold_ys = [r["y_test"] for r in fold_results]
    fold_ss = [r["s_resid"] for r in fold_results]

    replicate_stats = np.empty(n_boot, dtype=np.float64)
    for b, rep_seed in enumerate(child_seeds):
        rep_rng = np.random.default_rng(int(rep_seed))
        sampled_users = rep_rng.choice(unique_users, size=n_users, replace=True)
        fold_aucs: list[float] = []
        for y_fold, s_fold, u_to_rows in zip(fold_ys, fold_ss, fold_user_to_rows):
            chunks = [u_to_rows[int(u)] for u in sampled_users if int(u) in u_to_rows]
            if not chunks:
                continue
            row_idx = np.concatenate(chunks)
            y_b = y_fold[row_idx]
            s_b = s_fold[row_idx]
            if len(np.unique(y_b)) < 2:
                continue
            y_mx = mx.array(y_b.astype(np.float32))
            s_mx = mx.array(s_b.astype(np.float32))
            fold_aucs.append(gpu_auc_single(s_mx, y_mx))
        replicate_stats[b] = float(np.mean(fold_aucs)) if fold_aucs else 0.5

    return (
        float(np.quantile(replicate_stats, 0.025)),
        float(np.quantile(replicate_stats, 0.975)),
    )
