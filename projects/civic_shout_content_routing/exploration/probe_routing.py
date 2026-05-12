"""Probe-routing exploration for Civic Shout content routing.

Reads the derived wide parquet built by ``build_wide_engagement.py`` and
writes a single self-contained HTML file with one section per question:
prose explanation + base64-embedded PNG.

The HTML is incrementally extensible -- add more ``Section`` builders to
``SECTIONS`` and re-run.

Usage
-----
    python -m projects.civic_shout_content_routing.exploration.probe_routing
    python ... --in path/to/wide.parquet --out path/to/probe_routing.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sp

from entities.civic_shout_engagement.aggregate_emails_cache import (
    cache as agg_cache,
)
from entities.civic_shout_engagement.attributed_actions.cache import (
    cache as attr_cache,
)
from projects.civic_shout_content_routing.exploration.explore_users import (
    Section,
    _fig_to_b64,
    build_action_matrix,
)

ANCHOR_UTC = datetime(2026, 4, 21, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _tier_edges(counts: np.ndarray, n_tiers: int = 4) -> list[int]:
    """Return monotonic bucket edges over ``counts`` for ``n_tiers`` equal-quantile tiers."""
    eligible = counts[counts >= 2]
    qs = np.quantile(eligible, np.linspace(0.0, 1.0, n_tiers + 1)[1:-1])
    edges: list[int] = [2] + [int(q) + 1 for q in qs] + [int(eligible.max()) + 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    return edges


def _user_jaccard_topk_adj(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """User-side Jaccard top-K adjacency matrix.

    Returns a (n_users x n_users) sparse matrix where entry [u, v] == 1
    iff v is one of u's top-K Jaccard neighbours over shared emails.
    """
    from projects.civic_shout_content_routing.exploration.explore_users import (
        _build_topk_adj,
    )

    co = (M @ M.T).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -1.0)
    return _build_topk_adj(J, k)


def _chronological_email_split(
    wide: pl.DataFrame, n_folds: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling walk-forward folds based on ``sent_at``.

    Each fold spans a 12-month train window → 6-month test window, sliding
    6 months at a time.  Returns a list of (train_email_ids, test_email_ids).
    """
    sent = wide.select("email_id", "sent_at").unique("email_id").sort("sent_at")
    t_min = sent["sent_at"].min()
    t_max = sent["sent_at"].max()

    if t_min is None or t_max is None:
        return []

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        train_start = t_min + timedelta(days=180 * fold)
        train_end = train_start + timedelta(days=365)
        test_end = train_end + timedelta(days=180)

        train_ids = sent.filter(
            (pl.col("sent_at") >= train_start) & (pl.col("sent_at") < train_end)
        )["email_id"].to_numpy()
        test_ids = sent.filter((pl.col("sent_at") >= train_end) & (pl.col("sent_at") < test_end))[
            "email_id"
        ].to_numpy()
        if len(train_ids) > 0 and len(test_ids) > 0:
            folds.append((train_ids, test_ids))
    return folds


def _primary_split(wide: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """12-month/12-month primary chronological split on email ``sent_at``.

    The midpoint of [min(sent_at), max(sent_at)] divides train from test.
    """
    sent = wide.select("email_id", "sent_at").unique("email_id").sort("sent_at")
    t_min = sent["sent_at"].min()
    t_max = sent["sent_at"].max()

    if t_min is None or t_max is None:
        empty = np.array([], dtype=np.int64)
        return empty, empty

    mid = t_min + (t_max - t_min) / 2
    train_ids = sent.filter(pl.col("sent_at") < mid)["email_id"].to_numpy()
    test_ids = sent.filter(pl.col("sent_at") >= mid)["email_id"].to_numpy()
    return train_ids, test_ids


def _stratified_probe_partition(
    user_features_train: pl.DataFrame,
    probe_frac: float = 0.10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Stratified 10% probe partition on ``n_sent_train`` × ``n_actioned_train`` deciles.

    Returns (probe_user_ids, holdout_user_ids, smds) where smds maps each
    balance variable name → standardized mean difference.
    """
    rng = np.random.default_rng(seed)

    df = user_features_train.filter(
        (pl.col("n_actioned_train") >= 2) & (pl.col("n_sent_train") >= 5)
    )

    sent_vals = df["n_sent_train"].to_numpy().astype(float)
    act_vals = df["n_actioned_train"].to_numpy().astype(float)
    user_ids = df["user_id"].to_numpy()

    n_deciles = 10
    sent_edges = np.quantile(sent_vals, np.linspace(0, 1, n_deciles + 1))
    act_edges = np.quantile(act_vals, np.linspace(0, 1, n_deciles + 1))

    sent_decile = np.searchsorted(sent_edges[1:-1], sent_vals, side="right")
    act_decile = np.searchsorted(act_edges[1:-1], act_vals, side="right")
    strata = sent_decile * n_deciles + act_decile

    probe_mask = np.zeros(len(user_ids), dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        n_probe = max(1, int(round(len(idx) * probe_frac)))
        chosen = rng.choice(idx, size=min(n_probe, len(idx)), replace=False)
        probe_mask[chosen] = True

    probe_user_ids = user_ids[probe_mask]
    holdout_user_ids = user_ids[~probe_mask]

    def _smd(var: np.ndarray) -> float:
        p = var[probe_mask]
        h = var[~probe_mask]
        pooled_std = np.sqrt((np.var(p) + np.var(h)) / 2)
        if pooled_std == 0:
            return 0.0
        return float((np.mean(p) - np.mean(h)) / pooled_std)

    smds: dict[str, float] = {
        "n_sent_train": _smd(sent_vals),
        "n_actioned_train": _smd(act_vals),
    }
    return probe_user_ids, holdout_user_ids, smds


def _eb_shrink_rate(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Empirical-Bayes Beta-binomial shrinkage of numer/denom rates.

    Fits (alpha, beta) by method-of-moments; falls back to Laplace (alpha=beta=1)
    when fewer than 100 samples have non-zero variance in denom.
    """
    rates = np.where(denom > 0, numer / denom, 0.0)
    valid = denom > 0
    if valid.sum() >= 100 and np.var(rates[valid]) > 0:
        mu = float(np.mean(rates[valid]))
        var = float(np.var(rates[valid]))
        if var < mu * (1 - mu):
            phi = (mu * (1 - mu) / var) - 1
            alpha = mu * phi
            beta = (1 - mu) * phi
        else:
            alpha, beta = 1.0, 1.0
    else:
        alpha, beta = 1.0, 1.0
    return (numer + alpha) / (denom + alpha + beta)


def _sendable_set(wide: pl.DataFrame, test_email_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Map each test email_id → np.ndarray of user_ids who received it."""
    test_set = set(test_email_ids.tolist())
    result: dict[int, np.ndarray] = {}
    agg = (
        wide.filter(pl.col("email_id").is_in(list(test_set)))
        .group_by("email_id")
        .agg(pl.col("user_id").alias("user_ids"))
    )
    for row in agg.iter_rows(named=True):
        result[int(row["email_id"])] = np.array(row["user_ids"], dtype=np.int64)
    return result


def _score_holdouts_against_probe(
    M_probe: sp.csr_matrix,
    U_adj: sp.csr_matrix,
    test_email_col_idx: np.ndarray,
    holdout_user_row_idx: np.ndarray,
) -> np.ndarray:
    """Mean Jaccard similarity from each holdout user to probe signers per test email.

    Returns shape (n_holdout_users, n_test_emails).
    """
    probe_counts = np.asarray(M_probe[:, test_email_col_idx].sum(axis=0)).flatten()
    probe_counts = np.maximum(probe_counts, 1)
    raw = U_adj[holdout_user_row_idx, :] @ M_probe[:, test_email_col_idx]
    scores = np.asarray(raw.todense()) / probe_counts[None, :]
    return scores.astype(np.float32)


def _topX_action_rate(
    scores: np.ndarray,
    M_holdout_test: sp.csr_matrix,
    sendable_mask: np.ndarray,
    x: float,
) -> np.ndarray:
    """Per-test-email action rate when the top-X fraction of sendable holdouts are sent.

    Returns shape (n_test_emails,).
    """
    n_holdout, n_test_emails = scores.shape
    rates = np.zeros(n_test_emails, dtype=np.float32)
    for e_idx in range(n_test_emails):
        sendable_idx = np.where(sendable_mask[:, e_idx])[0]
        if len(sendable_idx) == 0:
            continue
        n_top = max(1, int(round(len(sendable_idx) * x)))
        e_scores = scores[sendable_idx, e_idx]
        top_idx = np.argpartition(-e_scores, min(n_top, len(e_scores) - 1))[:n_top]
        selected = sendable_idx[top_idx]
        actioned = np.asarray(M_holdout_test[selected, e_idx]).flatten()
        rates[e_idx] = float(actioned.mean()) if len(actioned) > 0 else 0.0
    return rates


def _hazard_curve(wide: pl.DataFrame, attributed: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative action probability by hours-since-send.

    Returns (hours_bins, cumulative_action_prob).
    Bins are log-spaced from 0.01h to 720h (30 days).
    """
    hours_bins = np.logspace(np.log10(0.01), np.log10(720), 200)

    lag_hours = (
        attributed.filter(pl.col("is_attributed").cast(pl.Boolean))
        .select("lag_seconds")
        .to_series()
        .to_numpy()
        .astype(float)
        / 3600.0
    )

    total_actions = wide.filter(pl.col("actioned"))["actioned"].len()
    if total_actions == 0 or len(lag_hours) == 0:
        return hours_bins, np.zeros(len(hours_bins))

    cumulative = np.array(
        [(lag_hours <= h).sum() / total_actions for h in hours_bins],
        dtype=np.float64,
    )
    return hours_bins, cumulative


def _cluster_bootstrap_users(
    holdout_user_ids: np.ndarray,
    fn: Callable,
    n_resamples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """User-cluster bootstrap: draw n_resamples results of fn(resampled_user_ids).

    Returns shape (n_resamples,).
    """
    rng = np.random.default_rng(seed)
    n = len(holdout_user_ids)
    results = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = holdout_user_ids[idx]
        results.append(fn(resampled))
    return np.array(results)


# ---------------------------------------------------------------------------
# Module-level probe seed cache (populated by _section_q2_partition).
# ---------------------------------------------------------------------------

_PROBE_SEED_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Setup helper: per-user train features.
# ---------------------------------------------------------------------------


def _build_user_features_train(
    wide: pl.DataFrame,
    train_email_ids: np.ndarray,
) -> pl.DataFrame:
    """Per-user features computed on train emails only.

    Returns one row per user with columns:
      user_id, n_sent_train, n_actioned_train, action_rate_train_eb,
      last_action_sent_at (nullable).
    Filtered to n_actioned_train >= 2 AND n_sent_train >= 5.
    """
    train_set = set(train_email_ids.tolist())
    train_wide = wide.filter(pl.col("email_id").is_in(list(train_set)))

    agg = (
        train_wide.group_by("user_id")
        .agg(
            pl.len().alias("n_sent_train"),
            pl.col("actioned").sum().cast(pl.Int64).alias("n_actioned_train"),
            pl.when(pl.col("actioned"))
            .then(pl.col("sent_at"))
            .otherwise(None)
            .max()
            .alias("last_action_sent_at"),
        )
        .filter((pl.col("n_actioned_train") >= 2) & (pl.col("n_sent_train") >= 5))
    )

    n_sent = agg["n_sent_train"].to_numpy().astype(float)
    n_act = agg["n_actioned_train"].to_numpy().astype(float)
    eb_rates = _eb_shrink_rate(n_act, n_sent)

    return agg.with_columns(pl.Series("action_rate_train_eb", eb_rates, dtype=pl.Float64))


# ---------------------------------------------------------------------------
# Section builders.
# ---------------------------------------------------------------------------


def _section_q1_baseline(
    wide: pl.DataFrame,
    user_features_train: pl.DataFrame,
    test_email_ids: np.ndarray,
) -> Section:
    test_set = set(test_email_ids.tolist())
    test_wide = wide.filter(pl.col("email_id").is_in(list(test_set)))

    overall_rate = float(test_wide["actioned"].mean() or 0.0)
    lift_target = overall_rate * 1.10

    eb_arr = user_features_train["action_rate_train_eb"].to_numpy()
    user_ids_arr = user_features_train["user_id"].to_numpy()

    q25, q50, q75 = (
        float(np.quantile(eb_arr, 0.25)),
        float(np.quantile(eb_arr, 0.50)),
        float(np.quantile(eb_arr, 0.75)),
    )
    tiers_1based = np.searchsorted([q25, q50, q75], eb_arr, side="right") + 1
    tiers_1based = np.clip(tiers_1based, 1, 4).astype(int)

    tier_df = pl.DataFrame({"user_id": user_ids_arr.tolist(), "tier": tiers_1based.tolist()})

    test_wide_with_tiers = test_wide.join(tier_df, on="user_id", how="left")

    overall_sends = int(test_wide_with_tiers.height)
    overall_actions = int(test_wide_with_tiers["actioned"].sum())
    overall_n_users = int(test_wide_with_tiers["user_id"].n_unique())

    tier_agg = (
        test_wide_with_tiers.filter(pl.col("tier").is_not_null())
        .group_by("tier")
        .agg(
            pl.len().alias("sends"),
            pl.col("actioned").sum().alias("actions"),
            pl.col("user_id").n_unique().alias("n_users"),
        )
        .sort("tier")
    )

    tier_labels = ["Overall", "Q1 (least active)", "Q2", "Q3", "Q4 (most active)"]

    tier_rows = [
        (
            overall_n_users,
            overall_sends,
            overall_actions,
            overall_rate,
            lift_target,
        )
    ]
    for t in range(1, 5):
        row = tier_agg.filter(pl.col("tier") == t)
        if row.height == 0:
            tier_rows.append((0, 0, 0, 0.0, 0.0))
        else:
            n_users = int(row["n_users"][0])
            n_sends = int(row["sends"][0])
            n_actions = int(row["actions"][0])
            rate = n_actions / n_sends if n_sends > 0 else 0.0
            tier_rows.append((n_users, n_sends, n_actions, rate, rate * 1.10))

    table_html_rows = ""
    for label, (n_users, n_sends, n_actions, rate, target) in zip(tier_labels, tier_rows):
        table_html_rows += (
            f"<tr><td>{label}</td><td>{n_users:,}</td><td>{n_sends:,}</td>"
            f"<td>{n_actions:,}</td><td>{rate:.2%}</td><td>{target:.2%}</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Slice</th><th>Users</th><th>Sends (test)</th>"
        "<th>Actions (test)</th><th>Action rate</th><th>10% lift target</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    tier_rates = [tier_rows[i + 1][3] for i in range(4)]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#94a3b8", "#60a5fa", "#3b82f6", "#1d4ed8"]
    bars = ax.barh(tier_labels[1:], tier_rates, color=colors)
    ax.axvline(
        lift_target,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"10% lift target ({lift_target:.2%})",
    )
    ax.axvline(
        overall_rate,
        color="#64748b",
        linestyle=":",
        linewidth=1.2,
        label=f"Overall ({overall_rate:.2%})",
    )
    ax.set_xlabel("Action rate (test emails)")
    ax.set_title("Q1. Action rate by user activity tier (train-only EB tiering)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    fig.tight_layout()

    prose = (
        f"Overall action rate on test emails: <b>{overall_rate:.2%}</b>. "
        f"10% relative lift target: <b>{lift_target:.2%}</b>. "
        f"Users are tiered by empirical-Bayes-shrunk train-only action rate. "
        f"The do-no-harm constraint requires no tier show significantly negative lift relative to random. "
        f"{table_html}"
    )

    return Section(
        slug="q1_baseline",
        heading="Q1. Baseline action rate & 10% lift target (train-only tiering)",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


def _section_q2_partition(
    user_features_train: pl.DataFrame,
    n_seeds: int = 50,
) -> Section:
    anchor_lit = pl.lit(ANCHOR_UTC).dt.replace_time_zone(None)

    recency_raw = (
        user_features_train.select(
            (anchor_lit - pl.col("last_action_sent_at").dt.replace_time_zone(None))
            .dt.total_days()
            .alias("recency_days")
        )["recency_days"]
        .to_numpy()
        .astype(float)
    )
    has_action = user_features_train["n_actioned_train"].to_numpy() > 0
    valid_lags = recency_raw[has_action & np.isfinite(recency_raw)]
    cap_recency = float(np.percentile(valid_lags, 99)) if len(valid_lags) > 0 else 9999.0
    recency_vals = np.where(has_action & np.isfinite(recency_raw), recency_raw, cap_recency)
    recency_vals = np.minimum(recency_vals, cap_recency)

    bal_vars = ["n_sent_train", "n_actioned_train", "recency"]
    all_smds: dict[str, list[float]] = {v: [] for v in bal_vars}

    user_ids_arr = user_features_train["user_id"].to_numpy()
    n_sent_arr = user_features_train["n_sent_train"].to_numpy().astype(float)
    n_act_arr = user_features_train["n_actioned_train"].to_numpy().astype(float)

    for seed in range(n_seeds):
        probe_ids, holdout_ids, smds = _stratified_probe_partition(
            user_features_train, probe_frac=0.10, seed=seed
        )
        _PROBE_SEED_CACHE[seed] = (probe_ids, holdout_ids)

        probe_set = set(probe_ids.tolist())
        probe_mask = np.isin(user_ids_arr, probe_ids)

        def _smd_vec(var: np.ndarray) -> float:
            p = var[probe_mask]
            h = var[~probe_mask]
            pooled_std = np.sqrt((np.var(p) + np.var(h)) / 2)
            if pooled_std == 0:
                return 0.0
            return float((np.mean(p) - np.mean(h)) / pooled_std)

        all_smds["n_sent_train"].append(_smd_vec(n_sent_arr))
        all_smds["n_actioned_train"].append(_smd_vec(n_act_arr))
        all_smds["recency"].append(_smd_vec(recency_vals))

    table_html_rows = ""
    for var in bal_vars:
        smds_arr = np.array(all_smds[var])
        mean_smd = float(np.mean(np.abs(smds_arr)))
        max_smd = float(np.max(np.abs(smds_arr)))
        pct_over = float((np.abs(smds_arr) > 0.1).mean() * 100)
        table_html_rows += (
            f"<tr><td><code>{var}</code></td>"
            f"<td>{mean_smd:.4f}</td><td>{max_smd:.4f}</td><td>{pct_over:.1f}%</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Balance variable</th><th>Mean |SMD| across seeds</th>"
        "<th>Max |SMD|</th><th>% seeds with |SMD|>0.1</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    max_abs_smds = [max(np.abs(np.array(all_smds[v]))) for v in bal_vars]
    any_bad = any(v > 0.1 for v in max_abs_smds)
    if any_bad:
        print(
            f"WARNING Q2: max |SMD| > 0.1 detected across seeds: "
            + ", ".join(f"{v}={max_abs_smds[i]:.4f}" for i, v in enumerate(bal_vars))
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, var in zip(axes, bal_vars):
        smds_arr = np.array(all_smds[var])
        ax.hist(smds_arr, bins=15, color="#3b82f6", edgecolor="white")
        ax.axvline(0.1, color="red", linestyle="--", linewidth=1.2, label="|SMD|=0.1")
        ax.axvline(-0.1, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"SMD: {var}", fontsize=10)
        ax.set_xlabel("SMD")
        ax.set_ylabel("Seeds")
        ax.legend(fontsize=8)

    fig.suptitle(f"Q2. SMD distribution across {n_seeds} probe seeds", fontsize=12)
    fig.tight_layout()

    prose = (
        f"Stratified 10% probe partition evaluated across <b>{n_seeds} seeds</b>, "
        f"stratified on <code>n_sent_train</code> decile × <code>n_actioned_train</code> decile. "
        f"All 50 seed partitions are cached for reuse in Q3+. "
        f"Recency is (anchor − last_action_sent_at).days; users with no train actions receive the "
        f"99th-percentile cap ({cap_recency} days). "
        f"{'<b style="color:red">WARNING: max |SMD| > 0.1 detected — stratification audit recommended.</b> ' if any_bad else ''}"
        f"{table_html}"
    )

    return Section(
        slug="q2_partition",
        heading="Q2. Stratified probe partition & multi-seed balance check",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


def _section_q3_coverage(
    wide: pl.DataFrame,
    test_email_ids: np.ndarray,
    user_features_train: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:

    test_set = set(test_email_ids.tolist())
    test_email_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )
    M_test_cols = M_full[:, test_email_col_idx]

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}

    sendable = _sendable_set(wide, test_email_ids)

    test_wide = wide.filter(pl.col("email_id").is_in(list(test_set)))
    sends_agg = test_wide.group_by("email_id").agg(pl.len().alias("n_sends"))
    sends_per_email: dict[int, int] = {
        int(row["email_id"]): int(row["n_sends"]) for row in sends_agg.iter_rows(named=True)
    }
    total_test_sends = sum(sends_per_email.values())

    n_test_emails = len(test_email_col_idx)
    n_seeds = len(_PROBE_SEED_CACHE)
    seeds_sorted = sorted(_PROBE_SEED_CACHE.keys())

    n_probe_signers = np.zeros((n_test_emails, n_seeds), dtype=np.float32)

    for s_idx, seed in enumerate(seeds_sorted):
        probe_ids, _ = _PROBE_SEED_CACHE[seed]
        probe_rows = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )
        if len(probe_rows) == 0:
            continue
        M_probe = M_test_cols[probe_rows, :]
        col_sums = np.asarray(M_probe.sum(axis=0)).flatten()
        n_probe_signers[:, s_idx] = col_sums

    median_signers = np.median(n_probe_signers, axis=1)
    thresholds = [1, 3, 5, 10, 25]
    min_signers = 3

    table_html_rows = ""
    for thresh in thresholds:
        pct_emails_clearing = float(np.mean(np.median(n_probe_signers >= thresh, axis=1)) * 100)

        fallback_sends = 0
        for e_col_idx, eid in zip(
            range(n_test_emails), [email_ids_all[c] for c in test_email_col_idx]
        ):
            eid_int = int(eid)
            med = float(np.median(n_probe_signers[e_col_idx, :]))
            if med < thresh:
                fallback_sends += sends_per_email.get(eid_int, 0)
        pct_sends_clearing = (
            float((1 - fallback_sends / total_test_sends) * 100) if total_test_sends > 0 else 0.0
        )

        label = f"≥ {thresh}" + (" (= min_signers)" if thresh == min_signers else "")
        table_html_rows += (
            f"<tr><td>{label}</td>"
            f"<td>{pct_emails_clearing:.1f}%</td>"
            f"<td>{pct_sends_clearing:.1f}%</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Probe signer threshold</th><th>% of test emails clearing</th>"
        "<th>% of test sends clearing (median over seeds)</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_median = np.sort(median_signers)
    ecdf_y = np.arange(1, len(sorted_median) + 1) / len(sorted_median)
    ax.plot(sorted_median + 0.01, ecdf_y, color="#3b82f6", linewidth=2, label="Median over seeds")

    p5_sorted = np.sort(np.percentile(n_probe_signers, 5, axis=1))
    p95_sorted = np.sort(np.percentile(n_probe_signers, 95, axis=1))
    ax.fill_betweenx(
        ecdf_y,
        p5_sorted + 0.01,
        p95_sorted + 0.01,
        alpha=0.2,
        color="#3b82f6",
        label="5th–95th pct band",
    )

    overall_median = float(np.median(sorted_median))
    ax.axvline(
        overall_median,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Median = {overall_median:.1f}",
    )
    ax.axvline(
        min_signers,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"min_signers = {min_signers}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("# probe signers per test email (log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("Q3. ECDF of probe signers per test email (median + seed band)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    probe_pct_clearing = float(np.mean(np.median(n_probe_signers >= min_signers, axis=1)) * 100)
    fallback_pct = 100.0 - probe_pct_clearing

    prose = (
        f"For each of the {n_test_emails} test emails and each of the "
        f"{n_seeds} cached probe seeds, the number of probe users who actioned that email. "
        f"With <code>min_signers = {min_signers}</code>, approximately "
        f"<b>{100.0 - fallback_pct:.1f}%</b> of test emails clear the threshold (median over seeds); "
        f"the remaining <b>{fallback_pct:.1f}%</b> fall back to the popularity baseline. "
        f"The ECDF (log x-axis) shows how thin the probe signal is for the long tail of niche emails. "
        f"{table_html}"
    )

    return Section(
        slug="q3_coverage",
        heading="Q3. Probe coverage on test emails & cold-start fallback rate",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Module-level Q4 winner (populated by _section_q4_popularity_head_to_head).
# ---------------------------------------------------------------------------

_Q4_BEST_POPULARITY_PREDICTOR: str = ""


# ---------------------------------------------------------------------------
# Setup helper: aggregate_emails v3 filtered to given email_ids.
# ---------------------------------------------------------------------------


def _load_agg_v3(email_ids: np.ndarray) -> pl.DataFrame:
    """Return agg v3 rows for the given email_ids with columns email_id, total_actions, total_sends."""
    df = agg_cache.get(3)
    id_set = set(int(x) for x in email_ids)
    filtered = df.filter(pl.col("email_id").is_in(list(id_set))).select(
        "email_id",
        "total_actions",
        pl.col("total_sent").alias("total_sends"),
    )
    return filtered


# ---------------------------------------------------------------------------
# Section builders: Q4 and Q5.
# ---------------------------------------------------------------------------


def _section_q4_popularity_head_to_head(
    wide: pl.DataFrame,
    test_email_ids: np.ndarray,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
    agg_v3_df: pl.DataFrame,
) -> Section:
    """Q4: compare probe predictors vs historical popularity at predicting holdout action rate."""
    global _Q4_BEST_POPULARITY_PREDICTOR

    test_set = set(test_email_ids.tolist())
    test_email_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )
    n_test_emails = len(test_email_col_idx)
    test_email_ids_ordered = email_ids_all[test_email_col_idx]

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}
    seeds_sorted = sorted(_PROBE_SEED_CACHE.keys())
    n_seeds = len(seeds_sorted)

    probe_signers_per_seed = np.zeros((n_test_emails, n_seeds), dtype=np.float32)
    probe_sent_per_seed = np.zeros((n_test_emails, n_seeds), dtype=np.float32)
    holdout_signers_per_seed = np.zeros((n_test_emails, n_seeds), dtype=np.float32)
    holdout_sent_per_seed = np.zeros((n_test_emails, n_seeds), dtype=np.float32)

    M_test_cols = M_full[:, test_email_col_idx]

    sendable_map = _sendable_set(wide, test_email_ids)
    test_wide = wide.filter(pl.col("email_id").is_in(list(test_set)))

    for s_idx, seed in enumerate(seeds_sorted):
        probe_ids, holdout_ids = _PROBE_SEED_CACHE[seed]
        probe_set = set(probe_ids.tolist())
        holdout_set = set(holdout_ids.tolist())

        probe_rows = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )
        holdout_rows = np.array(
            [uid_to_row[int(uid)] for uid in holdout_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )

        if len(probe_rows) > 0:
            M_probe_sub = M_test_cols[probe_rows, :]
            probe_signers_per_seed[:, s_idx] = np.asarray(M_probe_sub.sum(axis=0)).flatten()

        if len(holdout_rows) > 0:
            M_holdout_sub = M_test_cols[holdout_rows, :]
            holdout_signers_per_seed[:, s_idx] = np.asarray(M_holdout_sub.sum(axis=0)).flatten()

        for e_col, eid in enumerate(test_email_ids_ordered):
            eid_int = int(eid)
            sendable = sendable_map.get(eid_int, np.array([], dtype=np.int64))
            probe_sent_per_seed[e_col, s_idx] = np.isin(sendable, probe_ids).sum()
            holdout_sent_per_seed[e_col, s_idx] = np.isin(sendable, holdout_ids).sum()

    outcome = np.where(
        holdout_sent_per_seed.sum(axis=1) > 0,
        holdout_signers_per_seed.sum(axis=1) / holdout_sent_per_seed.sum(axis=1),
        np.nan,
    )

    probe_count = np.median(probe_signers_per_seed, axis=1)

    total_probe_signers = probe_signers_per_seed.sum(axis=1)
    total_probe_sent = probe_sent_per_seed.sum(axis=1)
    probe_action_rate = _eb_shrink_rate(total_probe_signers, total_probe_sent)

    agg_lookup = {
        int(row["email_id"]): (int(row["total_actions"]), int(row["total_sends"]))
        for row in agg_v3_df.iter_rows(named=True)
    }
    total_actions_arr = np.array(
        [agg_lookup.get(int(eid), (0, 0))[0] for eid in test_email_ids_ordered], dtype=np.float64
    )
    total_sends_arr = np.array(
        [agg_lookup.get(int(eid), (0, 1))[1] for eid in test_email_ids_ordered], dtype=np.float64
    )
    total_sends_arr = np.maximum(total_sends_arr, 1)
    total_action_rate_arr = total_actions_arr / total_sends_arr

    valid = np.isfinite(outcome) & (outcome >= 0)
    outcome_v = outcome[valid]

    predictors = {
        "probe_count": (probe_count[valid], True),
        "probe_action_rate (EB)": (probe_action_rate[valid], False),
        "total_actions (v3)": (total_actions_arr[valid], True),
        "total_action_rate (v3)": (total_action_rate_arr[valid], False),
    }

    from scipy.stats import spearmanr

    results: dict[str, dict] = {}
    for name, (x_raw, log_transform) in predictors.items():
        x_fit = np.log1p(x_raw) if log_transform else x_raw

        rho_val, _ = spearmanr(x_fit, outcome_v)
        rho = float(rho_val)

        coeffs = np.polyfit(x_fit, outcome_v, 1)
        y_pred = np.polyval(coeffs, x_fit)
        ss_res = np.sum((outcome_v - y_pred) ** 2)
        ss_tot = np.sum((outcome_v - np.mean(outcome_v)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        q75 = np.percentile(x_raw, 75)
        q25 = np.percentile(x_raw, 25)
        top_q_rate = float(np.mean(outcome_v[x_raw >= q75])) if (x_raw >= q75).any() else 0.0
        bot_q_rate = float(np.mean(outcome_v[x_raw <= q25])) if (x_raw <= q25).any() else 0.0
        uplift = top_q_rate / bot_q_rate if bot_q_rate > 0 else float("nan")

        results[name] = {
            "rho": rho,
            "r2": r2,
            "uplift": uplift,
            "x_raw": x_raw,
            "log": log_transform,
        }

    best_name = max(results, key=lambda n: abs(results[n]["rho"]))
    _Q4_BEST_POPULARITY_PREDICTOR = best_name
    print(f"Q4 winner: {best_name}")

    table_html_rows = ""
    for name, res in results.items():
        table_html_rows += (
            f"<tr><td><code>{name}</code></td>"
            f"<td>{res['rho']:.3f}</td>"
            f"<td>{res['r2']:.3f}</td>"
            f"<td>{res['uplift']:.2f}x</td></tr>"
        )
    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Predictor</th><th>Spearman ρ</th><th>OLS R²</th>"
        "<th>Top-Q vs Bottom-Q uplift</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    y_all = np.concatenate([res["x_raw"] for res in results.values()])
    y_min = float(np.percentile(outcome_v, 1))
    y_max = float(np.percentile(outcome_v, 99))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()
    predictor_items = list(results.items())

    for ax, (name, res) in zip(axes_flat, predictor_items):
        x_raw = res["x_raw"]
        log_t = res["log"]
        x_fit = np.log1p(x_raw) if log_t else x_raw

        ax.scatter(
            x_raw if not log_t else np.log1p(x_raw), outcome_v, s=8, alpha=0.4, color="#64748b"
        )

        coeffs = np.polyfit(x_fit, outcome_v, 1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color="#ef4444", linewidth=1.5, label="OLS fit")

        residuals = outcome_v - np.polyval(coeffs, x_fit)
        se = np.std(residuals)
        ax.fill_between(
            x_line,
            y_line - 1.96 * se,
            y_line + 1.96 * se,
            alpha=0.15,
            color="#ef4444",
            label="95% band",
        )

        ax.set_ylim(y_min, y_max)
        xlabel = f"log1p({name})" if log_t else name
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("holdout action rate", fontsize=9)
        ax.set_title(f"{name}\nSpearman ρ={res['rho']:.3f}", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("Q4. Popularity head-to-head: predictors vs holdout action rate", fontsize=12)
    fig.tight_layout()

    prose = (
        f"For each of the {n_test_emails} test emails the outcome is the "
        f"holdout action rate (mean over {n_seeds} probe seeds). Four predictors are compared. "
        f"<b>Strongest predictor by Spearman ρ: <code>{best_name}</code></b> — this becomes the "
        f"popularity-only arm in Q6/Q7. "
        f"{table_html}"
    )

    return Section(
        slug="q4_popularity",
        heading="Q4. Popularity head-to-head — is the probe just popularity rediscovered?",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


def _section_q5_user_jaccard_smoke(
    wide: pl.DataFrame,
    train_email_ids: np.ndarray,
    user_features_train: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:
    """Q5: user-user Jaccard smoke test against popularity-weighted null."""
    train_set = set(train_email_ids.tolist())
    train_col_idx = np.searchsorted(
        email_ids_all, np.array(sorted(train_set), dtype=email_ids_all.dtype)
    )
    train_col_idx = train_col_idx[train_col_idx < len(email_ids_all)]
    train_col_idx = train_col_idx[
        email_ids_all[train_col_idx]
        == np.array(sorted(train_set), dtype=email_ids_all.dtype)[: len(train_col_idx)]
    ]

    email_ids_sorted = np.sort(np.array(list(train_set), dtype=email_ids_all.dtype))
    train_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in train_set], dtype=np.intp
    )
    M_train = M_full[:, train_col_idx]

    eligible_user_ids = user_features_train["user_id"].to_numpy()
    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}
    eligible_rows = np.array(
        [uid_to_row[int(uid)] for uid in eligible_user_ids if int(uid) in uid_to_row],
        dtype=np.intp,
    )

    M_eligible = M_train[eligible_rows, :]

    rng = np.random.default_rng(0)
    n_sample = min(5000, M_eligible.shape[0])
    sample_idx = rng.choice(M_eligible.shape[0], size=n_sample, replace=False)
    M_s = M_eligible[sample_idx, :].toarray().astype(np.float32)

    co = M_s @ M_s.T
    counts = M_s.sum(axis=1)
    union = counts[:, None] + counts[None, :] - co
    with np.errstate(divide="ignore", invalid="ignore"):
        J_obs = np.where(union > 0, co / union, 0.0)
    np.fill_diagonal(J_obs, 0.0)
    mask = np.triu(np.ones((n_sample, n_sample), dtype=bool), k=1)
    obs_vals = J_obs[mask]

    email_popularity = np.asarray(M_train.sum(axis=0)).flatten().astype(np.float64)
    email_popularity = np.maximum(email_popularity, 1e-9)
    email_popularity /= email_popularity.sum()
    n_train_emails = M_train.shape[1]

    rng2 = np.random.default_rng(1)
    n_actioned_per_user = counts.astype(int)

    null_rows = []
    for i in range(n_sample):
        k = int(n_actioned_per_user[i])
        if k == 0:
            null_rows.append(np.zeros(n_train_emails, dtype=np.float32))
        else:
            chosen = rng2.choice(n_train_emails, size=k, replace=False, p=email_popularity)
            row = np.zeros(n_train_emails, dtype=np.float32)
            row[chosen] = 1.0
            null_rows.append(row)

    M_null = np.stack(null_rows, axis=0)
    co_null = M_null @ M_null.T
    counts_null = M_null.sum(axis=1)
    union_null = counts_null[:, None] + counts_null[None, :] - co_null
    with np.errstate(divide="ignore", invalid="ignore"):
        J_null = np.where(union_null > 0, co_null / union_null, 0.0)
    np.fill_diagonal(J_null, 0.0)
    null_vals = J_null[mask]

    percentiles = [50, 90, 99]
    table_html_rows = ""
    for pct in percentiles:
        obs_p = float(np.percentile(obs_vals, pct))
        null_p = float(np.percentile(null_vals, pct))
        ratio = obs_p / null_p if null_p > 0 else float("nan")
        table_html_rows += f"<tr><td>p{pct}</td><td>{obs_p:.4f}</td><td>{null_p:.4f}</td><td>{ratio:.2f}x</td></tr>"
    obs_mean = float(np.mean(obs_vals))
    null_mean = float(np.mean(null_vals))
    ratio_mean = obs_mean / null_mean if null_mean > 0 else float("nan")
    table_html_rows += f"<tr><td>mean</td><td>{obs_mean:.4f}</td><td>{null_mean:.4f}</td><td>{ratio_mean:.2f}x</td></tr>"

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Percentile</th><th>Observed user-pair Jaccard</th>"
        "<th>Popularity-null Jaccard</th><th>Ratio</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    max_j = max(float(np.percentile(obs_vals, 99.5)), float(np.percentile(null_vals, 99.5)))
    bins = np.linspace(0, max_j, 80)
    ax.hist(obs_vals, bins=bins, color="#0ea5e9", alpha=0.65, label="Observed", density=True)
    ax.hist(
        null_vals, bins=bins, color="#94a3b8", alpha=0.65, label="Popularity null", density=True
    )

    obs_max = float(obs_vals.max())
    null_max = float(null_vals.max())
    if obs_max > 0.02 or null_max > 0.02:
        ax.set_yscale("log")

    ax.set_xlabel("User-pair Jaccard similarity (train emails)")
    ax.set_ylabel("Density")
    ax.set_title(f"Q5. User-user Jaccard smoke test (n={n_sample:,} sampled users, upper triangle)")
    ax.legend(fontsize=10)
    fig.tight_layout()

    prose = (
        f"User-user Jaccard computed on train emails for {n_sample:,} sampled eligible users "
        f"(from {len(eligible_rows):,} total with n_actioned≥2 / n_sent≥5). "
        f"The popularity-weighted null preserves each user's <code>n_actioned_train</code> but "
        f"redraws which emails they signed proportional to per-email popularity. "
        f"If observed Jaccard systematically exceeds the null at the high tail, users share "
        f"email preferences beyond what popularity alone predicts. "
        f"{table_html}"
    )

    return Section(
        slug="q5_user_jaccard",
        heading="Q5. User-user Jaccard smoke test (popularity-weighted null)",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Module-level Q6 decision feed-forward.
# ---------------------------------------------------------------------------

_Q6_SCORE_FUNCTION_NAME: str = "mean-or-union Jaccard"
_Q6_USER_USER_K: int = 50


# ---------------------------------------------------------------------------
# Section builders: Q6 and Q7.
# ---------------------------------------------------------------------------


def _q6_worker(args: tuple) -> dict:
    """Per-seed worker for Q6 multiprocessing pool.

    Returns accumulated lift/rate lists and calibration data for this seed.
    """
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from scipy.stats import spearmanr

    (
        seed,
        s_idx,
        probe_ids,
        holdout_ids,
        M_train_filtered_data,
        M_train_filtered_indices,
        M_train_filtered_indptr,
        M_train_filtered_shape,
        M_probe_test_data,
        M_probe_test_indices,
        M_probe_test_indptr,
        M_probe_test_shape,
        M_test_rows_data,
        M_test_rows_indices,
        M_test_rows_indptr,
        M_test_rows_shape,
        probe_local_idx,
        eligible_rows_arr,
        user_ids_all,
        sendable_mask,
        holdout_local_idx,
        counts_h,
        X_values,
        n_random_subseeds,
        min_signers,
        n_test_emails,
    ) = args

    import numpy as np
    import scipy.sparse as sp

    M_train_filtered = sp.csr_matrix(
        (M_train_filtered_data, M_train_filtered_indices, M_train_filtered_indptr),
        shape=M_train_filtered_shape,
    )
    M_probe_test = sp.csr_matrix(
        (M_probe_test_data, M_probe_test_indices, M_probe_test_indptr),
        shape=M_probe_test_shape,
    )
    M_test_rows = sp.csr_matrix(
        (M_test_rows_data, M_test_rows_indices, M_test_rows_indptr),
        shape=M_test_rows_shape,
    )

    M_probe_train = M_train_filtered[probe_local_idx, :]

    n_probe_signers_per_email = np.asarray(M_probe_test.sum(axis=0)).flatten()

    combined = (M_probe_train.T @ M_probe_test).astype(bool).astype(np.float32)
    if sp.issparse(combined):
        combined = np.asarray(combined.todense())
    combined_sum_per_email = combined.sum(axis=0).flatten()

    M_holdout_train = M_train_filtered[holdout_local_idx, :]
    intersect_full = M_holdout_train @ combined
    if sp.issparse(intersect_full):
        intersect_full = np.asarray(intersect_full.todense())

    counts_h_holdout = counts_h[holdout_local_idx]
    union_full = counts_h_holdout[:, None] + combined_sum_per_email[None, :] - intersect_full
    scores_matrix = np.where(union_full > 0, intersect_full / union_full, 0.0).astype(np.float32)

    scores_matrix[:, n_probe_signers_per_email < min_signers] = 0.0

    M_test_holdout = M_test_rows[holdout_local_idx, :]
    actioned_matrix = np.asarray(M_test_holdout.todense()).astype(np.float32)

    lift_by_seed: dict[float, list[float]] = {x: [] for x in X_values}
    probe_rate_by_seed: dict[float, list[float]] = {x: [] for x in X_values}
    random_rate_by_seed: dict[float, list[float]] = {x: [] for x in X_values}

    decile_actioned_per_email: list[np.ndarray] = []
    spearman_rhos: list[float] = []
    spearman_pvals: list[float] = []

    rng = np.random.default_rng(42 + seed)

    sendable_holdout_mask = sendable_mask[holdout_local_idx, :]

    for e_idx in range(n_test_emails):
        sendable_local = np.where(sendable_holdout_mask[:, e_idx])[0]
        if len(sendable_local) == 0:
            continue

        score_h = scores_matrix[sendable_local, e_idx]
        actioned_h = actioned_matrix[sendable_local, e_idx]

        if s_idx == 0:
            n_deciles = 10
            if len(score_h) >= n_deciles:
                decile_edges = np.quantile(score_h, np.linspace(0, 1, n_deciles + 1))
                decile_labels = np.searchsorted(decile_edges[1:-1], score_h, side="right")
                per_decile = np.array(
                    [
                        float(np.mean(actioned_h[decile_labels == d]))
                        if (decile_labels == d).any()
                        else np.nan
                        for d in range(n_deciles)
                    ]
                )
                decile_actioned_per_email.append(per_decile)

            if len(score_h) >= 3:
                rho_val, p_val = spearmanr(score_h, actioned_h)
                spearman_rhos.append(float(rho_val))
                spearman_pvals.append(float(p_val))

        n_sendable = len(sendable_local)
        for x in X_values:
            n_top = max(1, int(np.floor(n_sendable * x)))
            top_idx = np.argpartition(-score_h, min(n_top, n_sendable) - 1)[:n_top]
            probe_rate = float(actioned_h[top_idx].mean()) if n_top > 0 else 0.0

            random_rates_sub = []
            for sub_seed in range(n_random_subseeds):
                rand_idx = rng.choice(n_sendable, size=n_top, replace=False)
                random_rates_sub.append(float(actioned_h[rand_idx].mean()) if n_top > 0 else 0.0)
            random_rate = float(np.mean(random_rates_sub))

            lift = probe_rate / random_rate if random_rate > 0 else 1.0
            lift_by_seed[x].append(lift)
            probe_rate_by_seed[x].append(probe_rate)
            random_rate_by_seed[x].append(random_rate)

    return {
        "lift_by_seed": lift_by_seed,
        "probe_rate_by_seed": probe_rate_by_seed,
        "random_rate_by_seed": random_rate_by_seed,
        "decile_actioned_per_email": decile_actioned_per_email,
        "spearman_rhos": spearman_rhos,
        "spearman_pvals": spearman_pvals,
    }


def _section_q6_probe_routing_lift(
    wide: pl.DataFrame,
    train_email_ids: np.ndarray,
    test_email_ids: np.ndarray,
    user_features_train: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:
    """Q6: probe-routing lift — calibration, lift-vs-X curve, and locked-X result."""
    import multiprocessing
    import os

    global _Q6_SCORE_FUNCTION_NAME, _Q6_USER_USER_K

    min_signers = 3
    X_values = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    locked_X = 0.25
    n_random_subseeds = 5

    print("Q6: building M_train_filtered")
    train_set = set(train_email_ids.tolist())
    test_set = set(test_email_ids.tolist())

    train_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in train_set], dtype=np.intp
    )
    test_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}

    eligible_user_ids = user_features_train["user_id"].to_numpy()
    eligible_rows = np.array(
        [uid_to_row[int(uid)] for uid in eligible_user_ids if int(uid) in uid_to_row],
        dtype=np.intp,
    )

    M_train_filtered = M_full[eligible_rows, :][:, train_col_idx].tocsr()
    counts_h = np.asarray(M_train_filtered.sum(axis=1)).flatten().astype(np.float32)

    n_seeds = len(_PROBE_SEED_CACHE)
    n_test_emails = len(test_col_idx)
    test_email_ids_ordered = email_ids_all[test_col_idx]

    seeds_sorted = sorted(_PROBE_SEED_CACHE.keys())
    n_seeds_use = n_seeds
    seeds_use = seeds_sorted

    eligible_row_to_local = {int(r): i for i, r in enumerate(eligible_rows)}

    print(f"Q6: computing sendable mask for {n_test_emails} test emails (vectorized)")
    sendable_map = _sendable_set(wide, test_email_ids)

    sendable_mask = np.zeros((len(eligible_rows), n_test_emails), dtype=bool)
    for e_idx, eid in enumerate(test_email_ids_ordered):
        sendable_uids = sendable_map.get(int(eid), np.array([], dtype=np.int64))
        if len(sendable_uids) == 0:
            continue
        local_indices = np.array(
            [
                eligible_row_to_local[uid_to_row[int(uid)]]
                for uid in sendable_uids
                if int(uid) in uid_to_row and uid_to_row[int(uid)] in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(local_indices) > 0:
            sendable_mask[local_indices, e_idx] = True

    M_test_full_rows = M_full[eligible_rows, :][:, test_col_idx].tocsr()

    print(f"Q6: building per-seed sparse matrices and dispatching {n_seeds_use} seeds to pool")

    worker_args = []
    for s_idx, seed in enumerate(seeds_use):
        probe_ids, holdout_ids = _PROBE_SEED_CACHE[seed]

        probe_rows_global = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )
        if len(probe_rows_global) == 0:
            continue

        probe_local_idx = np.array(
            [
                eligible_row_to_local[int(r)]
                for r in probe_rows_global
                if int(r) in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(probe_local_idx) == 0:
            continue

        holdout_ids_arr = np.asarray(holdout_ids)
        holdout_uids_in_eligible = np.isin(user_ids_all[eligible_rows], holdout_ids_arr)
        holdout_local_idx = np.where(holdout_uids_in_eligible)[0].astype(np.intp)
        if len(holdout_local_idx) == 0:
            continue

        M_probe_test = M_full[probe_rows_global, :][:, test_col_idx].tocsr()

        mt = M_train_filtered
        mpt = M_probe_test
        mtr = M_test_full_rows

        worker_args.append(
            (
                seed,
                s_idx,
                probe_ids,
                holdout_ids,
                mt.data,
                mt.indices,
                mt.indptr,
                mt.shape,
                mpt.data,
                mpt.indices,
                mpt.indptr,
                mpt.shape,
                mtr.data,
                mtr.indices,
                mtr.indptr,
                mtr.shape,
                probe_local_idx,
                eligible_rows,
                user_ids_all,
                sendable_mask,
                holdout_local_idx,
                counts_h,
                X_values,
                n_random_subseeds,
                min_signers,
                n_test_emails,
            )
        )

    n_processes = min(8, os.cpu_count() or 1)
    print(f"Q6: running {len(worker_args)} seeds across {n_processes} processes")
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(_q6_worker, worker_args)

    lift_by_seed: dict[float, list[float]] = {x: [] for x in X_values}
    probe_rate_by_seed: dict[float, list[float]] = {x: [] for x in X_values}
    random_rate_by_seed: dict[float, list[float]] = {x: [] for x in X_values}
    decile_actioned_per_email: list[np.ndarray] = []
    spearman_rhos: list[float] = []
    spearman_pvals: list[float] = []

    for r in results:
        for x in X_values:
            lift_by_seed[x].extend(r["lift_by_seed"][x])
            probe_rate_by_seed[x].extend(r["probe_rate_by_seed"][x])
            random_rate_by_seed[x].extend(r["random_rate_by_seed"][x])
        decile_actioned_per_email.extend(r["decile_actioned_per_email"])
        spearman_rhos.extend(r["spearman_rhos"])
        spearman_pvals.extend(r["spearman_pvals"])

    print("Q6: computing calibration and Spearman stats")
    if len(decile_actioned_per_email) > 0:
        pooled_deciles = np.nanmean(np.stack(decile_actioned_per_email, axis=0), axis=0)
        email_mean_rates = np.nanmean(
            np.stack(decile_actioned_per_email, axis=0), axis=1, keepdims=True
        )
        centered_decile_mat = np.stack(decile_actioned_per_email, axis=0) - email_mean_rates
        pooled_centered = np.nanmean(centered_decile_mat, axis=0)
    else:
        pooled_deciles = np.zeros(10)
        pooled_centered = np.zeros(10)

    n_emails_spearman = len(spearman_rhos)
    if n_emails_spearman > 0:
        bonferroni_alpha = 0.05 / n_emails_spearman
        n_sig = int(np.sum(np.array(spearman_pvals) < bonferroni_alpha))
        frac_sig = n_sig / n_emails_spearman
    else:
        n_sig = 0
        frac_sig = 0.0

    table_html_rows = ""
    locked_lift = float("nan")
    locked_probe_rate = float("nan")
    locked_random_rate = float("nan")
    for x in X_values:
        lifts = np.array(lift_by_seed[x])
        pr_arr = np.array(probe_rate_by_seed[x])
        rr_arr = np.array(random_rate_by_seed[x])
        mean_lift = float(np.mean(lifts)) if len(lifts) > 0 else float("nan")
        p5 = float(np.percentile(lifts, 5)) if len(lifts) > 0 else float("nan")
        p95 = float(np.percentile(lifts, 95)) if len(lifts) > 0 else float("nan")
        mean_pr = float(np.mean(pr_arr)) if len(pr_arr) > 0 else float("nan")
        mean_rr = float(np.mean(rr_arr)) if len(rr_arr) > 0 else float("nan")

        if abs(x - locked_X) < 1e-9:
            locked_lift = mean_lift
            locked_probe_rate = mean_pr
            locked_random_rate = mean_rr

        label = f"{int(x * 100)}%" + (" (locked)" if abs(x - locked_X) < 1e-9 else "")
        table_html_rows += (
            f"<tr><td>{label}</td>"
            f"<td>{mean_pr:.3%}</td>"
            f"<td>{mean_rr:.3%}</td>"
            f"<td>{mean_lift:.3f}x</td>"
            f"<td>({p5:.3f}x, {p95:.3f}x)</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>X</th><th>Probe-routed rate</th><th>Random rate</th>"
        "<th>Relative lift</th><th>Seed-variance band (p5, p95)</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    print(
        f"\n=== Q6 HEADLINE (locked X=25%) ===\n"
        f"  Probe-routed rate: {locked_probe_rate:.3%}\n"
        f"  Random rate:       {locked_random_rate:.3%}\n"
        f"  Relative lift:     {locked_lift:.3f}x\n"
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 14))

    ax0 = axes[0]
    x_pcts = [int(x * 100) for x in X_values]
    mean_lifts = [
        float(np.mean(lift_by_seed[x])) if lift_by_seed[x] else float("nan") for x in X_values
    ]
    p5_lifts = [
        float(np.percentile(lift_by_seed[x], 5)) if lift_by_seed[x] else float("nan")
        for x in X_values
    ]
    p95_lifts = [
        float(np.percentile(lift_by_seed[x], 95)) if lift_by_seed[x] else float("nan")
        for x in X_values
    ]
    ax0.plot(x_pcts, mean_lifts, color="#3b82f6", linewidth=2, marker="o", label="Mean lift")
    ax0.fill_between(x_pcts, p5_lifts, p95_lifts, alpha=0.2, color="#3b82f6", label="5th–95th pct")
    ax0.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1.2, label="No lift (1.0x)")
    ax0.axhline(1.10, color="#ef4444", linestyle="--", linewidth=1.2, label="10% lift target")
    ax0.axvline(25, color="#f59e0b", linestyle=":", linewidth=1.5, label="Locked X=25%")
    ax0.set_xlabel("X — top % of sendable holdouts routed")
    ax0.set_ylabel("Relative lift vs random")
    ax0.set_title("(a) Lift-vs-X curve with seed variance band")
    ax0.legend(fontsize=9)

    ax1 = axes[1]
    decile_x = np.arange(1, 11)
    ax1.bar(
        decile_x,
        pooled_centered,
        color="#0ea5e9",
        edgecolor="white",
        label="Pooled within-email (re-centered)",
    )
    ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Score decile (1=lowest, 10=highest)")
    ax1.set_ylabel("Mean action rate (re-centered within email)")
    ax1.set_title("(b) Pooled calibration: action rate by probe-score decile")
    ax1.legend(fontsize=9)

    ax2 = axes[2]
    if len(spearman_rhos) > 0:
        ax2.hist(spearman_rhos, bins=30, color="#8b5cf6", edgecolor="white")
        ax2.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2)
        ax2.set_title(
            f"(c) Per-email Spearman ρ histogram\n"
            f"Bonferroni-significant (α=0.05): {n_sig}/{n_emails_spearman} ({frac_sig:.1%})"
        )
    else:
        ax2.text(0.5, 0.5, "No Spearman data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Spearman ρ (score vs actioned)")
    ax2.set_ylabel("# emails")

    fig.suptitle("Q6. Probe-routing lift — calibration + lift-vs-X", fontsize=13)
    fig.tight_layout()

    prose = (
        f"Probe-routing evaluated across <b>{n_seeds_use} seeds</b> "
        f"and <b>{n_test_emails} test emails</b>. "
        f"Scoring method: <b>{_Q6_SCORE_FUNCTION_NAME}</b> (OR-union of probe-signer train signatures, "
        f"then per-holdout Jaccard against that combined signature). "
        f"Cold-start fallback (min_signers = {min_signers}): score set to 0 for emails with fewer than "
        f"{min_signers} probe signers (kept in blended metric). "
        f"Locked X = {int(locked_X * 100)}%: probe-routed rate <b>{locked_probe_rate:.3%}</b> "
        f"vs random <b>{locked_random_rate:.3%}</b> → relative lift <b>{locked_lift:.3f}x</b>. "
        f"Calibration: {n_sig}/{n_emails_spearman} emails ({frac_sig:.1%}) show Bonferroni-significant "
        f"positive Spearman between score and actioned. "
        f"Note: OR-union Jaccard is used instead of mean-over-probe-signers Jaccard for vectorization efficiency; "
        f"both compute how much a holdout user's train history overlaps the probe signers' combined footprint. "
        f"{table_html}"
    )

    return Section(
        slug="q6_probe_routing",
        heading="Q6. Probe-routing lift — calibration, lift-vs-X, and locked-X result",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


def _section_q7_activity_control(
    wide: pl.DataFrame,
    train_email_ids: np.ndarray,
    test_email_ids: np.ndarray,
    user_features_train: pl.DataFrame,
    agg_v3_df: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:
    """Q7: 5-arm fair activity control at locked X=25%."""
    locked_X = 0.25
    min_signers = 3
    n_random_subseeds = 5

    print("Q7: setting up data structures")
    train_set = set(train_email_ids.tolist())
    test_set = set(test_email_ids.tolist())

    train_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in train_set], dtype=np.intp
    )
    test_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}

    eligible_user_ids = user_features_train["user_id"].to_numpy()
    eligible_rows = np.array(
        [uid_to_row[int(uid)] for uid in eligible_user_ids if int(uid) in uid_to_row],
        dtype=np.intp,
    )

    eligible_row_to_local = {int(r): i for i, r in enumerate(eligible_rows)}

    M_train_filtered = M_full[eligible_rows, :][:, train_col_idx].tocsr()
    counts_h = np.asarray(M_train_filtered.sum(axis=1)).flatten().astype(np.float32)

    test_email_ids_ordered = email_ids_all[test_col_idx]
    n_test_emails = len(test_col_idx)

    action_rate_eb = user_features_train["action_rate_train_eb"].to_numpy().astype(np.float32)

    non_probe_predictors = ["total_actions (v3)", "total_action_rate (v3)"]
    non_probe_winner = "total_action_rate (v3)"

    agg_lookup = {
        int(row["email_id"]): (int(row["total_actions"]), int(row["total_sends"]))
        for row in agg_v3_df.iter_rows(named=True)
    }
    total_actions_arr = np.array(
        [agg_lookup.get(int(eid), (0, 0))[0] for eid in test_email_ids_ordered], dtype=np.float64
    )
    total_sends_arr = np.array(
        [agg_lookup.get(int(eid), (0, 1))[1] for eid in test_email_ids_ordered], dtype=np.float64
    )
    total_sends_arr = np.maximum(total_sends_arr, 1)
    total_action_rate_arr = total_actions_arr / total_sends_arr

    print(f"Q7: computing sendable mask for {n_test_emails} test emails (vectorized)")
    sendable_map = _sendable_set(wide, test_email_ids)

    sendable_mask = np.zeros((len(eligible_rows), n_test_emails), dtype=bool)
    for e_idx, eid in enumerate(test_email_ids_ordered):
        sendable_uids = sendable_map.get(int(eid), np.array([], dtype=np.int64))
        if len(sendable_uids) == 0:
            continue
        local_indices = np.array(
            [
                eligible_row_to_local[uid_to_row[int(uid)]]
                for uid in sendable_uids
                if int(uid) in uid_to_row and uid_to_row[int(uid)] in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(local_indices) > 0:
            sendable_mask[local_indices, e_idx] = True

    M_test_rows = M_full[eligible_rows, :][:, test_col_idx].tocsr()

    n_seeds_use = len(_PROBE_SEED_CACHE)
    seeds_sorted = sorted(_PROBE_SEED_CACHE.keys())

    arm_names = [
        "Random",
        "Activity-top (EB-shrunk rate)",
        "Best-popularity (Q4 non-probe winner)",
        "Probe-routed (Jaccard)",
        "Probe + activity blend",
    ]
    arm_rates: dict[str, list[float]] = {a: [] for a in arm_names}
    global arm_rates_q7

    rng_main = np.random.default_rng(99)

    print(f"Q7: running 5 arms over {n_seeds_use} seeds × {n_test_emails} test emails (vectorized)")
    for s_idx, seed in enumerate(seeds_sorted):
        if s_idx % 5 == 0:
            print(f"  seed {s_idx + 1}/{n_seeds_use}")

        probe_ids, holdout_ids = _PROBE_SEED_CACHE[seed]

        probe_rows_global = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )

        holdout_ids_arr = np.asarray(holdout_ids)
        holdout_uids_in_eligible = np.isin(user_ids_all[eligible_rows], holdout_ids_arr)
        holdout_local_idx = np.where(holdout_uids_in_eligible)[0].astype(np.intp)

        if len(probe_rows_global) == 0 or len(holdout_local_idx) == 0:
            continue

        probe_local_idx = np.array(
            [
                eligible_row_to_local[int(r)]
                for r in probe_rows_global
                if int(r) in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(probe_local_idx) == 0:
            continue

        M_probe_test = M_full[probe_rows_global, :][:, test_col_idx].tocsr()
        M_probe_train = M_train_filtered[probe_local_idx, :]

        n_probe_signers_per_email = np.asarray(M_probe_test.sum(axis=0)).flatten()

        combined = (M_probe_train.T @ M_probe_test).astype(bool).astype(np.float32)
        if sp.issparse(combined):
            combined = np.asarray(combined.todense())
        combined_sum_per_email = combined.sum(axis=0).flatten()

        M_holdout_train = M_train_filtered[holdout_local_idx, :]
        intersect_full = M_holdout_train @ combined
        if sp.issparse(intersect_full):
            intersect_full = np.asarray(intersect_full.todense())

        counts_h_holdout = counts_h[holdout_local_idx]
        union_full = counts_h_holdout[:, None] + combined_sum_per_email[None, :] - intersect_full
        probe_scores_matrix = np.where(union_full > 0, intersect_full / union_full, 0.0).astype(
            np.float32
        )
        probe_scores_matrix[:, n_probe_signers_per_email < min_signers] = 0.0

        actioned_matrix = np.asarray(M_test_rows[holdout_local_idx, :].todense()).astype(np.float32)
        activity_scores_holdout = action_rate_eb[holdout_local_idx]

        sendable_holdout_mask = sendable_mask[holdout_local_idx, :]

        for e_idx in range(n_test_emails):
            sendable_local = np.where(sendable_holdout_mask[:, e_idx])[0]
            if len(sendable_local) == 0:
                continue

            n_sendable = len(sendable_local)
            n_top = max(1, int(np.floor(n_sendable * locked_X)))

            actioned_h = actioned_matrix[sendable_local, e_idx]

            random_sub_rates = []
            for sub_seed in range(n_random_subseeds):
                rand_idx = rng_main.choice(n_sendable, size=n_top, replace=False)
                random_sub_rates.append(float(actioned_h[rand_idx].mean()))
            arm_rates["Random"].append(float(np.mean(random_sub_rates)))

            activity_scores = activity_scores_holdout[sendable_local]
            top_idx_act = np.argpartition(-activity_scores, min(n_top, n_sendable) - 1)[:n_top]
            arm_rates["Activity-top (EB-shrunk rate)"].append(float(actioned_h[top_idx_act].mean()))

            email_pop = float(total_action_rate_arr[e_idx])
            pop_scores = activity_scores * email_pop
            top_idx_pop = np.argpartition(-pop_scores, min(n_top, n_sendable) - 1)[:n_top]
            arm_rates["Best-popularity (Q4 non-probe winner)"].append(
                float(actioned_h[top_idx_pop].mean())
            )

            probe_score_h = probe_scores_matrix[sendable_local, e_idx]

            top_idx_probe = np.argpartition(-probe_score_h, min(n_top, n_sendable) - 1)[:n_top]
            arm_rates["Probe-routed (Jaccard)"].append(float(actioned_h[top_idx_probe].mean()))

            act_std = float(np.std(activity_scores)) if np.std(activity_scores) > 0 else 1.0
            activity_znorm = (activity_scores - float(np.mean(activity_scores))) / act_std

            probe_std = float(np.std(probe_score_h)) if np.std(probe_score_h) > 0 else 1.0
            probe_znorm = (probe_score_h - float(np.mean(probe_score_h))) / probe_std

            blend_score = activity_znorm + probe_znorm
            top_idx_blend = np.argpartition(-blend_score, min(n_top, n_sendable) - 1)[:n_top]
            arm_rates["Probe + activity blend"].append(float(actioned_h[top_idx_blend].mean()))

    print("Q7: computing per-arm summary stats")
    random_mean = float(np.mean(arm_rates["Random"])) if arm_rates["Random"] else float("nan")
    activity_mean = (
        float(np.mean(arm_rates["Activity-top (EB-shrunk rate)"]))
        if arm_rates["Activity-top (EB-shrunk rate)"]
        else float("nan")
    )
    probe_mean = (
        float(np.mean(arm_rates["Probe-routed (Jaccard)"]))
        if arm_rates["Probe-routed (Jaccard)"]
        else float("nan")
    )

    table_html_rows = ""
    arm_summary: dict[str, dict] = {}
    for arm in arm_names:
        rates = np.array(arm_rates[arm])
        if len(rates) == 0:
            arm_summary[arm] = {
                "mean": float("nan"),
                "lift": float("nan"),
                "p5": float("nan"),
                "p95": float("nan"),
                "beats_activity": "—",
            }
            continue
        mean_r = float(np.mean(rates))
        lift_vs_random = mean_r / random_mean if random_mean > 0 else float("nan")
        lift_vs_activity = (mean_r / activity_mean) if activity_mean > 0 else float("nan")
        beats_activity = "✓" if lift_vs_activity >= 1.10 else "✗" if arm != "Random" else "—"
        p5 = float(np.percentile(rates, 5))
        p95 = float(np.percentile(rates, 95))
        arm_summary[arm] = {
            "mean": mean_r,
            "lift": lift_vs_random,
            "p5": p5,
            "p95": p95,
            "beats_activity": beats_activity,
        }

        table_html_rows += (
            f"<tr><td>{arm}</td>"
            f"<td>{mean_r:.3%}</td>"
            f"<td>{lift_vs_random:.3f}x</td>"
            f"<td>{beats_activity}</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Arm</th><th>Action rate</th><th>Relative lift vs random</th>"
        "<th>Beats Activity-top by ≥10%?</th></tr></thead>"
        f"<tbody>{table_html_rows}</tbody></table>"
    )

    probe_lift = arm_summary["Probe-routed (Jaccard)"]["lift"]
    best_pop_lift = arm_summary["Best-popularity (Q4 non-probe winner)"]["lift"]
    activity_lift = arm_summary["Activity-top (EB-shrunk rate)"]["lift"]
    probe_beats_both = (
        not np.isnan(probe_lift)
        and not np.isnan(activity_lift)
        and not np.isnan(best_pop_lift)
        and (probe_lift / activity_lift) >= 1.10
        and (probe_lift / best_pop_lift) >= 1.10
    )

    print(
        f"\n=== Q7 HEADLINE (locked X=25%) ===\n"
        f"  Random rate:         {random_mean:.3%}\n"
        f"  Activity-top rate:   {activity_mean:.3%} (lift {activity_lift:.3f}x)\n"
        f"  Best-popularity rate:{arm_summary['Best-popularity (Q4 non-probe winner)']['mean']:.3%} (lift {best_pop_lift:.3f}x)\n"
        f"  Probe-routed rate:   {probe_mean:.3%} (lift {probe_lift:.3f}x)\n"
        f"  Probe beats Activity-top AND Best-popularity by ≥10%: {'YES' if probe_beats_both else 'NO'}\n"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    means = [arm_summary[a]["mean"] for a in arm_names]
    p5s = [arm_summary[a]["p5"] for a in arm_names]
    p95s = [arm_summary[a]["p95"] for a in arm_names]
    colors_bars = []
    for a in arm_names:
        if "Probe-routed" in a:
            colors_bars.append("#22c55e" if probe_beats_both else "#ef4444")
        elif "blend" in a:
            colors_bars.append("#8b5cf6")
        elif "Activity" in a:
            colors_bars.append("#f59e0b")
        elif "popularity" in a:
            colors_bars.append("#f97316")
        else:
            colors_bars.append("#94a3b8")

    short_labels = [
        "Random",
        "Activity-top",
        "Best-popularity",
        "Probe-routed",
        "Probe+Activity",
    ]
    x_pos = np.arange(len(arm_names))
    bars = ax.bar(x_pos, means, color=colors_bars, edgecolor="white", width=0.6)
    err_lo = [m - p5 for m, p5 in zip(means, p5s)]
    err_hi = [p95 - m for m, p95 in zip(means, p95s)]
    ax.errorbar(
        x_pos,
        means,
        yerr=[err_lo, err_hi],
        fmt="none",
        color="#0f172a",
        capsize=5,
        linewidth=1.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel("Mean action rate (locked X=25%)")
    ax.set_title(
        "Q7. Fair activity control — probe vs competing arms\n"
        f"Probe-routed bar: {'green (beats both by ≥10%)' if probe_beats_both else 'red (does not beat both by ≥10%)'}"
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    fig.tight_layout()

    prose = (
        f"5-arm head-to-head at locked X=25% over {n_seeds_use} seeds × {n_test_emails} test emails. "
        f"<b>Best-popularity arm</b> uses <code>non_probe_score[h,e] = action_rate_train_eb[h] × total_action_rate[e]</code> "
        f"(user propensity scaled by email popularity — the strongest non-probe-based ranking per Q4 non-probe predictors). "
        f"<b>Probe-routed arm</b> uses {_Q6_SCORE_FUNCTION_NAME} (same as Q6). "
        f"<b>Probe + activity blend</b> z-score normalizes both scores within each email then sums. "
        f"Probe-routed beats Activity-top AND Best-popularity by ≥10%: "
        f"<b style='color:{'green' if probe_beats_both else 'red'}'>{'YES' if probe_beats_both else 'NO'}</b>. "
        f"{table_html}"
    )

    arm_rates_q7.update(arm_rates)

    return Section(
        slug="q7_activity_control",
        heading="Q7. Fair activity control — does probe-routing beat 'just target active users'?",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Q8 worker.
# ---------------------------------------------------------------------------


def _q8_seed_worker(args: tuple) -> dict:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import numpy as np
    import scipy.sparse as sp

    (
        seed,
        M_train_filtered_data,
        M_train_filtered_indices,
        M_train_filtered_indptr,
        M_train_filtered_shape,
        M_probe_test_data,
        M_probe_test_indices,
        M_probe_test_indptr,
        M_probe_test_shape,
        M_test_rows_data,
        M_test_rows_indices,
        M_test_rows_indptr,
        M_test_rows_shape,
        probe_local_idx,
        holdout_local_idx,
        counts_h,
        sendable_mask,
        n_probe_signers_per_email,
        min_signers,
        n_test_emails,
        user_tier_labels,
        email_tier_labels,
        n_user_tiers,
        n_email_tiers,
        locked_X,
        n_random_subseeds,
    ) = args

    M_train_filtered = sp.csr_matrix(
        (M_train_filtered_data, M_train_filtered_indices, M_train_filtered_indptr),
        shape=M_train_filtered_shape,
    )
    M_probe_test = sp.csr_matrix(
        (M_probe_test_data, M_probe_test_indices, M_probe_test_indptr),
        shape=M_probe_test_shape,
    )
    M_test_rows = sp.csr_matrix(
        (M_test_rows_data, M_test_rows_indices, M_test_rows_indptr),
        shape=M_test_rows_shape,
    )

    M_probe_train = M_train_filtered[probe_local_idx, :]
    combined = (M_probe_train.T @ M_probe_test).astype(bool).astype(np.float32)
    if sp.issparse(combined):
        combined = np.asarray(combined.todense())
    combined_sum_per_email = combined.sum(axis=0).flatten()

    M_holdout_train = M_train_filtered[holdout_local_idx, :]
    intersect_full = M_holdout_train @ combined
    if sp.issparse(intersect_full):
        intersect_full = np.asarray(intersect_full.todense())

    counts_h_holdout = counts_h[holdout_local_idx]
    union_full = counts_h_holdout[:, None] + combined_sum_per_email[None, :] - intersect_full
    probe_scores = np.where(union_full > 0, intersect_full / union_full, 0.0).astype(np.float32)
    probe_scores[:, n_probe_signers_per_email < min_signers] = 0.0

    actioned_matrix = np.asarray(M_test_rows[holdout_local_idx, :].todense()).astype(np.float32)
    sendable_holdout_mask = sendable_mask[holdout_local_idx, :]

    holdout_user_tiers = user_tier_labels[holdout_local_idx]

    rng = np.random.default_rng(42 + seed)

    cell_probe_rates: dict[tuple, list[float]] = {}
    cell_random_rates: dict[tuple, list[float]] = {}
    cell_activity_rates: dict[tuple, list[float]] = {}

    for ut in range(n_user_tiers):
        for et in range(n_email_tiers):
            cell_probe_rates[(ut, et)] = []
            cell_random_rates[(ut, et)] = []
            cell_activity_rates[(ut, et)] = []

    email_tier_mask_per_tier = [np.where(email_tier_labels == et)[0] for et in range(n_email_tiers)]

    for e_idx in range(n_test_emails):
        et = int(email_tier_labels[e_idx])
        sendable_local = np.where(sendable_holdout_mask[:, e_idx])[0]
        if len(sendable_local) == 0:
            continue
        actioned_h = actioned_matrix[sendable_local, e_idx]
        probe_score_h = probe_scores[sendable_local, e_idx]
        user_tiers_local = holdout_user_tiers[sendable_local]

        for ut in range(n_user_tiers):
            ut_mask = user_tiers_local == ut
            ut_local = np.where(ut_mask)[0]
            if len(ut_local) < 2:
                continue

            n_top = max(1, int(np.floor(len(ut_local) * locked_X)))
            actioned_cell = actioned_h[ut_local]
            scores_cell = probe_score_h[ut_local]

            top_idx_probe = np.argpartition(-scores_cell, min(n_top, len(ut_local)) - 1)[:n_top]
            cell_probe_rates[(ut, et)].append(float(actioned_cell[top_idx_probe].mean()))

            rand_rates_sub = []
            for _ in range(n_random_subseeds):
                rand_idx = rng.choice(len(ut_local), size=n_top, replace=False)
                rand_rates_sub.append(float(actioned_cell[rand_idx].mean()))
            cell_random_rates[(ut, et)].append(float(np.mean(rand_rates_sub)))

    return {
        "cell_probe_rates": {str(k): v for k, v in cell_probe_rates.items()},
        "cell_random_rates": {str(k): v for k, v in cell_random_rates.items()},
    }


def _section_q8_joint_stratification(
    wide: pl.DataFrame,
    train_email_ids: np.ndarray,
    test_email_ids: np.ndarray,
    user_features_train: pl.DataFrame,
    agg_v3_df: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:
    """Q8: joint stratification — does probe-routing win in any sub-segment?"""
    import multiprocessing
    import os

    import matplotlib.patches as mpatches

    locked_X = 0.25
    min_signers = 3
    n_random_subseeds = 5
    n_bootstrap = 200
    n_user_tiers = 4
    n_email_tiers = 4

    PRE_SPECIFIED = {(0, 0), (0, 1), (1, 0), (1, 1)}

    print("Q8: setting up data structures")
    train_set = set(train_email_ids.tolist())
    test_set = set(test_email_ids.tolist())

    train_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in train_set], dtype=np.intp
    )
    test_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )
    test_email_ids_ordered = email_ids_all[test_col_idx]
    n_test_emails = len(test_col_idx)

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}

    eligible_user_ids = user_features_train["user_id"].to_numpy()
    eligible_rows = np.array(
        [uid_to_row[int(uid)] for uid in eligible_user_ids if int(uid) in uid_to_row],
        dtype=np.intp,
    )
    eligible_row_to_local = {int(r): i for i, r in enumerate(eligible_rows)}

    M_train_filtered = M_full[eligible_rows, :][:, train_col_idx].tocsr()
    counts_h = np.asarray(M_train_filtered.sum(axis=1)).flatten().astype(np.float32)

    print("Q8: computing email tiers from agg_v3 total_actions")
    agg_lookup_ta = {
        int(row["email_id"]): int(row["total_actions"]) for row in agg_v3_df.iter_rows(named=True)
    }
    total_actions_test = np.array(
        [agg_lookup_ta.get(int(eid), 0) for eid in test_email_ids_ordered], dtype=np.float64
    )
    email_tier_labels = np.zeros(n_test_emails, dtype=np.int32)
    email_tier_edges_q = np.quantile(total_actions_test, [0.25, 0.50, 0.75])
    for i, ta in enumerate(total_actions_test):
        email_tier_labels[i] = int(np.searchsorted(email_tier_edges_q, ta, side="right"))
    email_tier_labels = np.clip(email_tier_labels, 0, n_email_tiers - 1)

    print("Q8: computing user tiers from action_rate_train_eb")
    eb_arr = user_features_train["action_rate_train_eb"].to_numpy().astype(np.float32)
    user_tier_q = np.quantile(eb_arr, [0.25, 0.50, 0.75])
    user_tier_labels_full = np.clip(
        np.searchsorted(user_tier_q, eb_arr, side="right"), 0, n_user_tiers - 1
    ).astype(np.int32)

    print(f"Q8: computing sendable mask for {n_test_emails} test emails")
    sendable_map = _sendable_set(wide, test_email_ids)
    sendable_mask = np.zeros((len(eligible_rows), n_test_emails), dtype=bool)
    for e_idx, eid in enumerate(test_email_ids_ordered):
        sendable_uids = sendable_map.get(int(eid), np.array([], dtype=np.int64))
        if len(sendable_uids) == 0:
            continue
        local_indices = np.array(
            [
                eligible_row_to_local[uid_to_row[int(uid)]]
                for uid in sendable_uids
                if int(uid) in uid_to_row and uid_to_row[int(uid)] in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(local_indices) > 0:
            sendable_mask[local_indices, e_idx] = True

    M_test_rows = M_full[eligible_rows, :][:, test_col_idx].tocsr()

    seeds_sorted = sorted(_PROBE_SEED_CACHE.keys())
    n_seeds_use = len(seeds_sorted)

    print(f"Q8: building worker args for {n_seeds_use} seeds")
    worker_args = []
    for seed in seeds_sorted:
        probe_ids, holdout_ids = _PROBE_SEED_CACHE[seed]

        probe_rows_global = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )
        if len(probe_rows_global) == 0:
            continue

        probe_local_idx = np.array(
            [
                eligible_row_to_local[int(r)]
                for r in probe_rows_global
                if int(r) in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(probe_local_idx) == 0:
            continue

        holdout_ids_arr = np.asarray(holdout_ids)
        holdout_uids_in_eligible = np.isin(user_ids_all[eligible_rows], holdout_ids_arr)
        holdout_local_idx = np.where(holdout_uids_in_eligible)[0].astype(np.intp)
        if len(holdout_local_idx) == 0:
            continue

        M_probe_test = M_full[probe_rows_global, :][:, test_col_idx].tocsr()
        n_probe_signers_per_email = np.asarray(M_probe_test.sum(axis=0)).flatten()

        mt = M_train_filtered
        mpt = M_probe_test
        mtr = M_test_rows

        worker_args.append(
            (
                seed,
                mt.data,
                mt.indices,
                mt.indptr,
                mt.shape,
                mpt.data,
                mpt.indices,
                mpt.indptr,
                mpt.shape,
                mtr.data,
                mtr.indices,
                mtr.indptr,
                mtr.shape,
                probe_local_idx,
                holdout_local_idx,
                counts_h,
                sendable_mask,
                n_probe_signers_per_email,
                min_signers,
                n_test_emails,
                user_tier_labels_full,
                email_tier_labels,
                n_user_tiers,
                n_email_tiers,
                locked_X,
                n_random_subseeds,
            )
        )

    n_processes = min(8, os.cpu_count() or 1)
    print(f"Q8: running {len(worker_args)} seeds across {n_processes} processes")
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(_q8_seed_worker, worker_args)

    print("Q8: aggregating results across seeds")

    cell_probe_rates: dict[tuple, list[float]] = {}
    cell_random_rates: dict[tuple, list[float]] = {}
    for ut in range(n_user_tiers):
        for et in range(n_email_tiers):
            cell_probe_rates[(ut, et)] = []
            cell_random_rates[(ut, et)] = []

    for r in results:
        for k_str, v in r["cell_probe_rates"].items():
            k = tuple(int(x) for x in k_str.strip("()").split(", "))
            cell_probe_rates[k].extend(v)
        for k_str, v in r["cell_random_rates"].items():
            k = tuple(int(x) for x in k_str.strip("()").split(", "))
            cell_random_rates[k].extend(v)

    print("Q8: computing bootstrap CIs over seeds (200 resamples per cell)")
    rng_boot = np.random.default_rng(7)

    cell_lift_mean: dict[tuple, float] = {}
    cell_lift_lo: dict[tuple, float] = {}
    cell_lift_hi: dict[tuple, float] = {}

    for ut in range(n_user_tiers):
        for et in range(n_email_tiers):
            pr = np.array(cell_probe_rates[(ut, et)])
            rr = np.array(cell_random_rates[(ut, et)])
            if len(pr) == 0 or np.nanmean(rr) == 0:
                cell_lift_mean[(ut, et)] = float("nan")
                cell_lift_lo[(ut, et)] = float("nan")
                cell_lift_hi[(ut, et)] = float("nan")
                continue
            mean_lift = (
                float(np.nanmean(pr) / np.nanmean(rr)) if np.nanmean(rr) > 0 else float("nan")
            )
            n_obs = len(pr)
            boot_lifts = []
            for _ in range(n_bootstrap):
                idx = rng_boot.integers(0, n_obs, size=n_obs)
                br = float(np.nanmean(pr[idx]))
                rr_b = float(np.nanmean(rr[idx]))
                boot_lifts.append(br / rr_b if rr_b > 0 else mean_lift)
            boot_arr = np.array(boot_lifts)
            cell_lift_mean[(ut, et)] = mean_lift
            cell_lift_lo[(ut, et)] = float(np.percentile(boot_arr, 2.5))
            cell_lift_hi[(ut, et)] = float(np.percentile(boot_arr, 97.5))

    _q8_dnh = [
        (ut, et)
        for ut in range(n_user_tiers)
        for et in range(n_email_tiers)
        if not np.isnan(cell_lift_hi[(ut, et)]) and cell_lift_hi[(ut, et)] < 1.0
    ]
    global do_no_harm_violations
    do_no_harm_violations = _q8_dnh
    print(f"Q8: do-no-harm violations (95% CI upper < 1.0): {len(do_no_harm_violations)} cells")

    print("Q8: per-cell verdicts for pre-specified positive cells:")
    for ut, et in sorted(PRE_SPECIFIED):
        mean_l = cell_lift_mean.get((ut, et), float("nan"))
        lo = cell_lift_lo.get((ut, et), float("nan"))
        hi = cell_lift_hi.get((ut, et), float("nan"))
        pr_mean = (
            float(np.nanmean(cell_probe_rates[(ut, et)]))
            if cell_probe_rates[(ut, et)]
            else float("nan")
        )
        rr_mean = (
            float(np.nanmean(cell_random_rates[(ut, et)]))
            if cell_random_rates[(ut, et)]
            else float("nan")
        )
        verdict = "WIN" if (not np.isnan(lo) and lo >= 1.0) else "LOSE/UNCERTAIN"
        print(
            f"  Cell (Email Q{et + 1}, User Q{ut + 1}): lift={mean_l:.3f}x CI=({lo:.3f},{hi:.3f}) "
            f"probe_rate={pr_mean:.3%} random_rate={rr_mean:.3%} → {verdict}"
        )

    tier_labels_email = [f"Email Q{i + 1}" for i in range(n_email_tiers)]
    tier_labels_user = [f"User Q{i + 1}" for i in range(n_user_tiers)]

    table_rows = ""
    for et in range(n_email_tiers):
        row = f"<tr><td><b>{tier_labels_email[et]}</b></td>"
        for ut in range(n_user_tiers):
            mean_l = cell_lift_mean.get((ut, et), float("nan"))
            lo = cell_lift_lo.get((ut, et), float("nan"))
            hi = cell_lift_hi.get((ut, et), float("nan"))
            cell_text = f"{mean_l:.2f}x<br/>(CI {lo:.2f},{hi:.2f})"
            style = ""
            if (ut, et) in PRE_SPECIFIED:
                style = "border: 2px solid #16a34a;"
            if not np.isnan(hi) and hi < 1.0:
                style = "border: 2px solid #dc2626;"
            row += f"<td style='{style} padding:6px'>{cell_text}</td>"
        row += "</tr>"
        table_rows += row

    header = "<tr><th></th>" + "".join(f"<th>{l}</th>" for l in tier_labels_user) + "</tr>"
    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        f"<thead>{header}</thead><tbody>{table_rows}</tbody></table>"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax_heat = axes[0]
    heat_data = np.full((n_email_tiers, n_user_tiers), np.nan)
    for ut in range(n_user_tiers):
        for et in range(n_email_tiers):
            heat_data[et, ut] = cell_lift_mean.get((ut, et), np.nan)

    vmin = float(np.nanmin(heat_data))
    vmax = float(np.nanmax(heat_data))
    center = 1.0
    half_range = max(abs(vmax - center), abs(vmin - center), 0.01)
    im = ax_heat.imshow(
        heat_data,
        cmap="RdBu_r",
        aspect="auto",
        vmin=center - half_range,
        vmax=center + half_range,
    )
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    ax_heat.set_xticks(range(n_user_tiers))
    ax_heat.set_xticklabels(tier_labels_user, fontsize=9)
    ax_heat.set_yticks(range(n_email_tiers))
    ax_heat.set_yticklabels(tier_labels_email, fontsize=9)
    ax_heat.set_title("(a) Probe lift heatmap (4×4)\nGreen=pre-specified, Red=do-no-harm violation")

    for et in range(n_email_tiers):
        for ut in range(n_user_tiers):
            mean_l = cell_lift_mean.get((ut, et), float("nan"))
            lo = cell_lift_lo.get((ut, et), float("nan"))
            hi = cell_lift_hi.get((ut, et), float("nan"))
            txt = f"{mean_l:.2f}x\n({lo:.2f},{hi:.2f})" if not np.isnan(mean_l) else "N/A"
            ax_heat.text(ut, et, txt, ha="center", va="center", fontsize=7, color="black")

            is_pre = (ut, et) in PRE_SPECIFIED
            is_harm = not np.isnan(hi) and hi < 1.0
            edge_color = "#16a34a" if is_pre else ("#dc2626" if is_harm else None)
            if edge_color:
                rect = mpatches.Rectangle(
                    (ut - 0.5, et - 0.5), 1, 1, linewidth=2, edgecolor=edge_color, facecolor="none"
                )
                ax_heat.add_patch(rect)

    ax_bar = axes[1]
    pre_cells_sorted = sorted(PRE_SPECIFIED)
    bar_group_labels = [f"E-Q{et + 1}/U-Q{ut + 1}" for ut, et in pre_cells_sorted]
    x_g = np.arange(len(pre_cells_sorted))
    width = 0.25

    random_means_pre = []
    probe_means_pre = []

    for ut, et in pre_cells_sorted:
        rr = np.array(cell_random_rates[(ut, et)])
        pr = np.array(cell_probe_rates[(ut, et)])
        random_means_pre.append(float(np.nanmean(rr)) if len(rr) > 0 else 0.0)
        probe_means_pre.append(float(np.nanmean(pr)) if len(pr) > 0 else 0.0)

    ax_bar.bar(x_g - width / 2, random_means_pre, width, label="Random", color="#94a3b8")
    ax_bar.bar(x_g + width / 2, probe_means_pre, width, label="Probe-routed", color="#3b82f6")

    ax_bar.set_xticks(x_g)
    ax_bar.set_xticklabels(bar_group_labels, fontsize=9)
    ax_bar.set_ylabel("Mean action rate")
    ax_bar.set_title("(b) Pre-specified cells: random vs probe-routed")
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax_bar.legend(fontsize=9)

    fig.suptitle("Q8. Joint stratification — probe lift by email × user tier", fontsize=13)
    fig.tight_layout()

    n_pre_wins = sum(
        1
        for ut, et in PRE_SPECIFIED
        if not np.isnan(cell_lift_lo.get((ut, et), float("nan"))) and cell_lift_lo[(ut, et)] >= 1.0
    )

    prose = (
        f"4×4 joint stratification at locked X=25% over {n_seeds_use} seeds. "
        f"<b>Email tiers</b>: Q1–Q4 by <code>total_actions</code> from agg_v3 (ascending). "
        f"<b>User tiers</b>: Q1–Q4 by <code>action_rate_train_eb</code> (ascending). "
        f"Bootstrap CIs: 200 seed-resamples per cell. "
        f"<b>Pre-specified positive cells</b> (Email Q1–Q2 × User Q1–Q2, outlined green): "
        f"{n_pre_wins}/4 have 95% CI lower bound ≥ 1.0 (probe significantly better than random within cell). "
        f"<b>Do-no-harm violations</b> (95% CI upper < 1.0, outlined red): "
        f"<b>{len(do_no_harm_violations)}</b> cells. "
        f"{table_html}"
    )

    return Section(
        slug="q8_stratification",
        heading="Q8. Joint stratification — does probe-routing win in any sub-segment?",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Q7 arm rates global store (populated by _section_q7_activity_control result).
# ---------------------------------------------------------------------------

arm_rates_q7: dict[str, list[float]] = {}

# ---------------------------------------------------------------------------
# Q8 do-no-harm violations store (populated by _section_q8_joint_stratification).
# ---------------------------------------------------------------------------

do_no_harm_violations: list = []


# ---------------------------------------------------------------------------
# Q9 section builder.
# ---------------------------------------------------------------------------


def _section_q9_delay_confounding(
    wide: pl.DataFrame,
    test_email_ids: np.ndarray,
    attributed_df: pl.DataFrame,
    user_features_train: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:
    """Q9: send-delay confounding — does probe-routing survive operational reality?"""
    print("Q9: building hazard curve from attributed_df")
    hours_bins, cumulative = _hazard_curve(wide, attributed_df)

    delay_hours_list = [0, 1, 2, 6, 12, 24]

    def _f_at_delay(d: float) -> float:
        if d <= 0.0:
            return 0.0
        idx = int(np.searchsorted(hours_bins, d, side="right")) - 1
        idx = max(0, min(idx, len(cumulative) - 1))
        return float(cumulative[idx])

    f1 = _f_at_delay(1.0)
    f6 = _f_at_delay(6.0)
    f24 = _f_at_delay(24.0)
    print(f"Q9: F(1h)={f1:.4f}  F(6h)={f6:.4f}  F(24h)={f24:.4f}")

    random_rate = 0.1180
    activity_top_rate = 0.2335
    best_pop_rate = 0.2326
    probe_raw_rate = 0.2135

    if arm_rates_q7:
        rr_list = arm_rates_q7.get("Random", [])
        at_list = arm_rates_q7.get("Activity-top (EB-shrunk rate)", [])
        bp_list = arm_rates_q7.get("Best-popularity (Q4 non-probe winner)", [])
        pr_list = arm_rates_q7.get("Probe-routed (Jaccard)", [])
        if rr_list:
            random_rate = float(np.mean(rr_list))
        if at_list:
            activity_top_rate = float(np.mean(at_list))
        if bp_list:
            best_pop_rate = float(np.mean(bp_list))
        if pr_list:
            probe_raw_rate = float(np.mean(pr_list))

    table_rows = ""
    delay_adjusted_rates = []
    for d in delay_hours_list:
        fd = _f_at_delay(float(d))
        adjustment = 1.0 - fd
        adj_rate = probe_raw_rate * adjustment
        delay_adjusted_rates.append((d, fd, adj_rate))
        lift_vs_random = adj_rate / random_rate if random_rate > 0 else float("nan")
        gap_to_activity = activity_top_rate - adj_rate
        label = f"{d}h" + (" (raw)" if d == 0 else "")
        table_rows += (
            f"<tr><td>{label}</td>"
            f"<td>{fd:.4f}</td>"
            f"<td>{adj_rate:.3%}</td>"
            f"<td>{lift_vs_random:.3f}x</td>"
            f"<td>{gap_to_activity:+.3%}</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr>"
        "<th>Delay (h)</th><th>F(D) hazard</th>"
        "<th>Probe-routed delay-adj rate</th>"
        "<th>Lift vs random</th>"
        "<th>Gap to activity-top</th>"
        "</tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
    )

    print("\n=== Q9 HEADLINE (delay-adjusted probe lift) ===")
    for d, fd, adj_rate in delay_adjusted_rates:
        lift = adj_rate / random_rate if random_rate > 0 else float("nan")
        print(f"  D={d:2d}h: F(D)={fd:.4f}  probe_adj={adj_rate:.3%}  lift={lift:.3f}x")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax_haz = axes[0]
    ax_haz.plot(hours_bins, cumulative, color="#3b82f6", linewidth=2)
    for dv, color in [(1.0, "#f59e0b"), (6.0, "#ef4444"), (24.0, "#8b5cf6")]:
        ax_haz.axvline(dv, linestyle="--", linewidth=1.2, color=color, label=f"{dv:.0f}h")
    ax_haz.set_xscale("log")
    ax_haz.set_xlabel("Hours since send (log scale)")
    ax_haz.set_ylabel("Cumulative fraction of actions")
    ax_haz.set_title("(a) Hazard curve: cumulative action probability")
    ax_haz.legend(fontsize=9)
    ax_haz.set_xlim(hours_bins[0], hours_bins[-1])

    ax_bar = axes[1]
    bar_labels = ["Random", "Activity-top", "Best-pop"]
    bar_rates = [random_rate, activity_top_rate, best_pop_rate]
    bar_colors_base = ["#94a3b8", "#f59e0b", "#f97316"]

    probe_bar_labels = []
    probe_bar_rates_list = []
    probe_bar_colors = []
    color_map_d = {0: "#22c55e", 1: "#86efac", 6: "#fca5a5", 24: "#ef4444"}
    for d, fd, adj_rate in delay_adjusted_rates:
        if d in (0, 1, 6, 24):
            probe_bar_labels.append(f"Probe\n@{d}h")
            probe_bar_rates_list.append(adj_rate)
            probe_bar_colors.append(color_map_d.get(d, "#3b82f6"))

    all_labels = bar_labels + probe_bar_labels
    all_rates = bar_rates + probe_bar_rates_list
    all_colors = bar_colors_base + probe_bar_colors

    x_pos = np.arange(len(all_labels))
    ax_bar.bar(x_pos, all_rates, color=all_colors, edgecolor="white", width=0.6)
    ax_bar.axhline(
        random_rate * 1.10,
        linestyle="--",
        color="#64748b",
        linewidth=1.2,
        label="1.10x lift target",
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(all_labels, fontsize=9)
    ax_bar.set_ylabel("Mean action rate")
    ax_bar.set_title("(b) Delay-adjusted probe lift vs baseline arms")
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax_bar.legend(fontsize=9)

    fig.suptitle("Q9. Send-delay confounding — delay-penalized probe lift", fontsize=13)
    fig.tight_layout()

    f6_note = f"At the primary delay assumption (D=6h): F(6h)={f6:.4f}, probe-routed delay-adjusted rate = {probe_raw_rate * (1 - f6):.3%}."

    prose = (
        f"Operational reality: holdout users receive the email <i>D</i> hours after probe arm. "
        f"If action probability decays with hours-since-send, offline replay overestimates real-world lift. "
        f"<b>Important caveat</b>: this assumes action timing is independent of user identity. "
        f"Active users may sign faster, so the correction is a first-order approximation only. "
        f"Delay-adjustment factor for delay <i>D</i>: <code>(1 − F(D))</code>, where F is the "
        f"cumulative fraction of actions observed within D hours of send. "
        f"Non-probe arms (random, activity-top, best-popularity) do NOT take this penalty — they "
        f"do not need probe data and can fire at the same time as everyone else. "
        f"F(1h)={f1:.4f}, F(6h)={f6:.4f}, F(24h)={f24:.4f}. "
        f"{f6_note} "
        f"{table_html}"
    )

    return Section(
        slug="q9_delay_confounding",
        heading="Q9. Send-delay confounding — does probe-routing survive the operational reality?",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Q10 and Q11 section builders.
# ---------------------------------------------------------------------------

_Q10_FOLD_MIN_ADJ_LIFT: float = float("nan")
_Q10_ADJ_MEAN: float = float("nan")


def _q10_fold_worker(args: tuple) -> dict:
    """Per-seed worker for a single Q10 fold — mirrors _q6_worker interface."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import numpy as np
    import scipy.sparse as sp

    (
        seed,
        s_idx,
        probe_ids,
        holdout_ids,
        M_train_filtered_data,
        M_train_filtered_indices,
        M_train_filtered_indptr,
        M_train_filtered_shape,
        M_probe_test_data,
        M_probe_test_indices,
        M_probe_test_indptr,
        M_probe_test_shape,
        M_test_rows_data,
        M_test_rows_indices,
        M_test_rows_indptr,
        M_test_rows_shape,
        probe_local_idx,
        eligible_rows_arr,
        user_ids_all,
        sendable_mask,
        holdout_local_idx,
        counts_h,
        locked_X,
        min_signers,
        n_test_emails,
    ) = args

    M_train_filtered = sp.csr_matrix(
        (M_train_filtered_data, M_train_filtered_indices, M_train_filtered_indptr),
        shape=M_train_filtered_shape,
    )
    M_probe_test = sp.csr_matrix(
        (M_probe_test_data, M_probe_test_indices, M_probe_test_indptr),
        shape=M_probe_test_shape,
    )
    M_test_rows = sp.csr_matrix(
        (M_test_rows_data, M_test_rows_indices, M_test_rows_indptr),
        shape=M_test_rows_shape,
    )

    M_probe_train = M_train_filtered[probe_local_idx, :]
    n_probe_signers_per_email = np.asarray(M_probe_test.sum(axis=0)).flatten()

    combined = (M_probe_train.T @ M_probe_test).astype(bool).astype(np.float32)
    if sp.issparse(combined):
        combined = np.asarray(combined.todense())
    combined_sum_per_email = combined.sum(axis=0).flatten()

    M_holdout_train = M_train_filtered[holdout_local_idx, :]
    intersect_full = M_holdout_train @ combined
    if sp.issparse(intersect_full):
        intersect_full = np.asarray(intersect_full.todense())

    counts_h_holdout = counts_h[holdout_local_idx]
    union_full = counts_h_holdout[:, None] + combined_sum_per_email[None, :] - intersect_full
    scores_matrix = np.where(union_full > 0, intersect_full / union_full, 0.0).astype(np.float32)
    scores_matrix[:, n_probe_signers_per_email < min_signers] = 0.0

    M_test_holdout = M_test_rows[holdout_local_idx, :]
    actioned_matrix = np.asarray(M_test_holdout.todense()).astype(np.float32)

    probe_rates = []
    random_rates = []

    rng = np.random.default_rng(42 + seed)
    sendable_holdout_mask = sendable_mask[holdout_local_idx, :]

    for e_idx in range(n_test_emails):
        sendable_local = np.where(sendable_holdout_mask[:, e_idx])[0]
        if len(sendable_local) == 0:
            continue

        score_h = scores_matrix[sendable_local, e_idx]
        actioned_h = actioned_matrix[sendable_local, e_idx]

        n_sendable = len(sendable_local)
        n_top = max(1, int(np.floor(n_sendable * locked_X)))

        top_idx = np.argpartition(-score_h, min(n_top, n_sendable) - 1)[:n_top]
        probe_rates.append(float(actioned_h[top_idx].mean()) if n_top > 0 else 0.0)

        rand_rates_sub = []
        for _ in range(3):
            rand_idx = rng.choice(n_sendable, size=n_top, replace=False)
            rand_rates_sub.append(float(actioned_h[rand_idx].mean()))
        random_rates.append(float(np.mean(rand_rates_sub)))

    return {"probe_rates": probe_rates, "random_rates": random_rates}


def _run_fold_lift(
    wide: pl.DataFrame,
    fold_train_ids: np.ndarray,
    fold_test_ids: np.ndarray,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
    locked_X: float = 0.25,
    min_signers: int = 3,
    n_seeds: int = 50,
) -> tuple[float, float, float]:
    """Run Q6 locked-X metric on a single fold.

    Returns (mean_probe_lift, mean_random_rate, mean_probe_rate).
    """
    import multiprocessing
    import os

    fold_features = _build_user_features_train(wide, fold_train_ids)

    train_set = set(fold_train_ids.tolist())
    test_set = set(fold_test_ids.tolist())

    train_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in train_set], dtype=np.intp
    )
    test_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )
    n_test_emails = len(test_col_idx)
    if n_test_emails == 0:
        return float("nan"), float("nan"), float("nan")

    test_email_ids_ordered = email_ids_all[test_col_idx]

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}
    eligible_user_ids = fold_features["user_id"].to_numpy()
    eligible_rows = np.array(
        [uid_to_row[int(uid)] for uid in eligible_user_ids if int(uid) in uid_to_row],
        dtype=np.intp,
    )
    if len(eligible_rows) == 0:
        return float("nan"), float("nan"), float("nan")

    eligible_row_to_local = {int(r): i for i, r in enumerate(eligible_rows)}

    M_train_filtered = M_full[eligible_rows, :][:, train_col_idx].tocsr()
    counts_h = np.asarray(M_train_filtered.sum(axis=1)).flatten().astype(np.float32)

    sendable_map = _sendable_set(wide, fold_test_ids)
    sendable_mask = np.zeros((len(eligible_rows), n_test_emails), dtype=bool)
    for e_idx, eid in enumerate(test_email_ids_ordered):
        sendable_uids = sendable_map.get(int(eid), np.array([], dtype=np.int64))
        local_indices = np.array(
            [
                eligible_row_to_local[uid_to_row[int(uid)]]
                for uid in sendable_uids
                if int(uid) in uid_to_row and uid_to_row[int(uid)] in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(local_indices) > 0:
            sendable_mask[local_indices, e_idx] = True

    M_test_full_rows = M_full[eligible_rows, :][:, test_col_idx].tocsr()

    fold_probe_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in range(n_seeds):
        probe_ids, holdout_ids, _ = _stratified_probe_partition(
            fold_features, probe_frac=0.10, seed=seed
        )
        fold_probe_cache[seed] = (probe_ids, holdout_ids)

    worker_args = []
    for s_idx, seed in enumerate(range(n_seeds)):
        probe_ids, holdout_ids = fold_probe_cache[seed]

        probe_rows_global = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )
        if len(probe_rows_global) == 0:
            continue

        probe_local_idx = np.array(
            [
                eligible_row_to_local[int(r)]
                for r in probe_rows_global
                if int(r) in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(probe_local_idx) == 0:
            continue

        holdout_ids_arr = np.asarray(holdout_ids)
        holdout_uids_in_eligible = np.isin(user_ids_all[eligible_rows], holdout_ids_arr)
        holdout_local_idx = np.where(holdout_uids_in_eligible)[0].astype(np.intp)
        if len(holdout_local_idx) == 0:
            continue

        M_probe_test = M_full[probe_rows_global, :][:, test_col_idx].tocsr()

        mt = M_train_filtered
        mpt = M_probe_test
        mtr = M_test_full_rows

        worker_args.append(
            (
                seed,
                s_idx,
                probe_ids,
                holdout_ids,
                mt.data,
                mt.indices,
                mt.indptr,
                mt.shape,
                mpt.data,
                mpt.indices,
                mpt.indptr,
                mpt.shape,
                mtr.data,
                mtr.indices,
                mtr.indptr,
                mtr.shape,
                probe_local_idx,
                eligible_rows,
                user_ids_all,
                sendable_mask,
                holdout_local_idx,
                counts_h,
                locked_X,
                min_signers,
                n_test_emails,
            )
        )

    if not worker_args:
        return float("nan"), float("nan"), float("nan")

    n_processes = min(max(1, (os.cpu_count() or 1) - 4), 24)
    with multiprocessing.Pool(processes=n_processes) as pool:
        fold_results = pool.map(_q10_fold_worker, worker_args)

    all_probe_rates: list[float] = []
    all_random_rates: list[float] = []
    for r in fold_results:
        all_probe_rates.extend(r["probe_rates"])
        all_random_rates.extend(r["random_rates"])

    if not all_random_rates:
        return float("nan"), float("nan"), float("nan")

    mean_random = float(np.nanmean(all_random_rates))
    mean_probe = float(np.nanmean(all_probe_rates))
    mean_lift = mean_probe / mean_random if mean_random > 0 else float("nan")
    return mean_lift, mean_random, mean_probe


def _section_q10_rolling_folds(
    wide: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
    attributed_df: pl.DataFrame,
    agg_v3_df: pl.DataFrame,
) -> Section:
    """Q10: temporal robustness — 3 rolling walk-forward folds."""
    import time

    t0 = time.time()
    locked_X = 0.25
    min_signers = 3
    n_seeds = 50
    delay_hours = 6.0

    hours_bins, cumulative = _hazard_curve(wide, attributed_df)

    def _f_at_delay(d: float) -> float:
        if d <= 0.0:
            return 0.0
        idx = int(np.searchsorted(hours_bins, d, side="right")) - 1
        idx = max(0, min(idx, len(cumulative) - 1))
        return float(cumulative[idx])

    f6 = _f_at_delay(delay_hours)
    delay_factor_6h = 1.0 - f6
    print(f"Q10: F(6h)={f6:.4f}  delay factor={delay_factor_6h:.4f}")

    folds = _chronological_email_split(wide, n_folds=3)
    print(f"Q10: running {len(folds)} rolling folds × {n_seeds} seeds each")

    sent = wide.select("email_id", "sent_at").unique("email_id").sort("sent_at")
    sent_at_lookup: dict[int, "datetime"] = {
        int(row["email_id"]): row["sent_at"] for row in sent.iter_rows(named=True)
    }

    fold_results_summary: list[dict] = []
    for fold_idx, (fold_train_ids, fold_test_ids) in enumerate(folds):
        train_min = min(
            (sent_at_lookup[int(eid)] for eid in fold_train_ids if int(eid) in sent_at_lookup),
            default=None,
        )
        train_max = max(
            (sent_at_lookup[int(eid)] for eid in fold_train_ids if int(eid) in sent_at_lookup),
            default=None,
        )
        test_min = min(
            (sent_at_lookup[int(eid)] for eid in fold_test_ids if int(eid) in sent_at_lookup),
            default=None,
        )
        test_max = max(
            (sent_at_lookup[int(eid)] for eid in fold_test_ids if int(eid) in sent_at_lookup),
            default=None,
        )

        train_window = (
            f"{train_min.strftime('%Y-%m') if train_min else '?'} – "
            f"{train_max.strftime('%Y-%m') if train_max else '?'}"
        )
        test_window = (
            f"{test_min.strftime('%Y-%m') if test_min else '?'} – "
            f"{test_max.strftime('%Y-%m') if test_max else '?'}"
        )

        print(
            f"  Fold {fold_idx + 1}/3: train={len(fold_train_ids)} emails  "
            f"test={len(fold_test_ids)} emails"
        )

        raw_lift, mean_random, mean_probe = _run_fold_lift(
            wide,
            fold_train_ids,
            fold_test_ids,
            M_full,
            email_ids_all,
            user_ids_all,
            locked_X=locked_X,
            min_signers=min_signers,
            n_seeds=n_seeds,
        )

        delay_adj_probe = mean_probe * delay_factor_6h if not np.isnan(mean_probe) else float("nan")
        delay_adj_lift = (
            delay_adj_probe / mean_random
            if (not np.isnan(delay_adj_probe) and mean_random > 0)
            else float("nan")
        )

        fold_results_summary.append(
            {
                "fold": fold_idx + 1,
                "train_window": train_window,
                "test_window": test_window,
                "raw_lift": raw_lift,
                "delay_adj_lift": delay_adj_lift,
                "mean_random": mean_random,
                "mean_probe": mean_probe,
            }
        )
        print(f"    raw_lift={raw_lift:.3f}x  delay_adj_lift(6h)={delay_adj_lift:.3f}x")

    elapsed = time.time() - t0
    print(f"Q10: folds done in {elapsed:.1f}s")

    raw_lifts = np.array(
        [r["raw_lift"] for r in fold_results_summary if not np.isnan(r["raw_lift"])]
    )
    adj_lifts = np.array(
        [r["delay_adj_lift"] for r in fold_results_summary if not np.isnan(r["delay_adj_lift"])]
    )

    raw_mean = float(np.mean(raw_lifts)) if len(raw_lifts) > 0 else float("nan")
    raw_sd = float(np.std(raw_lifts)) if len(raw_lifts) > 1 else float("nan")
    adj_mean = float(np.mean(adj_lifts)) if len(adj_lifts) > 0 else float("nan")
    adj_sd = float(np.std(adj_lifts)) if len(adj_lifts) > 1 else float("nan")
    fold_min_adj = float(np.min(adj_lifts)) if len(adj_lifts) > 0 else float("nan")

    print(
        f"\n=== Q10 HEADLINE ===\n"
        f"  Per-fold lifts (raw):          "
        + "  ".join(f"fold{r['fold']}={r['raw_lift']:.3f}x" for r in fold_results_summary)
        + f"\n  Per-fold lifts (delay-adj 6h): "
        + "  ".join(f"fold{r['fold']}={r['delay_adj_lift']:.3f}x" for r in fold_results_summary)
        + f"\n  Raw mean±SD:        {raw_mean:.3f}x ± {raw_sd:.3f}"
        + f"\n  Delay-adj mean±SD:  {adj_mean:.3f}x ± {adj_sd:.3f}"
        + f"\n  Delay-adj fold-min: {fold_min_adj:.3f}x (threshold ≥1.0x for Q11 gate)"
    )

    table_rows = ""
    for r in fold_results_summary:
        table_rows += (
            f"<tr><td>Fold {r['fold']}</td>"
            f"<td>{r['train_window']}</td>"
            f"<td>{r['test_window']}</td>"
            f"<td>{r['raw_lift']:.3f}x</td>"
            f"<td>{r['delay_adj_lift']:.3f}x</td></tr>"
        )
    table_rows += (
        f"<tr><td><b>Mean (SD)</b></td><td>—</td><td>—</td>"
        f"<td><b>{raw_mean:.3f}x ({raw_sd:.3f})</b></td>"
        f"<td><b>{adj_mean:.3f}x ({adj_sd:.3f})</b></td></tr>"
    )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr><th>Fold</th><th>Train window</th><th>Test window</th>"
        "<th>Probe-routed lift (Q6, X=25%)</th>"
        "<th>Delay-adjusted lift (6h)</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fold_nums = [r["fold"] for r in fold_results_summary]
    raw_lifts_list = [r["raw_lift"] for r in fold_results_summary]
    adj_lifts_list = [r["delay_adj_lift"] for r in fold_results_summary]

    ax.plot(
        fold_nums,
        raw_lifts_list,
        color="#3b82f6",
        marker="o",
        linewidth=2,
        label="Raw lift (Q6 X=25%)",
    )
    ax.plot(
        fold_nums,
        adj_lifts_list,
        color="#f97316",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Delay-adjusted lift (6h)",
    )
    ax.axhline(1.10, color="#ef4444", linestyle="--", linewidth=1.5, label="1.10x target")
    ax.axhline(1.00, color="#94a3b8", linestyle=":", linewidth=1.2, label="1.00x (no lift)")
    ax.set_xticks(fold_nums)
    ax.set_xticklabels([f"Fold {f}" for f in fold_nums])
    ax.set_ylabel("Relative lift vs random")
    ax.set_title("Q10. Temporal robustness — 3 rolling walk-forward folds")
    ax.legend(fontsize=9)
    ax.set_ylim(
        min(0.5, float(np.nanmin(adj_lifts_list)) - 0.1),
        max(2.5, float(np.nanmax(raw_lifts_list)) + 0.2),
    )
    fig.tight_layout()

    prose = (
        f"Temporal robustness evaluated over <b>{len(folds)} rolling walk-forward folds</b>. "
        f"Each fold: 12-month train window → 6-month test window, sliding 6 months. "
        f"Per-fold: user features rebuilt on fold-train emails; 50 probe seeds redrawn; "
        f"Q6 locked-X=25% metric computed. "
        f"Delay-adjusted lift applies the F(6h)={f6:.4f} hazard correction (factor {delay_factor_6h:.4f}). "
        f"<b>Raw mean±SD: {raw_mean:.3f}x ± {raw_sd:.3f}</b>. "
        f"<b>Delay-adj mean±SD: {adj_mean:.3f}x ± {adj_sd:.3f}</b>. "
        f"Fold-min delay-adj lift: <b>{fold_min_adj:.3f}x</b> "
        f"(Q11 go-criterion requires ≥1.0x on all folds). "
        f"Wall time: {elapsed:.0f}s. "
        f"{table_html}"
    )

    global _Q10_FOLD_MIN_ADJ_LIFT, _Q10_ADJ_MEAN
    _Q10_FOLD_MIN_ADJ_LIFT = fold_min_adj
    _Q10_ADJ_MEAN = adj_mean

    return Section(
        slug="q10_rolling_folds",
        heading="Q10. Temporal robustness — 3 rolling walk-forward folds",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


def _q11_seed_worker(args: tuple) -> dict:
    """Per-seed worker for Q11 bootstrap: 100 user-cluster resamples for this seed."""
    import os
    import time

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import numpy as np
    import scipy.sparse as sp

    (
        seed,
        M_train_filtered_data,
        M_train_filtered_indices,
        M_train_filtered_indptr,
        M_train_filtered_shape,
        M_probe_test_data,
        M_probe_test_indices,
        M_probe_test_indptr,
        M_probe_test_shape,
        M_test_rows_data,
        M_test_rows_indices,
        M_test_rows_indptr,
        M_test_rows_shape,
        probe_local_idx,
        holdout_local_idx,
        counts_h,
        sendable_mask,
        n_probe_signers_per_email,
        min_signers,
        n_test_emails,
        activity_scores,
        total_action_rate_arr,
        n_resamples,
        locked_X,
        delay_factor_6h,
    ) = args

    M_train_filtered = sp.csr_matrix(
        (M_train_filtered_data, M_train_filtered_indices, M_train_filtered_indptr),
        shape=M_train_filtered_shape,
    )
    M_probe_test = sp.csr_matrix(
        (M_probe_test_data, M_probe_test_indices, M_probe_test_indptr),
        shape=M_probe_test_shape,
    )
    M_test_rows = sp.csr_matrix(
        (M_test_rows_data, M_test_rows_indices, M_test_rows_indptr),
        shape=M_test_rows_shape,
    )

    M_probe_train = M_train_filtered[probe_local_idx, :]
    combined = (M_probe_train.T @ M_probe_test).astype(bool).astype(np.float32)
    if sp.issparse(combined):
        combined = np.asarray(combined.todense())
    combined_sum_per_email = combined.sum(axis=0).flatten()

    M_holdout_train = M_train_filtered[holdout_local_idx, :]
    intersect_full = M_holdout_train @ combined
    if sp.issparse(intersect_full):
        intersect_full = np.asarray(intersect_full.todense())

    counts_h_holdout = counts_h[holdout_local_idx]
    union_full = counts_h_holdout[:, None] + combined_sum_per_email[None, :] - intersect_full
    probe_scores = np.where(union_full > 0, intersect_full / union_full, 0.0).astype(np.float32)
    probe_scores[:, n_probe_signers_per_email < min_signers] = 0.0

    actioned_matrix = np.asarray(M_test_rows[holdout_local_idx, :].todense()).astype(np.float32)
    sendable_holdout_mask = sendable_mask[holdout_local_idx, :]

    activity_scores_h = activity_scores[holdout_local_idx]

    n_holdout = len(holdout_local_idx)
    rng = np.random.default_rng(seed * 10000 + 7)

    t0 = time.time()

    arm_names_inner = ["Random", "Activity-top", "Best-popularity", "Probe-routed"]
    seed_results: dict[str, list[float]] = {a: [] for a in arm_names_inner}
    seed_results_adj: dict[str, list[float]] = {a: [] for a in arm_names_inner}

    for resample_i in range(n_resamples):
        resample_idx = rng.integers(0, n_holdout, size=n_holdout)

        per_arm_rates: dict[str, list[float]] = {a: [] for a in arm_names_inner}

        for e_idx in range(n_test_emails):
            sendable_local = np.where(sendable_holdout_mask[:, e_idx])[0]
            if len(sendable_local) == 0:
                continue

            resampled_sendable = sendable_local[np.isin(sendable_local, resample_idx)]
            if len(resampled_sendable) < 2:
                resampled_sendable = sendable_local

            n_sendable = len(resampled_sendable)
            n_top = max(1, int(np.floor(n_sendable * locked_X)))

            actioned_h = actioned_matrix[resampled_sendable, e_idx]

            rand_idx = rng.choice(n_sendable, size=n_top, replace=False)
            per_arm_rates["Random"].append(float(actioned_h[rand_idx].mean()))

            act_sc = activity_scores_h[resampled_sendable]
            top_act = np.argpartition(-act_sc, min(n_top, n_sendable) - 1)[:n_top]
            per_arm_rates["Activity-top"].append(float(actioned_h[top_act].mean()))

            pop_sc = activity_scores_h[resampled_sendable] * float(total_action_rate_arr[e_idx])
            top_pop = np.argpartition(-pop_sc, min(n_top, n_sendable) - 1)[:n_top]
            per_arm_rates["Best-popularity"].append(float(actioned_h[top_pop].mean()))

            probe_sc = probe_scores[resampled_sendable, e_idx]
            top_probe = np.argpartition(-probe_sc, min(n_top, n_sendable) - 1)[:n_top]
            per_arm_rates["Probe-routed"].append(float(actioned_h[top_probe].mean()))

        for a in arm_names_inner:
            if per_arm_rates[a]:
                r = float(np.mean(per_arm_rates[a]))
                seed_results[a].append(r)
                if a == "Probe-routed":
                    seed_results_adj[a].append(r * delay_factor_6h)
                else:
                    seed_results_adj[a].append(r)
            else:
                seed_results[a].append(float("nan"))
                seed_results_adj[a].append(float("nan"))

    elapsed = time.time() - t0
    if elapsed > 10.0:
        print(f"  WARNING Q11: seed={seed} 100-resample worker took {elapsed:.1f}s > 10s threshold")

    return {"seed_results": seed_results, "seed_results_adj": seed_results_adj}


def _section_q11_net_lift_go_nogo(
    wide: pl.DataFrame,
    train_email_ids: np.ndarray,
    test_email_ids: np.ndarray,
    user_features_train: pl.DataFrame,
    agg_v3_df: pl.DataFrame,
    attributed_df: pl.DataFrame,
    M_full: sp.csr_matrix,
    email_ids_all: np.ndarray,
    user_ids_all: np.ndarray,
) -> Section:
    """Q11: net lift & final Go/No-Go — user-cluster bootstrap, both variance sources composed."""
    import multiprocessing
    import os
    import time

    t0 = time.time()
    locked_X = 0.25
    min_signers = 3
    n_probe_seeds = 50
    n_resamples = 100

    print("Q11: building hazard curve for delay factor")
    hours_bins, cumulative = _hazard_curve(wide, attributed_df)

    def _f_at_delay(d: float) -> float:
        if d <= 0.0:
            return 0.0
        idx = int(np.searchsorted(hours_bins, d, side="right")) - 1
        idx = max(0, min(idx, len(cumulative) - 1))
        return float(cumulative[idx])

    f6 = _f_at_delay(6.0)
    delay_factor_6h = 1.0 - f6
    print(f"Q11: F(6h)={f6:.4f}  delay factor={delay_factor_6h:.4f}")

    print("Q11: setting up matrices")
    train_set = set(train_email_ids.tolist())
    test_set = set(test_email_ids.tolist())

    train_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in train_set], dtype=np.intp
    )
    test_col_idx = np.array(
        [i for i, eid in enumerate(email_ids_all) if eid in test_set], dtype=np.intp
    )
    test_email_ids_ordered = email_ids_all[test_col_idx]
    n_test_emails = len(test_col_idx)

    uid_to_row = {int(uid): i for i, uid in enumerate(user_ids_all)}
    eligible_user_ids = user_features_train["user_id"].to_numpy()
    eligible_rows = np.array(
        [uid_to_row[int(uid)] for uid in eligible_user_ids if int(uid) in uid_to_row],
        dtype=np.intp,
    )
    eligible_row_to_local = {int(r): i for i, r in enumerate(eligible_rows)}

    M_train_filtered = M_full[eligible_rows, :][:, train_col_idx].tocsr()
    counts_h = np.asarray(M_train_filtered.sum(axis=1)).flatten().astype(np.float32)

    print(f"Q11: computing sendable mask for {n_test_emails} test emails")
    sendable_map = _sendable_set(wide, test_email_ids)
    sendable_mask = np.zeros((len(eligible_rows), n_test_emails), dtype=bool)
    for e_idx, eid in enumerate(test_email_ids_ordered):
        sendable_uids = sendable_map.get(int(eid), np.array([], dtype=np.int64))
        local_indices = np.array(
            [
                eligible_row_to_local[uid_to_row[int(uid)]]
                for uid in sendable_uids
                if int(uid) in uid_to_row and uid_to_row[int(uid)] in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(local_indices) > 0:
            sendable_mask[local_indices, e_idx] = True

    M_test_full_rows = M_full[eligible_rows, :][:, test_col_idx].tocsr()

    activity_scores = user_features_train["action_rate_train_eb"].to_numpy().astype(np.float32)

    agg_lookup = {
        int(row["email_id"]): (int(row["total_actions"]), int(row["total_sends"]))
        for row in agg_v3_df.iter_rows(named=True)
    }
    total_action_rate_arr = np.array(
        [
            agg_lookup.get(int(eid), (0, 1))[0] / max(1, agg_lookup.get(int(eid), (0, 1))[1])
            for eid in test_email_ids_ordered
        ],
        dtype=np.float32,
    )
    print(
        f"Q11: dispatching {n_probe_seeds} seeds × {n_resamples} resamples "
        f"= {n_probe_seeds * n_resamples} total bootstrap estimates"
    )

    worker_args = []
    seeds_sorted = sorted(_PROBE_SEED_CACHE.keys())[:n_probe_seeds]

    for seed in seeds_sorted:
        probe_ids, holdout_ids = _PROBE_SEED_CACHE[seed]

        probe_rows_global = np.array(
            [uid_to_row[int(uid)] for uid in probe_ids if int(uid) in uid_to_row],
            dtype=np.intp,
        )
        if len(probe_rows_global) == 0:
            continue

        probe_local_idx = np.array(
            [
                eligible_row_to_local[int(r)]
                for r in probe_rows_global
                if int(r) in eligible_row_to_local
            ],
            dtype=np.intp,
        )
        if len(probe_local_idx) == 0:
            continue

        holdout_ids_arr = np.asarray(holdout_ids)
        holdout_uids_in_eligible = np.isin(user_ids_all[eligible_rows], holdout_ids_arr)
        holdout_local_idx = np.where(holdout_uids_in_eligible)[0].astype(np.intp)
        if len(holdout_local_idx) == 0:
            continue

        M_probe_test = M_full[probe_rows_global, :][:, test_col_idx].tocsr()
        n_probe_signers_per_email = np.asarray(M_probe_test.sum(axis=0)).flatten()

        mt = M_train_filtered
        mpt = M_probe_test
        mtr = M_test_full_rows

        worker_args.append(
            (
                seed,
                mt.data,
                mt.indices,
                mt.indptr,
                mt.shape,
                mpt.data,
                mpt.indices,
                mpt.indptr,
                mpt.shape,
                mtr.data,
                mtr.indices,
                mtr.indptr,
                mtr.shape,
                probe_local_idx,
                holdout_local_idx,
                counts_h,
                sendable_mask,
                n_probe_signers_per_email,
                min_signers,
                n_test_emails,
                activity_scores,
                total_action_rate_arr,
                n_resamples,
                locked_X,
                delay_factor_6h,
            )
        )

    n_processes = min(max(1, (os.cpu_count() or 1) - 4), 24)
    print(f"Q11: running {len(worker_args)} seed workers across {n_processes} processes")
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(_q11_seed_worker, worker_args)

    arm_names_outer = ["Random", "Activity-top", "Best-popularity", "Probe-routed"]
    all_raw: dict[str, list[float]] = {a: [] for a in arm_names_outer}
    all_adj: dict[str, list[float]] = {a: [] for a in arm_names_outer}
    for r in results:
        for a in arm_names_outer:
            all_raw[a].extend(r["seed_results"].get(a, []))
            all_adj[a].extend(r["seed_results_adj"].get(a, []))

    elapsed = time.time() - t0
    print(
        f"Q11: bootstrap done in {elapsed:.1f}s  ({len(worker_args)} seeds × {n_resamples} resamples)"
    )

    overall_rate = float(
        wide.filter(pl.col("email_id").is_in(list(test_set)))["actioned"].mean() or 0.0
    )

    display_names = {
        "Random": "Random top-25%",
        "Activity-top": "Activity-top top-25%",
        "Best-popularity": "Best-popularity top-25%",
        "Probe-routed": "Probe-routed top-25%",
    }

    summary: dict[str, dict] = {}
    random_raw_arr = np.array(all_raw["Random"])
    random_mean = float(np.nanmean(random_raw_arr)) if len(random_raw_arr) > 0 else overall_rate

    for a in arm_names_outer:
        raw_arr = np.array([x for x in all_raw[a] if not np.isnan(x)])
        adj_arr = np.array([x for x in all_adj[a] if not np.isnan(x)])
        if len(raw_arr) == 0:
            summary[a] = {
                "raw_rate": float("nan"),
                "raw_lift": float("nan"),
                "raw_ci_lo": float("nan"),
                "raw_ci_hi": float("nan"),
                "adj_rate": float("nan"),
                "adj_lift": float("nan"),
                "adj_ci_lo": float("nan"),
                "adj_ci_hi": float("nan"),
            }
            continue

        raw_rate = float(np.nanmean(raw_arr))
        raw_lift = raw_rate / random_mean if random_mean > 0 else float("nan")
        raw_lifts_arr = raw_arr / random_mean if random_mean > 0 else raw_arr
        raw_ci_lo = float(np.percentile(raw_lifts_arr, 2.5))
        raw_ci_hi = float(np.percentile(raw_lifts_arr, 97.5))

        adj_rate = float(np.nanmean(adj_arr)) if len(adj_arr) > 0 else float("nan")
        adj_lifts_arr = adj_arr / random_mean if random_mean > 0 else adj_arr
        adj_lift = float(np.nanmean(adj_lifts_arr)) if len(adj_lifts_arr) > 0 else float("nan")
        adj_ci_lo = (
            float(np.percentile(adj_lifts_arr, 2.5)) if len(adj_lifts_arr) > 0 else float("nan")
        )
        adj_ci_hi = (
            float(np.percentile(adj_lifts_arr, 97.5)) if len(adj_lifts_arr) > 0 else float("nan")
        )

        summary[a] = {
            "raw_rate": raw_rate,
            "raw_lift": raw_lift,
            "raw_ci_lo": raw_ci_lo,
            "raw_ci_hi": raw_ci_hi,
            "adj_rate": adj_rate,
            "adj_lift": adj_lift,
            "adj_ci_lo": adj_ci_lo,
            "adj_ci_hi": adj_ci_hi,
        }

    probe_adj_ci_lo = summary.get("Probe-routed", {}).get("adj_ci_lo", float("nan"))
    probe_adj_lift = summary.get("Probe-routed", {}).get("adj_lift", float("nan"))

    q10_fold_min_ok = (not np.isnan(_Q10_FOLD_MIN_ADJ_LIFT)) and (_Q10_FOLD_MIN_ADJ_LIFT >= 1.0)
    q11_probe_ok = (not np.isnan(probe_adj_ci_lo)) and (probe_adj_ci_lo >= 1.10)

    _dnh_val = do_no_harm_violations
    q8_no_harm_ok = len(_dnh_val) == 0

    is_go = q11_probe_ok and q8_no_harm_ok and q10_fold_min_ok

    verdict = "GO" if is_go else "NO-GO"
    print(
        f"\n=== Q11 FINAL VERDICT: {verdict} ===\n"
        f"  Criterion 1 — probe delay-adj CI lower bound ≥ 1.10x: "
        f"{'PASS' if q11_probe_ok else 'FAIL'} "
        f"(CI lo = {probe_adj_ci_lo:.3f}x, adj lift = {probe_adj_lift:.3f}x)\n"
        f"  Criterion 2 — Q8 zero do-no-harm violations:           "
        f"{'PASS' if q8_no_harm_ok else 'FAIL'} ({len(_dnh_val)} violations)\n"
        f"  Criterion 3 — Q10 delay-adj fold-min ≥ 1.0x:          "
        f"{'PASS' if q10_fold_min_ok else 'FAIL'} (fold-min = {_Q10_FOLD_MIN_ADJ_LIFT:.3f}x)\n"
        f"  → Final decision: {verdict}"
    )

    table_rows = (
        "<tr><td>Status quo (full blast)</td>"
        f"<td>{overall_rate:.3%}</td>"
        "<td>1.000x</td><td>—</td>"
        f"<td>{overall_rate * delay_factor_6h:.3%}</td>"
        "<td>1.000x</td><td>—</td></tr>"
    )
    for a in arm_names_outer:
        s = summary.get(a, {})
        raw_rate = s.get("raw_rate", float("nan"))
        raw_lift = s.get("raw_lift", float("nan"))
        raw_ci = f"({s.get('raw_ci_lo', float('nan')):.3f}, {s.get('raw_ci_hi', float('nan')):.3f})"
        adj_rate = s.get("adj_rate", float("nan"))
        adj_lift = s.get("adj_lift", float("nan"))
        adj_ci = f"({s.get('adj_ci_lo', float('nan')):.3f}, {s.get('adj_ci_hi', float('nan')):.3f})"
        table_rows += (
            f"<tr><td>{display_names.get(a, a)}</td>"
            f"<td>{raw_rate:.3%}</td>"
            f"<td>{raw_lift:.3f}x</td>"
            f"<td>{raw_ci}</td>"
            f"<td>{adj_rate:.3%}</td>"
            f"<td>{adj_lift:.3f}x</td>"
            f"<td>{adj_ci}</td></tr>"
        )

    table_html = (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        "<thead><tr>"
        "<th>Strategy</th>"
        "<th>Action rate (raw)</th><th>Lift (raw)</th><th>95% CI (cluster+seed)</th>"
        "<th>Action rate (delay-adj 6h)</th><th>Lift (delay-adj)</th><th>95% CI</th>"
        "</tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
    )

    all_strategies = ["Random", "Activity-top", "Best-popularity", "Probe-routed"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for panel_idx, (panel_ax, is_adj) in enumerate(zip(axes, [False, True])):
        panel_title = "Raw replay" if not is_adj else "Delay-adjusted (6h)"
        y_pos = np.arange(len(all_strategies))

        for idx, a in enumerate(all_strategies):
            s = summary.get(a, {})
            lift = s.get("adj_lift" if is_adj else "raw_lift", float("nan"))
            ci_lo = s.get("adj_ci_lo" if is_adj else "raw_ci_lo", float("nan"))
            ci_hi = s.get("adj_ci_hi" if is_adj else "raw_ci_hi", float("nan"))

            if np.isnan(lift):
                continue

            is_probe = a == "Probe-routed"
            above_target = (not np.isnan(ci_lo)) and (ci_lo >= 1.10)
            color = (
                "#22c55e"
                if (above_target and is_adj)
                else ("#86efac" if above_target else ("#94a3b8" if not is_probe else "#60a5fa"))
            )
            lw = 2.5 if is_probe else 1.0

            panel_ax.barh(
                idx,
                lift,
                color=color,
                edgecolor="black" if is_probe else "white",
                linewidth=lw,
                height=0.6,
            )
            if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                panel_ax.errorbar(
                    lift,
                    idx,
                    xerr=[[lift - ci_lo], [ci_hi - lift]],
                    fmt="none",
                    color="#0f172a",
                    capsize=5,
                    linewidth=1.5,
                )

        panel_ax.axvline(1.00, color="#94a3b8", linestyle=":", linewidth=1.2, label="1.00x")
        panel_ax.axvline(1.10, color="#ef4444", linestyle="--", linewidth=1.5, label="1.10x target")
        panel_ax.set_yticks(y_pos)
        panel_ax.set_yticklabels([display_names.get(a, a) for a in all_strategies], fontsize=9)
        panel_ax.set_xlabel("Relative lift vs random")
        panel_ax.set_title(f"({chr(97 + panel_idx)}) {panel_title}")
        panel_ax.legend(fontsize=8)

    fig.suptitle(
        f"Q11. Net lift forest plot — Final verdict: {verdict}\n"
        f"(Green bar = delay-adj CI lo ≥ 1.10x)",
        fontsize=13,
    )
    fig.tight_layout()

    criteria_prose = (
        f"<b>Go criterion (3 sub-criteria, all must pass):</b><br/>"
        f"1. Probe delay-adjusted CI lower bound ≥ 1.10x: "
        f"<b style='color:{'green' if q11_probe_ok else 'red'}'>{'PASS' if q11_probe_ok else 'FAIL'}</b> "
        f"(adj lift = {probe_adj_lift:.3f}x, CI lo = {probe_adj_ci_lo:.3f}x)<br/>"
        f"2. Q8 zero do-no-harm violations: "
        f"<b style='color:{'green' if q8_no_harm_ok else 'red'}'>{'PASS' if q8_no_harm_ok else 'FAIL'}</b> "
        f"({len(_dnh_val)} violations)<br/>"
        f"3. Q10 delay-adj fold-min ≥ 1.0x: "
        f"<b style='color:{'green' if q10_fold_min_ok else 'red'}'>{'PASS' if q10_fold_min_ok else 'FAIL'}</b> "
        f"(fold-min = {_Q10_FOLD_MIN_ADJ_LIFT:.3f}x)<br/>"
        f"<b style='font-size:1.2em; color:{'#16a34a' if is_go else '#dc2626'}'>Final decision: {verdict}</b>"
    )

    prose = (
        f"Final Go/No-Go. Bootstrap: {len(worker_args)} probe seeds × {n_resamples} user-cluster "
        f"resamples = {len(worker_args) * n_resamples} lift estimates per strategy. "
        f"Delay-adjustment: probe-routed arm multiplied by (1−F(6h)) = {delay_factor_6h:.4f}; "
        f"non-probe arms unchanged. "
        f"Bootstrap composes probe-seed variance with user-cluster sampling variance. "
        f"Wall time: {elapsed:.0f}s. "
        f"{criteria_prose} "
        f"{table_html}"
    )

    return Section(
        slug="q11_go_nogo",
        heading="Q11. Net lift & final Go/No-Go",
        prose=prose,
        png_b64=_fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Section builders list (populated by subsequent stages).
# ---------------------------------------------------------------------------

SECTIONS: list[Callable] = []


# ---------------------------------------------------------------------------
# HTML rendering.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Civic Shout content routing -- probe-routing exploration</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 980px;
         margin: 32px auto; padding: 0 16px; color: #0f172a; line-height: 1.5; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  .meta {{ color: #64748b; font-size: 13px; margin-bottom: 32px; }}
  .toc {{ background: #f1f5f9; border-radius: 8px; padding: 14px 18px;
          margin-bottom: 28px; }}
  .toc ul {{ margin: 4px 0 0 0; padding-left: 22px; }}
  section {{ margin-bottom: 44px; padding-bottom: 24px;
             border-bottom: 1px solid #e2e8f0; }}
  section:last-child {{ border-bottom: 0; }}
  h2 {{ font-size: 19px; color: #0f172a; }}
  .prose {{ color: #1e293b; }}
  .prose code {{ background: #e2e8f0; padding: 1px 5px; border-radius: 3px;
                 font-size: 0.92em; }}
  img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px;
         margin-top: 12px; }}
</style>
</head>
<body>
<h1>Civic Shout content routing -- probe-routing exploration</h1>
<div class="meta">
  Source: <code>{src}</code> &middot; Rows: {rows} &middot; Users: {users}
  &middot; Emails: {emails} &middot; Anchor: 2026-04-21 UTC
</div>
<div class="toc">
  <b>Sections:</b>
  <ul>{toc}</ul>
</div>
{body}
</body>
</html>
"""


def render_html(sections: list[Section], src: Path, rows: int, users: int, emails: int) -> str:
    toc = "".join(f'<li><a href="#{s.slug}">{html.escape(s.heading)}</a></li>' for s in sections)
    body_parts = []
    for s in sections:
        body_parts.append(
            f'<section id="{s.slug}">'
            f"<h2>{html.escape(s.heading)}</h2>"
            f'<div class="prose">{s.prose}</div>'
            f'<img alt="{html.escape(s.heading)}" '
            f'src="data:image/png;base64,{s.png_b64}">'
            f"</section>"
        )
    return _HTML_TEMPLATE.format(
        src=html.escape(str(src)),
        rows=f"{rows:,}",
        users=f"{users:,}",
        emails=f"{emails:,}",
        toc=toc,
        body="\n".join(body_parts),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in",
        dest="src",
        type=Path,
        default=Path(
            "data/projects/civic_shout_content_routing/exploration/wide_engagement.parquet"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("projects/civic_shout_content_routing/exploration/probe_routing.html"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"loading {args.src}")
    wide = pl.read_parquet(args.src)
    n_rows = wide.height
    n_emails = wide["email_id"].n_unique()
    n_users = wide["user_id"].n_unique()
    print(f"  rows={n_rows:,}  emails={n_emails:,}  users={n_users:,}")

    print("computing primary chronological split")
    train_email_ids, test_email_ids = _primary_split(wide)
    print(f"  train={len(train_email_ids):,} emails  test={len(test_email_ids):,} emails")

    assert len(set(train_email_ids.tolist()) & set(test_email_ids.tolist())) == 0, (
        "Leakage: train and test email sets overlap"
    )
    print("  leakage assertion passed: train/test email sets are disjoint")

    print("building action matrix")
    M, email_ids, user_ids = build_action_matrix(wide)
    print(f"  M shape={M.shape}  nnz={M.nnz:,}")

    print("computing user_features_train (train emails only, n_actioned>=2, n_sent>=5)")
    user_features_train = _build_user_features_train(wide, train_email_ids)
    print(f"  eligible users={len(user_features_train):,}")

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush(sections: list[Section]) -> None:
        html_str = render_html(sections, args.src, n_rows, n_users, n_emails)
        out_path.write_text(html_str, encoding="utf-8")
        print(f"flushed {len(sections)} sections to {out_path}")

    sections: list[Section] = []

    print("  -> _section_q1_baseline")
    sections.append(_section_q1_baseline(wide, user_features_train, test_email_ids))
    _flush(sections)

    print("  -> _section_q2_partition (50 seeds)")
    sections.append(_section_q2_partition(user_features_train, n_seeds=50))
    _flush(sections)

    print("  -> _section_q3_coverage")
    sections.append(
        _section_q3_coverage(wide, test_email_ids, user_features_train, M, email_ids, user_ids)
    )
    _flush(sections)

    print("loading agg_v3 for test emails")
    agg_v3_df = _load_agg_v3(test_email_ids)
    print(f"  agg_v3 rows={agg_v3_df.height:,}")

    print("  -> _section_q4_popularity_head_to_head")
    sections.append(
        _section_q4_popularity_head_to_head(wide, test_email_ids, M, email_ids, user_ids, agg_v3_df)
    )
    _flush(sections)

    print("  -> _section_q5_user_jaccard_smoke")
    sections.append(
        _section_q5_user_jaccard_smoke(
            wide, train_email_ids, user_features_train, M, email_ids, user_ids
        )
    )
    _flush(sections)

    print("  -> _section_q6_probe_routing_lift")
    sections.append(
        _section_q6_probe_routing_lift(
            wide, train_email_ids, test_email_ids, user_features_train, M, email_ids, user_ids
        )
    )
    _flush(sections)

    print("  -> _section_q7_activity_control")
    sections.append(
        _section_q7_activity_control(
            wide,
            train_email_ids,
            test_email_ids,
            user_features_train,
            agg_v3_df,
            M,
            email_ids,
            user_ids,
        )
    )
    _flush(sections)

    print("  -> _section_q8_joint_stratification")
    sections.append(
        _section_q8_joint_stratification(
            wide,
            train_email_ids,
            test_email_ids,
            user_features_train,
            agg_v3_df,
            M,
            email_ids,
            user_ids,
        )
    )
    _flush(sections)

    print("loading attributed_df (v2) for Q9")
    attributed_df = attr_cache.get(2)
    print(f"  attributed_df rows={attributed_df.height:,}")

    print("  -> _section_q9_delay_confounding")
    sections.append(
        _section_q9_delay_confounding(
            wide,
            test_email_ids,
            attributed_df,
            user_features_train,
            M,
            email_ids,
            user_ids,
        )
    )
    _flush(sections)

    print("  -> _section_q10_rolling_folds")
    sections.append(
        _section_q10_rolling_folds(
            wide,
            M,
            email_ids,
            user_ids,
            attributed_df,
            agg_v3_df,
        )
    )
    _flush(sections)

    print("  -> _section_q11_net_lift_go_nogo")
    sections.append(
        _section_q11_net_lift_go_nogo(
            wide,
            train_email_ids,
            test_email_ids,
            user_features_train,
            agg_v3_df,
            attributed_df,
            M,
            email_ids,
            user_ids,
        )
    )
    _flush(sections)

    for builder in SECTIONS:
        print(f"  -> {builder.__name__}")
        sections.append(builder(wide, train_email_ids, test_email_ids))
        _flush(sections)

    print(f"wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
