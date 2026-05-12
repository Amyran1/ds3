"""F06. Recency-weighted RFM baseline + send-frequency heterogeneity.

Baseline AUC of a model using ONLY prior-window engagement features:
action/click/open rates in 7d/30d/90d windows, days-since-last-*, sends-last-30d,
log1p(prior_action_count). Temporal split (80th pct of date_sent). Per-tenure-bucket AUC.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F06_recency_rfm_baseline
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from entities.civic_shout_engagement.email_activities_cache import cache as ea_cache
from projects.civic_shout_action_rate_increase.data import user_emails
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_JSONL = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)

_TENURE_BREAKS = [7, 30, 90, 180, 365, 730]
_TENURE_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr", "2yr+"]

_TIERS = [
    ("1pct", 0.01),
    ("5pct", 0.05),
    ("full", None),
]

_FEATURE_COLS = [
    "action_rate_7d",
    "action_rate_30d",
    "action_rate_90d",
    "open_rate_7d",
    "open_rate_30d",
    "open_rate_90d",
    "click_rate_7d",
    "click_rate_30d",
    "click_rate_90d",
    "days_since_last_action",
    "days_since_last_open",
    "days_since_last_click",
    "sends_last_30d",
    "log1p_prior_action_count",
]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/Users/aaronmyran/dev/ds3",
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _peak_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _load_ea_events() -> pl.DataFrame:
    """Load raw open/click events from email_activities v2 with timestamps."""
    return (
        ea_cache.scan(2)
        .filter(pl.col("action_type").is_in(["open", "click"]))
        .select(["user_id", "action_type", "created_at"])
        .collect()
    )


def _build_daily_cum(ea_events: pl.DataFrame, action_type: str, cum_col: str) -> pl.DataFrame:
    """Build daily cumulative counts of open or click events per user.

    Returns DataFrame sorted by (user_id, event_date) with columns:
        user_id, event_date, cum_col (cumulative count up to and including event_date).
    """
    return (
        ea_events.filter(pl.col("action_type") == action_type)
        .with_columns(pl.col("created_at").dt.date().alias("event_date"))
        .group_by(["user_id", "event_date"])
        .agg(pl.len().alias("n"))
        .sort(["user_id", "event_date"])
        .with_columns(pl.col("n").cum_sum().over("user_id").alias(cum_col))
        .drop("n")
    )


def _build_action_cum(ue_sorted: pl.DataFrame) -> pl.DataFrame:
    """Cumulative action and send counts per user from user_emails (sorted by user, date).

    Returns same-length DataFrame (indexed with _row_idx) with:
        _row_idx, user_id, date_sent, cum_actions_excl (excl. current row),
        cum_sends_excl.
    """
    return ue_sorted.with_row_index("_row_idx").with_columns(
        pl.col("actioned")
        .cast(pl.Int32)
        .cum_sum()
        .over("user_id")
        .shift(1)
        .fill_null(0)
        .alias("cum_actions_excl"),
        pl.lit(1).cum_sum().over("user_id").shift(1).fill_null(0).alias("cum_sends_excl"),
    )


def _asof_lookup(
    ue_df: pl.DataFrame,
    cum_df: pl.DataFrame,
    target_date_col: str,
    cum_col: str,
    out_col: str,
) -> pl.Series:
    """For each row in ue_df, find the cumulative count in cum_df at target_date_col.

    Uses join_asof (backward strategy) on (user_id, date). Returns a Series of
    length len(ue_df) in the same row order (sorted by user_id, target_date).

    ue_df must have: _row_idx, user_id, target_date_col (Date).
    cum_df must have: user_id, event_date (Date), cum_col.
    """
    tmp = ue_df.select(["_row_idx", "user_id", target_date_col]).sort(["user_id", target_date_col])
    joined = tmp.join_asof(
        cum_df.select(["user_id", "event_date", cum_col]).sort(["user_id", "event_date"]),
        left_on=target_date_col,
        right_on="event_date",
        by="user_id",
        strategy="backward",
    ).fill_null(0)
    # Restore original row order via _row_idx
    return joined.sort("_row_idx")[cum_col].rename(out_col)


def _build_features(sample: pl.DataFrame, ea_events: pl.DataFrame) -> pl.DataFrame:
    """Build prior-window RFM features for each (user_id, date_sent) row.

    Action features: from user_emails.actioned (uncontaminated per F03).
    Open/click features: from raw email_activities events (per F03 rule).

    Window computation uses cumulative-sum + asof-join approach:
    - sort rows by (user_id, date_sent)
    - build daily cumulative event counts
    - for each row, subtract cum_at(date_sent - W - 1) from cum_at(date_sent - 1)
    """
    # Step 1: sort and build cumulative action/send columns
    # Cast date_sent to Date (user_emails v2 stores it as datetime[us, UTC])
    ue_sorted = (
        sample.select(["user_id", "email_id", "date_sent", "actioned"])
        .with_columns(pl.col("date_sent").cast(pl.Date).alias("date_sent"))
        .sort(["user_id", "date_sent"])
    )

    act = _build_action_cum(ue_sorted)

    # First send date per user (for tenure)
    first_send = act.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))
    act = act.join(first_send, on="user_id")

    # Tenure
    act = act.with_columns(
        ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days()).alias("tenure_days")
    ).with_columns(
        pl.col("tenure_days").cut(_TENURE_BREAKS, labels=_TENURE_LABELS).alias("tenure_bucket")
    )

    # --- Action cumulative windows -----------------------------------------------
    # Daily cumulative actioned counts from user_emails (since we have actioned flag per row,
    # we build a per-user-date cumulative from act itself rather than a separate groupby)
    act_daily = (
        act.group_by(["user_id", "date_sent"])
        .agg(pl.col("actioned").cast(pl.Int32).sum().alias("n_act"), pl.len().alias("n_sends"))
        .sort(["user_id", "date_sent"])
        .with_columns(
            pl.col("n_act").cum_sum().over("user_id").alias("cum_act"),
            pl.col("n_sends").cum_sum().over("user_id").alias("cum_sends"),
        )
        .rename({"date_sent": "event_date"})
    )

    # For each row compute cum_act / cum_sends at:
    #   date_sent - 1 day (exclusive of current send day)
    #   date_sent - 7 days  (exclusive boundary: date_sent - 8 days)
    #   date_sent - 30 days (exclusive: date_sent - 31 days)
    #   date_sent - 90 days (exclusive: date_sent - 91 days)
    # Window count = cum_at(ds-1) - cum_at(ds - W - 1)

    def _date_offset_col(df: pl.DataFrame, offset_days: int, col_name: str) -> pl.DataFrame:
        """Add a date column = date_sent - offset_days."""
        return df.with_columns(
            (pl.col("date_sent") - pl.lit(timedelta(days=offset_days))).alias(col_name)
        )

    act2 = act.clone()
    act2 = _date_offset_col(act2, 1, "_ds_minus1")
    act2 = _date_offset_col(act2, 8, "_ds_minus8")
    act2 = _date_offset_col(act2, 31, "_ds_minus31")
    act2 = _date_offset_col(act2, 91, "_ds_minus91")

    # asof lookups for action cumulative
    cum_act_at_ds1 = _asof_lookup(act2, act_daily, "_ds_minus1", "cum_act", "cum_act_ds1")
    cum_act_at_ds8 = _asof_lookup(act2, act_daily, "_ds_minus8", "cum_act", "cum_act_ds8")
    cum_act_at_ds31 = _asof_lookup(act2, act_daily, "_ds_minus31", "cum_act", "cum_act_ds31")
    cum_act_at_ds91 = _asof_lookup(act2, act_daily, "_ds_minus91", "cum_act", "cum_act_ds91")

    cum_sends_at_ds1 = _asof_lookup(act2, act_daily, "_ds_minus1", "cum_sends", "cum_sends_ds1")
    cum_sends_at_ds8 = _asof_lookup(act2, act_daily, "_ds_minus8", "cum_sends", "cum_sends_ds8")
    cum_sends_at_ds31 = _asof_lookup(act2, act_daily, "_ds_minus31", "cum_sends", "cum_sends_ds31")
    cum_sends_at_ds91 = _asof_lookup(act2, act_daily, "_ds_minus91", "cum_sends", "cum_sends_ds91")

    act2 = act2.with_columns(
        cum_act_at_ds1,
        cum_act_at_ds8,
        cum_act_at_ds31,
        cum_act_at_ds91,
        cum_sends_at_ds1,
        cum_sends_at_ds8,
        cum_sends_at_ds31,
        cum_sends_at_ds91,
    )

    # Window counts
    act2 = act2.with_columns(
        (pl.col("cum_act_ds1") - pl.col("cum_act_ds8")).clip(0).alias("_act_7d"),
        (pl.col("cum_act_ds1") - pl.col("cum_act_ds31")).clip(0).alias("_act_30d"),
        (pl.col("cum_act_ds1") - pl.col("cum_act_ds91")).clip(0).alias("_act_90d"),
        (pl.col("cum_sends_ds1") - pl.col("cum_sends_ds8")).clip(0).alias("_sends_7d"),
        (pl.col("cum_sends_ds1") - pl.col("cum_sends_ds31")).clip(0).alias("_sends_30d"),
        (pl.col("cum_sends_ds1") - pl.col("cum_sends_ds91")).clip(0).alias("_sends_90d"),
    )

    # Action rates
    act2 = act2.with_columns(
        (pl.col("_act_7d") / (pl.col("_sends_7d") + 1e-9)).alias("action_rate_7d"),
        (pl.col("_act_30d") / (pl.col("_sends_30d") + 1e-9)).alias("action_rate_30d"),
        (pl.col("_act_90d") / (pl.col("_sends_90d") + 1e-9)).alias("action_rate_90d"),
        pl.col("_sends_30d").alias("sends_last_30d"),
        pl.col("cum_actions_excl").log1p().alias("log1p_prior_action_count"),
    )

    # Days since last action (forward-fill within user)
    act2 = act2.with_columns(
        pl.when(pl.col("actioned"))
        .then(pl.col("date_sent"))
        .otherwise(None)
        .shift(1)
        .forward_fill()
        .over("user_id")
        .alias("_last_action_date")
    ).with_columns(
        pl.when(pl.col("_last_action_date").is_not_null())
        .then((pl.col("date_sent") - pl.col("_last_action_date")).dt.total_days())
        .otherwise(999.0)
        .alias("days_since_last_action")
    )

    # --- Open/click cumulative windows from email_activities --------------------
    opens_cum = _build_daily_cum(ea_events, "open", "cum_opens")
    clicks_cum = _build_daily_cum(ea_events, "click", "cum_clicks")

    cum_opens_ds1 = _asof_lookup(act2, opens_cum, "_ds_minus1", "cum_opens", "cum_opens_ds1")
    cum_opens_ds8 = _asof_lookup(act2, opens_cum, "_ds_minus8", "cum_opens", "cum_opens_ds8")
    cum_opens_ds31 = _asof_lookup(act2, opens_cum, "_ds_minus31", "cum_opens", "cum_opens_ds31")
    cum_opens_ds91 = _asof_lookup(act2, opens_cum, "_ds_minus91", "cum_opens", "cum_opens_ds91")

    cum_clicks_ds1 = _asof_lookup(act2, clicks_cum, "_ds_minus1", "cum_clicks", "cum_clicks_ds1")
    cum_clicks_ds8 = _asof_lookup(act2, clicks_cum, "_ds_minus8", "cum_clicks", "cum_clicks_ds8")
    cum_clicks_ds31 = _asof_lookup(act2, clicks_cum, "_ds_minus31", "cum_clicks", "cum_clicks_ds31")
    cum_clicks_ds91 = _asof_lookup(act2, clicks_cum, "_ds_minus91", "cum_clicks", "cum_clicks_ds91")

    act2 = act2.with_columns(
        cum_opens_ds1,
        cum_opens_ds8,
        cum_opens_ds31,
        cum_opens_ds91,
        cum_clicks_ds1,
        cum_clicks_ds8,
        cum_clicks_ds31,
        cum_clicks_ds91,
    )

    # Open/click window counts and rates
    act2 = act2.with_columns(
        (pl.col("cum_opens_ds1") - pl.col("cum_opens_ds8")).clip(0).alias("_opens_7d"),
        (pl.col("cum_opens_ds1") - pl.col("cum_opens_ds31")).clip(0).alias("_opens_30d"),
        (pl.col("cum_opens_ds1") - pl.col("cum_opens_ds91")).clip(0).alias("_opens_90d"),
        (pl.col("cum_clicks_ds1") - pl.col("cum_clicks_ds8")).clip(0).alias("_clicks_7d"),
        (pl.col("cum_clicks_ds1") - pl.col("cum_clicks_ds31")).clip(0).alias("_clicks_30d"),
        (pl.col("cum_clicks_ds1") - pl.col("cum_clicks_ds91")).clip(0).alias("_clicks_90d"),
    ).with_columns(
        (pl.col("_opens_7d") / (pl.col("_sends_7d") + 1e-9)).alias("open_rate_7d"),
        (pl.col("_opens_30d") / (pl.col("_sends_30d") + 1e-9)).alias("open_rate_30d"),
        (pl.col("_opens_90d") / (pl.col("_sends_90d") + 1e-9)).alias("open_rate_90d"),
        (pl.col("_clicks_7d") / (pl.col("_sends_7d") + 1e-9)).alias("click_rate_7d"),
        (pl.col("_clicks_30d") / (pl.col("_sends_30d") + 1e-9)).alias("click_rate_30d"),
        (pl.col("_clicks_90d") / (pl.col("_sends_90d") + 1e-9)).alias("click_rate_90d"),
    )

    # Days since last open/click (asof join on ea event dates)
    ea_last_open = (
        ea_events.filter(pl.col("action_type") == "open")
        .with_columns(pl.col("created_at").dt.date().alias("event_date"))
        .group_by(["user_id", "event_date"])
        .agg(pl.len().alias("_n"))
        .sort(["user_id", "event_date"])
    )
    ea_last_click = (
        ea_events.filter(pl.col("action_type") == "click")
        .with_columns(pl.col("created_at").dt.date().alias("event_date"))
        .group_by(["user_id", "event_date"])
        .agg(pl.len().alias("_n"))
        .sort(["user_id", "event_date"])
    )

    # asof: find last open/click date strictly before date_sent (use _ds_minus1)
    tmp = act2.select(["_row_idx", "user_id", "_ds_minus1"]).sort(["user_id", "_ds_minus1"])

    last_open_joined = tmp.join_asof(
        ea_last_open.select(["user_id", "event_date"]).sort(["user_id", "event_date"]),
        left_on="_ds_minus1",
        right_on="event_date",
        by="user_id",
        strategy="backward",
    ).rename({"event_date": "last_open_date"})

    last_click_joined = tmp.join_asof(
        ea_last_click.select(["user_id", "event_date"]).sort(["user_id", "event_date"]),
        left_on="_ds_minus1",
        right_on="event_date",
        by="user_id",
        strategy="backward",
    ).rename({"event_date": "last_click_date"})

    act2 = act2.join(
        last_open_joined.select(["_row_idx", "last_open_date"]), on="_row_idx", how="left"
    ).join(last_click_joined.select(["_row_idx", "last_click_date"]), on="_row_idx", how="left")

    act2 = act2.with_columns(
        pl.when(pl.col("last_open_date").is_not_null())
        .then((pl.col("date_sent") - pl.col("last_open_date")).dt.total_days())
        .otherwise(999.0)
        .alias("days_since_last_open"),
        pl.when(pl.col("last_click_date").is_not_null())
        .then((pl.col("date_sent") - pl.col("last_click_date")).dt.total_days())
        .otherwise(999.0)
        .alias("days_since_last_click"),
    )

    return act2.fill_null(0.0)


def _temporal_split(df: pl.DataFrame, quantile: float = 0.80) -> tuple[pl.DataFrame, pl.DataFrame]:
    cutoff = df["date_sent"].quantile(quantile, interpolation="nearest")
    train = df.filter(pl.col("date_sent") < cutoff)
    test = df.filter(pl.col("date_sent") >= cutoff)
    return train, test


def _fit_lr(
    train: pl.DataFrame, test: pl.DataFrame, features: list[str]
) -> tuple[float, np.ndarray]:
    X_train = train.select(features).fill_nan(0.0).to_numpy()
    y_train = train["actioned"].cast(pl.Int32).to_numpy()
    X_test = test.select(features).fill_nan(0.0).to_numpy()
    y_test = test["actioned"].cast(pl.Int32).to_numpy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, n_jobs=-1)
    lr.fit(X_tr, y_train)
    proba = lr.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_test, proba)), proba


def _univariate_auc(df: pl.DataFrame, features: list[str]) -> dict[str, float]:
    train, test = _temporal_split(df)
    y_test = test["actioned"].cast(pl.Int32).to_numpy()
    result: dict[str, float] = {}
    for feat in features:
        try:
            sc = StandardScaler()
            X_tr = sc.fit_transform(train.select(feat).fill_nan(0.0).to_numpy())
            X_te = sc.transform(test.select(feat).fill_nan(0.0).to_numpy())
            lr = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
            lr.fit(X_tr, train["actioned"].cast(pl.Int32).to_numpy())
            result[feat] = float(roc_auc_score(y_test, lr.predict_proba(X_te)[:, 1]))
        except Exception:
            result[feat] = 0.5
    return result


def _per_tenure_auc(df: pl.DataFrame, features: list[str]) -> dict[str, float]:
    train, test = _temporal_split(df)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train.select(features).fill_nan(0.0).to_numpy())
    lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, n_jobs=-1)
    lr.fit(X_tr, train["actioned"].cast(pl.Int32).to_numpy())

    bucket_aucs: dict[str, float] = {}
    for bucket in _TENURE_LABELS:
        sub = test.filter(pl.col("tenure_bucket") == bucket)
        if sub.height < 50:
            bucket_aucs[bucket] = float("nan")
            continue
        y = sub["actioned"].cast(pl.Int32).to_numpy()
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            bucket_aucs[bucket] = float("nan")
            continue
        proba = lr.predict_proba(scaler.transform(sub.select(features).fill_nan(0.0).to_numpy()))[
            :, 1
        ]
        bucket_aucs[bucket] = float(roc_auc_score(y, proba))
    return bucket_aucs


def _send_freq_stats(df: pl.DataFrame) -> dict[str, float]:
    mature = df.filter(pl.col("tenure_days") >= 30)
    if mature.height == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    vals = mature["sends_last_30d"]
    return {
        "p10": float(vals.quantile(0.10, interpolation="nearest") or 0),
        "p50": float(vals.quantile(0.50, interpolation="nearest") or 0),
        "p90": float(vals.quantile(0.90, interpolation="nearest") or 0),
    }


def _build_chart(uni_aucs: dict[str, float], mv_auc: float, tenure_aucs: dict[str, float]) -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sorted_feats = sorted(uni_aucs.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f for f, _ in sorted_feats]
    feat_aucs_vals = [a for _, a in sorted_feats]
    y_pos = np.arange(len(feat_names))
    colors = ["#c5500f" if a == max(feat_aucs_vals) else "#2956a3" for a in feat_aucs_vals]
    ax1.barh(y_pos, feat_aucs_vals, color=colors, height=0.65)
    ax1.axvline(
        mv_auc, color="#b00020", linewidth=2, linestyle="--", label=f"Multi AUC={mv_auc:.3f}"
    )
    ax1.axvline(0.5, color="#999", linewidth=1, linestyle=":")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feat_names, fontsize=8)
    ax1.set_xlabel("AUC (univariate LR)")
    ax1.set_title("Univariate AUC per feature\n(multivariate = red dashed)")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0.45, max(max(feat_aucs_vals), mv_auc) + 0.05)
    ax1.invert_yaxis()
    ax1.grid(alpha=0.2, axis="x")

    valid_buckets = [b for b in _TENURE_LABELS if b in tenure_aucs and not np.isnan(tenure_aucs[b])]
    t_aucs = [tenure_aucs[b] for b in valid_buckets]
    x_t = np.arange(len(valid_buckets))
    ax2.bar(x_t, t_aucs, color="#207e3c", width=0.65)
    ax2.axhline(mv_auc, color="#b00020", linewidth=2, linestyle="--", label=f"Overall={mv_auc:.3f}")
    ax2.axhline(0.5, color="#999", linewidth=1, linestyle=":")
    ax2.set_xticks(x_t)
    ax2.set_xticklabels(valid_buckets, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("AUC")
    ax2.set_title("Per-tenure-bucket AUC\n(multivariate RFM model)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.45, max(t_aucs or [0.7]) + 0.08)
    for i, v in enumerate(t_aucs):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
    ax2.grid(alpha=0.2, axis="y")

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _numbers_table(
    mv_auc: float,
    uni_aucs: dict[str, float],
    tenure_aucs: dict[str, float],
    send_freq: dict[str, float],
    n_train: int,
    n_test: int,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.append(("Multivariate AUC (LR)", f"{mv_auc:.4f}"))
    best_feat = max(uni_aucs, key=lambda k: uni_aucs[k])
    rows.append(("Best univariate feature", best_feat))
    rows.append(("Best univariate AUC", f"{uni_aucs[best_feat]:.4f}"))
    rows.append(("Train rows", f"{n_train:,}"))
    rows.append(("Test rows", f"{n_test:,}"))
    rows.append(("", "--- per-tenure AUC ---"))
    for b in _TENURE_LABELS:
        v = tenure_aucs.get(b, float("nan"))
        rows.append((f"  {b}", "n/a" if np.isnan(v) else f"{v:.4f}"))
    rows.append(("", "--- send freq (mature, sends_last_30d) ---"))
    rows.append(("p10", f"{send_freq['p10']:.0f}"))
    rows.append(("p50", f"{send_freq['p50']:.0f}"))
    rows.append(("p90", f"{send_freq['p90']:.0f}"))
    return rows


def _impact_md(mv_auc: float, uni_aucs: dict[str, float], tenure_aucs: dict[str, float]) -> str:
    best_feat = max(uni_aucs, key=lambda k: uni_aucs[k])
    best_uni = uni_aucs[best_feat]
    weak_buckets = [
        b
        for b in _TENURE_LABELS
        if b in tenure_aucs and not np.isnan(tenure_aucs[b]) and tenure_aucs[b] < 0.55
    ]
    lines = [
        f"F06 multivariate AUC = {mv_auc:.4f} (temporal test split). "
        f"Best single feature: {best_feat} (AUC={best_uni:.4f}).",
        "",
        "This is the FLOOR. F07 (embedding affinity) and F10 (sequence features) must "
        f"beat {mv_auc:.4f} materially after history-depth controls to justify added complexity.",
        "",
    ]
    if weak_buckets:
        lines.append(
            f"Deployment risk: model is weak (AUC < 0.55) for {', '.join(weak_buckets)}. "
            "Cold-start fallback (population prior or email-level mean) is required for these segments."
        )
    else:
        lines.append(
            "Model generalises across all tenure buckets with AUC > 0.55 — no cold-start cliff."
        )
    lines += [
        "",
        "F03 constraint honoured: action rates from user_emails.actioned (uncontaminated); "
        "open/click rates from raw email_activities events only.",
    ]
    return "\n".join(lines)


def _run_tier(
    tier_label: str,
    fraction: float | None,
    ea_events: pl.DataFrame,
    git_sha: str,
) -> dict:
    t0 = time.time()
    print(f"\n[F06] Starting tier={tier_label} ...")

    sample = (
        user_emails.sample_percentage(fraction, seed=42)
        if fraction is not None
        else user_emails.full()
    )
    n_rows = sample.height
    print(f"  sample rows={n_rows:,}")

    print("  building features ...")
    df = _build_features(sample, ea_events)

    print("  fitting models ...")
    train, test = _temporal_split(df)
    n_train, n_test = train.height, test.height
    mv_auc, _ = _fit_lr(train, test, _FEATURE_COLS)
    uni_aucs = _univariate_auc(df, _FEATURE_COLS)
    tenure_aucs = _per_tenure_auc(df, _FEATURE_COLS)
    send_freq = _send_freq_stats(df)

    best_feat = max(uni_aucs, key=lambda k: uni_aucs[k])
    best_uni = uni_aucs[best_feat]
    tenure_str = "; ".join(
        f"{b}={v:.3f}" if not np.isnan(v) else f"{b}=n/a" for b, v in tenure_aucs.items()
    )

    wall = time.time() - t0
    print(
        f"F06 [{tier_label}] rows={n_rows:,} wall={wall:.1f}s; "
        f"AUC={mv_auc:.4f}; best_uni={best_uni:.4f} (feat: {best_feat}); "
        f"per_tenure_AUC={tenure_str}"
    )

    chart_bytes = _build_chart(uni_aucs, mv_auc, tenure_aucs)
    numbers_table = _numbers_table(mv_auc, uni_aucs, tenure_aucs, send_freq, n_train, n_test)
    impact = _impact_md(mv_auc, uni_aucs, tenure_aucs)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    peak_mb = _peak_mb()
    sample_label = f"{fraction:.0%} sample, seed=42" if fraction is not None else "full dataset"
    provenance = {
        "data": "user_emails v2 (actioned) + email_activities v2 (open/click per F03)",
        "sample": sample_label,
        "tier": tier_label,
        "wall_seconds": f"{wall:.1f}s",
        "peak_memory_mb": f"{peak_mb:.0f}",
        "ts": ts,
        "git": git_sha,
    }
    tldr = (
        f"RFM baseline AUC={mv_auc:.4f}; best univariate={best_feat} ({best_uni:.4f}). "
        f"[{tier_label} n={n_rows:,} wall={wall:.0f}s]"
    )

    html = render_finding(
        finding_id="F06",
        title="Recency-weighted RFM baseline + send-frequency heterogeneity",
        tldr=tldr,
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
        question_md=(
            "What is the AUC of a model trained exclusively on prior-window engagement history: "
            "action/click/open rates in 7d/30d/90d, days since last open/click/action, "
            "sends in last 30d, and log(prior action count)? "
            "This baseline floor must be beaten materially by F07 and F10. "
            "Open/click features are from raw email_activities (F03 rule). "
            "Action rates are from user_emails.actioned (uncontaminated)."
        ),
        plain_english_md=(
            "Imagine you only know a user's email history: how often they clicked or opened "
            "in the past month, when they last did anything, and how many emails they've received. "
            "Nothing about who they are or what the email says — just their track record.\n\n"
            "This is the simplest useful model. If embeddings or survival curves can't beat this, "
            "those features are not earning their complexity.\n\n"
            "Per-tenure AUC tells us whether the model works for new users (little history) vs "
            "veterans. New users with no history are the cold-start problem: nothing to learn from yet."
        ),
        method_py_source="""
# Sort by (user_id, date_sent) — required for cumulative ops
ue_sorted = sample.select(["user_id", "email_id", "date_sent", "actioned"]).sort(["user_id", "date_sent"])

# Build daily cumulative action/send counts from user_emails.actioned (uncontaminated per F03)
act_daily = (ue_sorted.group_by(["user_id", "date_sent"])
    .agg(n_act=pl.col("actioned").cast(Int32).sum(), n_sends=pl.len())
    .sort(["user_id", "date_sent"])
    .with_columns(cum_act=cum_sum("n_act").over("user_id"), cum_sends=cum_sum("n_sends").over("user_id")))

# For each row: window count = cum_at(date_sent-1) - cum_at(date_sent-W-1)
# via asof join (backward strategy) with date offset columns
for window in [7, 30, 90]:
    act_W = asof_lookup(act2, act_daily, f"_ds_minus1", "cum_act") - asof_lookup(act2, act_daily, f"_ds_minus{window+1}", "cum_act")
    sends_W = asof_lookup(act2, act_daily, "_ds_minus1", "cum_sends") - asof_lookup(act2, act_daily, f"_ds_minus{window+1}", "cum_sends")
    action_rate_Xd = act_W / (sends_W + 1e-9)

# Same for open/click rates from raw email_activities v2
opens_cum = ea_events.filter(action_type="open").group_by(["user_id", "event_date"]).agg(n=len).sort(...).with_columns(cum_opens=n.cum_sum().over("user_id"))
open_rate_Xd = (asof_cum_opens_at_ds1 - asof_cum_opens_at_ds_minus_X1) / sends_X

# Temporal split: train on date_sent < p80, test on date_sent >= p80
train, test = temporal_split(df, quantile=0.80)
lr = LogisticRegression(max_iter=500)
auc = roc_auc_score(y_test, lr.predict_proba(StandardScaler().fit_transform(X_test))[:, 1])
""",
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact,
        provenance=provenance,
        is_mockup=False,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "F06_recency_rfm_baseline.html"
    out_path.write_text(html, encoding="utf-8")

    timing_row = {
        "finding_id": "F06",
        "tier": tier_label,
        "sample_fraction": fraction if fraction is not None else 1.0,
        "rows_in": n_rows,
        "wall_seconds": round(wall, 2),
        "peak_memory_mb": round(peak_mb, 1),
        "timestamp_utc": ts,
        "git_sha": git_sha,
        "notes": f"AUC={mv_auc:.4f}; best_uni={best_feat}({best_uni:.4f}); train={n_train} test={n_test}",
    }
    with _TIMING_JSONL.open("a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    print(f"  wrote {out_path}")
    return {
        "tier": tier_label,
        "n_rows": n_rows,
        "wall": wall,
        "mv_auc": mv_auc,
        "uni_aucs": uni_aucs,
        "tenure_aucs": tenure_aucs,
        "send_freq": send_freq,
    }


def main() -> None:
    git_sha = _git_sha()
    print("[F06] Loading raw email_activities (open/click events) ...")
    t_ea = time.time()
    ea_events = _load_ea_events()
    print(f"  ea_events rows={ea_events.height:,} loaded in {time.time() - t_ea:.1f}s")

    eda_tiers_env = os.environ.get("EDA_TIERS")
    tiers = (
        [t for t in _TIERS if t[0] in {s.strip() for s in eda_tiers_env.split(",")}]
        if eda_tiers_env
        else _TIERS
    )

    last_result: dict = {}
    for tier_label, fraction in tiers:
        last_result = _run_tier(tier_label, fraction, ea_events, git_sha)

    print("\n=== F06 FINAL SUMMARY ===")
    mv_auc = last_result["mv_auc"]
    uni_aucs = last_result["uni_aucs"]
    tenure_aucs = last_result["tenure_aucs"]
    best_feat = max(uni_aucs, key=lambda k: uni_aucs[k])

    print(f"Multivariate AUC: {mv_auc:.4f}")
    print(f"Best univariate: {best_feat} = {uni_aucs[best_feat]:.4f}")
    print("Per-tenure-bucket AUC:")
    for b in _TENURE_LABELS:
        v = tenure_aucs.get(b, float("nan"))
        print(f"  {b:12s}: {'n/a' if np.isnan(v) else f'{v:.4f}'}")
    sf = last_result["send_freq"]
    print(
        f"Send-freq (mature, sends_last_30d): p10={sf['p10']:.0f} p50={sf['p50']:.0f} p90={sf['p90']:.0f}"
    )


if __name__ == "__main__":
    main()
