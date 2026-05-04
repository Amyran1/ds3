r"""Build civic_shout_engagement__aggregate_emails v3.

Inherits content + embeddings from ``aggregate_emails`` v2, then replaces the
``total_*`` counter columns with fresh counts aggregated from the 24-month
upstreams (``email_activities`` v2 + ``attributed_actions`` v2).

**What changes vs v2:**

- ``total_sent``, ``total_opens``, ``total_vo``, ``total_clicks`` — recomputed
  as n_unique(user_id) per action_type from ``email_activities`` v2 (24-month
  window).
- ``total_actions`` — recomputed as count of ``is_attributed=True`` rows per
  ``email_id`` from ``attributed_actions`` v2 (next-send-boundary, uncapped).
- Range-check flags (``sent_in_range``, ``opens_in_range``,
  ``clicks_in_range``) — recomputed against the fresh totals.
- ``stats_*`` columns, content columns, and embedding columns are passed
  through unchanged from v2.

**No Redshift re-pull.** v3 is a derived-only build; all inputs come from
existing entity caches.

Build anchor: ``BUILD_CUTOFF_UTC = datetime(2026, 4, 21, tzinfo=UTC)``.

Usage::

    ENVIRONMENT=PRODUCTION python -m libs.dsrun --timeout 1800 \
        entities/civic_shout_engagement/create_aggregate_emails_v3.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import polars as pl

from entities.civic_shout_engagement.aggregate_emails_cache import (
    cache as aggregate_emails_cache,
)
from entities.civic_shout_engagement.attributed_actions.cache import (
    cache as attributed_cache,
)
from entities.civic_shout_engagement.email_activities_cache import (
    cache as email_activities_cache,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc
BUILD_CUTOFF_UTC: datetime = datetime(2026, 4, 21, tzinfo=UTC)

# Validation thresholds (carried over from v1 gates).
_SENT_TOLERANCE = 0.25
_JOIN_COVERAGE_ABORT = 0.90
_JOIN_COVERAGE_WARN = 0.95
_SENT_IN_RANGE_ABORT = 0.75
_SENT_IN_RANGE_WARN = 0.90
_DENSE_VECTOR_MIN_PRESENT = 0.99  # Gate 2: dense_vector must be non-null >= 99%.
_IN_WINDOW_MIN_SENT = 0.95  # Gate 3: in-window emails with total_sent > 0.

# Columns inherited from v2 (content + stats + embeddings); never recomputed.
_V2_PASSTHROUGH_COLS = [
    "email_id",
    "date_sent",
    "subject",
    "sender",
    "email_body",
    "pre_header",
    "stats_sent",
    "stats_opens",
    "stats_vo",
    "stats_clicks",
    "stats_actions",
    "text_content",
    "dense_vector",
    "tfidf_vector",
    "sparse_vector",
]

# Counter columns produced by v2 — we replace these with fresh v2-upstream counts.
_V2_COUNTER_COLS = [
    "total_sent",
    "total_opens",
    "total_vo",
    "total_clicks",
    "total_actions",
    "sent_in_range",
    "opens_in_range",
    "clicks_in_range",
]


def _bool_mean(series: pl.Series) -> float:
    """Return the mean of a boolean Series as a Python float."""
    raw = series.mean()
    if not isinstance(raw, (int, float)):
        return 0.0
    return float(raw)


def _aggregate_activities(activities: pl.DataFrame) -> pl.DataFrame:
    """Per-email unique-user counts for each action_type from email_activities v2."""
    _sent = pl.col("user_id").filter(pl.col("action_type") == "sent").n_unique()
    _open = pl.col("user_id").filter(pl.col("action_type") == "open").n_unique()
    _vo = pl.col("user_id").filter(pl.col("action_type") == "verified_open").n_unique()
    _click = pl.col("user_id").filter(pl.col("action_type") == "click").n_unique()
    return activities.group_by("email_id").agg(
        _sent.alias("total_sent"),
        _open.alias("total_opens"),
        _vo.alias("total_vo"),
        _click.alias("total_clicks"),
    )


def _aggregate_attributed(attributed: pl.DataFrame) -> pl.DataFrame:
    """Per-email count of attributed signatures from attributed_actions v2."""
    return (
        attributed.filter(pl.col("is_attributed"))
        .group_by("email_id")
        .agg(pl.len().alias("total_actions"))
    )


def _join_and_compute(
    v2_content: pl.DataFrame,
    activity_agg: pl.DataFrame,
    action_agg: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join v2 content with fresh counter aggregates and recompute flags.

    Every v2 email is preserved (LEFT join).  Emails outside the v2 activities
    window keep null totals, filled to 0 (except total_vo — VO zero-to-null
    rule preserved from v1).
    """
    # Start from passthrough columns only — drop the stale v2 counter columns.
    base = v2_content.select(_V2_PASSTHROUGH_COLS)

    return (
        base.join(activity_agg, on="email_id", how="left")
        .with_columns(
            pl.col("total_sent").fill_null(0).cast(pl.Int64),
            pl.col("total_opens").fill_null(0).cast(pl.Int64),
            pl.col("total_clicks").fill_null(0).cast(pl.Int64),
        )
        .join(action_agg, on="email_id", how="left")
        .with_columns(
            pl.col("total_actions").fill_null(0).cast(pl.Int64),
            # VO zero-to-null: null stats_vo means AN never tracked verified-opens
            # for that email — treat as null (not zero) to match v1 semantics.
            pl.when(pl.col("stats_vo").is_null())
            .then(None)
            .otherwise(pl.col("total_vo"))
            .cast(pl.Int64)
            .alias("total_vo"),
        )
        # Recompute validation flags with fresh totals.
        .with_columns(
            (
                pl.col("stats_sent").is_not_null()
                & (pl.col("stats_sent") > 0)
                & (pl.col("total_sent") <= pl.col("stats_sent"))
                & (
                    (pl.col("stats_sent") - pl.col("total_sent")).cast(pl.Float64)
                    / pl.col("stats_sent").cast(pl.Float64)
                    < _SENT_TOLERANCE
                )
            )
            .fill_null(value=False)
            .alias("sent_in_range"),
            (
                pl.col("stats_opens").is_not_null()
                & (
                    ((pl.col("total_opens") == 0) & (pl.col("stats_opens") == 0))
                    | (pl.col("total_opens") <= pl.col("stats_opens"))
                )
            )
            .fill_null(value=False)
            .alias("opens_in_range"),
            (
                pl.col("stats_clicks").is_not_null()
                & (
                    ((pl.col("total_clicks") == 0) & (pl.col("stats_clicks") == 0))
                    | (pl.col("total_clicks") <= pl.col("stats_clicks"))
                )
            )
            .fill_null(value=False)
            .alias("clicks_in_range"),
        )
        # Final column ordering: match v1 schema order + append v2 additions.
        .select(
            "email_id",
            "date_sent",
            "subject",
            "sender",
            "email_body",
            "pre_header",
            "total_sent",
            "total_opens",
            "total_vo",
            "total_clicks",
            "total_actions",
            "stats_sent",
            "stats_opens",
            "stats_vo",
            "stats_clicks",
            "stats_actions",
            "sent_in_range",
            "opens_in_range",
            "clicks_in_range",
            "text_content",
            "dense_vector",
            "tfidf_vector",
            "sparse_vector",
        )
    )


def _run_validation_gates(
    df: pl.DataFrame,
    v2_row_count: int,
) -> None:
    """Run pre-write validation gates; sys.exit(1) on hard failure.

    Gates:
    1. Row count == v2 row count (same one-row-per-email universe).
    2. dense_vector non-null on >= _DENSE_VECTOR_MIN_PRESENT of rows.
    3. total_sent > 0 for >= _IN_WINDOW_MIN_SENT of in-window emails.
    """
    n_rows = len(df)

    # Gate 1: row count invariant.
    if n_rows != v2_row_count:
        logger.error(
            "ABORT: aggregate_emails v3 row count %d != v2 row count %d.",
            n_rows,
            v2_row_count,
        )
        sys.exit(1)
    logger.info("Gate 1 PASS: row count %d == v2 row count", n_rows)

    # Gate 2: dense_vector non-null >= _DENSE_VECTOR_MIN_PRESENT.
    dv_null_count = df["dense_vector"].null_count()
    dv_present_frac = 1.0 - (dv_null_count / n_rows)
    logger.info(
        "dense_vector non-null fraction: %.2f%% (%d nulls / %d rows)",
        100.0 * dv_present_frac,
        dv_null_count,
        n_rows,
    )
    if dv_present_frac < _DENSE_VECTOR_MIN_PRESENT:
        logger.error(
            "ABORT: dense_vector non-null fraction %.2f%% < %.0f%% — "
            "v2 embedding carryover may be incomplete.",
            100.0 * dv_present_frac,
            100.0 * _DENSE_VECTOR_MIN_PRESENT,
        )
        sys.exit(1)
    logger.info(
        "Gate 2 PASS: dense_vector >= %.0f%% non-null",
        100.0 * _DENSE_VECTOR_MIN_PRESENT,
    )

    # Gate 3: in-window emails should mostly have total_sent > 0.
    window_start = BUILD_CUTOFF_UTC.replace(year=BUILD_CUTOFF_UTC.year - 2)
    cutoff_lit = pl.lit(window_start).cast(
        pl.Datetime(time_unit="us", time_zone="UTC"),
    )
    in_window = df.filter(pl.col("date_sent") >= cutoff_lit)
    n_in_window = len(in_window)
    if n_in_window > 0:
        n_with_sent = len(in_window.filter(pl.col("total_sent") > 0))
        coverage = n_with_sent / n_in_window
        logger.info(
            "In-window emails (>= %s) with total_sent > 0: %.1f%% (%d / %d)",
            window_start.date().isoformat(),
            100.0 * coverage,
            n_with_sent,
            n_in_window,
        )
        if coverage < _IN_WINDOW_MIN_SENT:
            logger.error(
                "ABORT: only %.1f%% of in-window emails have total_sent > 0 "
                "(gate requires >= %.0f%%) — check email_activities v2 join coverage.",
                100.0 * coverage,
                100.0 * _IN_WINDOW_MIN_SENT,
            )
            sys.exit(1)
        logger.info(
            "Gate 3 PASS: in-window join coverage >= %.0f%%",
            100.0 * _IN_WINDOW_MIN_SENT,
        )
    else:
        logger.warning("No in-window emails found — Gate 3 skipped")


def main() -> None:
    """Load upstream v2 caches, rebuild counters, write aggregate_emails v3."""
    logger.info("BUILD_CUTOFF_UTC = %s", BUILD_CUTOFF_UTC.isoformat())

    logger.info("Loading aggregate_emails v2 (content + embeddings)...")
    v2_content = aggregate_emails_cache.get(2)
    v2_row_count = len(v2_content)
    n_v2_cols = len(v2_content.columns)
    logger.info("aggregate_emails v2: %d rows, %d columns", v2_row_count, n_v2_cols)

    logger.info("Loading email_activities v2 (24-month)...")
    activities = email_activities_cache.get(2)
    logger.info("email_activities v2: %d rows", len(activities))

    logger.info("Loading attributed_actions v2...")
    attributed = attributed_cache.get(2)
    logger.info("attributed_actions v2: %d rows", len(attributed))

    logger.info("Aggregating activity counters per email...")
    activity_agg = _aggregate_activities(activities)
    logger.info("Activity aggregation: %d email_id groups", len(activity_agg))

    logger.info("Aggregating attributed action counts per email...")
    action_agg = _aggregate_attributed(attributed)
    logger.info("Action aggregation: %d email_id groups", len(action_agg))

    logger.info("Joining v2 content with fresh counters...")
    df = _join_and_compute(v2_content, activity_agg, action_agg)
    n_v3_cols = len(df.columns)
    logger.info("aggregate_emails v3: %d rows, %d columns", len(df), n_v3_cols)

    # Log summary stats on freshly computed counters.
    total_sent_mean = df["total_sent"].mean()
    total_actions_sum = df["total_actions"].sum()
    sent_in_range_mean = _bool_mean(df["sent_in_range"])
    logger.info(
        "Counter summary: sent=%.1f, actions=%d, sent_in_range=%.1f%%",
        total_sent_mean if isinstance(total_sent_mean, (int, float)) else 0.0,
        total_actions_sum if isinstance(total_actions_sum, (int, float)) else 0,
        100.0 * sent_in_range_mean,
    )

    _run_validation_gates(df, v2_row_count)

    aggregate_emails_cache.put(3, df)
    logger.info("aggregate_emails v3 written to cache.")


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
