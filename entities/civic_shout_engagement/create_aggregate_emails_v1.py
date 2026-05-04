r"""Build civic_shout_engagement__aggregate_emails v1 from Redshift + caches.

Pulls one row per sent email for Action Network group 228691 from the Redshift
``emails`` table (subject, sender, body, pre_header, send_date, AN's own send/
action stats), then joins to per-email engagement counters aggregated from the
already-built ``email_activities_cache`` and ``attributed_actions_cache``.

The result is a 19-column parquet with two parallel engagement counter families:
"total_*" (our deduplicated unique-user counts) and "stats_*" (AN's raw counts),
plus three boolean range-check flags.

Usage::

    # Phase A (Redshift pull) + aggregate + cache write:
    PGPASSWORD=<secret> python -m \
        entities.civic_shout_engagement.create_aggregate_emails_v1

    # Skip Redshift pull, re-use an existing CSV at RAW_CSV:
    python -m entities.civic_shout_engagement.create_aggregate_emails_v1 \
        --skip-psql
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from entities.civic_shout_engagement._psql import run_psql_to_csv
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_ID = 228691
_SENT_TOLERANCE = 0.25
RAW_CSV = Path("data/raw/civic_shout_engagement/aggregate_emails.csv")

# Validation gate thresholds
_JOIN_COVERAGE_ABORT = 0.90
_JOIN_COVERAGE_WARN = 0.95
_STATS_NULL_ABORT = 0.25
_STATS_NULL_WARN = 0.10
_SENT_IN_RANGE_ABORT = 0.75
_SENT_IN_RANGE_WARN = 0.90

# All sent emails for the group — no date window filter.  Emails that predate
# the email_activities cache window will have total_*=0 via the LEFT join
# fill_null; their content columns remain accurate.
SQL = """
SELECT id AS email_id,
       subject,
       "from" AS sender,
       inlined_content AS email_body,
       pre_header,
       send_date AS date_sent,
       total_sent AS stats_sent,
       actions_count AS stats_actions,
       stats
FROM group_228691_indexed.emails
WHERE group_id = 228691
  AND status = 5
"""

# JSON parse resilience note:
# str.json_decode(dtype=...) with strict Int64 fields returns null for a
# struct field if the source JSON encodes the value as a string or float.
# For v1 we use strict decode.  If the first full run shows null rate >10%
# for any stats_* field despite the raw CSV carrying values, fall back to a
# permissive decode: use pl.Utf8 for the field then .cast(pl.Int64,
# strict=False) in a follow-up expression.  Document the decision in the
# commit message.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_mean(series: pl.Series) -> float:
    """Return the mean of a boolean Series as a Python float.

    Polars Series.mean() returns a wide union type that includes non-numeric
    variants (datetime, str, bytes, ...).  This helper narrows the return to
    float so pyright can reason about downstream arithmetic comparisons.
    """
    raw = series.mean()
    if not isinstance(raw, (int, float)):
        return 0.0
    return float(raw)


def _datetime_min(series: pl.Series) -> datetime | None:
    """Return the minimum of a datetime Series as a Python datetime, or None.

    Polars Series.min() returns a wide union type.  This helper narrows the
    return to datetime so pyright can reason about downstream comparisons.
    Returns None when the series is empty or the min is not a datetime.
    """
    raw = series.min()
    return raw if isinstance(raw, datetime) else None


# ---------------------------------------------------------------------------
# Phase A — Redshift pull + CSV ingest
# ---------------------------------------------------------------------------


def _fetch_content(*, skip_psql: bool) -> pl.DataFrame:
    """Pull Redshift emails table to CSV and ingest to a polars DataFrame.

    Args:
        skip_psql: When True, skip the Redshift pull and reuse an existing CSV
                   at ``RAW_CSV``.
    """
    if not skip_psql:
        logger.info("Phase A: pulling emails from Redshift → %s", RAW_CSV)
        run_psql_to_csv(SQL, RAW_CSV)
        logger.info("Phase A: done. CSV size: %d bytes", RAW_CSV.stat().st_size)

    logger.info("Phase A: ingesting %s", RAW_CSV)
    df = (
        pl.scan_csv(
            RAW_CSV,
            try_parse_dates=True,
            schema_overrides={
                "stats": pl.Utf8,
                "email_body": pl.Utf8,
                "pre_header": pl.Utf8,
                "subject": pl.Utf8,
                "sender": pl.Utf8,
            },
        )
        .with_columns(
            pl.col("email_id").cast(pl.Int64),
            pl.col("stats_sent").cast(pl.Int64),
            pl.col("stats_actions").cast(pl.Int64),
            pl.col("date_sent").cast(pl.Datetime("us", "UTC")),
        )
        .collect(engine="streaming")
    )
    logger.info("Phase A: ingested %d rows from CSV", len(df))
    return df


# ---------------------------------------------------------------------------
# Phase A.1 — JSON decode of stats column
# ---------------------------------------------------------------------------


def _parse_stats(content: pl.DataFrame) -> pl.DataFrame:
    """Decode the ``stats`` JSON string column into three typed counter columns.

    Extracts ``stats_opens``, ``stats_vo``, and ``stats_clicks`` from the JSON
    blob via a polars expression.  Malformed or missing JSON rows produce nulls
    in all three output columns; null rates are logged and trigger a WARNING if
    any field exceeds 10% of total row count.
    """
    parsed = (
        content.with_columns(
            pl.col("stats")
            .str.json_decode(
                dtype=pl.Struct(
                    [
                        pl.Field("open", pl.Int64),
                        pl.Field("verified_open", pl.Int64),
                        pl.Field("click", pl.Int64),
                    ],
                ),
            )
            .alias("_stats_struct"),
        )
        .with_columns(
            pl.col("_stats_struct").struct.field("open").alias("stats_opens"),
            pl.col("_stats_struct").struct.field("verified_open").alias("stats_vo"),
            pl.col("_stats_struct").struct.field("click").alias("stats_clicks"),
        )
        .drop(["stats", "_stats_struct"])
    )

    n_rows = len(parsed)
    threshold = int(0.10 * n_rows)
    for col in ("stats_opens", "stats_vo", "stats_clicks"):
        null_count = parsed[col].null_count()
        logger.info("%s null_count: %d / %d", col, null_count, n_rows)
        if null_count > threshold:
            logger.warning(
                "%s null rate %.1f%% exceeds 10%% — verify Redshift stats JSON keys",
                col,
                100.0 * null_count / n_rows,
            )

    return parsed


# ---------------------------------------------------------------------------
# Phase B — Aggregate from email_activities cache
# ---------------------------------------------------------------------------


def _aggregate_activities(activities: pl.DataFrame) -> pl.DataFrame:
    """Per-email unique-user counts for each action_type."""
    return activities.group_by("email_id").agg(
        pl.col("user_id").filter(pl.col("action_type") == "sent").n_unique().alias("total_sent"),
        pl.col("user_id").filter(pl.col("action_type") == "open").n_unique().alias("total_opens"),
        pl.col("user_id")
        .filter(pl.col("action_type") == "verified_open")
        .n_unique()
        .alias("total_vo"),
        pl.col("user_id").filter(pl.col("action_type") == "click").n_unique().alias("total_clicks"),
    )


# ---------------------------------------------------------------------------
# Phase B.1 — Aggregate from attributed_actions cache
# ---------------------------------------------------------------------------


def _aggregate_attributed(attributed: pl.DataFrame) -> pl.DataFrame:
    """Per-email count of signatures attributed via the 7-day rule."""
    return (
        attributed.filter(pl.col("is_attributed"))
        .group_by("email_id")
        .agg(pl.len().alias("total_actions"))
    )


# ---------------------------------------------------------------------------
# Phase C — Join and compute validation flags
# ---------------------------------------------------------------------------


def _join_and_validate(
    content: pl.DataFrame,
    activity_agg: pl.DataFrame,
    action_agg: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join content with both aggregation tables and compute range flags.

    Every Redshift email is preserved (LEFT join).  Emails outside the
    activities cache window keep null for all total_* columns, which are then
    filled to 0 (except total_vo — see VO zero-to-null rule).
    """
    return (
        content.join(activity_agg, on="email_id", how="left")
        # Fill zero-activity nulls for total_sent / opens / clicks
        .with_columns(
            pl.col("total_sent").fill_null(0).cast(pl.Int64),
            pl.col("total_opens").fill_null(0).cast(pl.Int64),
            pl.col("total_clicks").fill_null(0).cast(pl.Int64),
        )
        .join(action_agg, on="email_id", how="left")
        # Fill null attributed actions; apply VO zero-to-null rule
        .with_columns(
            pl.col("total_actions").fill_null(0).cast(pl.Int64),
            # VO zero-to-null: if AN has no stats_vo, treat pre-VO-era emails
            # as null (feature not tracked) rather than 0.
            pl.when(pl.col("stats_vo").is_null())
            .then(None)
            .otherwise(pl.col("total_vo"))
            .cast(pl.Int64)
            .alias("total_vo"),
        )
        # Compute validation flags (all vectorised, no Python loops)
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
        # Final column selection in §5.1 docstring order
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
        )
    )


# ---------------------------------------------------------------------------
# Validation gates helper
# ---------------------------------------------------------------------------


def _run_validation_gates(
    df: pl.DataFrame,
    activities: pl.DataFrame,
) -> None:
    """Compute in-window validation gates; sys.exit(1) on any ABORT.

    Logs info + warnings; returns normally if all gates pass or if no
    in-window emails exist (gates skipped with a single WARNING, consistent
    with the 'weird but shouldn't block the build' business case).

    All three gates are scoped to the in-window subset.  Emails that predate
    the activities cache window have total_*=0 from the LEFT join fill_null;
    including them in gates 2-3 would inflate null rates (pre-VO-era emails
    never have stats_vo) and deflate range-check pass rates (total_sent=0
    always fails sent_in_range for an email that was actually sent).  Scoping
    to in-window emails gives accurate signal on data quality for the emails
    we actually scored.

    If no in-window emails exist, all three gates are skipped with a WARNING
    and the caller proceeds to cache.put — the build is valid, just
    unverifiable.
    """
    n_rows = len(df)

    activity_window_start = _datetime_min(activities["created_at"])

    if activity_window_start is None:
        logger.warning(
            "activities cache appears empty — all validation gates skipped",
        )
        return

    in_window_df = df.filter(pl.col("date_sent") >= activity_window_start)
    n_in_window = in_window_df.height
    logger.info(
        "extracted %d emails total; %d fall inside activities window starting %s",
        n_rows,
        n_in_window,
        activity_window_start.strftime("%Y-%m-%d"),
    )

    if n_in_window == 0:
        logger.warning(
            "no emails inside activities window — all validation gates skipped",
        )
        return

    # Gate 1: join coverage — what fraction of in-window emails have
    # total_sent > 0 (i.e. matched at least one activity record)?
    n_in_range = in_window_df.filter(pl.col("total_sent") > 0).height
    join_coverage = n_in_range / n_in_window
    logger.info(
        "join coverage (total_sent > 0, in-window emails only): %.1f%%",
        100.0 * join_coverage,
    )
    if join_coverage < _JOIN_COVERAGE_ABORT:
        logger.error(
            "ABORT: join coverage %.1f%% < 90%% (measured over %d in-window "
            "emails) — check email_activities alignment or status=5 filter. "
            "cache.put skipped.",
            100.0 * join_coverage,
            n_in_window,
        )
        sys.exit(1)
    if join_coverage < _JOIN_COVERAGE_WARN:
        logger.warning(
            "join coverage %.1f%% is between 90-95%% (warn-and-continue, "
            "measured over %d in-window emails)",
            100.0 * join_coverage,
            n_in_window,
        )

    # Gate 2: stats_* null rates (in-window emails only).
    # Pre-window emails predate verified-open tracking and legitimately have
    # stats_vo=null; including them inflates the null rate as a false signal.
    for col in ("stats_opens", "stats_vo", "stats_clicks"):
        null_rate = in_window_df[col].null_count() / n_in_window
        logger.info(
            "%s null rate (in-window emails only): %.1f%%",
            col,
            100.0 * null_rate,
        )
        if null_rate > _STATS_NULL_ABORT:
            logger.error(
                "ABORT: %s null rate %.1f%% > 25%% (in-window emails only) — "
                "verify Redshift stats JSON key names. cache.put skipped.",
                col,
                100.0 * null_rate,
            )
            sys.exit(1)
        if null_rate > _STATS_NULL_WARN:
            logger.warning(
                "%s null rate %.1f%% is between 10-25%% (in-window emails only, warn-and-continue)",
                col,
                100.0 * null_rate,
            )

    # Gate 3: sent_in_range mean (in-window emails only).
    # Pre-window emails have total_sent=0 and sent_in_range=False by
    # construction; they would tank this metric even when the join is healthy.
    sent_in_range_mean = _bool_mean(in_window_df["sent_in_range"])
    opens_in_range_mean = _bool_mean(in_window_df["opens_in_range"])
    clicks_in_range_mean = _bool_mean(in_window_df["clicks_in_range"])
    logger.info(
        "range-check pass rates (in-window emails only): sent=%.1f%%, opens=%.1f%%, clicks=%.1f%%",
        100.0 * sent_in_range_mean,
        100.0 * opens_in_range_mean,
        100.0 * clicks_in_range_mean,
    )
    if sent_in_range_mean < _SENT_IN_RANGE_ABORT:
        logger.error(
            "ABORT: sent_in_range mean %.1f%% < 75%% (in-window emails only) — "
            "deduplicated total_sent diverges too far from stats_sent. "
            "cache.put skipped.",
            100.0 * sent_in_range_mean,
        )
        sys.exit(1)
    if sent_in_range_mean < _SENT_IN_RANGE_WARN:
        logger.warning(
            "sent_in_range mean %.1f%% is between 75-90%% (in-window emails only, "
            "warn-and-continue)",
            100.0 * sent_in_range_mean,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main(*, skip_psql: bool = False) -> None:
    """Build aggregate_emails v1 and write to the entity cache.

    Args:
        skip_psql: When True, skip the Redshift pull and reuse an existing CSV
                   at ``RAW_CSV``.
    """
    activities = email_activities_cache.get(1)
    attributed = attributed_cache.get(1)

    content = _fetch_content(skip_psql=skip_psql)
    content = _parse_stats(content)
    logger.info("content: %d emails from Redshift", len(content))

    activity_agg = _aggregate_activities(activities)
    action_agg = _aggregate_attributed(attributed)

    df = _join_and_validate(content, activity_agg, action_agg)
    logger.info("aggregate_emails v1: %d rows after join", len(df))

    _run_validation_gates(df, activities)

    # All gates passed (or skipped for empty window) — write cache.
    aggregate_emails_cache.put(1, df)
    logger.info("aggregate_emails v1 written to cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build civic_shout_engagement__aggregate_emails v1.",
    )
    parser.add_argument(
        "--skip-psql",
        action="store_true",
        help="Skip the Redshift pull and ingest an existing CSV at RAW_CSV.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(skip_psql=args.skip_psql)
