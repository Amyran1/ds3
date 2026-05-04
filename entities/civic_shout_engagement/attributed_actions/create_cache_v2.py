r"""Build attributed_actions v2 from civic_shout_engagement caches.

Attribution rule (v2):
- No absolute window cap (v1's 7-day CAP_SECONDS is removed).
- Every send participates: last send per user gets ``next_send_time =
  BUILD_CUTOFF_UTC`` (no ``exclude_last_send`` behaviour).
- ``window_seconds = next_send_time - send_time`` (full inter-send interval).
- Strict ``lag_seconds < window_seconds`` boundary: a signature at exactly the
  boundary goes to the NEXT send's window.
- ``is_bot_lag = lag_seconds < 10`` — flagged but NOT dropped.  Stage-3
  investigations decide the filter/reweight policy.

Reads from:
    civic_shout_engagement__email_activities v2
    civic_shout_engagement__actions v2

Writes:
    civic_shout_engagement__attributed_actions v2

Schema additions vs v1:
    is_bot_lag: bool   -- True when lag_seconds < 10 (bot-pipeline artifact flag)
    window_seconds:    -- no longer capped; full inter-send interval or
                          BUILD_CUTOFF_UTC - send_time for user's last send

Build anchor: ``BUILD_CUTOFF_UTC = datetime(2026, 4, 21, tzinfo=UTC)``.

Usage::

    ENVIRONMENT=PRODUCTION python -m libs.dsrun --timeout 3600 \
        entities/civic_shout_engagement/attributed_actions/create_cache_v2.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import polars as pl

from entities.civic_shout_engagement.actions_cache import cache as actions_cache
from entities.civic_shout_engagement.attributed_actions.cache import (
    cache as attributed_cache,
)
from entities.civic_shout_engagement.email_activities_cache import (
    cache as email_activities_cache,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc
BUILD_CUTOFF_UTC: datetime = datetime(2026, 4, 21, tzinfo=UTC)

# Bot-lag threshold: signatures with lag < this many seconds are flagged.
_BOT_LAG_SECONDS = 10


def build_attributed_actions_v2(
    activities: pl.DataFrame,
    actions: pl.DataFrame,
) -> pl.DataFrame:
    """Build the v2 attribution ledger from email activities and signatures.

    For each signature, finds the most recent prior send by the same user
    where sign_time - send_time < next_send_time - send_time (no cap).
    The user's most recent send (no later send in the window) uses
    ``BUILD_CUTOFF_UTC`` as ``next_send_time``.

    Within a (user_id, email_id) window the earliest signature (by
    created_at, tiebroken by signature_id) is ``is_first_in_window=True``;
    subsequent ones are an organic cascade.  ``is_bot_lag=True`` when
    lag_seconds < 10; bot rows are retained for Stage-3 analysis.

    Args:
        activities: email_activities v2 DataFrame (must include
            ``action_type``, ``user_id``, ``email_id``, ``created_at``).
        actions: actions v2 DataFrame (must include ``signature_id``,
            ``user_id``, ``petition_id``, ``created_at``).

    Returns:
        One row per signature with columns:
        ``signature_id, user_id, petition_id, sign_time, email_id, send_time,
        lag_seconds, window_seconds, is_first_in_window, is_attributed,
        is_bot_lag``.
    """
    # --- Prepare sends: filter to sent events, sort by (user_id, send_time). ---
    sends = (
        activities.filter(pl.col("action_type") == "sent")
        .select(
            pl.col("user_id"),
            pl.col("email_id"),
            pl.col("created_at").alias("send_time"),
        )
        .sort(["user_id", "send_time"])
    )

    # Attach next_send_time per user via shift(-1) within user group.
    sends = sends.with_columns(
        pl.col("send_time").shift(-1).over("user_id").alias("next_send_time"),
    )

    # Every send participates in v2 — fill the last send's null next_send_time
    # with BUILD_CUTOFF_UTC rather than dropping that row.
    cutoff_lit = pl.lit(BUILD_CUTOFF_UTC).cast(
        pl.Datetime(time_unit="us", time_zone="UTC"),
    )
    sends = sends.with_columns(
        pl.col("next_send_time").fill_null(cutoff_lit).alias("next_send_time"),
    )

    # Compute window_seconds = next_send_time - send_time (full interval, no cap).
    _window_delta = (pl.col("next_send_time") - pl.col("send_time")).dt.total_seconds()
    sends = sends.with_columns(_window_delta.alias("window_seconds"))

    # --- Prepare signatures: sort by (user_id, created_at, signature_id). ---
    sigs = actions.sort(["user_id", "created_at", "signature_id"])

    # --- join_asof backward: each sig gets its most recent prior send. ---
    joined = sigs.join_asof(
        sends.sort(["user_id", "send_time"]),
        left_on="created_at",
        right_on="send_time",
        by="user_id",
        strategy="backward",
    )

    # Compute lag_seconds = created_at (sign_time) - send_time.
    joined = joined.with_columns(
        ((pl.col("created_at") - pl.col("send_time")).dt.total_seconds()).alias(
            "lag_seconds",
        ),
    )

    # Enforce strict < window boundary (signature at exactly window_seconds
    # goes to the NEXT send's window — same semantics as v1).
    not_null = pl.col("send_time").is_not_null()
    in_window = pl.col("lag_seconds") < pl.col("window_seconds")
    joined = joined.with_columns(
        (not_null & in_window).alias("_in_window"),
    )

    # Null out all four attribution columns together when _in_window is False.
    def _keep(col: str) -> pl.Expr:
        _expr = pl.when(pl.col("_in_window")).then(pl.col(col)).otherwise(None)
        return _expr.alias(col)

    joined = joined.with_columns(
        _keep("email_id"),
        _keep("send_time"),
        _keep("lag_seconds"),
        _keep("window_seconds"),
    ).drop("_in_window")

    # Tag first-in-window: earliest by (created_at, signature_id) within
    # (user_id, email_id).  ordinal rank preserves the tiebreak correctly.
    sig_struct = pl.struct(["created_at", "signature_id"])
    joined = joined.with_columns(
        pl.when(pl.col("email_id").is_null())
        .then(pl.lit(value=False))
        .otherwise(
            sig_struct.rank(method="ordinal").over(["user_id", "email_id"]) == 1,
        )
        .alias("is_first_in_window"),
    )

    joined = joined.with_columns(
        (pl.col("email_id").is_not_null() & pl.col("is_first_in_window")).alias(
            "is_attributed",
        ),
    )

    # Add is_bot_lag: flag signatures with lag < _BOT_LAG_SECONDS as likely
    # bot artifacts. fill_null(False) handles unattributed rows (null lag).
    _bot_lag = (pl.col("lag_seconds") < _BOT_LAG_SECONDS).fill_null(value=False)
    joined = joined.with_columns(_bot_lag.alias("is_bot_lag"))

    return joined.select(
        "signature_id",
        "user_id",
        "petition_id",
        pl.col("created_at").alias("sign_time"),
        "email_id",
        "send_time",
        "lag_seconds",
        "window_seconds",
        "is_first_in_window",
        "is_attributed",
        "is_bot_lag",
    )


def _validate(df: pl.DataFrame, actions_row_count: int) -> None:
    """Run pre-write validation gates; sys.exit(1) on any failure.

    Gates:
    1. Row count == actions_row_count (1:1 invariant).  Hard abort.
    2. is_bot_lag fraction of is_attributed rows — informational only (log).
       v1 thresholds ([5%, 15%]) no longer apply: v2's longer inter-send
       windows dilute the denominator, pushing the fraction below the v1
       range.  Stage-3 investigations own the bot-lag filter policy.
    3. No nulls on required columns.  Hard abort.
    4. Attributed rows have non-null attribution columns.  Hard abort.
    """
    n_rows = len(df)

    # Gate 1: 1:1 row count invariant.
    if n_rows != actions_row_count:
        msg = (
            f"ABORT: attributed_actions v2 row count {n_rows} != actions v2 "
            f"row count {actions_row_count}. 1:1 invariant violated."
        )
        logger.error(msg)
        sys.exit(1)
    logger.info("Gate 1 PASS: row count %d == actions v2 row count", n_rows)

    # Gate 2 (informational): is_bot_lag fraction among attributed rows.
    # Not a blocker — v2 attribution window definition change diluted the
    # send-time cluster where bots live; the v1-calibrated thresholds no
    # longer apply.
    attributed = df.filter(pl.col("is_attributed"))
    n_attributed = len(attributed)
    if n_attributed > 0:
        n_bot = int(attributed["is_bot_lag"].sum())
        bot_fraction = n_bot / n_attributed
        logger.info(
            "is_bot_lag fraction of attributed rows: %.2f%% (%d / %d)",
            100.0 * bot_fraction,
            n_bot,
            n_attributed,
        )
    else:
        logger.warning("No attributed rows found — Gate 2 skipped")

    # Gate 3: required columns must have no nulls.
    required_no_null = [
        "signature_id",
        "user_id",
        "petition_id",
        "sign_time",
        "is_attributed",
        "is_bot_lag",
    ]
    for col in required_no_null:
        null_count = df[col].null_count()
        if null_count > 0:
            logger.error(
                "ABORT: column %s has %d null(s) — expected 0.",
                col,
                null_count,
            )
            sys.exit(1)
    logger.info("Gate 3 PASS: required columns have no nulls")

    # Gate 4: attributed rows must have non-null attribution columns.
    if n_attributed > 0:
        attribution_cols = ["email_id", "send_time", "lag_seconds", "window_seconds"]
        for col in attribution_cols:
            null_count = attributed[col].null_count()
            if null_count > 0:
                logger.error(
                    "ABORT: attributed rows have %d null(s) in %s.",
                    null_count,
                    col,
                )
                sys.exit(1)
        logger.info(
            "Gate 4 PASS: attributed rows have non-null attribution columns",
        )


def main() -> None:
    """Load upstream v2 caches, build v2 attribution ledger, write to cache."""
    logger.info(
        "BUILD_CUTOFF_UTC = %s",
        BUILD_CUTOFF_UTC.isoformat(),
    )

    logger.info("Loading email_activities v2...")
    activities = email_activities_cache.get(2)
    logger.info("Loaded email_activities v2: %d rows", len(activities))

    logger.info("Loading actions v2...")
    actions = actions_cache.get(2)
    logger.info("Loaded actions v2: %d rows", len(actions))
    actions_row_count = len(actions)

    logger.info("Building attributed_actions v2...")
    df = build_attributed_actions_v2(activities, actions)
    logger.info("Built attributed_actions v2: %d rows", len(df))

    n_attributed = int(df["is_attributed"].sum())
    _mean_raw = df["is_attributed"].mean()
    pct_attributed: float = 100.0 * _mean_raw if isinstance(_mean_raw, float) else 0.0
    n_cascade = int(
        (~df["is_first_in_window"] & df["email_id"].is_not_null()).sum(),
    )
    logger.info("Attributed: %d (%.1f%%)", n_attributed, pct_attributed)
    logger.info("Organic cascade (in-window, not first): %d", n_cascade)

    _validate(df, actions_row_count)

    attributed_cache.put(2, df)
    logger.info("attributed_actions v2 written to cache.")


if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=_fmt)
    main()
