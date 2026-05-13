"""Build prior_action_recency v1 feature cache.

Feature family: per-(user_id, email_id, date_sent) recency features derived
from the strict 24h-attributed action column (actioned_24h) in user_emails v3.

All 8 features are causally clean: each is computed from rows strictly prior
to the current row's date_sent via .shift(1).over("user_id") applied BEFORE
any window aggregate.

Output columns (3 row keys + 8 features):
    user_id, email_id, date_sent,
    actioned_last_1, actioned_last_3, actioned_last_5, actioned_last_10,
    sends_since_last_action, action_in_last_5, is_in_action_streak,
    lifetime_actions_prior

Usage:
    python -m projects.civic_shout_action_rate_increase.features.prior_action_recency.create_cache_v1 --tier 1pct
    python -m projects.civic_shout_action_rate_increase.features.prior_action_recency.create_cache_v1 --tier 5pct
    python -m projects.civic_shout_action_rate_increase.features.prior_action_recency.create_cache_v1 --tier full
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from projects.civic_shout_action_rate_increase.data import user_emails
from projects.civic_shout_action_rate_increase.features.prior_action_recency.cache import (
    cache as feature_cache,
)

logger = logging.getLogger(__name__)

_TIMING_JSONL = Path(
    "projects/civic_shout_action_rate_increase/features/prior_action_recency/timing_performance.jsonl"
)

_WINDOWS = [1, 3, 5, 10]


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


def build_v1(user_emails_df: pl.DataFrame) -> pl.DataFrame:
    """Compute all 8 prior_action_recency features from a user_emails v3 frame.

    Strict-causality invariant: every feature is derived from rows where
    date_sent < current row's date_sent, enforced by sort + shift(1) before
    any window aggregate.
    """
    df = user_emails_df.sort(["user_id", "date_sent"])

    rolling_exprs: list[pl.Expr] = []
    for n in _WINDOWS:
        rolling_exprs.append(
            pl.col("actioned_24h")
            .cast(pl.UInt8)
            .shift(1)
            .rolling_sum(window_size=n)
            .over("user_id")
            .fill_null(0)
            .cast(pl.UInt8)
            .alias(f"actioned_last_{n}")
        )

    df = df.with_columns(rolling_exprs)

    df = df.with_columns(
        (pl.col("actioned_last_5") >= 1).alias("action_in_last_5"),
        (pl.col("actioned_last_3") == 3).alias("is_in_action_streak"),
    )

    df = df.with_columns(
        pl.col("actioned_24h")
        .cast(pl.Int32)
        .shift(1, fill_value=0)
        .over("user_id")
        .alias("_prev_actioned"),
    )
    df = df.with_columns(
        pl.col("_prev_actioned").cum_sum().over("user_id").alias("_epoch"),
    )
    df = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .over(["user_id", "_epoch"])
        .alias("sends_since_last_action"),
    )

    df = df.with_columns(
        (
            pl.col("actioned_24h")
            .cast(pl.UInt32)
            .cum_sum()
            .over("user_id")
            .shift(1)
            .over("user_id")
            .fill_null(0)
        ).alias("lifetime_actions_prior"),
    )

    return df.select(
        [
            "user_id",
            "email_id",
            "date_sent",
            "actioned_last_1",
            "actioned_last_3",
            "actioned_last_5",
            "actioned_last_10",
            "sends_since_last_action",
            "action_in_last_5",
            "is_in_action_streak",
            "lifetime_actions_prior",
        ]
    )


def main(tier: str = "full") -> None:
    git_sha = _git_sha()
    t0 = time.time()

    fraction: float | None
    if tier == "1pct":
        fraction = 0.01
    elif tier == "5pct":
        fraction = 0.05
    elif tier == "full":
        fraction = None
    else:
        raise ValueError(f"Unknown tier: {tier!r}. Must be one of: 1pct, 5pct, full")

    logger.info("Loading user_emails v3 (tier=%s) ...", tier)
    if fraction is not None:
        raw = user_emails.sample_percentage(fraction, seed=42)
    else:
        raw = user_emails.full(version=3)

    n_input = raw.height
    logger.info("Loaded %d rows", n_input)

    logger.info("Building prior_action_recency v1 features ...")
    features_df = build_v1(raw)

    n_output = features_df.height
    logger.info("Feature frame: %d rows, %d columns", n_output, len(features_df.columns))

    if n_input != n_output:
        raise RuntimeError(
            f"Row count mismatch: input={n_input}, output={n_output}. Build is invalid."
        )

    wall = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if tier == "full":
        logger.info("Writing v1 cache via feature_cache.put ...")
        feature_cache.put(1, features_df)
        logger.info("Cache written successfully.")
    else:
        logger.info("Tier=%s: skipping cache write (full tier only).", tier)

    timing_row = {
        "feature_family": "prior_action_recency",
        "version": 1,
        "tier": tier,
        "rows": n_input,
        "wall_seconds": round(wall, 2),
        "fraction": fraction,
        "ts": ts,
        "git": git_sha,
    }
    _TIMING_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_JSONL.open("a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    logger.info(
        "prior_action_recency v1 [%s]: rows=%d wall=%.1fs",
        tier,
        n_input,
        wall,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=["1pct", "5pct", "full"],
        default="full",
        help="Sampling tier to run",
    )
    args = parser.parse_args()
    main(tier=args.tier)
