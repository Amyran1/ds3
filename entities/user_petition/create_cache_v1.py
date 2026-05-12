"""Build user_petition v1 from civic_shout_engagement actions v2.

Selects user_id, petition_id, and created_at (aliased to signed_at) from the
actions cache. No Redshift query is needed -- the source is already a local
parquet.

Usage::

    ENVIRONMENT=PRODUCTION python -m entities.user_petition.create_cache_v1
"""

from __future__ import annotations

import logging
import sys

import polars as pl

from entities.civic_shout_engagement.actions_cache import cache as actions_cache
from entities.user_petition.cache import cache

logger = logging.getLogger(__name__)

_KEY_COLS = ("user_id", "petition_id", "signed_at")


def _validate(df: pl.DataFrame, expected_rows: int) -> None:
    n = len(df)
    if n != expected_rows:
        logger.error(
            "ABORT: row count mismatch -- got %d, expected %d (actions v2 count).",
            n,
            expected_rows,
        )
        sys.exit(1)

    for col in _KEY_COLS:
        null_count = df[col].null_count()
        if null_count != 0:
            logger.error("ABORT: %s has %d nulls.", col, null_count)
            sys.exit(1)

    logger.info("Validation passed: %d rows, 0 nulls in all key columns.", n)


def main() -> None:
    logger.info("Loading actions v2 ...")
    actions_df = actions_cache.get(2)
    expected_rows = len(actions_df)
    logger.info("Actions v2 loaded: %d rows.", expected_rows)

    df = actions_df.select(
        pl.col("user_id"),
        pl.col("petition_id"),
        pl.col("created_at").alias("signed_at"),
    )

    _validate(df, expected_rows)
    cache.put(1, df)
    logger.info("user_petition v1 written to cache.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
