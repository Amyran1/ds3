r"""Build civic_shout_engagement__email_activities v1 from Redshift.

Pulls all event types (sent, open, click, verified_open, etc.) for Action
Network group 228691 email activities over the last 1 month + 14 days.  The
+14-day buffer (2x the 7-day attribution cap) ensures that sends which fall
just before the nominal 1-month boundary are still available when
``create_cache_v1.py`` in ``attributed_actions`` joins signatures to sends.

Usage::

    # Phase A (Redshift pull) + Phase B (ingest):
    PGPASSWORD=<secret> python -m \
        entities.civic_shout_engagement.create_email_activities_v1

    # Phase B only (re-run on an existing CSV, no Redshift pull):
    python -m entities.civic_shout_engagement.create_email_activities_v1 \
        --skip-psql
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from entities.civic_shout_engagement._psql import run_psql_to_csv
from entities.civic_shout_engagement.email_activities_cache import (
    cache as email_activities_cache,
)

logger = logging.getLogger(__name__)

SQL = """
SELECT id AS activity_id,
       email_id,
       recipient_id AS user_id,
       action_type,
       created_at
FROM group_228691_indexed.email_activities_16
-- Buffer window (1 month + 14 days) so day-1 actions can attribute to
-- earlier sends that fall outside the nominal 1-month boundary.
WHERE created_at >= current_date - interval '1 month 14 days'
"""

RAW_CSV = Path("data/raw/civic_shout_engagement/email_activities.csv")


def _action_type_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    """Return action_type value counts sorted descending by n."""
    by_count = df.group_by("action_type").agg(pl.len().alias("n"))
    return by_count.sort("n", descending=True)


def main(*, skip_psql: bool = False) -> None:
    """Pull email_activities from Redshift and write to the v1 entity cache.

    Args:
        skip_psql: If True, skip the Redshift pull and use an existing CSV at
                   ``RAW_CSV``.  Useful for re-running the ingest phase after a
                   failed first attempt.
    """
    if not skip_psql:
        logger.info("Phase A: pulling email_activities from Redshift → %s", RAW_CSV)
        run_psql_to_csv(SQL, RAW_CSV)
        logger.info("Phase A: done. CSV size: %d bytes", RAW_CSV.stat().st_size)

    logger.info("Phase B: ingesting %s", RAW_CSV)
    df = (
        pl.scan_csv(
            RAW_CSV,
            try_parse_dates=True,
            schema_overrides={"action_type": pl.Categorical},
        )
        .select(
            pl.col("activity_id").cast(pl.Int64),
            pl.col("email_id").cast(pl.Int64),
            pl.col("user_id").cast(pl.Int64),
            pl.col("action_type"),
            pl.col("created_at").cast(pl.Datetime("us", "UTC")),
        )
        .collect(engine="streaming")
    )

    logger.info("email_activities v1: %d rows", len(df))
    logger.info(
        "action_type breakdown (verify 'sent' is present):\n%s",
        _action_type_breakdown(df),
    )

    email_activities_cache.put(1, df)
    logger.info("email_activities v1 written to cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build civic_shout_engagement__email_activities v1.",
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
