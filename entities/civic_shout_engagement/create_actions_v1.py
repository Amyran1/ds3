"""Build civic_shout_engagement__actions v1 from Redshift.

Pulls petition signatures for Action Network group 228691 over the last
1 month.  Only signatures for petitions belonging to group 228691 are
included (enforced by the sub-select on ``group_228691_indexed.petitions``).

Usage::

    # Phase A (Redshift pull) + Phase B (ingest):
    PGPASSWORD=<secret> python -m entities.civic_shout_engagement.create_actions_v1

    # Phase B only (re-run on an existing CSV, no Redshift pull):
    python -m entities.civic_shout_engagement.create_actions_v1 --skip-psql
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from entities.civic_shout_engagement._psql import run_psql_to_csv
from entities.civic_shout_engagement.actions_cache import cache as actions_cache

logger = logging.getLogger(__name__)

SQL = """
SELECT id AS signature_id,
       user_id,
       petition_id,
       created_at
FROM group_228691_indexed.signatures
WHERE petition_id IN (
    SELECT id
    FROM group_228691_indexed.petitions
    WHERE group_id = 228691
)
  AND created_at >= current_date - interval '1 month'
"""

RAW_CSV = Path("data/raw/civic_shout_engagement/actions.csv")


def main(*, skip_psql: bool = False) -> None:
    """Pull actions (signatures) from Redshift and write to the v1 entity cache.

    Args:
        skip_psql: If True, skip the Redshift pull and use an existing CSV at
                   ``RAW_CSV``.
    """
    if not skip_psql:
        logger.info("Phase A: pulling actions from Redshift → %s", RAW_CSV)
        run_psql_to_csv(SQL, RAW_CSV)
        logger.info("Phase A: done. CSV size: %d bytes", RAW_CSV.stat().st_size)

    logger.info("Phase B: ingesting %s", RAW_CSV)
    df = (
        pl.scan_csv(RAW_CSV, try_parse_dates=True)
        .select(
            pl.col("signature_id").cast(pl.Int64),
            pl.col("user_id").cast(pl.Int64),
            pl.col("petition_id").cast(pl.Int64),
            pl.col("created_at").cast(pl.Datetime("us", "UTC")),
        )
        .collect(engine="streaming")
    )

    logger.info("actions v1: %d rows", len(df))

    actions_cache.put(1, df)
    logger.info("actions v1 written to cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build civic_shout_engagement__actions v1.",
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
