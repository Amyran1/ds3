"""Warm the news_sources entity cache from shared S3 export.

This entity is not rebuilt in this repo — the source-of-truth pipeline
lives in dev/backend/datascience (Perigon + Redis). Running this script
just downloads the existing parquet into the local cache and reports
shape/columns for a sanity check.

Usage:
    python -m entities.news_sources.create_cache_v1
"""

from __future__ import annotations

import logging

from entities.news_sources.cache import cache

logger = logging.getLogger(__name__)


def create() -> None:
    """Download the shared news_sources parquet and log summary stats."""
    logger.info("Warming news_sources cache from shared S3 export")
    df = cache.get(1)
    logger.info("news_sources v1 shape: %s", df.shape)
    logger.info("Columns: %s", df.columns)
    logger.info("Null counts: %s", df.null_count().to_dicts())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    create()
