"""News sources reference entity cache.

Reuses the shared Perigon-derived source metadata parquet published at
s3://chorus-content-assets/data-science/extractors/news_sources/news_sources.parquet
(built from dev/backend/datascience; not rebuilt here).

Usage:
    from entities.news_sources.cache import cache

    df = cache.get(1)  # polars DataFrame keyed by `domain`
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/news_sources")
_S3_PREFIX = "data-science/extractors/news_sources"

cache = EntityCache(
    entity="news_sources",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="news_sources", fmt="parquet"),
    },
)
