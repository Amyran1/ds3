"""News stories entity cache (production Mongo documents.news_stories).

This cache holds the full Civic Shout news story corpus (projected fields +
build-time join against the news_sources reference table). Unit of observation
is one story, keyed by `story_id`.

Usage:
    from entities.news_stories.cache import cache

    df = cache.get(1)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/news_stories")
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities/news_stories"
)

cache = EntityCache(
    entity="news_stories",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="news_stories_v1", fmt="parquet"),
    },
)
