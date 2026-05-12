"""EntityCache config for user_petition (petition signatures).

A semantically-clean view of the signature stream joining users to petitions.
Derived from civic_shout_engagement__actions v2 (24-month anchored window).

Schema (v1):
    user_id:     i64                -- signer
    petition_id: i64                -- which petition was signed
    signed_at:   datetime[us, UTC]  -- equals actions.created_at

Usage::

    from entities.user_petition.cache import cache

    df = cache.get(1)   # 24-month anchored window, 3-column signature stream
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/user_petition")
_S3_PREFIX = "autonomous-data-scientist/civic_shout_news_environment/entities/user_petition"

cache = EntityCache(
    entity="user_petition",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="user_petition_v1", fmt="parquet"),
    },
)
