"""EntityCache config for petition entity.

One row per petition signed in the 24-month window [2024-04-21, 2026-04-21)
for Action Network group 228691.

Schema (v1):
    petition_id:   i64                -- PK from petitions.id
    created_at:    datetime[us, UTC]  -- petitions.created_at
    text:          str                -- title + HTML-stripped description_info
    dense_vector:  list[f32]          -- 1536-dim text-embedding-3-small, L2-norm
    sparse_vector: struct{indices: list[u32], values: list[f32]}  -- BM25, L2-norm

Usage::

    from entities.petition.cache import cache

    df = cache.get(1)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/petition")
_S3_PREFIX = "autonomous-data-scientist/civic_shout_news_environment/entities/petition"

cache = EntityCache(
    entity="petition",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="petition_v1", fmt="parquet"),
    },
)
