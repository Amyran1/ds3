"""Bipartite story/attribute-value edges for the news story graph.

Long-form parquet produced after the eligibility filter (FLOOR=2, CAP=5000)
is applied to attribute values extracted from ``news_stories``. Excludes the
``locations`` attribute. Unit of observation is one (story, attribute, value)
edge.

Schema:
    story_id:       Utf8
    attribute_type: Utf8
    value:          Utf8

Usage::

    from entities import news_stories_graph

    df = news_stories_graph.bipartite_edges_cache.cache.get(1)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/bipartite_edges",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/news_stories_graph/bipartite_edges"
)

cache = EntityCache(
    entity="news_stories_graph__bipartite_edges",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="bipartite_edges_v1", fmt="parquet"),
    },
)
