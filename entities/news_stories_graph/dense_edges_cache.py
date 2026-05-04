"""Dense-similarity edges for the news story graph.

Undirected story/story edges (``src < dst``) produced by thresholding the
dense embedding cosine-similarity matrix at tau = 0.50.

Schema:
    src:        Utf8
    dst:        Utf8
    similarity: Float32

Usage::

    from entities import news_stories_graph

    df = news_stories_graph.dense_edges_cache.cache.get(1)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/dense_edges",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities/news_stories_graph/dense_edges"
)

cache = EntityCache(
    entity="news_stories_graph__dense_edges",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="dense_edges_v1", fmt="parquet"),
    },
)
