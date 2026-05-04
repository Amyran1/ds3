"""BM25 sparse-similarity edges for the news story graph.

Undirected story/story edges (``src < dst`` in string order) produced by
thresholding the BM25 cosine-similarity matrix at the Q9 winning
tau = 0.30. Populated by the ``create_bm25_edges_v1.py`` build script
(a later dispatch; this module is scaffold-only).

Schema:
    src:        Utf8
    dst:        Utf8
    similarity: Float32

Usage::

    from entities import news_stories_graph

    df = news_stories_graph.bm25_edges_cache.cache.get(1)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/bm25_edges",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities/news_stories_graph/bm25_edges"
)

cache = EntityCache(
    entity="news_stories_graph__bm25_edges",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="bm25_edges_v1", fmt="parquet"),
    },
)
