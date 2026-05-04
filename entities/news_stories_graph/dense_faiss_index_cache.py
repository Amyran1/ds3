"""Trained FAISS dense index cache for the full news story corpus.

Single-artifact custom cache that serializes a trained FAISS index
(typically an ``IndexIVFFlat`` trained over the dense embedding matrix
from ``dense_embeddings_cache``) via ``faiss.write_index`` /
``faiss.read_index``.

FAISS IVF (Inverted File) indexes partition vectors into Voronoi cells
during training; at query time, callers search only the ``nprobe``
nearest cells instead of scanning the entire corpus. Larger ``nprobe``
trades latency for recall. ``nprobe`` is NOT persisted inside the
index -- callers must set it explicitly after ``read_index``.

Usage::

    import faiss

    from entities import news_stories_graph

    index = news_stories_graph.dense_faiss_index_cache.cache.get(1)
    index.nprobe = 16  # caller-configured
    D, I = index.search(query_vectors, k=50)

Artifacts (per version):
    dense_faiss_index_v{version}.faiss - FAISS binary, written via
                                         ``faiss.write_index``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from libs.cache import s3

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/dense_faiss_index",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/news_stories_graph/dense_faiss_index"
)
_BUCKET = "chorus-content-assets"


class DenseFaissIndexCache:
    """Local + S3 cache for a trained FAISS index over news story embeddings."""

    def get(self, version: int = 1) -> faiss.Index:
        """Download and load the FAISS index. Caller decides ``nprobe``."""
        import faiss as _faiss

        local_path = self.local_path(version)
        if not local_path.exists():
            logger.info(
                "Cache miss: downloading dense faiss index v%d from S3",
                version,
            )
            s3.download(_BUCKET, self._s3_key(version), local_path)
        else:
            logger.debug("Cache hit: dense faiss index v%d", version)
        # faiss Python stubs are incomplete; read_index returns Any.
        return cast("faiss.Index", cast("Any", _faiss).read_index(str(local_path)))

    def put(self, index: faiss.Index, version: int = 1) -> None:
        """Serialize via ``faiss.write_index`` and upload to S3."""
        import faiss as _faiss

        local_path = self.local_path(version)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cast("Any", _faiss).write_index(index, str(local_path))
        s3.upload(local_path, _BUCKET, self._s3_key(version))

        logger.info("Saved dense_faiss_index v%d locally and to S3", version)

    def local_path(self, version: int = 1) -> Path:
        """Return the local path for callers that need it directly."""
        return _CACHE_DIR / f"dense_faiss_index_v{version}.faiss"

    def _s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/dense_faiss_index_v{version}.faiss"


cache = DenseFaissIndexCache()
