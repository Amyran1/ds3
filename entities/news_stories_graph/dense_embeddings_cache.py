"""Dense embedding matrix for the full news story corpus.

Float32 matrix with shape ~(3.5M, 1536), L2-normalized, produced by
``text-embedding-3-small`` over the news story text. Stored as a ``.npy``
locally and on S3, with a JSON side-car (``story_ids_v1.json``) recording the
row order so downstream code can map a ``story_id`` from the
``news_stories`` entity to its row index unambiguously.

Usage::

    from entities import news_stories_graph

    arr = news_stories_graph.dense_embeddings_cache.cache.get(1)
    ids = news_stories_graph.dense_embeddings_cache.cache.get_story_ids(1)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from libs.cache import s3

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/dense_embeddings",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/news_stories_graph/dense_embeddings"
)
_BUCKET = "chorus-content-assets"


class DenseEmbeddingsCache:
    """Local + S3 cache for news story dense embeddings."""

    def get(self, version: int = 1) -> np.ndarray:
        """Return the embedding matrix, pulling from S3 if needed."""
        local_path = self._local_path(version)
        if local_path.exists():
            logger.debug("Cache hit: dense embeddings v%d", version)
            return np.load(local_path)
        logger.info("Cache miss: downloading dense embeddings v%d from S3", version)
        s3.download(_BUCKET, self._s3_key(version), local_path)
        return np.load(local_path)

    def get_story_ids(self, version: int = 1) -> list[str]:
        """Return the row-ordered list of story_ids for this version."""
        local_path = self._ids_local_path(version)
        if not local_path.exists():
            logger.info("Cache miss: downloading story_ids v%d from S3", version)
            s3.download(_BUCKET, self._ids_s3_key(version), local_path)
        return [str(sid) for sid in json.loads(local_path.read_text())]

    def put(
        self,
        arr: np.ndarray,
        story_ids: list[str],
        version: int = 1,
    ) -> None:
        """Persist both the array and the id side-car locally and to S3."""
        if len(story_ids) != arr.shape[0]:
            msg = (
                f"story_ids length ({len(story_ids)}) does not match "
                f"array row count ({arr.shape[0]})"
            )
            raise ValueError(msg)

        local_path = self._local_path(version)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(local_path, arr)
        s3.upload(local_path, _BUCKET, self._s3_key(version))

        ids_path = self._ids_local_path(version)
        ids_path.write_text(json.dumps(story_ids))
        s3.upload(ids_path, _BUCKET, self._ids_s3_key(version))

        logger.info(
            "Saved dense_embeddings v%d (%s) + %d ids locally and to S3",
            version,
            arr.shape,
            len(story_ids),
        )

    def _local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"embeddings_v{version}.npy"

    def _ids_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"story_ids_v{version}.json"

    def _s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/embeddings_v{version}.npy"

    def _ids_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/story_ids_v{version}.json"


cache = DenseEmbeddingsCache()
