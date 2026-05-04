"""Sparse TF-IDF matrix cache for the full news story corpus.

Three-artifact cache holding a (~3.5M x 50K) L2-normalized TF-IDF matrix
alongside its vocabulary and a row-ordered ``story_id`` side-car.

TF-IDF config:
    ngram_range   = (1, 2)
    max_features  = 50_000
    min_df        = 5
    max_df        = 0.5
    stop_words    = "english"
    sublinear_tf  = True
    norm          = "l2"

Artifacts (per version):
    tfidf_v{version}.npz       - scipy sparse CSR matrix
    vocab_v{version}.txt       - 50K vocabulary tokens, row-ordered
    story_ids_v{version}.json  - list of story_ids, row order matches matrix

Usage::

    from entities import news_stories_graph

    matrix = news_stories_graph.sparse_tfidf_cache.cache.get(1)
    vocab = news_stories_graph.sparse_tfidf_cache.cache.get_vocab(1)
    ids = news_stories_graph.sparse_tfidf_cache.cache.get_story_ids(1)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from scipy.sparse import csr_matrix, load_npz, save_npz

from libs.cache import s3

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/sparse_tfidf",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/news_stories_graph/sparse_tfidf"
)
_BUCKET = "chorus-content-assets"


class SparseTfidfCache:
    """Local + S3 cache for the news story TF-IDF matrix + side-cars."""

    def get(self, version: int = 1) -> csr_matrix:
        """Return the TF-IDF matrix as a CSR, pulling from S3 if needed."""
        local_path = self._matrix_local_path(version)
        if not local_path.exists():
            logger.info("Cache miss: downloading tfidf v%d from S3", version)
            s3.download(_BUCKET, self._matrix_s3_key(version), local_path)
        else:
            logger.debug("Cache hit: tfidf v%d", version)
        matrix = load_npz(local_path)
        return matrix.tocsr()

    def get_vocab(self, version: int = 1) -> list[str]:
        """Return the row-ordered vocabulary for this version."""
        local_path = self._vocab_local_path(version)
        if not local_path.exists():
            logger.info("Cache miss: downloading tfidf vocab v%d from S3", version)
            s3.download(_BUCKET, self._vocab_s3_key(version), local_path)
        return local_path.read_text().splitlines()

    def get_story_ids(self, version: int = 1) -> list[str]:
        """Return the row-ordered list of story_ids for this version."""
        local_path = self._ids_local_path(version)
        if not local_path.exists():
            logger.info(
                "Cache miss: downloading tfidf story_ids v%d from S3",
                version,
            )
            s3.download(_BUCKET, self._ids_s3_key(version), local_path)
        return [str(sid) for sid in json.loads(local_path.read_text())]

    def put(
        self,
        matrix: csr_matrix,
        vocab: list[str],
        story_ids: list[str],
        version: int = 1,
    ) -> None:
        """Persist matrix, vocab, and story_ids locally and to S3."""
        shape = cast("tuple[int, int]", matrix.shape)
        n_rows, n_cols = shape
        if len(story_ids) != n_rows:
            got = len(story_ids)
            msg = f"story_ids length ({got}) does not match matrix row count ({n_rows})"
            raise ValueError(msg)
        if len(vocab) != n_cols:
            got = len(vocab)
            msg = f"vocab length ({got}) does not match matrix column count ({n_cols})"
            raise ValueError(msg)

        matrix_path = self._matrix_local_path(version)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(matrix_path, matrix)
        s3.upload(matrix_path, _BUCKET, self._matrix_s3_key(version))

        vocab_path = self._vocab_local_path(version)
        vocab_path.write_text("\n".join(vocab))
        s3.upload(vocab_path, _BUCKET, self._vocab_s3_key(version))

        ids_path = self._ids_local_path(version)
        ids_path.write_text(json.dumps(story_ids))
        s3.upload(ids_path, _BUCKET, self._ids_s3_key(version))

        logger.info(
            "Saved sparse_tfidf v%d (%d x %d, %d nnz) + vocab + %d ids",
            version,
            n_rows,
            n_cols,
            matrix.nnz,
            len(story_ids),
        )

    def _matrix_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"tfidf_v{version}.npz"

    def _vocab_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"vocab_v{version}.txt"

    def _ids_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"story_ids_v{version}.json"

    def _matrix_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/tfidf_v{version}.npz"

    def _vocab_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/vocab_v{version}.txt"

    def _ids_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/story_ids_v{version}.json"


cache = SparseTfidfCache()
