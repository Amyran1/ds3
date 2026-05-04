"""Sparse BM25 vectors cache for the full news story corpus.

Three-artifact cache (v1) / five-artifact cache (v2+) holding a (~N x K)
L2-normalized BM25 sparse matrix produced by
``pinecone_text.sparse.BM25Encoder``, alongside encoder hyperparameters and
a row-ordered ``story_id`` side-car.

Unlike the TF-IDF cache, BM25 vocabulary is hashed (not a human-readable
token list). Columns in the stored matrix correspond to hashed term ids
that have been compacted to a dense ``[0, K)`` range; ``n_cols`` in the
params captures ``K``.

v1 artifacts:
    bm25_v1.npz            - scipy sparse CSR matrix (L2-normalized)
    params_v1.json         - encoder hyperparameters (see required keys)
    story_ids_v1.json      - list of story_ids, row order matches matrix

v2+ additional artifacts:
    encoder_v{N}.pkl       - pickled fitted BM25Encoder; enables query encoding
    idx_map_v{N}.json      - JSON mapping str(hash_index) -> compact_col_index

Required ``params`` keys:
    encoder_class, k1, b, n_docs, n_cols, nnz

Optional ``params`` keys:
    nltk_data_version, encoder_pkl_size_bytes, idx_map_len
    (or any other producer-side provenance fields)

Usage::

    from entities import news_stories_graph

    matrix = news_stories_graph.bm25_vectors_cache.cache.get(1)
    params = news_stories_graph.bm25_vectors_cache.cache.get_params(1)
    ids = news_stories_graph.bm25_vectors_cache.cache.get_story_ids(1)

    # v2+ only:
    encoder = news_stories_graph.bm25_vectors_cache.cache.get_encoder(2)
    idx_map = news_stories_graph.bm25_vectors_cache.cache.get_idx_map(2)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, cast

from scipy.sparse import csr_matrix, load_npz, save_npz

from libs.cache import s3

if TYPE_CHECKING:
    from pinecone_text.sparse import BM25Encoder

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(
    "data/entities/news_stories_graph/bm25_vectors",
)
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/news_stories_graph/bm25_vectors"
)
_BUCKET = "chorus-content-assets"

_REQUIRED_PARAM_KEYS: frozenset[str] = frozenset(
    {"encoder_class", "k1", "b", "n_docs", "n_cols", "nnz"},
)


class Bm25VectorsCache:
    """Local + S3 cache for the news story BM25 matrix + side-cars."""

    def get(self, version: int = 1) -> csr_matrix:
        """Return the BM25 matrix as a CSR, pulling from S3 if needed."""
        local_path = self._matrix_local_path(version)
        if not local_path.exists():
            logger.info("Cache miss: downloading bm25 v%d from S3", version)
            s3.download(_BUCKET, self._matrix_s3_key(version), local_path)
        else:
            logger.debug("Cache hit: bm25 v%d", version)
        matrix = load_npz(local_path)
        return cast("csr_matrix", matrix.tocsr())

    def get_params(self, version: int = 1) -> dict[str, object]:
        """Return the encoder hyperparameters for this version."""
        local_path = self._params_local_path(version)
        if not local_path.exists():
            logger.info(
                "Cache miss: downloading bm25 params v%d from S3",
                version,
            )
            s3.download(_BUCKET, self._params_s3_key(version), local_path)
        return cast("dict[str, object]", json.loads(local_path.read_text()))

    def get_story_ids(self, version: int = 1) -> list[str]:
        """Return the row-ordered list of story_ids for this version."""
        local_path = self._ids_local_path(version)
        if not local_path.exists():
            logger.info(
                "Cache miss: downloading bm25 story_ids v%d from S3",
                version,
            )
            s3.download(_BUCKET, self._ids_s3_key(version), local_path)
        return [str(sid) for sid in json.loads(local_path.read_text())]

    def get_encoder(self, version: int = 2) -> BM25Encoder:
        """Return the fitted BM25Encoder for ``version`` (v2+).

        The encoder pickle may be 50-200 MB. Callers should cache the result
        in memory across calls (the ``_Caches`` lazy-load layer handles this).

        Raises ``FileNotFoundError`` if the encoder artifact was not persisted
        (i.e. the version was built before v2 support was added).
        """
        local_path = self._encoder_local_path(version)
        if not local_path.exists():
            logger.info(
                "Cache miss: downloading bm25 encoder v%d from S3",
                version,
            )
            s3.download(_BUCKET, self._encoder_s3_key(version), local_path)
        logger.info("Loading BM25Encoder pickle v%d", version)
        encoder = pickle.loads(local_path.read_bytes())  # noqa: S301
        return cast("BM25Encoder", encoder)

    def get_idx_map(self, version: int = 2) -> dict[int, int]:
        """Return the hash-index → compact-column mapping for ``version`` (v2+).

        Keys are the raw encoder hash indices (integers stored as JSON string
        keys); values are compact column indices ``[0, n_cols)``.

        Raises ``FileNotFoundError`` if the idx_map artifact was not persisted.
        """
        local_path = self._idx_map_local_path(version)
        if not local_path.exists():
            logger.info(
                "Cache miss: downloading bm25 idx_map v%d from S3",
                version,
            )
            s3.download(_BUCKET, self._idx_map_s3_key(version), local_path)
        raw: dict[str, int] = json.loads(local_path.read_text())
        return {int(k): v for k, v in raw.items()}

    def put(
        self,
        matrix: csr_matrix,
        params: dict[str, object],
        story_ids: list[str],
        version: int = 1,
        *,
        encoder: BM25Encoder | None = None,
        idx_map: dict[int, int] | None = None,
    ) -> None:
        """Persist matrix, params, and story_ids locally and to S3.

        ``encoder`` and ``idx_map`` are optional keyword-only arguments added
        in v2.  Passing them persists the encoder pickle and idx_map JSON
        alongside the existing three artifacts.  Omitting them (default) is
        backward-compatible with v1 builds.
        """
        shape = cast("tuple[int, int]", matrix.shape)
        n_rows, _n_cols = shape
        if len(story_ids) != n_rows:
            got = len(story_ids)
            msg = f"story_ids length ({got}) does not match matrix row count ({n_rows})"
            raise ValueError(msg)

        missing = _REQUIRED_PARAM_KEYS - set(params.keys())
        if missing:
            msg = f"params is missing required keys: {sorted(missing)}"
            raise ValueError(msg)

        matrix_path = self._matrix_local_path(version)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(matrix_path, matrix)
        s3.upload(matrix_path, _BUCKET, self._matrix_s3_key(version))

        params_path = self._params_local_path(version)
        params_path.write_text(json.dumps(params))
        s3.upload(params_path, _BUCKET, self._params_s3_key(version))

        ids_path = self._ids_local_path(version)
        ids_path.write_text(json.dumps(story_ids))
        s3.upload(ids_path, _BUCKET, self._ids_s3_key(version))

        if encoder is not None:
            encoder_path = self._encoder_local_path(version)
            encoder_bytes = pickle.dumps(encoder, protocol=pickle.HIGHEST_PROTOCOL)
            encoder_path.write_bytes(encoder_bytes)
            logger.info(
                "Uploading encoder pickle v%d (%.1f MB) to S3",
                version,
                len(encoder_bytes) / 1_048_576,
            )
            s3.upload(encoder_path, _BUCKET, self._encoder_s3_key(version))

        if idx_map is not None:
            idx_map_path = self._idx_map_local_path(version)
            # JSON requires string keys; store int keys as str.
            idx_map_path.write_text(
                json.dumps({str(k): v for k, v in idx_map.items()}),
            )
            s3.upload(idx_map_path, _BUCKET, self._idx_map_s3_key(version))

        logger.info(
            "Saved bm25_vectors v%d (%d x %d, %d nnz) + params + %d ids%s%s",
            version,
            n_rows,
            shape[1],
            matrix.nnz,
            len(story_ids),
            " + encoder" if encoder is not None else "",
            " + idx_map" if idx_map is not None else "",
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _matrix_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"bm25_v{version}.npz"

    def _params_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"params_v{version}.json"

    def _ids_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"story_ids_v{version}.json"

    def _encoder_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"encoder_v{version}.pkl"

    def _idx_map_local_path(self, version: int) -> Path:
        return _CACHE_DIR / f"idx_map_v{version}.json"

    def _matrix_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/bm25_v{version}.npz"

    def _params_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/params_v{version}.json"

    def _ids_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/story_ids_v{version}.json"

    def _encoder_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/encoder_v{version}.pkl"

    def _idx_map_s3_key(self, version: int) -> str:
        return f"{_S3_PREFIX}/idx_map_v{version}.json"


cache = Bm25VectorsCache()
