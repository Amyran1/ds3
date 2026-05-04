"""Lazy cache loaders for the StoryGraphAPI.

Each property on :class:`_Caches` pulls the underlying artifact from the
``news_stories_graph`` entity caches on first access and memoizes the result.
Instantiation of :class:`_Caches` is cheap (no disk IO). Every method on
:class:`StoryGraphAPI` triggers exactly the loads it needs.

Tradeoffs documented here (see also the class docstring on ``StoryGraphAPI``):

* **TF-IDF query vectorizer** — reconstructed from the cached vocabulary only
  (``sklearn.feature_extraction.text.TfidfVectorizer`` with
  ``vocabulary=<cached dict>``). The build-time IDF weights are NOT persisted,
  so query-side vectors use L2-normalized TF counts rather than true TF-IDF.
  For short, generic queries this is a small approximation; if exact build-time
  IDF is required, pass a csr row from the cached matrix instead of calling
  :meth:`StoryGraphAPI.tfidf_encode_text`. Refitting a full TfidfVectorizer on
  all 3.5M corpus texts (the exact match) costs ~10 min and ~8 GB of memory
  and is intentionally NOT done here.

* **BM25 query encoder (v1)** — ``pinecone_text.sparse.BM25Encoder`` can be
  refit on the same 3.5M text corpus with identical IDF weights, but the
  build-time ``idx_map`` (raw-hash to compacted-column) was NOT persisted in
  v1. This means a freshly-refit encoder's column ids do NOT line up with the
  stored matrix columns, so query encoding is not directly usable against the
  cached BM25 v1 matrix. Consequently :meth:`StoryGraphAPI.bm25_encode_text`
  raises ``NotImplementedError`` when constructed with ``version=1``.

* **BM25 query encoder (v2+)** — v2 persists both the fitted encoder pickle
  (``encoder_v2.pkl``) and the ``idx_map`` JSON (``idx_map_v2.json``). When
  ``version >= 2``, :attr:`bm25_encoder` and :attr:`bm25_idx_map` are
  available. The encoder may be 50-200 MB; it is loaded lazily and memoized
  on first access so subsequent calls to :meth:`bm25_encode_text` are cheap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from sklearn.feature_extraction.text import TfidfVectorizer

from entities.news_stories.cache import cache as _stories_cache
from entities.news_stories_graph import (
    bipartite_edges_cache as _bipartite_edges_cache,
)
from entities.news_stories_graph import (
    bm25_vectors_cache as _bm25_vectors_cache,
)
from entities.news_stories_graph import (
    dense_edges_cache as _dense_edges_cache,
)
from entities.news_stories_graph import (
    dense_embeddings_cache as _dense_embeddings_cache,
)
from entities.news_stories_graph import (
    dense_faiss_index_cache as _dense_faiss_index_cache,
)
from entities.news_stories_graph import (
    sparse_tfidf_cache as _sparse_tfidf_cache,
)

_bip = _bipartite_edges_cache.cache
_den = _dense_embeddings_cache.cache
_idx = _dense_faiss_index_cache.cache
_dedge = _dense_edges_cache.cache
_tfidf = _sparse_tfidf_cache.cache
_bm25 = _bm25_vectors_cache.cache

if TYPE_CHECKING:
    import faiss
    import numpy as np
    import polars as pl
    from pinecone_text.sparse import BM25Encoder
    from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

_STORY_METADATA_COLS: tuple[str, ...] = (
    "story_id",
    "created_at",
    "name",
    "summary",
    "article_count",
    "source_reach_total",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "source_bias_set",
)


class _Caches:
    """Lazy-load container for all news_stories_graph cache artifacts.

    Each attribute is ``None`` until first access via its ``@property``.
    Instances are not thread-safe on the first load; once loaded, all reads
    are thread-safe (numpy / polars / faiss are read-threadsafe).
    """

    def __init__(self, version: int = 1) -> None:
        self._version = version
        self._bipartite: pl.DataFrame | None = None
        self._dense_vectors: np.ndarray | None = None
        self._dense_story_ids: list[str] | None = None
        self._dense_story_id_to_idx: dict[str, int] | None = None
        self._dense_index: Any | None = None
        self._dense_edges: pl.DataFrame | None = None
        self._tfidf_matrix: csr_matrix | None = None
        self._tfidf_vocab: list[str] | None = None
        self._tfidf_story_ids: list[str] | None = None
        self._tfidf_story_id_to_idx: dict[str, int] | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._bm25_matrix: csr_matrix | None = None
        self._bm25_params: dict[str, object] | None = None
        self._bm25_story_ids: list[str] | None = None
        self._bm25_story_id_to_idx: dict[str, int] | None = None
        self._bm25_encoder: BM25Encoder | None = None
        self._bm25_idx_map: dict[int, int] | None = None
        self._story_metadata: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Bipartite
    # ------------------------------------------------------------------

    @property
    def bipartite(self) -> pl.DataFrame:
        if self._bipartite is None:
            logger.info("Loading bipartite_edges v%d", self._version)
            self._bipartite = _bip.get(self._version)
        return self._bipartite

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    @property
    def dense_vectors(self) -> np.ndarray:
        if self._dense_vectors is None:
            logger.info("Loading dense_embeddings v%d", self._version)
            self._dense_vectors = _den.get(self._version)
        return self._dense_vectors

    @property
    def dense_story_ids(self) -> list[str]:
        if self._dense_story_ids is None:
            logger.info("Loading dense_embeddings story_ids v%d", self._version)
            self._dense_story_ids = _den.get_story_ids(self._version)
        return self._dense_story_ids

    @property
    def dense_story_id_to_idx(self) -> dict[str, int]:
        if self._dense_story_id_to_idx is None:
            ids = self.dense_story_ids
            self._dense_story_id_to_idx = {s: i for i, s in enumerate(ids)}
        return self._dense_story_id_to_idx

    @property
    def dense_index(self) -> faiss.Index:
        if self._dense_index is None:
            logger.info("Loading dense_faiss_index v%d", self._version)
            self._dense_index = _idx.get(self._version)
        return cast("faiss.Index", self._dense_index)

    @property
    def dense_edges(self) -> pl.DataFrame:
        if self._dense_edges is None:
            logger.info("Loading dense_edges v%d", self._version)
            self._dense_edges = _dedge.get(self._version)
        return self._dense_edges

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------

    @property
    def tfidf_matrix(self) -> csr_matrix:
        if self._tfidf_matrix is None:
            logger.info("Loading sparse_tfidf v%d", self._version)
            self._tfidf_matrix = _tfidf.get(self._version)
        return self._tfidf_matrix

    @property
    def tfidf_vocab(self) -> list[str]:
        if self._tfidf_vocab is None:
            logger.info("Loading tfidf vocab v%d", self._version)
            self._tfidf_vocab = _tfidf.get_vocab(self._version)
        return self._tfidf_vocab

    @property
    def tfidf_story_ids(self) -> list[str]:
        if self._tfidf_story_ids is None:
            logger.info("Loading tfidf story_ids v%d", self._version)
            self._tfidf_story_ids = _tfidf.get_story_ids(self._version)
        return self._tfidf_story_ids

    @property
    def tfidf_story_id_to_idx(self) -> dict[str, int]:
        if self._tfidf_story_id_to_idx is None:
            ids = self.tfidf_story_ids
            self._tfidf_story_id_to_idx = {s: i for i, s in enumerate(ids)}
        return self._tfidf_story_id_to_idx

    @property
    def tfidf_vectorizer(self) -> TfidfVectorizer:
        if self._tfidf_vectorizer is None:
            vocab = self.tfidf_vocab
            vocab_dict = {tok: i for i, tok in enumerate(vocab)}
            vectorizer = TfidfVectorizer(
                vocabulary=vocab_dict,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
                norm="l2",
            )
            # Fit on a dummy doc so internal state is query-ready.
            vectorizer.fit(["dummy document for initialization"])
            self._tfidf_vectorizer = vectorizer
        return self._tfidf_vectorizer

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    @property
    def bm25_matrix(self) -> csr_matrix:
        if self._bm25_matrix is None:
            logger.info("Loading bm25_vectors v%d", self._version)
            self._bm25_matrix = _bm25.get(self._version)
        return self._bm25_matrix

    @property
    def bm25_params(self) -> dict[str, object]:
        if self._bm25_params is None:
            logger.info("Loading bm25 params v%d", self._version)
            self._bm25_params = _bm25.get_params(self._version)
        return self._bm25_params

    @property
    def bm25_story_ids(self) -> list[str]:
        if self._bm25_story_ids is None:
            logger.info("Loading bm25 story_ids v%d", self._version)
            self._bm25_story_ids = _bm25.get_story_ids(self._version)
        return self._bm25_story_ids

    @property
    def bm25_story_id_to_idx(self) -> dict[str, int]:
        if self._bm25_story_id_to_idx is None:
            ids = self.bm25_story_ids
            self._bm25_story_id_to_idx = {s: i for i, s in enumerate(ids)}
        return self._bm25_story_id_to_idx

    @property
    def bm25_encoder(self) -> BM25Encoder:
        """Fitted BM25Encoder (v2+). Loaded lazily and memoized.

        Raises ``FileNotFoundError`` if the encoder artifact is not present
        (i.e. this instance was constructed with ``version=1``).
        """
        if self._bm25_encoder is None:
            logger.info("Loading BM25Encoder pickle v%d", self._version)
            self._bm25_encoder = _bm25.get_encoder(self._version)
        return self._bm25_encoder

    @property
    def bm25_idx_map(self) -> dict[int, int]:
        """Hash-index → compact-column mapping (v2+). Loaded lazily and memoized.

        Raises ``FileNotFoundError`` if the idx_map artifact is not present.
        """
        if self._bm25_idx_map is None:
            logger.info("Loading BM25 idx_map v%d", self._version)
            self._bm25_idx_map = _bm25.get_idx_map(self._version)
        return self._bm25_idx_map

    @property
    def bm25_n_cols(self) -> int:
        """Number of compact columns in the BM25 matrix (from params)."""
        n_cols = self.bm25_params.get("n_cols")
        if not isinstance(n_cols, int):
            msg = f"bm25 params n_cols is not an int: {n_cols!r}"
            raise TypeError(msg)
        return n_cols

    # ------------------------------------------------------------------
    # Story metadata
    # ------------------------------------------------------------------

    @property
    def story_metadata(self) -> pl.DataFrame:
        if self._story_metadata is None:
            logger.info("Loading news_stories metadata (projected columns)")
            df = _stories_cache.get(self._version)
            self._story_metadata = df.select(list(_STORY_METADATA_COLS))
        return self._story_metadata
