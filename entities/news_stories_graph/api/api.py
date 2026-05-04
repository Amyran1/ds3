"""Story-graph retrieval API over the news_stories_graph entity.

``StoryGraphAPI`` exposes four projection layers over the full news story
corpus (3.5M stories as of v1):

* bipartite attribute edges (materialized; entity-share queries)
* dense embedding cosine via a trained FAISS IVF index
* TF-IDF cosine (matrix only; live query-time sparse matmul)
* BM25 cosine (matrix only; live query-time sparse matmul)

Plus a hybrid ``related_stories_quad`` that fuses all four layers and a batch
``batch_related_stories_sparse`` for downstream use cases.

**Lazy loads** — construction is cheap. Every method triggers only the cache
artifacts it needs. Typical cold start for a single dense query is dominated
by the FAISS index read (~60 s for 20 GB). Callers that need multiple layers
should keep a single ``StoryGraphAPI`` instance alive across calls.

**Limitations** (see ``loaders.py`` for the full rationale):

* :meth:`tfidf_encode_text` uses a vocab-only vectorizer without the build-time
  IDF weights — a small approximation acceptable for short queries. For exact
  matching against the cached matrix, pass a row of the cached matrix directly.
* :meth:`bm25_encode_text` requires ``version >= 2``. v1 did not persist the
  fitted encoder or the hash→column ``idx_map``. Construct with
  ``StoryGraphAPI(version=2)`` (or higher) to use live BM25 query encoding.
* ``dense_encode_text()`` runs a live ``text-embedding-3-small`` call
  via ``asyncio.run()``. Callers already inside an async event loop
  must use the private ``_dense_encode_text_async()`` coroutine
  instead.

**Dropped methods** (email-pipeline specific, not available in this repo):
* ``related_to_email`` — requires ``entities.emails.cache`` (not present).
* Eight scalar email-feature helpers (``ambient_relevance`` through
  ``general_news_volume``) — all wrap ``related_to_email``-style email context
  and are not needed for story-graph EDA.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

from entities.news_stories_graph.api.filters import (
    build_date_mask,
    filter_metadata_by_date,
    join_metadata,
)
from entities.news_stories_graph.api.loaders import _Caches
from libs.clients.openai import OpenAIClient
from libs.settings import Settings

logger = logging.getLogger(__name__)

_DEFAULT_FAISS_NPROBE = 16
# Minimum version that persists the fitted BM25Encoder and idx_map artifacts.
_BM25_MIN_VERSION = 2
_BIAS_TO_INT: dict[str, int] = {
    "left": -1,
    "center": 0,
    "right": 1,
}


class StoryGraphAPI:
    """Story-graph retrieval API over the news_stories_graph entity.

    Known limitations:
        - ``bm25_encode_text()`` requires ``version >= 2``. v1 did not
          persist the fitted encoder or the hash→column idx_map. Construct
          with ``StoryGraphAPI(version=2)`` to use live BM25 query encoding.
        - ``tfidf_encode_text()`` uses a vocab-only approximation (no
          build-time IDF weights). Short queries are close to exact; long
          queries drift. Use the exact-match path (raw matrix rows) for
          callers needing bit-level correctness.
        - ``dense_encode_text()`` runs a live ``text-embedding-3-small`` call
          via ``asyncio.run()``. Callers already inside an async event loop
          must use the private ``_dense_encode_text_async()`` coroutine
          instead.
    """

    def __init__(self, version: int = 1) -> None:
        self._version = version
        self._caches = _Caches(version=version)

    @classmethod
    def from_cache(cls, version: int = 1) -> StoryGraphAPI:
        """Construct a read-only API over the given graph version."""
        return cls(version=version)

    # ==================================================================
    # Core single-query similarity methods
    # ==================================================================

    def related_stories_dense(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        min_similarity: float = 0.50,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        """Return top-k dense-similar stories via the FAISS IVF index.

        ``query_embedding`` is a (1536,) or (1, 1536) L2-normalized float32.
        Date filtering is applied after the FAISS search; the search widens
        to ``max(k * 5, 2000)`` candidates to give the post-filter room to
        still return ``k`` rows after a 7-day date window reduces the pool.

        .. note:: The v1 FAISS index was built with ``IndexIVFFlat`` (no
            explicit ``metric_type``), which defaults to ``METRIC_L2``.  The
            raw ``distances`` values are therefore **squared L2 distances**,
            not inner products.  For unit-normalized vectors the exact cosine
            conversion is ``cos(θ) = 1 - L2² / 2``, applied here before the
            ``min_similarity`` threshold and before the ``similarity`` column
            is populated.  This layer correction replaces the previous
            (incorrect) treatment of raw L2² as cosine similarity.
        """
        q = _as_row_vector(query_embedding, dim=1536).astype(np.float32, copy=False)

        index = self._caches.dense_index
        _faiss_any = cast("Any", index)
        if getattr(index, "nprobe", None) != _DEFAULT_FAISS_NPROBE:
            with contextlib.suppress(AttributeError):
                _faiss_any.nprobe = _DEFAULT_FAISS_NPROBE

        # IndexIVFFlat returns squared L2 distances; vectors are unit-norm so
        # cos(θ) = 1 - L2² / 2. search_k widened so date-window post-filtering
        # still leaves a meaningful candidate pool per 7-day window.
        search_k = max(k * 5, 2000)
        distances, indices = _faiss_any.search(q, search_k)
        l2_sq = np.asarray(distances[0], dtype=np.float32)
        sims = 1.0 - l2_sq / 2.0
        idxs = np.asarray(indices[0], dtype=np.int64)

        valid = (idxs >= 0) & (sims >= min_similarity)
        sims = sims[valid]
        idxs = idxs[valid]
        if idxs.size == 0:
            return _empty_result_frame()

        story_ids_all = self._caches.dense_story_ids
        hit_story_ids = [story_ids_all[int(i)] for i in idxs]
        hits = pl.DataFrame(
            {
                "story_id": hit_story_ids,
                "similarity": sims.astype(np.float32),
            },
        )

        joined = join_metadata(hits, self._caches.story_metadata)
        joined = filter_metadata_by_date(joined, date_range)
        return joined.sort("similarity", descending=True).head(k)

    def related_stories_tfidf(
        self,
        query_sparse: csr_matrix,
        k: int = 20,
        min_similarity: float = 0.25,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        """Return top-k TF-IDF cosine-similar stories via a live sparse matmul."""
        return self._sparse_single_query(
            query_sparse=query_sparse,
            story_matrix=self._caches.tfidf_matrix,
            story_ids=self._caches.tfidf_story_ids,
            k=k,
            min_similarity=min_similarity,
            date_range=date_range,
        )

    def related_stories_bm25(
        self,
        query_sparse: csr_matrix | None = None,
        k: int = 20,
        min_similarity: float = 0.30,
        date_range: tuple[datetime, datetime] | None = None,
        *,
        query_text: str | None = None,
    ) -> pl.DataFrame:
        """Return top-k BM25 cosine-similar stories via a live sparse matmul.

        Accepts either a pre-encoded sparse query vector (``query_sparse``) or
        a raw text string (``query_text``). Exactly one must be provided.

        When ``query_text`` is given the text is encoded on-the-fly via
        :meth:`bm25_encode_text`, which requires ``version >= 2``.

        Args:
            query_sparse: A ``(1, n_cols)`` CSR matrix aligned to the BM25
                column space. Mutually exclusive with ``query_text``.
            k: Number of top results to return.
            min_similarity: Minimum cosine similarity threshold.
            date_range: Optional ``(start, end)`` datetime filter.
            query_text: Raw text to encode and query. Requires ``version >= 2``.
                Mutually exclusive with ``query_sparse``.

        Returns:
            DataFrame of top-k matching stories.
        """
        if query_text is not None and query_sparse is not None:
            msg = "Provide either query_text or query_sparse, not both."
            raise ValueError(msg)
        if query_text is None and query_sparse is None:
            msg = "One of query_text or query_sparse must be provided."
            raise ValueError(msg)

        if query_text is not None:
            query_sparse = self.bm25_encode_text(query_text)

        return self._sparse_single_query(
            query_sparse=cast("csr_matrix", query_sparse),
            story_matrix=self._caches.bm25_matrix,
            story_ids=self._caches.bm25_story_ids,
            k=k,
            min_similarity=min_similarity,
            date_range=date_range,
        )

    def related_stories_by_attribute(
        self,
        story_id: str,
        attribute_types: list[str] | None = None,
        min_shared: int = 2,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        """Return other stories sharing >= min_shared attrs with story_id."""
        if attribute_types is None:
            attribute_types = ["people", "companies"]

        bipartite = self._caches.bipartite
        type_mask = pl.col("attribute_type").is_in(attribute_types)
        anchor = bipartite.filter(
            (pl.col("story_id") == story_id) & type_mask,
        ).select(["attribute_type", "value"])

        if anchor.height == 0:
            return _empty_result_frame()

        candidates = (
            bipartite.filter(pl.col("attribute_type").is_in(attribute_types))
            .join(anchor, on=["attribute_type", "value"], how="inner")
            .filter(pl.col("story_id") != story_id)
            .group_by("story_id")
            .agg(pl.len().alias("shared_attributes"))
            .filter(pl.col("shared_attributes") >= min_shared)
            .rename({"shared_attributes": "similarity"})
            .with_columns(pl.col("similarity").cast(pl.Float32))
        )

        if candidates.height == 0:
            return _empty_result_frame()

        joined = join_metadata(candidates, self._caches.story_metadata)
        joined = filter_metadata_by_date(joined, date_range)
        return joined.sort("similarity", descending=True)

    def related_stories_quad(
        self,
        dense_query: np.ndarray,
        tfidf_query: csr_matrix,
        bm25_query: csr_matrix,
        weights: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20),
        attribute_anchor_story_id: str | None = None,
        k: int = 20,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        """Weighted hybrid across all four layers.

        Each layer retrieves ``k * 5`` candidates for a wider union; scores
        are min-max normalized within the candidate set and combined via
        ``weights`` = (attribute, dense, tfidf, bm25). If
        ``attribute_anchor_story_id`` is None, the attribute weight is
        redistributed proportionally to the remaining three.
        """
        w_attr, w_dense, w_tfidf, w_bm25 = weights
        if attribute_anchor_story_id is None:
            rest = w_dense + w_tfidf + w_bm25
            if rest <= 0:
                w_dense, w_tfidf, w_bm25 = 1 / 3, 1 / 3, 1 / 3
            else:
                scale = (w_attr + rest) / rest
                w_dense *= scale
                w_tfidf *= scale
                w_bm25 *= scale
            w_attr = 0.0

        wide_k = k * 5
        frames: list[pl.DataFrame] = []

        dense_hits = self.related_stories_dense(
            dense_query,
            k=wide_k,
            min_similarity=-1.0,
            date_range=date_range,
        )
        frames.append(_rescore_for_quad(dense_hits, "dense", w_dense))

        tfidf_hits = self.related_stories_tfidf(
            tfidf_query,
            k=wide_k,
            min_similarity=0.0,
            date_range=date_range,
        )
        frames.append(_rescore_for_quad(tfidf_hits, "tfidf", w_tfidf))

        bm25_hits = self.related_stories_bm25(
            bm25_query,
            k=wide_k,
            min_similarity=0.0,
            date_range=date_range,
        )
        frames.append(_rescore_for_quad(bm25_hits, "bm25", w_bm25))

        if attribute_anchor_story_id is not None and w_attr > 0:
            attr_hits = self.related_stories_by_attribute(
                attribute_anchor_story_id,
                min_shared=1,
                date_range=date_range,
            ).head(wide_k)
            frames.append(_rescore_for_quad(attr_hits, "attribute", w_attr))

        combined = _combine_quad_scores(frames)
        if combined.height == 0:
            return _empty_result_frame()

        joined = join_metadata(
            combined.select(["story_id", "similarity"]),
            self._caches.story_metadata,
        )
        return joined.sort("similarity", descending=True).head(k)

    # ==================================================================
    # Bulk batch method
    # ==================================================================

    def batch_related_stories_sparse(
        self,
        query_matrix: csr_matrix,
        method: Literal["tfidf", "bm25"] = "tfidf",
        k: int = 20,
        min_similarity: float | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        """Batch top-k retrieval via a single M x N sparse matmul.

        ``query_matrix`` is ``(M, n_cols)`` matching the chosen method's
        vocabulary. Before the matmul the story matrix is row-sliced by
        ``date_range`` (if provided) — this is the key optimization for the
        email panel use case.
        """
        if method == "tfidf":
            story_matrix = self._caches.tfidf_matrix
            story_ids = self._caches.tfidf_story_ids
            default_min = 0.25
        else:
            story_matrix = self._caches.bm25_matrix
            story_ids = self._caches.bm25_story_ids
            default_min = 0.30
        threshold = default_min if min_similarity is None else min_similarity

        mask = build_date_mask(
            self._caches.story_metadata,
            story_ids,
            date_range,
        )
        keep_idx = np.nonzero(mask)[0]
        if keep_idx.size == 0:
            return _empty_batch_frame()

        sliced = story_matrix[keep_idx, :]
        sims_block = cast("Any", query_matrix @ sliced.T)
        sims_dense = np.asarray(sims_block.todense())

        m = cast("tuple[int, int]", query_matrix.shape)[0]
        all_query_rows: list[int] = []
        all_story_ids: list[str] = []
        all_sims: list[float] = []

        effective_k = min(k, sims_dense.shape[1])
        for row_i in range(m):
            row = sims_dense[row_i]
            if effective_k <= 0:
                continue
            if effective_k >= row.size:
                top_local = np.argsort(-row)
            else:
                part = np.argpartition(-row, effective_k - 1)[:effective_k]
                top_local = part[np.argsort(-row[part])]
            top_sims = row[top_local]
            keep = top_sims >= threshold
            top_local = top_local[keep]
            top_sims = top_sims[keep]
            for local_i, sim in zip(top_local.tolist(), top_sims.tolist(), strict=True):
                all_query_rows.append(row_i)
                all_story_ids.append(story_ids[int(keep_idx[local_i])])
                all_sims.append(float(sim))

        if not all_story_ids:
            return _empty_batch_frame()

        out = pl.DataFrame(
            {
                "query_row": all_query_rows,
                "story_id": all_story_ids,
                "similarity": [np.float32(s) for s in all_sims],
            },
        )
        metadata = self._caches.story_metadata.select(
            [
                "story_id",
                "created_at",
                "name",
                "summary",
                "article_count",
                "source_reach_total",
            ],
        )
        return out.join(metadata, on="story_id", how="left").sort(
            ["query_row", "similarity"],
            descending=[False, True],
        )

    # ==================================================================
    # Encoder helpers
    # ==================================================================

    def dense_encode_text(self, text: str) -> np.ndarray:
        """Embed a single text via OpenAI text-embedding-3-small.

        Returns a ``(1536,)`` float32 L2-normalized vector. Uses the existing
        ``OpenAIClient`` wrapper which handles retry and rate-limit backoff.
        Cost is tracked via ``cost_key='news-stories-graph-api-queries'``.

        This is a live API call — typical latency 100-300 ms. Callers that
        need to embed many texts should batch them externally. If the caller
        is already inside an async event loop, use
        :meth:`_dense_encode_text_async` directly instead of this method.
        """
        return asyncio.run(self._dense_encode_text_async(text))

    async def _dense_encode_text_async(self, text: str) -> np.ndarray:
        """Async variant of :meth:`dense_encode_text` for async callers."""
        settings = Settings()
        async with OpenAIClient(
            api_key=settings.openai_api_key.get_secret_value(),
            max_concurrent=1,
        ) as client:
            result = await client.embed(
                [text],
                model="text-embedding-3-small",
                batch_size=1,
                cost_key="news-stories-graph-api-queries",
            )
        vec = np.asarray(result[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def tfidf_encode_text(self, text: str) -> csr_matrix:
        """Encode ``text`` via the vocab-only query-side TfidfVectorizer.

        Uses L2-normalized raw TF counts (no IDF weights) — see
        ``loaders.py`` for the rationale. Returns a ``(1, 50_000)`` csr_matrix.
        """
        vectorizer = self._caches.tfidf_vectorizer
        vec = cast("Any", vectorizer.transform([text]))
        return cast("csr_matrix", vec.tocsr())

    def bm25_encode_text(self, text: str) -> csr_matrix:
        """Encode ``text`` into the BM25 column space for query-time similarity.

        Requires ``version >= 2``. The fitted encoder and hash→column idx_map
        are loaded lazily on the first call (encoder may be 50-200 MB) and
        memoized for subsequent calls.

        Returns a ``(1, n_cols)`` L2-normalized CSR matrix aligned to the
        stored BM25 matrix column space. Out-of-vocabulary hashes (terms not
        seen during corpus fit) are silently dropped. If no in-vocabulary
        terms remain, returns a zero row.

        Raises:
            NotImplementedError: If constructed with ``version=1``, which did
                not persist the encoder or idx_map.
            FileNotFoundError: If the encoder/idx_map artifacts are missing
                from both local cache and S3.
        """
        if self._version < _BM25_MIN_VERSION:
            msg = (
                f"bm25_encode_text is not supported for version < {_BM25_MIN_VERSION}. "
                "The build-time hash-to-column idx_map was not persisted in "
                "v1. Construct StoryGraphAPI(version=2) to use live BM25 "
                "query encoding."
            )
            raise NotImplementedError(msg)

        encoder = self._caches.bm25_encoder
        idx_map = self._caches.bm25_idx_map
        n_cols = self._caches.bm25_n_cols

        encoded_list = cast(
            "list[dict[str, list[int] | list[float]]]",
            encoder.encode_documents([text]),
        )
        encoded = encoded_list[0]
        raw_indices = cast("list[int]", encoded["indices"])
        raw_values = cast("list[float]", encoded["values"])

        cols: list[int] = []
        vals: list[float] = []
        for h_idx, val in zip(raw_indices, raw_values, strict=True):
            compact = idx_map.get(h_idx)
            if compact is not None:
                cols.append(compact)
                vals.append(val)

        if not cols:
            return csr_matrix((1, n_cols), dtype=np.float32)

        row_vec = csr_matrix(
            (vals, ([0] * len(cols), cols)),
            shape=(1, n_cols),
            dtype=np.float64,
        )
        normalized = cast(
            "csr_matrix",
            sk_normalize(row_vec, norm="l2", axis=1),
        )
        return normalized.tocsr().astype(np.float32)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _sparse_single_query(
        self,
        query_sparse: csr_matrix,
        story_matrix: csr_matrix,
        story_ids: list[str],
        k: int,
        min_similarity: float,
        date_range: tuple[datetime, datetime] | None,
    ) -> pl.DataFrame:
        """Live single-query sparse matmul with date-window pre-slicing."""
        mask = build_date_mask(
            self._caches.story_metadata,
            story_ids,
            date_range,
        )
        keep_idx = np.nonzero(mask)[0]
        if keep_idx.size == 0:
            return _empty_result_frame()

        sliced = story_matrix[keep_idx, :]
        sims_block = cast("Any", sliced @ query_sparse.T)
        sims_dense = np.asarray(sims_block.todense()).ravel()

        above = np.nonzero(sims_dense >= min_similarity)[0]
        if above.size == 0:
            return _empty_result_frame()

        effective_k = min(k, above.size)
        cand_scores = sims_dense[above]
        if effective_k >= above.size:
            order = np.argsort(-cand_scores)
        else:
            part = np.argpartition(-cand_scores, effective_k - 1)[:effective_k]
            order = part[np.argsort(-cand_scores[part])]

        top_local = above[order]
        top_sims = sims_dense[top_local]
        hit_story_ids = [story_ids[int(keep_idx[i])] for i in top_local]

        hits = pl.DataFrame(
            {
                "story_id": hit_story_ids,
                "similarity": top_sims.astype(np.float32),
            },
        )
        return join_metadata(hits, self._caches.story_metadata)


# ======================================================================
# Module-level helpers
# ======================================================================


def _as_row_vector(arr: np.ndarray, dim: int) -> np.ndarray:
    """Reshape a (dim,) or (1, dim) array to a (1, dim) float32 row vector."""
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape != (1, dim):
        msg = f"Expected shape (1, {dim}) or ({dim},), got {a.shape}"
        raise ValueError(msg)
    return a


def _window_from(
    send_timestamp: datetime,
    active_window_days: int,
) -> tuple[datetime, datetime]:
    start = send_timestamp - timedelta(days=active_window_days)
    return (start, send_timestamp)


def _empty_result_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "story_id": pl.String,
            "similarity": pl.Float32,
            "created_at": pl.Datetime(time_unit="us"),
            "name": pl.String,
            "summary": pl.String,
            "article_count": pl.Int64,
            "source_reach_total": pl.Int64,
            "sentiment_positive": pl.Float64,
            "sentiment_negative": pl.Float64,
            "sentiment_neutral": pl.Float64,
            "source_bias_set": pl.List(pl.String),
        },
    )


def _empty_batch_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "query_row": pl.Int64,
            "story_id": pl.String,
            "similarity": pl.Float32,
            "created_at": pl.Datetime(time_unit="us"),
            "name": pl.String,
            "summary": pl.String,
            "article_count": pl.Int64,
            "source_reach_total": pl.Int64,
        },
    )


def _rescore_for_quad(
    frame: pl.DataFrame,
    layer: str,
    weight: float,
) -> pl.DataFrame:
    """Min-max normalize ``similarity`` within this layer and scale by weight."""
    if frame.height == 0 or weight <= 0:
        return pl.DataFrame(
            schema={"story_id": pl.String, "weighted_score": pl.Float64},
        )
    sims = frame["similarity"].to_numpy().astype(np.float64)
    lo = float(sims.min())
    hi = float(sims.max())
    norm = (sims - lo) / (hi - lo) if hi > lo else np.ones_like(sims)
    return pl.DataFrame(
        {
            "story_id": frame["story_id"],
            "weighted_score": (norm * weight).astype(np.float64),
        },
    )


def _combine_quad_scores(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Sum weighted per-layer scores across a candidate union."""
    non_empty = [f for f in frames if f.height > 0]
    if not non_empty:
        return pl.DataFrame(
            schema={"story_id": pl.String, "similarity": pl.Float64},
        )
    stacked = pl.concat(non_empty, how="vertical_relaxed")
    return (
        stacked.group_by("story_id")
        .agg(pl.col("weighted_score").sum().alias("similarity"))
        .sort("similarity", descending=True)
    )


def _bias_mean(bias_set: list[str] | None) -> float:
    """Map a ``source_bias_set`` value to its mean integer bias in {-1, 0, 1}."""
    if not bias_set:
        return 0.0
    vals: list[int] = []
    for b in bias_set:
        if b is None:
            continue
        v = _BIAS_TO_INT.get(b.strip().lower())
        if v is not None:
            vals.append(v)
    if not vals:
        return 0.0
    return float(np.mean(vals))
