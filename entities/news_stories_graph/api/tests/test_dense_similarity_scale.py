"""Tests for the L2->cosine conversion in related_stories_dense.

The v1 FAISS index was built as IndexIVFFlat (METRIC_L2), so raw
``distances`` values are squared L2 distances, not inner products.  The API
layer converts via ``cos(theta) = 1 - L2^2 / 2`` (exact for unit-norm
vectors).

Strategy: option (b) -- construct a minimal StoryGraphAPI instance and patch
``_caches.dense_index``, ``_caches.dense_story_ids``, and
``_caches.story_metadata`` with a small synthetic FAISS IndexIVFFlat
(quantizer=IndexFlatIP, nlist=4) over known unit-norm vectors, then call
``.related_stories_dense()`` end-to-end and assert:

    1. All returned ``similarity`` values are in [-1.0, 1.0 + 1e-5].
    2. Querying with a vector equal to an indexed vector returns cosine ~= 1.0.
    3. Two known vectors with hand-computed cosines produce similarities
       within 1e-4 of expected.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any, cast

import faiss
import numpy as np
import polars as pl
import pytest

from entities.news_stories_graph.api.api import StoryGraphAPI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 1536  # must match production dim used in related_stories_dense
_NLIST = 4  # tiny; we train on all vectors since n_vectors >= nlist


def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized version of v."""
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32)


def _build_synthetic_ivfflat(vectors: np.ndarray) -> Any:
    """Construct an IndexIVFFlat (metric=L2, quantizer=IndexFlatIP) over vectors.

    This mirrors the v1 production construction:
        quantizer = faiss.IndexFlatIP(DIM)
        index = faiss.IndexIVFFlat(quantizer, DIM, NLIST)
        # metric_type defaults to METRIC_L2 -- the bug
    """
    quantizer = faiss.IndexFlatIP(_DIM)
    # cast to Any: faiss Python stubs have inconsistent parameter signatures
    # across faiss versions; calling .train() / .add() works at runtime.
    idx: Any = faiss.IndexIVFFlat(quantizer, _DIM, _NLIST)
    idx.train(vectors)
    idx.add(vectors)
    idx.nprobe = 4  # search all clusters in test
    return idx


def _build_story_metadata(n: int) -> pl.DataFrame:
    """Return a minimal story_metadata frame with n rows."""
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
    return pl.DataFrame(
        {
            "story_id": [f"story_{i}" for i in range(n)],
            "created_at": [now] * n,
            "name": [f"Story {i}" for i in range(n)],
            "summary": [""] * n,
            "article_count": [1] * n,
            "source_reach_total": [0] * n,
            "sentiment_positive": [0.0] * n,
            "sentiment_negative": [0.0] * n,
            "sentiment_neutral": [1.0] * n,
            "source_bias_set": [None] * n,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_api_and_vectors() -> tuple[StoryGraphAPI, np.ndarray]:
    """Return a StoryGraphAPI with a patched in-memory IVFFlat index.

    We build 20 unit-norm random vectors so nlist=4 clustering is stable.
    The first vector (index 0) is our "self-query" baseline.
    Vector[2] queried against vector[5] gives a hand-computable cosine.
    """
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((20, _DIM)).astype(np.float32)
    vectors = np.stack([_unit(raw[i]) for i in range(raw.shape[0])])

    index = _build_synthetic_ivfflat(vectors)
    story_ids = [f"story_{i}" for i in range(len(vectors))]
    metadata = _build_story_metadata(len(vectors))

    api = StoryGraphAPI.__new__(StoryGraphAPI)
    # Bypass __init__ which triggers _Caches (file IO)
    object.__setattr__(api, "_version", 1)

    class _FakeCaches:
        dense_index = index
        dense_story_ids = story_ids
        story_metadata = metadata

    object.__setattr__(api, "_caches", _FakeCaches())

    return api, vectors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDenseSimilarityScale:
    """All returned similarities must be valid cosine values."""

    def test_similarities_in_range(
        self,
        synthetic_api_and_vectors: tuple[StoryGraphAPI, np.ndarray],
    ) -> None:
        """All returned similarity values must be in [-1.0, 1.0 + 1e-5]."""
        api, vectors = synthetic_api_and_vectors
        query = vectors[0]
        result = api.related_stories_dense(query, k=20, min_similarity=-2.0)
        assert result.height > 0, "Expected at least one hit"
        sims = result["similarity"].to_numpy()
        sims_min, sims_max = float(sims.min()), float(sims.max())
        assert np.all(sims >= -1.0 - 1e-5), f"min={sims_min:.6f} below -1.0"
        assert np.all(sims <= 1.0 + 1e-5), f"max={sims_max:.6f} above 1.0"

    def test_self_query_returns_cosine_one(
        self,
        synthetic_api_and_vectors: tuple[StoryGraphAPI, np.ndarray],
    ) -> None:
        """Querying with vector[0] should return cosine ~= 1.0 for story_0."""
        api, vectors = synthetic_api_and_vectors
        query = vectors[0]
        result = api.related_stories_dense(query, k=5, min_similarity=-2.0)
        assert result.height > 0
        top_story = result["story_id"][0]
        top_sim = float(result["similarity"][0])
        assert top_story == "story_0", f"Expected story_0 as top hit, got {top_story}"
        assert abs(top_sim - 1.0) < 1e-3, f"Self-query cosine ~= 1.0, got {top_sim:.6f}"

    def test_known_cosine_within_tolerance(
        self,
        synthetic_api_and_vectors: tuple[StoryGraphAPI, np.ndarray],
    ) -> None:
        """Return cosine(query, story_k) within 1e-4 of hand-computed value."""
        api, vectors = synthetic_api_and_vectors
        # Query with vector[2]; compute expected cosine against vector[5]
        query = vectors[2]
        target = vectors[5]
        # Hand-compute: cosine = dot product of unit vectors
        expected_cosine = float(np.dot(query, target))

        result = api.related_stories_dense(query, k=20, min_similarity=-2.0)
        assert result.height > 0

        hit = result.filter(pl.col("story_id") == "story_5")
        assert hit.height == 1, "story_5 not in top-20; check NLIST or k"
        returned_cosine = float(hit["similarity"][0])
        delta = abs(returned_cosine - expected_cosine)
        assert delta < 1e-4, (
            f"Expected cosine={expected_cosine:.6f} for story_5, "
            f"got {returned_cosine:.6f} (delta={delta:.6f})"
        )

    def test_min_similarity_filter_on_cosine_scale(
        self,
        synthetic_api_and_vectors: tuple[StoryGraphAPI, np.ndarray],
    ) -> None:
        """min_similarity=0.5 should filter by cosine, not by L2^2."""
        api, vectors = synthetic_api_and_vectors
        query = vectors[0]
        result_filtered = api.related_stories_dense(query, k=20, min_similarity=0.5)
        result_unfiltered = api.related_stories_dense(query, k=20, min_similarity=-2.0)
        if result_filtered.height > 0:
            sims = result_filtered["similarity"].to_numpy()
            assert np.all(sims >= 0.5 - 1e-5), (
                f"Filtered result has similarity < 0.5: min={sims.min():.6f}"
            )
        assert result_unfiltered.height >= result_filtered.height

    def test_raw_l2sq_would_fail_range_check(
        self,
        synthetic_api_and_vectors: tuple[StoryGraphAPI, np.ndarray],
    ) -> None:
        """Sanity: raw L2^2 distances are NOT in [-1, 1], confirming the bug.

        We directly call index.search() and verify that raw distances exceed
        1.0 for at least some neighbors -- proving that raw L2^2 interpretation
        is wrong and the conversion is necessary.
        """
        api, vectors = synthetic_api_and_vectors
        query = vectors[2].reshape(1, -1)
        faiss_any: Any = cast("Any", api._caches.dense_index)  # noqa: SLF001
        with contextlib.suppress(AttributeError):
            faiss_any.nprobe = 4
        distances, _indices = faiss_any.search(query, 10)
        raw = distances[0]
        # For unit-norm vectors, L2^2 = 2*(1 - cosine), so L2^2 in [0, 4].
        # At least some non-self-neighbors should have L2^2 > 1.0 (cosine < 0.5).
        assert np.any(raw > 1.0), (
            "Expected some L2^2 > 1.0 for non-identical neighbors; "
            "if all are <= 1.0, the test corpus is too clustered."
        )
