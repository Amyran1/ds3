from __future__ import annotations

import numpy as np
import pytest

from libs.autoloop.dedup import (
    DedupIndex,
    DuplicateMatch,
    canonicalize_hypothesis,
    content_hash,
)


class FakeOpenAI:
    def __init__(self, mapping: dict[str, list[float]] | None = None):
        self.mapping = mapping or {}

    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        cost_key: str | None = None,
    ) -> list[list[float]]:
        return [self.mapping.get(t, [0.0] * 1536) for t in texts]


class FakeContainer:
    def __init__(self, openai_mapping: dict[str, list[float]] | None = None):
        self.openai = FakeOpenAI(openai_mapping)


def _unit(dim: int, val: float, dim2: int | None = None, val2: float = 0.0) -> list[float]:
    v = [0.0] * 1536
    v[dim] = val
    if dim2 is not None:
        v[dim2] = val2
    return v


def _embed_text(title: str, hypothesis: str) -> str:
    return f"{title}\n\n{hypothesis}"


# ---------------------------------------------------------------------------
# content_hash tests
# ---------------------------------------------------------------------------


def test_content_hash_stable() -> None:
    h1 = content_hash("Title", "user", "hypothesis text")
    h2 = content_hash("Title", "user", "hypothesis text")
    assert h1 == h2


def test_content_hash_differs_for_different_inputs() -> None:
    h1 = content_hash("Title A", "user", "hypothesis")
    h2 = content_hash("Title B", "user", "hypothesis")
    assert h1 != h2


def test_content_hash_case_insensitive_title() -> None:
    h1 = content_hash("My Title", "user", "hypothesis")
    h2 = content_hash("MY TITLE", "user", "hypothesis")
    assert h1 == h2


def test_content_hash_whitespace_tolerant_hypothesis() -> None:
    h1 = content_hash("Title", "user", "  multiple   spaces  ")
    h2 = content_hash("Title", "user", "multiple spaces")
    assert h1 == h2


# ---------------------------------------------------------------------------
# canonicalize_hypothesis tests
# ---------------------------------------------------------------------------


def test_canonicalize_hypothesis_collapses_spaces() -> None:
    result = canonicalize_hypothesis("  lots   of   spaces  ")
    assert result == "lots of spaces"


def test_canonicalize_hypothesis_trims() -> None:
    result = canonicalize_hypothesis("   trimmed   ")
    assert result == "trimmed"


# ---------------------------------------------------------------------------
# DedupIndex tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_then_check_near_duplicate_returns_match(
    tmp_path: pytest.TempPathFactory,
) -> None:
    v1 = _unit(0, 1.0)
    v2 = _unit(0, 0.95, 1, float(np.sqrt(1 - 0.95**2)))

    key1 = _embed_text("Idea One", "first hypothesis")
    key2 = _embed_text("Idea Two", "second hypothesis")

    container = FakeContainer({key1: v1, key2: v2})
    idx = DedupIndex(tmp_path)
    idx.load()

    await idx.add(
        container, item_id="id1", title="Idea One", hypothesis="first hypothesis", project="proj"
    )

    result = await idx.check(
        container, title="Idea Two", hypothesis="second hypothesis", threshold=0.90, project="proj"
    )

    assert result is not None
    assert isinstance(result, DuplicateMatch)
    assert result.matched_id == "id1"
    assert result.cosine_similarity == pytest.approx(0.95, abs=1e-4)


@pytest.mark.asyncio
async def test_check_unrelated_item_returns_none(tmp_path: pytest.TempPathFactory) -> None:
    v1 = _unit(0, 1.0)
    v3 = _unit(5, 1.0)

    key1 = _embed_text("Idea One", "first hypothesis")
    key3 = _embed_text("Unrelated", "orthogonal hypothesis")

    container = FakeContainer({key1: v1, key3: v3})
    idx = DedupIndex(tmp_path)
    idx.load()

    await idx.add(
        container, item_id="id1", title="Idea One", hypothesis="first hypothesis", project="proj"
    )

    result = await idx.check(
        container,
        title="Unrelated",
        hypothesis="orthogonal hypothesis",
        threshold=0.90,
        project="proj",
    )

    assert result is None


@pytest.mark.asyncio
async def test_check_threshold_boundary_returns_match(tmp_path: pytest.TempPathFactory) -> None:
    cosine = 0.95
    v1 = _unit(0, 1.0)
    v2 = _unit(0, cosine, 1, float(np.sqrt(1 - cosine**2)))

    key1 = _embed_text("Idea", "hyp")
    key2 = _embed_text("Idea Near", "hyp near")

    container = FakeContainer({key1: v1, key2: v2})
    idx = DedupIndex(tmp_path)
    idx.load()

    await idx.add(container, item_id="id1", title="Idea", hypothesis="hyp", project="proj")

    result = await idx.check(
        container, title="Idea Near", hypothesis="hyp near", threshold=0.92, project="proj"
    )

    assert result is not None
    assert result.cosine_similarity == pytest.approx(0.95, abs=1e-4)


@pytest.mark.asyncio
async def test_save_load_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    v1 = _unit(0, 1.0)
    v2 = _unit(0, 0.97, 1, float(np.sqrt(1 - 0.97**2)))

    key1 = _embed_text("Round Trip", "original hypothesis")
    key2 = _embed_text("Round Trip Near", "near hypothesis")

    container = FakeContainer({key1: v1, key2: v2})

    idx = DedupIndex(tmp_path)
    idx.load()
    await idx.add(
        container,
        item_id="round-id",
        title="Round Trip",
        hypothesis="original hypothesis",
        project="proj",
    )
    idx.save()

    idx2 = DedupIndex(tmp_path)
    idx2.load()

    result = await idx2.check(
        container,
        title="Round Trip Near",
        hypothesis="near hypothesis",
        threshold=0.90,
        project="proj",
    )

    assert result is not None
    assert result.matched_id == "round-id"


@pytest.mark.asyncio
async def test_empty_index_check_returns_none(tmp_path: pytest.TempPathFactory) -> None:
    container = FakeContainer()
    idx = DedupIndex(tmp_path)
    idx.load()

    result = await idx.check(
        container, title="Anything", hypothesis="anything", threshold=0.90, project="proj"
    )

    assert result is None


@pytest.mark.asyncio
async def test_check_returns_best_match(tmp_path: pytest.TempPathFactory) -> None:
    cosine_a = 0.80
    cosine_b = 0.95

    v_a = _unit(0, cosine_a, 1, float(np.sqrt(1 - cosine_a**2)))
    v_b = _unit(0, cosine_b, 1, float(np.sqrt(1 - cosine_b**2)))
    v_query = _unit(0, 1.0)

    key_a = _embed_text("Idea A", "hyp a")
    key_b = _embed_text("Idea B", "hyp b")
    key_q = _embed_text("Query", "query hyp")

    container = FakeContainer({key_a: v_a, key_b: v_b, key_q: v_query})
    idx = DedupIndex(tmp_path)
    idx.load()

    await idx.add(container, item_id="id-a", title="Idea A", hypothesis="hyp a", project="proj")
    await idx.add(container, item_id="id-b", title="Idea B", hypothesis="hyp b", project="proj")

    result = await idx.check(
        container, title="Query", hypothesis="query hyp", threshold=0.70, project="proj"
    )

    assert result is not None
    assert result.matched_id == "id-b"
    assert result.cosine_similarity == pytest.approx(cosine_b, abs=1e-4)
