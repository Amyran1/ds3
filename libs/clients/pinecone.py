"""Async Pinecone client wrapper with rate limiting and retry."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from libs.costs import costs as _default_costs
from libs.costs.pricing import (
    estimate_pinecone_read,
    estimate_pinecone_write,
)
from libs.costs.tracker import CostRecord, CostTracker
from libs.resilience import RateLimiter, run_in_batches_concurrent, with_retry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from typing_extensions import Self

logger = logging.getLogger(__name__)

PINECONE_SPARSE_ENGLISH_V0 = "pinecone-sparse-english-v0"
_INFERENCE_MAX_BATCH = 96
_VALID_SPARSE_INPUT_TYPES = frozenset({"passage", "query"})


class SparseValues(BaseModel):
    """Sparse vector representation for hybrid search."""

    indices: list[int]
    values: list[float]


class SparseEmbedding(BaseModel):
    """Sparse embedding produced by Pinecone Inference.

    Carries the same ``indices`` / ``values`` as :class:`SparseValues`, plus the
    optional normalized ``tokens`` returned when ``return_tokens=True`` is set
    on the embed request. Used with the ``pinecone-sparse-english-v0`` model.
    """

    indices: list[int]
    values: list[float]
    tokens: list[str] | None = None

    def to_sparse_values(self) -> SparseValues:
        """Drop tokens and return the upsert-shaped :class:`SparseValues`."""
        return SparseValues(indices=self.indices, values=self.values)


class QueryMatch(BaseModel):
    """Single match result from a Pinecone query."""

    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertItem(BaseModel):
    """Item to upsert into a Pinecone index."""

    id: str
    values: list[float]
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None


def _to_sparse_embedding(item: object) -> SparseEmbedding:
    """Adapt a Pinecone Inference embedding item to :class:`SparseEmbedding`.

    The SDK returns OpenAPI-generated ``Embedding`` objects that expose
    ``sparse_indices`` / ``sparse_values`` / ``sparse_tokens`` either as
    attributes or via mapping access depending on SDK version; we accept both.
    Tokens are returned only if the request set ``return_tokens=True``; when
    absent on the response item, ``tokens`` is ``None``.
    """

    def _read(field: str) -> object:
        value: object = getattr(item, field, None)
        if value is None and isinstance(item, dict):
            value = item.get(field)
        return value

    raw_indices = _read("sparse_indices")
    raw_values = _read("sparse_values")
    raw_tokens = _read("sparse_tokens")
    indices = cast("list[int]", raw_indices) if raw_indices else []
    values = cast("list[float]", raw_values) if raw_values else []
    tokens = cast("list[str]", raw_tokens) if raw_tokens else None
    return SparseEmbedding(indices=indices, values=values, tokens=tokens)


@dataclass
class QueryOptions:
    """Options bundle for Pinecone query."""

    vector: list[float]
    namespace: str = ""
    top_k: int = 10
    filter: dict[str, Any] | None = None
    sparse_vector: SparseValues | None = None
    include_metadata: bool = True


class PineconeClient:
    """Async Pinecone client with rate limiting, retry, and batched upserts."""

    def __init__(
        self,
        api_key: str,
        host: str,
        max_concurrent: int = 25,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._api_key = api_key
        self._host = host
        self._rate_limiter = RateLimiter(max_concurrent)
        self._costs = cost_tracker or _default_costs
        self._pc: Any = None
        self._index: Any = None
        self._ctx: Any = None

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[None]:
        """Open an async index connection via the Pinecone SDK."""
        from pinecone import Pinecone  # noqa: PLC0415

        self._pc = Pinecone(api_key=self._api_key)
        async with self._pc.IndexAsyncio(host=self._host) as index:
            self._index = index
            try:
                yield
            finally:
                self._index = None

    async def __aenter__(self) -> Self:
        self._ctx = self._connect()
        await self._ctx.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._ctx.__aexit__(exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        """Close the Pinecone connection."""
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)

    def _require_index(self) -> object:
        """Raise if the client is used outside a context manager."""
        if self._index is None:
            msg = "PineconeClient must be used as an async context manager"
            raise RuntimeError(msg)
        return self._index

    @with_retry(max_attempts=3, retry_on=(Exception,))
    async def query(
        self,
        options: QueryOptions,
        cost_key: str | None = None,
    ) -> list[QueryMatch]:
        """Query the Pinecone index and return ranked matches."""
        index = self._require_index()

        kwargs: dict[str, Any] = {
            "vector": options.vector,
            "namespace": options.namespace,
            "top_k": options.top_k,
            "include_metadata": options.include_metadata,
        }
        if options.filter is not None:
            kwargs["filter"] = options.filter
        if options.sparse_vector is not None:
            kwargs["sparse_vector"] = {
                "indices": options.sparse_vector.indices,
                "values": options.sparse_vector.values,
            }

        async with self._rate_limiter:
            response = await index.query(**kwargs)

        matches = [
            QueryMatch(
                id=match["id"],
                score=match["score"],
                metadata=match.get("metadata", {}),
            )
            for match in response["matches"]
        ]

        if cost_key:
            read_units = response.get("usage", {}).get("read_units", 1)
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_pinecone_read(read_units),
                    provider="pinecone",
                    operation="query",
                    units=read_units,
                    unit_type="read_units",
                    source="actual",
                )
            )

        return matches

    async def upsert(
        self,
        items: list[UpsertItem],
        namespace: str = "",
        batch_size: int = 100,
        cost_key: str | None = None,
    ) -> int:
        """Upsert items into the Pinecone index in batches."""
        index = self._require_index()
        if not items:
            return 0

        @with_retry(max_attempts=3, retry_on=(Exception,))
        async def _upsert_batch(batch: list[UpsertItem]) -> list[int]:
            vectors: list[dict[str, Any]] = []
            for item in batch:
                vec: dict[str, Any] = {
                    "id": item.id,
                    "values": item.values,
                }
                if item.metadata is not None:
                    vec["metadata"] = item.metadata
                if item.sparse_values is not None:
                    vec["sparse_values"] = {
                        "indices": item.sparse_values.indices,
                        "values": item.sparse_values.values,
                    }
                vectors.append(vec)

            async with self._rate_limiter:
                result = await index.upsert(
                    vectors=vectors,
                    namespace=namespace,
                )
            return [result["upserted_count"]]

        results = await run_in_batches_concurrent(items, batch_size, _upsert_batch)
        total = sum(count for batch_counts in results for count in batch_counts)

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_pinecone_write(total),
                    provider="pinecone",
                    operation="upsert",
                    units=total,
                    unit_type="write_units",
                )
            )

        return total

    async def embed_sparse(  # ### DO NOT USE THIS YET###
        self,
        texts: list[str],
        *,
        model: str = PINECONE_SPARSE_ENGLISH_V0,
        input_type: str = "passage",
        batch_size: int = 25,
        max_batch_size: int = _INFERENCE_MAX_BATCH,
        return_tokens: bool = False,
    ) -> list[SparseEmbedding]:
        """### DO NOT USE THIS YET###

        Generate sparse embeddings via Pinecone Inference.

        Defaults target the ``pinecone-sparse-english-v0`` model. Batches are
        dispatched concurrently under the client's existing rate limiter; each
        request is retried up to 3 times on any exception.

        This call does not require the index context manager — it opens its own
        ``PineconeAsyncio`` HTTP session for the inference endpoint and closes
        it before returning.
        """
        if not texts:
            return []
        if input_type not in _VALID_SPARSE_INPUT_TYPES:
            valid = sorted(_VALID_SPARSE_INPUT_TYPES)
            msg = f"input_type must be one of {valid}, got {input_type!r}"
            raise ValueError(msg)
        if batch_size < 1 or batch_size > max_batch_size:
            msg = f"batch_size ({batch_size}) must be in [1, {max_batch_size}]"
            raise ValueError(msg)

        from pinecone import PineconeAsyncio  # noqa: PLC0415

        async with PineconeAsyncio(api_key=self._api_key) as pc:

            @with_retry(max_attempts=3, retry_on=(Exception,))
            async def _embed_batch(batch: list[str]) -> list[list[SparseEmbedding]]:
                async with self._rate_limiter:
                    response = await pc.inference.embed(
                        model=model,
                        inputs=list(batch),
                        parameters={
                            "input_type": input_type,
                            "return_tokens": return_tokens,
                        },
                    )
                return [[_to_sparse_embedding(item) for item in response]]

            batch_results = await run_in_batches_concurrent(
                texts,
                batch_size,
                _embed_batch,
            )

        all_embeddings: list[SparseEmbedding] = []
        for group in batch_results:
            for batch in group:
                all_embeddings.extend(batch)
        return all_embeddings

    @with_retry(max_attempts=3, retry_on=(Exception,))
    async def delete(
        self,
        ids: list[str],
        namespace: str = "",
        cost_key: str | None = None,
    ) -> None:
        """Delete vectors by ID from the Pinecone index."""
        index = self._require_index()
        async with self._rate_limiter:
            await index.delete(ids=ids, namespace=namespace)

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_pinecone_write(len(ids)),
                    provider="pinecone",
                    operation="delete",
                    units=len(ids),
                    unit_type="write_units",
                )
            )
