"""Async Cohere client wrapper with rate limiting and retry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from libs.costs import costs as _default_costs
from libs.costs.pricing import (
    estimate_cohere_embed,
    estimate_cohere_rerank,
)
from libs.costs.tracker import CostRecord, CostTracker
from libs.resilience import RateLimiter, run_in_batches_concurrent, with_retry

logger = logging.getLogger(__name__)

_RERANK_BATCH_SIZE = 500


class RerankResult(BaseModel):
    """Single reranking result with original document ID and relevance score."""

    index: int
    id: str
    relevance_score: float


@dataclass
class RerankOptions:
    """Bundled options for Cohere rerank requests."""

    model: str = "rerank-english-v3.0"
    top_n: int = 500
    cost_key: str | None = None


def _to_sparse_vector(embedding: list[float]) -> dict[str, list[Any]]:
    """Convert a dense embedding to a sparse vector dict."""
    indices: list[int] = []
    values: list[float] = []
    for i, val in enumerate(embedding):
        if val != 0.0:
            indices.append(i)
            values.append(val)
    return {"indices": indices, "values": values}


class CohereClient:
    """Async Cohere client with rate limiting, retry, and batched operations."""

    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 10,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        import cohere  # noqa: PLC0415

        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._rate_limiter = RateLimiter(max_concurrent)
        self._costs = cost_tracker or _default_costs

    async def close(self) -> None:
        """Close the underlying client."""
        # Cohere client doesn't have an explicit close

    async def rerank(
        self,
        query: str,
        documents: list[str],
        ids: list[str],
        options: RerankOptions | None = None,
    ) -> list[RerankResult]:
        """Rerank documents against a query in batches."""
        opts = options or RerankOptions()
        n_docs = len(documents)
        n_ids = len(ids)
        if n_docs != n_ids:
            msg = f"documents and ids must have same length: {n_docs} != {n_ids}"
            raise ValueError(msg)

        if not documents:
            return []

        batches = _create_paired_batches(
            documents,
            ids,
            _RERANK_BATCH_SIZE,
        )

        total_billed_units = 0

        @with_retry(max_attempts=3, retry_on=(Exception,))
        async def _rerank_batch(
            batch: list[tuple[list[str], list[str]]],
        ) -> list[list[RerankResult]]:
            nonlocal total_billed_units
            batch_docs, batch_ids = batch[0]
            async with self._rate_limiter:
                response = await self._client.rerank(
                    query=query,
                    documents=batch_docs,
                    model=opts.model,
                    top_n=opts.top_n,
                )
            billed: int = len(batch_docs)
            if response.meta and response.meta.billed_units:
                raw = response.meta.billed_units.search_units
                if raw is not None:
                    billed = int(raw)
            total_billed_units += billed
            return [
                [
                    RerankResult(
                        index=result.index,
                        id=batch_ids[result.index],
                        relevance_score=result.relevance_score,
                    )
                    for result in response.results
                ],
            ]

        batch_items = list(batches)
        # Wrap each paired batch as a single-element list for _rerank_batch
        all_batch_results = await run_in_batches_concurrent(
            items=batch_items,
            batch_size=1,
            process_fn=_rerank_batch,
            rate_limiter=None,  # rate limiting handled inside _rerank_batch
        )
        results: list[RerankResult] = []
        for batch_result_group in all_batch_results:
            for result_list in batch_result_group:
                results.extend(result_list)

        if opts.cost_key:
            has_actual = total_billed_units > 0
            billed = total_billed_units if has_actual else n_docs
            amount = estimate_cohere_rerank(billed)
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=amount,
                    provider="cohere",
                    operation="rerank",
                    model=opts.model,
                    units=billed,
                    unit_type="search_units",
                    source="actual" if has_actual else "estimated",
                )
            )

        return results

    async def embed_sparse(
        self,
        texts: list[str],
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        batch_size: int = 96,
        cost_key: str | None = None,
    ) -> list[dict[str, list[Any]]]:
        """Generate sparse embedding vectors from texts in batches."""
        if not texts:
            return []

        @with_retry(max_attempts=3, retry_on=(Exception,))
        async def _embed_batch(
            batch: list[str],
        ) -> list[list[dict[str, list[Any]]]]:
            async with self._rate_limiter:
                response = await self._client.embed(
                    texts=batch,
                    model=model,
                    input_type=input_type,
                    embedding_types=["float"],
                )
            float_embeddings = response.embeddings.float_
            if float_embeddings is None:
                return [
                    [{"indices": [], "values": []} for _ in batch],
                ]
            return [
                [_to_sparse_vector(emb) for emb in float_embeddings],
            ]

        batch_results = await run_in_batches_concurrent(
            texts,
            batch_size,
            _embed_batch,
            rate_limiter=None,  # rate limiting handled inside _embed_batch
        )

        all_sparse: list[dict[str, list[Any]]] = []
        for result_group in batch_results:
            for sparse_list in result_group:
                if isinstance(sparse_list, list):
                    all_sparse.extend(sparse_list)
                else:
                    all_sparse.append(sparse_list)

        if cost_key:
            amount = estimate_cohere_embed(model, texts)
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=amount,
                    provider="cohere",
                    operation="embed_sparse",
                    model=model,
                    units=len(texts),
                    unit_type="texts",
                )
            )

        return all_sparse


def _create_paired_batches(
    docs: list[str],
    ids: list[str],
    batch_size: int,
) -> list[tuple[list[str], list[str]]]:
    """Split documents and IDs into aligned batches."""
    return [
        (
            docs[i : i + batch_size],
            ids[i : i + batch_size],
        )
        for i in range(0, len(docs), batch_size)
    ]
