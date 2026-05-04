"""Async OpenAI client wrapper with rate limiting, retry, and structured output."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import instructor
import openai

from libs.costs import costs as _default_costs
from libs.costs.pricing import (
    estimate_openai_completion,
    estimate_openai_completion_from_messages,
    estimate_openai_embedding,
)
from libs.costs.tracker import CostRecord, CostTracker
from libs.resilience import RateLimiter, run_in_batches_concurrent, with_retry

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CompletionOptions:
    """Bundled options for OpenAI completion requests."""

    model: str = "gpt-4o"
    temperature: float = 0.5
    max_tokens: int = 1000
    cost_key: str | None = None


@dataclass
class StructuredCompletionOptions(CompletionOptions):
    """Options for structured completion with response model metadata."""

    response_model_kwargs: dict[str, object] = field(
        default_factory=dict,
    )


class OpenAIClient:
    """Async OpenAI client with rate limiting and retry support."""

    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 25,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._rate_limiter = RateLimiter(max_concurrent=max_concurrent)
        self._costs = cost_tracker or _default_costs

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.close()

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()

    async def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-large",
        batch_size: int = 2048,
        cost_key: str | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts, processing in batches with rate limiting."""

        @with_retry(retry_on=(openai.APIError, openai.RateLimitError))
        async def _embed_batch(batch: list[str]) -> list[list[float]]:
            cleaned = [text.replace("\n", " ") for text in batch]
            response = await self._client.embeddings.create(
                model=model,
                input=cleaned,
            )
            return [item.embedding for item in response.data]

        batch_results: list[list[list[float]]] = await run_in_batches_concurrent(
            items=texts,
            batch_size=batch_size,
            process_fn=_embed_batch,
            rate_limiter=self._rate_limiter,
        )

        embeddings: list[list[float]] = []
        for batch_result in batch_results:
            embeddings.extend(batch_result)

        if cost_key:
            amount = estimate_openai_embedding(model, texts)
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=amount,
                    provider="openai",
                    operation="embed",
                    model=model,
                    units=len(texts),
                    unit_type="texts",
                )
            )

        return embeddings

    @with_retry(retry_on=(openai.APIError, openai.RateLimitError))
    async def complete(
        self,
        messages: list[dict[str, str]],
        options: CompletionOptions | None = None,
    ) -> str:
        """Request a single chat completion."""
        opts = options or CompletionOptions()
        async with self._rate_limiter:
            response = await self._client.chat.completions.create(
                model=opts.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=opts.temperature,
                max_tokens=opts.max_tokens,
            )
        content = response.choices[0].message.content
        if content is None:
            msg = "OpenAI returned a response with no content"
            raise ValueError(msg)

        if opts.cost_key and response.usage:
            amount = estimate_openai_completion(
                opts.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=amount,
                    provider="openai",
                    operation="complete",
                    model=opts.model,
                    units=response.usage.total_tokens,
                    unit_type="tokens",
                    source="actual",
                )
            )

        return content

    @with_retry(retry_on=(openai.APIError, openai.RateLimitError))
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        options: CompletionOptions | None = None,
    ) -> T:
        """Request a structured completion via instructor."""
        opts = options or CompletionOptions()
        instructor_client = instructor.from_openai(self._client)
        async with self._rate_limiter:
            result = await instructor_client.chat.completions.create(
                model=opts.model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_model,
                temperature=opts.temperature,
                max_completion_tokens=opts.max_tokens,
            )

        if opts.cost_key:
            amount = estimate_openai_completion_from_messages(
                opts.model,
                messages,
                opts.max_tokens,
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=amount,
                    provider="openai",
                    operation="complete_structured",
                    model=opts.model,
                )
            )

        return result

    async def complete_many(
        self,
        messages_list: list[list[dict[str, str]]],
        options: CompletionOptions | None = None,
    ) -> list[str]:
        """Complete multiple message lists concurrently with rate limiting."""
        opts = options or CompletionOptions()

        # Use a per-call options copy without cost_key so individual calls
        # don't each record costs; we aggregate below.
        inner_opts = CompletionOptions(
            model=opts.model,
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            cost_key=None,
        )

        async def _single(idx: int, messages: list[dict[str, str]]) -> tuple[int, str]:
            async with self._rate_limiter:
                result = await self.complete(messages, inner_opts)
            return (idx, result)

        tasks = [asyncio.create_task(_single(i, msgs)) for i, msgs in enumerate(messages_list)]
        completed = await asyncio.gather(*tasks)
        sorted_results = sorted(completed, key=lambda x: x[0])

        if opts.cost_key:
            total_amount = sum(
                estimate_openai_completion_from_messages(opts.model, msgs, opts.max_tokens)
                for msgs in messages_list
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=total_amount,
                    provider="openai",
                    operation="complete_many",
                    model=opts.model,
                    units=len(messages_list),
                    unit_type="requests",
                )
            )

        return [result for _, result in sorted_results]

    async def complete_structured_many(
        self,
        messages_list: list[list[dict[str, str]]],
        response_model: type[T],
        options: CompletionOptions | None = None,
    ) -> list[T]:
        """Extract structured data from multiple inputs concurrently."""
        opts = options or CompletionOptions()

        inner_opts = CompletionOptions(
            model=opts.model,
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            cost_key=None,
        )

        async def _single(idx: int, messages: list[dict[str, str]]) -> tuple[int, T]:
            async with self._rate_limiter:
                result = await self.complete_structured(messages, response_model, inner_opts)
            return (idx, result)

        tasks = [asyncio.create_task(_single(i, msgs)) for i, msgs in enumerate(messages_list)]
        completed = await asyncio.gather(*tasks)
        sorted_results = sorted(completed, key=lambda x: x[0])

        if opts.cost_key:
            total_amount = sum(
                estimate_openai_completion_from_messages(opts.model, msgs, opts.max_tokens)
                for msgs in messages_list
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=total_amount,
                    provider="openai",
                    operation="complete_structured_many",
                    model=opts.model,
                    units=len(messages_list),
                    unit_type="requests",
                )
            )

        return [result for _, result in sorted_results]
