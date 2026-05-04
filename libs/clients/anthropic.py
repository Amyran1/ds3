"""Async Anthropic client wrapper with rate limiting and retry."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
import instructor

from libs.costs import costs as _default_costs
from libs.costs.pricing import (
    estimate_anthropic_completion,
    estimate_anthropic_completion_cached,
    estimate_anthropic_completion_from_messages,
)
from libs.costs.tracker import CostRecord, CostTracker
from libs.resilience import RateLimiter, with_retry

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


@dataclass
class CompletionOptions:
    """Bundled options for Anthropic completion requests."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.5
    max_tokens: int = 1000
    cost_key: str | None = None


def _extract_text(response: anthropic.types.Message) -> str:
    """Join all text blocks from an Anthropic message response."""
    return "".join(block.text for block in response.content if hasattr(block, "text"))


class AnthropicClient:
    """Thin async wrapper around the Anthropic API."""

    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 25,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._rate_limiter = RateLimiter(max_concurrent)
        self._costs = cost_tracker or _default_costs

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def complete(
        self,
        messages: list[dict[str, str]],
        options: CompletionOptions | None = None,
    ) -> str:
        """Send a completion request and return text."""
        opts = options or CompletionOptions()
        async with self._rate_limiter:
            response = await self._client.messages.create(
                model=opts.model,
                max_tokens=opts.max_tokens,
                messages=messages,
                temperature=opts.temperature,
            )

        if opts.cost_key:
            amount = estimate_anthropic_completion(
                opts.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=amount,
                    provider="anthropic",
                    operation="complete",
                    model=opts.model,
                    units=total_tokens,
                    unit_type="tokens",
                    source="actual",
                )
            )

        return _extract_text(response)

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def complete_cached(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        cost_key: str | None = None,
    ) -> str:
        """Send a completion with prompt caching and return text."""
        async with self._rate_limiter:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=temperature,
            )

        if cost_key:
            usage = response.usage
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            total_tokens = usage.input_tokens + usage.output_tokens + cache_creation + cache_read
            amount = estimate_anthropic_completion_cached(
                model,
                usage.input_tokens,
                usage.output_tokens,
                cache_creation,
                cache_read,
            )
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=amount,
                    provider="anthropic",
                    operation="complete_cached",
                    model=model,
                    units=total_tokens,
                    unit_type="tokens",
                    source="actual",
                )
            )

        return _extract_text(response)

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        options: CompletionOptions | None = None,
    ) -> T:
        """Send a completion and parse into a structured model via instructor."""
        opts = options or CompletionOptions()
        client = instructor.from_anthropic(self._client)

        async with self._rate_limiter:
            result = await client.messages.create(
                model=opts.model,
                max_tokens=opts.max_tokens,
                messages=messages,
                response_model=response_model,
                temperature=opts.temperature,
            )

        if opts.cost_key:
            amount = estimate_anthropic_completion_from_messages(
                opts.model,
                messages,
                opts.max_tokens,
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=amount,
                    provider="anthropic",
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
                estimate_anthropic_completion_from_messages(opts.model, msgs, opts.max_tokens)
                for msgs in messages_list
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=total_amount,
                    provider="anthropic",
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
                estimate_anthropic_completion_from_messages(opts.model, msgs, opts.max_tokens)
                for msgs in messages_list
            )
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=total_amount,
                    provider="anthropic",
                    operation="complete_structured_many",
                    model=opts.model,
                    units=len(messages_list),
                    unit_type="requests",
                )
            )

        return [result for _, result in sorted_results]

    async def complete_cached_many(
        self,
        system: list[dict[str, Any]],
        messages_list: list[list[dict[str, Any]]],
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        cost_key: str | None = None,
    ) -> list[str]:
        """Complete multiple cached message lists concurrently with rate limiting."""

        async def _single(idx: int, messages: list[dict[str, Any]]) -> tuple[int, str]:
            async with self._rate_limiter:
                result = await self.complete_cached(
                    system=system,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cost_key=None,
                )
            return (idx, result)

        tasks = [asyncio.create_task(_single(i, msgs)) for i, msgs in enumerate(messages_list)]
        completed = await asyncio.gather(*tasks)
        sorted_results = sorted(completed, key=lambda x: x[0])

        if cost_key:
            total_amount = sum(
                estimate_anthropic_completion_from_messages(model, msgs, max_tokens)
                for msgs in messages_list
            )
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=total_amount,
                    provider="anthropic",
                    operation="complete_cached_many",
                    model=model,
                    units=len(messages_list),
                    unit_type="requests",
                )
            )

        return [result for _, result in sorted_results]
