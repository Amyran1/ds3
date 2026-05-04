"""Shared resilience primitives: retry decorator and rate limiter."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from types import TracebackType

    from typing_extensions import Self

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Semaphore-based async concurrency limiter."""

    def __init__(self, max_concurrent: int = 25) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self) -> Self:
        await self._semaphore.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._semaphore.release()


def with_retry(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Factory for tenacity retry decorator with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "Retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else "unknown",
        ),
    )


async def run_in_batches(
    items: list[T],
    batch_size: int,
    process_fn: Callable[[list[T]], Coroutine[Any, Any, list[Any]]],
    rate_limiter: RateLimiter | None = None,
) -> list[Any]:
    """Process items in batches with optional rate limiting."""
    results: list[Any] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        if rate_limiter:
            async with rate_limiter:
                result = await process_fn(batch)
        else:
            result = await process_fn(batch)
        results.append(result)
    return results


async def run_in_batches_concurrent(
    items: list[T],
    batch_size: int,
    process_fn: Callable[[list[T]], Coroutine[Any, Any, list[Any]]],
    rate_limiter: RateLimiter | None = None,
) -> list[Any]:
    """Process items in batches with concurrent execution.

    Unlike run_in_batches (sequential), this fires all batches as tasks
    and uses the rate_limiter semaphore for backpressure.
    Results returned in original input order.
    """
    batches: list[tuple[int, list[T]]] = [
        (idx, items[i : i + batch_size]) for idx, i in enumerate(range(0, len(items), batch_size))
    ]

    async def _run_batch(idx: int, batch: list[T]) -> tuple[int, list[Any]]:
        if rate_limiter:
            async with rate_limiter:
                result = await process_fn(batch)
        else:
            result = await process_fn(batch)
        return (idx, result)

    tasks = [asyncio.create_task(_run_batch(idx, batch)) for idx, batch in batches]
    completed = await asyncio.gather(*tasks)

    # Sort by batch index to preserve original input order
    sorted_results = sorted(completed, key=lambda x: x[0])
    return [result for _, result in sorted_results]
