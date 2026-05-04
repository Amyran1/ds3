"""Resume-safe pipeline for expensive chunked operations."""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def auto_concurrency(
    *,
    total_items: int,
    measured_rate: float,
    target_minutes: float,
    max_concurrency: int = 16,
    min_concurrency: int = 1,
) -> int:
    """Back-solve pipeline concurrency from a measured single-worker rate.

    Given a single-worker throughput (items/sec) and a wall-clock ceiling,
    return the smallest concurrency that keeps the projected run within
    ``target_minutes``. Clamped to ``[min_concurrency, max_concurrency]``.

    Callers should:
        1. Run one calibration probe via ``libs.calibrate.calibrate`` to
           get a single-worker ``measured_rate``.
        2. Pass the result here along with the full-run size and ceiling.
        3. Log the returned value so the choice is auditable.

    Args:
        total_items: number of items in the full run.
        measured_rate: items/sec measured by a single-worker probe.
        target_minutes: wall-clock ceiling to aim for.
        max_concurrency: hard cap (avoids starving pools or RAM).
        min_concurrency: floor (>= 1).

    Returns:
        Recommended concurrency, an int in ``[min_concurrency, max_concurrency]``.
    """
    if measured_rate <= 0:
        return max_concurrency
    target_seconds = target_minutes * 60
    if target_seconds <= 0:
        return max_concurrency
    needed = math.ceil(total_items / (measured_rate * target_seconds))
    return max(min_concurrency, min(max_concurrency, needed))


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for a checkpointed pipeline."""

    name: str  # e.g., "mercari-desc-embeddings-v1"
    checkpoint_dir: Path  # e.g., Path("data/mercari/cache/.checkpoints")
    chunk_size: int = 50_000
    concurrency: int = 1  # number of chunks to process in parallel


class CheckpointedPipeline(Generic[T, R]):
    """Resume-safe pipeline for expensive chunked operations.

    Saves completed chunk results to disk. On restart, loads
    completed chunks and continues from where it left off.

    Usage:
        pipeline = CheckpointedPipeline(
            config=CheckpointConfig(
                name="embeddings",
                checkpoint_dir=Path(...),
                chunk_size=50_000,
            ),
            process_chunk=my_async_process_fn,  # async (items, chunk_idx) -> result
            save_chunk=lambda result, idx, dir: np.save(dir / f"chunk_{idx}.npy", result),
            load_chunk=lambda idx, dir: np.load(dir / f"chunk_{idx}.npy"),
            combine=lambda chunks: np.vstack(chunks),
        )
        result = await pipeline.run(all_items)
    """

    def __init__(
        self,
        config: CheckpointConfig,
        process_chunk: Callable[[list[T], int], Awaitable[R]],
        save_chunk: Callable[[R, int, Path], None],
        load_chunk: Callable[[int, Path], R],
        combine: Callable[[list[R]], R],
    ) -> None:
        self._config = config
        self._process_chunk = process_chunk
        self._save_chunk = save_chunk
        self._load_chunk = load_chunk
        self._combine = combine

    async def run(self, items: list[T]) -> R:
        """Process all items, resuming from checkpoints if available.

        When ``config.concurrency > 1``, pending chunks are processed
        concurrently behind an ``asyncio.Semaphore``. Completed chunks
        are still loaded sequentially (they're just disk reads) and the
        final ordering of results is preserved by chunk index.
        """
        self._config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Split items into chunks
        chunks: list[list[T]] = [
            items[i : i + self._config.chunk_size]
            for i in range(0, len(items), self._config.chunk_size)
        ]
        total = len(chunks)

        # Log resume info if any chunks already done
        already_done = self.completed_chunks()
        done_set = set(already_done)
        if already_done:
            first_pending = min(
                (i for i in range(total) if i not in done_set),
                default=total,
            )
            logger.info(
                "Resuming from chunk %d (%d/%d already done)",
                first_pending,
                len(already_done),
                total,
            )

        results: list[R | None] = [None] * total

        # Load already-completed chunks from disk (fast, sequential).
        for idx in already_done:
            if idx < total:
                results[idx] = self._load_chunk(idx, self._config.checkpoint_dir)

        pending = [idx for idx in range(total) if idx not in done_set]
        concurrency = max(1, self._config.concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = len(already_done)
        completed_lock = asyncio.Lock()

        async def _run_one(idx: int) -> None:
            nonlocal completed_count
            async with semaphore:
                result = await self._process_chunk(chunks[idx], idx)
                self._save_chunk(result, idx, self._config.checkpoint_dir)
                self._mark_complete(idx)
                results[idx] = result
            async with completed_lock:
                completed_count += 1
                pct = completed_count * 100 // total
                logger.info(
                    "Chunk %d complete — %d/%d done (%d%%)",
                    idx,
                    completed_count,
                    total,
                    pct,
                )

        if pending:
            logger.info(
                "Processing %d pending chunks with concurrency=%d",
                len(pending),
                concurrency,
            )
            await asyncio.gather(*(_run_one(idx) for idx in pending))

        # All slots filled now; cast for type-checkers.
        ordered: list[R] = [r for r in results if r is not None]
        if len(ordered) != total:
            msg = (
                f"Expected {total} results after run, got {len(ordered)} — "
                "some chunks failed to materialize."
            )
            raise RuntimeError(msg)
        return self._combine(ordered)

    def completed_chunks(self) -> list[int]:
        """Return sorted indices of completed chunks."""
        if not self._config.checkpoint_dir.exists():
            return []

        prefix = f"{self._config.name}_chunk_"
        suffix = ".done"
        indices: list[int] = []
        for path in self._config.checkpoint_dir.iterdir():
            name = path.name
            if name.startswith(prefix) and name.endswith(suffix):
                idx_str = name[len(prefix) : -len(suffix)]
                try:
                    indices.append(int(idx_str))
                except ValueError:
                    continue
        return sorted(indices)

    def clear_checkpoints(self) -> None:
        """Remove the entire checkpoint directory for this pipeline."""
        if self._config.checkpoint_dir.exists():
            shutil.rmtree(self._config.checkpoint_dir)

    def _is_complete(self, chunk_idx: int) -> bool:
        """Check if chunk has a .done marker file."""
        return self._done_path(chunk_idx).exists()

    def _mark_complete(self, chunk_idx: int) -> None:
        """Write .done marker file for chunk."""
        self._done_path(chunk_idx).touch()

    def _done_path(self, chunk_idx: int) -> Path:
        return self._config.checkpoint_dir / f"{self._config.name}_chunk_{chunk_idx}.done"
