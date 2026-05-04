"""Async batch processor with tqdm, Slack reporting, and 429 backoff.

Processes items in concurrent batches with adaptive rate limiting.
Extracted from the patterns in warm_story_cache.py and generalized.

Usage:
    from libs.batch_runner import BatchRunner, BatchRunnerConfig

    config = BatchRunnerConfig(
        name="emails",
        batch_size=50,
        slack_webhook=os.environ.get("CLAUDE_SLACK_WEBHOOK_URL"),
    )
    runner = BatchRunner(config)
    result = await runner.run(items, process_one_email)
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import logging
import os
import signal
import ssl
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import urlparse

if TYPE_CHECKING:
    from types import FrameType

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type aliases for process/skip function signatures
ProcessFn = Callable[[T], Awaitable[None]]
SkipFn = Callable[[T], bool]

_HTTP_429_STATUS = 429
_SLACK_TIMEOUT_SEC = 10


class RateLimitError(Exception):
    """Raised when a 429 rate limit response is detected."""


def is_rate_limit(exc: BaseException) -> bool:
    """Check if an exception indicates a 429 rate limit.

    Inspects status_code/status attributes and walks __cause__ chain.
    Works with httpx, requests, pinecone, and similar HTTP clients.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == _HTTP_429_STATUS:
        return True
    return bool(exc.__cause__ and is_rate_limit(exc.__cause__))


@dataclass
class BatchResult:
    """Outcome of a batch run."""

    name: str
    total: int
    processed: int
    failed: int
    errors: list[tuple[int, str]] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def throughput(self) -> float:
        """Records per second throughput."""
        if self.duration_sec > 0:
            return self.processed / self.duration_sec
        return 0.0

    def summary(self) -> str:
        """Human-readable summary of the batch run."""
        lines = [
            f"Batch run complete: {self.name}",
            f"  Total:      {self.total:,}",
            f"  Processed:  {self.processed:,}",
            f"  Failed:     {self.failed:,}",
            f"  Duration:   {self.duration_sec:.1f}s",
            f"  Throughput: {self.throughput:.1f} rec/s",
        ]
        if self.errors:
            lines.append("  First errors:")
            for idx, err in self.errors[:5]:
                lines.append(f"    [{idx}] {err}")
        return "\n".join(lines)


@dataclass
class BatchRunnerConfig:
    """Configuration for a BatchRunner instance.

    Args:
        name: Human-readable label for logs and Slack messages.
        batch_size: Items processed concurrently per batch.
        min_throttle: Minimum delay between batches (seconds).
        retries: Number of retry passes for failed items.
        slack_webhook: Slack incoming webhook URL (or None).
        slack_interval_sec: Seconds between Slack updates.
        progress_file: Path to write progress JSON (auto-detected
            from DSRUN_PROGRESS_DIR env var if not provided).
    """

    name: str
    batch_size: int = 50
    min_throttle: float = 0.5
    retries: int = 2
    slack_webhook: str | None = None
    slack_interval_sec: int = 300
    progress_file: Path | None = None


class BatchRunner:
    """Async batch processor with progress tracking and rate limit handling."""

    def __init__(self, config: BatchRunnerConfig) -> None:
        """Initialize the batch runner.

        Args:
            config: Runner configuration parameters.
        """
        self.name = config.name
        self.batch_size = config.batch_size
        self.min_throttle = config.min_throttle
        self.retries = config.retries
        self.slack_webhook = config.slack_webhook
        self.slack_interval_sec = config.slack_interval_sec

        progress_file = config.progress_file
        if progress_file is None:
            dsrun_dir = os.environ.get("DSRUN_PROGRESS_DIR")
            if dsrun_dir:
                progress_file = Path(dsrun_dir) / "progress.json"
        self.progress_file = progress_file
        self._shutdown_requested = False

    async def run(
        self,
        items: list[T],
        process_fn: Callable[[T], Awaitable[object]],
        *,
        skip_fn: Callable[[T], bool] | None = None,
    ) -> BatchResult:
        """Process items in batches with tqdm, Slack, and 429 handling.

        Args:
            items: List of items to process.
            process_fn: Async function that processes one item.
                Raise RateLimitError on 429. Other exceptions
                count as failures.
            skip_fn: Optional sync function to check if an item
                is already done. Items where skip_fn returns True
                are skipped (for resume support).

        Returns:
            BatchResult with counts, timing, and error details.
        """
        to_process, skipped = self._filter_items(items, skip_fn)

        if skipped:
            logger.info(
                "Skipped %d already-done, %d to process",
                skipped,
                len(to_process),
            )

        if not to_process:
            return BatchResult(
                name=self.name,
                total=len(items),
                processed=0,
                failed=0,
            )

        self._setup_shutdown_handler()

        t0 = time.monotonic()
        batch_state = await self._run_batches(to_process, process_fn, t0)
        processed = batch_state[0]
        all_failed = batch_state[1]
        backoff = batch_state[2]

        processed, all_failed, backoff = await self._retry_failed(
            all_failed,
            process_fn,
            t0,
            processed,
            backoff,
        )

        duration = time.monotonic() - t0

        # Write final progress state
        self._write_final_progress(processed, len(items), duration, all_failed)

        errors = [(i, str(e)) for i, e in enumerate(all_failed[:20])]

        result = BatchResult(
            name=self.name,
            total=len(items),
            processed=processed,
            failed=len(all_failed),
            errors=errors,
            duration_sec=duration,
        )

        logger.info(result.summary())
        self._send_slack(result.summary())
        return result

    def _setup_shutdown_handler(self) -> None:
        """Register SIGUSR1 graceful shutdown handler if under dsrun."""
        self._shutdown_requested = False
        if os.environ.get("DSRUN_ACTIVE") == "1":
            signal.signal(
                signal.SIGUSR1,
                self._sigusr1_handler,
            )

    async def _retry_failed(
        self,
        all_failed: list[T],
        process_fn: Callable[[T], Awaitable[object]],
        t0: float,
        processed: int,
        backoff: float,
    ) -> tuple[int, list[T], float]:
        """Retry failed items up to self.retries times.

        Args:
            all_failed: Items that failed on the first pass.
            process_fn: Async function to process each item.
            t0: Monotonic start time of the overall run.
            processed: Count of successfully processed items so far.
            backoff: Current backoff delay in seconds.

        Returns:
            Tuple of (processed count, remaining failures, backoff).
        """
        for retry_num in range(1, self.retries + 1):
            if not all_failed:
                break
            logger.info(
                "Retry %d/%d: %d failed items",
                retry_num,
                self.retries,
                len(all_failed),
            )
            await asyncio.sleep(backoff * 3)
            batch_state = await self._run_batches(all_failed, process_fn, t0)
            processed += batch_state[0]
            all_failed = batch_state[1]
            backoff = batch_state[2]
        return processed, all_failed, backoff

    def _sigusr1_handler(self, _signum: int, _frame: FrameType | None) -> None:
        """Handle SIGUSR1 for graceful shutdown under dsrun."""
        self._shutdown_requested = True
        logger.info("SIGUSR1 received -- finishing current batch")

    def _filter_items(
        self,
        items: list[T],
        skip_fn: Callable[[T], bool] | None,
    ) -> tuple[list[T], int]:
        """Separate items into to-process and skipped lists.

        Args:
            items: All items to consider.
            skip_fn: Predicate returning True for items to skip.

        Returns:
            Tuple of (items to process, count of skipped).
        """
        if skip_fn is None:
            return items, 0
        to_process = []
        skipped = 0
        for item in items:
            if skip_fn(item):
                skipped += 1
            else:
                to_process.append(item)
        return to_process, skipped

    async def _run_batches(
        self,
        items: list[T],
        process_fn: Callable[[T], Awaitable[object]],
        run_start: float,
    ) -> tuple[int, list[T], float]:
        """Process items in batches with progress tracking.

        Args:
            items: Items to process in this pass.
            process_fn: Async function to process each item.
            run_start: Monotonic timestamp of the run start.

        Returns:
            Tuple of (processed count, failed items, backoff).
        """
        tqdm = _import_tqdm()

        total = len(items)
        all_failed: list[T] = []
        processed = 0
        backoff = self.min_throttle
        last_slack = time.monotonic()

        pbar = tqdm(total=total, desc=self.name, unit="rec") if tqdm else None

        try:
            for i in range(0, total, self.batch_size):
                batch = items[i : i + self.batch_size]
                batch_result = await self._process_batch(batch, process_fn)
                failed = batch_result[0]
                hit_rate_limit = batch_result[1]

                if hit_rate_limit:
                    rl = await self._handle_rate_limit(failed, process_fn, backoff)
                    failed = rl[0]
                    backoff = rl[1]
                else:
                    backoff = max(backoff * 0.8, self.min_throttle)

                batch_ok = len(batch) - len(failed)
                processed += batch_ok
                all_failed.extend(failed)

                if pbar:
                    pbar.update(len(batch))

                now = time.monotonic()
                if now - last_slack >= self.slack_interval_sec:
                    self._send_progress(processed, total, now - run_start)
                    last_slack = now

                # Write progress JSON for dsrun/monitors
                if self.progress_file:
                    self._write_progress(
                        processed,
                        total,
                        now - run_start,
                        len(all_failed),
                    )

                # Check for graceful shutdown request
                if self._shutdown_requested:
                    logger.info(
                        "Graceful shutdown: %d/%d records",
                        processed,
                        total,
                    )
                    break

                if i + len(batch) < total:
                    await asyncio.sleep(backoff)
        finally:
            if pbar:
                pbar.close()

        return processed, all_failed, backoff

    async def _process_batch(
        self,
        batch: list[T],
        process_fn: Callable[[T], Awaitable[object]],
    ) -> tuple[list[T], bool]:
        """Process a single batch of items concurrently.

        Args:
            batch: Items in this batch.
            process_fn: Async function to process each item.

        Returns:
            Tuple of (failed items, whether rate limit was hit).
        """
        tasks = [process_fn(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed: list[T] = []
        hit_rate_limit = False

        for item, result in zip(batch, results, strict=False):
            if isinstance(result, RateLimitError) or (
                isinstance(result, BaseException) and is_rate_limit(result)
            ):
                hit_rate_limit = True
                failed.append(item)
            elif isinstance(result, BaseException):
                logger.warning("Failed for item: %s", result)
                failed.append(item)

        return failed, hit_rate_limit

    async def _handle_rate_limit(
        self,
        failed: list[T],
        process_fn: Callable[[T], Awaitable[object]],
        backoff: float,
    ) -> tuple[list[T], float]:
        """Back off and retry after a 429 rate limit.

        Args:
            failed: Items that failed due to rate limiting.
            process_fn: Async function to process each item.
            backoff: Current backoff delay in seconds.

        Returns:
            Tuple of (still-failed items, new backoff delay).
        """
        backoff = min(backoff * 2, 60.0)
        logger.warning(
            "Rate limited (429). Pausing %ds...",
            int(backoff),
        )
        await asyncio.sleep(backoff)

        retry_failed, retry_hit = await self._process_batch(failed, process_fn)
        if retry_hit:
            backoff = min(backoff * 2, 120.0)
            logger.warning(
                "Still rate limited. Pausing %ds...",
                int(backoff),
            )
            await asyncio.sleep(backoff)

        recovered = len(failed) - len(retry_failed)
        if recovered:
            logger.info("Recovered %d on retry", recovered)
        return retry_failed, backoff

    def _write_progress(
        self,
        done: int,
        total: int,
        elapsed: float,
        errors: int,
    ) -> None:
        """Write in-flight progress state to progress JSON.

        Args:
            done: Number of records processed so far.
            total: Total number of records.
            elapsed: Elapsed time in seconds.
            errors: Number of errors encountered.
        """
        if not self.progress_file:
            return
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else 0.0
        state = {
            "status": "running",
            "records_done": done,
            "records_total": total,
            "throughput_rec_s": round(rate, 2),
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(remaining, 1),
            "errors": errors,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        tmp = self.progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.progress_file)

    def _write_final_progress(
        self,
        done: int,
        total: int,
        elapsed: float,
        failed: list[object],
    ) -> None:
        """Write final progress state to progress JSON.

        Args:
            done: Number of records successfully processed.
            total: Total number of records.
            elapsed: Total elapsed time in seconds.
            failed: List of failed items.
        """
        if not self.progress_file:
            return
        rate = done / elapsed if elapsed > 0 else 0.0
        status = "complete" if not failed else "error"
        state = {
            "status": status,
            "records_done": done,
            "records_total": total,
            "throughput_rec_s": round(rate, 2),
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": 0.0,
            "errors": len(failed),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        tmp = self.progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.progress_file)

    def _send_progress(self, done: int, total: int, elapsed: float) -> None:
        """Send a progress update to Slack.

        Args:
            done: Number of records processed.
            total: Total number of records.
            elapsed: Elapsed time in seconds.
        """
        if not self.slack_webhook:
            return
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        pct = (done / total * 100) if total > 0 else 0
        msg = (
            f"*{self.name}*\n"
            f"Progress: {done:,}/{total:,} ({pct:.1f}%)\n"
            f"Speed: {rate:.1f} rec/s\n"
            f"ETA: {remaining / 60:.1f} min"
        )
        self._send_slack(msg)

    def _send_slack(self, text: str) -> None:
        """Send a message to Slack via incoming webhook.

        Args:
            text: Message text to send.
        """
        if not self.slack_webhook:
            return
        _post_webhook(self.slack_webhook, {"text": text})


def _post_webhook(url: str, payload: dict[str, str]) -> None:
    """POST JSON to a webhook URL using http.client.

    Only https:// URLs are supported. Other schemes are
    silently ignored for security.

    Args:
        url: The webhook URL (must be https).
        payload: JSON-serializable dict to POST.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    ctx = ssl.create_default_context()
    with contextlib.suppress(Exception):
        conn = http.client.HTTPSConnection(
            parsed.hostname,
            port=parsed.port,
            timeout=_SLACK_TIMEOUT_SEC,
            context=ctx,
        )
        try:
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            conn.request(
                "POST",
                path,
                body=data,
                headers=headers,
            )
            conn.getresponse().read()
        finally:
            conn.close()


def _import_tqdm() -> type | None:
    """Import tqdm if available, return None otherwise."""
    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:
        return None
    return tqdm
