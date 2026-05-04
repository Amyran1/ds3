"""Lightweight progress reporter writing JSON to a temp dsrun directory.

Scripts use ProgressReporter to report progress during long-running operations.
dsrun (or any monitor) reads the JSON file to surface throughput and ETA.

Usage:
    from libs.progress_tracking.progress import ProgressReporter

    reporter = ProgressReporter("my-extraction", total=5000)
    for batch in batches:
        process(batch)
        reporter.update(len(batch))
    reporter.complete()
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DSRUN_BASE_DIR = Path(tempfile.gettempdir()) / "dsrun"


class ProgressReporter:
    """Thread-safe progress reporter that writes JSON to disk.

    Attributes:
        name: Identifier used for the progress directory.
        total: Total number of records expected.
    """

    def __init__(self, name: str, total: int) -> None:
        """Initialize the reporter and write the initial progress file.

        Args:
            name: Unique name for this run (used as directory under dsrun/).
            total: Total number of records to process.
        """
        self.name = name
        self.total = total
        self._records_done = 0
        self._errors = 0
        self._error_message: str | None = None
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

        # Use DSRUN_PROGRESS_DIR if set (running under dsrun), else name-based dir
        dsrun_dir = os.environ.get("DSRUN_PROGRESS_DIR")
        self._progress_dir = Path(dsrun_dir) if dsrun_dir else DSRUN_BASE_DIR / name
        self._progress_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = self._progress_dir / "progress.json"

        self._write_state("running")

    def update(self, n: int, errors: int = 0) -> None:
        """Record that n records were processed (with optional error count).

        Args:
            n: Number of records processed in this batch.
            errors: Number of errors encountered in this batch.
        """
        with self._lock:
            self._records_done += n
            self._errors += errors
            self._write_state("running")

    def complete(self) -> None:
        """Mark the run as successfully completed."""
        with self._lock:
            self._write_state("complete")

    def error(self, msg: str) -> None:
        """Mark the run as failed with an error message.

        Args:
            msg: Human-readable error description.
        """
        with self._lock:
            self._error_message = msg
            self._write_state("error")

    def _write_state(self, status: str) -> None:
        """Write current state to the progress JSON file.

        Args:
            status: One of "running", "complete", or "error".
        """
        elapsed = time.monotonic() - self._start_time
        throughput = self._records_done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self._records_done
        eta = remaining / throughput if throughput > 0 else 0.0

        state: dict[str, Any] = {
            "status": status,
            "records_done": self._records_done,
            "records_total": self.total,
            "throughput_rec_s": round(throughput, 2),
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 1),
            "errors": self._errors,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if self._error_message is not None:
            state["error_message"] = self._error_message

        # Atomic write via temp file + rename to avoid partial reads.
        tmp_path = self._progress_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2))
        tmp_path.replace(self._progress_file)


def read_progress(name: str) -> dict[str, Any] | None:
    """Read the progress file for a given run name.

    Args:
        name: The run name (directory under dsrun/).

    Returns:
        Parsed progress dict, or None if the file does not exist or is unreadable.
    """
    progress_file = DSRUN_BASE_DIR / name / "progress.json"
    try:
        data: dict[str, Any] = json.loads(progress_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data
