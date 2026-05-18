"""Per-project failure-shell row writer for results.jsonl.

When harness() raises before producing a HarnessResponse, the surrounding run script
catches the exception, calls write_failure_shell_row(...), and re-raises so the
caller sees a non-zero exit. The failure row is recognizable to autoloop's
supervisor (status="failed_harness" → dashboard renders × red).

Uses libs.ledgers.LedgerWriter to preserve the managed-ledger sentinel contract
(civic_shout's results.jsonl IS managed — sentinel added in commit 42f8222).
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from libs.ledgers import LedgerWriter


def write_failure_shell_row(
    *,
    results_path: Path,
    run_id: str,
    project: str,
    comparison_group: str,
    scope: str,
    status: str = "failed_harness",
    error_msg: str,
    git_sha: str | None = None,
    extra: dict | None = None,
) -> None:
    assert status.startswith("failed_"), f"status must start with 'failed_'; got {status!r}"
    row = {
        "run_id": run_id,
        "project": project,
        "comparison_group": comparison_group,
        "scope": scope,
        "status": status,
        "primary_metric_name": None,
        "primary_metric_value": None,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "notes": f"harness exception: {error_msg[:500]}",
    }
    if extra:
        row.update(extra)

    # results_path = <project_root>/projects/<project>/runs/results.jsonl
    project_root = results_path.parent.parent.parent
    writer = LedgerWriter(project_root, project)
    writer._ensure_header(results_path)
    writer._append_line(results_path, json.dumps(row))


def write_error_traceback(*, run_dir: Path, error: BaseException) -> None:
    """Persist the full traceback to runs/{run_id}/error.txt for debugging."""
    run_dir.mkdir(parents=True, exist_ok=True)
    error_path = run_dir / "error.txt"
    with error_path.open("w") as f:
        f.write(f"Exception type: {type(error).__name__}\n")
        f.write(f"Exception message: {error}\n\n")
        f.write("Traceback:\n")
        f.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))
