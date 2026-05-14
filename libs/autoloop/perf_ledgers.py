"""Append-only ledger writers for perf-bounty attempt and win records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "PerfAttempt",
    "PerfWin",
    "append_attempt",
    "append_win",
    "save_diff",
]

_ATTEMPT_MANAGED_BY = "libs.autoloop.perf_ledgers"
_ATTEMPT_SCHEMA_FAMILY = "perf_attempt"
_WIN_SCHEMA_FAMILY = "perf_win"


class PerfAttempt(BaseModel):
    model_config = {"populate_by_name": True}

    schema_version: Literal["perf_attempt/v1"] = Field(default="perf_attempt/v1", alias="schema")
    ts: str
    project: str
    iteration_id: int
    run_id: str
    target_file: str
    bottleneck_stage: str
    fingerprint: str
    technique: str
    wall_before_s: float
    wall_after_s: float | None
    speedup: float | None
    equivalence_passed: bool
    failure_reason: str | None
    drift_pct: float | None


class PerfWin(BaseModel):
    model_config = {"populate_by_name": True}

    schema_version: Literal["perf_win/v1"] = Field(default="perf_win/v1", alias="schema")
    ts: str
    project: str
    iteration_id: int
    run_id: str
    target_file: str
    bottleneck_stage: str
    fingerprint: str
    technique: str
    wall_before_s: float
    wall_after_s: float
    speedup: float
    seconds_saved: float
    commit_sha: str
    diff_bank_path: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_sentinel(f: "os.IO[str]", managed_by: str, schema_family: str) -> None:
    sentinel = json.dumps(
        {
            "_managed_by": managed_by,
            "_schema_family": schema_family,
            "_created_utc": _now_iso(),
        }
    )
    f.write(sentinel + "\n")
    f.flush()
    os.fsync(f.fileno())


def _ensure_header(path: Path, managed_by: str, schema_family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w") as f:
            _write_sentinel(f, managed_by, schema_family)


def _append_line(path: Path, line: str) -> None:
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_attempt(repo_root: Path, row: PerfAttempt) -> None:
    path = repo_root / "knowledge" / "perf_attempts.jsonl"
    _ensure_header(path, _ATTEMPT_MANAGED_BY, _ATTEMPT_SCHEMA_FAMILY)
    _append_line(path, row.model_dump_json(by_alias=True))


def append_win(repo_root: Path, project_root: Path, win: PerfWin) -> None:
    path = project_root / "autoloop" / "perf_wins.jsonl"
    _ensure_header(path, _ATTEMPT_MANAGED_BY, _WIN_SCHEMA_FAMILY)
    _append_line(path, win.model_dump_json(by_alias=True))


def save_diff(
    repo_root: Path,
    fingerprint_tag: str,
    project: str,
    run_id: str,
    diff_text: str,
    meta: dict,
) -> Path:
    tag_dir = fingerprint_tag if fingerprint_tag else "_untagged"
    iso_ts = _now_iso().replace(":", "-").replace("+", "Z").split(".")[0]
    filename = f"{iso_ts}_{project}_{run_id}.diff"
    diff_path = repo_root / "knowledge" / "perf_diff_bank" / tag_dir / filename
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(diff_text)
    meta_path = diff_path.with_suffix(".diff.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return diff_path
