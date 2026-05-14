from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.autoloop.perf_ledgers import (
    PerfAttempt,
    PerfWin,
    append_attempt,
    append_win,
    save_diff,
)
from libs.ledgers import LEDGER_SENTINEL_KEY, LEDGER_SENTINEL_VALUE, LedgerError, LedgerWriter

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_attempt(**overrides: object) -> PerfAttempt:
    defaults: dict[str, object] = {
        "ts": "2026-05-14T00:00:00+00:00",
        "project": "civic_shout",
        "iteration_id": 1,
        "run_id": "run_01",
        "target_file": "harness.py",
        "bottleneck_stage": "aggregate",
        "fingerprint": "iterrows_api_call",
        "technique": "vectorize",
        "wall_before_s": 200.0,
        "wall_after_s": None,
        "speedup": None,
        "equivalence_passed": False,
        "failure_reason": "drift",
        "drift_pct": None,
    }
    defaults.update(overrides)
    return PerfAttempt(**defaults)


def _make_win(**overrides: object) -> PerfWin:
    defaults: dict[str, object] = {
        "ts": "2026-05-14T00:00:00+00:00",
        "project": "civic_shout",
        "iteration_id": 1,
        "run_id": "run_01",
        "target_file": "harness.py",
        "bottleneck_stage": "aggregate",
        "fingerprint": "iterrows_api_call",
        "technique": "vectorize",
        "wall_before_s": 200.0,
        "wall_after_s": 160.0,
        "speedup": 1.25,
        "seconds_saved": 40.0,
        "commit_sha": "abc123",
        "diff_bank_path": "knowledge/perf_diff_bank/iterrows_api_call/ts_civic_shout_run_01.diff",
    }
    defaults.update(overrides)
    return PerfWin(**defaults)


# ---------------------------------------------------------------------------
# append_attempt
# ---------------------------------------------------------------------------


def test_append_attempt_creates_sentinel_on_first_write(tmp_path: Path) -> None:
    attempt = _make_attempt()
    append_attempt(tmp_path, attempt)

    ledger_path = tmp_path / "knowledge" / "perf_attempts.jsonl"
    assert ledger_path.exists()

    lines = [l for l in ledger_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    sentinel = json.loads(lines[0])
    assert sentinel["_managed_by"] == "libs.autoloop.perf_ledgers"
    assert sentinel["_schema_family"] == "perf_attempt"

    row = json.loads(lines[1])
    assert row["schema"] == "perf_attempt/v1"
    assert row["project"] == "civic_shout"


def test_append_attempt_appends_subsequent_rows(tmp_path: Path) -> None:
    append_attempt(tmp_path, _make_attempt(iteration_id=1))
    append_attempt(tmp_path, _make_attempt(iteration_id=2))

    ledger_path = tmp_path / "knowledge" / "perf_attempts.jsonl"
    lines = [l for l in ledger_path.read_text().splitlines() if l.strip()]

    assert len(lines) == 3
    assert json.loads(lines[0]).get("_managed_by") == "libs.autoloop.perf_ledgers"
    assert json.loads(lines[1])["iteration_id"] == 1
    assert json.loads(lines[2])["iteration_id"] == 2


# ---------------------------------------------------------------------------
# append_win
# ---------------------------------------------------------------------------


def test_append_win_creates_project_local_ledger(tmp_path: Path) -> None:
    project_root = tmp_path / "projects" / "civic_shout"
    win = _make_win()
    append_win(repo_root=tmp_path, project_root=project_root, win=win)

    ledger_path = project_root / "autoloop" / "perf_wins.jsonl"
    assert ledger_path.exists()

    lines = [l for l in ledger_path.read_text().splitlines() if l.strip()]
    sentinel = json.loads(lines[0])
    assert sentinel["_schema_family"] == "perf_win"

    row = json.loads(lines[1])
    assert row["schema"] == "perf_win/v1"


# ---------------------------------------------------------------------------
# save_diff
# ---------------------------------------------------------------------------


def test_save_diff_writes_diff_and_meta(tmp_path: Path) -> None:
    diff_path = save_diff(
        repo_root=tmp_path,
        fingerprint_tag="iterrows_api_call",
        project="civic_shout",
        run_id="run_01",
        diff_text="diff content",
        meta={"speedup": 2.0, "seconds_saved": 100.0},
    )

    assert diff_path.exists()
    assert diff_path.read_text() == "diff content"
    assert "iterrows_api_call" in str(diff_path)
    assert "civic_shout" in diff_path.name

    meta_path = diff_path.with_suffix(".diff.meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["speedup"] == 2.0
    assert meta["seconds_saved"] == 100.0


def test_save_diff_uses_untagged_for_empty_fingerprint(tmp_path: Path) -> None:
    diff_path = save_diff(
        repo_root=tmp_path,
        fingerprint_tag="",
        project="civic_shout",
        run_id="run_01",
        diff_text="diff content",
        meta={"speedup": 1.5, "seconds_saved": 50.0},
    )

    assert "_untagged" in str(diff_path)


# ---------------------------------------------------------------------------
# Regression: LedgerWriter.assert_managed contract
# ---------------------------------------------------------------------------


def test_ledger_write_rejected_without_sentinel(tmp_path: Path) -> None:
    ledger_path = tmp_path / "perf_attempts.jsonl"
    row = json.dumps({"ts": "2026-05-14T00:00:00Z", "project": "test", "outcome": "noise"})
    ledger_path.write_text(row + "\n")

    with pytest.raises(LedgerError):
        LedgerWriter.assert_managed(ledger_path)
