from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.ledgers import LedgerError, LedgerWriter
from libs.responses import RunResultRow


def _make_result_row(**overrides: object) -> RunResultRow:
    defaults: dict[str, object] = {
        "project": "test_project",
        "run_id": "run_01",
        "comparison_group": "cg_v1",
        "scope": "smoke",
        "status": "completed",
        "verdict": "smoke_only",
        "recorded_at_utc": "2026-01-01T00:00:00+00:00",
        "git_sha": "abc1234",
        "outcome_version": "v1",
        "primary_metric_name": "roc_auc",
        "primary_metric_value": 0.6,
        "baseline_metric_value": 0.5,
        "lift_vs_baseline": 0.1,
        "metric_direction": "higher_is_better",
        "n_rows": 1000,
        "sample_frac": 0.01,
        "is_full_run": False,
        "n_features": 5,
        "total_seconds": 10.0,
        "cpu_utilization_pct": None,
        "fingerprint": "",
        "notes": "",
    }
    defaults.update(overrides)
    return RunResultRow.model_validate(defaults)


def _sentinel_obj() -> dict[str, str]:
    return {
        "_managed_by": "libs.ledgers.LedgerWriter",
        "schema_version": "ledger/v1",
    }


def test_assert_managed_raises_on_unmanaged_file(tmp_path: Path) -> None:
    ledger = tmp_path / "results.jsonl"
    row = _make_result_row()
    ledger.write_text(row.model_dump_json(by_alias=True) + "\n")
    with pytest.raises(LedgerError, match="was not created by LedgerWriter"):
        LedgerWriter.assert_managed(ledger)


def test_assert_managed_passes_after_sentinel_prepend(tmp_path: Path) -> None:
    ledger = tmp_path / "results.jsonl"
    row = _make_result_row()
    ledger.write_text(row.model_dump_json(by_alias=True) + "\n")

    with pytest.raises(LedgerError):
        LedgerWriter.assert_managed(ledger)

    sentinel = json.dumps(
        {
            "_managed_by": "libs.ledgers.LedgerWriter",
            "schema_version": "ledger/v1",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
        }
    )
    existing = ledger.read_text()
    ledger.write_text(sentinel + "\n" + existing)

    result = LedgerWriter.assert_managed(ledger)
    assert result is None


def test_append_atomic_succeeds_against_sentineled_existing_file(tmp_path: Path) -> None:
    writer = LedgerWriter(tmp_path, "test_project")
    runs_dir = tmp_path / "projects" / "test_project" / "runs"
    runs_dir.mkdir(parents=True)

    row1 = _make_result_row(run_id="run_01")
    row2 = _make_result_row(run_id="run_02")
    sentinel = json.dumps(
        {
            "_managed_by": "libs.ledgers.LedgerWriter",
            "schema_version": "ledger/v1",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
        }
    )
    results_path = runs_dir / "results.jsonl"
    results_path.write_text(
        sentinel
        + "\n"
        + row1.model_dump_json(by_alias=True)
        + "\n"
        + row2.model_dump_json(by_alias=True)
        + "\n"
    )

    row3 = _make_result_row(run_id="run_03")
    writer.append_atomic(row3, None, None)

    lines = [l for l in results_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4
    appended = json.loads(lines[-1])
    assert appended["run_id"] == "run_03"
