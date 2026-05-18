from __future__ import annotations

import datetime
import json

import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.runs._ledger_fallback import (
    write_error_traceback,
    write_failure_shell_row,
)


def test_write_failure_shell_row_appends(tmp_path):
    results_path = tmp_path / "results.jsonl"
    write_failure_shell_row(
        results_path=results_path,
        run_id="run_test",
        project="civic_shout",
        comparison_group="g",
        scope="comparison",
        error_msg="boom",
    )
    all_lines = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    data_rows = [r for r in all_lines if "_managed_by" not in r]
    assert len(data_rows) == 1
    row = data_rows[0]
    assert row["run_id"] == "run_test"
    assert row["status"] == "failed_harness"
    assert "boom" in row["notes"]
    assert "recorded_at_utc" in row
    assert row["primary_metric_value"] is None


def test_write_failure_shell_row_preserves_existing(tmp_path):
    sentinel = {
        "_managed_by": "libs.ledgers.LedgerWriter",
        "schema_version": "ledger/v1",
        "created_at_utc": "2026-01-01T00:00:00",
    }
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(sentinel) + "\n" + '{"run_id": "run_prior", "status": "completed"}\n'
    )
    write_failure_shell_row(
        results_path=results_path,
        run_id="run_test",
        project="civic_shout",
        comparison_group="g",
        scope="comparison",
        error_msg="boom",
    )
    all_lines = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    data_rows = [r for r in all_lines if "_managed_by" not in r]
    assert len(data_rows) == 2
    assert data_rows[0]["run_id"] == "run_prior"
    assert data_rows[1]["status"] == "failed_harness"


def test_write_failure_shell_row_creates_parent_dir(tmp_path):
    results_path = tmp_path / "new_dir" / "results.jsonl"
    write_failure_shell_row(
        results_path=results_path,
        run_id="run_test",
        project="civic_shout",
        comparison_group="g",
        scope="comparison",
        error_msg="boom",
    )
    assert results_path.exists()


def test_status_must_start_with_failed(tmp_path):
    results_path = tmp_path / "results.jsonl"
    with pytest.raises(AssertionError):
        write_failure_shell_row(
            results_path=results_path,
            run_id="run_test",
            project="civic_shout",
            comparison_group="g",
            scope="comparison",
            status="completed",
            error_msg="x",
        )


def test_write_error_traceback(tmp_path):
    try:
        raise ValueError("test error")
    except ValueError as e:
        write_error_traceback(run_dir=tmp_path, error=e)
    error_path = tmp_path / "error.txt"
    assert error_path.exists()
    content = error_path.read_text()
    assert "ValueError" in content
    assert "test error" in content
    assert "Traceback" in content


def test_run_01_writes_failure_row_on_harness_exception(tmp_path, monkeypatch):
    """Integration: monkeypatch harness() to raise → run_01 main writes failure_harness row + re-raises."""
    from projects.civic_shout_action_rate_increase.runs import run_01

    sentinel = {
        "_managed_by": "libs.ledgers.LedgerWriter",
        "schema_version": "ledger/v1",
        "created_at_utc": "2026-01-01T00:00:00",
    }
    fake_ledger = tmp_path / "results.jsonl"
    fake_ledger.write_text(json.dumps(sentinel) + "\n")

    monkeypatch.setattr(run_01, "_LEDGER", fake_ledger)

    minimal_df = pl.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "email_id": ["e1", "e2"],
            "date_sent": pl.Series([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]),
            "actioned_24h": [0, 1],
            "actioned_last_1": [0, 0],
            "actioned_last_3": [0, 0],
            "actioned_last_5": [0, 0],
            "actioned_last_10": [0, 0],
            "sends_since_last_action": [0, 0],
            "action_in_last_5": [0, 0],
            "is_in_action_streak": [0, 0],
            "lifetime_actions_prior": [0, 0],
        }
    )

    class _FakeCache:
        def get(self, version: int) -> pl.DataFrame:
            return minimal_df

    fake_cache = _FakeCache()
    monkeypatch.setattr(run_01, "user_emails_cache", fake_cache)
    monkeypatch.setattr(run_01, "recency_feature_cache", fake_cache)

    def _fake_harness(*args, **kwargs):
        raise RuntimeError("planted_failure_for_test")

    monkeypatch.setattr(run_01, "harness", _fake_harness)

    with pytest.raises(RuntimeError, match="planted_failure_for_test"):
        run_01.main(
            run_id="test_failure_integration",
            sample_frac=None,
            scope="comparison",
        )

    rows = [json.loads(line) for line in fake_ledger.read_text().splitlines() if line]
    failure_rows = [r for r in rows if r.get("run_id") == "test_failure_integration"]
    assert len(failure_rows) == 1
    assert failure_rows[0]["status"] == "failed_harness"
    assert "planted_failure_for_test" in failure_rows[0].get("notes", "")
