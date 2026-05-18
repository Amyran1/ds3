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


def test_run_01_writes_error_txt_on_run_record_exception(tmp_path, monkeypatch):
    """Integration: monkeypatch run_record() to raise → run_01 main re-raises and writes error.txt.

    Ledger failure-shell rows are tested in tests/libs/test_run_record.py (libs scope).
    """
    from libs import run_record as run_record_mod
    from projects.civic_shout_action_rate_increase.runs import run_01

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

    def _fake_run_record(**kwargs):
        raise RuntimeError("planted_failure_for_test")

    monkeypatch.setattr(run_record_mod, "run_record", _fake_run_record)
    monkeypatch.setattr(run_01, "run_record", _fake_run_record)

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    monkeypatch.setattr(run_01, "_RUNS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="planted_failure_for_test"):
        run_01.main(
            run_id="run_01",
            sample_frac=None,
            scope="smoke",
        )

    error_txt = tmp_path / "run_01" / "error.txt"
    assert error_txt.exists(), "error.txt should be written on harness failure"
    assert "planted_failure_for_test" in error_txt.read_text()
