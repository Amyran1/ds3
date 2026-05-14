from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from projects.civic_shout_action_rate_increase.runs.timing_helper import emit_timing_row


def _make_mock_response(n_rows: int = 5000, n_features: int = 8) -> MagicMock:
    response = MagicMock()

    stage1 = MagicMock()
    stage1.stage = "data_conversion"
    stage1.seconds = 12.5

    stage2 = MagicMock()
    stage2.stage = "fit"
    stage2.seconds = 25.0

    perf = MagicMock()
    perf.total_seconds = 42.0
    perf.stage_timings = [stage1, stage2]
    perf.rows_per_second = 500.0
    perf.bottleneck_stage = "fit"
    perf.peak_memory_mb = None
    perf.parallelism = MagicMock()
    perf.parallelism.cpu_utilization_pct = 350.0
    response.performance = perf

    data = MagicMock()
    data.n_rows = n_rows
    data.n_features = n_features
    data.train_rows = n_rows * 4
    data.sample_frac = 0.001
    response.data = data

    result_row = MagicMock()
    result_row.n_rows = n_rows
    result_row.n_feature_cols = n_features
    result_row.walk_forward_n_folds = 8
    response.to_result_row.return_value = result_row

    return response


def _make_metadata(run_id: str = "test_run") -> MagicMock:
    meta = MagicMock()
    meta.run_id = run_id
    meta.project = "civic_shout_action_rate_increase"
    meta.scope = "smoke"
    meta.comparison_group = "test_group_v1"
    meta.sample_frac = 0.001
    meta.git_sha = "abc1234"
    meta.recorded_at_utc = "2026-05-14T00:00:00+00:00"
    return meta


def test_emit_timing_row_creates_file(tmp_path: Path) -> None:
    metadata = _make_metadata("test_run_create")
    response = _make_mock_response()

    result_path = emit_timing_row(metadata, response, tmp_path)

    assert result_path.exists()
    lines = result_path.read_text().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])
    assert row["run_id"] == "test_run_create"
    assert row["project"] == "civic_shout_action_rate_increase"
    assert row["scope"] == "smoke"
    assert row["total_seconds"] == 42.0
    assert row["bottleneck_stage"] == "fit"
    assert row["cpu_utilization_pct"] == 350.0
    assert row["n_test_rows"] == 5000
    assert row["walk_forward_n_folds"] == 8
    assert isinstance(row["stage_seconds"], dict)
    assert len(row["stage_seconds"]) >= 1
    assert row["stage_seconds"]["data_conversion"] == 12.5
    assert row["stage_seconds"]["fit"] == 25.0


def test_emit_timing_row_appends(tmp_path: Path) -> None:
    meta1 = _make_metadata("run_a")
    meta2 = _make_metadata("run_b")
    response = _make_mock_response()

    emit_timing_row(meta1, response, tmp_path)
    emit_timing_row(meta2, response, tmp_path)

    ledger = tmp_path / "timing_performance.jsonl"
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2

    row1 = json.loads(lines[0])
    row2 = json.loads(lines[1])
    assert row1["run_id"] == "run_a"
    assert row2["run_id"] == "run_b"
