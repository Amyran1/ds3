"""Fixture-driven render tests for libs.leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.leaderboard import render


def _managed_by_sentinel() -> dict:
    return {
        "_managed_by": "libs.ledgers.LedgerWriter",
        "schema_version": "ledger/v1",
        "created_at_utc": "2026-05-13T00:00:00Z",
    }


def _harness_row(
    run_id: str, primary: float, scope: str = "comparison", sample_frac: float = 0.05
) -> dict:
    return {
        "run_id": run_id,
        "project": "test_project",
        "comparison_group": "group_v1",
        "scope": scope,
        "status": "completed",
        "verdict": "smoke_only",
        "recorded_at_utc": "2026-05-13T10:00:00Z",
        "git_sha": "abc1234",
        "data_fingerprint": "",
        "outcome_version": "v1",
        "primary_metric_name": "roc_auc_residualized",
        "primary_metric_value": primary,
        "baseline_metric_value": 0.5,
        "lift_vs_baseline": primary - 0.5,
        "metric_direction": "higher_is_better",
        "n_rows": 100000,
        "sample_frac": sample_frac,
        "is_full_run": False,
        "n_features": 8,
        "total_seconds": 120.0,
        "cpu_utilization_pct": 700.0,
        "fingerprint": "fp123",
        "notes": "",
        "ci_low": primary - 0.01,
        "ci_high": primary + 0.01,
        "elapsed_seconds": 120.0,
        "roc_auc_raw_pair": primary + 0.20,
    }


def _eda_row(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "project": "test_project",
        "comparison_group": "group_v1",
        "scope": "comparison",
        "recorded_at_utc": "2026-05-13T10:00:00Z",
        "git_sha": "abc1234",
        "eda_outcome_prevalence": 0.09,
        "eda_drift_score": 0.1,
        "eda_n_findings_warning": 2,
        "eda_n_findings_error": 0,
        "eda_top_finding_id": "eda_01",
        "eda_total_seconds": 3.5,
    }


def _disc_row(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "project": "test_project",
        "comparison_group": "group_v1",
        "scope": "comparison",
        "recorded_at_utc": "2026-05-13T10:00:00Z",
        "git_sha": "abc1234",
        "discovery_top_slice_residual_lift": 0.05,
        "discovery_embedding_region_count": 0,
        "discovery_label_noise_rate": 0.01,
        "discovery_priority_score_max": 0.7,
        "discovery_overall_health": "green",
        "discovery_n_findings_warning": 1,
        "discovery_n_findings_error": 0,
        "discovery_total_seconds": 10.0,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_renders_with_only_harness_ledger(tmp_path: Path) -> None:
    runs_dir = tmp_path / "projects" / "test_project" / "runs"
    runs_dir.mkdir(parents=True)

    _write_jsonl(
        runs_dir / "results.jsonl",
        [
            _managed_by_sentinel(),
            _harness_row("run_01", 0.6755, scope="comparison", sample_frac=0.05),
            _harness_row("run_01_smoke", 0.6554, scope="smoke", sample_frac=0.001),
            _harness_row("run_01_1pct", 0.6732, scope="smoke", sample_frac=0.01),
        ],
    )

    out_path = tmp_path / "out" / "leaderboard.html"

    import libs.leaderboard as lb

    orig_root = lb.ROOT
    lb.ROOT = tmp_path
    try:
        result = render("test_project", out_path=out_path)
    finally:
        lb.ROOT = orig_root

    assert result == out_path
    assert out_path.exists()
    content = out_path.read_text()

    assert len(content) > 5000
    assert "Champion" in content
    assert "X × 1.10" in content or "0.7430" in content or "target" in content.lower()
    assert "EDA + discovery ledgers not yet present" in content
    assert "drift check deferred" in content


def test_renders_three_way_join(tmp_path: Path) -> None:
    runs_dir = tmp_path / "projects" / "test_project" / "runs"
    runs_dir.mkdir(parents=True)

    _write_jsonl(
        runs_dir / "results.jsonl",
        [
            _harness_row("run_01", 0.6755, scope="comparison"),
            _harness_row("run_01_smoke", 0.6554, scope="smoke", sample_frac=0.001),
            _harness_row("run_02", 0.6800, scope="comparison"),
        ],
    )
    _write_jsonl(
        runs_dir / "eda_results.jsonl",
        [
            _eda_row("run_01"),
            _eda_row("run_01_smoke"),
            _eda_row("run_02"),
        ],
    )
    _write_jsonl(
        runs_dir / "discovery_results.jsonl",
        [
            _disc_row("run_01"),
            _disc_row("run_01_smoke"),
            _disc_row("run_02"),
        ],
    )

    out_path = tmp_path / "out" / "leaderboard.html"

    import libs.leaderboard as lb

    orig_root = lb.ROOT
    lb.ROOT = tmp_path
    try:
        result = render("test_project", out_path=out_path)
    finally:
        lb.ROOT = orig_root

    assert result == out_path
    content = out_path.read_text()

    assert len(content) > 5000
    assert "Champion" in content
    assert "eda_warn" in content or "eda_n_findings" in content or "eda_warn" in content
    assert "n/a" not in content.split("Three-Way Drift")[0] or "eda_warn" in content
    assert "EDA + discovery ledgers not yet present" not in content
