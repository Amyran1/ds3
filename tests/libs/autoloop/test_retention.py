from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from libs.autoloop.retention import prune_old_predictions


def _make_run(runs_dir: Path, run_id: str, file_size: int = 100) -> Path:
    parquet = runs_dir / run_id / "artifacts" / "predictions.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"x" * file_size)
    return parquet


def _write_results(runs_dir: Path, rows: list[dict]) -> None:
    results = runs_dir / "results.jsonl"
    with results.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _results_row(
    run_id: str,
    metric: float,
    comparison_group: str = "cg1",
    scope: str = "comparison",
    status: str = "completed",
    primary_metric: str = "roc_auc",
) -> dict:
    return {
        "run_id": run_id,
        primary_metric: metric,
        "comparison_group": comparison_group,
        "scope": scope,
        "status": status,
    }


def _make_project(tmp_path: Path, project: str = "proj") -> tuple[Path, Path]:
    project_root = tmp_path
    runs_dir = project_root / "projects" / project / "runs"
    runs_dir.mkdir(parents=True)
    return project_root, runs_dir


def test_prune_keeps_top_k_plus_current(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    run_ids = [f"run_0{i}" for i in range(1, 9)]
    for run_id in run_ids:
        _make_run(runs_dir, run_id)

    rows = [_results_row(rid, metric=float(i)) for i, rid in enumerate(run_ids)]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=3,
        current_run_id="run_01",
    )

    assert "run_08" in report.kept_run_ids
    assert "run_07" in report.kept_run_ids
    assert "run_06" in report.kept_run_ids
    assert "run_01" in report.kept_run_ids

    for rid in ["run_02", "run_03", "run_04", "run_05"]:
        assert not (runs_dir / rid / "artifacts" / "predictions.parquet").exists()


def test_prune_higher_is_better_keeps_highest(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 6):
        _make_run(runs_dir, f"run_0{i}")

    rows = [_results_row(f"run_0{i}", metric=float(i)) for i in range(1, 6)]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=2,
    )

    assert "run_05" in report.kept_run_ids
    assert "run_04" in report.kept_run_ids
    for rid in ["run_01", "run_02", "run_03"]:
        assert rid not in report.kept_run_ids


def test_prune_lower_is_better_keeps_lowest(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 6):
        _make_run(runs_dir, f"run_0{i}")

    rows = [_results_row(f"run_0{i}", metric=float(i)) for i in range(1, 6)]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="lower_is_better",
        top_k=2,
    )

    assert "run_01" in report.kept_run_ids
    assert "run_02" in report.kept_run_ids
    for rid in ["run_03", "run_04", "run_05"]:
        assert rid not in report.kept_run_ids


def test_prune_current_run_id_always_protected(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 4):
        _make_run(runs_dir, f"run_0{i}")

    rows = [_results_row(f"run_0{i}", metric=float(i)) for i in range(1, 4)]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=1,
        current_run_id="run_01",
    )

    assert "run_01" in report.kept_run_ids


def test_prune_champion_md_run_id_protected(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 6):
        _make_run(runs_dir, f"run_0{i}")

    rows = [_results_row(f"run_0{i}", metric=float(i)) for i in range(1, 6)]
    _write_results(runs_dir, rows)

    champion_md = runs_dir / "CHAMPION.md"
    champion_md.write_text("Champion: run_02\n\nSome other content.")

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=1,
    )

    assert "run_02" in report.kept_run_ids


def test_prune_scope_filter_excludes_reproduction(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 4):
        _make_run(runs_dir, f"run_0{i}")

    rows = [
        _results_row("run_01", metric=0.99, scope="reproduction"),
        _results_row("run_02", metric=0.60),
        _results_row("run_03", metric=0.70),
    ]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=1,
    )

    assert "run_01" not in report.kept_run_ids


def test_prune_no_runs_dir_returns_empty_report(tmp_path: Path) -> None:
    project_root = tmp_path

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
    )

    assert report.kept_run_ids == []
    assert report.files_removed == 0


def test_prune_jsonl_malformed_line_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 4):
        _make_run(runs_dir, f"run_0{i}")

    results = runs_dir / "results.jsonl"
    with results.open("w") as f:
        f.write("{bad json line\n")
        f.write(json.dumps(_results_row("run_02", metric=0.8)) + "\n")
        f.write(json.dumps(_results_row("run_03", metric=0.9)) + "\n")

    with caplog.at_level(logging.WARNING):
        report = prune_old_predictions(
            project_root=project_root,
            project="proj",
            comparison_group="cg1",
            primary_metric="roc_auc",
            metric_direction="higher_is_better",
            top_k=1,
        )

    assert "run_03" in report.kept_run_ids


def test_prune_bytes_freed_reflects_deleted_sizes(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i, size in zip(range(1, 5), [200, 300, 400, 500]):
        _make_run(runs_dir, f"run_0{i}", file_size=size)

    rows = [_results_row(f"run_0{i}", metric=float(i)) for i in range(1, 5)]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=2,
    )

    assert report.bytes_freed == 200 + 300
    assert report.files_removed == 2


def test_prune_kept_run_ids_matches_protected_set(tmp_path: Path) -> None:
    project_root, runs_dir = _make_project(tmp_path)

    for i in range(1, 5):
        _make_run(runs_dir, f"run_0{i}")

    rows = [_results_row(f"run_0{i}", metric=float(i)) for i in range(1, 5)]
    _write_results(runs_dir, rows)

    report = prune_old_predictions(
        project_root=project_root,
        project="proj",
        comparison_group="cg1",
        primary_metric="roc_auc",
        metric_direction="higher_is_better",
        top_k=2,
        current_run_id="run_01",
    )

    assert set(report.kept_run_ids) == {"run_03", "run_04", "run_01"}
