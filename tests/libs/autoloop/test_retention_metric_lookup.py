from __future__ import annotations

import json
from pathlib import Path

from libs.autoloop.retention import (
    _detect_champion_from_results,
    _lookup_row_metric,
    prune_old_predictions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_typed_row_lookup(tmp_path: Path) -> None:
    """primary_metric_name + primary_metric_value (typed schema) → metric found."""
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "run_id": "run_01",
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
                "primary_metric_name": "auc",
                "primary_metric_value": 0.85,
            },
            {
                "run_id": "run_02",
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
                "primary_metric_name": "auc",
                "primary_metric_value": 0.90,
            },
        ],
    )
    champion = _detect_champion_from_results(
        results_path,
        comparison_group="cg1",
        primary_metric="auc",
        metric_direction="higher_is_better",
    )
    assert champion == "run_02", f"expected run_02 (higher AUC), got {champion}"


def test_legacy_flat_lookup_fallback(tmp_path: Path) -> None:
    """{"auc": 0.85} legacy flat schema → fallback still works."""
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "run_id": "run_01",
                "auc": 0.85,
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
            },
            {
                "run_id": "run_02",
                "auc": 0.90,
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
            },
        ],
    )
    champion = _detect_champion_from_results(
        results_path,
        comparison_group="cg1",
        primary_metric="auc",
        metric_direction="higher_is_better",
    )
    assert champion == "run_02"


def test_mixed_typed_and_legacy(tmp_path: Path) -> None:
    """One typed + one legacy row in same file → both contribute to champion selection."""
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "run_id": "run_01",
                "auc": 0.85,
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
            },
            {
                "run_id": "run_02",
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
                "primary_metric_name": "auc",
                "primary_metric_value": 0.90,
            },
        ],
    )
    champion = _detect_champion_from_results(
        results_path,
        comparison_group="cg1",
        primary_metric="auc",
        metric_direction="higher_is_better",
    )
    assert champion == "run_02"


def test_empty_ledger(tmp_path: Path) -> None:
    """No rows → no champion, no exception."""
    results_path = tmp_path / "results.jsonl"
    results_path.touch()
    champion = _detect_champion_from_results(
        results_path,
        comparison_group="cg1",
        primary_metric="auc",
        metric_direction="higher_is_better",
    )
    assert champion is None


def test_prune_protects_champion(tmp_path: Path) -> None:
    """prune_old_predictions does NOT prune the champion's predictions directory."""
    project = "test_proj"
    runs_dir = tmp_path / "projects" / project / "runs"
    runs_dir.mkdir(parents=True)

    results_path = runs_dir / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "run_id": "run_01",
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
                "primary_metric_name": "auc",
                "primary_metric_value": 0.85,
            },
            {
                "run_id": "run_02",
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
                "primary_metric_name": "auc",
                "primary_metric_value": 0.95,
            },
            {
                "run_id": "run_03",
                "status": "completed",
                "comparison_group": "cg1",
                "scope": "comparison",
                "primary_metric_name": "auc",
                "primary_metric_value": 0.80,
            },
        ],
    )

    for rid in ["run_01", "run_02", "run_03"]:
        artifacts = runs_dir / rid / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "predictions.parquet").write_text("fake")

    report = prune_old_predictions(
        project_root=tmp_path,
        project=project,
        comparison_group="cg1",
        primary_metric="auc",
        metric_direction="higher_is_better",
        top_k=1,
    )

    champion_parquet = runs_dir / "run_02" / "artifacts" / "predictions.parquet"
    assert champion_parquet.exists(), "champion run_02 predictions.parquet was pruned — BUG"
    assert "run_02" not in report.removed_run_ids
    assert "run_01" in report.removed_run_ids, "run_01 should have been pruned"
    assert "run_03" in report.removed_run_ids, "run_03 should have been pruned"
    assert not (runs_dir / "run_01" / "artifacts" / "predictions.parquet").exists()
    assert not (runs_dir / "run_03" / "artifacts" / "predictions.parquet").exists()


def test_typed_row_lookup_lower_is_better(tmp_path: Path) -> None:
    """metric_direction='lower_is_better' → lowest typed metric wins."""
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "run_id": "run_01",
                "status": "completed",
                "comparison_group": "g",
                "scope": "comparison",
                "primary_metric_name": "rmse",
                "primary_metric_value": 0.10,
            },
            {
                "run_id": "run_02",
                "status": "completed",
                "comparison_group": "g",
                "scope": "comparison",
                "primary_metric_name": "rmse",
                "primary_metric_value": 0.20,
            },
        ],
    )
    champion = _detect_champion_from_results(
        results_path,
        comparison_group="g",
        primary_metric="rmse",
        metric_direction="lower_is_better",
    )
    assert champion == "run_01", f"expected run_01 (lower RMSE), got {champion}"
