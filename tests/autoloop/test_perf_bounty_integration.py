from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from libs.autoloop.perf_bounty import PerfBountyResult, run_perf_bounty

from libs.responses import (
    HarnessDataProfile,
    HarnessParallelism,
    HarnessPerformance,
    HarnessResponse,
    HarnessStageTiming,
    HarnessSummary,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PREDICTIONS_COLS = [
    "user_id",
    "email_id",
    "date_sent",
    "score",
    "score_residualized",
    "label",
    "row_loss",
    "fold",
]


def _write_frozen_predictions(path: Path, score_offset: float = 0.0) -> None:
    n = 50
    df = pl.DataFrame(
        {
            "user_id": pl.Series(list(range(n)), dtype=pl.Int64),
            "email_id": pl.Series([i % 5 for i in range(n)], dtype=pl.Int64),
            "date_sent": pl.Series(
                [f"2024-0{(i % 9) + 1}-01" for i in range(n)],
                dtype=pl.Utf8,
            ).str.to_datetime(),
            "score": pl.Series(
                [0.1 + i * 0.01 + score_offset for i in range(n)],
                dtype=pl.Float64,
            ),
            "score_residualized": pl.Series(
                [0.05 + i * 0.005 for i in range(n)],
                dtype=pl.Float64,
            ),
            "label": pl.Series([i % 2 for i in range(n)], dtype=pl.Int32),
            "row_loss": pl.Series([0.3 + i * 0.01 for i in range(n)], dtype=pl.Float64),
            "fold": pl.Series([i % 4 for i in range(n)], dtype=pl.Int32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _write_frozen_response(path: Path, total_seconds: float = 200.0) -> None:
    response = HarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc",
            primary_metric_value=0.72,
            baseline_metric_value=0.64,
            lift_vs_baseline=0.08,
            secondary_metrics={"pr_auc": 0.38},
            direction="higher_is_better",
        ),
        performance=HarnessPerformance(
            total_seconds=total_seconds,
            parallelism=HarnessParallelism(
                outer_n_jobs=1,
                inner_threads=1,
                cpu_utilization_pct=None,
            ),
            stage_timings=[
                HarnessStageTiming(stage="aggregate", seconds=total_seconds * 0.51),
                HarnessStageTiming(stage="score", seconds=total_seconds * 0.44),
            ],
        ),
        data_profile=HarnessDataProfile(
            n_rows=50,
            n_rows_source=5000,
            sample_frac=0.01,
            is_full_run=False,
            n_features=8,
        ),
        artifacts=[],
        fingerprint="deadbeef00000000",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.model_dump_json())


def _make_timing_row(wall_seconds: float = 200.0) -> dict[str, object]:
    return {
        "run_id": "run_test_01",
        "total_seconds": wall_seconds,
        "bottleneck_stage": "aggregate",
        "stage_seconds": {"aggregate": wall_seconds * 0.51, "score": wall_seconds * 0.44},
    }


# ---------------------------------------------------------------------------
# Scenario 1: Drift-injecting modification → outcome drift_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perf_bounty_drift_injecting_modification(tmp_path: Path) -> None:
    """run_perf_bounty returns outcome=drift_failed when candidate predictions drift from frozen oracle."""
    fixture_dir = tmp_path / "fixtures" / "drift"
    frozen_response = fixture_dir / "harness_response.json"
    frozen_predictions = fixture_dir / "predictions.parquet"

    _write_frozen_response(frozen_response, total_seconds=200.0)
    _write_frozen_predictions(frozen_predictions, score_offset=0.0)

    candidate_harness = (
        Path(__file__).parent.parent / "fixtures" / "perf_bounty" / "harness_drift.py"
    )

    result: PerfBountyResult = await run_perf_bounty(
        project="test_project",
        iteration_id=1,
        run_id="run_drift_01",
        frozen_response_path=frozen_response,
        frozen_predictions_path=frozen_predictions,
        timing_row=_make_timing_row(200.0),
        project_root=tmp_path,
        candidate_harness_path=candidate_harness,
    )

    assert result.outcome == "drift_failed", (
        f"Expected outcome='drift_failed' for drift-injecting harness, got {result.outcome!r}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Benign no-op modification → outcome noise (speedup <10%)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perf_bounty_noop_modification(tmp_path: Path) -> None:
    """run_perf_bounty returns outcome=noise when candidate is equivalent but has <10% speedup."""
    fixture_dir = tmp_path / "fixtures" / "noop"
    frozen_response = fixture_dir / "harness_response.json"
    frozen_predictions = fixture_dir / "predictions.parquet"

    _write_frozen_response(frozen_response, total_seconds=200.0)
    _write_frozen_predictions(frozen_predictions, score_offset=0.0)

    candidate_harness = (
        Path(__file__).parent.parent / "fixtures" / "perf_bounty" / "harness_noop.py"
    )

    result: PerfBountyResult = await run_perf_bounty(
        project="test_project",
        iteration_id=2,
        run_id="run_noop_01",
        frozen_response_path=frozen_response,
        frozen_predictions_path=frozen_predictions,
        timing_row=_make_timing_row(200.0),
        project_root=tmp_path,
        candidate_harness_path=candidate_harness,
    )

    assert result.outcome == "noise", (
        f"Expected outcome='noise' for no-op harness (speedup <10%), got {result.outcome!r}"
    )
    assert result.speedup is not None and result.speedup < 1.10, (
        f"Expected speedup < 1.10 for no-op, got {result.speedup}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Valid speedup modification → outcome merged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perf_bounty_valid_speedup_modification(tmp_path: Path) -> None:
    """run_perf_bounty returns outcome=merged when candidate is equivalent and has >=10% speedup."""
    fixture_dir = tmp_path / "fixtures" / "speedup"
    frozen_response = fixture_dir / "harness_response.json"
    frozen_predictions = fixture_dir / "predictions.parquet"

    _write_frozen_response(frozen_response, total_seconds=200.0)
    _write_frozen_predictions(frozen_predictions, score_offset=0.0)

    candidate_harness = (
        Path(__file__).parent.parent / "fixtures" / "perf_bounty" / "harness_fast.py"
    )

    result: PerfBountyResult = await run_perf_bounty(
        project="test_project",
        iteration_id=3,
        run_id="run_speedup_01",
        frozen_response_path=frozen_response,
        frozen_predictions_path=frozen_predictions,
        timing_row=_make_timing_row(200.0),
        project_root=tmp_path,
        candidate_harness_path=candidate_harness,
    )

    assert result.outcome == "merged", (
        f"Expected outcome='merged' for fast-equivalent harness (speedup >=10%), got {result.outcome!r}"
    )
    assert result.speedup is not None and result.speedup >= 1.10, (
        f"Expected speedup >= 1.10 for merged outcome, got {result.speedup}"
    )
