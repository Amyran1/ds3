"""Fast-equivalent harness fixture for perf_bounty integration tests.

Copies frozen predictions unchanged (equivalence passes) and reports
wall time 50% faster than the oracle (speedup >= 10% → outcome=merged).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from libs.responses import (
    HarnessDataProfile,
    HarnessParallelism,
    HarnessPerformance,
    HarnessResponse,
    HarnessStageTiming,
    HarnessSummary,
)


def run_harness(
    frozen_predictions_path: Path,
    output_predictions_path: Path,
    oracle_wall_seconds: float = 200.0,
    **_kwargs: Any,
) -> HarnessResponse:
    """Copy frozen predictions without modification; report 50% speedup."""
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen_predictions_path, output_predictions_path)

    total_seconds = oracle_wall_seconds * 0.50
    return HarnessResponse(
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
            parallelism=HarnessParallelism(outer_n_jobs=1, inner_threads=1),
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
