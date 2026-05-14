"""Drift-injecting harness fixture for perf_bounty integration tests.

Returns predictions with scores shifted by 0.1 relative to the frozen oracle,
causing predictions_equivalent to fail the correlation threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

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
    **_kwargs: Any,
) -> HarnessResponse:
    """Read frozen predictions, inject drift into scores, write drifted parquet."""
    df = pl.read_parquet(frozen_predictions_path)
    drifted = df.with_columns((pl.col("score") + 0.1).alias("score"))
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    drifted.write_parquet(output_predictions_path)

    total_seconds = 180.0
    return HarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            baseline_metric_value=0.64,
            lift_vs_baseline=0.16,
            secondary_metrics={"pr_auc": 0.45},
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
        fingerprint="drifted00000000",
    )
