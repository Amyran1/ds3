from __future__ import annotations

import time

import polars as pl

from libs.responses import (
    EDAConfig,
    EDAPerformance,
    EDAResponse,
    EDAScores,
    HarnessStageTiming,
)


def eda(
    *,
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    eda_config: EDAConfig,
    outcome_version: str = "v1",
) -> EDAResponse:
    """Toy stub: returns an empty EDAResponse. Real projects implement findings here."""
    t0 = time.monotonic()
    return EDAResponse(
        findings=[],
        scores=EDAScores(),
        performance=EDAPerformance(
            total_seconds=time.monotonic() - t0,
            stage_timings=[HarnessStageTiming(stage="stub", seconds=0.0)],
        ),
        outcome_version=outcome_version,
        fingerprint="",
    )
