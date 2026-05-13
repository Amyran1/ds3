from __future__ import annotations

import time

import polars as pl

from libs.responses import (
    DiscoveryConfig,
    DiscoveryPerformance,
    DiscoveryReproducibility,
    DiscoveryResponse,
    DiscoveryScores,
    HarnessResponse,
    HarnessStageTiming,
)


def discover_gaps(
    *,
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    prediction_cols: dict[str, str],
    harness_response: HarnessResponse,
    discovery_config: DiscoveryConfig,
    outcome_version: str = "v1",
) -> DiscoveryResponse:
    """Toy stub: returns an empty DiscoveryResponse. Real projects implement findings here."""
    t0 = time.monotonic()
    return DiscoveryResponse(
        findings=[],
        scores=DiscoveryScores(),
        reproducibility=DiscoveryReproducibility(),
        performance=DiscoveryPerformance(
            total_seconds=time.monotonic() - t0,
            stage_timings=[HarnessStageTiming(stage="stub", seconds=0.0)],
        ),
        outcome_version=outcome_version,
    )
