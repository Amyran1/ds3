"""Smoke run — Ridge + population_per_household (Population / AveOccup) on 10% sample.

Autoloop iteration 4, brainstorm BS-2026-05-13-pph004 (population_per_household).
Pre-samples to sample_frac=0.10 (deterministic seed) before `run_record(...)`
because `libs.run_record.run_record` passes `sample_frac=None` to the harness;
the autoloop directive "sample_frac=0.1 passed through to the harness call" is
honored by feeding the harness a pre-sampled DataFrame.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from libs.responses import EDAConfig, RunResultMetadata
from libs.run_record import RunRecordOutput, run_record
from projects.california_housing_demo.data import (
    OUTCOME_VARIABLE,
    RAW_FEATURE_COLS,
    load_data,
)
from projects.california_housing_demo.features.population_per_household.cache import (
    FEATURE_COLS,
    add_features,
)

SAMPLE_FRAC = 0.10
SAMPLE_SEED = 42


def main() -> RunRecordOutput:
    project_root = Path(__file__).resolve().parents[3]
    metadata = RunResultMetadata(
        run_id="run_05",
        project="california_housing_demo",
        comparison_group="california_housing_v1",
        scope="smoke",
        verdict="smoke_only",
        recorded_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        outcome_version="v1",
        notes="population_per_household; smoke sample_frac=0.10; autoloop iter 4",
    )

    full = load_data()
    sampled = full.sample(
        n=max(50, int(full.height * SAMPLE_FRAC)),
        seed=SAMPLE_SEED,
        with_replacement=False,
    )
    data = add_features(sampled)
    feature_cols = RAW_FEATURE_COLS + FEATURE_COLS

    return run_record(
        metadata=metadata,
        data=data,
        feature_cols=feature_cols,
        outcome_variable=OUTCOME_VARIABLE,
        model_cfg={"alpha": 1.0, "seed": SAMPLE_SEED},
        eda_config=EDAConfig(),
        discovery_config=None,
        project_root=project_root,
    )


if __name__ == "__main__":
    out = main()
    if out.harness_response is not None:
        print(f"run_05 complete: r2={out.harness_response.summary.primary_metric_value:.4f}")
