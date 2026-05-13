"""Baseline run — Ridge regression on raw features."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from libs.responses import DiscoveryConfig, EDAConfig, RunResultMetadata
from libs.run_record import RunRecordOutput, run_record
from projects.california_housing_demo.data import OUTCOME_VARIABLE, RAW_FEATURE_COLS, load_data


def main() -> RunRecordOutput:
    project_root = Path(__file__).resolve().parents[3]
    metadata = RunResultMetadata(
        run_id="run_01",
        project="california_housing_demo",
        comparison_group="california_housing_v1",
        scope="comparison",
        verdict="promote",
        recorded_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        outcome_version="v1",
        notes="baseline_ridge_alpha_1",
    )
    data = load_data()
    return run_record(
        metadata=metadata,
        data=data,
        feature_cols=RAW_FEATURE_COLS,
        outcome_variable=OUTCOME_VARIABLE,
        model_cfg={"alpha": 1.0, "seed": 42},
        eda_config=EDAConfig(),
        discovery_config=DiscoveryConfig(),
        project_root=project_root,
    )


if __name__ == "__main__":
    out = main()
    if out.harness_response is not None:
        print(f"run_01 complete: r2={out.harness_response.summary.primary_metric_value:.4f}")
