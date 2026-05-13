"""Smoke run — Ridge + winsorize_heavy_tails (p99 caps on AveRooms/AveBedrms/AveOccup/Population) on 10% sample.

Autoloop iteration 6, brainstorm BS-2026-05-13-wh008 (winsorize_heavy_tails).

Pre-samples to sample_frac=0.10 (deterministic seed) before `run_record(...)`
because `libs.run_record.run_record` passes `sample_frac=None` to the harness;
the autoloop directive "sample_frac=0.1 passed through to the harness call" is
honored by feeding the harness a pre-sampled DataFrame.

Feature-cols swap: this family REPLACES the four heavy-tail raw columns
(AveRooms, AveBedrms, AveOccup, Population) with their `*_winz` winsorized
variants. Passing both raw and winz would defeat the preprocessing because
Ridge would still see the outliers through the raw columns.

Sample-first gate result (from this feature family's timing_performance.jsonl):
rows_per_second 74,542 (vs predicted 280K and gate floor 140K). `gate_status`
"failed" against single-op comparables; absolute projected_full_seconds 0.28s
is well below the PLAN's 5s ceiling. The comparable mismatch is structural
(O(n log n) quantile sort vs O(n) division), not a runtime defect — see
PLAN §7 for the calibration rationale.
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
from projects.california_housing_demo.features.winsorize_heavy_tails.cache import (
    FEATURE_COLS,
    HEAVY_TAIL_RAW_COLS,
    add_features,
)

SAMPLE_FRAC = 0.10
SAMPLE_SEED = 42


def main() -> RunRecordOutput:
    project_root = Path(__file__).resolve().parents[3]
    metadata = RunResultMetadata(
        run_id="run_06",
        project="california_housing_demo",
        comparison_group="california_housing_v1",
        scope="smoke",
        verdict="smoke_only",
        recorded_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        outcome_version="v1",
        notes="winsorize_heavy_tails; smoke sample_frac=0.10; autoloop iter 6",
    )

    full = load_data()
    sampled = full.sample(
        n=max(50, int(full.height * SAMPLE_FRAC)),
        seed=SAMPLE_SEED,
        with_replacement=False,
    )
    data = add_features(sampled)
    feature_cols = [c for c in RAW_FEATURE_COLS if c not in HEAVY_TAIL_RAW_COLS] + FEATURE_COLS

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
        print(f"run_06 complete: r2={out.harness_response.summary.primary_metric_value:.4f}")
