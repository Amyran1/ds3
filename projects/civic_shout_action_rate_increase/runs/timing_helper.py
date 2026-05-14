"""Harness timing ledger emitter for civic_shout_action_rate_increase.

Appends one JSONL row to projects/{project}/timing_performance.jsonl per harness
run. Schema mirrors HarnessPerformance stage_timings with project/run metadata
flattened for easy leaderboard joins.

Row schema:
  run_id, project, scope, comparison_group, sample_frac, git_sha,
  recorded_at_utc, total_seconds, rows_per_second, bottleneck_stage,
  cpu_utilization_pct, peak_memory_mb, n_test_rows, n_train_rows,
  n_features, walk_forward_n_folds, stage_seconds
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def emit_timing_row(
    metadata: Any,
    harness_response: Any,
    project_root: Path,
) -> Path:
    perf = harness_response.performance

    stage_seconds: dict[str, float] = {st.stage: st.seconds for st in perf.stage_timings}

    cpu_utilization_pct: float | None = None
    if perf.parallelism is not None:
        cpu_utilization_pct = perf.parallelism.cpu_utilization_pct

    recorded_at = metadata.recorded_at_utc
    if isinstance(recorded_at, datetime):
        recorded_at = recorded_at.isoformat()

    result_row = harness_response.to_result_row(metadata)

    n_test_rows: int | None = getattr(result_row, "n_rows", None)
    n_train_rows: int | None = None
    data = getattr(harness_response, "data", None)
    if data is not None:
        n_train_rows = getattr(data, "train_rows", None)

    n_features: int = getattr(result_row, "n_feature_cols", None) or getattr(
        harness_response.data, "n_features", 0
    )

    walk_forward_n_folds: int | None = getattr(result_row, "walk_forward_n_folds", None)

    row: dict[str, Any] = {
        "run_id": metadata.run_id,
        "project": metadata.project,
        "scope": metadata.scope,
        "comparison_group": metadata.comparison_group,
        "sample_frac": getattr(metadata, "sample_frac", None)
        or getattr(harness_response.data, "sample_frac", None),
        "git_sha": metadata.git_sha,
        "recorded_at_utc": recorded_at,
        "total_seconds": perf.total_seconds,
        "rows_per_second": getattr(perf, "rows_per_second", None),
        "bottleneck_stage": getattr(perf, "bottleneck_stage", None),
        "cpu_utilization_pct": cpu_utilization_pct,
        "peak_memory_mb": getattr(perf, "peak_memory_mb", None),
        "n_test_rows": n_test_rows,
        "n_train_rows": n_train_rows,
        "n_features": n_features,
        "walk_forward_n_folds": walk_forward_n_folds,
        "stage_seconds": stage_seconds,
    }

    ledger_path = project_root / "timing_performance.jsonl"
    with ledger_path.open("a") as f:
        f.write(json.dumps(row) + "\n")

    return ledger_path.resolve()
