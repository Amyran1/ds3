"""One-shot backfill: emit timing rows for all historical runs.

Reads harness_response.json + results.jsonl for each existing run directory,
reconstructs timing rows, and appends to timing_performance.jsonl in
chronological order (sorted by recorded_at_utc).

Run with:
  python -m projects.civic_shout_action_rate_increase.runs._backfill_timing
"""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent
_RUNS_DIR = _PROJECT_DIR / "runs"
_LEDGER = _PROJECT_DIR / "timing_performance.jsonl"
_RESULTS_JSONL = _RUNS_DIR / "results.jsonl"


def _build_results_index() -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    if not _RESULTS_JSONL.exists():
        return index
    for line in _RESULTS_JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = row.get("run_id", "")
        index.setdefault(run_id, []).append(row)
    return index


def _backfill() -> None:
    results_index = _build_results_index()

    candidates: list[tuple[str, dict, dict]] = []

    for run_dir in sorted(_RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        harness_path = run_dir / "harness_response.json"
        if not harness_path.exists():
            continue

        try:
            harness_data = json.loads(harness_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        perf = harness_data.get("performance", {})
        if not perf:
            continue

        run_id = run_dir.name
        ledger_rows = results_index.get(run_id, [])
        if not ledger_rows:
            continue

        meta = ledger_rows[-1]

        recorded_at = meta.get("recorded_at_utc", "")
        candidates.append((recorded_at, meta, harness_data))

    candidates.sort(key=lambda t: t[0])

    emitted = 0
    for recorded_at, meta, harness_data in candidates:
        perf = harness_data.get("performance", {})
        data_profile = harness_data.get("data", {})

        stage_seconds: dict[str, float] = {
            st["stage"]: st["seconds"]
            for st in perf.get("stage_timings", [])
            if "stage" in st and "seconds" in st
        }

        parallelism = perf.get("parallelism") or {}
        cpu_utilization_pct = parallelism.get("cpu_utilization_pct")

        row = {
            "run_id": meta.get("run_id"),
            "project": meta.get("project"),
            "scope": meta.get("scope"),
            "comparison_group": meta.get("comparison_group"),
            "sample_frac": meta.get("sample_frac"),
            "git_sha": meta.get("git_sha"),
            "recorded_at_utc": recorded_at,
            "total_seconds": perf.get("total_seconds"),
            "rows_per_second": perf.get("rows_per_second"),
            "bottleneck_stage": perf.get("bottleneck_stage"),
            "cpu_utilization_pct": cpu_utilization_pct,
            "peak_memory_mb": perf.get("peak_memory_mb"),
            "n_test_rows": data_profile.get("n_rows"),
            "n_train_rows": data_profile.get("train_rows"),
            "n_features": data_profile.get("n_features"),
            "walk_forward_n_folds": meta.get("walk_forward_n_folds"),
            "stage_seconds": stage_seconds,
        }

        with _LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")

        total = perf.get("total_seconds", "?")
        scope = meta.get("scope", "?")
        print(f"  ({meta.get('run_id')!r}, {total}, {scope!r})")
        emitted += 1

    print(f"\nBackfilled {emitted} row(s) → {_LEDGER}")


if __name__ == "__main__":
    _backfill()
