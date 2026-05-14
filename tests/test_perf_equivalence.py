from __future__ import annotations

from pathlib import Path

import polars as pl
from libs.perf_equivalence import (
    compute_predictions_hash,
    harness_response_equivalent,
    predictions_equivalent,
)

from libs.responses import (
    HarnessDataProfile,
    HarnessParallelism,
    HarnessPerformance,
    HarnessResponse,
    HarnessStageTiming,
    HarnessSummary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PARQUET_COLS = [
    "user_id",
    "email_id",
    "date_sent",
    "score",
    "score_residualized",
    "label",
    "row_loss",
    "fold",
]


def _make_predictions_df(n_rows: int = 20, score_offset: float = 0.0) -> pl.DataFrame:
    """Build a minimal predictions parquet matching harness.py:1843 schema."""
    return pl.DataFrame(
        {
            "user_id": pl.Series(list(range(n_rows)), dtype=pl.Int64),
            "email_id": pl.Series([i % 5 for i in range(n_rows)], dtype=pl.Int64),
            "date_sent": pl.Series(
                [f"2024-0{(i % 9) + 1}-01" for i in range(n_rows)],
                dtype=pl.Utf8,
            ).str.to_datetime(),
            "score": pl.Series(
                [0.1 + (i * 0.01) + score_offset for i in range(n_rows)],
                dtype=pl.Float64,
            ),
            "score_residualized": pl.Series(
                [0.05 + (i * 0.005) for i in range(n_rows)],
                dtype=pl.Float64,
            ),
            "label": pl.Series([i % 2 for i in range(n_rows)], dtype=pl.Int32),
            "row_loss": pl.Series([0.3 + (i * 0.01) for i in range(n_rows)], dtype=pl.Float64),
            "fold": pl.Series([i % 4 for i in range(n_rows)], dtype=pl.Int32),
        }
    )


def _make_harness_response(
    primary_value: float = 0.75,
    lift: float = 0.08,
    total_seconds: float = 100.0,
    fingerprint: str = "abc123",
) -> HarnessResponse:
    return HarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc",
            primary_metric_value=primary_value,
            baseline_metric_value=primary_value - lift,
            lift_vs_baseline=lift,
            secondary_metrics={"pr_auc": 0.40, "brier": 0.18},
            direction="higher_is_better",
        ),
        performance=HarnessPerformance(
            total_seconds=total_seconds,
            parallelism=HarnessParallelism(
                outer_n_jobs=4,
                inner_threads=2,
                cpu_utilization_pct=350.0,
            ),
            stage_timings=[
                HarnessStageTiming(stage="aggregate", seconds=total_seconds * 0.51),
                HarnessStageTiming(stage="score", seconds=total_seconds * 0.44),
            ],
        ),
        data_profile=HarnessDataProfile(
            n_rows=10000,
            n_rows_source=200000,
            sample_frac=0.05,
            is_full_run=False,
            n_features=8,
        ),
        artifacts=[],
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Test 1: predictions_equivalent — identical parquets → passed=True
# ---------------------------------------------------------------------------


def test_predictions_equivalent_identical_parquets(tmp_path: Path) -> None:
    """predictions_equivalent returns passed=True when two parquets are identical."""
    df = _make_predictions_df()
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    df.write_parquet(baseline)
    df.write_parquet(candidate)

    result = predictions_equivalent(baseline, candidate)

    assert result.passed is True, f"Expected passed=True for identical parquets, got {result}"


def test_predictions_equivalent_perturbed_score_fails(tmp_path: Path) -> None:
    """predictions_equivalent returns passed=False when one row's score is perturbed by 1e-3."""
    baseline_df = _make_predictions_df()
    candidate_df = _make_predictions_df(score_offset=1e-3)
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    baseline_df.write_parquet(baseline)
    candidate_df.write_parquet(candidate)

    result = predictions_equivalent(baseline, candidate)

    assert result.passed is False, (
        f"Expected passed=False when score perturbed by 1e-3, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 2: compute_predictions_hash — deterministic across two reads
# ---------------------------------------------------------------------------


def test_compute_predictions_hash_deterministic(tmp_path: Path) -> None:
    """compute_predictions_hash returns the same sha256 on two independent reads of the same parquet."""
    df = _make_predictions_df()
    parquet_path = tmp_path / "preds.parquet"
    df.write_parquet(parquet_path)

    hash_first = compute_predictions_hash(parquet_path)
    hash_second = compute_predictions_hash(parquet_path)

    assert hash_first == hash_second, (
        f"Hash not deterministic: first={hash_first!r}, second={hash_second!r}"
    )
    assert len(hash_first) == 64, f"Expected sha256 hex string (64 chars), got {len(hash_first)}"


# ---------------------------------------------------------------------------
# Test 3: harness_response_equivalent — ignores performance.* and timestamps
# ---------------------------------------------------------------------------


def test_harness_response_equivalent_ignores_performance_and_timestamps(tmp_path: Path) -> None:
    """harness_response_equivalent treats two responses equal when they differ only in performance.* fields."""
    baseline = _make_harness_response(total_seconds=100.0, fingerprint="abc123")
    candidate = _make_harness_response(total_seconds=55.0, fingerprint="abc123")

    result = harness_response_equivalent(baseline, candidate, metric_tol=1e-6)

    assert result.passed is True, (
        "Expected passed=True when responses differ only in performance.total_seconds, "
        f"got {result}"
    )


def test_harness_response_equivalent_detects_metric_drift(tmp_path: Path) -> None:
    """harness_response_equivalent returns passed=False when summary.value differs beyond metric_tol."""
    baseline = _make_harness_response(primary_value=0.75, fingerprint="abc123")
    candidate = _make_harness_response(primary_value=0.76, fingerprint="abc123")

    result = harness_response_equivalent(baseline, candidate, metric_tol=1e-6)

    assert result.passed is False, (
        f"Expected passed=False when primary_metric_value differs by 0.01, got {result}"
    )
