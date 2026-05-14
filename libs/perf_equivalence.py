"""Prediction and harness-response equivalence checks for the perf-bounty autoloop."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from libs.responses import HarnessResponse

__all__ = [
    "EquivResult",
    "predictions_equivalent",
    "harness_response_equivalent",
    "compute_predictions_hash",
]

_SORT_KEYS = ["user_id", "email_id", "date_sent", "fold"]


@dataclass
class EquivResult:
    passed: bool
    score_corr: float | None
    residual_corr: float | None
    hash_match: bool
    drift_pct: float


def compute_predictions_hash(parquet_path: Path) -> str:
    df = pl.read_parquet(parquet_path)
    row_hash_sum = df.hash_rows().sum()
    raw = struct.pack(">Q", int(row_hash_sum) & 0xFFFFFFFFFFFFFFFF)
    return hashlib.sha256(raw).hexdigest()


def predictions_equivalent(
    baseline_parquet: Path,
    candidate_parquet: Path,
    *,
    corr_min: float = 0.99999,
) -> EquivResult:
    baseline_hash = compute_predictions_hash(baseline_parquet)
    candidate_hash = compute_predictions_hash(candidate_parquet)
    hash_match = baseline_hash == candidate_hash

    b = pl.read_parquet(baseline_parquet).sort(_SORT_KEYS)
    c = pl.read_parquet(candidate_parquet).sort(_SORT_KEYS)

    row_count_match = len(b) == len(c)

    if not row_count_match:
        return EquivResult(
            passed=False,
            score_corr=None,
            residual_corr=None,
            hash_match=hash_match,
            drift_pct=float("inf"),
        )

    b_score = b["score"].to_numpy()
    c_score = c["score"].to_numpy()
    b_res = b["score_residualized"].to_numpy()
    c_res = c["score_residualized"].to_numpy()

    score_corr = float(np.corrcoef(b_score, c_score)[0, 1])
    residual_corr = float(np.corrcoef(b_res, c_res)[0, 1])

    # Avoid division by zero; rows where baseline score is zero use absolute drift
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_drift = np.where(
            b_score != 0,
            np.abs((c_score - b_score) / b_score),
            np.abs(c_score - b_score),
        )
    drift_pct = float(rel_drift.max() * 100)

    passed = score_corr >= corr_min and residual_corr >= corr_min and row_count_match and hash_match

    return EquivResult(
        passed=passed,
        score_corr=score_corr,
        residual_corr=residual_corr,
        hash_match=hash_match,
        drift_pct=drift_pct,
    )


def harness_response_equivalent(
    baseline: HarnessResponse,
    candidate: HarnessResponse,
    *,
    metric_tol: float = 1e-6,
) -> EquivResult:
    def _float_close(a: float | None, b: float | None) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(a - b) <= metric_tol

    summary_ok = (
        baseline.summary.primary_metric_name == candidate.summary.primary_metric_name
        and _float_close(
            baseline.summary.primary_metric_value, candidate.summary.primary_metric_value
        )
        and _float_close(
            baseline.summary.baseline_metric_value, candidate.summary.baseline_metric_value
        )
        and _float_close(baseline.summary.lift_vs_baseline, candidate.summary.lift_vs_baseline)
        and baseline.summary.direction == candidate.summary.direction
        and set(baseline.summary.secondary_metrics) == set(candidate.summary.secondary_metrics)
        and all(
            _float_close(
                baseline.summary.secondary_metrics[k], candidate.summary.secondary_metrics[k]
            )
            for k in baseline.summary.secondary_metrics
        )
    )

    data_profile_ok = baseline.data_profile == candidate.data_profile

    b_fp = baseline.fingerprint or ""
    c_fp = candidate.fingerprint or ""
    fingerprint_ok = (b_fp == "" and c_fp == "") or (b_fp == c_fp)

    passed = summary_ok and data_profile_ok and fingerprint_ok

    return EquivResult(
        passed=passed,
        score_corr=None,
        residual_corr=None,
        hash_match=fingerprint_ok,
        drift_pct=0.0 if summary_ok else float("nan"),
    )
