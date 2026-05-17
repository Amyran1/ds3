from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numba
import numpy as np
import polars as pl
import psutil
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict, Field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from libs.perf_equivalence import compute_predictions_hash
from projects.civic_shout_action_rate_increase.harness_cache import (
    HarnessCache,
    data_fingerprint,
    fold_train_fingerprint,
    full_work_df_fingerprint,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LeakageError(Exception):
    pass


class ConfoundJoinError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Stratification constants
# ---------------------------------------------------------------------------

_TENURE_BREAKS = [7, 30, 90, 180, 365, 730]
_TENURE_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr", "2yr+"]
_PRIOR_ACTION_BINS = [0, 1, 2, 5, 20, 1_000_000]
_PRIOR_ACTION_LABELS = ["0", "1", "2-4", "5-19", "20+"]

# Denylist of outcome and outcome-derived columns. Allowlist-by-feature-dictionary
# is a follow-up when > 2 feature families exist.
_FORBIDDEN_COLS = {
    "opened",
    "clicked",
    "verified_opened",
    "actioned",
    "actioned_24h",
    "n_actions_24h",
    "unsubscribed",
}


# ---------------------------------------------------------------------------
# Pydantic shapes — canonical contract
# ---------------------------------------------------------------------------


class HarnessSummary(BaseModel):
    primary_metric_name: str
    primary_metric_value: float
    baseline_metric_value: float | None = None
    lift_vs_baseline: float | None = None
    result_notes: list[str] = Field(default_factory=list)


class HarnessMethod(BaseModel):
    name: str
    metric_direction: Literal["higher_is_better", "lower_is_better"]
    n_splits: int | None = None
    split_strategy: str
    outcome_transform: str | None = None
    outcome_inverse_transform: str | None = None


class HarnessDataProfile(BaseModel):
    n_rows: int
    n_features: int
    feature_cols: list[str]
    outcome_variable: str
    train_rows: int | None = None
    validation_rows: int | None = None
    excluded_rows: int = 0
    sample_frac: float | None = None
    sample_seed: int | None = None
    sample_strategy: str | None = None
    is_full_run: bool
    population_n_rows: int | None = None
    column_mapping: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class HarnessMetric(BaseModel):
    name: str
    value: float
    split: str | None = None
    fold: int | None = None
    direction: Literal["higher_is_better", "lower_is_better"]
    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class HarnessMetrics(BaseModel):
    primary: HarnessMetric
    by_fold: list[HarnessMetric] = Field(default_factory=list)
    secondary: list[HarnessMetric] = Field(default_factory=list)
    baseline: list[HarnessMetric] = Field(default_factory=list)


class HarnessStageTiming(BaseModel):
    stage: Literal[
        "outcome_transform",
        "outcome_inverse_transform",
        "column_selection",
        "data_conversion",
        "split_generation",
        "fit",
        "predict",
        "score",
        "aggregate",
        "residualize_scores",
        "cross_fit_isotonic",
        "bootstrap_ci",
        "persist_predictions",
    ]
    seconds: float
    owner: Literal["harness", "model"]
    calls: int = 1
    notes: list[str] = Field(default_factory=list)


class HarnessParallelism(BaseModel):
    outer_n_jobs: int
    inner_threads: int
    backend: Literal["loky", "threading", "multiprocessing", "dask", "sequential"]
    cpu_count_observed: int
    cpu_utilization_pct: float | None = None
    notes: list[str] = Field(default_factory=list)


class HarnessPerformance(BaseModel):
    total_seconds: float
    stage_timings: list[HarnessStageTiming]
    bottleneck_stage: str | None = None
    rows_per_second: float | None = None
    folds_per_second: float | None = None
    harness_owned_seconds: float | None = None
    model_owned_seconds: float | None = None
    peak_memory_mb: float | None = None
    parallelism: HarnessParallelism | None = None
    hardware: str | None = None
    sample_size: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class HarnessDiagnostic(BaseModel):
    name: str
    severity: Literal["info", "warning", "error"]
    message: str
    values: dict[str, Any] = Field(default_factory=dict)


class HarnessArtifact(BaseModel):
    name: str = "predictions"
    kind: Literal["table", "plot", "model", "predictions", "report", "other"]
    uri: str
    description: str | None = None


class HarnessReproducibility(BaseModel):
    seed: int | None = None
    ml_model_type: str
    ml_model_args: dict[str, Any]
    data_fingerprint: str | None = None
    code_version: str | None = None


class ML_Model_Type(Enum):
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
    LOGISTIC_REGRESSION = "logistic_regression"


class ML_Model_Config(BaseModel):
    ml_model_type: ML_Model_Type
    args: dict[str, Any]
    column_mapping: dict[str, str] = Field(default_factory=dict)


class RunResultMetadata(BaseModel):
    run_id: str
    project: str
    comparison_group: str
    scope: Literal[
        "smoke", "reproduction", "comparison", "champion_candidate", "fast_iter", "comparison_fast"
    ]
    status: str = "completed"
    verdict: str | None = None
    recorded_at_utc: Any = None
    git_sha: str | None = None
    data_fingerprint: str | None = None


class RunResultRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    project: str
    comparison_group: str
    scope: str
    status: str
    verdict: str | None = None
    recorded_at_utc: Any = None
    git_sha: str | None = None
    data_fingerprint: str | None = None
    outcome_variable: str
    primary_metric_name: str
    primary_metric_value: float
    baseline_metric_value: float | None = None
    lift_vs_baseline: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_rows: int
    sample_frac: float | None = None
    is_full_run: bool
    elapsed_seconds: float
    model_type: str
    n_feature_cols: int
    feature_cols: list[str]
    cpu_utilization_pct: float | None = None


class HarnessResponse(BaseModel):
    summary: HarnessSummary
    method: HarnessMethod
    data: HarnessDataProfile
    metrics: HarnessMetrics
    performance: HarnessPerformance
    diagnostics: list[HarnessDiagnostic] = Field(default_factory=list)
    artifacts: list[HarnessArtifact] = Field(default_factory=list)
    reproducibility: HarnessReproducibility
    model_interpretability: dict[str, Any] | None = None
    fingerprint: str = ""

    def to_result_row(self, run_metadata: RunResultMetadata) -> RunResultRow:
        return RunResultRow(
            run_id=run_metadata.run_id,
            project=run_metadata.project,
            comparison_group=run_metadata.comparison_group,
            scope=run_metadata.scope,
            status=run_metadata.status,
            verdict=run_metadata.verdict,
            recorded_at_utc=run_metadata.recorded_at_utc,
            git_sha=run_metadata.git_sha,
            data_fingerprint=run_metadata.data_fingerprint or self.reproducibility.data_fingerprint,
            outcome_variable=self.data.outcome_variable,
            primary_metric_name=self.summary.primary_metric_name,
            primary_metric_value=self.summary.primary_metric_value,
            baseline_metric_value=self.summary.baseline_metric_value,
            lift_vs_baseline=self.summary.lift_vs_baseline,
            ci_low=self.metrics.primary.ci_low,
            ci_high=self.metrics.primary.ci_high,
            n_rows=self.data.n_rows,
            sample_frac=self.data.sample_frac,
            is_full_run=self.data.is_full_run,
            elapsed_seconds=self.performance.total_seconds,
            model_type=self.reproducibility.ml_model_type,
            n_feature_cols=self.data.n_features,
            feature_cols=self.data.feature_cols,
            cpu_utilization_pct=(
                self.performance.parallelism.cpu_utilization_pct
                if self.performance.parallelism
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Project subclasses
# ---------------------------------------------------------------------------


class CivicShoutResultRow(RunResultRow):
    roc_auc_raw_pair: float | None = None
    roc_auc_confound_only_pair: float | None = None
    roc_auc_residualized_user_prior_pair: float | None = None
    roc_auc_residualized_user_prior_x_email_popularity_pair: float | None = None
    roc_auc_residualized_user_main_effect_pair: float | None = None
    roc_auc_residualized_user_main_effect_x_email_popularity_pair: float | None = None
    roc_auc_residualized_email_popularity_pair: float | None = None
    pr_auc_residualized_user_prior_x_email_popularity_pair: float | None = None
    roc_auc_residualized_user_weighted_pair: float | None = None
    roc_auc_residualized_pair_tenure_0_7d: float | None = None
    roc_auc_residualized_pair_tenure_8_30d: float | None = None
    roc_auc_residualized_pair_tenure_31_90d: float | None = None
    roc_auc_residualized_pair_tenure_91_180d: float | None = None
    roc_auc_residualized_pair_tenure_181_365d: float | None = None
    roc_auc_residualized_pair_tenure_1_2yr: float | None = None
    roc_auc_residualized_pair_tenure_2yr_plus: float | None = None
    roc_auc_residualized_non_streak_users_pair: float | None = None
    roc_auc_residualized_cold_start_pair: float | None = None
    primary_metric_temporal_cv: float | None = None
    auc_reverse_time: float | None = None
    walk_forward_strategy: str | None = None
    walk_forward_n_folds: int | None = None
    walk_forward_fold_aucs: list[float] | None = None
    walk_forward_fold_labels: list[str] | None = None
    walk_forward_fold_std: float | None = None
    walk_forward_fold_min: float | None = None
    walk_forward_fold_max: float | None = None
    bootstrap_method: str | None = None
    isotonic_method: str | None = None
    n_test_users: int | None = None
    n_test_emails: int | None = None
    n_test_rows: int | None = None
    score_psi_train_test: float | None = None
    score_psi_max_by_tenure: float | None = None
    per_stratum_auc_cv: float | None = None
    confound_correlation_pearson: float | None = None
    residual_collapse_flag: bool | None = None
    pooled_positive_rate_train: float | None = None
    pooled_positive_rate_test: float | None = None
    confound_overlap_min_positives_per_cell: int | None = None
    confound_overlap_n_cells_below_threshold: int | None = None
    class_balance_min_positives_per_cell: int | None = None
    class_balance_n_cells_excluded: int | None = None


class CivicShoutHarnessResponse(HarnessResponse):
    def to_result_row(self, run_metadata: RunResultMetadata) -> CivicShoutResultRow:
        base = super().to_result_row(run_metadata).model_dump()
        sec = {m.name: m.value for m in self.metrics.secondary}
        diag = {d.name: d for d in self.diagnostics}

        tenure_map = {
            "0-7d": "roc_auc_residualized_pair_tenure_0_7d",
            "8-30d": "roc_auc_residualized_pair_tenure_8_30d",
            "31-90d": "roc_auc_residualized_pair_tenure_31_90d",
            "91-180d": "roc_auc_residualized_pair_tenure_91_180d",
            "181-365d": "roc_auc_residualized_pair_tenure_181_365d",
            "1-2yr": "roc_auc_residualized_pair_tenure_1_2yr",
            "2yr+": "roc_auc_residualized_pair_tenure_2yr_plus",
        }

        def _diag_val(name: str, key: str) -> Any:
            d = diag.get(name)
            return d.values.get(key) if d else None

        fold_aucs: list[float] | None = None
        fold_labels: list[str] | None = None
        by_fold_aucs = [m.value for m in self.metrics.by_fold]
        by_fold_splits = [m.split for m in self.metrics.by_fold]
        if by_fold_aucs:
            fold_aucs = by_fold_aucs
            fold_labels = [s for s in by_fold_splits if s is not None]

        extras: dict[str, Any] = {
            "roc_auc_raw_pair": sec.get("roc_auc_raw_pair"),
            "roc_auc_confound_only_pair": sec.get("roc_auc_confound_only_pair"),
            "roc_auc_residualized_user_prior_pair": sec.get("roc_auc_residualized_user_prior_pair"),
            "roc_auc_residualized_user_prior_x_email_popularity_pair": sec.get(
                "roc_auc_residualized_user_prior_x_email_popularity_pair"
            ),
            "roc_auc_residualized_user_main_effect_pair": sec.get(
                "roc_auc_residualized_user_main_effect_pair"
            ),
            "roc_auc_residualized_user_main_effect_x_email_popularity_pair": sec.get(
                "roc_auc_residualized_user_main_effect_x_email_popularity_pair"
            ),
            "roc_auc_residualized_email_popularity_pair": sec.get(
                "roc_auc_residualized_email_popularity_pair"
            ),
            "pr_auc_residualized_user_prior_x_email_popularity_pair": sec.get(
                "pr_auc_residualized_user_prior_x_email_popularity_pair"
            ),
            "roc_auc_residualized_user_weighted_pair": sec.get(
                "roc_auc_residualized_user_weighted_pair"
            ),
            "roc_auc_residualized_non_streak_users_pair": sec.get(
                "roc_auc_residualized_non_streak_users_pair"
            ),
            "roc_auc_residualized_cold_start_pair": sec.get("roc_auc_residualized_cold_start_pair"),
            "primary_metric_temporal_cv": sec.get("primary_metric_temporal_cv"),
            "auc_reverse_time": sec.get("auc_reverse_time"),
            "n_test_users": int(sec["n_test_users"])
            if sec.get("n_test_users") is not None
            else None,
            "n_test_emails": int(sec["n_test_emails"])
            if sec.get("n_test_emails") is not None
            else None,
            "n_test_rows": int(sec["n_test_rows"]) if sec.get("n_test_rows") is not None else None,
            "pooled_positive_rate_train": sec.get("pooled_positive_rate_train"),
            "pooled_positive_rate_test": sec.get("pooled_positive_rate_test"),
            "walk_forward_strategy": "quarterly_cumulative_train",
            "walk_forward_n_folds": len(fold_aucs) if fold_aucs else None,
            "walk_forward_fold_aucs": fold_aucs,
            "walk_forward_fold_labels": fold_labels or None,
            "walk_forward_fold_std": float(np.std(fold_aucs, ddof=1)) if fold_aucs else None,
            "walk_forward_fold_min": float(min(fold_aucs)) if fold_aucs else None,
            "walk_forward_fold_max": float(max(fold_aucs)) if fold_aucs else None,
            "bootstrap_method": "user_block_500",
            "isotonic_method": "cross_fit_2way",
            "score_psi_train_test": _diag_val("score_psi_train_test", "psi"),
            "score_psi_max_by_tenure": _diag_val("score_psi_by_tenure_bucket", "max_psi"),
            "per_stratum_auc_cv": _diag_val("per_stratum_auc_dispersion", "cv"),
            "confound_correlation_pearson": _diag_val("confound_correlation_pearson", "pearson_r"),
            "residual_collapse_flag": bool(
                _diag_val("residual_collapse_check", "collapsed") or False
            ),
            "confound_overlap_min_positives_per_cell": _diag_val(
                "confound_overlap_min_across_folds", "min_positives_per_cell"
            ),
            "confound_overlap_n_cells_below_threshold": _diag_val(
                "confound_overlap_min_across_folds", "n_cells_below_threshold"
            ),
            "class_balance_min_positives_per_cell": _diag_val(
                "class_balance", "min_positives_per_cell"
            ),
            "class_balance_n_cells_excluded": _diag_val("class_balance", "n_cells_excluded"),
        }

        for label, col in tenure_map.items():
            extras[col] = sec.get(f"roc_auc_residualized_{label}_pair")

        return CivicShoutResultRow(**base, **extras)


# ---------------------------------------------------------------------------
# Fast-iter mode gate
# ---------------------------------------------------------------------------


def _is_fast_iter(scope: str) -> bool:
    return scope in ("fast_iter", "comparison_fast")


# ---------------------------------------------------------------------------
# Confound score helpers  (computed on FULL data, not the sample)
# ---------------------------------------------------------------------------


def _compute_confound_scores_full_data(full_df: pl.DataFrame) -> pl.DataFrame:
    """Compute confound scores on the FULL frame; caller joins to sample.

    Returns df with columns (user_id, email_id, date_sent,
    user_prior_engagement_score, email_popularity_score).

    Strict-prior semantics: group_by to (user/email, date_sent) collapses
    same-day events to one row BEFORE cum_sum + shift(1), so the prior at
    date t excludes all events on date t (not just row-order prior).
    Two sub-plans collected in parallel via pl.collect_all (v3.6a).
    """
    user_date_lazy = (
        full_df.lazy()
        .group_by(["user_id", "date_sent"])
        .agg(pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"))
        .with_columns(
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("user_id", order_by="date_sent")
            .fill_null(0)
            .alias("_prior_action_count")
        )
        .with_columns(
            (pl.col("_prior_action_count") + 1)
            .log(base=2.718281828459045)
            .alias("user_prior_engagement_score")
        )
        .select(["user_id", "date_sent", "user_prior_engagement_score"])
    )

    email_date_lazy = (
        full_df.lazy()
        .group_by(["email_id", "date_sent"])
        .agg(
            pl.len().cast(pl.Int64).alias("_sends_on_date"),
            pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"),
        )
        .with_columns(
            pl.col("_sends_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id", order_by="date_sent")
            .fill_null(0)
            .alias("_prior_sends"),
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id", order_by="date_sent")
            .fill_null(0)
            .alias("_prior_acts"),
        )
        .with_columns(
            ((pl.col("_prior_acts") + 1) / (pl.col("_prior_sends") + 2)).alias(
                "email_popularity_score"
            )
        )
        .select(["email_id", "date_sent", "email_popularity_score"])
    )

    n_unique_keys = full_df.select(
        pl.struct(["user_id", "email_id", "date_sent"]).n_unique()
    ).item()
    if n_unique_keys != len(full_df):
        raise ConfoundJoinError(
            f"Confound join aborted: {len(full_df) - n_unique_keys} duplicate "
            "(user_id, email_id, date_sent) keys detected in full_df."
        )

    user_date_level, email_date_level = pl.collect_all([user_date_lazy, email_date_lazy])

    joined = (
        full_df.join(user_date_level, on=["user_id", "date_sent"], how="left")
        .join(email_date_level, on=["email_id", "date_sent"], how="left")
        .select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "user_prior_engagement_score",
                "email_popularity_score",
            ]
        )
    )

    if len(joined) != len(full_df):
        raise ConfoundJoinError(
            f"Confound join produced {len(joined)} rows but expected {len(full_df)} "
            f"(full_df). Row count mismatch after join."
        )
    return joined


# ---------------------------------------------------------------------------
# Strict-causality audit
# ---------------------------------------------------------------------------


def _assert_no_future_leak(
    df: pl.DataFrame,
    score_col: str,
    time_col: str,
    group_col: str,
    n_sample: int = 1000,
    seed: int = 42,
    audit_sample_size: int | None = None,
) -> HarnessDiagnostic:
    """Re-derive score for a sample via vectorized cumsum and confirm max |diff| < 1e-9.

    Replaces the per-row filter loop with a single group_by + cumsum + shift(1)
    per confound. Strict `< t` semantics are preserved: aggregating to
    (group, time) then shift(1) means the cumsum at time t excludes time-t rows.
    """
    if df[time_col].is_null().any():
        raise LeakageError(
            f"Future-leak audit aborted: null values found in {time_col!r}. "
            "All rows must have a non-null time value."
        )

    _effective_sample = audit_sample_size if audit_sample_size is not None else n_sample
    rng = np.random.default_rng(seed)
    n_audited = min(_effective_sample, len(df))
    sample_idx = rng.choice(len(df), size=n_audited, replace=False)
    sample = df[sample_idx]

    if score_col == "user_prior_engagement_score":
        date_level = (
            df.group_by([group_col, time_col])
            .agg(pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"))
            .sort([group_col, time_col])
            .with_columns(
                pl.col("_acts_on_date")
                .cum_sum()
                .shift(1)
                .over(group_col)
                .fill_null(0)
                .alias("_prior_acts")
            )
            .select([group_col, time_col, "_prior_acts"])
        )
        joined = sample.join(date_level, on=[group_col, time_col], how="left").with_columns(
            pl.col("_prior_acts").fill_null(0)
        )
        expected = np.log1p(joined["_prior_acts"].to_numpy().astype(np.float64))

    elif score_col == "email_popularity_score":
        date_level = (
            df.group_by([group_col, time_col])
            .agg(
                pl.len().cast(pl.Int64).alias("_sends_on_date"),
                pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"),
            )
            .sort([group_col, time_col])
            .with_columns(
                pl.col("_sends_on_date")
                .cum_sum()
                .shift(1)
                .over(group_col)
                .fill_null(0)
                .alias("_prior_sends"),
                pl.col("_acts_on_date")
                .cum_sum()
                .shift(1)
                .over(group_col)
                .fill_null(0)
                .alias("_prior_acts"),
            )
            .select([group_col, time_col, "_prior_sends", "_prior_acts"])
        )
        joined = sample.join(date_level, on=[group_col, time_col], how="left").with_columns(
            pl.col("_prior_sends").fill_null(0),
            pl.col("_prior_acts").fill_null(0),
        )
        prior_acts = joined["_prior_acts"].to_numpy().astype(np.float64)
        prior_sends = joined["_prior_sends"].to_numpy().astype(np.float64)
        expected = (prior_acts + 1) / (prior_sends + 2)

    else:
        return HarnessDiagnostic(
            name="future_leak_audit",
            severity="info",
            message=f"Strict-causality audit skipped for unknown score_col={score_col!r}",
            values={"n_audited": 0, "max_diff": 0.0, "score_col": score_col},
        )

    actual = joined[score_col].to_numpy().astype(np.float64)
    diffs = np.abs(actual - expected)
    max_diff = float(diffs.max()) if len(diffs) > 0 else 0.0

    if max_diff > 1e-9:
        raise LeakageError(
            f"Future-leak detected in {score_col}: max |diff| = {max_diff:.2e} > 1e-9"
        )

    return HarnessDiagnostic(
        name="future_leak_audit",
        severity="info",
        message=f"Strict-causality audit passed for {score_col}: n_audited={n_audited}, max_diff={max_diff:.2e}",
        values={"n_audited": n_audited, "max_diff": max_diff, "score_col": score_col},
    )


# ---------------------------------------------------------------------------
# Cross-fit isotonic residualization
# ---------------------------------------------------------------------------


def _compute_residualized_auc_cross_fit(
    y_test: np.ndarray,
    s_test: np.ndarray,
    c_u_test: np.ndarray,
    c_e_test: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray, float]:
    """Two-way cross-fit sequential isotonic residualization.

    Returns (auc_residualized, auc_direction_a, auc_direction_b, s_resid_xfit, iso_seconds).
    s_resid_xfit is aligned to the input row order: each row's residual is
    the output of the smoother fitted on the half it was NOT in.
    iso_seconds is the cumulative wall time spent on IsotonicRegression.fit() calls only.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    half_a_idx = rng.choice(n, size=n // 2, replace=False)
    half_b_idx = np.setdiff1d(np.arange(n), half_a_idx)

    y_a, s_a = y_test[half_a_idx], s_test[half_a_idx]
    c_u_a, c_e_a = c_u_test[half_a_idx], c_e_test[half_a_idx]

    y_b, s_b = y_test[half_b_idx], s_test[half_b_idx]
    c_u_b, c_e_b = c_u_test[half_b_idx], c_e_test[half_b_idx]

    _iso_seconds = 0.0

    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata

        return rankdata(arr, method="average") / len(arr)

    # T1.3: precompute confound ranks once per half (each set of 4 ranks is used twice
    # across the two _residualize_sequential calls — once as fit-side, once as apply-side).
    rank_u_a = _rank_normalize(c_u_a)
    rank_e_a = _rank_normalize(c_e_a)
    rank_u_b = _rank_normalize(c_u_b)
    rank_e_b = _rank_normalize(c_e_b)

    def _residualize_sequential(
        s_fit: np.ndarray,
        r_u_fit: np.ndarray,
        r_e_fit: np.ndarray,
        s_apply: np.ndarray,
        r_u_apply: np.ndarray,
        r_e_apply: np.ndarray,
    ) -> np.ndarray:
        nonlocal _iso_seconds

        iso_u = IsotonicRegression(out_of_bounds="clip")
        _t_iso = time.perf_counter()
        iso_u.fit(r_u_fit, s_fit)
        _iso_seconds += time.perf_counter() - _t_iso
        s_hat_u_fit = iso_u.predict(r_u_fit)
        s_hat_u_apply = iso_u.predict(r_u_apply)

        resid_fit = s_fit - s_hat_u_fit

        iso_e = IsotonicRegression(out_of_bounds="clip")
        _t_iso = time.perf_counter()
        iso_e.fit(r_e_fit, resid_fit)
        _iso_seconds += time.perf_counter() - _t_iso
        s_hat_e_apply = iso_e.predict(r_e_apply)

        return s_apply - s_hat_u_apply - s_hat_e_apply

    resid_b = _residualize_sequential(s_a, rank_u_a, rank_e_a, s_b, rank_u_b, rank_e_b)
    resid_a = _residualize_sequential(s_b, rank_u_b, rank_e_b, s_a, rank_u_a, rank_e_a)

    s_resid_xfit = np.empty_like(s_test)
    s_resid_xfit[half_a_idx] = resid_a
    s_resid_xfit[half_b_idx] = resid_b

    if len(np.unique(y_b)) < 2 or len(np.unique(y_a)) < 2:
        return 0.5, 0.5, 0.5, s_resid_xfit, _iso_seconds

    auc_b = float(roc_auc_score(y_b, resid_b))
    auc_a = float(roc_auc_score(y_a, resid_a))

    n_a, n_b = len(half_a_idx), len(half_b_idx)
    auc_combined = (n_a * auc_a + n_b * auc_b) / (n_a + n_b)
    return auc_combined, auc_a, auc_b, s_resid_xfit, _iso_seconds


# ---------------------------------------------------------------------------
# Single-fit isotonic residualization (T4.3 — fast_iter only)
# ---------------------------------------------------------------------------


def _compute_residualized_auc_single_fit(
    y_test: np.ndarray,
    s_test: np.ndarray,
    c_u_test: np.ndarray,
    c_e_test: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray, float]:
    """Single-fit isotonic residualization (T4.3 — fast_iter mode only).

    Fits isotonic regression ONCE on the full test fold and applies it to itself.
    Single-fit adds test-noise-leak risk; OK for directional fast-iter verdicts.
    Returns same 5-tuple as _compute_residualized_auc_cross_fit for drop-in routing.
    """
    from scipy.stats import rankdata

    _iso_seconds = 0.0

    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        return rankdata(arr, method="average") / len(arr)

    rank_u = _rank_normalize(c_u_test)
    rank_e = _rank_normalize(c_e_test)

    iso_u = IsotonicRegression(out_of_bounds="clip")
    _t = time.perf_counter()
    iso_u.fit(rank_u, s_test)
    _iso_seconds += time.perf_counter() - _t
    resid = s_test - iso_u.predict(rank_u)

    iso_e = IsotonicRegression(out_of_bounds="clip")
    _t = time.perf_counter()
    iso_e.fit(rank_e, resid)
    _iso_seconds += time.perf_counter() - _t
    s_resid = resid - iso_e.predict(rank_e)

    if len(np.unique(y_test)) < 2:
        return 0.5, 0.5, 0.5, s_resid, _iso_seconds

    auc = float(roc_auc_score(y_test, s_resid))
    return auc, auc, auc, s_resid, _iso_seconds


# ---------------------------------------------------------------------------
# Vectorized rank-based AUC (T1.2 — replaces sklearn in bootstrap hot loop)
# ---------------------------------------------------------------------------


def _rank_auc_numpy(y: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata

    if np.any(np.isnan(y)) or np.any(np.isnan(scores)):
        raise ValueError("scores/labels contain NaN")
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# cvAUC point estimate (T3b — comparison_fast fast-path cross-check)
# ---------------------------------------------------------------------------


def _cv_auc_point_estimate(fold_results: list[dict]) -> float:
    """Compute the cross-validated AUC point estimate via Mann-Whitney U
    averaged across folds. Fast cross-check against the bootstrap mean —
    NOT a CI replacement. Returns a single float.
    """
    fold_aucs = []
    for fold in fold_results:
        y = fold["y_test"]
        s = fold["s_resid"]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fold_aucs.append(_rank_auc_numpy(y, s))
    if not fold_aucs:
        return float("nan")
    return float(np.mean(fold_aucs))


# ---------------------------------------------------------------------------
# Weighted-rank AUC (T2.1 — pre-sorted input, integer multiplicity weights)
# ---------------------------------------------------------------------------


def _weighted_rank_auc(
    y_sorted_asc: np.ndarray,
    s_sorted_asc: np.ndarray,
    weights: np.ndarray,
) -> float:
    """AUC via Mann-Whitney U prefix-sum on a pre-sorted (ascending score) array.

    Args:
        y_sorted_asc: Binary labels in ascending-score order (already sorted).
        s_sorted_asc: Scores in ascending order (same permutation as y_sorted_asc).
        weights: Integer multiplicity per row (e.g. 2 = row drawn twice in resample).

    Returns float("nan") for degenerate inputs (W_pos == 0 or W_neg == 0).

    Tie handling: rows with identical scores form a tie group. Each positive
    is credited with (cum_neg_below + W_neg_in_group / 2), which is the weighted
    average-rank identity for the Mann-Whitney U statistic and matches
    scipy.stats.rankdata(method="average") on the equivalent expanded array.
    """
    is_pos = y_sorted_asc == 1
    w_pos_row = weights * is_pos
    w_neg_row = weights * ~is_pos
    W_pos = float(w_pos_row.sum())
    W_neg = float(w_neg_row.sum())
    if W_pos == 0.0 or W_neg == 0.0:
        return float("nan")

    # Vectorized tie-group reduction. Group boundaries = where score changes.
    # np.add.reduceat at these indices sums each tied-score block in pure C.
    n = len(s_sorted_asc)
    if n == 1:
        group_starts = np.array([0], dtype=np.int64)
    else:
        change_mask = np.empty(n, dtype=bool)
        change_mask[0] = True
        np.not_equal(s_sorted_asc[1:], s_sorted_asc[:-1], out=change_mask[1:])
        group_starts = np.flatnonzero(change_mask)

    group_w_pos = np.add.reduceat(w_pos_row.astype(np.float64), group_starts)
    group_w_neg = np.add.reduceat(w_neg_row.astype(np.float64), group_starts)

    cum_neg_above = np.cumsum(group_w_neg)
    cum_neg_below = cum_neg_above - group_w_neg  # exclusive prefix sum
    u_stat = float(np.sum(group_w_pos * (cum_neg_below + group_w_neg * 0.5)))

    return u_stat / (W_pos * W_neg)


# ---------------------------------------------------------------------------
# Numba-JIT weighted-AUC kernel (T3.1)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, nogil=True)
def _weighted_rank_auc_numba(
    y_sorted_asc: np.ndarray,
    s_sorted_asc: np.ndarray,
    weights: np.ndarray,
) -> float:
    n = len(y_sorted_asc)
    W_pos = 0.0
    W_neg = 0.0
    for k in range(n):
        w = weights[k]
        if y_sorted_asc[k] == 1:
            W_pos += w
        else:
            W_neg += w
    if W_pos == 0.0 or W_neg == 0.0:
        return np.nan
    cum_neg_below = 0.0
    u_stat = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted_asc[j] == s_sorted_asc[i]:
            j += 1
        group_w_pos = 0.0
        group_w_neg = 0.0
        for k in range(i, j):
            w = weights[k]
            if y_sorted_asc[k] == 1:
                group_w_pos += w
            else:
                group_w_neg += w
        u_stat += group_w_pos * (cum_neg_below + group_w_neg * 0.5)
        cum_neg_below += group_w_neg
        i = j
    return u_stat / (W_pos * W_neg)


# Warm the JIT at import time so the first real call doesn't pay compile latency.
_weighted_rank_auc_numba(
    np.array([1, 0], dtype=np.int64),
    np.array([0.5, 0.7], dtype=np.float64),
    np.array([1, 1], dtype=np.int32),
)


def _build_sorted_user_map(
    y: np.ndarray,
    s_resid: np.ndarray,
    user_ids: np.ndarray,
    unique_users: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-sort by ascending score; return sorted labels, sorted scores, and row→user-index array.

    The returned `row_to_user_idx[i]` gives the index of row i's user in `unique_users`
    (which is sorted ascending). Used to vectorize per-resample weight construction via
    `weights = user_multiplicity[row_to_user_idx]`.
    """
    sort_idx = np.argsort(s_resid, kind="stable")
    y_sorted = y[sort_idx]
    s_sorted = s_resid[sort_idx]
    uid_sorted = user_ids[sort_idx]
    row_to_user_idx = np.searchsorted(unique_users, uid_sorted).astype(np.int64)
    return y_sorted, s_sorted, row_to_user_idx


# ---------------------------------------------------------------------------
# User-block bootstrap
# ---------------------------------------------------------------------------


def _user_block_bootstrap(
    y: np.ndarray,
    s_resid: np.ndarray,
    user_ids: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    bootstrap_n_jobs: int = -1,
    *,
    sorted_user_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Sample whole users with replacement, preserving multiplicity, recompute AUC per resample.

    T2.1: pre-sort scores once per fold, then compute bootstrap AUCs via
    per-row integer weights rather than full row-index expansion.
    """
    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)
    n_users = len(unique_users)

    if sorted_user_map is None:
        y_sorted, s_sorted, row_to_user_idx = _build_sorted_user_map(
            y, s_resid, user_ids, unique_users
        )
    else:
        y_sorted, s_sorted, row_to_user_idx = sorted_user_map

    child_seeds = rng.integers(0, 2**32 - 1, size=n_boot)
    uniform_pvals = np.full(n_users, 1.0 / n_users)

    def _one_replicate(rep_seed: int) -> float:
        rep_rng = np.random.default_rng(rep_seed)
        user_multiplicity = rep_rng.multinomial(n_users, uniform_pvals).astype(np.int32, copy=False)
        weights = user_multiplicity[row_to_user_idx]
        return _weighted_rank_auc_numba(
            y_sorted.astype(np.int64, copy=False),
            s_sorted.astype(np.float64, copy=False),
            weights,
        )

    _resolved_jobs = bootstrap_n_jobs if bootstrap_n_jobs != -1 else n_jobs
    _effective_jobs = 1 if (_resolved_jobs == 1 or len(y) < 50_000) else _resolved_jobs
    results = Parallel(n_jobs=_effective_jobs, prefer="threads")(
        delayed(_one_replicate)(int(s)) for s in child_seeds
    )
    return np.array(results, dtype=np.float64)


def _pooled_user_block_bootstrap_mean_of_folds(
    fold_results: list[dict],
    n_boot: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    bootstrap_n_jobs: int = -1,
) -> tuple[float, float]:
    """Cluster-bootstrap the mean-across-folds AUC estimator.

    Cluster unit = user (a user can appear in multiple folds' test sets).
    Each replicate resamples users globally with replacement, recomputes
    each fold's AUC using only the rows belonging to sampled users in
    that fold (with multiplicity preserved), then takes the mean of the
    per-fold AUCs. CI = quantiles of replicate statistics.

    T2.1: pre-sorts scores once per fold; uses weighted-AUC prefix-sum
    instead of expanded row-index concatenation.
    """
    rng = np.random.default_rng(seed)

    all_user_ids = np.concatenate([r["user_ids"] for r in fold_results])
    unique_users_global = np.unique(all_user_ids)
    n_users_global = len(unique_users_global)

    # Build per-fold sorted data and precompute global→fold user-index mapping once.
    # Shape of fold entry: (y_sorted, s_sorted, row_to_user_idx, local_idx, in_mask, n_fold_users)
    fold_sorted_data: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
    ] = []
    for r in fold_results:
        uids = r["user_ids"]
        fold_unique = np.unique(uids)
        n_fold = len(fold_unique)
        if "y_sorted" in r:
            y_s = r["y_sorted"]
            s_s = r["s_sorted"]
            fold_row_to_user_idx = r["row_to_user_idx"]
        else:
            y_s, s_s, fold_row_to_user_idx = _build_sorted_user_map(
                r["y_test"], r["s_resid"], uids, fold_unique
            )
        # For each global user, its index in fold_unique; -1 if absent.
        pos = np.searchsorted(fold_unique, unique_users_global)
        pos_clipped = np.clip(pos, 0, n_fold - 1)
        in_fold = fold_unique[pos_clipped] == unique_users_global
        mapping = np.where(in_fold, pos_clipped, -1).astype(np.int32)
        in_mask = mapping >= 0
        local_idx = mapping[in_mask].astype(np.int32)
        fold_sorted_data.append((y_s, s_s, fold_row_to_user_idx, local_idx, in_mask, n_fold))

    child_seeds = rng.integers(0, 2**32 - 1, size=n_boot)
    uniform_pvals_global = np.full(n_users_global, 1.0 / n_users_global)

    def _one_replicate(rep_seed: int) -> float:
        rep_rng = np.random.default_rng(rep_seed)
        counts_global = rep_rng.multinomial(n_users_global, uniform_pvals_global).astype(
            np.int32, copy=False
        )
        fold_aucs_replicate: list[float] = []
        for (
            y_sorted,
            s_sorted_fold,
            fold_row_to_user_idx,
            local_idx,
            in_mask,
            n_fold,
        ) in fold_sorted_data:
            # Scatter global counts into fold-local via precomputed mapping — no per-fold searchsorted.
            counts_fold = np.bincount(
                local_idx,
                weights=counts_global[in_mask],
                minlength=n_fold,
            ).astype(np.int32, copy=False)
            if counts_fold.sum() == 0:
                continue
            weights = counts_fold[fold_row_to_user_idx]
            auc_b = _weighted_rank_auc_numba(
                y_sorted.astype(np.int64, copy=False),
                s_sorted_fold.astype(np.float64, copy=False),
                weights,
            )
            if not np.isnan(auc_b):
                fold_aucs_replicate.append(auc_b)
        if not fold_aucs_replicate:
            return 0.5
        return float(np.mean(fold_aucs_replicate))

    _resolved_jobs = bootstrap_n_jobs if bootstrap_n_jobs != -1 else n_jobs
    _effective_jobs = 1 if _resolved_jobs == 1 else _resolved_jobs
    results = Parallel(n_jobs=_effective_jobs, prefer="threads")(
        delayed(_one_replicate)(int(s)) for s in child_seeds
    )
    replicate_stats = np.array(results, dtype=np.float64)

    return (
        float(np.nanquantile(replicate_stats, 0.025)),
        float(np.nanquantile(replicate_stats, 0.975)),
    )


# ---------------------------------------------------------------------------
# Walk-forward fold construction
# ---------------------------------------------------------------------------


def _quarterly_folds(
    df: pl.DataFrame,
    date_col: str = "date_sent",
) -> list[tuple[str, pl.DataFrame, pl.DataFrame]]:
    """Returns [(quarter_label, train_df, test_df), ...] sorted by quarter.

    Quarter labels: "YYYY-QN" (e.g. "2024-Q2").
    First quarter (no prior history) is skipped.
    Cumulative-train: fold k trains on quarters [Q1..Q_{k-1}].
    """
    quarters = df.select(
        (
            pl.col(date_col).dt.year().cast(pl.Utf8)
            + "-Q"
            + pl.col(date_col).dt.quarter().cast(pl.Utf8)
        )
    ).to_series()

    quarter_labels: list[str] = sorted(set(quarters.to_list()))

    folds = []
    for k, q_label in enumerate(quarter_labels):
        if k == 0:
            continue
        train_mask = quarters.is_in(quarter_labels[:k])
        test_mask = quarters == q_label
        train_df = df.filter(train_mask)
        test_df = df.filter(test_mask)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        folds.append((q_label, train_df, test_df))

    return folds


# ---------------------------------------------------------------------------
# PSI helper
# ---------------------------------------------------------------------------


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    bins = np.linspace(0, 1, n_bins + 1)
    bins[0], bins[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = (exp_counts + 1e-8) / (exp_counts.sum() + 1e-8 * len(exp_counts))
    act_pct = (act_counts + 1e-8) / (act_counts.sum() + 1e-8 * len(act_counts))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _psi_against_ref(
    actual: np.ndarray,
    ref_counts: np.ndarray,
    ref_bins: np.ndarray,
) -> float:
    """PSI of `actual` against a pre-computed reference histogram."""
    act_counts, _ = np.histogram(actual, bins=ref_bins)
    exp_pct = (ref_counts + 1e-8) / (ref_counts.sum() + 1e-8 * len(ref_counts))
    act_pct = (act_counts + 1e-8) / (act_counts.sum() + 1e-8 * len(act_counts))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ---------------------------------------------------------------------------
# Tenure bucket helper
# ---------------------------------------------------------------------------


def _assign_tenure_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Adds tenure_bucket column based on user-level days since first send."""
    first_send = df.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send"))
    df = df.join(first_send, on="user_id", how="left")
    df = df.with_columns(
        ((pl.col("date_sent") - pl.col("first_send")).dt.total_days())
        .cast(pl.Int64)
        .alias("_tenure_days")
    )
    return df.with_columns(
        pl.col("_tenure_days")
        .cut(breaks=_TENURE_BREAKS, labels=_TENURE_LABELS)
        .alias("tenure_bucket")
    ).drop(["first_send", "_tenure_days"])


# ---------------------------------------------------------------------------
# Per-stratum residualized AUC
# ---------------------------------------------------------------------------


def _per_tenure_residualized_auc(
    y: np.ndarray,
    s_resid: np.ndarray,
    tenure_buckets: np.ndarray,
    c_u: np.ndarray,
    c_e: np.ndarray,
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for label in _TENURE_LABELS:
        mask = tenure_buckets == label
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            results[label] = None
            continue
        try:
            results[label] = float(roc_auc_score(y[mask], s_resid[mask]))
        except Exception:
            results[label] = None
    return results


# ---------------------------------------------------------------------------
# Single-confound residualization (Recipe 1)
# ---------------------------------------------------------------------------


def _residualize_single_confound(
    y: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    seed: int = 42,
) -> tuple[float, float]:
    """Residualize s against a single confound c via isotonic regression.

    Returns (auc, iso_seconds) where iso_seconds is cumulative IsotonicRegression.fit() time.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    half_a_idx = rng.choice(n, size=n // 2, replace=False)
    half_b_idx = np.setdiff1d(np.arange(n), half_a_idx)

    from scipy.stats import rankdata

    _iso_seconds = 0.0

    def _do(fit_idx: np.ndarray, apply_idx: np.ndarray) -> float:
        nonlocal _iso_seconds
        r_fit = rankdata(c[fit_idx], method="average") / len(fit_idx)
        r_apply = rankdata(c[apply_idx], method="average") / len(apply_idx)
        iso = IsotonicRegression(out_of_bounds="clip")
        _t_iso = time.perf_counter()
        iso.fit(r_fit, s[fit_idx])
        _iso_seconds += time.perf_counter() - _t_iso
        resid = s[apply_idx] - iso.predict(r_apply)
        if len(np.unique(y[apply_idx])) < 2:
            return 0.5
        return float(roc_auc_score(y[apply_idx], resid))

    auc_b = _do(half_a_idx, half_b_idx)
    auc_a = _do(half_b_idx, half_a_idx)
    auc = (auc_a * len(half_a_idx) + auc_b * len(half_b_idx)) / (len(half_a_idx) + len(half_b_idx))
    return auc, _iso_seconds


# ---------------------------------------------------------------------------
# Confound overlap diagnostic
# ---------------------------------------------------------------------------


def _compute_confound_overlap_diagnostic(
    fold_label: str,
    c_u_test: np.ndarray,
    c_e_test: np.ndarray,
    y_test: np.ndarray,
    n_quartiles: int = 4,
    min_positives_threshold: int = 100,
) -> HarnessDiagnostic:
    """4×4 joint quartile positive-count diagnostic on a single test fold."""
    u_unique = len(np.unique(c_u_test))
    e_unique = len(np.unique(c_e_test))
    if u_unique < n_quartiles or e_unique < n_quartiles:
        return HarnessDiagnostic(
            name=f"confound_overlap_{fold_label}",
            severity="info",
            message=f"Confound has < {n_quartiles} distinct test-fold values; quartile binning collapsed",
            values={
                "fold": fold_label,
                "n_cells_below_threshold": 0,
                "min_positives_per_cell": int(y_test.sum()),
            },
        )

    q_u = np.quantile(c_u_test, np.linspace(0, 1, n_quartiles + 1))
    q_e = np.quantile(c_e_test, np.linspace(0, 1, n_quartiles + 1))

    u_bin = np.searchsorted(q_u[1:-1], c_u_test, side="right")
    e_bin = np.searchsorted(q_e[1:-1], c_e_test, side="right")

    cell_counts = np.zeros((n_quartiles, n_quartiles), dtype=np.int64)
    np.add.at(cell_counts, (u_bin, e_bin), y_test.astype(np.int64))

    n_cells_below = int((cell_counts < min_positives_threshold).sum())
    min_pos = int(cell_counts.min())

    sev: Literal["info", "warning"] = "warning" if n_cells_below > 0 else "info"
    return HarnessDiagnostic(
        name=f"confound_overlap_{fold_label}",
        severity=sev,
        message=(
            f"Fold {fold_label}: {n_cells_below}/{n_quartiles**2} confound overlap cells "
            f"have < {min_positives_threshold} positives (min={min_pos})"
        ),
        values={
            "fold": fold_label,
            "n_cells_below_threshold": n_cells_below,
            "min_positives_per_cell": min_pos,
        },
    )


# ---------------------------------------------------------------------------
# Cell-level class balance diagnostic
# ---------------------------------------------------------------------------


def _compute_cell_class_balance_diagnostic(
    combined: pl.DataFrame,
    y_all: np.ndarray,
    tenure_all: np.ndarray,
    min_positives_threshold: int = 5,
) -> tuple[HarnessDiagnostic, set[tuple[str, str]]]:
    """Per-(tenure_bucket, prior_action_bin) positive count on the test fold.

    Returns the diagnostic and the set of excluded (tenure, prior_action_bin) cells.
    """
    prior_action_col = (
        combined["lifetime_actions_prior"] if "lifetime_actions_prior" in combined.columns else None
    )

    if prior_action_col is not None:
        prior_vals = prior_action_col.to_numpy().astype(np.int64)
        prior_bin_idx = np.searchsorted(_PRIOR_ACTION_BINS[1:], prior_vals, side="right")
        prior_bin_arr = np.array(_PRIOR_ACTION_LABELS)[prior_bin_idx]
    else:
        prior_bin_arr = np.full(len(y_all), "0")

    diag_df = pl.DataFrame(
        {
            "tenure": tenure_all,
            "prior_action_bin": prior_bin_arr,
            "y": y_all.astype(np.int64),
        }
    )
    cell_counts_df = diag_df.group_by(["tenure", "prior_action_bin"]).agg(
        pl.col("y").sum().alias("positives")
    )
    observed: dict[tuple[str, str], int] = {
        (str(row["tenure"]), str(row["prior_action_bin"])): int(row["positives"])
        for row in cell_counts_df.iter_rows(named=True)
    }
    cell_positives: dict[tuple[str, str], int] = {}
    for t_label in _TENURE_LABELS:
        for p_label in _PRIOR_ACTION_LABELS:
            cell_positives[(t_label, p_label)] = observed.get((t_label, p_label), 0)

    excluded: set[tuple[str, str]] = {
        k for k, v in cell_positives.items() if v < min_positives_threshold
    }
    n_excluded = len(excluded)
    min_pos = min(cell_positives.values()) if cell_positives else 0

    sev: Literal["info", "error"] = "error" if n_excluded > 0 else "info"
    return (
        HarnessDiagnostic(
            name="class_balance",
            severity=sev,
            message=(
                f"{n_excluded}/{len(cell_positives)} (tenure×prior_action) cells excluded "
                f"(< {min_positives_threshold} positives); min positives per cell = {min_pos}"
            ),
            values={
                "n_cells_excluded": n_excluded,
                "min_positives_per_cell": min_pos,
                "excluded_cells": [f"{t}|{p}" for t, p in sorted(excluded)],
            },
        ),
        excluded,
    )


# ---------------------------------------------------------------------------
# Per-fold user main effect (LightGBM on user-side features only)
# ---------------------------------------------------------------------------


def _compute_user_main_effect_lgbm(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    ml_model_config: ML_Model_Config,
    outcome_variable: str,
    seed: int,
    inner_threads: int,
    fold_k: int = 0,
    cache: HarnessCache | None = None,
) -> np.ndarray:
    """Train a lightweight LightGBM on train_df and return test_df probabilities.

    Uses n_estimators=100 (T2.3 reverted: 50-estimator predictions caused a primary-metric
    drift of -0.029 on the 5% probe, well beyond the Codex drift gate). Cache bypass on
    warm runs via T1.1. Train/test separation guarantees no future leak by construction.
    """
    fingerprint: str | None = None
    if cache is not None:
        fingerprint = fold_train_fingerprint(train_df, fold_k=fold_k, user_features=feature_cols)
        cached = cache.get_user_main_effect_predictions(fingerprint=fingerprint, fold_k=fold_k)
        if cached is not None:
            return cached["predictions"].to_numpy().astype(np.float64)

    args = dict(ml_model_config.args)
    args["n_estimators"] = 100
    args["random_state"] = seed
    args["n_jobs"] = inner_threads
    args["verbose"] = -1

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df[outcome_variable].to_numpy().astype(np.float32)
    X_test = test_df.select(feature_cols).fill_null(0).to_numpy()

    model = LGBMClassifier(**args)
    model.fit(X_train, y_train)
    preds = model.predict(X_test, raw_score=True)

    if cache is not None and fingerprint is not None:
        cache.put_user_main_effect_predictions(
            fingerprint=fingerprint,
            fold_k=fold_k,
            df=pl.DataFrame({"predictions": preds}),
        )

    return preds


# ---------------------------------------------------------------------------
# Per-fold user main effect (LogisticRegression on user-side features only)
# ---------------------------------------------------------------------------


def _compute_user_main_effect_lr(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    ml_model_config: ML_Model_Config,
    outcome_variable: str,
    seed: int,
    inner_threads: int,
    fold_k: int = 0,
    cache: HarnessCache | None = None,
) -> np.ndarray:
    _LGBM_ONLY_KEYS = {
        "num_leaves",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "force_col_wise",
        "feature_pre_filter",
        "bin_construct_sample_cnt",
        "verbose",
    }
    lr_args: dict[str, Any] = {
        k: v for k, v in dict(ml_model_config.args).items() if k not in _LGBM_ONLY_KEYS
    }
    lr_args.setdefault("random_state", seed)
    lr_args.setdefault("max_iter", 200)
    lr_args.setdefault("tol", 1e-4)

    penalty = lr_args.get("penalty", "l2")
    if "solver" not in lr_args:
        if penalty == "l2":
            lr_args["solver"] = "newton-cholesky"
        elif penalty == "l1":
            lr_args["solver"] = "liblinear"
        elif penalty == "elasticnet":
            lr_args["solver"] = "saga"
            lr_args.setdefault("l1_ratio", 0.5)

    fp = fold_train_fingerprint(train_df, fold_k, feature_cols) + "_lr"
    if cache is not None:
        cached = cache.get_user_main_effect_predictions(fp, fold_k)
        if cached is not None:
            return cached["predictions"].to_numpy().astype(np.float64)

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy().astype(np.float64)
    y_train = train_df.get_column(outcome_variable).to_numpy().astype(np.int32)
    X_test = test_df.select(feature_cols).fill_null(0).to_numpy().astype(np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(**lr_args)
    model.fit(X_train_scaled, y_train)
    raw_test = model.decision_function(X_test_scaled).astype(np.float64)

    if cache is not None:
        cache.put_user_main_effect_predictions(fp, fold_k, pl.DataFrame({"predictions": raw_test}))
    return raw_test


# ---------------------------------------------------------------------------
# Per-fold worker (pure function — safe for joblib loky dispatch)
# ---------------------------------------------------------------------------


def _run_one_fold(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    q_label: str,
    feature_cols: list[str],
    outcome_variable: str,
    ml_model_config: ML_Model_Config,
    inner_threads: int,
    confound_overlap_threshold: int,
    seed: int,
    bootstrap_n_jobs: int,
    fold_k: int = 0,
    bootstrap_n_resamples: int = 500,
    cache: HarnessCache | None = None,
    scope: str = "comparison",
    report_old_rfm: bool = False,
) -> dict[str, Any]:
    """Execute one quarterly fold: fit → predict → residualize → score → bootstrap CI."""
    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df[outcome_variable].to_numpy().astype(np.float32)
    X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test = test_df[outcome_variable].to_numpy().astype(np.float32)

    c_u_test = test_df["user_prior_engagement_score"].to_numpy()
    c_e_test = test_df["email_popularity_score"].to_numpy()

    fold_lr_coefs: dict[str, float] | None = None
    fold_lr_intercept: float | None = None
    fold_lr_feature_means: list[float] | None = None
    fold_lr_feature_stds: list[float] | None = None

    # === Dispatch on model type ===
    if ml_model_config.ml_model_type == ML_Model_Type.LIGHTGBM_CLASSIFIER:
        lgbm_args = dict(ml_model_config.args)
        lgbm_args["n_jobs"] = inner_threads
        lgbm_args["random_state"] = seed
        lgbm_args["verbose"] = -1

        model_lgbm = LGBMClassifier(**lgbm_args)
        model_lgbm.fit(X_train, y_train)

        # Single inference per split; derive proba once for PSI/row_loss path.
        raw_test = model_lgbm.predict(X_test, raw_score=True)
        raw_train = model_lgbm.predict(X_train, raw_score=True)

    elif ml_model_config.ml_model_type == ML_Model_Type.LOGISTIC_REGRESSION:
        _LGBM_ONLY_KEYS = {
            "num_leaves",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "force_col_wise",
            "feature_pre_filter",
            "bin_construct_sample_cnt",
            "verbose",
        }
        lr_args: dict[str, Any] = {
            k: v for k, v in dict(ml_model_config.args).items() if k not in _LGBM_ONLY_KEYS
        }
        lr_args.setdefault("random_state", seed)
        lr_args.setdefault("max_iter", 200)
        lr_args.setdefault("tol", 1e-4)
        penalty = lr_args.get("penalty", "l2")
        if "solver" not in lr_args:
            if penalty == "l2":
                lr_args["solver"] = "newton-cholesky"
            elif penalty == "l1":
                lr_args["solver"] = "liblinear"
            elif penalty == "elasticnet":
                lr_args["solver"] = "saga"
                lr_args.setdefault("l1_ratio", 0.5)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        X_test_scaled = scaler.transform(X_test.astype(np.float64))

        model_lr = LogisticRegression(**lr_args)
        model_lr.fit(X_train_scaled, y_train)
        raw_test = model_lr.decision_function(X_test_scaled).astype(np.float64)
        raw_train = model_lr.decision_function(X_train_scaled).astype(np.float64)

        fold_lr_coefs = dict(zip(feature_cols, model_lr.coef_[0].tolist()))
        fold_lr_intercept = float(model_lr.intercept_[0])
        fold_lr_feature_means = scaler.mean_.tolist()
        fold_lr_feature_stds = scaler.scale_.tolist()

    else:
        raise ValueError(f"Unsupported ml_model_type: {ml_model_config.ml_model_type}")

    # AUC + residualization: rank-invariant under sigmoid — use raw scores.
    s_test = raw_test
    s_train = raw_train

    # Proba for consumers that require [0,1]: PSI bins and row_loss (cross-entropy).
    proba_test = 1.0 / (1.0 + np.exp(-raw_test))
    proba_train = 1.0 / (1.0 + np.exp(-raw_train))

    if ml_model_config.ml_model_type == ML_Model_Type.LIGHTGBM_CLASSIFIER:
        c_u_main_effect_test = _compute_user_main_effect_lgbm(
            train_df,
            test_df,
            feature_cols,
            ml_model_config,
            outcome_variable,
            seed=seed,
            inner_threads=inner_threads,
            fold_k=fold_k,
            cache=cache,
        )
    elif ml_model_config.ml_model_type == ML_Model_Type.LOGISTIC_REGRESSION:
        c_u_main_effect_test = _compute_user_main_effect_lr(
            train_df,
            test_df,
            feature_cols,
            ml_model_config,
            outcome_variable,
            seed=seed,
            inner_threads=inner_threads,
            fold_k=fold_k,
            cache=cache,
        )
    else:
        raise ValueError(f"Unsupported ml_model_type: {ml_model_config.ml_model_type}")

    _t_resid = time.perf_counter()
    # T4.3: single-fit adds test-noise-leak risk; OK for directional fast-iter verdicts.
    _resid_fn = (
        _compute_residualized_auc_single_fit
        if _is_fast_iter(scope)
        else _compute_residualized_auc_cross_fit
    )
    auc_residualized, auc_dir_a, auc_dir_b, s_resid_xfit, _iso_secs_primary = _resid_fn(
        y_test, s_test, c_u_main_effect_test, c_e_test, seed=seed
    )

    if report_old_rfm:
        auc_residualized_old, _, _, s_resid_old, _iso_secs_old = _resid_fn(
            y_test, s_test, c_u_test, c_e_test, seed=seed
        )
    else:
        auc_residualized_old = None
        s_resid_old = None
        _iso_secs_old = 0.0
    _resid_seconds = time.perf_counter() - _t_resid
    _iso_fold_seconds = _iso_secs_primary + _iso_secs_old

    user_ids_test = test_df["user_id"].to_numpy()
    unique_users_test = np.unique(user_ids_test)

    # Precompute once per fold; reused by _user_block_bootstrap and stored for
    # _pooled_user_block_bootstrap_mean_of_folds to avoid a duplicate call.
    _fold_sorted_user_map = _build_sorted_user_map(
        y_test, s_resid_xfit, user_ids_test, unique_users_test
    )
    y_sorted_fold, s_sorted_fold, row_to_user_idx_fold = _fold_sorted_user_map

    _t_boot = time.perf_counter()
    boot_aucs = _user_block_bootstrap(
        y_test,
        s_resid_xfit,
        user_ids_test,
        n_boot=bootstrap_n_resamples,
        seed=seed,
        n_jobs=-1,
        bootstrap_n_jobs=bootstrap_n_jobs,
        sorted_user_map=_fold_sorted_user_map,
    )
    _boot_seconds = time.perf_counter() - _t_boot
    ci_low = float(np.nanquantile(boot_aucs, 0.025))
    ci_high = float(np.nanquantile(boot_aucs, 0.975))

    fold_overlap_diag = _compute_confound_overlap_diagnostic(
        fold_label=q_label,
        c_u_test=c_u_main_effect_test,
        c_e_test=c_e_test,
        y_test=y_test,
        min_positives_threshold=confound_overlap_threshold,
    )

    pred_rows = test_df.with_columns(
        [
            pl.Series("score", proba_test),
            pl.Series("score_residualized", s_resid_xfit),
            pl.lit(q_label).alias("fold"),
            pl.Series(
                "row_loss",
                (
                    -y_test * np.log(np.clip(proba_test, 1e-7, 1 - 1e-7))
                    - (1 - y_test) * np.log(np.clip(1 - proba_test, 1e-7, 1 - 1e-7))
                ),
            ),
        ]
    )

    return {
        "q_label": q_label,
        "auc_residualized": auc_residualized,
        "auc_residualized_old": auc_residualized_old,
        "auc_dir_a": auc_dir_a,
        "auc_dir_b": auc_dir_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_test": y_test,
        "s_test": proba_test,
        "s_train": proba_train,
        "s_resid": s_resid_xfit,
        "s_resid_old": s_resid_old,
        "c_u_test": c_u_test,
        "c_u_main_effect_test": c_u_main_effect_test,
        "c_e_test": c_e_test,
        "user_ids": user_ids_test,
        "unique_users": unique_users_test,
        "y_sorted": y_sorted_fold,
        "s_sorted": s_sorted_fold,
        "row_to_user_idx": row_to_user_idx_fold,
        "test_df": test_df,
        "train_df": train_df,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fold_overlap_diag": fold_overlap_diag,
        "pred_rows": pred_rows,
        "_resid_seconds": _resid_seconds,
        "_iso_fold_seconds": _iso_fold_seconds,
        "_boot_seconds": _boot_seconds,
        "lr_coefs": fold_lr_coefs,
        "lr_intercept": fold_lr_intercept,
        "lr_feature_means": fold_lr_feature_means,
        "lr_feature_stds": fold_lr_feature_stds,
    }


# ---------------------------------------------------------------------------
# Public harness
# ---------------------------------------------------------------------------


def harness(
    data: pl.DataFrame,
    feature_cols: list[str],
    ml_model_config: ML_Model_Config,
    outcome_variable: str = "actioned_24h",
    *,
    sample_frac: float | None = None,
    sample_seed: int | None = None,
    predictions_dir: str | None = None,
    fold_n_jobs: int = 8,
    inner_threads: int | None = None,
    disable_cache: bool = False,
    bootstrap_n_resamples: int | None = None,
    scope: str = "comparison",
    report_supplementary: bool | None = None,
    report_old_rfm: bool | None = None,
    skip_audit_on_cache_hit: bool | None = None,
) -> CivicShoutHarnessResponse:
    """Evaluate features against actioned_24h via quarterly walk-forward.

    Primary metric: roc_auc_residualized_user_main_effect_x_email_popularity_pair.
    User main effect computed per-fold via LightGBM (train/test split). Email confound
    computed on FULL data (Laplace-smoothed). Old RFM metric kept as secondary.

    Scopes: smoke, reproduction, comparison (8-fold, 500 resamples, decision-bearing),
    champion_candidate (8-fold, 500 resamples, full secondary), fast_iter (3-fold, 100 resamples,
    directional only), comparison_fast (4-fold, 100 resamples, pilot-only — not decision-bearing).
    """
    wall_start = time.perf_counter()
    # T4.4: default resamples adapts to scope; callers can always override explicitly.
    if bootstrap_n_resamples is None:
        bootstrap_n_resamples = 100 if _is_fast_iter(scope) else 500
    # T3.2: supplementary single-confound residualizations are expensive (16 isotonic fits).
    # fast_iter and comparison_fast always skip; champion_candidate always runs; comparison defaults off.
    if _is_fast_iter(scope):
        report_supplementary = False
    elif report_supplementary is None:
        report_supplementary = scope == "champion_candidate"
    # T2a: old-RFM secondary residualization gated per scope.
    # champion_candidate defaults on; comparison + comparison_fast default off.
    if report_old_rfm is None:
        report_old_rfm = scope == "champion_candidate"
    _report_old_rfm: bool = report_old_rfm
    # v3.12: audit sample size — 200 for comparison/comparison_fast/fast_iter, 1000 for champion_candidate.
    _audit_sample_size: int = 200 if scope != "champion_candidate" else 1000
    # v3.12: skip audit on cache hit — default True for non-champion scopes.
    if skip_audit_on_cache_hit is None:
        skip_audit_on_cache_hit = scope != "champion_candidate"
    _skip_audit_on_cache_hit: bool = skip_audit_on_cache_hit
    stage_timings: list[HarnessStageTiming] = []
    diagnostics: list[HarnessDiagnostic] = []
    artifacts: list[HarnessArtifact] = []

    cpu_count = os.cpu_count() or 8
    _parallel_folds = fold_n_jobs != 1
    _inner_threads: int = (
        inner_threads
        if inner_threads is not None
        else (3 if _parallel_folds else max(cpu_count - 1, 1))
    )
    bootstrap_n_jobs = 3 if _parallel_folds else -1

    _is_smoke = sample_frac is not None and sample_frac <= 0.01
    _confound_overlap_threshold = 25 if _is_smoke else 100
    _class_balance_threshold = 50 if _is_smoke else 5

    # ------------------------------------------------------------------
    # Stage 1: column_selection
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    required_cols = {"user_id", "email_id", "date_sent", outcome_variable}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    forbidden_in_features = _FORBIDDEN_COLS & set(feature_cols)
    if forbidden_in_features:
        raise ValueError(
            f"Feature columns contain forbidden columns (F03 — post-treatment proxies): "
            f"{forbidden_in_features}. Remove them before calling harness()."
        )

    if outcome_variable != "actioned_24h":
        raise ValueError(
            f"harness v1 only supports outcome_variable='actioned_24h'; got '{outcome_variable}'"
        )

    all_needed = list(required_cols) + [c for c in feature_cols if c not in required_cols]
    work_df = data.select([c for c in all_needed if c in data.columns])
    work_df = work_df.with_columns(pl.col("actioned_24h").cast(pl.Int8))

    stage_timings.append(
        HarnessStageTiming(
            stage="column_selection", seconds=time.perf_counter() - t0, owner="harness"
        )
    )

    # ------------------------------------------------------------------
    # Stage 2: data_conversion — compute confound scores on FULL data,
    #           then optionally sample
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    population_n_rows = len(work_df)

    _harness_cache = HarnessCache()
    _confound_fp = data_fingerprint(work_df)

    _cached_confound = None if disable_cache else _harness_cache.get_confound_scores(_confound_fp)
    if _cached_confound is not None:
        confound_df = _cached_confound
        diagnostics.append(
            HarnessDiagnostic(
                name="confound_cache_hit",
                severity="info",
                message="Confound scores loaded from disk cache.",
                values={"hit": True, "fingerprint": _confound_fp[:8]},
            )
        )
    else:
        confound_df = _compute_confound_scores_full_data(work_df)
        if not disable_cache:
            _harness_cache.put_confound_scores(_confound_fp, confound_df)
        diagnostics.append(
            HarnessDiagnostic(
                name="confound_cache_hit",
                severity="info",
                message="Confound scores computed and written to disk cache.",
                values={"hit": False, "fingerprint": _confound_fp[:8]},
            )
        )

    _full_work_fp = full_work_df_fingerprint(work_df, outcome_variable)
    _cached_full_work_df = None if disable_cache else _harness_cache.get_full_work_df(_full_work_fp)
    if _cached_full_work_df is not None:
        full_work_df = _cached_full_work_df
        _full_work_df_cache_hit = True
        diagnostics.append(
            HarnessDiagnostic(
                name="full_work_df_cache_hit",
                severity="info",
                message="full_work_df loaded from disk cache (post-confound-join, pre-sample).",
                values={"hit": True, "fingerprint": _full_work_fp[:8]},
            )
        )
    else:
        # Preserve the full frame for the leakage audit before sampling narrows it.
        # The audit re-derives scores from strictly-prior rows; on a sample it would
        # miss most prior history and produce false positives on high-volume users.
        _confound_joined = work_df.join(
            confound_df.select(
                [
                    "user_id",
                    "email_id",
                    "date_sent",
                    "user_prior_engagement_score",
                    "email_popularity_score",
                ]
            ),
            on=["user_id", "email_id", "date_sent"],
            how="left",
        )
        _null_user = _confound_joined["user_prior_engagement_score"].is_null().sum()
        _null_email = _confound_joined["email_popularity_score"].is_null().sum()
        if _null_user > 0 or _null_email > 0:
            raise ConfoundJoinError(
                f"Confound left-join left {_null_user} null user_prior_engagement_score "
                f"and {_null_email} null email_popularity_score rows. "
                "Some work_df (user_id, email_id, date_sent) keys have no match in confound_df."
            )
        full_work_df = _confound_joined.with_columns(
            pl.col("user_prior_engagement_score").fill_null(0.0),
            pl.col("email_popularity_score").fill_null(0.5),
        )
        if not disable_cache:
            _harness_cache.put_full_work_df(_full_work_fp, full_work_df)
        _full_work_df_cache_hit = False
        diagnostics.append(
            HarnessDiagnostic(
                name="full_work_df_cache_hit",
                severity="info",
                message="full_work_df built and written to disk cache.",
                values={"hit": False, "fingerprint": _full_work_fp[:8]},
            )
        )

    if sample_frac is not None and sample_seed is not None:
        rng_sample = np.random.default_rng(sample_seed)
        n_sample = max(1, int(len(work_df) * sample_frac))
        sample_indices = rng_sample.choice(len(work_df), size=n_sample, replace=False)
        sample_indices_sorted = np.sort(sample_indices)
        work_df = work_df[sample_indices_sorted]
    elif sample_frac is not None:
        raise ValueError("sample_frac requires sample_seed")

    work_df = work_df.join(
        full_work_df.select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "user_prior_engagement_score",
                "email_popularity_score",
            ]
        ),
        on=["user_id", "email_id", "date_sent"],
        how="left",
    ).with_columns(
        pl.col("user_prior_engagement_score").fill_null(0.0),
        pl.col("email_popularity_score").fill_null(0.5),
    )

    work_df = work_df.sort("date_sent")

    stage_timings.append(
        HarnessStageTiming(
            stage="data_conversion", seconds=time.perf_counter() - t0, owner="harness"
        )
    )

    # ------------------------------------------------------------------
    # Stage 3: split_generation — quarterly folds + diagnostic bins
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    work_df = _assign_tenure_bucket(work_df)

    folds = _quarterly_folds(work_df)
    if len(folds) == 0:
        raise ValueError("No valid quarterly walk-forward folds found. Check date range in data.")

    if scope == "comparison_fast" and len(folds) > 4:
        # T2c: 4-fold pilot tier — evenly spaced quarters (Q1/Q2/Q3/Q4 of temporal range).
        step = (len(folds) - 1) / 3
        indices = [round(i * step) for i in range(4)]
        folds = [folds[i] for i in indices]
    elif scope == "fast_iter" and len(folds) > 3:
        # T4.2: subsample to 3 folds (first/middle/last) to preserve temporal structure
        # while cutting ~60-70% of per-fold loop wall. Uses all folds when fewer than 3.
        mid = len(folds) // 2
        folds = [folds[0], folds[mid], folds[-1]]

    thin_train_threshold = 50_000
    for q_label, train_df, _ in folds:
        if len(train_df) < thin_train_threshold:
            diagnostics.append(
                HarnessDiagnostic(
                    name="thin_train_fold",
                    severity="warning",
                    message=f"Fold {q_label} train has {len(train_df)} rows < {thin_train_threshold} threshold.",
                    values={"fold": q_label, "train_rows": len(train_df)},
                )
            )

    stage_timings.append(
        HarnessStageTiming(
            stage="split_generation",
            seconds=time.perf_counter() - t0,
            owner="harness",
            calls=len(folds),
        )
    )

    # ------------------------------------------------------------------
    # Stages 4–7: fit / predict / residualize_scores / score per fold
    # ------------------------------------------------------------------

    # Leakage audit: run once in main process before parallel dispatch.
    # v3.12: skip audit on warm cache hit for non-champion scopes to reduce wall time.
    leak_diag_user: HarnessDiagnostic | None = None
    leak_diag_email: HarnessDiagnostic | None = None
    _run_audit = not (_skip_audit_on_cache_hit and _full_work_df_cache_hit)
    if _run_audit:
        try:
            leak_diag_user = _assert_no_future_leak(
                full_work_df,
                "user_prior_engagement_score",
                "date_sent",
                "user_id",
                audit_sample_size=_audit_sample_size,
            )
            leak_diag_email = _assert_no_future_leak(
                full_work_df,
                "email_popularity_score",
                "date_sent",
                "email_id",
                audit_sample_size=_audit_sample_size,
            )
        except LeakageError:
            raise

    # Pre-spawn per-fold seeds for determinism (#22 seed-pre-spawn pattern).
    rng_folds = np.random.default_rng(sample_seed if sample_seed is not None else 0)
    n_folds = len(folds)
    fold_seeds = rng_folds.integers(0, 2**32 - 1, size=n_folds)

    # CPU utilization measurement across the parallel fold batch.
    import threading

    cpu_util_samples: list[float] = []
    proc = psutil.Process()

    def _sample_cpu() -> None:
        for _ in range(6):
            time.sleep(1.0)
            try:
                cpu_util_samples.append(proc.cpu_percent(interval=None))
            except Exception:
                pass

    cpu_thread = threading.Thread(target=_sample_cpu, daemon=True)
    cpu_thread.start()

    t_fold_wall_start = time.perf_counter()

    _effective_fold_jobs = fold_n_jobs if fold_n_jobs != 1 else 1
    raw_fold_results: list[dict[str, Any]] = Parallel(
        n_jobs=_effective_fold_jobs, prefer="threads"
    )(
        delayed(_run_one_fold)(
            train_df,
            test_df,
            q_label,
            feature_cols,
            outcome_variable,
            ml_model_config,
            _inner_threads,
            _confound_overlap_threshold,
            int(fold_seeds[k]),
            bootstrap_n_jobs,
            fold_k=k,
            bootstrap_n_resamples=bootstrap_n_resamples,
            cache=_harness_cache if not disable_cache else None,
            scope=scope,
            report_old_rfm=_report_old_rfm,
        )
        for k, (q_label, train_df, test_df) in enumerate(folds)
    )  # type: ignore[assignment]

    t_fold_wall_total = time.perf_counter() - t_fold_wall_start
    cpu_thread.join(timeout=2.0)

    fold_results: list[dict[str, Any]] = raw_fold_results
    all_rows: list[pl.DataFrame] = [r["pred_rows"] for r in fold_results]
    all_train_scores: list[np.ndarray] = [r["s_train"] for r in fold_results]

    for r in fold_results:
        diagnostics.append(r["fold_overlap_diag"])

    stage_timings.append(
        HarnessStageTiming(stage="fit", seconds=t_fold_wall_total, owner="model", calls=n_folds)
    )
    stage_timings.append(
        HarnessStageTiming(stage="predict", seconds=0.0, owner="model", calls=n_folds)
    )
    stage_timings.append(
        HarnessStageTiming(stage="score", seconds=0.0, owner="harness", calls=n_folds)
    )

    _fold_resid_seconds = sum(r["_resid_seconds"] for r in fold_results)
    _fold_iso_seconds = sum(r["_iso_fold_seconds"] for r in fold_results)
    _fold_boot_seconds = sum(r["_boot_seconds"] for r in fold_results)
    stage_timings.append(
        HarnessStageTiming(
            stage="residualize_scores",
            seconds=_fold_resid_seconds,
            owner="harness",
            calls=n_folds,
        )
    )
    stage_timings.append(
        HarnessStageTiming(
            stage="cross_fit_isotonic",
            seconds=_fold_iso_seconds,
            owner="harness",
            calls=n_folds,
        )
    )
    stage_timings.append(
        HarnessStageTiming(
            stage="bootstrap_ci",
            seconds=_fold_boot_seconds,
            owner="harness",
            calls=n_folds,
        )
    )

    if leak_diag_user:
        diagnostics.append(leak_diag_user)
    if leak_diag_email:
        diagnostics.append(leak_diag_email)

    # ------------------------------------------------------------------
    # Stage 8: aggregate
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    fold_aucs = [r["auc_residualized"] for r in fold_results]
    fold_aucs_old = [r["auc_residualized_old"] for r in fold_results]
    fold_labels = [r["q_label"] for r in fold_results]
    primary_value = float(np.mean(fold_aucs))

    if ml_model_config.ml_model_type == ML_Model_Type.LOGISTIC_REGRESSION:
        _lr_coef_dicts = [r["lr_coefs"] for r in fold_results if r["lr_coefs"] is not None]
        _lr_intercepts = [r["lr_intercept"] for r in fold_results if r["lr_intercept"] is not None]
        _feature_names = list(_lr_coef_dicts[0].keys()) if _lr_coef_dicts else list(feature_cols)
        coefs_per_fold = np.array([[d[f] for f in _feature_names] for d in _lr_coef_dicts])
        intercepts = np.array(_lr_intercepts)
        _coef_entries = [
            {
                "feature": _feature_names[i],
                "value": float(coefs_per_fold[:, i].mean()),
                "ci_low": None,
                "ci_high": None,
                "is_zero": bool(abs(coefs_per_fold[:, i].mean()) < 1e-12),
                "rank_by_abs_value": None,
            }
            for i in range(len(_feature_names))
        ]
        _ranked = sorted(_coef_entries, key=lambda c: abs(c["value"]), reverse=True)
        for _rank, _coef in enumerate(_ranked, start=1):
            _coef["rank_by_abs_value"] = _rank
        model_interpretability: dict[str, Any] | None = {
            "model_type": "logistic_regression",
            "penalty": ml_model_config.args.get("penalty", "l2"),
            "C": ml_model_config.args.get("C", 1.0),
            "coefficients": _coef_entries,
            "intercept": float(intercepts.mean()),
            "intercept_ci_low": None,
            "intercept_ci_high": None,
            "n_features_nonzero": int((np.abs(coefs_per_fold.mean(axis=0)) > 1e-12).sum()),
            "sparsity": float((np.abs(coefs_per_fold.mean(axis=0)) <= 1e-12).mean()),
            "feature_means": dict(zip(_feature_names, fold_results[0]["lr_feature_means"] or [])),
            "feature_stds": dict(zip(_feature_names, fold_results[0]["lr_feature_stds"] or [])),
        }
    else:
        model_interpretability = None
    _old_rfm_fold_values = [v for v in fold_aucs_old if v is not None]
    old_primary_value: float | None = (
        float(np.mean(_old_rfm_fold_values)) if _old_rfm_fold_values else None
    )
    fold_std = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0

    k = len(fold_aucs)
    if k > 1:
        # All fold threads have finished; no over-subscription risk — use all cores.
        primary_ci_low, primary_ci_high = _pooled_user_block_bootstrap_mean_of_folds(
            fold_results,
            n_boot=bootstrap_n_resamples,
            seed=42,
            n_jobs=-1,
            bootstrap_n_jobs=-1,
        )
    else:
        primary_ci_low = fold_results[0]["ci_low"]
        primary_ci_high = fold_results[0]["ci_high"]

    # Aggregate test set for secondary metrics
    y_all = np.concatenate([r["y_test"] for r in fold_results])
    s_all = np.concatenate([r["s_test"] for r in fold_results])
    s_resid_all = np.concatenate([r["s_resid"] for r in fold_results])
    c_u_all = np.concatenate([r["c_u_test"] for r in fold_results])
    c_u_main_effect_all = np.concatenate([r["c_u_main_effect_test"] for r in fold_results])
    c_e_all = np.concatenate([r["c_e_test"] for r in fold_results])
    user_ids_all = np.concatenate([r["user_ids"] for r in fold_results])
    tenure_all = np.concatenate([r["test_df"]["tenure_bucket"].to_numpy() for r in fold_results])

    train_y_all = np.concatenate([r["train_df"]["actioned_24h"].to_numpy() for r in fold_results])
    s_train_all = np.concatenate(all_train_scores)

    # Secondary: raw AUC
    roc_auc_raw = float(roc_auc_score(y_all, s_all)) if len(np.unique(y_all)) >= 2 else 0.5

    # Secondary: confound-only AUC (RFM floor — Gemini's finding)
    confound_combined = c_u_all + c_e_all
    roc_auc_confound_only = (
        float(roc_auc_score(y_all, confound_combined)) if len(np.unique(y_all)) >= 2 else 0.5
    )

    # Secondary: single-confound residualized AUCs (old RFM user confound)
    # T3.2 / T4.5: skip supplementary residualizations unless report_supplementary is True.
    # fast_iter forces False; champion_candidate defaults True; comparison defaults False.
    _t_supp_resid = time.perf_counter()
    if report_supplementary:
        roc_auc_user_prior_only, _iso_supp_1 = _residualize_single_confound(
            y_all, s_all, c_u_all, seed=42
        )
        roc_auc_email_pop_only, _iso_supp_2 = _residualize_single_confound(
            y_all, s_all, c_e_all, seed=42
        )
        roc_auc_user_main_effect_only, _iso_supp_3 = _residualize_single_confound(
            y_all, s_all, c_u_main_effect_all, seed=42
        )
        _supp_iso_seconds = _iso_supp_1 + _iso_supp_2 + _iso_supp_3
    else:
        roc_auc_user_prior_only = None
        roc_auc_email_pop_only = None
        roc_auc_user_main_effect_only = None
        _supp_iso_seconds = 0.0
    _supp_resid_seconds = time.perf_counter() - _t_supp_resid

    # Accumulate supplementary residualization into the fold-level substage timings.
    for st in stage_timings:
        if st.stage == "residualize_scores":
            st.seconds += _supp_resid_seconds
        elif st.stage == "cross_fit_isotonic":
            st.seconds += _supp_iso_seconds

    # PR-AUC variant (uses new primary residualized scores)
    pr_auc_residualized = (
        float(average_precision_score(y_all, s_resid_all)) if len(np.unique(y_all)) >= 2 else 0.0
    )

    # Per-tenure bucket residualized AUCs (uses new primary residualized scores)
    tenure_aucs = _per_tenure_residualized_auc(
        y_all, s_resid_all, tenure_all, c_u_main_effect_all, c_e_all
    )

    # Non-streak users (Gemini — streak-bridge guard)
    all_test_dfs = pl.concat([r["test_df"] for r in fold_results])
    if "is_in_action_streak" in all_test_dfs.columns:
        non_streak_mask = (~all_test_dfs["is_in_action_streak"].fill_null(False)).to_numpy()
        if non_streak_mask.sum() >= 50 and len(np.unique(y_all[non_streak_mask])) >= 2:
            roc_auc_non_streak = float(
                roc_auc_score(y_all[non_streak_mask], s_resid_all[non_streak_mask])
            )
        else:
            roc_auc_non_streak = None
    else:
        roc_auc_non_streak = None

    # Cold-start users (Codex — new users with 0 lifetime_actions_prior)
    if "lifetime_actions_prior" in all_test_dfs.columns:
        cold_start_mask = (all_test_dfs["lifetime_actions_prior"].fill_null(0) == 0).to_numpy()
        if cold_start_mask.sum() >= 50 and len(np.unique(y_all[cold_start_mask])) >= 2:
            roc_auc_cold_start = float(
                roc_auc_score(y_all[cold_start_mask], s_resid_all[cold_start_mask])
            )
        else:
            roc_auc_cold_start = None
    else:
        roc_auc_cold_start = None

    # Calendar-month CV (Gemini, Concern 1)
    month_labels = (
        all_test_dfs.with_columns(
            (
                pl.col("date_sent").dt.year().cast(pl.Utf8)
                + "-"
                + pl.col("date_sent").dt.month().cast(pl.Utf8).str.zfill(2)
            ).alias("_month")
        )
        .get_column("_month")
        .to_numpy()
    )
    unique_months = np.unique(month_labels)
    monthly_aucs: list[float] = []
    secondary_monthly: list[HarnessMetric] = []
    for m in unique_months:
        mask = month_labels == m
        if mask.sum() < 20 or len(np.unique(y_all[mask])) < 2:
            continue
        try:
            mauc = float(roc_auc_score(y_all[mask], s_resid_all[mask]))
            monthly_aucs.append(mauc)
            secondary_monthly.append(
                HarnessMetric(
                    name=f"roc_auc_residualized_pair_calendar_month_{m}",
                    value=mauc,
                    direction="higher_is_better",
                )
            )
        except Exception:
            pass

    temporal_cv = (
        float(np.std(monthly_aucs) / np.mean(monthly_aucs))
        if monthly_aucs and np.mean(monthly_aucs) != 0
        else 0.0
    )

    # PSI: train vs test score distribution
    psi_train_test = _compute_psi(s_train_all, s_all)

    # PSI by tenure bucket — hoist reference histogram once for all 7 buckets
    _psi_ref_bins = np.linspace(0, 1, 11)
    _psi_ref_bins[0], _psi_ref_bins[-1] = -np.inf, np.inf
    _psi_ref_counts, _psi_ref_edges = np.histogram(s_all, bins=_psi_ref_bins)
    tenure_psi_vals: list[float] = []
    for label in _TENURE_LABELS:
        mask = tenure_all == label
        if mask.sum() >= 20:
            psi = _psi_against_ref(s_all[mask], _psi_ref_counts, _psi_ref_edges)
            tenure_psi_vals.append(psi)
    max_tenure_psi = float(max(tenure_psi_vals)) if tenure_psi_vals else 0.0

    # Cell-level class balance: per-(tenure_bucket, prior_action_bin) positive counts
    cell_balance_diag, excluded_cells = _compute_cell_class_balance_diagnostic(
        combined=all_test_dfs,
        y_all=y_all,
        tenure_all=tenure_all,
        min_positives_threshold=_class_balance_threshold,
    )

    # Per-stratum AUC CV — drop excluded (tenure, prior_action_bin) cells from CV
    stratum_auc_vals = [
        v
        for label, v in tenure_aucs.items()
        if v is not None and not any(label == t for t, _ in excluded_cells)
    ]
    stratum_cv = (
        float(np.std(stratum_auc_vals) / np.mean(stratum_auc_vals))
        if len(stratum_auc_vals) >= 2 and np.mean(stratum_auc_vals) != 0
        else 0.0
    )

    # Confound correlation (Gemini — collapse-into-one-dimension check)
    confound_r = float(np.corrcoef(c_u_all, c_e_all)[0, 1])

    # Residual collapse check
    residual_collapsed = primary_value < 0.505

    # Pooled positive rates (kept for to_result_row; pooled_positive_rate_* fields)
    positive_rate_test = float(np.mean(y_all))
    positive_rate_train = float(np.mean(train_y_all))

    # User counts
    n_test_users = len(np.unique(user_ids_all))
    n_test_emails = int(all_test_dfs["email_id"].n_unique())
    n_test_rows = len(y_all)

    # Confound overlap summary across folds
    fold_overlap_diags = [d for d in diagnostics if d.name.startswith("confound_overlap_")]
    _overlap_min_positives = [
        d.values.get("min_positives_per_cell", 0)
        for d in fold_overlap_diags
        if isinstance(d.values.get("min_positives_per_cell"), int)
    ]
    _overlap_min = min(_overlap_min_positives) if _overlap_min_positives else None
    _overlap_n_cells_below = sum(
        d.values.get("n_cells_below_threshold", 0)
        for d in fold_overlap_diags
        if isinstance(d.values.get("n_cells_below_threshold"), int)
    )
    _overlap_summary_sev: Literal["info", "warning"] = (
        "warning" if any(d.severity == "warning" for d in fold_overlap_diags) else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="confound_overlap_min_across_folds",
            severity=_overlap_summary_sev,
            message=(
                f"Confound overlap summary: min positives/cell across folds = {_overlap_min}, "
                f"total cells below threshold = {_overlap_n_cells_below}"
            ),
            values={
                "min_positives_per_cell": _overlap_min,
                "n_cells_below_threshold": _overlap_n_cells_below,
                "threshold": _confound_overlap_threshold,
            },
        )
    )

    # Assemble diagnostics
    psi_sev: Literal["info", "warning", "error"] = (
        "error" if psi_train_test > 0.25 else "warning" if psi_train_test > 0.10 else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="score_psi_train_test",
            severity=psi_sev,
            message=f"Score PSI between train and test: {psi_train_test:.4f}",
            values={"psi": psi_train_test},
        )
    )

    tenure_psi_sev: Literal["info", "warning", "error"] = (
        "warning" if max_tenure_psi > 0.15 else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="score_psi_by_tenure_bucket",
            severity=tenure_psi_sev,
            message=f"Max per-tenure PSI: {max_tenure_psi:.4f}",
            values={"max_psi": max_tenure_psi},
        )
    )

    stratum_sev: Literal["info", "warning", "error"] = "warning" if stratum_cv > 0.5 else "info"
    diagnostics.append(
        HarnessDiagnostic(
            name="per_stratum_auc_dispersion",
            severity=stratum_sev,
            message=f"CV of residualized AUC across tenure buckets: {stratum_cv:.4f}",
            values={"cv": stratum_cv},
        )
    )

    collapse_sev: Literal["info", "warning", "error"] = "warning" if residual_collapsed else "info"
    diagnostics.append(
        HarnessDiagnostic(
            name="residual_collapse_check",
            severity=collapse_sev,
            message=f"Residualized AUC {'collapsed' if residual_collapsed else 'above floor'}: {primary_value:.4f}",
            values={"collapsed": residual_collapsed, "value": primary_value},
        )
    )

    confound_corr_sev: Literal["info", "warning", "error"] = (
        "warning" if abs(confound_r) > 0.5 else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="confound_correlation_pearson",
            severity=confound_corr_sev,
            message=f"Pearson r between user_prior and email_popularity scores: {confound_r:.4f}",
            values={"pearson_r": confound_r},
        )
    )

    temporal_sev: Literal["info", "warning", "error"] = "warning" if temporal_cv > 0.30 else "info"
    diagnostics.append(
        HarnessDiagnostic(
            name="temporal_drift_check",
            severity=temporal_sev,
            message=f"Calendar-month AUC CV: {temporal_cv:.4f}",
            values={"cv": temporal_cv, "n_months": len(monthly_aucs)},
        )
    )

    diagnostics.append(cell_balance_diag)

    cpu_util = float(np.mean(cpu_util_samples)) if cpu_util_samples else None
    if cpu_util is not None and cpu_util < 50:
        diagnostics.append(
            HarnessDiagnostic(
                name="cpu_underutilization",
                severity="warning",
                message=f"CPU utilization {cpu_util:.1f}% below 50% threshold",
                values={"cpu_utilization_pct": cpu_util},
            )
        )

    # T3b: cvAUC point estimate — comparison_fast only.
    # primary_value == mean(fold_aucs) by construction; cv_auc_pe should match to 1e-9.
    _perf_metadata: dict[str, Any] = {"full_work_df_cache_hit": _full_work_df_cache_hit}
    if scope == "comparison_fast":
        cv_auc_pe = _cv_auc_point_estimate(fold_results)
        _perf_metadata["cv_auc_point_estimate"] = cv_auc_pe
        _perf_metadata["cv_auc_pe_vs_bootstrap_mean"] = float(primary_value - cv_auc_pe)

    stage_timings.append(
        HarnessStageTiming(stage="aggregate", seconds=time.perf_counter() - t0, owner="harness")
    )

    # ------------------------------------------------------------------
    # Stage 9: persist_predictions
    # ------------------------------------------------------------------
    t_persist = time.perf_counter()
    if predictions_dir is not None and all_rows:
        pred_df = pl.concat(all_rows).select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "score",
                "score_residualized",
                pl.col("actioned_24h").alias("label"),
                "row_loss",
                "fold",
            ]
        )
        pred_path = Path(predictions_dir) / "predictions.parquet"
        pred_df.write_parquet(pred_path)
        byte_size = pred_path.stat().st_size
        artifacts.append(
            HarnessArtifact(
                name="predictions",
                kind="predictions",
                uri=str(pred_path),
                byte_size=byte_size,
                description="Per-row predictions with score, residualized score, label, row_loss, and fold.",
            )
        )
    stage_timings.append(
        HarnessStageTiming(
            stage="persist_predictions",
            seconds=time.perf_counter() - t_persist,
            owner="harness",
        )
    )

    # ------------------------------------------------------------------
    # Build secondary metrics
    # ------------------------------------------------------------------
    secondary_metrics: list[HarnessMetric] = [
        HarnessMetric(name="roc_auc_raw_pair", value=roc_auc_raw, direction="higher_is_better"),
        HarnessMetric(
            name="roc_auc_confound_only_pair",
            value=roc_auc_confound_only,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
            value=primary_value,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="pr_auc_residualized_user_prior_x_email_popularity_pair",
            value=pr_auc_residualized,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="primary_metric_temporal_cv", value=temporal_cv, direction="lower_is_better"
        ),
        HarnessMetric(name="n_test_users", value=float(n_test_users), direction="higher_is_better"),
        HarnessMetric(
            name="n_test_emails", value=float(n_test_emails), direction="higher_is_better"
        ),
        HarnessMetric(name="n_test_rows", value=float(n_test_rows), direction="higher_is_better"),
        HarnessMetric(
            name="pooled_positive_rate_train",
            value=positive_rate_train,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="pooled_positive_rate_test", value=positive_rate_test, direction="higher_is_better"
        ),
    ]

    # T2a: old-RFM secondary is absent when report_old_rfm=False; emit only when computed.
    if old_primary_value is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_user_prior_x_email_popularity_pair",
                value=old_primary_value,
                direction="higher_is_better",
            )
        )

    # T4.5: supplementary single-confound metrics are None in fast_iter; emit only when computed.
    if roc_auc_user_prior_only is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_user_prior_pair",
                value=roc_auc_user_prior_only,
                direction="higher_is_better",
            )
        )
    if roc_auc_user_main_effect_only is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_user_main_effect_pair",
                value=roc_auc_user_main_effect_only,
                direction="higher_is_better",
            )
        )
    if roc_auc_email_pop_only is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_email_popularity_pair",
                value=roc_auc_email_pop_only,
                direction="higher_is_better",
            )
        )

    if roc_auc_non_streak is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_non_streak_users_pair",
                value=roc_auc_non_streak,
                direction="higher_is_better",
            )
        )

    if roc_auc_cold_start is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_cold_start_pair",
                value=roc_auc_cold_start,
                direction="higher_is_better",
            )
        )

    tenure_name_map = {
        "0-7d": "0_7d",
        "8-30d": "8_30d",
        "31-90d": "31_90d",
        "91-180d": "91_180d",
        "181-365d": "181_365d",
        "1-2yr": "1_2yr",
        "2yr+": "2yr_plus",
    }
    for label, val in tenure_aucs.items():
        if val is not None:
            slug = tenure_name_map.get(label, label.replace("-", "_").replace("+", "_plus"))
            secondary_metrics.append(
                HarnessMetric(
                    name=f"roc_auc_residualized_{slug}_pair",
                    value=val,
                    direction="higher_is_better",
                )
            )

    secondary_metrics.extend(secondary_monthly)

    by_fold_metrics = [
        HarnessMetric(
            name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
            value=r["auc_residualized"],
            split=r["q_label"],
            fold=i,
            direction="higher_is_better",
            ci_low=r["ci_low"],
            ci_high=r["ci_high"],
        )
        for i, r in enumerate(fold_results)
    ]

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    total_seconds = time.perf_counter() - wall_start
    n_rows_evaluated = len(work_df)

    harness_secs = sum(st.seconds for st in stage_timings if st.owner == "harness")
    model_secs = sum(st.seconds for st in stage_timings if st.owner == "model")

    bottleneck = max(stage_timings, key=lambda st: st.seconds).stage

    if predictions_dir is not None:
        _pred_path = Path(predictions_dir) / "predictions.parquet"
        _fingerprint = hashlib.sha256(
            compute_predictions_hash(_pred_path).encode()
            + str(round(primary_value, 10)).encode()
            + str(round(primary_value - 0.5, 10)).encode()
        ).hexdigest()[:16]
    else:
        _fingerprint = ""

    response = CivicShoutHarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
            primary_metric_value=primary_value,
            baseline_metric_value=0.5,
            lift_vs_baseline=primary_value - 0.5,
            result_notes=[
                f"Quarterly walk-forward: {len(fold_results)} valid folds over {fold_labels[0]}–{fold_labels[-1]}.",
                f"Fold AUCs: {[round(a, 4) for a in fold_aucs]}",
                f"Old RFM-floor secondary: {round(old_primary_value, 4) if old_primary_value is not None else 'n/a'} (user_prior_x_email_popularity).",
                f"User main effect computed per-fold via {'LogisticRegression' if ml_model_config.ml_model_type == ML_Model_Type.LOGISTIC_REGRESSION else 'LightGBM'} on train split (train/test separation guarantees no leak).",
            ],
        ),
        method=HarnessMethod(
            name="quarterly_walk_forward_residualized_auc",
            metric_direction="higher_is_better",
            n_splits=len(fold_results),
            split_strategy="quarterly_cumulative_train_expanding",
            outcome_transform=None,
        ),
        data=HarnessDataProfile(
            n_rows=n_rows_evaluated,
            n_features=len(feature_cols),
            feature_cols=feature_cols,
            outcome_variable=outcome_variable,
            train_rows=sum(r["n_train"] for r in fold_results),
            validation_rows=sum(r["n_test"] for r in fold_results),
            sample_frac=sample_frac,
            sample_seed=sample_seed,
            sample_strategy="random_row_sample" if sample_frac is not None else None,
            is_full_run=sample_frac is None or sample_frac >= 1.0,
            population_n_rows=population_n_rows if sample_frac is not None else None,
            column_mapping={"group": "user_id"},
            notes=["Confound scores computed on full data before sampling."],
        ),
        metrics=HarnessMetrics(
            primary=HarnessMetric(
                name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
                value=primary_value,
                direction="higher_is_better",
                std=fold_std,
                ci_low=primary_ci_low,
                ci_high=primary_ci_high,
            ),
            by_fold=by_fold_metrics,
            secondary=secondary_metrics,
            baseline=[
                HarnessMetric(
                    name="roc_auc_random", value=0.5, split="baseline", direction="higher_is_better"
                ),
            ],
        ),
        performance=HarnessPerformance(
            total_seconds=total_seconds,
            stage_timings=stage_timings,
            bottleneck_stage=bottleneck,
            rows_per_second=n_rows_evaluated / total_seconds if total_seconds > 0 else None,
            folds_per_second=len(fold_results) / total_seconds if total_seconds > 0 else None,
            harness_owned_seconds=harness_secs,
            model_owned_seconds=model_secs,
            peak_memory_mb=None,
            parallelism=HarnessParallelism(
                outer_n_jobs=_effective_fold_jobs,
                inner_threads=_inner_threads,
                backend="threading" if _effective_fold_jobs != 1 else "sequential",
                cpu_count_observed=cpu_count,
                cpu_utilization_pct=cpu_util,
            ),
            hardware="local cpu",
            sample_size=n_rows_evaluated,
            metadata=_perf_metadata,
        ),
        diagnostics=diagnostics,
        artifacts=artifacts,
        reproducibility=HarnessReproducibility(
            seed=sample_seed,
            ml_model_type=ml_model_config.ml_model_type.value,
            ml_model_args=ml_model_config.args,
            data_fingerprint=None,
        ),
        model_interpretability=model_interpretability,
        fingerprint=_fingerprint,
    )

    return response
