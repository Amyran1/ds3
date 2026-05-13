from __future__ import annotations

import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import psutil
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict, Field
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LeakageError(Exception):
    pass


# ---------------------------------------------------------------------------
# Stratification constants
# ---------------------------------------------------------------------------

_TENURE_BREAKS = [7, 30, 90, 180, 365, 730]
_TENURE_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr", "2yr+"]
_PRIOR_ACTION_BINS = [0, 1, 2, 5, 20, 1_000_000]
_PRIOR_ACTION_LABELS = ["0", "1", "2-4", "5-19", "20+"]

_FORBIDDEN_COLS = {"opened", "clicked", "verified_opened"}


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


class ML_Model_Config(BaseModel):
    ml_model_type: ML_Model_Type
    args: dict[str, Any]
    column_mapping: dict[str, str] = Field(default_factory=dict)


class RunResultMetadata(BaseModel):
    run_id: str
    project: str
    comparison_group: str
    scope: Literal["smoke", "reproduction", "comparison", "champion_candidate"]
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
            "walk_forward_fold_std": float(np.std(fold_aucs)) if fold_aucs else None,
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
        }

        for label, col in tenure_map.items():
            extras[col] = sec.get(f"roc_auc_residualized_{label}_pair")

        return CivicShoutResultRow(**base, **extras)


# ---------------------------------------------------------------------------
# Confound score helpers  (computed on FULL data, not the sample)
# ---------------------------------------------------------------------------


def _user_prior_engagement_score(df: pl.DataFrame) -> pl.Series:
    """log1p(cumulative prior actioned_24h per user, strict prior).

    Aggregates to (user_id, date_sent) level first so all rows sharing the
    same date receive the same score — matching the audit's strict date_sent < t
    semantics. Within-date row order is irrelevant to the score.
    """
    date_level = (
        df.group_by(["user_id", "date_sent"])
        .agg(pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"))
        .sort(["user_id", "date_sent"])
        .with_columns(
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("user_id")
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
    return (
        df.sort(["user_id", "email_id", "date_sent"])
        .join(date_level, on=["user_id", "date_sent"], how="left")
        .get_column("user_prior_engagement_score")
    )


def _email_popularity_score(df: pl.DataFrame) -> pl.Series:
    """Laplace-smoothed prior action rate per email, strict prior.

    score = (cumulative_prior_actions + 1) / (cumulative_prior_sends + 2)
    First send of any new email: score = 0.5 (uninformative prior).

    Aggregates to date level first so that all rows sharing the same (email_id,
    date_sent) receive the same score — matching the strict-prior semantics of
    the audit (prior = date_sent < t, not <=).
    """
    date_level = (
        df.group_by(["email_id", "date_sent"])
        .agg(
            pl.len().cast(pl.Int64).alias("_sends_on_date"),
            pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"),
        )
        .sort(["email_id", "date_sent"])
        .with_columns(
            pl.col("_sends_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id")
            .fill_null(0)
            .alias("_prior_sends"),
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id")
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
    return (
        df.sort(["user_id", "email_id", "date_sent"])
        .join(date_level, on=["email_id", "date_sent"], how="left")
        .get_column("email_popularity_score")
    )


def _compute_confound_scores_full_data(full_df: pl.DataFrame) -> pl.DataFrame:
    """Compute confound scores on the FULL frame; caller joins to sample.

    Returns df with columns (user_id, email_id, date_sent,
    user_prior_engagement_score, email_popularity_score).
    """
    sorted_df = full_df.sort(["user_id", "email_id", "date_sent"])

    user_date_level = (
        sorted_df.group_by(["user_id", "date_sent"])
        .agg(pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"))
        .sort(["user_id", "date_sent"])
        .with_columns(
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("user_id")
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

    user_scores = sorted_df.join(user_date_level, on=["user_id", "date_sent"], how="left").select(
        ["user_id", "email_id", "date_sent", "user_prior_engagement_score"]
    )

    email_date_level = (
        sorted_df.group_by(["email_id", "date_sent"])
        .agg(
            pl.len().cast(pl.Int64).alias("_sends_on_date"),
            pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"),
        )
        .sort(["email_id", "date_sent"])
        .with_columns(
            pl.col("_sends_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id")
            .fill_null(0)
            .alias("_prior_sends"),
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id")
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

    email_scores = sorted_df.join(
        email_date_level, on=["email_id", "date_sent"], how="left"
    ).select(["user_id", "email_id", "date_sent", "email_popularity_score"])

    return user_scores.join(email_scores, on=["user_id", "email_id", "date_sent"], how="inner")


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
) -> HarnessDiagnostic:
    """Re-derive score offline for a sample and confirm max |diff| < 1e-9."""
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    sample = df[sample_idx]

    max_diff = 0.0
    for row in sample.iter_rows(named=True):
        group_val = row[group_col]
        t = row[time_col]
        prior = df.filter((pl.col(group_col) == group_val) & (pl.col(time_col) < t))
        if score_col == "user_prior_engagement_score":
            n_prior_actions = prior["actioned_24h"].cast(pl.Int32).sum() if len(prior) > 0 else 0
            expected = float(np.log1p(n_prior_actions))
        elif score_col == "email_popularity_score":
            n_prior_acts = int(prior["actioned_24h"].cast(pl.Int32).sum()) if len(prior) > 0 else 0
            n_prior_sends = len(prior)
            expected = (n_prior_acts + 1) / (n_prior_sends + 2)
        else:
            continue

        actual = float(row[score_col])
        diff = abs(actual - expected)
        max_diff = max(max_diff, diff)

    if max_diff > 1e-9:
        raise LeakageError(
            f"Future-leak detected in {score_col}: max |diff| = {max_diff:.2e} > 1e-9"
        )

    return HarnessDiagnostic(
        name="future_leak_audit",
        severity="info",
        message=f"Strict-causality audit passed for {score_col}: n_audited={min(n_sample, len(df))}, max_diff={max_diff:.2e}",
        values={"n_audited": min(n_sample, len(df)), "max_diff": max_diff, "score_col": score_col},
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
) -> tuple[float, float, float]:
    """Two-way cross-fit sequential isotonic residualization.

    Returns (auc_residualized, auc_direction_a, auc_direction_b).
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    half_a_idx = rng.choice(n, size=n // 2, replace=False)
    half_b_idx = np.setdiff1d(np.arange(n), half_a_idx)

    y_a, s_a = y_test[half_a_idx], s_test[half_a_idx]
    c_u_a, c_e_a = c_u_test[half_a_idx], c_e_test[half_a_idx]

    y_b, s_b = y_test[half_b_idx], s_test[half_b_idx]
    c_u_b, c_e_b = c_u_test[half_b_idx], c_e_test[half_b_idx]

    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata

        return rankdata(arr, method="average") / len(arr)

    def _residualize_sequential(
        s_fit: np.ndarray,
        c_u_fit: np.ndarray,
        c_e_fit: np.ndarray,
        s_apply: np.ndarray,
        c_u_apply: np.ndarray,
        c_e_apply: np.ndarray,
    ) -> np.ndarray:
        r_u_fit = _rank_normalize(c_u_fit)
        r_e_fit = _rank_normalize(c_e_fit)
        r_u_apply = _rank_normalize(c_u_apply)
        r_e_apply = _rank_normalize(c_e_apply)

        iso_u = IsotonicRegression(out_of_bounds="clip")
        iso_u.fit(r_u_fit, s_fit)
        s_hat_u_fit = iso_u.predict(r_u_fit)
        s_hat_u_apply = iso_u.predict(r_u_apply)

        resid_fit = s_fit - s_hat_u_fit

        iso_e = IsotonicRegression(out_of_bounds="clip")
        iso_e.fit(r_e_fit, resid_fit)
        s_hat_e_apply = iso_e.predict(r_e_apply)

        return s_apply - s_hat_u_apply - s_hat_e_apply

    resid_b = _residualize_sequential(s_a, c_u_a, c_e_a, s_b, c_u_b, c_e_b)
    resid_a = _residualize_sequential(s_b, c_u_b, c_e_b, s_a, c_u_a, c_e_a)

    if len(np.unique(y_b)) < 2 or len(np.unique(y_a)) < 2:
        return 0.5, 0.5, 0.5

    auc_b = float(roc_auc_score(y_b, resid_b))
    auc_a = float(roc_auc_score(y_a, resid_a))

    n_a, n_b = len(half_a_idx), len(half_b_idx)
    auc_combined = (n_a * auc_a + n_b * auc_b) / (n_a + n_b)
    return auc_combined, auc_a, auc_b


# ---------------------------------------------------------------------------
# User-block bootstrap
# ---------------------------------------------------------------------------


def _user_block_bootstrap(
    y: np.ndarray,
    s_resid: np.ndarray,
    user_ids: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
    c_u: np.ndarray | None = None,
    c_e: np.ndarray | None = None,
) -> np.ndarray:
    """Sample whole users with replacement, recompute AUC per resample."""
    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)
    boot_aucs = np.empty(n_boot)

    for i in range(n_boot):
        sampled_users = rng.choice(unique_users, size=len(unique_users), replace=True)
        mask = np.isin(user_ids, sampled_users)
        if mask.sum() == 0:
            boot_aucs[i] = 0.5
            continue
        y_b, s_b = y[mask], s_resid[mask]
        if len(np.unique(y_b)) < 2:
            boot_aucs[i] = 0.5
            continue
        boot_aucs[i] = float(roc_auc_score(y_b, s_b))

    return boot_aucs


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
    df = df.with_columns(
        pl.col(date_col).dt.year().alias("_year"),
        pl.col(date_col).dt.quarter().alias("_quarter"),
    ).with_columns(
        (pl.col("_year").cast(pl.Utf8) + "-Q" + pl.col("_quarter").cast(pl.Utf8)).alias("_q_label")
    )

    quarter_labels: list[str] = (
        df.select("_q_label").unique().sort("_q_label").get_column("_q_label").to_list()
    )

    folds = []
    for k, q_label in enumerate(quarter_labels):
        if k == 0:
            continue
        train_quarters = quarter_labels[:k]
        train_df = df.filter(pl.col("_q_label").is_in(train_quarters)).drop(
            ["_year", "_quarter", "_q_label"]
        )
        test_df = df.filter(pl.col("_q_label") == q_label).drop(["_year", "_quarter", "_q_label"])
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
    bucket_expr = pl.lit("2yr+")
    for break_val, label in zip(reversed(_TENURE_BREAKS), reversed(_TENURE_LABELS[:-1])):
        bucket_expr = (
            pl.when(pl.col("_tenure_days") <= break_val).then(pl.lit(label)).otherwise(bucket_expr)
        )
    return df.with_columns(bucket_expr.alias("tenure_bucket")).drop(["first_send", "_tenure_days"])


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
) -> float:
    """Residualize s against a single confound c via isotonic regression."""
    rng = np.random.default_rng(seed)
    n = len(y)
    half_a_idx = rng.choice(n, size=n // 2, replace=False)
    half_b_idx = np.setdiff1d(np.arange(n), half_a_idx)

    from scipy.stats import rankdata

    def _do(fit_idx: np.ndarray, apply_idx: np.ndarray) -> float:
        r_fit = rankdata(c[fit_idx], method="average") / len(fit_idx)
        r_apply = rankdata(c[apply_idx], method="average") / len(apply_idx)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(r_fit, s[fit_idx])
        resid = s[apply_idx] - iso.predict(r_apply)
        if len(np.unique(y[apply_idx])) < 2:
            return 0.5
        return float(roc_auc_score(y[apply_idx], resid))

    auc_b = _do(half_a_idx, half_b_idx)
    auc_a = _do(half_b_idx, half_a_idx)
    return (auc_a * len(half_a_idx) + auc_b * len(half_b_idx)) / (len(half_a_idx) + len(half_b_idx))


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
) -> CivicShoutHarnessResponse:
    """Evaluate features against actioned_24h via quarterly walk-forward.

    Primary metric: roc_auc_residualized_user_prior_x_email_popularity_pair.
    Confound scores are computed on the FULL data before any sampling.
    """
    wall_start = time.perf_counter()
    stage_timings: list[HarnessStageTiming] = []
    diagnostics: list[HarnessDiagnostic] = []
    artifacts: list[HarnessArtifact] = []

    cpu_count = os.cpu_count() or 8
    inner_threads = max(cpu_count - 1, 1)

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

    confound_df = _compute_confound_scores_full_data(work_df)

    # Preserve the full frame for the leakage audit before sampling narrows it.
    # The audit re-derives scores from strictly-prior rows; on a sample it would
    # miss most prior history and produce false positives on high-volume users.
    full_work_df = work_df.join(
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
    ).with_columns(
        pl.col("user_prior_engagement_score").fill_null(0.0),
        pl.col("email_popularity_score").fill_null(0.5),
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
    lgbm_args = dict(ml_model_config.args)
    lgbm_args["n_jobs"] = inner_threads

    fold_results: list[dict[str, Any]] = []
    all_train_scores: list[np.ndarray] = []
    all_test_scores: list[np.ndarray] = []
    all_rows: list[pl.DataFrame] = []

    t_fit_total = 0.0
    t_predict_total = 0.0
    t_score_total = 0.0

    leak_diag_user: HarnessDiagnostic | None = None
    leak_diag_email: HarnessDiagnostic | None = None

    # CPU utilization measurement during fit of first fold
    cpu_util_samples: list[float] = []
    proc = psutil.Process()

    for fold_idx, (q_label, train_df, test_df) in enumerate(folds):
        X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
        y_train = train_df["actioned_24h"].to_numpy().astype(np.float32)
        X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
        y_test = test_df["actioned_24h"].to_numpy().astype(np.float32)

        c_u_test = test_df["user_prior_engagement_score"].to_numpy()
        c_e_test = test_df["email_popularity_score"].to_numpy()

        # Fit
        t_fit_start = time.perf_counter()
        model = LGBMClassifier(**lgbm_args)

        if fold_idx == 0:
            # Sample CPU utilization during first fold's fit
            import threading

            def _sample_cpu() -> None:
                for _ in range(3):
                    time.sleep(0.5)
                    try:
                        cpu_util_samples.append(proc.cpu_percent(interval=None))
                    except Exception:
                        pass

            cpu_thread = threading.Thread(target=_sample_cpu, daemon=True)
            cpu_thread.start()
            model.fit(X_train, y_train)
            cpu_thread.join(timeout=5.0)
        else:
            model.fit(X_train, y_train)

        t_fit_total += time.perf_counter() - t_fit_start

        # Predict
        t_pred_start = time.perf_counter()
        s_test = model.predict_proba(X_test)[:, 1]
        s_train = model.predict_proba(X_train)[:, 1]
        t_predict_total += time.perf_counter() - t_pred_start

        all_train_scores.append(s_train)
        all_test_scores.append(s_test)

        # Residualize + score
        t_score_start = time.perf_counter()

        if fold_idx == 0 and leak_diag_user is None:
            try:
                leak_diag_user = _assert_no_future_leak(
                    full_work_df, "user_prior_engagement_score", "date_sent", "user_id"
                )
                leak_diag_email = _assert_no_future_leak(
                    full_work_df, "email_popularity_score", "date_sent", "email_id"
                )
            except LeakageError:
                raise

        auc_residualized, auc_dir_a, auc_dir_b = _compute_residualized_auc_cross_fit(
            y_test, s_test, c_u_test, c_e_test, seed=42
        )

        # User block bootstrap for this fold
        user_ids_test = test_df["user_id"].to_numpy()

        # Compute residualized scores for the full test set (needed for bootstrap)
        from scipy.stats import rankdata as _rankdata

        def _full_resid(s: np.ndarray, c_u: np.ndarray, c_e: np.ndarray) -> np.ndarray:
            r_u = _rankdata(c_u, method="average") / len(c_u)
            r_e = _rankdata(c_e, method="average") / len(c_e)
            iso_u = IsotonicRegression(out_of_bounds="clip")
            iso_u.fit(r_u, s)
            s_hat_u = iso_u.predict(r_u)
            resid = s - s_hat_u
            iso_e = IsotonicRegression(out_of_bounds="clip")
            iso_e.fit(r_e, resid)
            s_hat_e = iso_e.predict(r_e)
            return s - s_hat_u - s_hat_e

        s_resid_full = _full_resid(s_test, c_u_test, c_e_test)

        boot_aucs = _user_block_bootstrap(y_test, s_resid_full, user_ids_test, n_boot=500, seed=42)
        ci_low = float(np.quantile(boot_aucs, 0.025))
        ci_high = float(np.quantile(boot_aucs, 0.975))

        t_score_total += time.perf_counter() - t_score_start

        fold_results.append(
            {
                "q_label": q_label,
                "auc_residualized": auc_residualized,
                "auc_dir_a": auc_dir_a,
                "auc_dir_b": auc_dir_b,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "y_test": y_test,
                "s_test": s_test,
                "s_resid": s_resid_full,
                "c_u_test": c_u_test,
                "c_e_test": c_e_test,
                "user_ids": user_ids_test,
                "test_df": test_df,
                "train_df": train_df,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )

        all_rows.append(
            test_df.with_columns(
                [
                    pl.Series("score", s_test),
                    pl.Series("score_residualized", s_resid_full),
                    pl.lit(q_label).alias("fold"),
                    pl.Series(
                        "row_loss",
                        (
                            -y_test * np.log(np.clip(s_test, 1e-7, 1 - 1e-7))
                            - (1 - y_test) * np.log(np.clip(1 - s_test, 1e-7, 1 - 1e-7))
                        ),
                    ),
                ]
            )
        )

    stage_timings.append(
        HarnessStageTiming(stage="fit", seconds=t_fit_total, owner="model", calls=len(folds))
    )
    stage_timings.append(
        HarnessStageTiming(
            stage="predict", seconds=t_predict_total, owner="model", calls=len(folds)
        )
    )
    stage_timings.append(
        HarnessStageTiming(stage="score", seconds=t_score_total, owner="harness", calls=len(folds))
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
    fold_labels = [r["q_label"] for r in fold_results]
    primary_value = float(np.mean(fold_aucs))
    fold_std = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else 0.0

    # t-based CI across folds
    k = len(fold_aucs)
    if k > 1:
        from scipy import stats as scipy_stats

        se = fold_std / np.sqrt(k)
        t_crit = float(scipy_stats.t.ppf(0.975, df=k - 1))
        primary_ci_low = primary_value - t_crit * se
        primary_ci_high = primary_value + t_crit * se
    else:
        primary_ci_low = fold_results[0]["ci_low"]
        primary_ci_high = fold_results[0]["ci_high"]

    # Aggregate test set for secondary metrics
    y_all = np.concatenate([r["y_test"] for r in fold_results])
    s_all = np.concatenate([r["s_test"] for r in fold_results])
    s_resid_all = np.concatenate([r["s_resid"] for r in fold_results])
    c_u_all = np.concatenate([r["c_u_test"] for r in fold_results])
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

    # Secondary: single-confound residualized AUCs
    roc_auc_user_prior_only = _residualize_single_confound(y_all, s_all, c_u_all, seed=42)
    roc_auc_email_pop_only = _residualize_single_confound(y_all, s_all, c_e_all, seed=42)

    # PR-AUC variant
    pr_auc_residualized = (
        float(average_precision_score(y_all, s_resid_all)) if len(np.unique(y_all)) >= 2 else 0.0
    )

    # Per-tenure bucket residualized AUCs
    tenure_aucs = _per_tenure_residualized_auc(y_all, s_resid_all, tenure_all, c_u_all, c_e_all)

    # Non-streak users (Gemini — streak-bridge guard)
    all_test_dfs = pl.concat([r["test_df"] for r in fold_results])
    if "is_in_action_streak" in all_test_dfs.columns:
        non_streak_mask = (all_test_dfs["is_in_action_streak"].fill_null(False) == False).to_numpy()
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
    all_test_dates = all_test_dfs["date_sent"]
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

    # PSI by tenure bucket
    tenure_psi_vals: list[float] = []
    for label in _TENURE_LABELS:
        mask = tenure_all == label
        if mask.sum() >= 20:
            # compare to overall test distribution
            psi = _compute_psi(s_all, s_all[mask])
            tenure_psi_vals.append(psi)
    max_tenure_psi = float(max(tenure_psi_vals)) if tenure_psi_vals else 0.0

    # Per-stratum AUC CV
    stratum_auc_vals = [v for v in tenure_aucs.values() if v is not None]
    stratum_cv = (
        float(np.std(stratum_auc_vals) / np.mean(stratum_auc_vals))
        if len(stratum_auc_vals) >= 2 and np.mean(stratum_auc_vals) != 0
        else 0.0
    )

    # Confound correlation (Gemini — collapse-into-one-dimension check)
    confound_r = float(np.corrcoef(c_u_all, c_e_all)[0, 1])

    # Residual collapse check
    residual_collapsed = primary_value < 0.505

    # Class balance
    positive_rate_test = float(np.mean(y_all))
    positive_rate_train = float(np.mean(train_y_all))

    # User counts
    n_test_users = len(np.unique(user_ids_all))
    n_test_emails = int(all_test_dfs["email_id"].n_unique())
    n_test_rows = len(y_all)

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

    diagnostics.append(
        HarnessDiagnostic(
            name="class_balance",
            severity="info",
            message=f"Positive rate — train: {positive_rate_train:.4f}, test: {positive_rate_test:.4f}",
            values={
                "positive_rate_train": positive_rate_train,
                "positive_rate_test": positive_rate_test,
            },
        )
    )

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
                description="Per-row predictions with score, residualized score, label, row_loss, and fold.",
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
            name="roc_auc_residualized_user_prior_pair",
            value=roc_auc_user_prior_only,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_email_popularity_pair",
            value=roc_auc_email_pop_only,
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
            name="roc_auc_residualized_user_prior_x_email_popularity_pair",
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

    response = CivicShoutHarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc_residualized_user_prior_x_email_popularity_pair",
            primary_metric_value=primary_value,
            baseline_metric_value=0.5,
            lift_vs_baseline=primary_value - 0.5,
            result_notes=[
                f"Quarterly walk-forward: {len(fold_results)} valid folds over {fold_labels[0]}–{fold_labels[-1]}.",
                f"Fold AUCs: {[round(a, 4) for a in fold_aucs]}",
                "Confound scores computed on FULL data per Codex recommendation.",
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
                name="roc_auc_residualized_user_prior_x_email_popularity_pair",
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
                outer_n_jobs=1,
                inner_threads=inner_threads,
                backend="sequential",
                cpu_count_observed=cpu_count,
                cpu_utilization_pct=cpu_util,
            ),
            hardware="local cpu",
            sample_size=n_rows_evaluated,
        ),
        diagnostics=diagnostics,
        artifacts=artifacts,
        reproducibility=HarnessReproducibility(
            seed=sample_seed,
            ml_model_type=ml_model_config.ml_model_type.value,
            ml_model_args=ml_model_config.args,
            data_fingerprint=None,
        ),
    )

    return response
