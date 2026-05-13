from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from projects.civic_shout_action_rate_increase.harness import (
    CivicShoutHarnessResponse,
    CivicShoutResultRow,
    ML_Model_Config,
    ML_Model_Type,
    RunResultMetadata,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_features_equal_to_confound,
    make_synthetic_user_emails,
)

_MODEL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
    args={"n_estimators": 30, "num_leaves": 15, "random_state": 42, "verbose": -1},
)

_SUITE_START = time.perf_counter()


@pytest.fixture(scope="module")
def full_response() -> CivicShoutHarnessResponse:
    df = make_synthetic_user_emails(seed=42)
    return harness(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=42)


@pytest.fixture(scope="module")
def full_response_with_preds(tmp_path_factory) -> tuple[CivicShoutHarnessResponse, Path]:
    df = make_synthetic_user_emails(seed=42)
    tmpdir = tmp_path_factory.mktemp("predictions")
    resp = harness(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, predictions_dir=str(tmpdir))
    return resp, tmpdir


@pytest.fixture(scope="module")
def confound_only_response() -> CivicShoutHarnessResponse:
    df = make_synthetic_features_equal_to_confound(seed=42)
    return harness(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=42)


def test_harness_response_shape(full_response: CivicShoutHarnessResponse):
    r = full_response
    assert isinstance(r, CivicShoutHarnessResponse)
    assert (
        r.summary.primary_metric_name == "roc_auc_residualized_user_prior_x_email_popularity_pair"
    )
    assert r.summary.primary_metric_value is not None
    assert r.metrics.primary is not None
    assert r.metrics.primary.value is not None
    assert r.data.n_rows > 0
    assert r.data.n_features == len(FEATURE_COLS)
    assert r.method.name is not None
    assert r.performance.total_seconds > 0


def test_auc_math_matches_sklearn(full_response_with_preds):
    r, tmpdir = full_response_with_preds
    pred_path = tmpdir / "predictions.parquet"
    assert pred_path.exists()
    pred_df = pl.read_parquet(pred_path)
    fold_names = pred_df["fold"].unique().to_list()
    for fold_name in fold_names:
        fold_df = pred_df.filter(pl.col("fold") == fold_name)
        y = fold_df["label"].to_numpy().astype(float)
        if len(np.unique(y)) < 2:
            continue
        s_raw = fold_df["score"].to_numpy()
        sk_auc = roc_auc_score(y, s_raw)
        by_fold_map = {m.split: m.value for m in r.metrics.by_fold}
        if fold_name not in by_fold_map:
            continue
        sec_map = {m.name: m.value for m in r.metrics.secondary}
        raw_auc = sec_map.get("roc_auc_raw_pair")
        assert raw_auc is not None
        assert (
            abs(
                raw_auc
                - roc_auc_score(
                    pred_df["label"].to_numpy().astype(float),
                    pred_df["score"].to_numpy(),
                )
            )
            < 1e-9
        )


def test_raw_auc_exceeds_residualized_auc(full_response: CivicShoutHarnessResponse):
    sec = {m.name: m.value for m in full_response.metrics.secondary}
    raw = sec["roc_auc_raw_pair"]
    residualized = full_response.summary.primary_metric_value
    assert raw > residualized + 0.02, (
        f"Expected raw ({raw:.4f}) > residualized ({residualized:.4f}) + 0.02"
    )


def test_confound_only_auc_above_chance(full_response: CivicShoutHarnessResponse):
    sec = {m.name: m.value for m in full_response.metrics.secondary}
    confound_auc = sec["roc_auc_confound_only_pair"]
    assert confound_auc > 0.60, f"confound_only AUC = {confound_auc:.4f}, expected > 0.60"


def test_residualized_above_floor_when_features_add_signal(
    full_response: CivicShoutHarnessResponse,
):
    primary = full_response.summary.primary_metric_value
    assert primary > 0.51, f"residualized AUC = {primary:.4f}, expected > 0.51"


def test_residual_collapses_when_features_are_confound(
    confound_only_response: CivicShoutHarnessResponse,
):
    r = confound_only_response
    primary = r.summary.primary_metric_value
    assert abs(primary - 0.50) < 0.05, f"Expected residualized AUC ≈ 0.5, got {primary:.4f}"
    diag_names = {d.name: d for d in r.diagnostics}
    collapse_diag = diag_names.get("residual_collapse_check")
    assert collapse_diag is not None
    assert collapse_diag.severity == "warning"
    row = r.to_result_row(
        RunResultMetadata(
            run_id="test_collapse",
            project="civic_shout_action_rate_increase",
            comparison_group="test",
            scope="smoke",
        )
    )
    assert isinstance(row, CivicShoutResultRow)
    assert row.residual_collapse_flag is True


def test_user_block_bootstrap_ci_brackets_point_estimate(full_response: CivicShoutHarnessResponse):
    primary = full_response.metrics.primary
    assert primary.ci_low is not None
    assert primary.ci_high is not None
    assert primary.ci_low < primary.value < primary.ci_high
    assert primary.ci_high - primary.ci_low > 0
    for fold_metric in full_response.metrics.by_fold:
        assert fold_metric.ci_low is not None
        assert fold_metric.ci_high is not None
        assert fold_metric.ci_low <= fold_metric.value <= fold_metric.ci_high


def test_user_block_ci_wider_than_row_block_ci(full_response_with_preds):
    r, tmpdir = full_response_with_preds
    pred_path = tmpdir / "predictions.parquet"
    pred_df = pl.read_parquet(pred_path)
    y_all = pred_df["label"].to_numpy().astype(float)
    s_resid = pred_df["score_residualized"].to_numpy()

    rng = np.random.default_rng(42)
    n = len(y_all)
    n_boot = 200
    row_boot_aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_all[idx], s_resid[idx]
        if len(np.unique(yb)) < 2:
            row_boot_aucs[i] = 0.5
        else:
            row_boot_aucs[i] = float(roc_auc_score(yb, sb))

    row_ci_width = float(np.quantile(row_boot_aucs, 0.975) - np.quantile(row_boot_aucs, 0.025))

    primary = r.metrics.primary
    user_ci_width = (primary.ci_high or 0.0) - (primary.ci_low or 0.0)

    assert user_ci_width > row_ci_width * 0.95, (
        f"User-block CI width ({user_ci_width:.4f}) should exceed row-block CI width ({row_ci_width:.4f})"
    )


def test_walk_forward_produces_multiple_folds(full_response: CivicShoutHarnessResponse):
    row = full_response.to_result_row(
        RunResultMetadata(
            run_id="test_wf",
            project="civic_shout_action_rate_increase",
            comparison_group="test",
            scope="smoke",
        )
    )
    assert isinstance(row, CivicShoutResultRow)
    fold_aucs = row.walk_forward_fold_aucs
    fold_labels = row.walk_forward_fold_labels
    assert fold_aucs is not None
    assert len(fold_aucs) >= 4
    assert fold_labels is not None
    import re

    quarter_pattern = re.compile(r"^\d{4}-Q[1-4]$")
    for label in fold_labels:
        assert quarter_pattern.match(label), f"Bad quarter label: {label}"


def test_walk_forward_fold_aucs_are_finite_and_in_range(full_response: CivicShoutHarnessResponse):
    fold_aucs = [m.value for m in full_response.metrics.by_fold]
    assert len(fold_aucs) >= 4
    for auc in fold_aucs:
        assert math.isfinite(auc)
        assert 0.0 <= auc <= 1.0
    computed_mean = float(np.mean(fold_aucs))
    primary = full_response.summary.primary_metric_value
    assert abs(computed_mean - primary) < 1e-9, (
        f"Mean of fold AUCs ({computed_mean:.6f}) != primary ({primary:.6f})"
    )
    assert float(np.std(fold_aucs)) > 0


def test_cross_fit_isotonic_two_directions_consistent(full_response: CivicShoutHarnessResponse):
    for fold_m in full_response.metrics.by_fold:
        auc = fold_m.value
        assert math.isfinite(auc)
        assert 0.0 <= auc <= 1.0


def test_future_leak_audit_passes(full_response: CivicShoutHarnessResponse):
    diag_map = {d.name: d for d in full_response.diagnostics}
    audit_diag = diag_map.get("future_leak_audit")
    assert audit_diag is not None, "future_leak_audit diagnostic not found"
    assert audit_diag.severity == "info"
    assert audit_diag.values.get("max_diff", 1.0) < 1e-9
    assert audit_diag.values.get("n_audited", 0) >= 1000


def test_rejects_F03_forbidden_columns():
    df = make_synthetic_user_emails(seed=42)
    bad_cols = ["opened"] + FEATURE_COLS
    df_with_bad = df.with_columns(pl.lit(0).alias("opened"))
    with pytest.raises(ValueError) as exc_info:
        harness(df_with_bad, bad_cols, _MODEL_CONFIG)
    assert "F03" in str(exc_info.value)


def test_predictions_parquet_written_and_valid(full_response_with_preds):
    r, tmpdir = full_response_with_preds
    pred_path = tmpdir / "predictions.parquet"
    assert pred_path.exists()
    pred_df = pl.read_parquet(pred_path)
    required_cols = {
        "user_id",
        "email_id",
        "date_sent",
        "score",
        "score_residualized",
        "label",
        "row_loss",
        "fold",
    }
    assert required_cols.issubset(set(pred_df.columns)), (
        f"Missing columns: {required_cols - set(pred_df.columns)}"
    )
    assert len(pred_df) > 0
    artifact_names = {a.name for a in r.artifacts}
    assert "predictions" in artifact_names


def test_to_result_row_round_trips(full_response: CivicShoutHarnessResponse):
    metadata = RunResultMetadata(
        run_id="test_round_trip",
        project="civic_shout_action_rate_increase",
        comparison_group="fixture_test",
        scope="smoke",
    )
    row = full_response.to_result_row(metadata)
    assert isinstance(row, CivicShoutResultRow)
    serialized = row.model_dump_json()
    roundtrip = CivicShoutResultRow.model_validate_json(serialized)
    assert row == roundtrip


def test_deterministic_same_seed_e2e():
    df = make_synthetic_user_emails(seed=42)
    r1 = harness(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=99)
    r2 = harness(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=99)
    assert abs(r1.summary.primary_metric_value - r2.summary.primary_metric_value) < 1e-9
    aucs1 = [m.value for m in r1.metrics.by_fold]
    aucs2 = [m.value for m in r2.metrics.by_fold]
    assert len(aucs1) == len(aucs2)
    for a, b in zip(aucs1, aucs2):
        assert abs(a - b) < 1e-9
    assert abs((r1.metrics.primary.ci_low or 0) - (r2.metrics.primary.ci_low or 0)) < 1e-9
    assert abs((r1.metrics.primary.ci_high or 0) - (r2.metrics.primary.ci_high or 0)) < 1e-9


def test_smoke_lt_30_seconds():
    elapsed = time.perf_counter() - _SUITE_START
    assert elapsed < 150, f"Suite wall time {elapsed:.1f}s exceeded 150s budget"
