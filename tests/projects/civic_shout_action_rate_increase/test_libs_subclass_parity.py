from __future__ import annotations

import libs.responses as libs_resp
from projects.civic_shout_action_rate_increase.harness import (
    CivicShoutHarnessResponse,
    CivicShoutResultRow,
    HarnessDataProfile,
    HarnessMethod,
    HarnessMetric,
    HarnessMetrics,
    HarnessPerformance,
    HarnessReproducibility,
    HarnessResponse,
    HarnessSummary,
    RunResultMetadata,
    RunResultRow,
)


def test_project_harness_response_is_libs_subclass():
    assert issubclass(HarnessResponse, libs_resp.HarnessResponse)
    assert issubclass(CivicShoutHarnessResponse, libs_resp.HarnessResponse)


def test_project_run_result_metadata_is_libs_subclass():
    assert issubclass(RunResultMetadata, libs_resp.RunResultMetadata)


def test_project_run_result_row_is_libs_subclass():
    assert issubclass(RunResultRow, libs_resp.RunResultRow)
    assert issubclass(CivicShoutResultRow, libs_resp.RunResultRow)


def test_civic_shout_instance_isinstance_libs_harness_response():
    instance = CivicShoutHarnessResponse.model_construct()
    assert isinstance(instance, libs_resp.HarnessResponse)


def test_existing_harness_response_fields_preserved():
    fields = CivicShoutHarnessResponse.model_fields
    assert "model_interpretability" in fields
    assert "summary" in fields
    assert "metrics" in fields
    assert "method" in fields
    assert "data" in fields
    assert "reproducibility" in fields
    assert "diagnostics" in fields


def test_to_result_row_populates_libs_base_fields():
    primary_metric = HarnessMetric(
        name="roc_auc",
        value=0.72,
        direction="higher_is_better",
        ci_low=0.71,
        ci_high=0.73,
    )
    response = HarnessResponse.model_construct(
        summary=HarnessSummary(
            primary_metric_name="roc_auc",
            primary_metric_value=0.72,
            baseline_metric_value=0.50,
            lift_vs_baseline=0.22,
        ),
        method=HarnessMethod(
            name="walk_forward_cv",
            metric_direction="higher_is_better",
            split_strategy="quarterly_cumulative_train",
        ),
        data=HarnessDataProfile(
            n_rows=5000,
            n_features=8,
            feature_cols=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
            outcome_variable="actioned_24h",
            is_full_run=False,
            sample_frac=0.05,
        ),
        metrics=HarnessMetrics(primary=primary_metric),
        performance=HarnessPerformance(
            total_seconds=12.5,
            stage_timings=[],
        ),
        reproducibility=HarnessReproducibility(
            ml_model_type="lightgbm_classifier",
            ml_model_args={"n_estimators": 100},
            data_fingerprint="abc123",
        ),
        fingerprint="fp-test-abc",
        diagnostics=[],
        artifacts=[],
        model_interpretability=None,
        data_profile=None,
    )
    metadata = RunResultMetadata.model_construct(
        run_id="run_test",
        project="civic_shout_action_rate_increase",
        comparison_group="baseline",
        scope="smoke",
        status="completed",
        verdict="smoke_only",
        recorded_at_utc="2026-01-01T00:00:00Z",
        git_sha="abc1234",
        outcome_version="v3",
        notes="test run",
        data_fingerprint="",
    )

    row = response.to_result_row(metadata)

    assert row.n_features == 8
    assert row.total_seconds == 12.5
    assert row.fingerprint == "fp-test-abc"
    assert row.outcome_version == "v3"
    assert row.notes == "test run"
    assert row.metric_direction == "higher_is_better"
    assert row.sample_frac == 0.05

    row.model_dump_json()
