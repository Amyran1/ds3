from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from projects.civic_shout_action_rate_increase.harness import (
    HarnessDataProfile,
    HarnessMethod,
    HarnessMetric,
    HarnessMetrics,
    HarnessPerformance,
    HarnessReproducibility,
    HarnessResponse,
    HarnessSummary,
    RunResultMetadata,
)
from projects.civic_shout_action_rate_increase.runs.manifest_helper import emit_manifest


def _make_metadata() -> RunResultMetadata:
    return RunResultMetadata(
        run_id="run_test",
        project="civic_shout_action_rate_increase",
        comparison_group="action_rate_increase_temporal_holdout_residualized_auc_v1",
        scope="smoke",
        git_sha="abc1234",
        recorded_at_utc=None,
        data_fingerprint=None,
    )


def _make_harness_response() -> HarnessResponse:
    return HarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc_residualized",
            primary_metric_value=0.65,
        ),
        method=HarnessMethod(
            name="quarterly_walk_forward",
            metric_direction="higher_is_better",
            split_strategy="quarterly_cumulative",
        ),
        data=HarnessDataProfile(
            n_rows=50000,
            n_features=8,
            feature_cols=["f1", "f2"],
            outcome_variable="actioned_24h",
            sample_frac=0.05,
            sample_seed=42,
            is_full_run=False,
        ),
        metrics=HarnessMetrics(
            primary=HarnessMetric(
                name="roc_auc_residualized",
                value=0.65,
                direction="higher_is_better",
            )
        ),
        performance=HarnessPerformance(
            total_seconds=12.3,
            stage_timings=[],
            bottleneck_stage="fit",
        ),
        reproducibility=HarnessReproducibility(
            ml_model_type="lightgbm_classifier",
            ml_model_args={"objective": "binary", "n_estimators": 200},
        ),
    )


def test_emit_and_read_roundtrip(tmp_path: Path) -> None:
    metadata = _make_metadata()
    harness_response = _make_harness_response()

    manifest_path = emit_manifest(metadata, harness_response, tmp_path)

    assert manifest_path.exists()
    doc = yaml.safe_load(manifest_path.read_text())

    assert doc["run_id"] == "run_test"
    assert doc["project"] == "civic_shout_action_rate_increase"
    assert doc["git_sha"] == "abc1234"
    assert doc["outcome"]["version"] == "v1"
    assert doc["data"]["source_cache"] == "civic_shout_user_emails"
    assert doc["data"]["source_version"] == 3
    assert doc["model"]["type"] == "lightgbm_classifier"
    assert doc["data"]["n_rows"] == 50000
    assert doc["data"]["fingerprint"] == "null"
    assert doc["sample"]["frac"] == 0.05
    assert doc["sample"]["seed"] == 42
    assert doc["performance"]["elapsed_seconds"] == pytest.approx(12.3)
    assert doc["performance"]["bottleneck_stage"] == "fit"


def test_emit_with_feature_dict_fixture(tmp_path: Path) -> None:
    metadata = _make_metadata()
    harness_response = _make_harness_response()

    dict_path = tmp_path / "dictionary.yaml"
    dict_path.write_text("feature_family: prior_action_recency\nversion: v1\n")

    manifest_path = emit_manifest(
        metadata,
        harness_response,
        tmp_path / "run_dir",
        feature_dictionaries={"prior_action_recency": dict_path},
    )

    doc = yaml.safe_load(manifest_path.read_text())
    feat = doc["features"]["prior_action_recency"]
    assert feat["version"] == "v1"
    assert feat["dict_fingerprint"].startswith("sha256:")
    assert len(feat["dict_fingerprint"]) == len("sha256:") + 16
