from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

import projects.civic_shout_action_rate_increase.harness_cache as hc_module
from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from projects.civic_shout_action_rate_increase.harness_cache import (
    FULL_WORK_DF_CACHE_SCHEMA_VERSION,
    HarnessCache,
    full_work_df_fingerprint,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_user_emails,
)


@pytest.fixture()
def df() -> pl.DataFrame:
    return make_synthetic_user_emails(seed=42)


@pytest.fixture()
def model_config() -> ML_Model_Config:
    return ML_Model_Config(
        ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
        args={
            "n_estimators": 10,
            "num_leaves": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    )


def test_full_work_df_cache_round_trip(
    df: pl.DataFrame,
    model_config: ML_Model_Config,
    tmp_path: pytest.TempPathFactory,
) -> None:
    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        r1 = harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)

    diag_miss = [d for d in r1.diagnostics if d.name == "full_work_df_cache_hit"]
    assert len(diag_miss) == 1
    assert diag_miss[0].values["hit"] is False

    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        r2 = harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)

    diag_hit = [d for d in r2.diagnostics if d.name == "full_work_df_cache_hit"]
    assert len(diag_hit) == 1
    assert diag_hit[0].values["hit"] is True

    assert r1.summary.primary_metric_value == r2.summary.primary_metric_value
    assert r1.data.n_rows == r2.data.n_rows


def test_full_work_df_cache_invalidates_on_version_bump(
    df: pl.DataFrame,
    tmp_path: pytest.TempPathFactory,
) -> None:
    cache = HarnessCache(project_root=tmp_path)

    fp_v1 = full_work_df_fingerprint(df, "actioned_24h")
    cache.put_full_work_df(fp_v1, df.head(10))

    with patch.object(
        hc_module, "FULL_WORK_DF_CACHE_SCHEMA_VERSION", FULL_WORK_DF_CACHE_SCHEMA_VERSION + 1
    ):
        fp_v2 = full_work_df_fingerprint(df, "actioned_24h")

    assert fp_v1 != fp_v2
    assert cache.get_full_work_df(fp_v2) is None


def test_full_work_df_cache_invalidates_on_data_change(
    df: pl.DataFrame,
    tmp_path: pytest.TempPathFactory,
) -> None:
    cache = HarnessCache(project_root=tmp_path)

    df_a = df.head(100)
    df_b = df.head(101)

    fp_a = full_work_df_fingerprint(df_a, "actioned_24h")
    fp_b = full_work_df_fingerprint(df_b, "actioned_24h")

    assert fp_a != fp_b

    cache.put_full_work_df(fp_a, df_a.head(5))
    assert cache.get_full_work_df(fp_b) is None


def test_warm_cache_path_skips_audit(
    df: pl.DataFrame,
    model_config: ML_Model_Config,
    tmp_path: pytest.TempPathFactory,
) -> None:
    # The leakage audit runs on both cache miss and cache hit (full_work_df is loaded from cache
    # and still passed through the audit to preserve diagnostics). This test verifies that the
    # audit is called on both invocations: 2 calls per run (user + email score cols).
    audit_target = "projects.civic_shout_action_rate_increase.harness._assert_no_future_leak"

    with (
        patch(
            "projects.civic_shout_action_rate_increase.harness.HarnessCache",
            lambda: HarnessCache(project_root=tmp_path),
        ),
        patch(audit_target) as mock_audit,
    ):
        mock_audit.return_value = None
        harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)
        calls_after_miss = mock_audit.call_count

    with (
        patch(
            "projects.civic_shout_action_rate_increase.harness.HarnessCache",
            lambda: HarnessCache(project_root=tmp_path),
        ),
        patch(audit_target) as mock_audit,
    ):
        mock_audit.return_value = None
        harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)
        calls_after_hit = mock_audit.call_count

    assert calls_after_miss == 2, f"Expected 2 audit calls on miss, got {calls_after_miss}"
    assert calls_after_hit == 2, f"Expected 2 audit calls on cache hit, got {calls_after_hit}"
