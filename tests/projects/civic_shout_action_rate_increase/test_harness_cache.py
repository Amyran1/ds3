from __future__ import annotations

import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from projects.civic_shout_action_rate_increase.harness_cache import (
    HarnessCache,
    data_fingerprint,
    fold_train_fingerprint,
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
            "n_estimators": 20,
            "num_leaves": 8,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    )


def test_cache_hit_produces_same_primary_metric(
    df: pl.DataFrame, model_config: ML_Model_Config, tmp_path: pytest.TempPathFactory
) -> None:
    from unittest.mock import patch

    cache_root = tmp_path / "harness_cache"

    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        r1 = harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)
        r2 = harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)

    assert r1.summary.primary_metric_value == r2.summary.primary_metric_value


def test_cache_disable_recomputes(
    df: pl.DataFrame, model_config: ML_Model_Config, tmp_path: pytest.TempPathFactory
) -> None:
    from unittest.mock import patch

    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        harness(
            df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42, disable_cache=False
        )

    cache_dir = (
        tmp_path / "data" / "projects" / "civic_shout_action_rate_increase" / "harness_cache"
    )
    files_after_first = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
    assert len(files_after_first) > 0, "Cache should have written files on first run"

    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        harness(
            df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42, disable_cache=True
        )

    files_after_disable = list(cache_dir.glob("*.parquet"))
    assert len(files_after_disable) == len(files_after_first), (
        "disable_cache=True should not write additional files"
    )


def test_cache_invalidates_on_user_features_change(df: pl.DataFrame) -> None:
    subset = df.head(100)
    fp1 = fold_train_fingerprint(
        subset, fold_k=0, user_features=["actioned_last_1", "actioned_last_3"]
    )
    fp2 = fold_train_fingerprint(subset, fold_k=0, user_features=["actioned_last_1"])
    assert fp1 != fp2


def test_cache_invalidates_on_data_change(df: pl.DataFrame) -> None:
    fp1 = data_fingerprint(df.head(100))
    fp2 = data_fingerprint(df.head(200))
    assert fp1 != fp2


def test_diagnostic_logged_on_hit_and_miss(
    df: pl.DataFrame, model_config: ML_Model_Config, tmp_path: pytest.TempPathFactory
) -> None:
    from unittest.mock import patch

    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        r_miss = harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)

    confound_diags_miss = [d for d in r_miss.diagnostics if d.name == "confound_cache_hit"]
    assert len(confound_diags_miss) == 1
    assert confound_diags_miss[0].values["hit"] is False
    assert confound_diags_miss[0].severity == "info"

    ume_diags_miss = [
        d for d in r_miss.diagnostics if d.name.startswith("user_main_effect_cache_hit_fold_")
    ]
    assert len(ume_diags_miss) > 0
    for d in ume_diags_miss:
        assert d.values["hit"] is False
        assert d.severity == "info"

    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        r_hit = harness(df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42)

    confound_diags_hit = [d for d in r_hit.diagnostics if d.name == "confound_cache_hit"]
    assert len(confound_diags_hit) == 1
    assert confound_diags_hit[0].values["hit"] is True
    assert confound_diags_hit[0].severity == "info"

    ume_diags_hit = [
        d for d in r_hit.diagnostics if d.name.startswith("user_main_effect_cache_hit_fold_")
    ]
    assert len(ume_diags_hit) > 0
    for d in ume_diags_hit:
        assert d.values["hit"] is True
        assert d.severity == "info"
