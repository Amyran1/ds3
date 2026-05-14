from __future__ import annotations

import pytest

from projects.civic_shout_action_rate_increase.harness import (
    CivicShoutHarnessResponse,
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_user_emails,
)

_SMALL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
    args={"n_estimators": 20, "num_leaves": 8, "random_state": 42, "verbose": -1},
)


def _small_frame(seed: int = 42):
    df = make_synthetic_user_emails(seed=seed)
    return df.sample(n=1000, seed=seed, with_replacement=False)


def test_returns_civic_shout_response_on_synthetic_data():
    df = _small_frame()
    response = harness(df, FEATURE_COLS, _SMALL_CONFIG, sample_seed=42)
    assert isinstance(response, CivicShoutHarnessResponse)
    assert (
        response.summary.primary_metric_name
        == "roc_auc_residualized_user_main_effect_x_email_popularity_pair"
    )
    v = response.summary.primary_metric_value
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0
    assert response.metrics.primary is not None


def test_deterministic_same_seed():
    df = _small_frame(seed=42)
    r1 = harness(df, FEATURE_COLS, _SMALL_CONFIG, sample_seed=7)
    r2 = harness(df, FEATURE_COLS, _SMALL_CONFIG, sample_seed=7)
    assert abs(r1.summary.primary_metric_value - r2.summary.primary_metric_value) < 1e-9
    aucs1 = [m.value for m in r1.metrics.by_fold]
    aucs2 = [m.value for m in r2.metrics.by_fold]
    assert len(aucs1) == len(aucs2)
    for a, b in zip(aucs1, aucs2):
        assert abs(a - b) < 1e-9


def test_rejects_F03_forbidden_columns():
    df = _small_frame()
    bad_cols = ["opened"] + FEATURE_COLS
    with pytest.raises(ValueError) as exc_info:
        harness(
            df.with_columns(__import__("polars").lit(0).alias("opened")), bad_cols, _SMALL_CONFIG
        )
    assert "F03" in str(exc_info.value)
