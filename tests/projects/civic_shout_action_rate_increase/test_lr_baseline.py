from __future__ import annotations

import numpy as np
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_features_equal_to_confound,
    make_synthetic_with_interaction,
)

_LR_MODEL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
    args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-4, "random_state": 42},
)


def test_lr_runs_end_to_end() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
    )
    assert resp.model_interpretability is not None
    assert resp.model_interpretability["model_type"] == "logistic_regression"
    assert len(resp.model_interpretability["coefficients"]) == len(FEATURE_COLS)


def test_lr_primary_above_floor_on_interaction() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
    )
    assert resp.summary.primary_metric_value > 0.50


def test_lr_residual_collapse_on_confound_only() -> None:
    df = make_synthetic_features_equal_to_confound(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
    )
    assert abs(resp.summary.primary_metric_value - 0.5) < 0.05


def test_lr_coefficient_ranking() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
    )
    assert resp.model_interpretability is not None
    coefs = resp.model_interpretability["coefficients"]
    ranks = [c["rank_by_abs_value"] for c in coefs]
    assert set(ranks) == set(range(1, len(coefs) + 1))


def test_lgbm_unchanged_by_lr_addition() -> None:
    from tests.projects.civic_shout_action_rate_increase.test_harness_toy_e2e import _MODEL_CONFIG

    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_MODEL_CONFIG,
        scope="smoke",
    )
    assert resp.model_interpretability is None
    assert resp.summary.primary_metric_value > 0.50


def test_lr_l1_penalty_works() -> None:
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l1", "C": 0.1, "max_iter": 200, "tol": 1e-4, "random_state": 42},
    )
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg,
        scope="smoke",
    )
    assert resp.model_interpretability is not None
    assert resp.model_interpretability["sparsity"] >= 0.0
