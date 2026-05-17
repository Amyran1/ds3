from __future__ import annotations

import numpy as np
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    FitFoldResult,
    ML_Model_Config,
    ML_Model_Type,
    _fit_lgbm_fold,
    _fit_lr_fold,
)


def test_fit_lgbm_fold_returns_fit_fold_result() -> None:
    np.random.seed(42)
    X_train = np.random.randn(200, 6).astype(np.float64)
    y_train = (np.random.rand(200) < 0.1).astype(np.int32)
    X_test = np.random.randn(50, 6).astype(np.float64)
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
        args={"n_estimators": 10, "num_leaves": 5, "random_state": 42, "verbose": -1},
    )
    result = _fit_lgbm_fold(
        X_train,
        y_train,
        X_test,
        ["f0", "f1", "f2", "f3", "f4", "f5"],
        cfg,
        seed=42,
        inner_threads=1,
    )
    assert isinstance(result, FitFoldResult)
    assert result.raw_test.shape == (50,)
    assert result.proba_test.shape == (50,)
    assert result.lgbm_feature_importance is not None
    assert result.lr_coefs is None


def test_fit_lr_fold_returns_fit_fold_result() -> None:
    np.random.seed(42)
    X_train = np.random.randn(200, 6).astype(np.float64)
    y_train = (np.random.rand(200) < 0.1).astype(np.int32)
    X_test = np.random.randn(50, 6).astype(np.float64)
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l2", "C": 1.0, "random_state": 42},
    )
    result = _fit_lr_fold(
        X_train, y_train, X_test, ["f0", "f1", "f2", "f3", "f4", "f5"], cfg, seed=42
    )
    assert isinstance(result, FitFoldResult)
    assert result.raw_test.shape == (50,)
    assert result.lr_coefs is not None
    assert len(result.lr_coefs) == 6
    assert result.lgbm_feature_importance is None


def test_fit_lr_fold_with_user_ids_populates_ci() -> None:
    np.random.seed(42)
    n_train = 200
    X_train = np.random.randn(n_train, 6).astype(np.float64)
    y_train = (np.random.rand(n_train) < 0.1).astype(np.int32)
    X_test = np.random.randn(50, 6).astype(np.float64)
    user_ids = np.repeat(np.arange(50), 4)
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l2", "C": 1.0, "random_state": 42},
    )
    result = _fit_lr_fold(
        X_train,
        y_train,
        X_test,
        ["f0", "f1", "f2", "f3", "f4", "f5"],
        cfg,
        seed=42,
        user_ids_train=user_ids,
    )
    assert result.lr_coef_ci_low is not None
    assert result.lr_coef_ci_high is not None
    assert len(result.lr_coef_ci_low) == 6


def test_run_one_fold_lgbm_e2e_unchanged() -> None:
    from projects.civic_shout_action_rate_increase.harness import harness
    from tests.projects.civic_shout_action_rate_increase._fixtures import (
        FEATURE_COLS,
        make_synthetic_with_interaction,
    )

    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
        args={"n_estimators": 30, "num_leaves": 15, "random_state": 42, "verbose": -1},
    )
    df = make_synthetic_with_interaction(n_users=200, seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg,
        scope="smoke",
    )
    assert resp.summary.primary_metric_value > 0.50
    assert resp.model_interpretability is not None
    assert resp.model_interpretability["model_type"] == "lightgbm_classifier"
