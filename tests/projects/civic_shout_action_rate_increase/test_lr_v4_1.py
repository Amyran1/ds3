from __future__ import annotations

from projects.civic_shout_action_rate_increase.harness import (
    _SKLEARN_HAS_NEWTON_CHOLESKY,
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)


def test_l2_resolves_to_newton_cholesky_when_available() -> None:
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-3, "random_state": 42},
    )
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg,
        scope="smoke",
    )
    used_solver = resp.reproducibility.ml_model_args.get("solver")
    expected = "newton-cholesky" if _SKLEARN_HAS_NEWTON_CHOLESKY else "lbfgs"
    assert used_solver == expected


def test_l1_resolves_to_liblinear() -> None:
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l1", "C": 0.1, "max_iter": 200, "tol": 1e-3, "random_state": 42},
    )
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg,
        scope="smoke",
    )
    assert resp.reproducibility.ml_model_args.get("solver") == "liblinear"


def test_elasticnet_resolves_to_saga() -> None:
    cfg = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={
            "penalty": "elasticnet",
            "C": 1.0,
            "max_iter": 200,
            "tol": 1e-3,
            "random_state": 42,
            "l1_ratio": 0.5,
        },
    )
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg,
        scope="smoke",
    )
    assert resp.reproducibility.ml_model_args.get("solver") == "saga"


def test_tol_1e3_drift_within_gate() -> None:
    cfg_loose = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-3, "random_state": 42},
    )
    cfg_tight = ML_Model_Config(
        ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
        args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-4, "random_state": 42},
    )
    df = make_synthetic_with_interaction(seed=42)
    r_loose = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg_loose,
        scope="smoke",
    )
    r_tight = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=cfg_tight,
        scope="smoke",
    )
    drift = abs(r_loose.summary.primary_metric_value - r_tight.summary.primary_metric_value)
    assert drift < 0.000675, f"tol=1e-3 drifts {drift:.6f} vs tol=1e-4"
