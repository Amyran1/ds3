from __future__ import annotations

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)

_LR_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
    args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-3, "random_state": 42},
)


def test_lr_runs_with_50_resamples() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=50,
    )
    assert resp.summary.primary_metric_value > 0.50


def test_lr_skip_audit_kwarg_works() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=50,
        lr_skip_audit=True,
    )
    assert resp.summary.primary_metric_value > 0.50
    diag_names = {d.name for d in resp.diagnostics}
    assert "future_leak_user_prior_engagement_score" not in diag_names
    assert "future_leak_email_popularity_score" not in diag_names


def test_lr_fold_backend_processes_runs() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=50,
        lr_fold_backend="processes",
    )
    assert resp.summary.primary_metric_value > 0.50
