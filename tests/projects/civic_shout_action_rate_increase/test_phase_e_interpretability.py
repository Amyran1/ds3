from __future__ import annotations

import pytest

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)
from tests.projects.civic_shout_action_rate_increase.test_harness_toy_e2e import (
    _MODEL_CONFIG,
)

_LR_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
    args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-4, "random_state": 42},
)


def test_lr_coefficients_have_bootstrap_cis():
    df = make_synthetic_with_interaction(n_users=200, seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
    )
    mi = resp.model_interpretability
    assert mi is not None
    for coef in mi["coefficients"]:
        assert coef["ci_low"] is not None
        assert coef["ci_high"] is not None
        assert coef["ci_low"] <= coef["value"] <= coef["ci_high"], (
            f"Point estimate {coef['value']} not in CI [{coef['ci_low']}, {coef['ci_high']}]"
        )


def test_lr_ci_width_is_reasonable():
    df = make_synthetic_with_interaction(n_users=300, seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
    )
    mi = resp.model_interpretability
    assert mi is not None
    for coef in mi["coefficients"]:
        width = coef["ci_high"] - coef["ci_low"]
        assert width > 0
        assert width < 10.0


def test_lgbm_interpretability_populated():
    df = make_synthetic_with_interaction(n_users=300, seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_MODEL_CONFIG,
        scope="smoke",
    )
    mi = resp.model_interpretability
    assert mi is not None
    assert mi["model_type"] == "lightgbm_classifier"
    assert "feature_importance_gain" in mi
    assert len(mi["feature_importance_gain"]) == len(FEATURE_COLS)
    ranks = sorted([item["rank"] for item in mi["feature_importance_gain"]])
    assert ranks == list(range(1, len(ranks) + 1))


def test_lr_coefficients_rank_match_importance_directionality():
    df = make_synthetic_with_interaction(n_users=400, seed=42)

    resp_lr = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
    )
    resp_lgbm = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_MODEL_CONFIG,
        scope="smoke",
    )

    lr_top2 = sorted(
        resp_lr.model_interpretability["coefficients"],
        key=lambda c: abs(c["value"]),
        reverse=True,
    )[:2]
    lgbm_top2 = sorted(
        resp_lgbm.model_interpretability["feature_importance_gain"],
        key=lambda f: f["importance"],
        reverse=True,
    )[:2]

    lr_top2_features = {c["feature"] for c in lr_top2}
    lgbm_top2_features = {f["feature"] for f in lgbm_top2}
    overlap = lr_top2_features & lgbm_top2_features
    assert len(overlap) >= 1, (
        f"LR top-2 {lr_top2_features} and LGBM top-2 {lgbm_top2_features} should share >=1 feature"
    )
