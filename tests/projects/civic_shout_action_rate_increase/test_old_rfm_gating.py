from __future__ import annotations

import math

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)

_MODEL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
    args={"n_estimators": 30, "num_leaves": 15, "random_state": 42, "verbose": -1},
)

_OLD_RFM_METRIC = "roc_auc_residualized_user_prior_x_email_popularity_pair"


def _df():
    return make_synthetic_with_interaction(seed=42)


def test_report_old_rfm_false_omits_secondary():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, report_old_rfm=False)
    sec_names = {m.name for m in r.metrics.secondary}
    assert _OLD_RFM_METRIC not in sec_names, (
        f"Expected old-RFM metric absent when report_old_rfm=False, but found it"
    )


def test_report_old_rfm_true_includes_secondary():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, report_old_rfm=True)
    sec = {m.name: m.value for m in r.metrics.secondary}
    assert _OLD_RFM_METRIC in sec, f"Expected old-RFM metric present when report_old_rfm=True"
    val = sec[_OLD_RFM_METRIC]
    assert val is not None and math.isfinite(val), f"Expected finite value, got {val}"


def test_scope_comparison_defaults_old_rfm_off():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="comparison")
    sec_names = {m.name for m in r.metrics.secondary}
    assert _OLD_RFM_METRIC not in sec_names, (
        f"Expected old-RFM metric absent for scope='comparison', but found it"
    )


def test_scope_champion_candidate_defaults_old_rfm_on():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="champion_candidate")
    sec = {m.name: m.value for m in r.metrics.secondary}
    assert _OLD_RFM_METRIC in sec, f"Expected old-RFM metric present for scope='champion_candidate'"
    val = sec[_OLD_RFM_METRIC]
    assert val is not None and math.isfinite(val), f"Expected finite value, got {val}"
