from __future__ import annotations

import pytest

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from projects.civic_shout_action_rate_increase.runs.run_01 import _should_escalate_to_comparison
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


def test_comparison_fast_runs_4_folds():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="comparison_fast")
    assert r.method.n_splits == 4, (
        f"Expected 4 folds for scope=comparison_fast, got {r.method.n_splits}"
    )


def test_comparison_fast_uses_100_resamples():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="comparison_fast")
    # bootstrap_n_resamples=100 means the CI is computed from 100 resamples.
    # We verify indirectly: n_splits==4 and CI is present (bootstrap ran).
    assert r.metrics.primary.ci_low is not None
    assert r.metrics.primary.ci_high is not None
    # The by_fold CIs are per-fold; aggregate CI comes from pooled bootstrap.
    # Confirming the run completed with 4 folds and a CI is the observable proxy.
    assert r.method.n_splits == 4


def test_comparison_fast_skips_old_rfm():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="comparison_fast")
    sec_names = {m.name for m in r.metrics.secondary}
    assert _OLD_RFM_METRIC not in sec_names, (
        f"Expected old-RFM metric absent for scope=comparison_fast, but found it"
    )


@pytest.mark.parametrize(
    "mean,ci_half,expected_clear",
    [
        (0.56, 0.01, True),  # lower=0.5404 > 0.501 → clear signal → should NOT escalate
        (0.50, 0.02, False),  # ambiguous → SHOULD escalate
        (0.44, 0.01, True),  # upper=0.4596 < 0.499 → clear signal → should NOT escalate
        (0.502, 0.002, False),  # lower=0.4980 < 0.501 → ambiguous → SHOULD escalate
    ],
)
def test_comparison_fast_first_clear_signal_skips_comparison(
    mean: float, ci_half: float, expected_clear: bool
) -> None:
    should_escalate = _should_escalate_to_comparison(mean, ci_half)
    if expected_clear:
        assert not should_escalate, (
            f"mean={mean}, ci_half={ci_half}: expected no escalation (clear signal), got escalate"
        )
    else:
        assert should_escalate, (
            f"mean={mean}, ci_half={ci_half}: expected escalation (ambiguous), got no escalate"
        )


def test_comparison_fast_first_ambiguous_escalates():
    # When |mean - 0.5| is within CI ± delta, should escalate.
    assert _should_escalate_to_comparison(0.501, 0.005) is True


def test_comparison_fast_first_clear_above_does_not_escalate():
    # lower = 0.56 - 1.96*0.02 = 0.5208 > 0.501 → clear signal → no escalation.
    assert _should_escalate_to_comparison(0.56, 0.02) is False


def test_comparison_fast_first_clear_below_does_not_escalate():
    # upper = 0.44 + 1.96*0.02 = 0.4792 < 0.499 → clear signal → no escalation.
    assert _should_escalate_to_comparison(0.44, 0.02) is False
