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

_LR_MODEL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
    args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-3, "random_state": 42},
)


def test_lr_runs_with_100_resamples() -> None:
    """LR pipeline accepts bootstrap_n_resamples=100 and produces sensible CI."""
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=100,
    )
    assert resp.summary.primary_metric_value > 0.50
    assert (
        resp.metrics.primary.ci_low
        < resp.summary.primary_metric_value
        < resp.metrics.primary.ci_high
    )


def test_lr_100_vs_500_ci_within_tolerance() -> None:
    """100-resample CI should be within ~2.5× the 500-resample CI half-width on toy data."""
    df = make_synthetic_with_interaction(seed=42)
    r_100 = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=100,
    )
    r_500 = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_MODEL_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=500,
    )
    hw_100 = (r_100.metrics.primary.ci_high - r_100.metrics.primary.ci_low) / 2
    hw_500 = (r_500.metrics.primary.ci_high - r_500.metrics.primary.ci_low) / 2
    assert abs(r_100.summary.primary_metric_value - r_500.summary.primary_metric_value) < 0.01
    assert hw_100 < hw_500 * 3.0  # expect ~sqrt(5)x = 2.24x
