from __future__ import annotations

from pathlib import Path

import polars as pl

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from projects.civic_shout_action_rate_increase.runs.run_01 import (
    _LR_DEFAULT_BOOTSTRAP_N_COMPARISON,
    _LR_DEFAULT_BOOTSTRAP_N_COMPARISON_FAST,
    _build_harness_kwargs,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)

_LR_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
    args={"penalty": "l2", "C": 1.0, "max_iter": 200, "tol": 1e-3, "random_state": 42},
)

_DUMMY_DF = pl.DataFrame({"user_id": [1]})
_DUMMY_DIR = Path("/tmp")


def test_lr_comparison_uses_b_30_default() -> None:
    """When model=lr and scope=comparison, bootstrap_n_resamples defaults to 30."""
    kwargs = _build_harness_kwargs(
        joined=_DUMMY_DF,
        artifacts_dir=_DUMMY_DIR,
        sample_frac=0.01,
        sample_seed=42,
        scope="comparison",
        bootstrap_n_resamples=None,
        model="lr",
    )
    assert kwargs["bootstrap_n_resamples"] == _LR_DEFAULT_BOOTSTRAP_N_COMPARISON
    assert kwargs["bootstrap_n_resamples"] == 30


def test_lr_comparison_fast_uses_b_20_default() -> None:
    """When model=lr and scope=comparison_fast, bootstrap_n_resamples defaults to 20."""
    kwargs = _build_harness_kwargs(
        joined=_DUMMY_DF,
        artifacts_dir=_DUMMY_DIR,
        sample_frac=0.01,
        sample_seed=42,
        scope="comparison_fast",
        bootstrap_n_resamples=None,
        model="lr",
    )
    assert kwargs["bootstrap_n_resamples"] == _LR_DEFAULT_BOOTSTRAP_N_COMPARISON_FAST
    assert kwargs["bootstrap_n_resamples"] == 20


def test_lr_with_b_30_still_runs_clean() -> None:
    """LR with B=30 produces a valid response above 0.50 primary metric."""
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=30,
    )
    assert resp.summary.primary_metric_value > 0.50
