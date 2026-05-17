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


def test_lr_skip_audit_default_true_via_cli() -> None:
    """Argparse default for --lr-skip-audit is True; --no-lr-skip-audit disables it."""
    import argparse
    import sys
    from unittest.mock import patch

    import_path = "projects.civic_shout_action_rate_increase.runs.run_01"
    import importlib

    run_01 = importlib.import_module(import_path)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr-skip-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="lr_skip_audit",
    )

    args_default = parser.parse_args([])
    assert args_default.lr_skip_audit is True

    args_disabled = parser.parse_args(["--no-lr-skip-audit"])
    assert args_disabled.lr_skip_audit is False


def test_lr_with_skip_audit_true_runs_clean() -> None:
    """LR with lr_skip_audit=True produces a valid response with no future_leak audit."""
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
        lr_skip_audit=True,
    )
    assert resp.summary.primary_metric_value > 0.50
    diag_names = {d.name for d in resp.diagnostics}
    assert "future_leak_audit" not in diag_names
    assert "future_leak_user_prior_engagement_score" not in diag_names
    assert "future_leak_email_popularity_score" not in diag_names
