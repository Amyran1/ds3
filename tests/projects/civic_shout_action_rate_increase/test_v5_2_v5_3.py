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


def test_cpu_sampler_not_in_critical_path_for_lr_comparison() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="smoke",
        bootstrap_n_resamples=50,
        lr_skip_audit=True,
        lr_skip_heavy_diagnostics=False,
    )
    diag_names = {d.name for d in resp.diagnostics}
    assert "cpu_underutilization" not in diag_names


def test_heavy_diagnostics_skipped_on_lr_comparison() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="comparison",
        bootstrap_n_resamples=50,
        lr_skip_audit=True,
        lr_skip_heavy_diagnostics=True,
    )
    diag_names = {d.name for d in resp.diagnostics}
    assert "confound_overlap_min_across_folds" not in diag_names
    assert "score_psi_train_test" not in diag_names
    assert "score_psi_by_tenure_bucket" not in diag_names
    assert "class_balance" not in diag_names


def test_heavy_diagnostics_preserved_on_champion_candidate() -> None:
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="champion_candidate",
        bootstrap_n_resamples=50,
        lr_skip_audit=True,
        lr_skip_heavy_diagnostics=True,
    )
    diag_names = {d.name for d in resp.diagnostics}
    assert "confound_overlap_min_across_folds" in diag_names
    assert "score_psi_train_test" in diag_names
