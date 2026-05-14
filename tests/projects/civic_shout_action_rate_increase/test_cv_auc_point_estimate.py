from __future__ import annotations

import numpy as np
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    _cv_auc_point_estimate,
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


def _df():
    return make_synthetic_with_interaction(seed=42)


def test_cv_auc_pe_matches_bootstrap_mean_in_comparison_fast():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="comparison_fast")
    assert "cv_auc_point_estimate" in r.performance.metadata, (
        "cv_auc_point_estimate should be present in performance.metadata for comparison_fast"
    )
    cv_auc_pe = r.performance.metadata["cv_auc_point_estimate"]
    primary = r.summary.primary_metric_value
    assert abs(cv_auc_pe - primary) < 1e-6, (
        f"cv_auc_pe={cv_auc_pe:.10f} should match primary_metric_value={primary:.10f} "
        f"(diff={abs(cv_auc_pe - primary):.2e})"
    )


def test_cv_auc_pe_absent_on_comparison_scope():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="comparison")
    assert "cv_auc_point_estimate" not in r.performance.metadata, (
        "cv_auc_point_estimate should NOT be present for scope=comparison"
    )


def test_cv_auc_pe_absent_on_champion_candidate():
    r = harness(_df(), FEATURE_COLS, _MODEL_CONFIG, sample_seed=42, scope="champion_candidate")
    assert "cv_auc_point_estimate" not in r.performance.metadata, (
        "cv_auc_point_estimate should NOT be present for scope=champion_candidate"
    )


def test_cv_auc_pe_handles_degenerate_fold():
    y_all_pos = np.ones(20, dtype=np.float32)
    y_good = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
    s_good = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.45], dtype=np.float32)

    fold_results = [
        {"y_test": y_all_pos, "s_resid": np.random.default_rng(0).random(20).astype(np.float32)},
        {"y_test": y_good, "s_resid": s_good},
    ]
    result = _cv_auc_point_estimate(fold_results)
    assert not np.isnan(result), (
        "Should return a valid float when at least one fold is non-degenerate"
    )
    assert 0.0 <= result <= 1.0


def test_cv_auc_pe_unit_returns_nan_if_all_folds_degenerate():
    y_all_pos = np.ones(10, dtype=np.float32)
    y_all_neg = np.zeros(10, dtype=np.float32)

    fold_results = [
        {"y_test": y_all_pos, "s_resid": np.random.default_rng(1).random(10).astype(np.float32)},
        {"y_test": y_all_neg, "s_resid": np.random.default_rng(2).random(10).astype(np.float32)},
    ]
    result = _cv_auc_point_estimate(fold_results)
    assert np.isnan(result), f"Expected NaN when all folds are degenerate, got {result}"
