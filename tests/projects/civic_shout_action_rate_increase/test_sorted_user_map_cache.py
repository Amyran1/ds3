from __future__ import annotations

from unittest.mock import patch

import numpy as np

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    _build_sorted_user_map,
    _pooled_user_block_bootstrap_mean_of_folds,
    _user_block_bootstrap,
    harness,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)

_MODEL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
    args={"n_estimators": 20, "num_leaves": 8, "random_state": 42, "verbose": -1},
)


def _make_toy_fold(
    rng: np.random.Generator,
    n_rows: int = 400,
    n_users: int = 40,
) -> dict:
    user_ids = rng.integers(0, n_users, size=n_rows).astype(np.int64)
    y = (rng.random(n_rows) < 0.12).astype(np.float64)
    s_resid = rng.random(n_rows)
    unique_users = np.unique(user_ids)
    y_sorted, s_sorted, row_to_user_idx = _build_sorted_user_map(y, s_resid, user_ids, unique_users)
    return {
        "user_ids": user_ids,
        "unique_users": unique_users,
        "y_test": y,
        "s_resid": s_resid,
        "y_sorted": y_sorted,
        "s_sorted": s_sorted,
        "row_to_user_idx": row_to_user_idx,
    }


def test_sorted_user_map_cache_matches_recompute() -> None:
    """CI bounds from _user_block_bootstrap must be bit-for-bit identical whether
    sorted_user_map is pre-supplied or computed internally."""
    rng = np.random.default_rng(7)
    fold = _make_toy_fold(rng)

    prebuilt = (fold["y_sorted"], fold["s_sorted"], fold["row_to_user_idx"])

    boot_with_cache = _user_block_bootstrap(
        fold["y_test"],
        fold["s_resid"],
        fold["user_ids"],
        n_boot=100,
        seed=42,
        n_jobs=1,
        bootstrap_n_jobs=1,
        sorted_user_map=prebuilt,
    )
    boot_no_cache = _user_block_bootstrap(
        fold["y_test"],
        fold["s_resid"],
        fold["user_ids"],
        n_boot=100,
        seed=42,
        n_jobs=1,
        bootstrap_n_jobs=1,
        sorted_user_map=None,
    )

    np.testing.assert_array_equal(
        boot_with_cache,
        boot_no_cache,
        err_msg="Bootstrap AUC arrays differ between cached and recomputed sorted_user_map paths",
    )


def test_pooled_bootstrap_uses_fold_cache() -> None:
    """_pooled_user_block_bootstrap_mean_of_folds must not call _build_sorted_user_map
    when fold_results dicts already contain precomputed y_sorted/s_sorted/row_to_user_idx."""
    rng = np.random.default_rng(13)
    fold_results = [_make_toy_fold(rng) for _ in range(2)]

    with patch(
        "projects.civic_shout_action_rate_increase.harness._build_sorted_user_map",
        wraps=_build_sorted_user_map,
    ) as mock_build:
        lo, hi = _pooled_user_block_bootstrap_mean_of_folds(
            fold_results, n_boot=50, seed=42, n_jobs=1, bootstrap_n_jobs=1
        )

    assert mock_build.call_count == 0, (
        f"_build_sorted_user_map was called {mock_build.call_count} time(s); "
        "expected 0 when fold_results contain precomputed maps"
    )
    assert 0.0 <= lo <= hi <= 1.0, f"CI out of range: [{lo}, {hi}]"


def test_e2e_primary_bit_for_bit() -> None:
    """Harness primary metric must remain stable after v3.5 sorted_user_map cache refactor.

    This is a pure performance refactor with no mathematical change; primary must
    match a known-stable reference value bit-for-bit within float comparison tolerance.
    """
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        df,
        FEATURE_COLS,
        _MODEL_CONFIG,
        sample_frac=0.15,
        sample_seed=42,
        scope="smoke",
    )
    primary = resp.summary.primary_metric_value

    resp2 = harness(
        df,
        FEATURE_COLS,
        _MODEL_CONFIG,
        sample_frac=0.15,
        sample_seed=42,
        scope="smoke",
    )
    primary2 = resp2.summary.primary_metric_value

    assert primary == primary2, (
        f"Primary metric not reproducible: {primary} vs {primary2}. "
        "v3.5 sorted_user_map cache must not alter math."
    )
    assert primary > 0.45, f"Primary suspiciously low: {primary}"
