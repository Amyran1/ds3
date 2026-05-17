from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    _quarterly_folds,
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


# ---------------------------------------------------------------------------
# Cv5.5: fold equivalence
# ---------------------------------------------------------------------------


def _quarterly_folds_filter_reference(
    df: pl.DataFrame,
    date_col: str = "date_sent",
) -> list[tuple[str, pl.DataFrame, pl.DataFrame]]:
    """Original filter-based implementation preserved as a reference for equivalence tests."""
    quarters = df.select(
        (
            pl.col(date_col).dt.year().cast(pl.Utf8)
            + "-Q"
            + pl.col(date_col).dt.quarter().cast(pl.Utf8)
        )
    ).to_series()

    quarter_labels: list[str] = sorted(set(quarters.to_list()))

    folds = []
    for k, q_label in enumerate(quarter_labels):
        if k == 0:
            continue
        train_mask = quarters.is_in(quarter_labels[:k])
        test_mask = quarters == q_label
        train_df = df.filter(train_mask)
        test_df = df.filter(test_mask)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        folds.append((q_label, train_df, test_df))

    return folds


def test_fold_construction_equivalence_via_slice() -> None:
    """Slice-based _quarterly_folds produces identical row sets to the old filter-based version."""
    df = make_synthetic_with_interaction(seed=42)

    slice_folds = _quarterly_folds(df)
    filter_folds = _quarterly_folds_filter_reference(df)

    assert len(slice_folds) == len(filter_folds), (
        f"fold count mismatch: slice={len(slice_folds)}, filter={len(filter_folds)}"
    )

    key_cols = ["user_id", "email_id", "date_sent"]

    for (slice_label, slice_train, slice_test), (filter_label, filter_train, filter_test) in zip(
        slice_folds, filter_folds
    ):
        assert slice_label == filter_label, (
            f"quarter label mismatch: {slice_label} vs {filter_label}"
        )

        for split_name, slice_df, filter_df in [
            ("train", slice_train, filter_train),
            ("test", slice_test, filter_test),
        ]:
            assert len(slice_df) == len(filter_df), (
                f"fold {slice_label} {split_name} row count mismatch: "
                f"slice={len(slice_df)}, filter={len(filter_df)}"
            )

            slice_keys = set(tuple(r) for r in slice_df.select(key_cols).iter_rows())
            filter_keys = set(tuple(r) for r in filter_df.select(key_cols).iter_rows())
            assert slice_keys == filter_keys, (
                f"fold {slice_label} {split_name}: row key sets differ "
                f"(only_in_slice={len(slice_keys - filter_keys)}, "
                f"only_in_filter={len(filter_keys - slice_keys)})"
            )


# ---------------------------------------------------------------------------
# Cv5.4: LR comparison_fast skips predictions by default
# ---------------------------------------------------------------------------


def test_lr_compfast_skips_predictions_by_default() -> None:
    """LR comparison_fast does not write predictions.parquet when write_predictions=False."""
    from projects.civic_shout_action_rate_increase.runs.run_01 import _build_harness_kwargs

    df = make_synthetic_with_interaction(seed=42)

    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)
        kwargs = _build_harness_kwargs(
            joined=df,
            artifacts_dir=artifacts_dir,
            sample_frac=None,
            sample_seed=42,
            scope="comparison_fast",
            bootstrap_n_resamples=10,
            model="lr",
            write_predictions=False,
        )
        assert kwargs["predictions_dir"] is None, (
            "LR comparison_fast should skip predictions by default"
        )


def test_lr_compfast_writes_predictions_when_opted_in() -> None:
    """LR comparison_fast writes predictions when --write-predictions is set."""
    from projects.civic_shout_action_rate_increase.runs.run_01 import _build_harness_kwargs

    df = make_synthetic_with_interaction(seed=42)

    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)
        kwargs = _build_harness_kwargs(
            joined=df,
            artifacts_dir=artifacts_dir,
            sample_frac=None,
            sample_seed=42,
            scope="comparison_fast",
            bootstrap_n_resamples=10,
            model="lr",
            write_predictions=True,
        )
        assert kwargs["predictions_dir"] == str(artifacts_dir), (
            "LR comparison_fast should write predictions when opted in"
        )


def test_lr_comparison_always_writes_predictions() -> None:
    """LR comparison scope always writes predictions (decision-bearing), even without flag."""
    from projects.civic_shout_action_rate_increase.runs.run_01 import _build_harness_kwargs

    df = make_synthetic_with_interaction(seed=42)

    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)
        kwargs = _build_harness_kwargs(
            joined=df,
            artifacts_dir=artifacts_dir,
            sample_frac=None,
            sample_seed=42,
            scope="comparison",
            bootstrap_n_resamples=10,
            model="lr",
            write_predictions=False,
        )
        assert kwargs["predictions_dir"] == str(artifacts_dir), (
            "LR comparison scope must always write predictions"
        )


# ---------------------------------------------------------------------------
# Cv5.6: bootstrap dtype hoist — structural smoke
# ---------------------------------------------------------------------------


def test_bootstrap_dtype_hoist_no_regression() -> None:
    """Bootstrap produces valid CI bounds after dtype hoist (smoke: no NaN, CI ordered)."""
    df = make_synthetic_with_interaction(seed=42)
    resp = harness(
        data=df,
        feature_cols=FEATURE_COLS,
        outcome_variable="actioned_24h",
        ml_model_config=_LR_CONFIG,
        scope="comparison_fast",
        bootstrap_n_resamples=20,
        lr_skip_audit=True,
        lr_skip_heavy_diagnostics=True,
    )
    ci_low = resp.metrics.primary.ci_low
    ci_high = resp.metrics.primary.ci_high
    assert ci_low is not None and ci_high is not None
    assert not np.isnan(ci_low) and not np.isnan(ci_high)
    assert ci_low <= ci_high
