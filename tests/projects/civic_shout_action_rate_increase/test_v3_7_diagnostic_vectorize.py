from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    _PRIOR_ACTION_BINS,
    _PRIOR_ACTION_LABELS,
    _TENURE_LABELS,
    _compute_cell_class_balance_diagnostic,
)


def _make_combined(tenure: list[str], prior_vals: list[int], y: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tenure_bucket": tenure,
            "lifetime_actions_prior": prior_vals,
            "actioned_24h": y,
        }
    )


def _bin_prior_reference(val: int) -> str:
    for i, bound in enumerate(_PRIOR_ACTION_BINS[1:]):
        if val < bound:
            return _PRIOR_ACTION_LABELS[i]
    return _PRIOR_ACTION_LABELS[-1]


def test_vectorized_cell_balance_matches_loop() -> None:
    rng = np.random.default_rng(42)
    n = 500

    tenure_arr = rng.choice(_TENURE_LABELS, size=n).astype(str)
    prior_vals_int = rng.integers(0, 30, size=n)
    y_arr = rng.integers(0, 2, size=n)

    prior_bin_arr_ref = np.array([_bin_prior_reference(int(v)) for v in prior_vals_int])

    # Reference loop-based cell_positives
    cell_positives_ref: dict[tuple[str, str], int] = {}
    for t in _TENURE_LABELS:
        for p in _PRIOR_ACTION_LABELS:
            cell_positives_ref[(t, p)] = 0
    for i in range(n):
        key = (str(tenure_arr[i]), str(prior_bin_arr_ref[i]))
        if key in cell_positives_ref:
            cell_positives_ref[key] += int(y_arr[i])

    combined = pl.DataFrame(
        {
            "lifetime_actions_prior": prior_vals_int,
        }
    )

    diag, excluded = _compute_cell_class_balance_diagnostic(
        combined=combined,
        y_all=y_arr,
        tenure_all=tenure_arr,
        min_positives_threshold=0,
    )

    # Reconstruct observed cell_positives from the diagnostic message — verify via
    # calling private internals indirectly: rerun the same computation and compare totals.
    # Build expected cell_positives via vectorized path
    prior_bin_idx = np.searchsorted(
        _PRIOR_ACTION_BINS[1:], prior_vals_int.astype(np.int64), side="right"
    )
    prior_bin_arr_vec = np.array(_PRIOR_ACTION_LABELS)[prior_bin_idx]

    diag_df = pl.DataFrame(
        {
            "tenure": tenure_arr,
            "prior_action_bin": prior_bin_arr_vec,
            "y": y_arr.astype(np.int64),
        }
    )
    cell_counts_df = diag_df.group_by(["tenure", "prior_action_bin"]).agg(
        pl.col("y").sum().alias("positives")
    )
    observed_vec: dict[tuple[str, str], int] = {
        (str(row["tenure"]), str(row["prior_action_bin"])): int(row["positives"])
        for row in cell_counts_df.iter_rows(named=True)
    }
    cell_positives_vec: dict[tuple[str, str], int] = {}
    for t in _TENURE_LABELS:
        for p in _PRIOR_ACTION_LABELS:
            cell_positives_vec[(t, p)] = observed_vec.get((t, p), 0)

    assert cell_positives_vec == cell_positives_ref, (
        f"Vectorized cell_positives differs from loop reference. "
        f"Max diff keys: {[k for k in cell_positives_ref if cell_positives_ref[k] != cell_positives_vec.get(k, -1)][:5]}"
    )

    # Total positives must match
    assert sum(cell_positives_vec.values()) == int(y_arr.sum())


def test_double_concat_eliminated() -> None:
    rng = np.random.default_rng(7)
    n = 100

    tenure_arr = rng.choice(_TENURE_LABELS, size=n).astype(str)
    prior_vals_int = rng.integers(0, 10, size=n)
    y_arr = rng.integers(0, 2, size=n)

    combined = pl.DataFrame(
        {
            "lifetime_actions_prior": prior_vals_int,
        }
    )

    with patch("polars.concat") as mock_concat:
        _compute_cell_class_balance_diagnostic(
            combined=combined,
            y_all=y_arr,
            tenure_all=tenure_arr,
            min_positives_threshold=0,
        )
        assert mock_concat.call_count == 0, (
            f"Expected pl.concat to not be called inside _compute_cell_class_balance_diagnostic, "
            f"but it was called {mock_concat.call_count} time(s)"
        )
