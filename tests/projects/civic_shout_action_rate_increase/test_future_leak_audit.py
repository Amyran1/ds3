from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    LeakageError,
    _assert_no_future_leak,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    user_ids: list[int],
    email_ids: list[int],
    dates: list[int],
    actioned: list[int],
    user_scores: list[float],
    email_scores: list[float],
) -> pl.DataFrame:
    """Build a minimal work_df compatible with the audit function."""
    return pl.DataFrame(
        {
            "user_id": user_ids,
            "email_id": email_ids,
            "date_sent": dates,
            "actioned_24h": actioned,
            "user_prior_engagement_score": user_scores,
            "email_popularity_score": email_scores,
        }
    )


def _per_row_user_score(df: pl.DataFrame, row: dict) -> float:
    """Reference implementation: per-row filter for user_prior_engagement_score."""
    prior = df.filter(
        (pl.col("user_id") == row["user_id"]) & (pl.col("date_sent") < row["date_sent"])
    )
    n_prior = int(prior["actioned_24h"].cast(pl.Int32).sum()) if len(prior) > 0 else 0
    return math.log1p(n_prior)


def _per_row_email_score(df: pl.DataFrame, row: dict) -> float:
    """Reference implementation: per-row filter for email_popularity_score."""
    prior = df.filter(
        (pl.col("email_id") == row["email_id"]) & (pl.col("date_sent") < row["date_sent"])
    )
    n_acts = int(prior["actioned_24h"].cast(pl.Int32).sum()) if len(prior) > 0 else 0
    n_sends = len(prior)
    return (n_acts + 1) / (n_sends + 2)


def _compute_correct_user_scores(df: pl.DataFrame) -> list[float]:
    scores = []
    for row in df.iter_rows(named=True):
        scores.append(_per_row_user_score(df, row))
    return scores


def _compute_correct_email_scores(df: pl.DataFrame) -> list[float]:
    scores = []
    for row in df.iter_rows(named=True):
        scores.append(_per_row_email_score(df, row))
    return scores


# ---------------------------------------------------------------------------
# T1: planted leak detection
# ---------------------------------------------------------------------------


def test_audit_catches_planted_leak() -> None:
    """A row with a wrong user_score (includes own day) should raise LeakageError."""
    # user=1 sends on days 1 and 2; day-1 actioned=0, day-2 actioned=1.
    # Correct day-2 user_prior_engagement_score = log1p(0) = 0.0 (nothing on day 1).
    # We plant score=log1p(1)=0.693 to simulate including own-day's action as a prior.
    df = _make_df(
        user_ids=[1, 1],
        email_ids=[10, 11],
        dates=[1, 2],
        actioned=[0, 1],
        user_scores=[0.0, math.log1p(1)],  # day-2 score is WRONG: correct is log1p(0)=0.0
        email_scores=[0.5, 0.5],
    )
    with pytest.raises(LeakageError, match="Future-leak detected in user_prior_engagement_score"):
        _assert_no_future_leak(
            df, "user_prior_engagement_score", "date_sent", "user_id", n_sample=100
        )


# ---------------------------------------------------------------------------
# T2: clean data passes silently
# ---------------------------------------------------------------------------


def test_audit_passes_on_clean_data() -> None:
    """Correctly computed scores must not raise."""
    df = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "email_id": [10, 11, 10, 12],
            "date_sent": [1, 2, 1, 3],
            "actioned_24h": [1, 0, 0, 1],
            "user_prior_engagement_score": [
                math.log1p(0),  # u1 d1: no prior
                math.log1p(1),  # u1 d2: 1 prior action
                math.log1p(0),  # u2 d1: no prior
                math.log1p(0),  # u2 d3: 0 prior actions (actioned_24h on d1 was 0)
            ],
            "email_popularity_score": [
                1 / 2,  # e10 d1: no prior → (0+1)/(0+2)
                1 / 2,  # e11 d2: no prior → (0+1)/(0+2)
                1 / 2,  # e10 d1: same date as row 0, shared date_level → (0+1)/(0+2)
                1 / 2,  # e12 d3: no prior for e12 → (0+1)/(0+2)
            ],
        }
    )
    result_user = _assert_no_future_leak(
        df, "user_prior_engagement_score", "date_sent", "user_id", n_sample=100
    )
    result_email = _assert_no_future_leak(
        df, "email_popularity_score", "date_sent", "email_id", n_sample=100
    )
    assert result_user.name == "future_leak_audit"
    assert result_email.name == "future_leak_audit"


# ---------------------------------------------------------------------------
# T3: parity with per-row reference across edge-case scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,user_ids,email_ids,dates,actioned",
    [
        (
            "single_user_single_email",
            [1, 1, 1],
            [10, 10, 10],
            [1, 2, 3],
            [0, 1, 1],
        ),
        (
            "ties_on_time",
            [1, 1, 1, 1],
            [10, 11, 10, 11],
            [1, 1, 2, 2],
            [1, 0, 0, 1],
        ),
        (
            "group_no_history",
            [1, 2],
            [10, 20],
            [5, 5],
            [1, 0],
        ),
        (
            "all_same_time",
            [1, 1, 1],
            [10, 11, 12],
            [7, 7, 7],
            [1, 1, 1],
        ),
        (
            "multi_group_mixed_dates",
            [1, 1, 2, 2, 3],
            [10, 10, 20, 20, 30],
            [1, 3, 2, 4, 1],
            [1, 0, 1, 1, 0],
        ),
    ],
)
def test_audit_parity_with_per_row_path(
    label: str,
    user_ids: list[int],
    email_ids: list[int],
    dates: list[int],
    actioned: list[int],
) -> None:
    """Vectorized path must produce expected scores identical to the per-row reference."""
    correct_user = _compute_correct_user_scores(
        pl.DataFrame(
            {
                "user_id": user_ids,
                "email_id": email_ids,
                "date_sent": dates,
                "actioned_24h": actioned,
            }
        )
    )
    correct_email = _compute_correct_email_scores(
        pl.DataFrame(
            {
                "user_id": user_ids,
                "email_id": email_ids,
                "date_sent": dates,
                "actioned_24h": actioned,
            }
        )
    )

    df = _make_df(
        user_ids=user_ids,
        email_ids=email_ids,
        dates=dates,
        actioned=actioned,
        user_scores=correct_user,
        email_scores=correct_email,
    )

    # Both audits must pass with zero diff — this exercises the vectorized path's
    # score derivation, not just whether it raises.
    result_user = _assert_no_future_leak(
        df, "user_prior_engagement_score", "date_sent", "user_id", n_sample=1000
    )
    result_email = _assert_no_future_leak(
        df, "email_popularity_score", "date_sent", "email_id", n_sample=1000
    )
    assert result_user.values["max_diff"] < 1e-9, (
        f"[{label}] user max_diff={result_user.values['max_diff']:.2e}"
    )
    assert result_email.values["max_diff"] < 1e-9, (
        f"[{label}] email max_diff={result_email.values['max_diff']:.2e}"
    )


# ---------------------------------------------------------------------------
# T4: null time_col raises
# ---------------------------------------------------------------------------


def test_audit_handles_null_time() -> None:
    """A null in date_sent must cause an early LeakageError before any scan."""
    df = pl.DataFrame(
        {
            "user_id": [1, 2],
            "email_id": [10, 11],
            "date_sent": pl.Series([1, None], dtype=pl.Int32),
            "actioned_24h": [0, 0],
            "user_prior_engagement_score": [0.0, 0.0],
            "email_popularity_score": [0.5, 0.5],
        }
    )
    with pytest.raises(LeakageError, match="null values found in"):
        _assert_no_future_leak(df, "user_prior_engagement_score", "date_sent", "user_id")


# ---------------------------------------------------------------------------
# T5: new group with no history → default uninformative scores
# ---------------------------------------------------------------------------


def test_audit_new_group_with_no_history() -> None:
    """A group's first appearance must yield user=0.0, email=0.5; audit passes."""
    df = _make_df(
        user_ids=[42],
        email_ids=[99],
        dates=[1],
        actioned=[0],
        user_scores=[math.log1p(0)],  # = 0.0
        email_scores=[1 / 2],  # = 0.5
    )
    result_user = _assert_no_future_leak(
        df, "user_prior_engagement_score", "date_sent", "user_id", n_sample=100
    )
    result_email = _assert_no_future_leak(
        df, "email_popularity_score", "date_sent", "email_id", n_sample=100
    )
    assert result_user.values["max_diff"] < 1e-9
    assert result_email.values["max_diff"] < 1e-9
