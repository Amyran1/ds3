from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.harness import (
    LeakageError,
    ML_Model_Config,
    ML_Model_Type,
    _assert_no_future_leak,
    harness,
)
from projects.civic_shout_action_rate_increase.harness_cache import HarnessCache
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_user_emails,
)


@pytest.fixture()
def df() -> pl.DataFrame:
    return make_synthetic_user_emails(seed=42)


@pytest.fixture()
def model_config() -> ML_Model_Config:
    return ML_Model_Config(
        ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
        args={
            "n_estimators": 10,
            "num_leaves": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    )


# ---------------------------------------------------------------------------
# T1: audit_sample_size default 200 for comparison scope
# ---------------------------------------------------------------------------


def test_audit_sample_size_default_200(
    df: pl.DataFrame,
    model_config: ML_Model_Config,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """comparison scope: audit diagnostic must report n_audited <= 200."""
    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        result = harness(
            df,
            FEATURE_COLS,
            model_config,
            sample_frac=0.05,
            sample_seed=42,
            scope="comparison",
        )

    audit_diags = [d for d in result.diagnostics if d.name == "future_leak_audit"]
    assert len(audit_diags) == 2, f"Expected 2 audit diagnostics, got {len(audit_diags)}"
    for diag in audit_diags:
        n_audited = diag.values["n_audited"]
        assert n_audited <= 200, f"Expected n_audited <= 200 for comparison scope, got {n_audited}"


# ---------------------------------------------------------------------------
# T2: audit_sample_size 1000 for champion_candidate scope
# ---------------------------------------------------------------------------


def test_audit_sample_size_1000_on_champion(
    df: pl.DataFrame,
    model_config: ML_Model_Config,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """champion_candidate scope: audit diagnostic must report n_audited == 1000 (or len(df))."""
    with patch(
        "projects.civic_shout_action_rate_increase.harness.HarnessCache",
        lambda: HarnessCache(project_root=tmp_path),
    ):
        result = harness(
            df,
            FEATURE_COLS,
            model_config,
            sample_frac=0.05,
            sample_seed=42,
            scope="champion_candidate",
        )

    audit_diags = [d for d in result.diagnostics if d.name == "future_leak_audit"]
    assert len(audit_diags) == 2, f"Expected 2 audit diagnostics, got {len(audit_diags)}"
    for diag in audit_diags:
        n_audited = diag.values["n_audited"]
        assert n_audited > 200, (
            f"Expected n_audited > 200 for champion_candidate scope, got {n_audited}"
        )


# ---------------------------------------------------------------------------
# T3: 200-sample audit still catches a leak affecting 20% of rows
# ---------------------------------------------------------------------------


def test_audit_planted_leak_still_caught_at_200() -> None:
    """A leak affecting 20% of rows must be caught by the 200-sample audit.

    With n=200 and p=0.2, expected leaked rows = 40 — clear signal well above noise.
    """
    rng = np.random.default_rng(0)
    n_rows = 2000
    n_users = 200

    user_ids = rng.integers(0, n_users, size=n_rows).tolist()
    email_ids = rng.integers(0, 50, size=n_rows).tolist()
    dates = sorted(rng.integers(1, 500, size=n_rows).tolist())
    actioned = rng.integers(0, 2, size=n_rows).tolist()

    # Compute correct user scores
    df_raw = pl.DataFrame(
        {
            "user_id": user_ids,
            "email_id": email_ids,
            "date_sent": dates,
            "actioned_24h": actioned,
        }
    )
    date_level = (
        df_raw.group_by(["user_id", "date_sent"])
        .agg(pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts"))
        .sort(["user_id", "date_sent"])
        .with_columns(
            pl.col("_acts").cum_sum().shift(1).over("user_id").fill_null(0).alias("_prior_acts")
        )
    )
    df_joined = df_raw.join(
        date_level.select(["user_id", "date_sent", "_prior_acts"]),
        on=["user_id", "date_sent"],
        how="left",
    )
    correct_user_scores = np.log1p(df_joined["_prior_acts"].to_numpy().astype(np.float64)).tolist()

    # Plant a leak: corrupt 20% of rows by adding 1.0 to the score
    corrupt_mask = rng.random(size=n_rows) < 0.20
    leaked_scores = [
        s + (1.0 if corrupt_mask[i] else 0.0) for i, s in enumerate(correct_user_scores)
    ]

    df = pl.DataFrame(
        {
            "user_id": pl.Series(user_ids, dtype=pl.Int32),
            "email_id": pl.Series(email_ids, dtype=pl.Int32),
            "date_sent": pl.Series(dates, dtype=pl.Int32),
            "actioned_24h": pl.Series(actioned, dtype=pl.Int32),
            "user_prior_engagement_score": pl.Series(leaked_scores, dtype=pl.Float64),
            "email_popularity_score": pl.Series([0.5] * n_rows, dtype=pl.Float64),
        }
    )

    with pytest.raises(LeakageError, match="Future-leak detected"):
        _assert_no_future_leak(
            df,
            "user_prior_engagement_score",
            "date_sent",
            "user_id",
            audit_sample_size=200,
            seed=42,
        )


# ---------------------------------------------------------------------------
# T4: skip audit on cache hit for comparison scope
# ---------------------------------------------------------------------------


def test_skip_audit_on_cache_hit_comparison(
    df: pl.DataFrame,
    model_config: ML_Model_Config,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """comparison scope: audit called on miss (2 calls), skipped on cache hit (0 calls)."""
    audit_target = "projects.civic_shout_action_rate_increase.harness._assert_no_future_leak"

    with (
        patch(
            "projects.civic_shout_action_rate_increase.harness.HarnessCache",
            lambda: HarnessCache(project_root=tmp_path),
        ),
        patch(audit_target) as mock_audit,
    ):
        mock_audit.return_value = None
        harness(
            df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42, scope="comparison"
        )
        calls_miss = mock_audit.call_count

    with (
        patch(
            "projects.civic_shout_action_rate_increase.harness.HarnessCache",
            lambda: HarnessCache(project_root=tmp_path),
        ),
        patch(audit_target) as mock_audit,
    ):
        mock_audit.return_value = None
        harness(
            df, FEATURE_COLS, model_config, sample_frac=0.05, sample_seed=42, scope="comparison"
        )
        calls_hit = mock_audit.call_count

    assert calls_miss == 2, f"Expected 2 audit calls on miss, got {calls_miss}"
    assert calls_hit == 0, f"Expected 0 audit calls on cache hit (comparison), got {calls_hit}"


# ---------------------------------------------------------------------------
# T5: audit NOT skipped on cache hit for champion_candidate
# ---------------------------------------------------------------------------


def test_skip_audit_on_cache_hit_off_for_champion(
    df: pl.DataFrame,
    model_config: ML_Model_Config,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """champion_candidate scope: audit called on both miss and cache hit (2 calls each)."""
    audit_target = "projects.civic_shout_action_rate_increase.harness._assert_no_future_leak"

    with (
        patch(
            "projects.civic_shout_action_rate_increase.harness.HarnessCache",
            lambda: HarnessCache(project_root=tmp_path),
        ),
        patch(audit_target) as mock_audit,
    ):
        mock_audit.return_value = None
        harness(
            df,
            FEATURE_COLS,
            model_config,
            sample_frac=0.05,
            sample_seed=42,
            scope="champion_candidate",
        )
        calls_miss = mock_audit.call_count

    with (
        patch(
            "projects.civic_shout_action_rate_increase.harness.HarnessCache",
            lambda: HarnessCache(project_root=tmp_path),
        ),
        patch(audit_target) as mock_audit,
    ):
        mock_audit.return_value = None
        harness(
            df,
            FEATURE_COLS,
            model_config,
            sample_frac=0.05,
            sample_seed=42,
            scope="champion_candidate",
        )
        calls_hit = mock_audit.call_count

    assert calls_miss == 2, f"Expected 2 audit calls on miss (champion_candidate), got {calls_miss}"
    assert calls_hit == 2, (
        f"Expected 2 audit calls on cache hit (champion_candidate), got {calls_hit}"
    )
