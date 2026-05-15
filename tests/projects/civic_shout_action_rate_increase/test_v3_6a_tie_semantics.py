from __future__ import annotations

import datetime
import math

import numpy as np
import polars as pl

from projects.civic_shout_action_rate_increase.harness import (
    _assert_no_future_leak,
    _compute_confound_scores_full_data,
)


def _make_full_df_from_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("date_sent").cast(pl.Date),
        pl.col("actioned_24h").cast(pl.Int32),
    )


def test_strict_prior_under_same_day_ties() -> None:
    rows = [
        {"user_id": "A", "email_id": "e1", "date_sent": "2024-01-01", "actioned_24h": 1},
        {"user_id": "A", "email_id": "e2", "date_sent": "2024-01-01", "actioned_24h": 0},
        {"user_id": "A", "email_id": "e3", "date_sent": "2024-01-02", "actioned_24h": 1},
        {"user_id": "A", "email_id": "e4", "date_sent": "2024-01-03", "actioned_24h": 0},
    ]
    full_df = _make_full_df_from_rows(rows)
    scores = _compute_confound_scores_full_data(full_df)

    by_date = {
        row["date_sent"]: row["user_prior_engagement_score"]
        for row in scores.filter(pl.col("user_id") == "A").sort("date_sent").iter_rows(named=True)
    }

    date1 = datetime.date(2024, 1, 1)
    date2 = datetime.date(2024, 1, 2)
    date3 = datetime.date(2024, 1, 3)

    # date 1: no prior events → prior_action_count=0 → log(0+1)=0
    assert abs(by_date[date1] - math.log(0 + 1)) < 1e-9, f"date1={by_date[date1]}"
    # date 2: prior = date-1 sum = 1 → log(1+1)
    assert abs(by_date[date2] - math.log(1 + 1)) < 1e-9, f"date2={by_date[date2]}"
    # date 3: prior = date1(1) + date2(1) = 2 → log(2+1)
    assert abs(by_date[date3] - math.log(2 + 1)) < 1e-9, f"date3={by_date[date3]}"


def test_strict_prior_consistent_with_per_row_audit() -> None:
    rng = np.random.default_rng(42)
    n_users = 5
    dates = [f"2024-0{m}-{d:02d}" for m in range(1, 5) for d in [1, 5, 10, 15, 20]]

    # Each row gets a unique email_id (event-scoped) to avoid duplicate (user,email,date) keys.
    # Dates are sampled with replacement so same-day ties exist at the (user, date) level.
    rows: list[dict] = []
    event_id = 0
    for u in range(n_users):
        for _ in range(20):
            date = rng.choice(dates)
            actioned = int(rng.random() < 0.15)
            rows.append(
                {
                    "user_id": f"user_{u}",
                    "email_id": f"email_{event_id}",
                    "date_sent": date,
                    "actioned_24h": actioned,
                }
            )
            event_id += 1

    full_df = _make_full_df_from_rows(rows)
    scores = _compute_confound_scores_full_data(full_df)

    # _assert_no_future_leak needs actioned_24h alongside the score columns.
    # Scores preserves full_df row count/order; hstack is safe here.
    audit_df = full_df.hstack(
        scores.select(["user_prior_engagement_score", "email_popularity_score"])
    )

    diag_user = _assert_no_future_leak(
        audit_df,
        score_col="user_prior_engagement_score",
        time_col="date_sent",
        group_col="user_id",
        n_sample=len(audit_df),
        seed=7,
    )
    assert diag_user.severity != "error", diag_user.message

    diag_email = _assert_no_future_leak(
        audit_df,
        score_col="email_popularity_score",
        time_col="date_sent",
        group_col="email_id",
        n_sample=len(audit_df),
        seed=7,
    )
    assert diag_email.severity != "error", diag_email.message


def test_new_lazy_dag_matches_old_eager_pipeline_when_no_ties() -> None:
    rng = np.random.default_rng(99)
    n_rows = 100
    # Ensure unique (user, date) and unique (email, date) pairs by construction
    dates = [np.datetime64("2024-01-01", "D") + np.timedelta64(i, "D") for i in range(n_rows)]
    user_ids = [f"user_{i % 10}" for i in range(n_rows)]
    email_ids = [f"email_{i % 7}" for i in range(n_rows)]
    actioned = rng.integers(0, 2, size=n_rows).tolist()

    full_df = pl.DataFrame(
        {
            "user_id": user_ids,
            "email_id": email_ids,
            "date_sent": [str(d) for d in dates],
            "actioned_24h": actioned,
        }
    ).with_columns(
        pl.col("date_sent").cast(pl.Date),
        pl.col("actioned_24h").cast(pl.Int32),
    )

    # --- old eager pipeline (inlined from original implementation) ---
    sorted_df = full_df.sort(["user_id", "email_id", "date_sent"])

    user_date_level_eager = (
        sorted_df.group_by(["user_id", "date_sent"])
        .agg(pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"))
        .sort(["user_id", "date_sent"])
        .with_columns(
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("user_id")
            .fill_null(0)
            .alias("_prior_action_count")
        )
        .with_columns(
            (pl.col("_prior_action_count") + 1)
            .log(base=2.718281828459045)
            .alias("user_prior_engagement_score")
        )
        .select(["user_id", "date_sent", "user_prior_engagement_score"])
    )
    user_scores_eager = sorted_df.join(
        user_date_level_eager, on=["user_id", "date_sent"], how="left"
    ).select(["user_id", "email_id", "date_sent", "user_prior_engagement_score"])

    email_date_level_eager = (
        sorted_df.group_by(["email_id", "date_sent"])
        .agg(
            pl.len().cast(pl.Int64).alias("_sends_on_date"),
            pl.col("actioned_24h").cast(pl.Int32).sum().alias("_acts_on_date"),
        )
        .sort(["email_id", "date_sent"])
        .with_columns(
            pl.col("_sends_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id")
            .fill_null(0)
            .alias("_prior_sends"),
            pl.col("_acts_on_date")
            .cum_sum()
            .shift(1)
            .over("email_id")
            .fill_null(0)
            .alias("_prior_acts"),
        )
        .with_columns(
            ((pl.col("_prior_acts") + 1) / (pl.col("_prior_sends") + 2)).alias(
                "email_popularity_score"
            )
        )
        .select(["email_id", "date_sent", "email_popularity_score"])
    )
    email_scores_eager = sorted_df.join(
        email_date_level_eager, on=["email_id", "date_sent"], how="left"
    ).select(["user_id", "email_id", "date_sent", "email_popularity_score"])

    old_joined = user_scores_eager.join(
        email_scores_eager, on=["user_id", "email_id", "date_sent"], how="inner"
    ).sort(["user_id", "email_id", "date_sent"])

    # --- new lazy DAG ---
    new_joined = _compute_confound_scores_full_data(full_df).sort(
        ["user_id", "email_id", "date_sent"]
    )

    assert len(new_joined) == len(old_joined)

    old_user_scores = old_joined["user_prior_engagement_score"].to_numpy()
    new_user_scores = new_joined["user_prior_engagement_score"].to_numpy()
    assert np.max(np.abs(old_user_scores - new_user_scores)) < 1e-9

    old_email_scores = old_joined["email_popularity_score"].to_numpy()
    new_email_scores = new_joined["email_popularity_score"].to_numpy()
    assert np.max(np.abs(old_email_scores - new_email_scores)) < 1e-9
