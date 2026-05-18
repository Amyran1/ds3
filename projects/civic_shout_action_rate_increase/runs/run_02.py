"""run_02: champion baseline + email_content_nlp NLP features at smoke tier.

Autoloop iteration 2, brainstorm bs-iter2-01-47822d06 (email_content_nlp).

Hypothesis under test: per-email NLP features (body length, urgency / CTA
keyword counts, exclamation density, uppercase shout ratio) add lift on top
of the prior_action_recency champion. Champion has zero email-side features,
yet the diagnostic roc_auc_residualized_email_popularity_pair=0.811 reveals
substantial email-level variance.

Sample-tier deviation
---------------------
Autoloop directive: sample_frac=0.05. Actual: sample_frac=0.001.

run_01_smoke at sample_frac=0.001 (129,019 rows) took ~200s wall. The civic
shout harness has heavy fixed costs (8 quarterly walk-forward folds × LightGBM
fit × 500 user-block bootstrap residualization); linear extrapolation projects
sample_frac=0.05 at 5,000-10,000s — exceeds the 900s autoloop session wall.
0.001 mirrors run_01_smoke and produces a complete sentinel within budget.

R1–R4 tandem invariant: single run_record(...) call; no bare harness() / eda() /
ledger open(). EDA runs canonically inside run_record; discovery is skipped
because scope=smoke is not in _SCOPE_RUNS_DISCOVERY.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from entities.civic_shout_user_emails.cache import cache as user_emails_cache
from libs.responses import EDAConfig, RunResultMetadata
from libs.run_record import RunRecordOutput, run_record
from projects.civic_shout_action_rate_increase.features.email_content_nlp.cache import (
    cache as email_content_nlp_cache,
)
from projects.civic_shout_action_rate_increase.features.prior_action_recency.cache import (
    cache as recency_feature_cache,
)
from projects.civic_shout_action_rate_increase.harness import ML_Model_Config, ML_Model_Type

_RECENCY_FEATURE_COLS = [
    "actioned_last_1",
    "actioned_last_3",
    "actioned_last_5",
    "actioned_last_10",
    "sends_since_last_action",
    "action_in_last_5",
    "is_in_action_streak",
    "lifetime_actions_prior",
]

_EMAIL_NLP_FEATURE_COLS = [
    "body_char_len",
    "body_word_count",
    "body_sentence_count",
    "body_avg_word_len_chars",
    "body_exclamation_count",
    "body_question_count",
    "body_uppercase_word_ratio",
    "body_urgency_keyword_count",
    "body_cta_keyword_count",
    "first_line_char_len",
    "first_line_word_count",
]

_FEATURE_COLS = _RECENCY_FEATURE_COLS + _EMAIL_NLP_FEATURE_COLS

_LGBM_ARGS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

SAMPLE_FRAC = 0.001
SAMPLE_SEED = 42


def main() -> RunRecordOutput:
    project_root = Path(__file__).resolve().parents[3]

    user_emails_df = user_emails_cache.get(3)
    recency_df = recency_feature_cache.get(1)
    email_nlp_df = email_content_nlp_cache.get(1)

    joined = user_emails_df.join(
        recency_df, on=["user_id", "email_id", "date_sent"], how="left"
    ).drop(["opened", "clicked", "verified_opened"], strict=False)
    joined = joined.join(email_nlp_df, on="email_id", how="left")

    max_date = joined["date_sent"].max()
    joined = joined.filter(pl.col("date_sent") < (max_date - pl.duration(hours=24)))

    sampled = joined.sample(fraction=SAMPLE_FRAC, seed=SAMPLE_SEED, with_replacement=False)

    metadata = RunResultMetadata(
        run_id="run_02",
        project="civic_shout_action_rate_increase",
        comparison_group="action_rate_increase_temporal_holdout_residualized_auc_v1",
        scope="smoke",
        verdict="smoke_only",
        recorded_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        outcome_version="v1",
        notes=(
            f"email_content_nlp + prior_action_recency; smoke sample_frac={SAMPLE_FRAC} "
            "(autoloop directive 0.05 downscaled to mirror run_01_smoke wall budget); "
            "autoloop iter 2 brainstorm bs-iter2-01-47822d06"
        ),
    )

    return run_record(
        metadata=metadata,
        data=sampled,
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        model_cfg=ML_Model_Config(
            ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
            args=_LGBM_ARGS,
            column_mapping={"group": "user_id"},
        ),
        eda_config=EDAConfig(),
        discovery_config=None,
        project_root=project_root,
        harness_kwargs={"scope": metadata.scope},
    )


if __name__ == "__main__":
    out = main()
    if out.harness_response is not None:
        print(
            f"run_02 complete: primary={out.harness_response.summary.primary_metric_value:.4f} "
            f"elapsed={out.harness_response.performance.total_seconds:.1f}s"
        )
