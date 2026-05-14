"""run_01_smoke_gpu: 0.1% smoke variant calling harness_gpu instead of harness."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from entities.civic_shout_user_emails.cache import cache as user_emails_cache
from projects.civic_shout_action_rate_increase.eda import eda
from projects.civic_shout_action_rate_increase.features.prior_action_recency.cache import (
    cache as recency_feature_cache,
)
from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    RunResultMetadata,
)
from projects.civic_shout_action_rate_increase.harness_gpu import harness
from projects.civic_shout_action_rate_increase.runs.manifest_helper import emit_manifest
from projects.civic_shout_action_rate_increase.runs.timing_helper import emit_timing_row

_RUNS_DIR = Path(__file__).parent
_LEDGER = _RUNS_DIR / "results.jsonl"
_EDA_LEDGER = _RUNS_DIR / "eda_results.jsonl"

_FEATURE_COLS = [
    "actioned_last_1",
    "actioned_last_3",
    "actioned_last_5",
    "actioned_last_10",
    "sends_since_last_action",
    "action_in_last_5",
    "is_in_action_streak",
    "lifetime_actions_prior",
]

_LGBM_ARGS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def main(
    run_id: str = "run_01_smoke_gpu",
    sample_frac: float | None = 0.001,
    sample_seed: int = 42,
    comparison_group: str = "action_rate_increase_temporal_holdout_residualized_auc_v1",
    scope: str = "smoke",
) -> None:
    user_emails_df = user_emails_cache.get(3)
    features_df = recency_feature_cache.get(1)

    joined = user_emails_df.join(
        features_df, on=["user_id", "email_id", "date_sent"], how="left"
    ).drop(["opened", "clicked", "verified_opened"], strict=False)

    max_date = joined["date_sent"].max()
    joined = joined.filter(pl.col("date_sent") < (max_date - pl.duration(hours=24)))

    run_dir = _RUNS_DIR / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    recorded_at = datetime.now(timezone.utc)
    metadata = RunResultMetadata(
        run_id=run_id,
        project="civic_shout_action_rate_increase",
        comparison_group=comparison_group,
        scope=scope,  # type: ignore[arg-type]
        recorded_at_utc=recorded_at.isoformat(),
        git_sha=_git_sha(),
        data_fingerprint=None,
    )

    response = harness(
        data=joined,
        feature_cols=_FEATURE_COLS,
        ml_model_config=ML_Model_Config(
            ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
            args=_LGBM_ARGS,
            column_mapping={"group": "user_id"},
        ),
        outcome_variable="actioned_24h",
        sample_frac=sample_frac,
        sample_seed=sample_seed,
        predictions_dir=str(artifacts_dir),
    )

    row = response.to_result_row(metadata)
    with _LEDGER.open("a") as f:
        f.write(row.model_dump_json() + "\n")

    (run_dir / "harness_response.json").write_text(response.model_dump_json(indent=2))

    eda_response = eda(
        data=joined,
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        eda_config=None,
    )
    (run_dir / "eda_response.json").write_text(eda_response.model_dump_json(indent=2))
    (run_dir / "eda_summary.md").write_text(eda_response.to_summary_md(metadata))

    with _EDA_LEDGER.open("a") as f:
        f.write(eda_response.to_eda_summary_row(metadata).model_dump_json() + "\n")

    emit_manifest(metadata, response, run_dir)
    emit_timing_row(metadata, response, _RUNS_DIR.parent)

    print(
        f"[{run_id}] primary={response.summary.primary_metric_value:.4f} "
        f"ci=[{response.metrics.primary.ci_low:.4f}, {response.metrics.primary.ci_high:.4f}] "
        f"elapsed={response.performance.total_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
