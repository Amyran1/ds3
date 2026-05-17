"""run_01: forward quarterly walk-forward at 5% sample tier.

The first canonical ship run. Wires user_emails v3 + prior_action_recency v1
through harness() with LightGBM classifier args from the plan's resolved decisions.

Direct append to results.jsonl — libs/ledgers.py deferred per plan scope.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")

import polars as pl

from entities.civic_shout_user_emails.cache import cache as user_emails_cache
from projects.civic_shout_action_rate_increase.features.prior_action_recency.cache import (
    cache as recency_feature_cache,
)
from projects.civic_shout_action_rate_increase.harness import (
    HarnessResponse,
    ML_Model_Config,
    ML_Model_Type,
    RunResultMetadata,
    harness,
)
from projects.civic_shout_action_rate_increase.runs.manifest_helper import emit_manifest
from projects.civic_shout_action_rate_increase.runs.timing_helper import emit_timing_row

_RUNS_DIR = Path(__file__).parent
_LEDGER = _RUNS_DIR / "results.jsonl"

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
    "force_col_wise": True,
    "feature_pre_filter": True,
}

_LR_ARGS = {
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 200,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": None,
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


def _build_harness_kwargs(
    joined: pl.DataFrame,
    artifacts_dir: Path,
    sample_frac: float | None,
    sample_seed: int,
    scope: str,
    bootstrap_n_resamples: int | None,
    model: str = "lgbm",
) -> dict:
    if model == "lr":
        ml_model_type = ML_Model_Type.LOGISTIC_REGRESSION
        model_args: dict = _LR_ARGS
    else:
        ml_model_type = ML_Model_Type.LIGHTGBM_CLASSIFIER
        model_args = _LGBM_ARGS

    kwargs: dict = dict(
        data=joined,
        feature_cols=_FEATURE_COLS,
        ml_model_config=ML_Model_Config(
            ml_model_type=ml_model_type,
            args=model_args,
            column_mapping={"group": "user_id"},
        ),
        outcome_variable="actioned_24h",
        sample_frac=sample_frac,
        sample_seed=sample_seed,
        predictions_dir=str(artifacts_dir),
        scope=scope,
    )
    if bootstrap_n_resamples is not None:
        kwargs["bootstrap_n_resamples"] = bootstrap_n_resamples
    return kwargs


def _write_result(
    response: object,
    metadata: RunResultMetadata,
    run_dir: Path,
) -> None:
    assert isinstance(response, HarnessResponse)

    row = response.to_result_row(metadata)
    with _LEDGER.open("a") as f:
        f.write(row.model_dump_json() + "\n")

    (run_dir / "harness_response.json").write_text(response.model_dump_json(indent=2))

    emit_manifest(metadata, response, run_dir)
    emit_timing_row(metadata, response, _RUNS_DIR.parent)

    print(
        f"[{metadata.run_id}] scope={metadata.scope} "
        f"primary={response.summary.primary_metric_value:.4f} "
        f"ci=[{response.metrics.primary.ci_low:.4f}, {response.metrics.primary.ci_high:.4f}] "
        f"elapsed={response.performance.total_seconds:.1f}s"
    )


def _should_escalate_to_comparison(mean: float, ci_half_width: float, delta: float = 0.001) -> bool:
    """Return True if the comparison_fast pilot result is ambiguous (warrants full comparison).

    Returns False (skip full comparison) when the CI is clearly above or below the
    null boundary 0.5 ± delta.  Returns True (escalate) when ambiguous.
    """
    lower = mean - 1.96 * ci_half_width
    upper = mean + 1.96 * ci_half_width
    clearly_above = lower > 0.5 + delta
    clearly_below = upper < 0.5 - delta
    return not (clearly_above or clearly_below)


def main(
    run_id: str = "run_01",
    sample_frac: float | None = 0.05,
    sample_seed: int = 42,
    comparison_group: str = "action_rate_increase_temporal_holdout_residualized_auc_v1",
    scope: str = "comparison",
    profile: bool = False,
    bootstrap_n_resamples: int | None = None,
    auto_promote: bool = False,
    comparison_fast_first: bool = False,
    model: str = "lgbm",
) -> None:
    user_emails_df = user_emails_cache.get(3)
    features_df = recency_feature_cache.get(1)

    joined = user_emails_df.join(
        features_df, on=["user_id", "email_id", "date_sent"], how="left"
    ).drop(["opened", "clicked", "verified_opened"], strict=False)

    # exclude_last_send: drop rows within 24h of the latest send to avoid
    # outcome label instability (Codex scoring-stability finding)
    max_date = joined["date_sent"].max()
    joined = joined.filter(pl.col("date_sent") < (max_date - pl.duration(hours=24)))

    run_dir = _RUNS_DIR / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    sha = _git_sha()
    metadata = RunResultMetadata(
        run_id=run_id,
        project="civic_shout_action_rate_increase",
        comparison_group=comparison_group,
        scope=scope,  # type: ignore[arg-type]
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=sha,
        data_fingerprint=None,
    )

    harness_kwargs = _build_harness_kwargs(
        joined, artifacts_dir, sample_frac, sample_seed, scope, bootstrap_n_resamples, model
    )

    response: object = None

    def _run_harness_inner() -> None:
        nonlocal response
        response = harness(**harness_kwargs)

    if profile:
        prof = cProfile.Profile()
        prof.enable()
        _run_harness_inner()
        prof.disable()

        pstats_path = artifacts_dir / "profile.pstats"
        prof.dump_stats(str(pstats_path))

        txt_path = artifacts_dir / "profile.txt"
        stream = StringIO()
        ps = pstats.Stats(prof, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(30)
        ps.sort_stats("tottime")
        ps.print_stats(30)
        txt_path.write_text(stream.getvalue())
    else:
        _run_harness_inner()

    # T2c: comparison_fast_first — run 4-fold pilot, escalate to full comparison only when ambiguous.
    if comparison_fast_first:
        assert isinstance(response, HarnessResponse)
        pilot_mean = response.summary.primary_metric_value
        ci_low_pilot = response.metrics.primary.ci_low
        ci_high_pilot = response.metrics.primary.ci_high
        ci_half_width: float = 0.0
        if ci_low_pilot is not None and ci_high_pilot is not None:
            ci_half_width = (ci_high_pilot - ci_low_pilot) / 2.0
        lower = pilot_mean - 1.96 * ci_half_width
        upper = pilot_mean + 1.96 * ci_half_width

        _write_result(response, metadata, run_dir)

        if not _should_escalate_to_comparison(pilot_mean, ci_half_width):
            print(
                f"[comparison_fast pilot] clear directional signal "
                f"(mean4={pilot_mean:.4f}, CI=[{lower:.4f}, {upper:.4f}]); "
                f"skipping full comparison run."
            )
            return

        print(
            f"[comparison_fast pilot] ambiguous "
            f"(|mean4 - 0.5| within CI ± 0.001); running full comparison."
        )
        full_run_id = run_id + "_comparison"
        full_run_dir = _RUNS_DIR / full_run_id
        full_artifacts_dir = full_run_dir / "artifacts"
        full_run_dir.mkdir(parents=True, exist_ok=True)
        full_artifacts_dir.mkdir(exist_ok=True)

        full_metadata = RunResultMetadata(
            run_id=full_run_id,
            project="civic_shout_action_rate_increase",
            comparison_group=comparison_group,
            scope="comparison",
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            git_sha=sha,
            data_fingerprint=None,
        )
        full_kwargs = _build_harness_kwargs(
            joined,
            full_artifacts_dir,
            sample_frac,
            sample_seed,
            "comparison",
            bootstrap_n_resamples,
            model,
        )
        full_response = harness(**full_kwargs)
        _write_result(full_response, full_metadata, full_run_dir)
        return

    # T4.6: adaptive promotion gate — auto-promote fast_iter to comparison when
    # the directional verdict is unambiguous (|primary - 0.5| > 1.5 × CI half-width).
    if auto_promote and scope == "fast_iter":
        assert isinstance(response, HarnessResponse)
        primary = response.summary.primary_metric_value
        ci_low = response.metrics.primary.ci_low
        ci_high = response.metrics.primary.ci_high
        half_width: float = 0.0
        if ci_low is not None and ci_high is not None:
            half_width = (ci_high - ci_low) / 2.0
            directional_clear = abs(primary - 0.5) > 1.5 * half_width
        else:
            directional_clear = False

        if directional_clear:
            print(
                f"[{run_id}] directional verdict clear "
                f"(|{primary:.4f} - 0.5| > 1.5 × {half_width:.4f}); promoting to comparison."
            )
            _write_result(response, metadata, run_dir)

            promoted_run_id = run_id + "_promoted"
            promoted_run_dir = _RUNS_DIR / promoted_run_id
            promoted_artifacts_dir = promoted_run_dir / "artifacts"
            promoted_run_dir.mkdir(parents=True, exist_ok=True)
            promoted_artifacts_dir.mkdir(exist_ok=True)

            promoted_metadata = RunResultMetadata(
                run_id=promoted_run_id,
                project="civic_shout_action_rate_increase",
                comparison_group=comparison_group,
                scope="comparison",
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                git_sha=sha,
                data_fingerprint=None,
            )
            promoted_kwargs = _build_harness_kwargs(
                joined,
                promoted_artifacts_dir,
                sample_frac,
                sample_seed,
                "comparison",
                bootstrap_n_resamples,
                model,
            )
            promoted_response = harness(**promoted_kwargs)
            _write_result(promoted_response, promoted_metadata, promoted_run_dir)
        else:
            print(
                f"[{run_id}] directional verdict ambiguous "
                f"(|{primary:.4f} - 0.5| ≤ 1.5 × {half_width:.4f}); not promoting."
            )
            ambiguous_metadata = RunResultMetadata(
                run_id=run_id,
                project="civic_shout_action_rate_increase",
                comparison_group=comparison_group,
                scope="fast_iter",  # type: ignore[arg-type]
                verdict="ambiguous",
                recorded_at_utc=metadata.recorded_at_utc,
                git_sha=sha,
                data_fingerprint=None,
            )
            _write_result(response, ambiguous_metadata, run_dir)
    else:
        _write_result(response, metadata, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_01: quarterly walk-forward at configurable sample tier."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Wrap harness() in cProfile; write profile.pstats and profile.txt to artifacts/.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.05,
        dest="sample_frac",
        help="Fraction of data to sample (default: 0.05). Use 0.01 for fast iteration probes.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="run_01",
        dest="run_id",
        help="Run identifier (default: run_01).",
    )
    parser.add_argument(
        "--bootstrap-n-resamples",
        type=int,
        default=None,
        dest="bootstrap_n_resamples",
        help="Bootstrap resample count (default: harness default of 500). Use 100 for fast-iter mode.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="comparison",
        choices=[
            "smoke",
            "comparison",
            "champion_candidate",
            "reproduction",
            "fast_iter",
            "comparison_fast",
        ],
        help="Evaluation scope (default: comparison). Use fast_iter for rapid directional iteration, comparison_fast for 4-fold pilot.",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        default=False,
        dest="auto_promote",
        help=(
            "When scope=fast_iter: auto-promote to scope=comparison if directional verdict is clear. "
            "Rejected if --scope is not fast_iter."
        ),
    )
    parser.add_argument(
        "--comparison-fast-first",
        action="store_true",
        default=False,
        dest="comparison_fast_first",
        help=(
            "Run scope=comparison_fast (4-fold pilot) first. If directional signal is clear "
            "(CI excludes 0.5 ± 0.001), skip the full comparison. Otherwise escalate to "
            "scope=comparison with a separate run_id (<run_id>_comparison)."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "lr"],
        default="lgbm",
        help="Model type: lgbm (default, backward-compat) or lr (logistic regression).",
    )
    args = parser.parse_args()

    if args.auto_promote and args.scope != "fast_iter":
        parser.error("--auto-promote requires --scope fast_iter")

    if args.comparison_fast_first and args.scope != "comparison_fast":
        parser.error("--comparison-fast-first requires --scope comparison_fast")

    main(
        run_id=args.run_id,
        sample_frac=args.sample_frac,
        profile=args.profile,
        bootstrap_n_resamples=args.bootstrap_n_resamples,
        scope=args.scope,
        auto_promote=args.auto_promote,
        comparison_fast_first=args.comparison_fast_first,
        model=args.model,
    )
