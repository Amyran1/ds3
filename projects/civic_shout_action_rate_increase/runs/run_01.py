"""run_01: forward quarterly walk-forward at 5% sample tier.

The first canonical ship run. Wires user_emails v3 + prior_action_recency v1
through libs.run_record.run_record(...) with LightGBM classifier args from
the plan's resolved decisions.

R1–R4 tandem invariant: all non-profile branches route through a single
run_record(...) call per logical run. The comparison_fast_first path issues
one call for the pilot and a second call for the full run if the pilot is
ambiguous; the auto_promote path issues one call for the fast_iter run and a
second call for the promoted comparison if the verdict is clear.

--profile is a developer diagnostic and intentionally bypasses
run_record/ledgers. Use only for performance profiling; never as a
decision-bearing artifact.

Predictions write policy is scope-gated by libs.run_record (_SCOPE_WRITES_PREDICTIONS).
comparison_fast is excluded from that set; the --write-predictions flag was removed.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")

import polars as pl

from entities.civic_shout_user_emails.cache import cache as user_emails_cache
from libs.ledgers import LedgerWriter
from libs.responses import DiscoveryConfig, EDAConfig, RunResultMetadata
from libs.run_record import RunRecordOutput, run_record
from projects.civic_shout_action_rate_increase.features.prior_action_recency.cache import (
    cache as recency_feature_cache,
)
from projects.civic_shout_action_rate_increase.harness import (
    HarnessResponse,
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from projects.civic_shout_action_rate_increase.runs._ledger_fallback import (
    write_error_traceback,
    write_failure_shell_row,
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
    "tol": 1e-3,  # was 1e-4 in Phase C; tighter tol verified within drift gate (< 0.000675 AUC)
    "fit_intercept": True,
    "class_weight": None,
    "random_state": 42,
}

# LR bootstrap defaults are scope-aware:
#   comparison      B=30  — ~2× faster than LGBM on equivalent data; CI std error inflates
#                           by sqrt(500/30)=4.1× vs LGBM. Acceptable for directional lift question.
#   comparison_fast B=20  — 4 folds + tighter bootstrap; sub-10s target at 5% sample.
#                           Std error inflates by sqrt(500/20)=5× — still directional.
# For sub-10s LR use --scope comparison_fast (4 folds + B=20).
# Override via --bootstrap-n-resamples N for tighter CIs (slower).
_LR_DEFAULT_BOOTSTRAP_N_COMPARISON = 30
_LR_DEFAULT_BOOTSTRAP_N_COMPARISON_FAST = 20


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


def _build_model_cfg(model: str) -> ML_Model_Config:
    if model == "lr":
        return ML_Model_Config(
            ml_model_type=ML_Model_Type.LOGISTIC_REGRESSION,
            args=_LR_ARGS,
            column_mapping={"group": "user_id"},
        )
    return ML_Model_Config(
        ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
        args=_LGBM_ARGS,
        column_mapping={"group": "user_id"},
    )


def _build_extra_harness_kwargs(
    scope: str,
    bootstrap_n_resamples: int | None,
    model: str,
    lr_warm_start: bool,
    lr_skip_audit: bool,
    lr_fold_backend: str,
    lr_skip_heavy_diagnostics: bool,
) -> dict:
    """Build the non-reserved harness kwargs (everything except the 7 reserved keys)."""
    if model == "lr" and bootstrap_n_resamples is None:
        if scope == "comparison_fast":
            resolved_bootstrap_n: int | None = _LR_DEFAULT_BOOTSTRAP_N_COMPARISON_FAST
        else:
            resolved_bootstrap_n = _LR_DEFAULT_BOOTSTRAP_N_COMPARISON
    else:
        resolved_bootstrap_n = bootstrap_n_resamples

    kwargs: dict = {"scope": scope}
    if resolved_bootstrap_n is not None:
        kwargs["bootstrap_n_resamples"] = resolved_bootstrap_n
    if lr_warm_start and model == "lr":
        kwargs["lr_warm_start"] = True
        kwargs["fold_n_jobs"] = 1
    if lr_skip_audit and model == "lr":
        kwargs["lr_skip_audit"] = True
    if model == "lr" and lr_fold_backend == "processes":
        kwargs["lr_fold_backend"] = "processes"
    if lr_skip_heavy_diagnostics and model == "lr":
        kwargs["lr_skip_heavy_diagnostics"] = True
    return kwargs


def _build_harness_kwargs(
    joined: pl.DataFrame,
    artifacts_dir: Path,
    sample_frac: float | None,
    sample_seed: int,
    scope: str,
    bootstrap_n_resamples: int | None,
    model: str = "lgbm",
    lr_warm_start: bool = False,
    lr_skip_audit: bool = False,
    lr_fold_backend: str = "threads",
    lr_skip_heavy_diagnostics: bool = False,
) -> dict:
    if model == "lr":
        ml_model_type = ML_Model_Type.LOGISTIC_REGRESSION
        model_args: dict = _LR_ARGS
    else:
        ml_model_type = ML_Model_Type.LIGHTGBM_CLASSIFIER
        model_args = _LGBM_ARGS

    if model == "lr" and bootstrap_n_resamples is None:
        if scope == "comparison_fast":
            resolved_bootstrap_n: int | None = _LR_DEFAULT_BOOTSTRAP_N_COMPARISON_FAST
        else:
            resolved_bootstrap_n = _LR_DEFAULT_BOOTSTRAP_N_COMPARISON
    else:
        resolved_bootstrap_n = bootstrap_n_resamples

    # Cv5.4: LR comparison_fast is a pilot tier — predictions.parquet not needed
    # for the decision-bearing flow. Predictions write is scope-gated by libs.run_record
    # (_SCOPE_WRITES_PREDICTIONS excludes comparison_fast). Profile path uses this helper
    # directly and can pass any scope; the harness itself governs predictions_dir.
    _skip_predictions = model == "lr" and scope == "comparison_fast"
    predictions_dir_val: str | None = None if _skip_predictions else str(artifacts_dir)

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
        predictions_dir=predictions_dir_val,
        scope=scope,
    )
    if resolved_bootstrap_n is not None:
        kwargs["bootstrap_n_resamples"] = resolved_bootstrap_n
    if lr_warm_start and model == "lr":
        kwargs["lr_warm_start"] = True
        kwargs["fold_n_jobs"] = 1
    if lr_skip_audit and model == "lr":
        kwargs["lr_skip_audit"] = True
    if model == "lr" and lr_fold_backend == "processes":
        kwargs["lr_fold_backend"] = "processes"
    if lr_skip_heavy_diagnostics and model == "lr":
        kwargs["lr_skip_heavy_diagnostics"] = True
    return kwargs


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


def _post_run_record_artifacts(
    out: RunRecordOutput,
    project_root: Path,
) -> None:
    run_dir = out.artifacts_dir.parent
    proj_dir = project_root / "projects" / out.metadata.project
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "harness_response.json").write_text(out.harness_response.model_dump_json(indent=2))
    emit_manifest(out.metadata, out.harness_response, run_dir)
    emit_timing_row(out.metadata, out.harness_response, proj_dir)
    print(
        f"[{out.metadata.run_id}] scope={out.metadata.scope} "
        f"primary={out.harness_response.summary.primary_metric_value:.4f} "
        f"ci=[{out.harness_response.metrics.primary.ci_low:.4f}, "
        f"{out.harness_response.metrics.primary.ci_high:.4f}] "
        f"elapsed={out.harness_response.performance.total_seconds:.1f}s"
    )


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
    lr_warm_start: bool = False,
    lr_skip_audit: bool = False,
    lr_fold_backend: str = "threads",
    lr_skip_heavy_diagnostics: bool = True,
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

    project_root = Path(__file__).resolve().parents[3]
    run_dir = _RUNS_DIR / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    sha = _git_sha()

    # --profile: direct harness() bypass — does not route through run_record or ledgers.
    if profile:
        metadata = RunResultMetadata(
            run_id=run_id,
            project="civic_shout_action_rate_increase",
            comparison_group=comparison_group,
            scope=scope,  # type: ignore[arg-type]
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            git_sha=sha,
            data_fingerprint="",
            outcome_version="v1",
        )
        harness_kwargs = _build_harness_kwargs(
            joined,
            artifacts_dir,
            sample_frac,
            sample_seed,
            scope,
            bootstrap_n_resamples,
            model,
            lr_warm_start=lr_warm_start,
            lr_skip_audit=lr_skip_audit,
            lr_fold_backend=lr_fold_backend,
            lr_skip_heavy_diagnostics=lr_skip_heavy_diagnostics,
        )
        prof = cProfile.Profile()
        prof.enable()
        try:
            response = harness(**harness_kwargs)
        except Exception as e:
            prof.disable()
            try:
                write_error_traceback(run_dir=run_dir, error=e)
            except Exception as write_err:
                print(f"WARN: failed to write error.txt: {write_err}", file=sys.stderr)
            try:
                write_failure_shell_row(
                    results_path=_LEDGER,
                    run_id=run_id,
                    project="civic_shout_action_rate_increase",
                    comparison_group=comparison_group,
                    scope=scope,
                    status="failed_harness",
                    error_msg=str(e),
                    git_sha=sha,
                )
            except Exception as write_err:
                print(f"WARN: failed to write failure row to ledger: {write_err}", file=sys.stderr)
            raise
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
        return

    # T2c: comparison_fast_first — run 4-fold pilot, escalate to full comparison only when ambiguous.
    if comparison_fast_first:
        pilot_metadata = RunResultMetadata(
            run_id=run_id,
            project="civic_shout_action_rate_increase",
            comparison_group=comparison_group,
            scope="comparison_fast",  # type: ignore[arg-type]
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            git_sha=sha,
            data_fingerprint="",
            outcome_version="v1",
        )
        pilot_extra_kwargs = _build_extra_harness_kwargs(
            scope="comparison_fast",
            bootstrap_n_resamples=bootstrap_n_resamples,
            model=model,
            lr_warm_start=lr_warm_start,
            lr_skip_audit=lr_skip_audit,
            lr_fold_backend=lr_fold_backend,
            lr_skip_heavy_diagnostics=lr_skip_heavy_diagnostics,
        )
        try:
            pilot_out = run_record(
                metadata=pilot_metadata,
                data=joined,
                feature_cols=_FEATURE_COLS,
                outcome_variable="actioned_24h",
                model_cfg=_build_model_cfg(model),
                sample_frac=sample_frac,
                sample_seed=sample_seed,
                harness_kwargs=pilot_extra_kwargs,
                eda_config=EDAConfig(),
                discovery_config=None,
                project_root=project_root,
            )
        except Exception as e:
            try:
                write_error_traceback(run_dir=run_dir, error=e)
            except Exception as write_err:
                print(f"WARN: failed to write error.txt: {write_err}", file=sys.stderr)
            raise

        _post_run_record_artifacts(pilot_out, project_root)

        assert pilot_out.harness_response is not None
        pilot_mean = pilot_out.harness_response.summary.primary_metric_value
        ci_low_pilot = pilot_out.harness_response.metrics.primary.ci_low
        ci_high_pilot = pilot_out.harness_response.metrics.primary.ci_high
        ci_half_width: float = 0.0
        if ci_low_pilot is not None and ci_high_pilot is not None:
            ci_half_width = (ci_high_pilot - ci_low_pilot) / 2.0
        lower = pilot_mean - 1.96 * ci_half_width
        upper = pilot_mean + 1.96 * ci_half_width

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
        full_run_dir.mkdir(parents=True, exist_ok=True)

        full_metadata = RunResultMetadata(
            run_id=full_run_id,
            project="civic_shout_action_rate_increase",
            comparison_group=comparison_group,
            scope="comparison",  # type: ignore[arg-type]
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            git_sha=sha,
            data_fingerprint="",
            outcome_version="v1",
        )
        full_extra_kwargs = _build_extra_harness_kwargs(
            scope="comparison",
            bootstrap_n_resamples=bootstrap_n_resamples,
            model=model,
            lr_warm_start=False,
            lr_skip_audit=lr_skip_audit,
            lr_fold_backend=lr_fold_backend,
            lr_skip_heavy_diagnostics=lr_skip_heavy_diagnostics,
        )
        try:
            full_out = run_record(
                metadata=full_metadata,
                data=joined,
                feature_cols=_FEATURE_COLS,
                outcome_variable="actioned_24h",
                model_cfg=_build_model_cfg(model),
                sample_frac=sample_frac,
                sample_seed=sample_seed,
                harness_kwargs=full_extra_kwargs,
                eda_config=EDAConfig(),
                discovery_config=DiscoveryConfig(),
                project_root=project_root,
            )
        except Exception as e:
            try:
                write_error_traceback(run_dir=full_run_dir, error=e)
            except Exception as write_err:
                print(f"WARN: failed to write error.txt: {write_err}", file=sys.stderr)
            raise

        _post_run_record_artifacts(full_out, project_root)
        return

    # T4.6: adaptive promotion gate — auto-promote fast_iter to comparison when
    # the directional verdict is unambiguous (|primary - 0.5| > 1.5 × CI half-width).
    if auto_promote and scope == "fast_iter":
        # Do not set verdict here — run_record commits the fast_iter row eagerly
        # before we know promotion outcome. Ambiguous signal written separately below.
        fast_metadata = RunResultMetadata(
            run_id=run_id,
            project="civic_shout_action_rate_increase",
            comparison_group=comparison_group,
            scope="fast_iter",  # type: ignore[arg-type]
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            git_sha=sha,
            data_fingerprint="",
            outcome_version="v1",
        )
        fast_extra_kwargs = _build_extra_harness_kwargs(
            scope="fast_iter",
            bootstrap_n_resamples=bootstrap_n_resamples,
            model=model,
            lr_warm_start=lr_warm_start,
            lr_skip_audit=lr_skip_audit,
            lr_fold_backend=lr_fold_backend,
            lr_skip_heavy_diagnostics=lr_skip_heavy_diagnostics,
        )
        try:
            fast_out = run_record(
                metadata=fast_metadata,
                data=joined,
                feature_cols=_FEATURE_COLS,
                outcome_variable="actioned_24h",
                model_cfg=_build_model_cfg(model),
                sample_frac=sample_frac,
                sample_seed=sample_seed,
                harness_kwargs=fast_extra_kwargs,
                eda_config=EDAConfig(),
                discovery_config=None,
                project_root=project_root,
            )
        except Exception as e:
            try:
                write_error_traceback(run_dir=run_dir, error=e)
            except Exception as write_err:
                print(f"WARN: failed to write error.txt: {write_err}", file=sys.stderr)
            raise

        _post_run_record_artifacts(fast_out, project_root)

        assert fast_out.harness_response is not None
        primary = fast_out.harness_response.summary.primary_metric_value
        ci_low = fast_out.harness_response.metrics.primary.ci_low
        ci_high = fast_out.harness_response.metrics.primary.ci_high
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
            promoted_run_id = run_id + "_promoted"
            promoted_run_dir = _RUNS_DIR / promoted_run_id
            promoted_run_dir.mkdir(parents=True, exist_ok=True)

            promoted_metadata = RunResultMetadata(
                run_id=promoted_run_id,
                project="civic_shout_action_rate_increase",
                comparison_group=comparison_group,
                scope="comparison",  # type: ignore[arg-type]
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                git_sha=sha,
                data_fingerprint="",
                outcome_version="v1",
            )
            promoted_extra_kwargs = _build_extra_harness_kwargs(
                scope="comparison",
                bootstrap_n_resamples=bootstrap_n_resamples,
                model=model,
                lr_warm_start=False,
                lr_skip_audit=lr_skip_audit,
                lr_fold_backend=lr_fold_backend,
                lr_skip_heavy_diagnostics=lr_skip_heavy_diagnostics,
            )
            try:
                promoted_out = run_record(
                    metadata=promoted_metadata,
                    data=joined,
                    feature_cols=_FEATURE_COLS,
                    outcome_variable="actioned_24h",
                    model_cfg=_build_model_cfg(model),
                    sample_frac=sample_frac,
                    sample_seed=sample_seed,
                    harness_kwargs=promoted_extra_kwargs,
                    eda_config=EDAConfig(),
                    discovery_config=DiscoveryConfig(),
                    project_root=project_root,
                )
            except Exception as e:
                try:
                    write_error_traceback(run_dir=promoted_run_dir, error=e)
                except Exception as write_err:
                    print(f"WARN: failed to write error.txt: {write_err}", file=sys.stderr)
                raise

            _post_run_record_artifacts(promoted_out, project_root)
        else:
            print(
                f"[{run_id}] directional verdict ambiguous "
                f"(|{primary:.4f} - 0.5| ≤ 1.5 × {half_width:.4f}); not promoting."
            )
            # Write a second atomic ledger row with verdict="ambiguous" so the signal
            # is preserved as an explicit annotation rather than implied by absence.
            # eda_row reuse is intentional (R4 break by design — annotation-only writes
            # are not yet supported by LedgerWriter).
            if fast_out.result_row is not None:
                ambiguous_row = fast_out.result_row.model_copy(update={"verdict": "ambiguous"})
                LedgerWriter(project_root, "civic_shout_action_rate_increase").append_atomic(
                    result_row=ambiguous_row,
                    eda_row=fast_out.eda_row,
                    discovery_row=None,
                )
                print(f"[{run_id}] Wrote ambiguous-verdict annotation row.")
        return

    # Default single-call branch.
    discovery_config = DiscoveryConfig() if scope in {"comparison", "champion_candidate"} else None
    metadata = RunResultMetadata(
        run_id=run_id,
        project="civic_shout_action_rate_increase",
        comparison_group=comparison_group,
        scope=scope,  # type: ignore[arg-type]
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=sha,
        data_fingerprint="",
        outcome_version="v1",
    )
    extra_kwargs = _build_extra_harness_kwargs(
        scope=scope,
        bootstrap_n_resamples=bootstrap_n_resamples,
        model=model,
        lr_warm_start=lr_warm_start,
        lr_skip_audit=lr_skip_audit,
        lr_fold_backend=lr_fold_backend,
        lr_skip_heavy_diagnostics=lr_skip_heavy_diagnostics,
    )

    try:
        out = run_record(
            metadata=metadata,
            data=joined,
            feature_cols=_FEATURE_COLS,
            outcome_variable="actioned_24h",
            model_cfg=_build_model_cfg(model),
            sample_frac=sample_frac,
            sample_seed=sample_seed,
            harness_kwargs=extra_kwargs,
            eda_config=EDAConfig(),
            discovery_config=discovery_config,
            project_root=project_root,
        )
    except Exception as e:
        try:
            write_error_traceback(run_dir=run_dir, error=e)
        except Exception as write_err:
            print(f"WARN: failed to write error.txt: {write_err}", file=sys.stderr)
        raise

    _post_run_record_artifacts(out, project_root)


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
    parser.add_argument(
        "--lr-warm-start",
        action="store_true",
        default=False,
        dest="lr_warm_start",
        help=(
            "Enable warm-start across walk-forward folds for LR (sequential execution only). "
            "Each fold k+1 initializes from fold k's fitted coefficients. "
            "Activating this flag sets fold_n_jobs=1 automatically. "
            "Off by default; folds remain parallel in the default path."
        ),
    )
    parser.add_argument(
        "--lr-skip-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="lr_skip_audit",
        help=(
            "Skip _assert_no_future_leak audit entirely for LR runs (default: True). "
            "Safe when LGBM ran first against the same data (audit is model-agnostic). "
            "Use --no-lr-skip-audit to re-enable the audit."
        ),
    )
    parser.add_argument(
        "--lr-fold-backend",
        choices=["threads", "processes"],
        default="threads",
        dest="lr_fold_backend",
        help=(
            "Joblib backend for LR fold parallelism. "
            "'threads' (default): lower overhead, GIL may limit CPU for non-BLAS ops. "
            "'processes' (loky): no GIL contention but incurs pickling overhead. "
            "Run 1pct probe with both and pick the faster one."
        ),
    )
    parser.add_argument(
        "--lr-skip-heavy-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="lr_skip_heavy_diagnostics",
        help=(
            "Skip PSI, per-tenure PSI, cell balance, and confound overlap diagnostics for LR "
            "runs in comparison/comparison_fast scopes (default: True). champion_candidate and "
            "smoke always run full diagnostics regardless of this flag. "
            "Use --no-lr-skip-heavy-diagnostics to re-enable."
        ),
    )
    args = parser.parse_args()

    if args.auto_promote and args.scope != "fast_iter":
        parser.error("--auto-promote requires --scope fast_iter")

    if args.comparison_fast_first and args.scope != "comparison_fast":
        parser.error("--comparison-fast-first requires --scope comparison_fast")

    if args.profile and (args.comparison_fast_first or args.auto_promote):
        parser.error("--profile cannot be combined with --comparison-fast-first or --auto-promote")

    main(
        run_id=args.run_id,
        sample_frac=args.sample_frac,
        profile=args.profile,
        bootstrap_n_resamples=args.bootstrap_n_resamples,
        scope=args.scope,
        auto_promote=args.auto_promote,
        comparison_fast_first=args.comparison_fast_first,
        model=args.model,
        lr_warm_start=args.lr_warm_start,
        lr_skip_audit=args.lr_skip_audit,
        lr_fold_backend=args.lr_fold_backend,
        lr_skip_heavy_diagnostics=args.lr_skip_heavy_diagnostics,
    )
