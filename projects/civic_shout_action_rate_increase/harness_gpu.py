from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import psutil
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict, Field
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from libs.perf_equivalence import compute_predictions_hash
from projects.civic_shout_action_rate_increase._gpu_bootstrap import (
    gpu_pooled_bootstrap,
    gpu_user_block_bootstrap,
)
from projects.civic_shout_action_rate_increase.harness import (
    _FORBIDDEN_COLS,
    _TENURE_LABELS,
    CivicShoutHarnessResponse,
    CivicShoutResultRow,
    ConfoundJoinError,
    HarnessArtifact,
    HarnessDataProfile,
    HarnessDiagnostic,
    HarnessMethod,
    HarnessMetric,
    HarnessMetrics,
    HarnessParallelism,
    HarnessPerformance,
    HarnessReproducibility,
    HarnessStageTiming,
    HarnessSummary,
    LeakageError,
    ML_Model_Config,
    ML_Model_Type,
    RunResultMetadata,
    RunResultRow,
    _assert_no_future_leak,
    _assign_tenure_bucket,
    _compute_cell_class_balance_diagnostic,
    _compute_confound_overlap_diagnostic,
    _compute_confound_scores_full_data,
    _compute_psi,
    _compute_residualized_auc_cross_fit,
    _compute_user_main_effect_lgbm,
    _per_tenure_residualized_auc,
    _quarterly_folds,
    _residualize_single_confound,
)
from projects.civic_shout_action_rate_increase.harness_cache import (
    HarnessCache,
    data_fingerprint,
    fold_train_fingerprint,
)

# ---------------------------------------------------------------------------
# Per-fold worker — GPU variant: per-fold bootstrap routed through MLX
# ---------------------------------------------------------------------------


def _run_one_fold_gpu(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    q_label: str,
    feature_cols: list[str],
    outcome_variable: str,
    ml_model_config: ML_Model_Config,
    inner_threads: int,
    confound_overlap_threshold: int,
    seed: int,
    bootstrap_n_jobs: int,
) -> dict[str, Any]:
    """Execute one quarterly fold with GPU-accelerated per-fold bootstrap CI."""
    lgbm_args = dict(ml_model_config.args)
    lgbm_args["n_jobs"] = inner_threads
    lgbm_args["random_state"] = seed
    lgbm_args["verbose"] = -1

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df[outcome_variable].to_numpy().astype(np.float32)
    X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test = test_df[outcome_variable].to_numpy().astype(np.float32)

    c_u_test = test_df["user_prior_engagement_score"].to_numpy()
    c_e_test = test_df["email_popularity_score"].to_numpy()

    model = LGBMClassifier(**lgbm_args)
    model.fit(X_train, y_train)

    s_test = model.predict_proba(X_test)[:, 1]
    s_train = model.predict_proba(X_train)[:, 1]

    c_u_main_effect_test = _compute_user_main_effect_lgbm(
        train_df,
        test_df,
        feature_cols,
        ml_model_config,
        outcome_variable,
        seed=seed,
        inner_threads=inner_threads,
    )

    auc_residualized, auc_dir_a, auc_dir_b, s_resid_xfit = _compute_residualized_auc_cross_fit(
        y_test, s_test, c_u_main_effect_test, c_e_test, seed=seed
    )

    auc_residualized_old, _, _, s_resid_old = _compute_residualized_auc_cross_fit(
        y_test, s_test, c_u_test, c_e_test, seed=seed
    )

    user_ids_test = test_df["user_id"].to_numpy()

    # GPU-accelerated per-fold bootstrap (swap 1 of 2)
    boot_aucs = gpu_user_block_bootstrap(
        y_test,
        s_resid_xfit,
        user_ids_test,
        n_boot=500,
        seed=seed,
    )
    ci_low = float(np.quantile(boot_aucs, 0.025))
    ci_high = float(np.quantile(boot_aucs, 0.975))

    fold_overlap_diag = _compute_confound_overlap_diagnostic(
        fold_label=q_label,
        c_u_test=c_u_main_effect_test,
        c_e_test=c_e_test,
        y_test=y_test,
        min_positives_threshold=confound_overlap_threshold,
    )

    pred_rows = test_df.with_columns(
        [
            pl.Series("score", s_test),
            pl.Series("score_residualized", s_resid_xfit),
            pl.lit(q_label).alias("fold"),
            pl.Series(
                "row_loss",
                (
                    -y_test * np.log(np.clip(s_test, 1e-7, 1 - 1e-7))
                    - (1 - y_test) * np.log(np.clip(1 - s_test, 1e-7, 1 - 1e-7))
                ),
            ),
        ]
    )

    return {
        "q_label": q_label,
        "auc_residualized": auc_residualized,
        "auc_residualized_old": auc_residualized_old,
        "auc_dir_a": auc_dir_a,
        "auc_dir_b": auc_dir_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_test": y_test,
        "s_test": s_test,
        "s_train": s_train,
        "s_resid": s_resid_xfit,
        "s_resid_old": s_resid_old,
        "c_u_test": c_u_test,
        "c_u_main_effect_test": c_u_main_effect_test,
        "c_e_test": c_e_test,
        "user_ids": user_ids_test,
        "test_df": test_df,
        "train_df": train_df,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fold_overlap_diag": fold_overlap_diag,
        "pred_rows": pred_rows,
    }


# ---------------------------------------------------------------------------
# Public GPU harness — mirrors harness() with two bootstrap swaps
# ---------------------------------------------------------------------------


def harness(
    data: pl.DataFrame,
    feature_cols: list[str],
    ml_model_config: ML_Model_Config,
    outcome_variable: str = "actioned_24h",
    *,
    sample_frac: float | None = None,
    sample_seed: int | None = None,
    predictions_dir: str | None = None,
    fold_n_jobs: int = 8,
    inner_threads: int | None = None,
    disable_cache: bool = False,
) -> CivicShoutHarnessResponse:
    """GPU-accelerated harness: mirrors harness.py with MLX bootstrap (sort-based AUC).

    Both bootstrap calls (_user_block_bootstrap and _pooled_user_block_bootstrap_mean_of_folds)
    are routed through gpu_user_block_bootstrap and gpu_pooled_bootstrap respectively.
    All other logic is identical to harness.py.
    """
    wall_start = time.perf_counter()
    stage_timings: list[HarnessStageTiming] = []
    diagnostics: list[HarnessDiagnostic] = []
    artifacts: list[HarnessArtifact] = []

    cpu_count = os.cpu_count() or 8
    _parallel_folds = fold_n_jobs != 1
    _inner_threads: int = (
        inner_threads
        if inner_threads is not None
        else (3 if _parallel_folds else max(cpu_count - 1, 1))
    )
    bootstrap_n_jobs = 3 if _parallel_folds else -1

    _is_smoke = sample_frac is not None and sample_frac <= 0.01
    _confound_overlap_threshold = 25 if _is_smoke else 100
    _class_balance_threshold = 50 if _is_smoke else 5

    # ------------------------------------------------------------------
    # Stage 1: column_selection
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    required_cols = {"user_id", "email_id", "date_sent", outcome_variable}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    forbidden_in_features = _FORBIDDEN_COLS & set(feature_cols)
    if forbidden_in_features:
        raise ValueError(
            f"Feature columns contain forbidden columns (F03 — post-treatment proxies): "
            f"{forbidden_in_features}. Remove them before calling harness()."
        )

    if outcome_variable != "actioned_24h":
        raise ValueError(
            f"harness v1 only supports outcome_variable='actioned_24h'; got '{outcome_variable}'"
        )

    all_needed = list(required_cols) + [c for c in feature_cols if c not in required_cols]
    work_df = data.select([c for c in all_needed if c in data.columns])
    work_df = work_df.with_columns(pl.col("actioned_24h").cast(pl.Int8))

    stage_timings.append(
        HarnessStageTiming(
            stage="column_selection", seconds=time.perf_counter() - t0, owner="harness"
        )
    )

    # ------------------------------------------------------------------
    # Stage 2: data_conversion
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    population_n_rows = len(work_df)

    _harness_cache = HarnessCache()
    _confound_fp = data_fingerprint(work_df)

    _cached_confound = None if disable_cache else _harness_cache.get_confound_scores(_confound_fp)
    if _cached_confound is not None:
        confound_df = _cached_confound
        diagnostics.append(
            HarnessDiagnostic(
                name="confound_cache_hit",
                severity="info",
                message="Confound scores loaded from disk cache.",
                values={"hit": True, "fingerprint": _confound_fp[:8]},
            )
        )
    else:
        confound_df = _compute_confound_scores_full_data(work_df)
        if not disable_cache:
            _harness_cache.put_confound_scores(_confound_fp, confound_df)
        diagnostics.append(
            HarnessDiagnostic(
                name="confound_cache_hit",
                severity="info",
                message="Confound scores computed and written to disk cache.",
                values={"hit": False, "fingerprint": _confound_fp[:8]},
            )
        )

    _confound_joined = work_df.join(
        confound_df.select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "user_prior_engagement_score",
                "email_popularity_score",
            ]
        ),
        on=["user_id", "email_id", "date_sent"],
        how="left",
    )
    _null_user = _confound_joined["user_prior_engagement_score"].is_null().sum()
    _null_email = _confound_joined["email_popularity_score"].is_null().sum()
    if _null_user > 0 or _null_email > 0:
        raise ConfoundJoinError(
            f"Confound left-join left {_null_user} null user_prior_engagement_score "
            f"and {_null_email} null email_popularity_score rows. "
            "Some work_df (user_id, email_id, date_sent) keys have no match in confound_df."
        )

    full_work_df = _confound_joined.with_columns(
        pl.col("user_prior_engagement_score").fill_null(0.0),
        pl.col("email_popularity_score").fill_null(0.5),
    )

    if sample_frac is not None and sample_seed is not None:
        rng_sample = np.random.default_rng(sample_seed)
        n_sample = max(1, int(len(work_df) * sample_frac))
        sample_indices = rng_sample.choice(len(work_df), size=n_sample, replace=False)
        sample_indices_sorted = np.sort(sample_indices)
        work_df = work_df[sample_indices_sorted]
    elif sample_frac is not None:
        raise ValueError("sample_frac requires sample_seed")

    work_df = work_df.join(
        full_work_df.select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "user_prior_engagement_score",
                "email_popularity_score",
            ]
        ),
        on=["user_id", "email_id", "date_sent"],
        how="left",
    ).with_columns(
        pl.col("user_prior_engagement_score").fill_null(0.0),
        pl.col("email_popularity_score").fill_null(0.5),
    )

    work_df = work_df.sort("date_sent")

    stage_timings.append(
        HarnessStageTiming(
            stage="data_conversion", seconds=time.perf_counter() - t0, owner="harness"
        )
    )

    # ------------------------------------------------------------------
    # Stage 3: split_generation
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    work_df = _assign_tenure_bucket(work_df)

    folds = _quarterly_folds(work_df)
    if len(folds) == 0:
        raise ValueError("No valid quarterly walk-forward folds found. Check date range in data.")

    thin_train_threshold = 50_000
    for q_label, train_df, _ in folds:
        if len(train_df) < thin_train_threshold:
            diagnostics.append(
                HarnessDiagnostic(
                    name="thin_train_fold",
                    severity="warning",
                    message=f"Fold {q_label} train has {len(train_df)} rows < {thin_train_threshold} threshold.",
                    values={"fold": q_label, "train_rows": len(train_df)},
                )
            )

    stage_timings.append(
        HarnessStageTiming(
            stage="split_generation",
            seconds=time.perf_counter() - t0,
            owner="harness",
            calls=len(folds),
        )
    )

    # ------------------------------------------------------------------
    # Stages 4–7: fit / predict / residualize_scores / score per fold
    # ------------------------------------------------------------------

    leak_diag_user: HarnessDiagnostic | None = None
    leak_diag_email: HarnessDiagnostic | None = None
    try:
        leak_diag_user = _assert_no_future_leak(
            full_work_df, "user_prior_engagement_score", "date_sent", "user_id"
        )
        leak_diag_email = _assert_no_future_leak(
            full_work_df, "email_popularity_score", "date_sent", "email_id"
        )
    except LeakageError:
        raise

    rng_folds = np.random.default_rng(sample_seed if sample_seed is not None else 0)
    n_folds = len(folds)
    fold_seeds = rng_folds.integers(0, 2**32 - 1, size=n_folds)

    fold_pairs = [(q_label, train_df, test_df) for q_label, train_df, test_df in folds]

    import threading

    cpu_util_samples: list[float] = []
    proc = psutil.Process()

    def _sample_cpu() -> None:
        for _ in range(6):
            time.sleep(1.0)
            try:
                cpu_util_samples.append(proc.cpu_percent(interval=None))
            except Exception:
                pass

    cpu_thread = threading.Thread(target=_sample_cpu, daemon=True)
    cpu_thread.start()

    t_fold_wall_start = time.perf_counter()

    _effective_fold_jobs = fold_n_jobs if fold_n_jobs != 1 else 1
    raw_fold_results: list[dict[str, Any]] = Parallel(n_jobs=_effective_fold_jobs, backend="loky")(
        delayed(_run_one_fold_gpu)(
            train_df,
            test_df,
            q_label,
            feature_cols,
            outcome_variable,
            ml_model_config,
            _inner_threads,
            _confound_overlap_threshold,
            int(fold_seeds[k]),
            bootstrap_n_jobs,
        )
        for k, (q_label, train_df, test_df) in enumerate(fold_pairs)
    )  # type: ignore[assignment]

    t_fold_wall_total = time.perf_counter() - t_fold_wall_start
    cpu_thread.join(timeout=2.0)

    fold_results: list[dict[str, Any]] = raw_fold_results
    all_rows: list[pl.DataFrame] = [r["pred_rows"] for r in fold_results]
    all_train_scores: list[np.ndarray] = [r["s_train"] for r in fold_results]

    for r in fold_results:
        diagnostics.append(r["fold_overlap_diag"])

    stage_timings.append(
        HarnessStageTiming(stage="fit", seconds=t_fold_wall_total, owner="model", calls=n_folds)
    )
    stage_timings.append(
        HarnessStageTiming(stage="predict", seconds=0.0, owner="model", calls=n_folds)
    )
    stage_timings.append(
        HarnessStageTiming(stage="score", seconds=0.0, owner="harness", calls=n_folds)
    )

    if leak_diag_user:
        diagnostics.append(leak_diag_user)
    if leak_diag_email:
        diagnostics.append(leak_diag_email)

    # ------------------------------------------------------------------
    # Stage 8: aggregate
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    fold_aucs = [r["auc_residualized"] for r in fold_results]
    fold_aucs_old = [r["auc_residualized_old"] for r in fold_results]
    fold_labels = [r["q_label"] for r in fold_results]
    primary_value = float(np.mean(fold_aucs))
    old_primary_value = float(np.mean(fold_aucs_old))
    fold_std = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0

    k = len(fold_aucs)
    if k > 1:
        # GPU-accelerated pooled bootstrap (swap 2 of 2)
        primary_ci_low, primary_ci_high = gpu_pooled_bootstrap(fold_results, n_boot=500, seed=42)
    else:
        primary_ci_low = fold_results[0]["ci_low"]
        primary_ci_high = fold_results[0]["ci_high"]

    y_all = np.concatenate([r["y_test"] for r in fold_results])
    s_all = np.concatenate([r["s_test"] for r in fold_results])
    s_resid_all = np.concatenate([r["s_resid"] for r in fold_results])
    s_resid_old_all = np.concatenate([r["s_resid_old"] for r in fold_results])
    c_u_all = np.concatenate([r["c_u_test"] for r in fold_results])
    c_u_main_effect_all = np.concatenate([r["c_u_main_effect_test"] for r in fold_results])
    c_e_all = np.concatenate([r["c_e_test"] for r in fold_results])
    user_ids_all = np.concatenate([r["user_ids"] for r in fold_results])
    tenure_all = np.concatenate([r["test_df"]["tenure_bucket"].to_numpy() for r in fold_results])

    train_y_all = np.concatenate([r["train_df"]["actioned_24h"].to_numpy() for r in fold_results])
    s_train_all = np.concatenate(all_train_scores)

    roc_auc_raw = float(roc_auc_score(y_all, s_all)) if len(np.unique(y_all)) >= 2 else 0.5

    confound_combined = c_u_all + c_e_all
    roc_auc_confound_only = (
        float(roc_auc_score(y_all, confound_combined)) if len(np.unique(y_all)) >= 2 else 0.5
    )

    roc_auc_user_prior_only = _residualize_single_confound(y_all, s_all, c_u_all, seed=42)
    roc_auc_email_pop_only = _residualize_single_confound(y_all, s_all, c_e_all, seed=42)

    roc_auc_user_main_effect_only = _residualize_single_confound(
        y_all, s_all, c_u_main_effect_all, seed=42
    )

    pr_auc_residualized = (
        float(average_precision_score(y_all, s_resid_all)) if len(np.unique(y_all)) >= 2 else 0.0
    )

    tenure_aucs = _per_tenure_residualized_auc(
        y_all, s_resid_all, tenure_all, c_u_main_effect_all, c_e_all
    )

    all_test_dfs = pl.concat([r["test_df"] for r in fold_results])
    if "is_in_action_streak" in all_test_dfs.columns:
        non_streak_mask = (all_test_dfs["is_in_action_streak"].fill_null(False) == False).to_numpy()
        if non_streak_mask.sum() >= 50 and len(np.unique(y_all[non_streak_mask])) >= 2:
            roc_auc_non_streak = float(
                roc_auc_score(y_all[non_streak_mask], s_resid_all[non_streak_mask])
            )
        else:
            roc_auc_non_streak = None
    else:
        roc_auc_non_streak = None

    if "lifetime_actions_prior" in all_test_dfs.columns:
        cold_start_mask = (all_test_dfs["lifetime_actions_prior"].fill_null(0) == 0).to_numpy()
        if cold_start_mask.sum() >= 50 and len(np.unique(y_all[cold_start_mask])) >= 2:
            roc_auc_cold_start = float(
                roc_auc_score(y_all[cold_start_mask], s_resid_all[cold_start_mask])
            )
        else:
            roc_auc_cold_start = None
    else:
        roc_auc_cold_start = None

    all_test_dates = all_test_dfs["date_sent"]
    month_labels = (
        all_test_dfs.with_columns(
            (
                pl.col("date_sent").dt.year().cast(pl.Utf8)
                + "-"
                + pl.col("date_sent").dt.month().cast(pl.Utf8).str.zfill(2)
            ).alias("_month")
        )
        .get_column("_month")
        .to_numpy()
    )
    unique_months = np.unique(month_labels)
    monthly_aucs: list[float] = []
    secondary_monthly: list[HarnessMetric] = []
    for m in unique_months:
        mask = month_labels == m
        if mask.sum() < 20 or len(np.unique(y_all[mask])) < 2:
            continue
        try:
            mauc = float(roc_auc_score(y_all[mask], s_resid_all[mask]))
            monthly_aucs.append(mauc)
            secondary_monthly.append(
                HarnessMetric(
                    name=f"roc_auc_residualized_pair_calendar_month_{m}",
                    value=mauc,
                    direction="higher_is_better",
                )
            )
        except Exception:
            pass

    temporal_cv = (
        float(np.std(monthly_aucs) / np.mean(monthly_aucs))
        if monthly_aucs and np.mean(monthly_aucs) != 0
        else 0.0
    )

    psi_train_test = _compute_psi(s_train_all, s_all)

    tenure_psi_vals: list[float] = []
    for label in _TENURE_LABELS:
        mask = tenure_all == label
        if mask.sum() >= 20:
            psi = _compute_psi(s_all, s_all[mask])
            tenure_psi_vals.append(psi)
    max_tenure_psi = float(max(tenure_psi_vals)) if tenure_psi_vals else 0.0

    cell_balance_diag, excluded_cells = _compute_cell_class_balance_diagnostic(
        all_test_dfs=[r["test_df"] for r in fold_results],
        y_all=y_all,
        tenure_all=tenure_all,
        min_positives_threshold=_class_balance_threshold,
    )

    stratum_auc_vals = [
        v
        for label, v in tenure_aucs.items()
        if v is not None and not any(label == t for t, _ in excluded_cells)
    ]
    stratum_cv = (
        float(np.std(stratum_auc_vals) / np.mean(stratum_auc_vals))
        if len(stratum_auc_vals) >= 2 and np.mean(stratum_auc_vals) != 0
        else 0.0
    )

    confound_r = float(np.corrcoef(c_u_all, c_e_all)[0, 1])

    residual_collapsed = primary_value < 0.505

    positive_rate_test = float(np.mean(y_all))
    positive_rate_train = float(np.mean(train_y_all))

    n_test_users = len(np.unique(user_ids_all))
    n_test_emails = int(all_test_dfs["email_id"].n_unique())
    n_test_rows = len(y_all)

    fold_overlap_diags = [d for d in diagnostics if d.name.startswith("confound_overlap_")]
    _overlap_min_positives = [
        d.values.get("min_positives_per_cell", 0)
        for d in fold_overlap_diags
        if isinstance(d.values.get("min_positives_per_cell"), int)
    ]
    _overlap_min = min(_overlap_min_positives) if _overlap_min_positives else None
    _overlap_n_cells_below = sum(
        d.values.get("n_cells_below_threshold", 0)
        for d in fold_overlap_diags
        if isinstance(d.values.get("n_cells_below_threshold"), int)
    )
    _overlap_summary_sev: Literal["info", "warning"] = (
        "warning" if any(d.severity == "warning" for d in fold_overlap_diags) else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="confound_overlap_min_across_folds",
            severity=_overlap_summary_sev,
            message=(
                f"Confound overlap summary: min positives/cell across folds = {_overlap_min}, "
                f"total cells below threshold = {_overlap_n_cells_below}"
            ),
            values={
                "min_positives_per_cell": _overlap_min,
                "n_cells_below_threshold": _overlap_n_cells_below,
                "threshold": _confound_overlap_threshold,
            },
        )
    )

    psi_sev: Literal["info", "warning", "error"] = (
        "error" if psi_train_test > 0.25 else "warning" if psi_train_test > 0.10 else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="score_psi_train_test",
            severity=psi_sev,
            message=f"Score PSI between train and test: {psi_train_test:.4f}",
            values={"psi": psi_train_test},
        )
    )

    tenure_psi_sev: Literal["info", "warning", "error"] = (
        "warning" if max_tenure_psi > 0.15 else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="score_psi_by_tenure_bucket",
            severity=tenure_psi_sev,
            message=f"Max per-tenure PSI: {max_tenure_psi:.4f}",
            values={"max_psi": max_tenure_psi},
        )
    )

    stratum_sev: Literal["info", "warning", "error"] = "warning" if stratum_cv > 0.5 else "info"
    diagnostics.append(
        HarnessDiagnostic(
            name="per_stratum_auc_dispersion",
            severity=stratum_sev,
            message=f"CV of residualized AUC across tenure buckets: {stratum_cv:.4f}",
            values={"cv": stratum_cv},
        )
    )

    collapse_sev: Literal["info", "warning", "error"] = "warning" if residual_collapsed else "info"
    diagnostics.append(
        HarnessDiagnostic(
            name="residual_collapse_check",
            severity=collapse_sev,
            message=f"Residualized AUC {'collapsed' if residual_collapsed else 'above floor'}: {primary_value:.4f}",
            values={"collapsed": residual_collapsed, "value": primary_value},
        )
    )

    confound_corr_sev: Literal["info", "warning", "error"] = (
        "warning" if abs(confound_r) > 0.5 else "info"
    )
    diagnostics.append(
        HarnessDiagnostic(
            name="confound_correlation_pearson",
            severity=confound_corr_sev,
            message=f"Pearson r between user_prior and email_popularity scores: {confound_r:.4f}",
            values={"pearson_r": confound_r},
        )
    )

    temporal_sev: Literal["info", "warning", "error"] = "warning" if temporal_cv > 0.30 else "info"
    diagnostics.append(
        HarnessDiagnostic(
            name="temporal_drift_check",
            severity=temporal_sev,
            message=f"Calendar-month AUC CV: {temporal_cv:.4f}",
            values={"cv": temporal_cv, "n_months": len(monthly_aucs)},
        )
    )

    diagnostics.append(cell_balance_diag)

    cpu_util = float(np.mean(cpu_util_samples)) if cpu_util_samples else None
    if cpu_util is not None and cpu_util < 50:
        diagnostics.append(
            HarnessDiagnostic(
                name="cpu_underutilization",
                severity="warning",
                message=f"CPU utilization {cpu_util:.1f}% below 50% threshold",
                values={"cpu_utilization_pct": cpu_util},
            )
        )

    stage_timings.append(
        HarnessStageTiming(stage="aggregate", seconds=time.perf_counter() - t0, owner="harness")
    )

    # ------------------------------------------------------------------
    # Stage 9: persist_predictions
    # ------------------------------------------------------------------
    if predictions_dir is not None and all_rows:
        pred_df = pl.concat(all_rows).select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "score",
                "score_residualized",
                pl.col("actioned_24h").alias("label"),
                "row_loss",
                "fold",
            ]
        )
        pred_path = Path(predictions_dir) / "predictions.parquet"
        pred_df.write_parquet(pred_path)
        artifacts.append(
            HarnessArtifact(
                name="predictions",
                kind="predictions",
                uri=str(pred_path),
                description="Per-row predictions with score, residualized score, label, row_loss, and fold.",
            )
        )

    # ------------------------------------------------------------------
    # Build secondary metrics
    # ------------------------------------------------------------------
    secondary_metrics: list[HarnessMetric] = [
        HarnessMetric(name="roc_auc_raw_pair", value=roc_auc_raw, direction="higher_is_better"),
        HarnessMetric(
            name="roc_auc_confound_only_pair",
            value=roc_auc_confound_only,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_user_prior_pair",
            value=roc_auc_user_prior_only,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_user_prior_x_email_popularity_pair",
            value=old_primary_value,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_user_main_effect_pair",
            value=roc_auc_user_main_effect_only,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
            value=primary_value,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="roc_auc_residualized_email_popularity_pair",
            value=roc_auc_email_pop_only,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="pr_auc_residualized_user_prior_x_email_popularity_pair",
            value=pr_auc_residualized,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="primary_metric_temporal_cv", value=temporal_cv, direction="lower_is_better"
        ),
        HarnessMetric(name="n_test_users", value=float(n_test_users), direction="higher_is_better"),
        HarnessMetric(
            name="n_test_emails", value=float(n_test_emails), direction="higher_is_better"
        ),
        HarnessMetric(name="n_test_rows", value=float(n_test_rows), direction="higher_is_better"),
        HarnessMetric(
            name="pooled_positive_rate_train",
            value=positive_rate_train,
            direction="higher_is_better",
        ),
        HarnessMetric(
            name="pooled_positive_rate_test", value=positive_rate_test, direction="higher_is_better"
        ),
    ]

    if roc_auc_non_streak is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_non_streak_users_pair",
                value=roc_auc_non_streak,
                direction="higher_is_better",
            )
        )

    if roc_auc_cold_start is not None:
        secondary_metrics.append(
            HarnessMetric(
                name="roc_auc_residualized_cold_start_pair",
                value=roc_auc_cold_start,
                direction="higher_is_better",
            )
        )

    tenure_name_map = {
        "0-7d": "0_7d",
        "8-30d": "8_30d",
        "31-90d": "31_90d",
        "91-180d": "91_180d",
        "181-365d": "181_365d",
        "1-2yr": "1_2yr",
        "2yr+": "2yr_plus",
    }
    for label, val in tenure_aucs.items():
        if val is not None:
            slug = tenure_name_map.get(label, label.replace("-", "_").replace("+", "_plus"))
            secondary_metrics.append(
                HarnessMetric(
                    name=f"roc_auc_residualized_{slug}_pair",
                    value=val,
                    direction="higher_is_better",
                )
            )

    secondary_metrics.extend(secondary_monthly)

    by_fold_metrics = [
        HarnessMetric(
            name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
            value=r["auc_residualized"],
            split=r["q_label"],
            fold=i,
            direction="higher_is_better",
            ci_low=r["ci_low"],
            ci_high=r["ci_high"],
        )
        for i, r in enumerate(fold_results)
    ]

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    total_seconds = time.perf_counter() - wall_start
    n_rows_evaluated = len(work_df)

    harness_secs = sum(st.seconds for st in stage_timings if st.owner == "harness")
    model_secs = sum(st.seconds for st in stage_timings if st.owner == "model")

    bottleneck = max(stage_timings, key=lambda st: st.seconds).stage

    if predictions_dir is not None:
        _pred_path = Path(predictions_dir) / "predictions.parquet"
        _fingerprint = hashlib.sha256(
            compute_predictions_hash(_pred_path).encode()
            + str(round(primary_value, 10)).encode()
            + str(round(primary_value - 0.5, 10)).encode()
        ).hexdigest()[:16]
    else:
        _fingerprint = ""

    response = CivicShoutHarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
            primary_metric_value=primary_value,
            baseline_metric_value=0.5,
            lift_vs_baseline=primary_value - 0.5,
            result_notes=[
                f"Quarterly walk-forward: {len(fold_results)} valid folds over {fold_labels[0]}–{fold_labels[-1]}.",
                f"Fold AUCs: {[round(a, 4) for a in fold_aucs]}",
                f"Old RFM-floor secondary: {round(old_primary_value, 4)} (user_prior_x_email_popularity).",
                "User main effect computed per-fold via LightGBM on train split (train/test separation guarantees no leak).",
                "Bootstrap: GPU-accelerated MLX sort-based AUC (gpu_user_block_bootstrap + gpu_pooled_bootstrap).",
            ],
        ),
        method=HarnessMethod(
            name="quarterly_walk_forward_residualized_auc_gpu",
            metric_direction="higher_is_better",
            n_splits=len(fold_results),
            split_strategy="quarterly_cumulative_train_expanding",
            outcome_transform=None,
        ),
        data=HarnessDataProfile(
            n_rows=n_rows_evaluated,
            n_features=len(feature_cols),
            feature_cols=feature_cols,
            outcome_variable=outcome_variable,
            train_rows=sum(r["n_train"] for r in fold_results),
            validation_rows=sum(r["n_test"] for r in fold_results),
            sample_frac=sample_frac,
            sample_seed=sample_seed,
            sample_strategy="random_row_sample" if sample_frac is not None else None,
            is_full_run=sample_frac is None or sample_frac >= 1.0,
            population_n_rows=population_n_rows if sample_frac is not None else None,
            column_mapping={"group": "user_id"},
            notes=["Confound scores computed on full data before sampling."],
        ),
        metrics=HarnessMetrics(
            primary=HarnessMetric(
                name="roc_auc_residualized_user_main_effect_x_email_popularity_pair",
                value=primary_value,
                direction="higher_is_better",
                std=fold_std,
                ci_low=primary_ci_low,
                ci_high=primary_ci_high,
            ),
            by_fold=by_fold_metrics,
            secondary=secondary_metrics,
            baseline=[
                HarnessMetric(
                    name="roc_auc_random", value=0.5, split="baseline", direction="higher_is_better"
                ),
            ],
        ),
        performance=HarnessPerformance(
            total_seconds=total_seconds,
            stage_timings=stage_timings,
            bottleneck_stage=bottleneck,
            rows_per_second=n_rows_evaluated / total_seconds if total_seconds > 0 else None,
            folds_per_second=len(fold_results) / total_seconds if total_seconds > 0 else None,
            harness_owned_seconds=harness_secs,
            model_owned_seconds=model_secs,
            peak_memory_mb=None,
            parallelism=HarnessParallelism(
                outer_n_jobs=_effective_fold_jobs,
                inner_threads=_inner_threads,
                backend="loky" if _effective_fold_jobs != 1 else "sequential",
                cpu_count_observed=cpu_count,
                cpu_utilization_pct=cpu_util,
            ),
            hardware="apple_mlx_gpu",
            sample_size=n_rows_evaluated,
        ),
        diagnostics=diagnostics,
        artifacts=artifacts,
        reproducibility=HarnessReproducibility(
            seed=sample_seed,
            ml_model_type=ml_model_config.ml_model_type.value,
            ml_model_args=ml_model_config.args,
            data_fingerprint=None,
        ),
        fingerprint=_fingerprint,
    )

    return response
