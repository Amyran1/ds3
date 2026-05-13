from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from libs.responses import (
    HarnessArtifact,
    HarnessDataProfile,
    HarnessParallelism,
    HarnessPerformance,
    HarnessResponse,
    HarnessStageTiming,
    HarnessSummary,
)

# rough headline; baseline_metric_value reflects actual baseline-on-raw-features measured per-run
_BASELINE_R2 = 0.60


def harness(
    *,
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    ml_model_config: dict[str, Any],
    sample_frac: float | None = None,
    predictions_dir: str | None = None,
) -> HarnessResponse:
    """5-fold CV Ridge regression with R² scoring. Writes predictions.parquet if predictions_dir set."""
    t0 = time.monotonic()

    stage_timings: list[HarnessStageTiming] = []
    n_rows_source = data.height
    if sample_frac is not None and sample_frac < 1.0:
        n = max(50, int(data.height * sample_frac))
        data = data.sample(n=n, seed=ml_model_config.get("seed", 0), with_replacement=False)
    is_full_run = sample_frac is None or sample_frac >= 1.0
    stage_timings.append(HarnessStageTiming(stage="sample", seconds=time.monotonic() - t0))

    t1 = time.monotonic()
    X = data.select(feature_cols).to_numpy()
    y = data.select(outcome_variable).to_numpy().ravel()
    stage_timings.append(HarnessStageTiming(stage="prepare", seconds=time.monotonic() - t1))

    t2 = time.monotonic()
    kf = KFold(n_splits=5, shuffle=True, random_state=ml_model_config.get("seed", 0))
    alpha = float(ml_model_config.get("alpha", 1.0))
    fold_r2: list[float] = []
    y_pred_full = np.zeros_like(y, dtype=float)
    fold_ids = np.zeros_like(y, dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        model = Ridge(alpha=alpha, random_state=ml_model_config.get("seed", 0))
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        y_pred_full[test_idx] = preds
        fold_ids[test_idx] = fold_idx
        fold_r2.append(r2_score(y[test_idx], preds))
    stage_timings.append(HarnessStageTiming(stage="fit_predict", seconds=time.monotonic() - t2))

    t3 = time.monotonic()
    primary_r2 = float(np.mean(fold_r2))
    stage_timings.append(HarnessStageTiming(stage="score", seconds=time.monotonic() - t3))

    artifacts: list[HarnessArtifact] = []
    if predictions_dir is not None:
        t4 = time.monotonic()
        out_dir = Path(predictions_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / "predictions.parquet"
        pl.DataFrame({"y_true": y, "y_pred": y_pred_full, "fold": fold_ids}).write_parquet(
            pred_path
        )
        artifacts.append(
            HarnessArtifact(
                kind="predictions",
                path=(
                    str(pred_path.relative_to(Path.cwd()))
                    if pred_path.is_absolute()
                    else str(pred_path)
                ),
                bytes=pred_path.stat().st_size,
            )
        )
        stage_timings.append(
            HarnessStageTiming(stage="write_predictions", seconds=time.monotonic() - t4)
        )

    total = time.monotonic() - t0

    return HarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="r2",
            primary_metric_value=primary_r2,
            baseline_metric_value=_BASELINE_R2,
            lift_vs_baseline=primary_r2 - _BASELINE_R2,
            secondary_metrics={"r2_std": float(np.std(fold_r2))},
            direction="higher_is_better",
        ),
        performance=HarnessPerformance(
            total_seconds=total,
            parallelism=HarnessParallelism(outer_n_jobs=1, inner_threads=1),
            stage_timings=stage_timings,
        ),
        data_profile=HarnessDataProfile(
            n_rows=data.height,
            n_rows_source=n_rows_source,
            sample_frac=(sample_frac if sample_frac is not None else 1.0),
            is_full_run=is_full_run,
            n_features=len(feature_cols),
        ),
        artifacts=artifacts,
        fingerprint="",
    )
