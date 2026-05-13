from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from libs.responses import (
    DiscoveryConfig,
    DiscoveryPerformance,
    DiscoveryReproducibility,
    DiscoveryResponse,
    DiscoveryScores,
    FeatureGapFinding,
    HarnessResponse,
    HarnessStageTiming,
)

_LIFT_THRESHOLD = 1.3
_T0_THRESHOLD = 2.0
_T1_THRESHOLD = 1.5
_N_QUANTILES = 10


def _tier(ratio: float) -> str:
    if ratio >= _T0_THRESHOLD:
        return "T0"
    if ratio >= _T1_THRESHOLD:
        return "T1"
    return "T2"


def _load_predictions(harness_response: HarnessResponse, project_root: Path) -> pl.DataFrame | None:
    for artifact in harness_response.artifacts:
        if artifact.kind == "predictions":
            p = Path(artifact.path)
            if not p.is_absolute():
                p = project_root / p
            if p.exists():
                return pl.read_parquet(p)
    return None


def discover_gaps(
    *,
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    prediction_cols: dict[str, str],
    harness_response: HarnessResponse,
    discovery_config: DiscoveryConfig,
    outcome_version: str = "v1",
) -> DiscoveryResponse:
    t0 = time.monotonic()
    stage_timings: list[HarnessStageTiming] = []

    # Resolve project root relative to this file so predictions path resolves correctly.
    project_root = Path(__file__).resolve().parents[2]

    t_load = time.monotonic()
    preds_df = _load_predictions(harness_response, project_root)
    if preds_df is None:
        # Fallback: champion predictions at the known fixed path.
        fallback = (
            project_root
            / "projects/california_housing_demo/runs/run_01/artifacts/predictions.parquet"
        )
        if fallback.exists():
            preds_df = pl.read_parquet(fallback)

    stage_timings.append(
        HarnessStageTiming(stage="load_predictions", seconds=time.monotonic() - t_load)
    )

    if preds_df is None or data.height == 0:
        return DiscoveryResponse(
            findings=[],
            scores=DiscoveryScores(),
            reproducibility=DiscoveryReproducibility(),
            performance=DiscoveryPerformance(
                total_seconds=time.monotonic() - t0,
                stage_timings=stage_timings,
            ),
            outcome_version=outcome_version,
        )

    t_slice = time.monotonic()

    # The harness may sample data; predictions rows align 1:1 with the sampled frame.
    # Use the smaller of the two to stay safe, then slice data to match.
    n = min(data.height, preds_df.height)
    feat_df = data.slice(0, n).select(feature_cols)
    y_true = preds_df["y_true"].slice(0, n)
    y_pred = preds_df["y_pred"].slice(0, n)
    residuals = (y_true - y_pred).abs()
    overall_mae = residuals.mean()

    if overall_mae is None or overall_mae == 0.0:
        return DiscoveryResponse(
            findings=[],
            scores=DiscoveryScores(),
            reproducibility=DiscoveryReproducibility(),
            performance=DiscoveryPerformance(
                total_seconds=time.monotonic() - t0,
                stage_timings=stage_timings,
            ),
            outcome_version=outcome_version,
        )

    overall_mae_val = float(overall_mae)
    residuals_list = residuals.to_list()

    findings: list[FeatureGapFinding] = []

    for feat in feature_cols:
        feat_vals = feat_df[feat].to_list()
        pairs = sorted(zip(feat_vals, residuals_list), key=lambda x: x[0])
        chunk = max(1, len(pairs) // _N_QUANTILES)

        worst_mae = 0.0
        worst_q_idx = 0
        worst_q_min: float = pairs[0][0]
        worst_q_max: float = pairs[0][0]
        n_coverage = 0

        for q in range(_N_QUANTILES):
            start = q * chunk
            end = start + chunk if q < _N_QUANTILES - 1 else len(pairs)
            q_pairs = pairs[start:end]
            q_mae = sum(r for _, r in q_pairs) / len(q_pairs)
            if q_mae > worst_mae:
                worst_mae = q_mae
                worst_q_idx = q
                worst_q_min = q_pairs[0][0]
                worst_q_max = q_pairs[-1][0]
                n_coverage = len(q_pairs)

        ratio = worst_mae / overall_mae_val
        if ratio < _LIFT_THRESHOLD:
            continue

        tier = _tier(ratio)
        decile_label = f"q{worst_q_idx}"
        finding_id = f"slice_{feat}_{decile_label}"

        if worst_q_idx == _N_QUANTILES - 1:
            slice_def = f"{feat} >= {worst_q_min:.4g} (top decile)"
            desc = (
                f"Residuals in the top decile of `{feat}` (>= {worst_q_min:.4g}) are "
                f"{ratio:.2f}x the overall MAE — model struggles with high-{feat} examples."
            )
        elif worst_q_idx == 0:
            slice_def = f"{feat} <= {worst_q_max:.4g} (bottom decile)"
            desc = (
                f"Residuals in the bottom decile of `{feat}` (<= {worst_q_max:.4g}) are "
                f"{ratio:.2f}x the overall MAE — model struggles with low-{feat} examples."
            )
        else:
            slice_def = f"{feat} in [{worst_q_min:.4g}, {worst_q_max:.4g}] (decile {worst_q_idx})"
            desc = (
                f"Residuals in decile {worst_q_idx} of `{feat}` "
                f"[{worst_q_min:.4g}, {worst_q_max:.4g}] are "
                f"{ratio:.2f}x the overall MAE."
            )

        findings.append(
            FeatureGapFinding(
                id=finding_id,
                kind="residual_slice",
                slice_definition=slice_def,
                residual_lift=round(ratio, 4),
                severity=round(ratio, 4),
                priority=round(ratio, 4),
                coverage=round(n_coverage / n, 4),
                recommended_feature_family=f"{feat}_nonlinear",
                feature_plan_stub=desc,
            )
        )

    stage_timings.append(
        HarnessStageTiming(stage="slice_search", seconds=time.monotonic() - t_slice)
    )

    findings.sort(key=lambda f: f.residual_lift, reverse=True)
    top_k = discovery_config.top_k_findings
    findings = findings[:top_k]

    top_lift = findings[0].residual_lift if findings else 0.0
    n_warn = sum(1 for f in findings if f.residual_lift < _T1_THRESHOLD)
    n_err = sum(1 for f in findings if f.residual_lift >= _T1_THRESHOLD)
    health: str
    if top_lift >= _T0_THRESHOLD:
        health = "red"
    elif top_lift >= _T1_THRESHOLD:
        health = "yellow"
    else:
        health = "green"

    return DiscoveryResponse(
        findings=findings,
        scores=DiscoveryScores(
            top_slice_residual_lift=top_lift,
            priority_score_max=top_lift,
            overall_health=health,  # type: ignore[arg-type]
            n_findings_warning=n_warn,
            n_findings_error=n_err,
        ),
        reproducibility=DiscoveryReproducibility(),
        performance=DiscoveryPerformance(
            total_seconds=time.monotonic() - t0,
            stage_timings=stage_timings,
        ),
        outcome_version=outcome_version,
    )
