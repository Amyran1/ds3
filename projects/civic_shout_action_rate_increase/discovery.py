"""Project-level gap-finder for civic_shout_action_rate_increase."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor

from libs.responses import (
    DiscoveryConfig,
    DiscoveryPerformance,
    DiscoveryReproducibility,
    DiscoveryResponse,
    DiscoveryResultRow,
    DiscoveryScores,
    FeatureGapFinding,
    HarnessResponse,
    HarnessStageTiming,
    RunResultMetadata,
)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_DISCOVERY_METHOD_REGISTRY: dict[str, Callable[..., list[FeatureGapFinding]]] = {}


def register_discovery_method(
    name: str,
) -> Callable[[Callable[..., list[FeatureGapFinding]]], Callable[..., list[FeatureGapFinding]]]:
    def decorator(
        fn: Callable[..., list[FeatureGapFinding]],
    ) -> Callable[..., list[FeatureGapFinding]]:
        _DISCOVERY_METHOD_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Project subclass
# ---------------------------------------------------------------------------


class CivicShoutDiscoveryResponse(DiscoveryResponse):
    """DiscoveryResponse with civic_shout projection columns."""

    n_label_noise_candidates: int = 0
    top_slice_loss_uplift_pct: float = 0.0

    def to_discovery_summary_row(self, metadata: RunResultMetadata) -> CivicShoutDiscoveryResultRow:
        base = super().to_discovery_summary_row(metadata).model_dump()
        base["n_label_noise_candidates"] = self.n_label_noise_candidates
        base["top_slice_loss_uplift_pct"] = self.top_slice_loss_uplift_pct
        return CivicShoutDiscoveryResultRow(**base)


class CivicShoutDiscoveryResultRow(DiscoveryResultRow):
    """DiscoveryResultRow with civic_shout-specific columns."""

    n_label_noise_candidates: int = 0
    top_slice_loss_uplift_pct: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PREDICTIONS_REQUIRED_COLS = {
    "user_id",
    "email_id",
    "date_sent",
    "score",
    "label",
    "row_loss",
    "fold",
}


def _load_predictions(harness_response: HarnessResponse) -> pl.DataFrame:
    """Load predictions parquet from the artifact URI recorded on the harness response."""
    pred_artifact = next(
        (
            a
            for a in harness_response.artifacts
            if getattr(a, "name", None) == "predictions"
            or getattr(a, "kind", None) == "predictions"
        ),
        None,
    )
    if pred_artifact is None:
        raise ValueError(
            "HarnessResponse has no 'predictions' artifact. "
            "Ensure harness.py wrote predictions.parquet with name='predictions' or kind='predictions'."
        )
    # Support both libs.responses.HarnessArtifact (path=) and project-local (uri=)
    uri: str = getattr(pred_artifact, "uri", None) or getattr(pred_artifact, "path", None)  # type: ignore[assignment]
    if not uri:
        raise ValueError("predictions artifact has neither 'uri' nor 'path' field")
    df = pl.read_parquet(uri)
    missing = _PREDICTIONS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions.parquet is missing required columns: {sorted(missing)}. "
            f"Present columns: {sorted(df.columns)}"
        )
    return df


def _extract_tree_path(
    tree: DecisionTreeRegressor,
    node_id: int,
    feature_names: list[str],
) -> list[tuple[str, str, float]]:
    """Walk from root to node_id and return list of (feature, operator, threshold)."""
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    # Build parent map
    parent: dict[int, tuple[int, str]] = {}
    stack = [0]
    while stack:
        nid = stack.pop()
        left = children_left[nid]
        right = children_right[nid]
        if left != -1:
            parent[left] = (nid, "<=")
            stack.append(left)
        if right != -1:
            parent[right] = (nid, ">")
            stack.append(right)

    path: list[tuple[str, str, float]] = []
    cur = node_id
    while cur in parent:
        par, op = parent[cur]
        feat_idx = feature[par]
        thresh = float(threshold[par])
        path.append((feature_names[feat_idx], op, thresh))
        cur = par

    return list(reversed(path))


def _slice_def_str(path: list[tuple[str, str, float]]) -> str:
    return " AND ".join(f"{feat} {op} {thresh:.4g}" for feat, op, thresh in path)


def _fingerprint_df(df: pl.DataFrame) -> str:
    cols = sorted(df.columns)
    h = hashlib.sha256()
    for row in df.select(cols).iter_rows():
        h.update(str(row).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Method: residual_slice_tree
# ---------------------------------------------------------------------------


@register_discovery_method(name="residual_slice_tree")
def residual_slice_tree(
    predictions_df: pl.DataFrame,
    feature_cols: list[str],
    data: pl.DataFrame,
    cfg: DiscoveryConfig,
) -> list[FeatureGapFinding]:
    """Decision-tree slice-miner on row_loss from predictions.parquet."""
    join_keys = ["user_id", "email_id", "date_sent"]
    available_feature_cols = [c for c in feature_cols if c in data.columns]

    if not available_feature_cols:
        return []

    joined = predictions_df.join(
        data.select(join_keys + available_feature_cols), on=join_keys, how="left"
    )

    feature_matrix = joined.select(available_feature_cols).to_numpy()
    row_loss_values = joined["row_loss"].to_numpy()

    valid_mask = ~np.isnan(feature_matrix).any(axis=1) & ~np.isnan(row_loss_values)
    feature_matrix = feature_matrix[valid_mask]
    row_loss_values = row_loss_values[valid_mask]

    if len(row_loss_values) == 0:
        return []

    min_leaf = int(getattr(cfg, "min_slice_size", None) or 1000)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=min_leaf, random_state=42)
    tree.fit(feature_matrix, row_loss_values)

    overall_mean = float(np.mean(row_loss_values))
    threshold_10pct = overall_mean * 1.10

    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    node_values = tree.tree_.value.squeeze()
    n_node_samples = tree.tree_.n_node_samples

    findings: list[FeatureGapFinding] = []
    finding_idx = 0

    for node_id in range(tree.tree_.node_count):
        is_leaf = children_left[node_id] == -1 and children_right[node_id] == -1
        if not is_leaf:
            continue

        leaf_mean = float(node_values[node_id])
        n_rows = int(n_node_samples[node_id])

        if leaf_mean <= threshold_10pct:
            continue

        uplift_pct = (leaf_mean - overall_mean) / overall_mean * 100.0
        severity_float = 1.0 if uplift_pct > 25.0 else 0.5

        path = _extract_tree_path(tree, node_id, available_feature_cols)
        slice_def = _slice_def_str(path)

        findings.append(
            FeatureGapFinding(
                id=f"residual_slice_tree_{finding_idx:03d}",
                kind="residual_slice",
                slice_definition=slice_def,
                residual_lift=uplift_pct / 100.0,
                severity=severity_float,
                priority=uplift_pct / 100.0,
                coverage=n_rows / len(row_loss_values),
                recommended_feature_family="",
                feature_plan_stub=(
                    f"Slice: {slice_def[:80]}. "
                    f"Mean loss uplift: {uplift_pct:.1f}%. "
                    f"N rows: {n_rows}."
                ),
            )
        )
        finding_idx += 1

    findings.sort(key=lambda f: f.residual_lift, reverse=True)
    return findings


# ---------------------------------------------------------------------------
# Method: label_noise_scan
# ---------------------------------------------------------------------------


@register_discovery_method(name="label_noise_scan")
def label_noise_scan(
    predictions_df: pl.DataFrame,
    cfg: DiscoveryConfig,
) -> list[FeatureGapFinding]:
    """Log-loss outlier detection: rows the model strongly disagreed with."""
    top_n = int(getattr(cfg, "label_noise_top_n", None) or 100)

    scores = predictions_df["score"].to_numpy()
    labels = predictions_df["label"].to_numpy()

    p99_score = float(np.percentile(scores, 99))
    p1_score = float(np.percentile(scores, 1))

    false_neg_mask = (labels == 0) & (scores > p99_score)
    false_pos_mask = (labels == 1) & (scores < p1_score)
    candidate_mask = false_neg_mask | false_pos_mask

    candidate_df = predictions_df.filter(pl.Series(candidate_mask))
    candidate_scores = candidate_df["score"].to_numpy()
    candidate_labels = candidate_df["label"].to_numpy()
    outlier_strength = np.abs(candidate_scores - candidate_labels)

    order = np.argsort(outlier_strength)[::-1][:top_n]
    top_candidates = candidate_df[order.tolist()]

    findings: list[FeatureGapFinding] = []
    for i, row in enumerate(top_candidates.iter_rows(named=True)):
        findings.append(
            FeatureGapFinding(
                id=f"label_noise_scan_{i:04d}",
                kind="label_noise",
                slice_definition=json.dumps(
                    {
                        "type": "label_noise_candidate",
                        "user_id": row["user_id"],
                        "email_id": row["email_id"],
                        "date_sent": str(row["date_sent"]),
                        "score": float(row["score"]),
                        "label": int(row["label"]),
                    }
                ),
                residual_lift=float(abs(row["score"] - row["label"])),
                severity=0.5,
                priority=float(abs(row["score"] - row["label"])),
                coverage=1.0 / max(len(predictions_df), 1),
                recommended_feature_family="",
                feature_plan_stub="Inspect these rows for possible label errors.",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def discover_gaps(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    prediction_cols: dict[str, str],
    harness_response: HarnessResponse,
    discovery_config: DiscoveryConfig,
    outcome_version: str | None = None,
) -> CivicShoutDiscoveryResponse:
    """Run all registered discovery methods and return a CivicShoutDiscoveryResponse."""
    t_total_start = time.perf_counter()

    predictions_df = _load_predictions(harness_response)

    enabled = (
        set(discovery_config.enabled_methods)
        if discovery_config.enabled_methods
        else set(_DISCOVERY_METHOD_REGISTRY)
    )

    stage_timings: list[HarnessStageTiming] = []
    all_findings: list[FeatureGapFinding] = []
    method_results: dict[str, Any] = {}

    for method_name, method_fn in _DISCOVERY_METHOD_REGISTRY.items():
        if method_name not in enabled:
            continue

        t_start = time.perf_counter()

        if method_name == "residual_slice_tree":
            findings = method_fn(predictions_df, feature_cols, data, discovery_config)
        elif method_name == "label_noise_scan":
            findings = method_fn(predictions_df, discovery_config)
        else:
            findings = []

        elapsed = time.perf_counter() - t_start
        stage_timings.append(HarnessStageTiming(stage=method_name, seconds=elapsed))
        all_findings.extend(findings)
        method_results[method_name] = {"n_findings": len(findings), "seconds": elapsed}

    # Sort by severity desc then residual_lift desc
    all_findings.sort(key=lambda f: (f.severity, f.residual_lift), reverse=True)
    top_k = discovery_config.top_k_findings
    all_findings = all_findings[:top_k]

    n_warning = sum(1 for f in all_findings if f.severity < 1.0 and f.severity >= 0.4)
    n_error = sum(1 for f in all_findings if f.severity >= 1.0)

    top_slice_findings = [f for f in all_findings if f.kind == "residual_slice"]
    top_slice_lift = top_slice_findings[0].residual_lift if top_slice_findings else 0.0
    top_slice_uplift_pct = top_slice_lift * 100.0

    label_noise_findings = [f for f in all_findings if f.kind == "label_noise"]
    n_noise = len(label_noise_findings)
    label_noise_rate = n_noise / max(len(predictions_df), 1)

    overall_health: str
    if n_error > 0:
        overall_health = "red"
    elif n_warning > 0:
        overall_health = "yellow"
    else:
        overall_health = "green"

    scores = DiscoveryScores(
        top_slice_residual_lift=top_slice_lift,
        embedding_region_count=0,
        label_noise_rate=label_noise_rate,
        priority_score_max=all_findings[0].priority if all_findings else 0.0,
        overall_health=overall_health,
        n_findings_warning=n_warning,
        n_findings_error=n_error,
    )

    harness_hash = hashlib.sha256(harness_response.model_dump_json().encode()).hexdigest()[:16]

    valid_pred_cols = sorted(c for c in prediction_cols.values() if c in predictions_df.columns)
    pred_fingerprint = (
        _fingerprint_df(predictions_df.select(valid_pred_cols)) if valid_pred_cols else ""
    )

    reproducibility = DiscoveryReproducibility(
        harness_response_hash=harness_hash,
        predictions_fingerprint=pred_fingerprint,
        data_fingerprint="",
    )

    total_seconds = time.perf_counter() - t_total_start
    stage_timings.append(HarnessStageTiming(stage="total", seconds=total_seconds))

    performance = DiscoveryPerformance(
        total_seconds=total_seconds,
        stage_timings=stage_timings,
    )

    # Write summary markdown artifact
    _write_discovery_summary(all_findings, method_results, performance)

    # Append timing row to ledger
    _append_timing_row(method_results, total_seconds)

    return CivicShoutDiscoveryResponse(
        findings=all_findings,
        scores=scores,
        reproducibility=reproducibility,
        performance=performance,
        outcome_version=outcome_version or "v1",
        n_label_noise_candidates=n_noise,
        top_slice_loss_uplift_pct=top_slice_uplift_pct,
    )


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent


def _write_discovery_summary(
    findings: list[FeatureGapFinding],
    method_results: dict[str, Any],
    performance: DiscoveryPerformance,
) -> None:
    lines = [
        "# Discovery Summary",
        "",
        f"Total elapsed: {performance.total_seconds:.2f}s",
        f"Total findings: {len(findings)}",
        "",
        "## Method Results",
        "",
    ]
    for name, res in method_results.items():
        lines.append(f"- **{name}**: {res['n_findings']} findings in {res['seconds']:.2f}s")
    lines.append("")
    lines.append("## Top Findings")
    lines.append("")
    for i, f in enumerate(findings[:10]):
        sev = "ERROR" if f.severity >= 1.0 else "WARNING"
        lines.append(f"### Finding {i + 1} [{sev}] — {f.id}")
        lines.append(f"- Kind: {f.kind}")
        lines.append(f"- Residual lift: {f.residual_lift:.4f}")
        lines.append(f"- Priority: {f.priority:.4f}")
        lines.append(f"- Coverage: {f.coverage:.4f}")
        if f.slice_definition:
            lines.append(f"- Slice: `{f.slice_definition[:120]}`")
        lines.append("")

    out_path = _PROJECT_ROOT / "discovery_summary.md"
    out_path.write_text("\n".join(lines))


def _append_timing_row(method_results: dict[str, Any], total_seconds: float) -> None:
    import datetime

    row = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_seconds": total_seconds,
        "methods": method_results,
    }
    ledger = _PROJECT_ROOT / "discovery_timing_performance.jsonl"
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")
