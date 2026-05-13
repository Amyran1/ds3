from __future__ import annotations

import hashlib
import importlib
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.ledgers import LedgerError, LedgerWriter
from libs.responses import (
    DiscoveryConfig,
    DiscoveryResponse,
    DiscoveryResultRow,
    EDAConfig,
    EDAResponse,
    EDAResultRow,
    HarnessResponse,
    RunResultMetadata,
    RunResultRow,
)

logger = logging.getLogger(__name__)

_SCOPE_WRITES_PREDICTIONS = frozenset({"smoke", "comparison", "champion_candidate"})
_SCOPE_RUNS_DISCOVERY = frozenset({"champion_candidate", "comparison"})


@dataclass
class RunRecordOutput:
    metadata: RunResultMetadata
    harness_response: HarnessResponse | None
    eda_response: EDAResponse | None
    discovery_response: DiscoveryResponse | None
    result_row: RunResultRow | None
    eda_row: EDAResultRow | None
    discovery_row: DiscoveryResultRow | None
    artifacts_dir: Path


def _resolve_git_sha(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _compute_data_fingerprint(data: Any) -> str:
    shape = getattr(data, "shape", None)
    columns = sorted(getattr(data, "columns", []))
    payload = f"{shape}:{columns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _failure_result_row(metadata: RunResultMetadata, reason: str) -> RunResultRow:
    return RunResultRow(
        project=metadata.project,
        run_id=metadata.run_id,
        comparison_group=metadata.comparison_group,
        scope=metadata.scope,
        status=metadata.status,
        verdict=metadata.verdict,
        recorded_at_utc=metadata.recorded_at_utc,
        git_sha=metadata.git_sha,
        outcome_version=metadata.outcome_version,
        primary_metric_name="",
        primary_metric_value=None,
        baseline_metric_value=None,
        lift_vs_baseline=None,
        metric_direction="higher_is_better",
        n_rows=0,
        sample_frac=0.0,
        is_full_run=False,
        n_features=0,
        total_seconds=0.0,
        cpu_utilization_pct=None,
        fingerprint="",
        notes=reason,
    )


def _failure_eda_row(metadata: RunResultMetadata) -> EDAResultRow:
    return EDAResultRow(
        project=metadata.project,
        run_id=metadata.run_id,
        comparison_group=metadata.comparison_group,
        scope=metadata.scope,
        recorded_at_utc=metadata.recorded_at_utc,
        git_sha=metadata.git_sha,
        eda_outcome_prevalence=None,
        eda_drift_score=0.0,
        eda_n_findings_warning=0,
        eda_n_findings_error=0,
        eda_top_finding_id=None,
        eda_total_seconds=0.0,
    )


def _failure_discovery_row(metadata: RunResultMetadata) -> DiscoveryResultRow:
    return DiscoveryResultRow(
        project=metadata.project,
        run_id=metadata.run_id,
        comparison_group=metadata.comparison_group,
        scope=metadata.scope,
        recorded_at_utc=metadata.recorded_at_utc,
        git_sha=metadata.git_sha,
        discovery_top_slice_residual_lift=0.0,
        discovery_embedding_region_count=0,
        discovery_label_noise_rate=0.0,
        discovery_priority_score_max=0.0,
        discovery_overall_health="red",
        discovery_n_findings_warning=0,
        discovery_n_findings_error=0,
        discovery_total_seconds=0.0,
    )


def _write_failure_shell_rows(
    metadata: RunResultMetadata,
    status: str,
    reason: str,
    project_root: Path,
    include_discovery: bool = False,
) -> None:
    # Shell rows preserve R3/R4: the leaderboard drift check finds the failure rather than missing rows.
    failed_meta = metadata.model_copy(update={"status": status, "notes": reason})
    result_row = _failure_result_row(failed_meta, reason)
    eda_row = _failure_eda_row(failed_meta)
    discovery_row = _failure_discovery_row(failed_meta) if include_discovery else None
    try:
        writer = LedgerWriter(project_root, metadata.project)
        writer.append_atomic(result_row, eda_row, discovery_row)
    except Exception:
        logger.exception("Failed to write failure shell rows for run %s", metadata.run_id)


def run_record(
    *,
    metadata: RunResultMetadata,
    data: Any,
    feature_cols: list[str],
    outcome_variable: str,
    model_cfg: dict[str, Any],
    eda_config: EDAConfig | None = None,
    discovery_config: DiscoveryConfig | None = None,
    project_root: Path | None = None,
) -> RunRecordOutput:
    import re

    resolved_root = project_root if project_root is not None else Path.cwd()
    proj_dir = resolved_root / "projects" / metadata.project
    runs_dir = proj_dir / "runs"
    artifacts_dir = runs_dir / metadata.run_id / "artifacts"

    if not re.fullmatch(r"run_\d+", metadata.run_id):
        raise ValueError(f"run_id must match 'run_N+' pattern, got {metadata.run_id!r}")

    if not metadata.git_sha:
        metadata = metadata.model_copy(update={"git_sha": _resolve_git_sha(resolved_root)})
    if not metadata.data_fingerprint:
        metadata = metadata.model_copy(update={"data_fingerprint": _compute_data_fingerprint(data)})
    if not metadata.recorded_at_utc:
        metadata = metadata.model_copy(
            update={"recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        )

    try:
        harness_mod = importlib.import_module(f"projects.{metadata.project}.harness")
        eda_mod = importlib.import_module(f"projects.{metadata.project}.eda")
        discovery_mod = importlib.import_module(f"projects.{metadata.project}.discovery")
    except ModuleNotFoundError as e:
        _write_failure_shell_rows(metadata, "failed_validation", str(e), resolved_root)
        raise

    try:
        eda_response = eda_mod.eda(
            data=data,
            feature_cols=feature_cols,
            outcome_variable=outcome_variable,
            eda_config=(eda_config or EDAConfig()),
            outcome_version=metadata.outcome_version,
        )
    except Exception as e:
        _write_failure_shell_rows(metadata, "failed_harness", f"EDA failed: {e}", resolved_root)
        raise

    if not isinstance(eda_response, EDAResponse):
        _write_failure_shell_rows(
            metadata,
            "failed_harness",
            f"eda() returned {type(eda_response).__name__}, expected EDAResponse",
            resolved_root,
        )
        raise TypeError(f"eda() must return EDAResponse, got {type(eda_response).__name__}")

    predictions_dir: Path | None = None
    if metadata.scope in _SCOPE_WRITES_PREDICTIONS:
        predictions_dir = artifacts_dir
        predictions_dir.mkdir(parents=True, exist_ok=True)

    try:
        harness_response = harness_mod.harness(
            data=data,
            feature_cols=feature_cols,
            outcome_variable=outcome_variable,
            ml_model_config=model_cfg,
            sample_frac=None,
            predictions_dir=str(predictions_dir) if predictions_dir is not None else None,
        )
    except Exception as e:
        _write_failure_shell_rows(metadata, "failed_harness", f"Harness failed: {e}", resolved_root)
        raise

    if not isinstance(harness_response, HarnessResponse):
        _write_failure_shell_rows(
            metadata,
            "failed_harness",
            f"harness() returned {type(harness_response).__name__}, expected HarnessResponse",
            resolved_root,
        )
        raise TypeError(
            f"harness() must return HarnessResponse, got {type(harness_response).__name__}"
        )

    discovery_response: DiscoveryResponse | None = None
    if metadata.scope in _SCOPE_RUNS_DISCOVERY and discovery_config is not None:
        try:
            discovery_response = discovery_mod.discover_gaps(
                data=data,
                feature_cols=feature_cols,
                outcome_variable=outcome_variable,
                prediction_cols={"score": "y_pred"},
                harness_response=harness_response,
                discovery_config=discovery_config,
                outcome_version=metadata.outcome_version,
            )
        except Exception as e:
            _write_failure_shell_rows(
                metadata,
                "failed_harness",
                f"Discovery failed: {e}",
                resolved_root,
                include_discovery=True,
            )
            raise

    result_row = harness_response.to_result_row(metadata)
    eda_row = eda_response.to_eda_summary_row(metadata)
    discovery_row = (
        discovery_response.to_discovery_summary_row(metadata)
        if discovery_response is not None
        else None
    )

    writer = LedgerWriter(resolved_root, metadata.project)
    try:
        writer.append_atomic(result_row, eda_row, discovery_row)
    except LedgerError as e:
        _write_failure_shell_rows(
            metadata,
            "failed_recording",
            str(e),
            resolved_root,
            include_discovery=discovery_response is not None,
        )
        raise

    return RunRecordOutput(
        metadata=metadata,
        harness_response=harness_response,
        eda_response=eda_response,
        discovery_response=discovery_response,
        result_row=result_row,
        eda_row=eda_row,
        discovery_row=discovery_row,
        artifacts_dir=artifacts_dir,
    )
