from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class PruneReport:
    kept_run_ids: list[str] = field(default_factory=list)
    removed_run_ids: list[str] = field(default_factory=list)
    files_removed: int = 0
    bytes_freed: int = 0
    skipped_missing: list[str] = field(default_factory=list)
    error_count: int = 0


def _parse_champion_run_id(champion_path: Path) -> str | None:
    try:
        lines = champion_path.read_text().splitlines()[:50]
    except OSError:
        return None
    for line in lines:
        m = re.search(r"\brun_\d+\b", line)
        if m:
            return m.group()
    return None


def _detect_champion_from_results(
    results_path: Path,
    comparison_group: str,
    primary_metric: str,
    metric_direction: Literal["higher_is_better", "lower_is_better"],
) -> str | None:
    if not results_path.exists():
        return None
    champion_candidates: list[tuple[float, str]] = []
    for lineno, raw in enumerate(results_path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("results.jsonl line %d: JSONDecodeError — skipping", lineno)
            continue
        if row.get("comparison_group") != comparison_group:
            continue
        if row.get("scope") not in {"comparison", "champion_candidate"}:
            continue
        if row.get("status") != "completed":
            continue
        value = row.get(primary_metric)
        if value is None:
            continue
        run_id = row.get("run_id")
        if not run_id:
            continue
        champion_candidates.append((float(value), str(run_id)))
    if not champion_candidates:
        return None
    reverse = metric_direction == "higher_is_better"
    champion_candidates.sort(key=lambda t: t[0], reverse=reverse)
    return champion_candidates[0][1]


def prune_old_predictions(
    project_root: Path,
    project: str,
    comparison_group: str,
    primary_metric: str,
    metric_direction: Literal["higher_is_better", "lower_is_better"],
    *,
    top_k: int = 5,
    current_run_id: str | None = None,
) -> PruneReport:
    runs_dir = project_root / "projects" / project / "runs"
    if not runs_dir.exists():
        return PruneReport()

    results_path = runs_dir / "results.jsonl"
    eligible: list[tuple[float, str]] = []

    if results_path.exists():
        for lineno, raw in enumerate(results_path.read_text().splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("results.jsonl line %d: JSONDecodeError — skipping", lineno)
                continue

            if row.get("comparison_group") != comparison_group:
                continue
            if row.get("scope") not in {"smoke", "comparison", "champion_candidate"}:
                continue
            if row.get("status") != "completed":
                continue

            value = row.get(primary_metric)
            if value is None:
                continue

            run_id = row.get("run_id")
            if not run_id:
                continue

            eligible.append((float(value), str(run_id)))

    reverse = metric_direction == "higher_is_better"
    eligible.sort(key=lambda t: t[0], reverse=reverse)
    protected: set[str] = {run_id for _, run_id in eligible[:top_k]}

    if current_run_id is not None:
        protected.add(current_run_id)

    champion_path = runs_dir / "CHAMPION.md"
    champion_run_id = _parse_champion_run_id(champion_path)
    if champion_run_id is None:
        champion_run_id = _detect_champion_from_results(
            results_path, comparison_group, primary_metric, metric_direction
        )
        if champion_run_id is None:
            logger.warning(
                "No champion found for comparison_group=%s — champion pinning skipped",
                comparison_group,
            )

    if champion_run_id is not None:
        champion_predictions = runs_dir / champion_run_id / "artifacts" / "predictions.parquet"
        if champion_predictions.exists():
            protected.add(champion_run_id)
            logger.info(
                "Pinning champion %s (predictions.parquet protected from prune)",
                champion_run_id,
            )

    report = PruneReport(kept_run_ids=sorted(protected))

    for parquet_path in runs_dir.glob("run_*/artifacts/predictions.parquet"):
        run_id = parquet_path.parent.parent.name
        if run_id in protected:
            continue
        try:
            size = parquet_path.stat().st_size
            parquet_path.unlink()
            report.removed_run_ids.append(run_id)
            report.files_removed += 1
            report.bytes_freed += size
        except FileNotFoundError:
            report.skipped_missing.append(run_id)
            report.error_count += 1
            logger.warning("predictions.parquet for %s not found during prune — skipping", run_id)
        except OSError as exc:
            report.skipped_missing.append(run_id)
            report.error_count += 1
            logger.warning("failed to remove predictions.parquet for %s: %s", run_id, exc)

    return report
