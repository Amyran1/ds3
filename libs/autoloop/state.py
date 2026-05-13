from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic file utilities
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_json(path: Path, obj: dict | BaseModel) -> None:
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2, by_alias=True)
    else:
        text = json.dumps(obj, indent=2)
    atomic_write_text(path, text)


def append_jsonl(path: Path, model: BaseModel) -> None:
    with path.open("a") as f:
        f.write(model.model_dump_json(by_alias=True) + "\n")
        f.flush()


def read_jsonl_iter(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("JSONDecodeError on line %d of %s — skipping", lineno, path)


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class GapFinderCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    cadence: Literal["never", "every_iter", "every_n_iters", "on_demand"] = "every_n_iters"
    n: int = 5
    champion_only: bool = True


class BudgetCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    iterations_max: int = 70
    dollars_max: float = 80.0
    wall_seconds_max: int = 86400
    per_session_dollars_max: float = 5.0
    per_session_wall_seconds_max: int = 4200


class PlateauCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    top_k: int = 3
    window_iters: int = 6


class LoopDetectionCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    dedup_reject_streak: int = 4
    identical_args_streak: int = 3


class CurriculumCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    explore_fraction: float = 0.4
    exploit_top_k: int = 5


class CheckpointCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    human_every_n_iters: int = 0
    slack_milestone_iters: list[int] = Field(default_factory=lambda: [1, 10, 25, 50, 70])


class SessionCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    planner_model: str = "claude-opus-4-7"
    executor_model: str = "claude-opus-4-7"
    cli_path: str = ""

    def resolved_cli_path(self) -> str:
        if self.cli_path:
            return self.cli_path
        return shutil.which("claude") or "claude"


# ---------------------------------------------------------------------------
# AutoloopConfig
# ---------------------------------------------------------------------------


class AutoloopConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    project: str
    outcome_variable: str
    comparison_group: str
    primary_metric: str
    metric_direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    sample_frac: float = 0.10
    gap_finder: GapFinderCfg = Field(default_factory=GapFinderCfg)
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    plateau: PlateauCfg = Field(default_factory=PlateauCfg)
    loop_detection: LoopDetectionCfg = Field(default_factory=LoopDetectionCfg)
    curriculum: CurriculumCfg = Field(default_factory=CurriculumCfg)
    checkpoint: CheckpointCfg = Field(default_factory=CheckpointCfg)
    session: SessionCfg = Field(default_factory=SessionCfg)

    @classmethod
    def load(cls, project_root: Path) -> AutoloopConfig:
        # project_root is the repo root; config lives alongside the state files
        # at projects/{project}/autoloop/config.yaml — but since we don't know
        # the project name before parsing, callers may also pass the autoloop dir
        # directly. We try both: autoloop/config.yaml (caller = project dir)
        # then fall back to config.yaml (caller = autoloop dir).
        candidate = project_root / "autoloop" / "config.yaml"
        if not candidate.exists():
            candidate = project_root / "config.yaml"
        raw = yaml.safe_load(candidate.read_text())
        return cls.model_validate(raw)

    def state_dir(self, project_root: Path) -> Path:
        return project_root / "projects" / self.project / "autoloop"


# ---------------------------------------------------------------------------
# Brainstorm models
# ---------------------------------------------------------------------------


class BrainstormSource(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    kind: Literal["discovery", "eda", "user", "seed", "reflection", "planner"]
    ref: str | None = None
    finding_id: str | None = None


class BrainstormItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["brainstorm/v1"] = Field(default="brainstorm/v1", alias="schema")
    id: str
    ts: str
    title: str
    hypothesis: str
    source: BrainstormSource
    feature_family_name_hint: str
    row_grain: str
    estimated_cost_dollars: float = 0.5
    estimated_implement_minutes: int = 30
    content_hash: str
    embedding_hash: str | None = None
    embedding_uri: str | None = None
    tier: Literal["T0", "T1", "T2", "T3"] = "T2"
    tier_score: float = 0.5
    tier_rationale: str = ""
    status: Literal[
        "queued",
        "in_progress",
        "run_completed",
        "run_failed",
        "deduped_rejected",
        "deferred",
        "bailed_timing_blowup",
        "bailed_unfeasible",
    ] = "queued"
    iteration_id: int | None = None
    run_id: str | None = None
    iteration_history: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sentinel / summary models
# ---------------------------------------------------------------------------


class PlannerSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["planner_summary/v1"] = Field(
        default="planner_summary/v1", alias="schema"
    )
    iteration_id: int
    added: int
    reranked: int
    skipped: int
    ran_gap_finder: bool


class ExecutorSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["executor_summary/v1"] = Field(
        default="executor_summary/v1", alias="schema"
    )
    iteration_id: int
    brainstorm_id: str
    run_id: str | None
    metric: float | None
    baseline_metric: float | None
    lift: float | None
    verdict: Literal["completed", "bailed_timing_blowup", "bailed_unfeasible", "run_failed"]


# ---------------------------------------------------------------------------
# Session result models
# ---------------------------------------------------------------------------


class SessionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    role: Literal["planner", "executor"]
    session_id: str | None
    exit_code: int
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    dollars: float = 0.0
    log_path: str
    wall_seconds: float
    tools_used_counts: dict[str, int] = Field(default_factory=dict)
    wrote_files: list[str] = Field(default_factory=list)
    kill_reason: Literal["normal", "wall_cap", "dollar_cap", "crashed"] | None = None


class PlannerResult(SessionResult):
    ran_gap_finder: bool = False
    brainstorm_added: int = 0
    brainstorm_reranked: int = 0
    brainstorm_dedup_rejected: int = 0


class ExecutorResult(SessionResult):
    brainstorm_id: str | None = None
    run_id: str | None = None
    harness_metric: float | None = None
    baseline_metric: float | None = None
    lift: float | None = None
    run_record_status: Literal[
        "completed",
        "failed_validation",
        "failed_harness",
        "failed_recording",
        "missing",
        "unknown",
    ] = "unknown"


# ---------------------------------------------------------------------------
# Loop-control models
# ---------------------------------------------------------------------------


class TerminationCheck(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    iter_cap_hit: bool = False
    dollar_cap_hit: bool = False
    wall_cap_hit: bool = False
    plateau_hit: bool = False
    dedup_streak: int = 0
    identical_args_streak: int = 0
    manual_stop_present: bool = False
    planner_crash_streak: int = 0
    executor_crash_streak: int = 0
    brainstorm_exhausted: bool = False


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    iterations_used: int = 0
    iterations_max: int
    dollars_used: float = 0.0
    dollars_max: float
    wall_seconds_used: float = 0.0
    wall_seconds_max: float
    last_top3_run_ids: list[str] = Field(default_factory=list)
    last_top3_metrics: list[float] = Field(default_factory=list)
    iters_since_top3_change: int = 0
    updated_at: str


class IterationRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["iteration/v1"] = Field(default="iteration/v1", alias="schema")
    iteration_id: int
    ts_start: str
    ts_end: str | None = None
    phase_timings: dict[str, float] = Field(default_factory=dict)
    planner: PlannerResult | None = None
    executor: ExecutorResult | None = None
    budget_after: BudgetSnapshot | None = None
    termination_checks: TerminationCheck = Field(default_factory=TerminationCheck)
    iteration_fingerprint: str = ""


# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------


class State:
    def __init__(self, project_root: Path, cfg: AutoloopConfig) -> None:
        self._project_root = project_root
        self._cfg = cfg

    @property
    def state_dir(self) -> Path:
        return self._cfg.state_dir(self._project_root)

    def _iterations_path(self) -> Path:
        return self.state_dir / "iterations.jsonl"

    def _brainstorm_path(self) -> Path:
        return self.state_dir / "brainstorm.jsonl"

    def next_iteration_id(self) -> int:
        path = self._iterations_path()
        if not path.exists():
            return 1
        max_id = 0
        for row in read_jsonl_iter(path):
            try:
                iter_id = int(row["iteration_id"])
                max_id = max(max_id, iter_id)
            except (KeyError, TypeError, ValueError):
                pass
        return max_id + 1

    def append_iteration(self, rec: IterationRecord) -> None:
        path = self._iterations_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = rec.model_dump_json(by_alias=True) + "\n"
        with path.open("a") as f:
            f.write(line)
            f.flush()
            # fsync so that a crash mid-loop doesn't lose the completed record
            os.fsync(f.fileno())

    def recent_iterations(self, n: int = 5) -> list[IterationRecord]:
        path = self._iterations_path()
        rows: list[dict] = []
        for row in read_jsonl_iter(path):
            if row.get("schema") == "iteration/v1":
                rows.append(row)
        tail = rows[-n:] if len(rows) > n else rows
        result: list[IterationRecord] = []
        for row in tail:
            try:
                result.append(IterationRecord.model_validate(row))
            except Exception:
                logger.warning("Could not parse IterationRecord from row — skipping")
        return result

    def load_brainstorm(self) -> list[BrainstormItem]:
        """Event-sourced replay: last brainstorm/v1 line per id wins."""
        path = self._brainstorm_path()
        latest: dict[str, dict] = {}
        for row in read_jsonl_iter(path):
            if row.get("schema") == "brainstorm/v1":
                item_id = row.get("id")
                if item_id:
                    latest[item_id] = row
        result: list[BrainstormItem] = []
        for row in latest.values():
            try:
                result.append(BrainstormItem.model_validate(row))
            except ValidationError as e:
                logger.warning("Could not parse BrainstormItem — skipping: %s", e)
        return result

    def pop_top_queued(self) -> BrainstormItem | None:
        items = self.load_brainstorm()
        queued = [i for i in items if i.status == "queued"]
        if not queued:
            return None
        return max(queued, key=lambda i: i.tier_score)

    def mark_consumed(
        self,
        item_id: str,
        *,
        iteration_id: int,
        run_id: str | None,
        status: str,
    ) -> None:
        items = self.load_brainstorm()
        original = next((i for i in items if i.id == item_id), None)
        if original is None:
            logger.warning("mark_consumed: brainstorm id %s not found", item_id)
            return
        updated = original.model_copy(
            update={
                "status": status,
                "iteration_id": iteration_id,
                "run_id": run_id,
                "iteration_history": original.iteration_history + [iteration_id],
            }
        )
        path = self._brainstorm_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(path, updated)

    def mark_failed(self, item_id: str, *, iteration_id: int, reason: str) -> None:
        self.mark_consumed(item_id, iteration_id=iteration_id, run_id=None, status="run_failed")
