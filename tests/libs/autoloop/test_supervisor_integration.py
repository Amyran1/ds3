from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from libs.autoloop.state import (
    ExecutorResult,
    PlannerResult,
    SessionResult,
)
from libs.autoloop.supervisor import SupervisorArgs, SupervisorExitCode, run_supervisor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_result(**overrides: Any) -> SessionResult:
    defaults: dict[str, Any] = {
        "role": "planner",
        "session_id": "sess-001",
        "exit_code": 0,
        "model": "claude-test",
        "tokens_in": 100,
        "tokens_out": 50,
        "dollars": 0.10,
        "log_path": "",
        "wall_seconds": 1.0,
        "tools_used_counts": {},
        "wrote_files": [],
        "kill_reason": "normal",
    }
    defaults.update(overrides)
    return SessionResult.model_validate(defaults)


def _seed_brainstorm_item(
    autoloop_dir: Path,
    item_id: str = "BS-seed-001",
    status: str = "queued",
    tier_score: float = 0.5,
) -> None:
    item = {
        "schema": "brainstorm/v1",
        "id": item_id,
        "ts": "2026-01-01T00:00:00Z",
        "title": "demo seed",
        "hypothesis": "test feature",
        "source": {"kind": "seed"},
        "feature_family_name_hint": "demo_seed",
        "row_grain": "(row)",
        "content_hash": "abc",
        "tier": "T1",
        "tier_score": tier_score,
        "tier_rationale": "seed",
        "status": status,
    }
    brainstorm_path = autoloop_dir / "brainstorm.jsonl"
    with brainstorm_path.open("a") as f:
        f.write(json.dumps(item) + "\n")


def _write_config(autoloop_dir: Path, iterations_max: int = 3) -> None:
    autoloop_dir.mkdir(parents=True, exist_ok=True)
    (autoloop_dir / "config.yaml").write_text(
        f"""\
project: demo
outcome_variable: actioned
comparison_group: demo_v1
primary_metric: roc_auc
metric_direction: higher_is_better
sample_frac: 0.10
gap_finder:
  cadence: never
budget:
  iterations_max: {iterations_max}
  dollars_max: 20.0
  wall_seconds_max: 3600
  per_session_dollars_max: 2.0
  per_session_wall_seconds_max: 120
plateau:
  top_k: 3
  window_iters: 2
loop_detection:
  dedup_reject_streak: 4
  identical_args_streak: 3
curriculum:
  explore_fraction: 0.4
  exploit_top_k: 3
checkpoint:
  human_every_n_iters: 0
  slack_milestone_iters: []
session:
  planner_model: claude-opus-4-7
  executor_model: claude-opus-4-7
  cli_path: ""
"""
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_project(tmp_path: Path) -> Path:
    """Build a minimal autoloop-ready project tree; returns project_root."""
    project_root = tmp_path
    proj = project_root / "projects" / "demo"
    proj.mkdir(parents=True)

    (proj / "GOAL.md").write_text("Demo project goal for integration tests.")
    (proj / "harness.py").write_text("def harness(*args, **kwargs): pass\n")
    (proj / "eda.py").write_text("def eda(*args, **kwargs): pass\n")
    (proj / "discovery.py").write_text("def discover_gaps(*args, **kwargs): pass\n")

    runs = proj / "runs"
    runs.mkdir()
    (runs / "run_01.py").write_text("# baseline\n")
    (runs / "results.jsonl").write_text(
        json.dumps(
            {
                "project": "demo",
                "run_id": "run_01",
                "status": "completed",
                "comparison_group": "demo_v1",
                "scope": "comparison",
                "primary_metric_name": "roc_auc",
                "primary_metric_value": 0.75,
            }
        )
        + "\n"
    )

    (proj / "outcomes" / "actioned" / "v1").mkdir(parents=True)
    (proj / "outcomes" / "actioned" / "v1" / "dictionary.yaml").write_text("name: actioned\n")

    libs = project_root / "libs"
    libs.mkdir()
    (libs / "run_record.py").write_text("def run_record(*args, **kwargs): pass\n")
    (libs / "ledgers.py").write_text("class LedgerWriter: pass\n")

    _write_config(proj / "autoloop")
    _seed_brainstorm_item(proj / "autoloop")

    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )

    return project_root


class _FakeCostTracker:
    """Minimal CostTracker substitute that returns zero cost for any key."""

    class _Summary:
        total: float = 0.0

    def get(self, key: str) -> _Summary:
        return self._Summary()

    def get_all(self) -> dict:
        return {}


class _FakeContainer:
    def __init__(self) -> None:
        self.costs = _FakeCostTracker()

    async def __aenter__(self) -> _FakeContainer:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prereq_failure_exits_early(tmp_path: Path) -> None:
    """Supervisor exits with PREREQ_FAILED when harness.py is absent."""
    project_root = tmp_path
    proj = project_root / "projects" / "demo"
    proj.mkdir(parents=True)
    (proj / "GOAL.md").write_text("Goal.")

    args = SupervisorArgs(project="demo", project_root=project_root)

    exit_code = asyncio.run(run_supervisor(args))

    assert exit_code == SupervisorExitCode.PREREQ_FAILED


@pytest.mark.slow
def test_iter_cap_termination(fake_project: Path) -> None:
    """Supervisor exits with ITER_BUDGET_HIT after iterations_max iters."""
    autoloop_dir = fake_project / "projects" / "demo" / "autoloop"
    _write_config(autoloop_dir, iterations_max=1)

    args = SupervisorArgs(project="demo", project_root=fake_project)

    from libs.autoloop.prereqs import PrereqResult

    _git_clean_ok = PrereqResult(name="git working tree clean", passed=True)

    planner_result = PlannerResult(
        role="planner",
        session_id="sess-planner",
        exit_code=0,
        model="claude-test",
        log_path="",
        wall_seconds=0.5,
        kill_reason="normal",
        brainstorm_added=1,
        brainstorm_reranked=0,
        brainstorm_dedup_rejected=0,
        ran_gap_finder=False,
    )

    planner_summary_row = json.dumps(
        {
            "schema": "planner_summary/v1",
            "iteration_id": 1,
            "added": 1,
            "reranked": 0,
            "skipped": 0,
            "ran_gap_finder": False,
        }
    )

    executor_result = ExecutorResult(
        role="executor",
        session_id="sess-exec",
        exit_code=0,
        model="claude-test",
        log_path="",
        wall_seconds=0.5,
        kill_reason="normal",
        brainstorm_id="BS-seed-001",
        run_id="run_02",
        harness_metric=0.80,
        baseline_metric=0.75,
        lift=0.05,
        run_record_status="completed",
    )

    executor_summary_row = json.dumps(
        {
            "schema": "executor_summary/v1",
            "iteration_id": 1,
            "brainstorm_id": "BS-seed-001",
            "run_id": "run_02",
            "metric": 0.80,
            "baseline_metric": 0.75,
            "lift": 0.05,
            "verdict": "completed",
        }
    )

    results_jsonl_row = json.dumps(
        {
            "run_id": "run_02",
            "status": "completed",
            "project": "demo",
            "comparison_group": "demo_v1",
            "scope": "comparison",
        }
    )

    def fake_run_claude_session(**kwargs: Any) -> SessionResult:
        if kwargs.get("role") == "planner":
            state_dir = fake_project / "projects" / "demo" / "autoloop"
            brainstorm_path = state_dir / "brainstorm.jsonl"
            with brainstorm_path.open("a") as f:
                f.write(planner_summary_row + "\n")
            return planner_result
        else:
            state_dir = fake_project / "projects" / "demo" / "autoloop"
            brainstorm_path = state_dir / "brainstorm.jsonl"
            with brainstorm_path.open("a") as f:
                f.write(executor_summary_row + "\n")
            results_path = fake_project / "projects" / "demo" / "runs" / "results.jsonl"
            with results_path.open("a") as f:
                f.write(results_jsonl_row + "\n")
            return executor_result

    with (
        patch(
            "libs.autoloop.supervisor.session.run_claude_session",
            side_effect=fake_run_claude_session,
        ),
        patch("libs.autoloop.supervisor.Container", return_value=_FakeContainer()),
        patch("libs.autoloop.failure_modes.extract_failure_fingerprint", new_callable=AsyncMock),
        patch("libs.autoloop.supervisor.retention.prune_old_predictions"),
        patch("libs.autoloop.prereqs._check_git_clean", return_value=_git_clean_ok),
    ):
        exit_code = asyncio.run(run_supervisor(args))

    assert exit_code == SupervisorExitCode.ITER_BUDGET_HIT


@pytest.mark.slow
def test_planner_crash_streak(fake_project: Path) -> None:
    """Supervisor exits with PLANNER_FAILED after 2 consecutive planner failures."""
    args = SupervisorArgs(project="demo", project_root=fake_project)

    crashed_session = SessionResult(
        role="planner",
        session_id=None,
        exit_code=1,
        model="claude-test",
        log_path="",
        wall_seconds=0.2,
        kill_reason="crashed",
    )

    def fake_run_claude_session(**kwargs: Any) -> SessionResult:
        return crashed_session

    with (
        patch(
            "libs.autoloop.supervisor.session.run_claude_session",
            side_effect=fake_run_claude_session,
        ),
        patch("libs.autoloop.supervisor.Container", return_value=_FakeContainer()),
        patch("libs.autoloop.failure_modes.extract_failure_fingerprint", new_callable=AsyncMock),
    ):
        exit_code = asyncio.run(run_supervisor(args))

    assert exit_code == SupervisorExitCode.PLANNER_FAILED


@pytest.mark.slow
def test_manual_stop_file(fake_project: Path) -> None:
    """Supervisor exits immediately with MANUAL_STOP when STOP file exists."""
    from libs.autoloop.prereqs import PrereqResult

    stop_path = fake_project / "projects" / "demo" / "autoloop" / "STOP"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("stop\n")

    args = SupervisorArgs(project="demo", project_root=fake_project)

    _git_clean_ok = PrereqResult(name="git working tree clean", passed=True)

    with (
        patch("libs.autoloop.supervisor.Container", return_value=_FakeContainer()),
        patch("libs.autoloop.prereqs._check_git_clean", return_value=_git_clean_ok),
    ):
        exit_code = asyncio.run(run_supervisor(args))

    assert exit_code == SupervisorExitCode.MANUAL_STOP
