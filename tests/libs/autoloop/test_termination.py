from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from libs.autoloop.budget import AutoloopBudget
from libs.autoloop.state import (
    AutoloopConfig,
    ExecutorResult,
    IterationRecord,
    PlannerResult,
    State,
)
from libs.autoloop.termination import TerminationVerdict, brainstorm_has_queued, check_all

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    project: str = "test_proj",
    iterations_max: int = 70,
    dollars_max: float = 80.0,
    wall_seconds_max: int = 86400,
    top_k: int = 3,
    window_iters: int = 6,
    metric_direction: str = "higher_is_better",
    dedup_reject_streak: int = 4,
) -> AutoloopConfig:
    raw: dict[str, Any] = {
        "project": project,
        "outcome_variable": "signed",
        "comparison_group": "cg1",
        "primary_metric": "auc",
        "metric_direction": metric_direction,
        "budget": {
            "iterations_max": iterations_max,
            "dollars_max": dollars_max,
            "wall_seconds_max": wall_seconds_max,
        },
        "plateau": {"top_k": top_k, "window_iters": window_iters},
        "loop_detection": {"dedup_reject_streak": dedup_reject_streak},
    }
    return AutoloopConfig.model_validate(raw)


def _make_state(tmp_path: Path, cfg: AutoloopConfig) -> State:
    state = State(tmp_path, cfg)
    state.state_dir.mkdir(parents=True, exist_ok=True)
    return state


def _make_budget(
    state: State,
    cfg: AutoloopConfig,
    *,
    first_iter_ts: datetime | None = None,
) -> AutoloopBudget:
    tracker = MagicMock()
    tracker.get_all.return_value = {}
    return AutoloopBudget(
        cfg=cfg,
        state=state,
        tracker=tracker,
        first_iter_ts=first_iter_ts,
    )


def _iter_rec(
    iteration_id: int,
    *,
    run_id: str | None = None,
    harness_metric: float | None = None,
    planner_added: int = 1,
    planner_reranked: int = 1,
    planner_exit_code: int = 0,
    executor_exit_code: int = 0,
    fingerprint: str = "",
) -> IterationRecord:
    planner = PlannerResult(
        role="planner",
        session_id=None,
        exit_code=planner_exit_code,
        model="claude-opus",
        log_path="",
        wall_seconds=1.0,
        brainstorm_added=planner_added,
        brainstorm_reranked=planner_reranked,
    )
    executor = ExecutorResult(
        role="executor",
        session_id=None,
        exit_code=executor_exit_code,
        model="claude-opus",
        log_path="",
        wall_seconds=1.0,
        run_id=run_id,
        harness_metric=harness_metric,
    )
    return IterationRecord(
        iteration_id=iteration_id,
        ts_start="2025-01-01T00:00:00+00:00",
        planner=planner,
        executor=executor,
        iteration_fingerprint=fingerprint,
    )


def _write_iterations(state: State, records: list[IterationRecord]) -> None:
    for rec in records:
        state.append_iteration(rec)


def _call_check_all(
    state: State, cfg: AutoloopConfig, budget: AutoloopBudget
) -> TerminationVerdict:
    return check_all(state=state, cfg=cfg, budget=budget, project_root=state.state_dir.parent)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_iterations_should_not_stop(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    budget = _make_budget(state, cfg)

    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is False
    assert verdict.reason is None


def test_manual_stop_file_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)
    budget = _make_budget(state, cfg)
    (state.state_dir / "STOP").touch()

    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "manual_stop"


def test_iter_cap_reached_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg(iterations_max=3)
    state = _make_state(tmp_path, cfg)

    for i in range(1, 4):
        state.append_iteration(_iter_rec(i))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "iter_cap"


def test_dollar_cap_reached_triggers_termination(tmp_path: Path) -> None:
    from libs.costs.models import CostEntry, CostSummary

    cfg = _make_cfg(dollars_max=10.0)
    state = _make_state(tmp_path, cfg)

    # Budget filters claude_p_session from entries; pass a non-CLI entry so it counts.
    entries = [
        CostEntry(
            key=f"autoloop:{cfg.project}:something",
            amount=15.0,
            provider="openai",
            operation="openai_embedding",
        )
    ]
    tracker = MagicMock()
    tracker.get_all.return_value = {
        f"autoloop:{cfg.project}:something": CostSummary(
            key=f"autoloop:{cfg.project}:something",
            total=15.0,
            count=1,
            entries=entries,
        )
    }
    budget = AutoloopBudget(cfg=cfg, state=state, tracker=tracker, first_iter_ts=None)

    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "dollar_cap"


def test_wall_cap_reached_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg(wall_seconds_max=60)
    state = _make_state(tmp_path, cfg)
    tracker = MagicMock()
    tracker.get_all.return_value = {}
    first_iter = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
    budget = AutoloopBudget(cfg=cfg, state=state, tracker=tracker, first_iter_ts=first_iter)

    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "wall_cap"


def test_plateau_top3_unchanged_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg(top_k=3, window_iters=6)
    state = _make_state(tmp_path, cfg)

    run_ids = ["run-a", "run-b", "run-c"]
    for i in range(1, 9):
        rid = run_ids[i % len(run_ids)]
        state.append_iteration(_iter_rec(i, run_id=rid, harness_metric=0.8 + (i % 3) * 0.01))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "plateau"


def test_plateau_no_stop_when_top3_changed(tmp_path: Path) -> None:
    cfg = _make_cfg(top_k=3, window_iters=6)
    state = _make_state(tmp_path, cfg)

    for i in range(1, 7):
        state.append_iteration(_iter_rec(i, run_id=f"run-{i}", harness_metric=float(i)))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is False


def test_loop_fingerprint_identical_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)

    for i in range(1, 4):
        state.append_iteration(_iter_rec(i, fingerprint="same-fp"))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "loop_fingerprint"


def test_loop_fingerprint_different_does_not_stop(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)

    for i in range(1, 4):
        state.append_iteration(_iter_rec(i, fingerprint=f"fp-{i}"))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is False


def test_dedup_rejected_streak_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg(dedup_reject_streak=4)
    state = _make_state(tmp_path, cfg)

    for i in range(1, 5):
        state.append_iteration(_iter_rec(i, planner_added=0, planner_reranked=0))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "dedup_streak"


def test_planner_crash_streak_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)

    for i in range(1, 3):
        state.append_iteration(_iter_rec(i, planner_exit_code=1))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "planner_crash_streak"


def test_executor_crash_streak_triggers_termination(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)

    for i in range(1, 4):
        state.append_iteration(_iter_rec(i, executor_exit_code=1))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "executor_crash_streak"


def test_manual_stop_takes_priority_over_dollar_cap(tmp_path: Path) -> None:
    from libs.costs.models import CostSummary

    cfg = _make_cfg(dollars_max=10.0)
    state = _make_state(tmp_path, cfg)
    (state.state_dir / "STOP").touch()

    tracker = MagicMock()
    tracker.get_all.return_value = {
        f"autoloop:{cfg.project}:x": CostSummary(
            key=f"autoloop:{cfg.project}:x", total=99.0, count=1, entries=[]
        )
    }
    budget = AutoloopBudget(cfg=cfg, state=state, tracker=tracker, first_iter_ts=None)

    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "manual_stop"


def test_brainstorm_has_queued_true(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)

    brainstorm_path = state.state_dir / "brainstorm.jsonl"
    item = {
        "schema": "brainstorm/v1",
        "id": "b1",
        "ts": "2025-01-01T00:00:00+00:00",
        "title": "Test idea",
        "hypothesis": "hyp",
        "source": {"kind": "seed"},
        "feature_family_name_hint": "ff",
        "row_grain": "user",
        "content_hash": "abc",
        "status": "queued",
    }
    brainstorm_path.write_text(json.dumps(item) + "\n")

    assert brainstorm_has_queued(state) is True


def test_brainstorm_has_queued_false_when_only_consumed(tmp_path: Path) -> None:
    cfg = _make_cfg()
    state = _make_state(tmp_path, cfg)

    brainstorm_path = state.state_dir / "brainstorm.jsonl"
    item = {
        "schema": "brainstorm/v1",
        "id": "b1",
        "ts": "2025-01-01T00:00:00+00:00",
        "title": "Test idea",
        "hypothesis": "hyp",
        "source": {"kind": "seed"},
        "feature_family_name_hint": "ff",
        "row_grain": "user",
        "content_hash": "abc",
        "status": "run_completed",
    }
    brainstorm_path.write_text(json.dumps(item) + "\n")

    assert brainstorm_has_queued(state) is False


def test_lower_is_better_plateau_sorted_ascending(tmp_path: Path) -> None:
    cfg = _make_cfg(metric_direction="lower_is_better", top_k=2, window_iters=4)
    state = _make_state(tmp_path, cfg)

    run_ids = ["run-x", "run-y"]
    for i in range(1, 6):
        rid = run_ids[i % 2]
        state.append_iteration(_iter_rec(i, run_id=rid, harness_metric=0.1 + (i % 2) * 0.01))

    budget = _make_budget(state, cfg)
    verdict = _call_check_all(state, cfg, budget)

    assert verdict.should_stop is True
    assert verdict.reason == "plateau"
