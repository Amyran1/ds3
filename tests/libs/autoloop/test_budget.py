from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from libs.autoloop.budget import AutoloopBudget
from libs.autoloop.state import AutoloopConfig, BudgetSnapshot, State


class FakeCostTracker:
    """Test double with both .total (legacy) and .entries (current) on summaries.

    AutoloopBudget.dollars_used filters out claude_p_session records from
    summary.entries; tests pass non-claude_p_session entries so they all count.
    """

    def __init__(self, totals: dict[str, float]) -> None:
        self._totals = totals

    def _make_summary(self, key: str, total: float) -> object:
        entry = type("E", (), {"amount": total, "operation": "openai_embedding"})()
        return type("S", (), {"total": total, "entries": [entry]})()

    def get_all(self) -> dict[str, object]:
        return {k: self._make_summary(k, v) for k, v in self._totals.items()}

    def get(self, key: str) -> object:
        return self._make_summary(key, self._totals.get(key, 0.0))


def _make_cfg(
    project: str = "proj",
    iterations_max: int = 70,
    dollars_max: float = 80.0,
    wall_seconds_max: int = 86400,
) -> AutoloopConfig:
    return AutoloopConfig.model_validate(
        {
            "project": project,
            "outcome_variable": "y",
            "comparison_group": "cg",
            "primary_metric": "roc_auc",
            "budget": {
                "iterations_max": iterations_max,
                "dollars_max": dollars_max,
                "wall_seconds_max": wall_seconds_max,
            },
        }
    )


def _make_state_with_next_id(next_id: int) -> MagicMock:
    state = MagicMock(spec=State)
    state.next_iteration_id.return_value = next_id
    return state


def _make_budget(
    tmp_path: Path,
    project: str = "proj",
    next_iter_id: int = 1,
    cost_totals: dict[str, float] | None = None,
    first_iter_ts: datetime | None = None,
    iterations_max: int = 70,
    dollars_max: float = 80.0,
    wall_seconds_max: int = 86400,
) -> AutoloopBudget:
    cfg = _make_cfg(
        project=project,
        iterations_max=iterations_max,
        dollars_max=dollars_max,
        wall_seconds_max=wall_seconds_max,
    )
    state = _make_state_with_next_id(next_iter_id)
    state.state_dir = tmp_path / "projects" / project / "autoloop"
    tracker = FakeCostTracker(cost_totals or {})
    return AutoloopBudget(cfg=cfg, state=state, tracker=tracker, first_iter_ts=first_iter_ts)


def test_iterations_used_reads_from_state(tmp_path: Path) -> None:
    budget = _make_budget(tmp_path, next_iter_id=6)

    assert budget.iterations_used() == 5


def test_dollars_used_sums_only_matching_prefix(tmp_path: Path) -> None:
    totals = {
        "autoloop:proj:planner:iter_01": 1.50,
        "autoloop:proj:executor:iter_01": 2.00,
        "autoloop:other_proj:planner:iter_01": 99.0,
        "unrelated:key": 10.0,
    }
    budget = _make_budget(tmp_path, project="proj", cost_totals=totals)

    assert abs(budget.dollars_used() - 3.50) < 1e-9


def test_dollars_used_ignores_other_prefixes(tmp_path: Path) -> None:
    totals = {
        "autoloop:other:planner": 50.0,
        "unrelated": 100.0,
    }
    budget = _make_budget(tmp_path, project="proj", cost_totals=totals)

    assert budget.dollars_used() == 0.0


def test_wall_seconds_used_returns_elapsed(tmp_path: Path) -> None:
    first_ts = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
    budget = _make_budget(tmp_path, first_iter_ts=first_ts)

    elapsed = budget.wall_seconds_used()
    assert 115 <= elapsed <= 130


def test_wall_seconds_used_returns_zero_when_no_ts(tmp_path: Path) -> None:
    budget = _make_budget(tmp_path, first_iter_ts=None)

    assert budget.wall_seconds_used() == 0.0


def test_caps_exceeded_iter_cap_hit(tmp_path: Path) -> None:
    budget = _make_budget(tmp_path, next_iter_id=71, iterations_max=70)

    status = budget.caps_exceeded()

    assert status.iter_cap_hit is True
    assert status.dollar_cap_hit is False
    assert status.wall_cap_hit is False


def test_caps_exceeded_dollar_cap_hit(tmp_path: Path) -> None:
    totals = {"autoloop:proj:planner": 85.0}
    budget = _make_budget(tmp_path, cost_totals=totals, dollars_max=80.0)

    status = budget.caps_exceeded()

    assert status.dollar_cap_hit is True
    assert status.iter_cap_hit is False


def test_caps_exceeded_wall_cap_hit(tmp_path: Path) -> None:
    first_ts = datetime.now(tz=timezone.utc) - timedelta(seconds=90000)
    budget = _make_budget(tmp_path, first_iter_ts=first_ts, wall_seconds_max=86400)

    status = budget.caps_exceeded()

    assert status.wall_cap_hit is True


def test_snapshot_writes_budget_json(tmp_path: Path) -> None:
    state_dir = tmp_path / "projects" / "proj" / "autoloop"
    state_dir.mkdir(parents=True)

    cfg = _make_cfg()
    state = _make_state_with_next_id(3)
    state.state_dir = state_dir
    tracker = FakeCostTracker({"autoloop:proj:planner": 5.0})
    budget = AutoloopBudget(cfg=cfg, state=state, tracker=tracker, first_iter_ts=None)

    budget.snapshot(
        last_top3_run_ids=["run_01", "run_02"],
        last_top3_metrics=[0.8, 0.75],
        iters_since_top3_change=2,
    )

    budget_path = state_dir / "budget.json"
    assert budget_path.exists()
    tmp_file = budget_path.with_suffix(".tmp")
    assert not tmp_file.exists()

    parsed = json.loads(budget_path.read_text())
    assert parsed["iterations_used"] == 2
    assert parsed["dollars_used"] == 5.0
    assert parsed["last_top3_run_ids"] == ["run_01", "run_02"]


def test_snapshot_returns_budget_snapshot_object(tmp_path: Path) -> None:
    state_dir = tmp_path / "projects" / "proj" / "autoloop"
    state_dir.mkdir(parents=True)

    cfg = _make_cfg()
    state = _make_state_with_next_id(5)
    state.state_dir = state_dir
    tracker = FakeCostTracker({})
    budget = AutoloopBudget(cfg=cfg, state=state, tracker=tracker, first_iter_ts=None)

    snap = budget.snapshot(
        last_top3_run_ids=[],
        last_top3_metrics=[],
        iters_since_top3_change=0,
    )

    assert isinstance(snap, BudgetSnapshot)
    assert snap.iterations_used == 4
    assert snap.iterations_max == 70
    assert snap.iters_since_top3_change == 0
    assert snap.updated_at != ""
