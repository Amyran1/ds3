from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from libs.autoloop.state import AutoloopConfig, BudgetSnapshot, State, atomic_write_json
from libs.costs.tracker import CostTracker


@dataclass
class CapStatus:
    iter_cap_hit: bool
    dollar_cap_hit: bool
    wall_cap_hit: bool


class AutoloopBudget:
    def __init__(
        self,
        *,
        cfg: AutoloopConfig,
        state: State,
        tracker: CostTracker,
        first_iter_ts: datetime | None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._tracker = tracker
        self._first_iter_ts = first_iter_ts

    def iterations_used(self) -> int:
        return self._state.next_iteration_id() - 1

    def dollars_used(self) -> float:
        # API spend only — exclude claude_p_session estimates because Claude Code
        # subprocess invocations are free under an auth-membership plan. Historical
        # records (before that decoupling) live in .costs/ledger.jsonl with
        # operation="claude_p_session"; this filter ignores them at read time so
        # we don't have to mutate the ledger.
        prefix = f"autoloop:{self._cfg.project}:"
        total = 0.0
        for key, summary in self._tracker.get_all().items():
            if not key.startswith(prefix):
                continue
            for entry in summary.entries:
                if entry.operation == "claude_p_session":
                    continue
                total += entry.amount
        return total

    def wall_seconds_used(self) -> float:
        if self._first_iter_ts is None:
            return 0.0
        now = datetime.now(tz=timezone.utc)
        return (now - self._first_iter_ts).total_seconds()

    def caps_exceeded(self) -> CapStatus:
        return CapStatus(
            iter_cap_hit=self.iterations_used() >= self._cfg.budget.iterations_max,
            dollar_cap_hit=self.dollars_used() >= self._cfg.budget.dollars_max,
            wall_cap_hit=self.wall_seconds_used() >= self._cfg.budget.wall_seconds_max,
        )

    def snapshot(
        self,
        *,
        last_top3_run_ids: list[str],
        last_top3_metrics: list[float],
        iters_since_top3_change: int,
    ) -> BudgetSnapshot:
        snap = BudgetSnapshot(
            iterations_used=self.iterations_used(),
            iterations_max=self._cfg.budget.iterations_max,
            dollars_used=self.dollars_used(),
            dollars_max=self._cfg.budget.dollars_max,
            wall_seconds_used=self.wall_seconds_used(),
            wall_seconds_max=self._cfg.budget.wall_seconds_max,
            last_top3_run_ids=last_top3_run_ids,
            last_top3_metrics=last_top3_metrics,
            iters_since_top3_change=iters_since_top3_change,
            updated_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        out_path = self._state.state_dir / "budget.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path, snap)
        return snap
