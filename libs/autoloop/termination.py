from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from libs.autoloop.budget import AutoloopBudget
from libs.autoloop.state import AutoloopConfig, IterationRecord, State, TerminationCheck


@dataclass
class TerminationVerdict:
    should_stop: bool
    reason: str | None
    details: dict
    checks: TerminationCheck


def _top_k_run_ids(
    records: list[IterationRecord],
    k: int,
    direction: Literal["higher_is_better", "lower_is_better"],
) -> list[str]:
    pairs: list[tuple[str, float]] = []
    for rec in records:
        if (
            rec.executor is not None
            and rec.executor.run_id
            and rec.executor.harness_metric is not None
        ):
            pairs.append((rec.executor.run_id, rec.executor.harness_metric))

    if not pairs:
        return []

    reverse = direction == "higher_is_better"
    pairs.sort(key=lambda p: p[1], reverse=reverse)

    seen: list[str] = []
    for run_id, _ in pairs:
        if run_id not in seen:
            seen.append(run_id)
        if len(seen) == k:
            break
    return seen


def _streak_count(
    records: list[IterationRecord],
    predicate: Callable[[IterationRecord], bool],
) -> int:
    count = 0
    for rec in reversed(records):
        if predicate(rec):
            count += 1
        else:
            break
    return count


def brainstorm_has_queued(state: State) -> bool:
    """Returns True if there is at least one queued item in brainstorm.jsonl.
    Called by the supervisor after the planner phase — not part of check_all.
    brainstorm exhaustion is checked by the supervisor via state.pop_top_queued;
    this helper gives a cheap boolean for logging/early-exit decisions.
    """
    items = state.load_brainstorm()
    return any(i.status == "queued" for i in items)


def check_all(
    *,
    state: State,
    cfg: AutoloopConfig,
    budget: AutoloopBudget,
    project_root: Path,
) -> TerminationVerdict:
    state_dir = state.state_dir
    checks = TerminationCheck()
    details: dict = {}

    all_records = state.recent_iterations(n=10_000)

    # 1. Manual STOP file
    checks.manual_stop_present = (state_dir / "STOP").exists()
    if checks.manual_stop_present:
        return TerminationVerdict(
            should_stop=True,
            reason="manual_stop",
            details=details,
            checks=checks,
        )

    # 2–4. Budget caps
    caps = budget.caps_exceeded()
    checks.iter_cap_hit = caps.iter_cap_hit
    checks.dollar_cap_hit = caps.dollar_cap_hit
    checks.wall_cap_hit = caps.wall_cap_hit

    details["iterations_used"] = budget.iterations_used()
    details["dollars_used"] = budget.dollars_used()
    details["wall_seconds_used"] = budget.wall_seconds_used()

    if caps.iter_cap_hit:
        return TerminationVerdict(
            should_stop=True, reason="iter_cap", details=details, checks=checks
        )
    if caps.dollar_cap_hit:
        return TerminationVerdict(
            should_stop=True, reason="dollar_cap", details=details, checks=checks
        )
    if caps.wall_cap_hit:
        return TerminationVerdict(
            should_stop=True, reason="wall_cap", details=details, checks=checks
        )

    if not all_records:
        return TerminationVerdict(should_stop=False, reason=None, details=details, checks=checks)

    # 5. Plateau — compare top-K across last window_iters windows
    k = cfg.plateau.top_k
    window = cfg.plateau.window_iters
    direction = cfg.metric_direction

    current_top_k = _top_k_run_ids(all_records, k, direction)
    details["last_top_k"] = current_top_k

    if len(all_records) >= window and current_top_k:
        plateau_hit = True
        for w in range(1, window):
            window_top_k = _top_k_run_ids(all_records[:-w], k, direction)
            if window_top_k != current_top_k:
                plateau_hit = False
                break
        checks.plateau_hit = plateau_hit
        if plateau_hit:
            return TerminationVerdict(
                should_stop=True, reason="plateau", details=details, checks=checks
            )

    # 6. Loop fingerprint — last 3 identical and non-empty
    recent_fps = [rec.iteration_fingerprint for rec in all_records[-3:]]
    if (
        len(recent_fps) == 3
        and all(fp == recent_fps[0] for fp in recent_fps)
        and recent_fps[0] != ""
    ):
        return TerminationVerdict(
            should_stop=True,
            reason="loop_fingerprint",
            details=details,
            checks=checks,
        )

    # 7. Dedup-rejected streak
    dedup_streak = _streak_count(
        all_records,
        lambda rec: (
            rec.planner is not None
            and rec.planner.brainstorm_added == 0
            and rec.planner.brainstorm_reranked == 0
        ),
    )
    checks.dedup_streak = dedup_streak
    details["dedup_streak"] = dedup_streak
    if dedup_streak >= cfg.loop_detection.dedup_reject_streak:
        return TerminationVerdict(
            should_stop=True, reason="dedup_streak", details=details, checks=checks
        )

    # 9. Planner-crash streak
    planner_crash_streak = _streak_count(
        all_records,
        lambda rec: rec.planner is not None and rec.planner.exit_code != 0,
    )
    checks.planner_crash_streak = planner_crash_streak
    details["planner_crash_streak"] = planner_crash_streak
    if planner_crash_streak >= 2:
        return TerminationVerdict(
            should_stop=True,
            reason="planner_crash_streak",
            details=details,
            checks=checks,
        )

    # 10. Executor-crash streak
    executor_crash_streak = _streak_count(
        all_records,
        lambda rec: rec.executor is not None and rec.executor.exit_code != 0,
    )
    checks.executor_crash_streak = executor_crash_streak
    details["executor_crash_streak"] = executor_crash_streak
    if executor_crash_streak >= 3:
        return TerminationVerdict(
            should_stop=True,
            reason="executor_crash_streak",
            details=details,
            checks=checks,
        )

    return TerminationVerdict(should_stop=False, reason=None, details=details, checks=checks)
