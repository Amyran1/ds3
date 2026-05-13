from __future__ import annotations

import fnmatch
import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from libs.autoloop import (
    curriculum,
    prereqs,
    registry,
    retention,
    session,
    slack,
    termination,
)
from libs.autoloop.budget import AutoloopBudget
from libs.autoloop.failure_modes import (
    append_failure_mode,
    extract_failure_fingerprint,
    format_for_prompt,
    recent_failures,
)
from libs.autoloop.progress import Heartbeat, event, fmt_dur
from libs.autoloop.state import (
    AutoloopConfig,
    BrainstormItem,
    BudgetSnapshot,
    ExecutorResult,
    ExecutorSummary,
    IterationRecord,
    PlannerResult,
    PlannerSummary,
    SessionResult,
    State,
    TerminationCheck,
    read_jsonl_iter,
)
from libs.container import Container

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class SupervisorArgs:
    project: str
    project_root: Path
    budget_override_iters: int | None = None
    budget_override_dollars: float | None = None
    wall_hours_override: float | None = None
    dry_run: bool = False
    resume: bool = False
    once: bool = False
    no_gap_finder: bool = False
    from_iter: int | None = None
    verbose: bool = False


class SupervisorExitCode:
    SUCCESS = 0
    ITER_BUDGET_HIT = 10
    DOLLAR_BUDGET_HIT = 11
    WALL_BUDGET_HIT = 12
    PLATEAU = 13
    LOOP_REPEATING = 14
    LOOP_NO_NEW_IDEAS = 15
    BRAINSTORM_EXHAUSTED = 16
    PLANNER_FAILED = 20
    EXECUTOR_FAILED = 21
    MANUAL_STOP = 30
    PREREQ_FAILED = 40
    UNEXPECTED_ERROR = 99


_REASON_TO_EXIT_CODE: dict[str, int] = {
    "iter_cap": SupervisorExitCode.ITER_BUDGET_HIT,
    "dollar_cap": SupervisorExitCode.DOLLAR_BUDGET_HIT,
    "wall_cap": SupervisorExitCode.WALL_BUDGET_HIT,
    "plateau": SupervisorExitCode.PLATEAU,
    "loop_fingerprint": SupervisorExitCode.LOOP_REPEATING,
    "dedup_streak": SupervisorExitCode.LOOP_NO_NEW_IDEAS,
    "brainstorm_exhausted": SupervisorExitCode.BRAINSTORM_EXHAUSTED,
    "planner_crash_streak": SupervisorExitCode.PLANNER_FAILED,
    "executor_crash_streak": SupervisorExitCode.EXECUTOR_FAILED,
    "manual_stop": SupervisorExitCode.MANUAL_STOP,
}


async def run_supervisor(args: SupervisorArgs) -> int:
    report = prereqs.check_prereqs(args.project_root, args.project)
    if not report.all_passed:
        print(report.format_table(use_color=True))
        return SupervisorExitCode.PREREQ_FAILED

    cfg = AutoloopConfig.load(args.project_root / "projects" / args.project)
    if args.budget_override_iters is not None:
        cfg.budget.iterations_max = args.budget_override_iters
    if args.budget_override_dollars is not None:
        cfg.budget.dollars_max = args.budget_override_dollars
    if args.wall_hours_override is not None:
        cfg.budget.wall_seconds_max = int(args.wall_hours_override * 3600)

    state = State(args.project_root, cfg)
    state.state_dir.mkdir(parents=True, exist_ok=True)
    (state.state_dir / "logs").mkdir(exist_ok=True)
    (state.state_dir / "dedup_index").mkdir(exist_ok=True)

    async with Container() as container:
        budget = AutoloopBudget(
            cfg=cfg,
            state=state,
            tracker=container.costs,
            first_iter_ts=_get_or_set_first_iter_ts(state),
        )

        slack.notify_start(cfg)

        try:
            return await _outer_loop(args, cfg, state, container, budget)
        except Exception as e:
            logger.exception("Unexpected supervisor error")
            slack.notify_hard_failure(cfg, f"Unexpected: {e}", iter_id=None)
            return SupervisorExitCode.UNEXPECTED_ERROR


async def _outer_loop(
    args: SupervisorArgs,
    cfg: AutoloopConfig,
    state: State,
    container: Container,
    budget: AutoloopBudget,
) -> int:
    planner_fail_streak = 0
    executor_fail_streak = 0

    while True:
        verdict = termination.check_all(
            state=state,
            cfg=cfg,
            budget=budget,
            project_root=args.project_root,
        )
        if verdict.should_stop:
            snap = budget.snapshot(
                last_top3_run_ids=verdict.details.get("last_top_k", [])[:3],
                last_top3_metrics=[],
                iters_since_top3_change=0,
            )
            event(
                f"TERMINATE · reason={verdict.reason} · iters_used={snap.iterations_used}"
                f" · dollars_used=${snap.dollars_used:.2f}"
            )
            slack.notify_termination(cfg, verdict.reason or "unknown", snap)
            _write_final_report(state, cfg, verdict, snap)
            return _verdict_to_exit_code(verdict.reason)

        iter_id = state.next_iteration_id()

        if args.from_iter is not None and iter_id < args.from_iter:
            state.append_iteration(
                IterationRecord(
                    iteration_id=iter_id,
                    ts_start=datetime.now(tz=timezone.utc).isoformat(),
                    ts_end=datetime.now(tz=timezone.utc).isoformat(),
                )
            )
            continue

        ts_start = datetime.now(tz=timezone.utc).isoformat()
        iter_rec = IterationRecord(iteration_id=iter_id, ts_start=ts_start)

        _iter_start_snap = budget.snapshot(
            last_top3_run_ids=[], last_top3_metrics=[], iters_since_top3_change=0
        )
        decision = curriculum.decide(
            iter_id, cfg.budget.iterations_max, cfg.curriculum.explore_fraction
        )
        run_gap_finder = not args.no_gap_finder and (
            (cfg.gap_finder.cadence == "every_n_iters" and (iter_id % cfg.gap_finder.n == 0))
            or cfg.gap_finder.cadence == "every_iter"
        )

        queue_size = len([i for i in state.load_brainstorm() if i.status == "queued"])
        event(
            f"iter {iter_id}/{cfg.budget.iterations_max} · start"
            f" · queue={queue_size}"
            f" · phase={decision.phase.lower()}"
            f" · cumulative=${_iter_start_snap.dollars_used:.2f}/${_iter_start_snap.dollars_max:.2f}"
        )

        event(f"iter {iter_id}/{cfg.budget.iterations_max} · planner.spawn")
        _planner_hb = Heartbeat(f"iter {iter_id}/{cfg.budget.iterations_max} · planner")
        _planner_hb.start()
        try:
            planner_result = await _run_planner(
                args, cfg, state, container, budget, iter_id, decision, run_gap_finder
            )
            iter_rec.planner = planner_result
            _planner_hb.stop()
            if planner_result.exit_code != 0:
                planner_fail_streak += 1
                event(
                    f"iter {iter_id}/{cfg.budget.iterations_max} · planner.crashed"
                    f" · kill_reason={planner_result.kill_reason}"
                    f" · ${planner_result.dollars:.2f}"
                    f" · {fmt_dur(planner_result.wall_seconds)}"
                )
            else:
                planner_fail_streak = 0
                event(
                    f"iter {iter_id}/{cfg.budget.iterations_max} · planner.done"
                    f" · added={planner_result.brainstorm_added}"
                    f" reranked={planner_result.brainstorm_reranked}"
                    f" · ${planner_result.dollars:.2f}"
                    f" · {fmt_dur(planner_result.wall_seconds)}"
                    f" · exit={planner_result.kill_reason or 'normal'}"
                )
        except Exception:
            logger.exception("planner phase crashed")
            _planner_hb.stop()
            planner_fail_streak += 1
            iter_rec.planner = PlannerResult(
                role="planner",
                session_id=None,
                exit_code=-1,
                model=cfg.session.planner_model,
                log_path="",
                wall_seconds=0.0,
                kill_reason="crashed",
            )
            event(
                f"iter {iter_id}/{cfg.budget.iterations_max} · planner.crashed"
                f" · kill_reason=exception · $0.00 · 0s"
            )

        if iter_rec.planner is not None and iter_rec.planner.exit_code != 0:
            await _capture_failure(
                container, state, iter_id, "planner", iter_rec.planner, args.project
            )
            if planner_fail_streak >= 2:
                state.append_iteration(_finalize_iter_rec(iter_rec, budget))
                slack.notify_hard_failure(cfg, "planner crash streak", iter_id)
                return SupervisorExitCode.PLANNER_FAILED
            state.append_iteration(_finalize_iter_rec(iter_rec, budget))
            continue

        registry.rebuild_registry(state)

        if not termination.brainstorm_has_queued(state):
            iter_rec.executor = None
            iter_rec.termination_checks = TerminationCheck(brainstorm_exhausted=True)
            state.append_iteration(_finalize_iter_rec(iter_rec, budget))
            snap = budget.snapshot(
                last_top3_run_ids=[], last_top3_metrics=[], iters_since_top3_change=0
            )
            slack.notify_termination(cfg, "brainstorm_exhausted", snap)
            _write_final_report(
                state,
                cfg,
                termination.TerminationVerdict(
                    should_stop=True,
                    reason="brainstorm_exhausted",
                    checks=iter_rec.termination_checks,
                    details={},
                ),
                snap,
            )
            return SupervisorExitCode.BRAINSTORM_EXHAUSTED

        top = state.pop_top_queued()
        if top is None:
            state.append_iteration(_finalize_iter_rec(iter_rec, budget))
            continue

        _title = top.title[:60] + ("..." if len(top.title) > 60 else "")
        event(f'iter {iter_id}/{cfg.budget.iterations_max} · executor.spawn · idea="{_title}"')
        _executor_hb = Heartbeat(f"iter {iter_id}/{cfg.budget.iterations_max} · executor")
        _executor_hb.start()
        try:
            executor_result = await _run_executor(args, cfg, state, container, budget, iter_id, top)
            iter_rec.executor = executor_result
            _executor_hb.stop()
            if executor_result.exit_code != 0 or executor_result.run_record_status not in (
                "completed",
            ):
                executor_fail_streak += 1
                state.mark_failed(
                    top.id, iteration_id=iter_id, reason=str(executor_result.kill_reason)
                )
                event(
                    f"iter {iter_id}/{cfg.budget.iterations_max} · executor.crashed"
                    f" · kill_reason={executor_result.kill_reason}"
                    f" · ${executor_result.dollars:.2f}"
                    f" · {fmt_dur(executor_result.wall_seconds)}"
                )
            else:
                executor_fail_streak = 0
                state.mark_consumed(
                    top.id,
                    iteration_id=iter_id,
                    run_id=executor_result.run_id,
                    status="run_completed",
                )
                _outcome = (
                    "shipped"
                    if executor_result.run_record_status == "completed"
                    else (
                        "skipped"
                        if executor_result.run_record_status == "failed_validation"
                        else "failed"
                    )
                )
                _metric_str = (
                    f"{executor_result.harness_metric:.4f}"
                    if executor_result.harness_metric is not None
                    else "NA"
                )
                event(
                    f"iter {iter_id}/{cfg.budget.iterations_max} · executor.done"
                    f" · run_id={executor_result.run_id or 'NA'}"
                    f" · metric={_metric_str}"
                    f" · ${executor_result.dollars:.2f}"
                    f" · {fmt_dur(executor_result.wall_seconds)}"
                    f" · outcome={_outcome}"
                )
        except Exception:
            logger.exception("executor phase crashed")
            _executor_hb.stop()
            executor_fail_streak += 1
            iter_rec.executor = ExecutorResult(
                role="executor",
                session_id=None,
                exit_code=-1,
                model=cfg.session.executor_model,
                log_path="",
                wall_seconds=0.0,
                kill_reason="crashed",
                brainstorm_id=top.id,
                run_id=None,
            )
            state.mark_failed(top.id, iteration_id=iter_id, reason="exception")
            event(
                f"iter {iter_id}/{cfg.budget.iterations_max} · executor.crashed"
                f" · kill_reason=exception · $0.00 · 0s"
            )

        if iter_rec.executor is not None and iter_rec.executor.exit_code != 0:
            await _capture_failure(
                container, state, iter_id, "executor", iter_rec.executor, args.project
            )
            if executor_fail_streak >= 3:
                state.append_iteration(_finalize_iter_rec(iter_rec, budget))
                slack.notify_hard_failure(cfg, "executor crash streak", iter_id)
                return SupervisorExitCode.EXECUTOR_FAILED

        snap = budget.snapshot(
            last_top3_run_ids=[],
            last_top3_metrics=[],
            iters_since_top3_change=0,
        )
        retention.prune_old_predictions(
            args.project_root,
            args.project,
            cfg.comparison_group,
            cfg.primary_metric,
            cfg.metric_direction,
            top_k=cfg.curriculum.exploit_top_k,
            current_run_id=(iter_rec.executor.run_id if iter_rec.executor else None),
        )
        iter_rec.budget_after = snap
        iter_rec.termination_checks = verdict.checks
        iter_rec.iteration_fingerprint = _compute_iteration_fingerprint(top, iter_rec.executor)
        iter_rec.ts_end = datetime.now(tz=timezone.utc).isoformat()

        state.append_iteration(iter_rec)

        commit_sha = _commit_iteration(
            project_root=args.project_root,
            project=args.project,
            iter_id=iter_id,
            brainstorm_id=top.id if top is not None else None,
            family_name=top.feature_family_name_hint if top is not None else None,
            metric=iter_rec.executor.harness_metric if iter_rec.executor else None,
            lift=iter_rec.executor.lift if iter_rec.executor else None,
        )
        if commit_sha:
            logger.info("autoloop iter %d committed as %s", iter_id, commit_sha[:8])
            _commit_subject = _get_commit_subject(args.project_root)
            event(
                f"iter {iter_id}/{cfg.budget.iterations_max} · committed"
                f" · sha={commit_sha[:7]}"
                f' · "{_commit_subject}"'
            )

        _iter_end_snap = budget.snapshot(
            last_top3_run_ids=[], last_top3_metrics=[], iters_since_top3_change=0
        )
        try:
            _iter_wall = (
                (
                    datetime.fromisoformat(iter_rec.ts_end)
                    - datetime.fromisoformat(iter_rec.ts_start)
                ).total_seconds()
                if iter_rec.ts_end and iter_rec.ts_start
                else 0.0
            )
        except (ValueError, TypeError):
            _iter_wall = 0.0
        _total_wall = _iter_end_snap.wall_seconds_used
        event(
            f"iter {iter_id}/{cfg.budget.iterations_max} · iter.done"
            f" · cumulative=${_iter_end_snap.dollars_used:.2f}/${_iter_end_snap.dollars_max:.2f}"
            f" · wall={fmt_dur(_iter_wall)}/{fmt_dur(_total_wall)}"
        )

        slack.notify_milestone(cfg, iter_id, iter_rec, snap)

        if args.once:
            return SupervisorExitCode.SUCCESS


def _render_template(template_path: Path, substitutions: dict[str, str]) -> str:
    text = template_path.read_text()
    for key, value in substitutions.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def _latest_champion_run_id(project_root: Path, project: str) -> str:
    champion_path = project_root / "projects" / project / "runs" / "CHAMPION.md"
    if not champion_path.exists():
        return ""
    try:
        content = champion_path.read_text()
        m = re.search(r"run_\d+", content)
        if m:
            return m.group()
    except OSError:
        pass
    return ""


def _next_run_number(project_root: Path, project: str) -> int:
    runs_dir = project_root / "projects" / project / "runs"
    max_n = 1
    for p in runs_dir.glob("run_*.py"):
        m = re.match(r"run_(\d+)\.py$", p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _phase_instruction(decision: curriculum.CurriculumDecision) -> str:
    if decision.phase == "EXPLORE":
        return (
            "EXPLORE phase: favor novel, diverse ideas. "
            "Eligible tiers: T0, T1, T2. "
            "Similarity threshold: 0.85. "
            "Rank by novelty and signal equally."
        )
    return (
        "EXPLOIT phase: favor high-signal ideas with evidence of success. "
        "Eligible tiers: T0, T1. "
        "Similarity threshold: 0.94. "
        "Rank heavily by signal and similar_success."
    )


def _write_settings(
    settings_dst: Path,
    hook_path: str,
) -> None:
    template_path = _PROMPTS_DIR / "session_settings_template.json"
    rendered = _render_template(template_path, {"hook_path": hook_path})
    settings_dst.write_text(rendered)


async def _run_planner(
    args: SupervisorArgs,
    cfg: AutoloopConfig,
    state: State,
    container: Container,
    budget: AutoloopBudget,
    iter_id: int,
    decision: curriculum.CurriculumDecision,
    run_gap_finder: bool,
) -> PlannerResult:
    failures = recent_failures(state.state_dir)
    failure_block = format_for_prompt(failures)
    dollars_remaining = cfg.budget.dollars_max - budget.dollars_used()
    champion_run_id = _latest_champion_run_id(args.project_root, args.project)

    rendered_prompt = _render_template(
        _PROMPTS_DIR / "planner.md",
        {
            "project": args.project,
            "iteration_id": str(iter_id),
            "iterations_max": str(cfg.budget.iterations_max),
            "dollars_remaining": f"{dollars_remaining:.2f}",
            "phase": decision.phase,
            "phase_instruction": _phase_instruction(decision),
            "run_gap_finder": "true" if run_gap_finder else "false",
            "latest_champion_run_id": champion_run_id,
            "failure_modes_block": failure_block,
        },
    )

    log_dir = state.state_dir / "logs"
    settings_path = log_dir / f"iter_{iter_id:03d}_planner_settings.json"
    hook_path = str(
        (args.project_root / "libs" / "autoloop" / "hooks" / "autoloop-write-guard.sh").resolve()
    )
    _write_settings(settings_path, hook_path)

    log_path = log_dir / f"iter_{iter_id:03d}_planner.jsonl"
    sess = session.run_claude_session(
        role="planner",
        prompt=rendered_prompt,
        cwd=args.project_root,
        settings_path=settings_path,
        allowed_tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash", "Agent"],
        denied_tools=["WebFetch", "WebSearch"],
        max_wall_seconds=300,
        max_dollars=cfg.budget.per_session_dollars_max,
        cost_tracker=container.costs,
        cost_key=f"autoloop:{args.project}:iter_{iter_id:03d}:planner",
        log_path=log_path,
        cli_path=cfg.session.resolved_cli_path(),
        project=args.project,
        iteration_id=iter_id,
        model=cfg.session.planner_model,
    )

    planner_summary = _read_planner_sentinel(state.state_dir, iter_id)
    if planner_summary is None and sess.exit_code == 0:
        sess = SessionResult(
            **{
                **sess.model_dump(),
                "exit_code": 1,
                "kill_reason": "crashed",
            }
        )

    added = planner_summary.added if planner_summary else 0
    reranked = planner_summary.reranked if planner_summary else 0
    skipped = planner_summary.skipped if planner_summary else 0
    ran_gf = planner_summary.ran_gap_finder if planner_summary else False

    return PlannerResult(
        **{
            **sess.model_dump(),
            "ran_gap_finder": ran_gf,
            "brainstorm_added": added,
            "brainstorm_reranked": reranked,
            "brainstorm_dedup_rejected": skipped,
        }
    )


def _read_planner_sentinel(state_dir: Path, iter_id: int) -> PlannerSummary | None:
    brainstorm_path = state_dir / "brainstorm.jsonl"
    if not brainstorm_path.exists():
        return None
    last_summary: PlannerSummary | None = None
    for row in read_jsonl_iter(brainstorm_path):
        if row.get("schema") == "planner_summary/v1" and row.get("iteration_id") == iter_id:
            try:
                last_summary = PlannerSummary.model_validate(row)
            except Exception:
                pass
    return last_summary


async def _run_executor(
    args: SupervisorArgs,
    cfg: AutoloopConfig,
    state: State,
    container: Container,
    budget: AutoloopBudget,
    iter_id: int,
    top: BrainstormItem,
) -> ExecutorResult:
    failures = recent_failures(state.state_dir)
    failure_block = format_for_prompt(failures)
    next_n = _next_run_number(args.project_root, args.project)

    rendered_prompt = _render_template(
        _PROMPTS_DIR / "executor.md",
        {
            "project": args.project,
            "iteration_id": str(iter_id),
            "per_session_wall_seconds_max": str(cfg.budget.per_session_wall_seconds_max),
            "per_session_dollars_max": f"{cfg.budget.per_session_dollars_max:.2f}",
            "brainstorm_item_json_pretty": top.model_dump_json(indent=2),
            "family_name": top.feature_family_name_hint,
            "N": str(next_n).zfill(2),
            "comparison_group": cfg.comparison_group,
            "sample_frac": str(cfg.sample_frac),
            "failure_modes_block": failure_block,
        },
    )

    log_dir = state.state_dir / "logs"
    settings_path = log_dir / f"iter_{iter_id:03d}_executor_settings.json"
    hook_path = str(
        (args.project_root / "libs" / "autoloop" / "hooks" / "autoloop-write-guard.sh").resolve()
    )
    _write_settings(settings_path, hook_path)

    log_path = log_dir / f"iter_{iter_id:03d}_executor.jsonl"
    sess = session.run_claude_session(
        role="executor",
        prompt=rendered_prompt,
        cwd=args.project_root,
        settings_path=settings_path,
        allowed_tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash", "Agent"],
        denied_tools=["WebFetch", "WebSearch"],
        max_wall_seconds=cfg.budget.per_session_wall_seconds_max,
        max_dollars=cfg.budget.per_session_dollars_max,
        cost_tracker=container.costs,
        cost_key=f"autoloop:{args.project}:iter_{iter_id:03d}:executor",
        log_path=log_path,
        cli_path=cfg.session.resolved_cli_path(),
        project=args.project,
        iteration_id=iter_id,
        model=cfg.session.executor_model,
    )

    executor_summary = _read_executor_sentinel(state.state_dir, iter_id)

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

    if executor_summary is not None:
        run_id = executor_summary.run_id
        harness_metric = executor_summary.metric
        baseline_metric = executor_summary.baseline_metric
        lift = executor_summary.lift
        verdict = executor_summary.verdict
        if verdict == "completed":
            ledger_status = _check_ledger_row(args.project_root, args.project, run_id)
            run_record_status = ledger_status
        elif verdict in ("bailed_timing_blowup", "bailed_unfeasible"):
            run_record_status = "failed_validation"
        else:
            run_record_status = "failed_harness"

    if executor_summary is None and sess.exit_code == 0:
        sess = SessionResult(
            **{
                **sess.model_dump(),
                "exit_code": 1,
                "kill_reason": "crashed",
            }
        )

    return ExecutorResult(
        **{
            **sess.model_dump(),
            "brainstorm_id": top.id,
            "run_id": run_id,
            "harness_metric": harness_metric,
            "baseline_metric": baseline_metric,
            "lift": lift,
            "run_record_status": run_record_status,
        }
    )


def _read_executor_sentinel(state_dir: Path, iter_id: int) -> ExecutorSummary | None:
    brainstorm_path = state_dir / "brainstorm.jsonl"
    if not brainstorm_path.exists():
        return None
    last_summary: ExecutorSummary | None = None
    for row in read_jsonl_iter(brainstorm_path):
        if row.get("schema") == "executor_summary/v1" and row.get("iteration_id") == iter_id:
            try:
                last_summary = ExecutorSummary.model_validate(row)
            except Exception:
                pass
    return last_summary


def _check_ledger_row(
    project_root: Path,
    project: str,
    run_id: str | None,
) -> Literal[
    "completed", "failed_validation", "failed_harness", "failed_recording", "missing", "unknown"
]:
    if not run_id:
        return "missing"
    results_path = project_root / "projects" / project / "runs" / "results.jsonl"
    if not results_path.exists():
        return "missing"
    for row in read_jsonl_iter(results_path):
        if row.get("run_id") == run_id:
            status = row.get("status", "")
            if status == "completed":
                return "completed"
            if isinstance(status, str) and status.startswith("failed_"):
                tail = status[len("failed_") :]
                if tail == "validation":
                    return "failed_validation"
                if tail == "recording":
                    return "failed_recording"
                return "failed_harness"
    return "missing"


async def _capture_failure(
    container: Container,
    state: State,
    iter_id: int,
    role: Literal["planner", "executor"],
    result: SessionResult,
    project: str,
) -> None:
    try:
        fm = await extract_failure_fingerprint(
            container=container,
            iteration_id=iter_id,
            role=role,
            session_log_path=Path(result.log_path),
            exit_code=result.exit_code,
            kill_reason=result.kill_reason,
            project=project,
        )
        append_failure_mode(state.state_dir, fm)
    except Exception:
        logger.exception("failure-mode extraction failed (non-fatal)")


def _compute_iteration_fingerprint(
    top: BrainstormItem | None,
    executor_result: ExecutorResult | None,
) -> str:
    if top is None:
        return ""
    tools_part = (
        ",".join(sorted((executor_result.tools_used_counts or {}).keys()))
        if executor_result
        else ""
    )
    files_part = ",".join((executor_result.wrote_files or [])[:5]) if executor_result else ""
    raw = f"{top.id}|{tools_part}|{files_part}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_or_set_first_iter_ts(state: State) -> datetime | None:
    iterations_path = state.state_dir / "iterations.jsonl"
    if iterations_path.exists():
        for row in read_jsonl_iter(iterations_path):
            ts_start = row.get("ts_start")
            if ts_start:
                try:
                    return datetime.fromisoformat(ts_start)
                except ValueError:
                    pass
    return datetime.now(tz=timezone.utc)


def _verdict_to_exit_code(reason: str | None) -> int:
    if reason is None:
        return SupervisorExitCode.SUCCESS
    return _REASON_TO_EXIT_CODE.get(reason, SupervisorExitCode.UNEXPECTED_ERROR)


def _write_final_report(
    state: State,
    cfg: AutoloopConfig,
    verdict: termination.TerminationVerdict,
    snap: BudgetSnapshot,
) -> None:
    now_str = datetime.now(tz=timezone.utc).isoformat()
    content = (
        f"# Autoloop final report\n\n"
        f"Project: {cfg.project}\n"
        f"Stopped at: {now_str}\n"
        f"Reason: {verdict.reason}\n"
        f"Iterations: {snap.iterations_used} / {snap.iterations_max}\n"
        f"Dollars: ${snap.dollars_used:.2f} / ${snap.dollars_max:.2f}\n"
        f"Wall: {snap.wall_seconds_used / 3600:.1f}h\n\n"
        f"Top runs: {snap.last_top3_run_ids}\n"
    )
    out_path = state.state_dir / "FINAL_REPORT.md"
    out_path.write_text(content)
    logger.info("Final report written to %s", out_path)


def _commit_iteration(
    *,
    project_root: Path,
    project: str,
    iter_id: int,
    brainstorm_id: str | None,
    family_name: str | None,
    metric: float | None,
    lift: float | None,
) -> str | None:
    # Scope-limited: only paths matching the autoloop allowlist are ever staged — never git-add -A.
    allowlist_globs = [
        f"projects/{project}/features/*/*/**",
        f"projects/{project}/features/*/**",
        f"projects/{project}/runs/run_*.py",
        f"projects/{project}/runs/run_*/**",
        f"projects/{project}/runs/results.jsonl",
        f"projects/{project}/runs/eda_results.jsonl",
        f"projects/{project}/runs/discovery_results.jsonl",
        f"projects/{project}/runs/CHAMPION.md",
        f"projects/{project}/autoloop/brainstorm.jsonl",
        f"projects/{project}/autoloop/iterations.jsonl",
        f"projects/{project}/autoloop/budget.json",
        f"projects/{project}/autoloop/failure_modes.jsonl",
        f"projects/{project}/autoloop/idea_registry.md",
        f"projects/{project}/autoloop/logs/iter_*.jsonl",
    ]

    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if status_proc.returncode != 0:
            logger.warning("auto-commit failed: git status returned %d", status_proc.returncode)
            return None

        dirty_paths: set[str] = set()
        for line in status_proc.stdout.splitlines():
            if len(line) < 3:
                continue
            path = line[3:].strip()
            if path:
                dirty_paths.add(path)

        to_add: list[str] = []
        seen: set[str] = set()
        for dirty in dirty_paths:
            for glob in allowlist_globs:
                if fnmatch.fnmatch(dirty, glob) and dirty not in seen:
                    to_add.append(dirty)
                    seen.add(dirty)
                    break

        if not to_add:
            logger.info("no autoloop-scope changes to commit for iter %d", iter_id)
            return None

        add_proc = subprocess.run(
            ["git", "add", "--", *to_add],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if add_proc.returncode != 0:
            logger.warning(
                "auto-commit failed: git add returned %d: %s", add_proc.returncode, add_proc.stderr
            )
            return None

        metric_str = f"{metric:.4f}" if metric is not None else "?"
        lift_str = f"{lift:+.4f}" if lift is not None else "?"
        family_str = family_name or "no-family"
        bid_str = brainstorm_id or "?"
        message = (
            f"autoloop iter {iter_id}: {family_str} ({bid_str})\n\n"
            f"metric={metric_str} lift={lift_str}\n\n"
            f"Auto-committed by libs.autoloop.supervisor."
        )

        commit_proc = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_proc.returncode != 0:
            logger.warning(
                "auto-commit failed: git commit returned %d: %s",
                commit_proc.returncode,
                commit_proc.stderr,
            )
            return None

        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if sha_proc.returncode != 0:
            logger.warning("auto-commit failed: git rev-parse returned %d", sha_proc.returncode)
            return None

        return sha_proc.stdout.strip()

    except Exception as err:
        logger.warning("auto-commit failed: %s", err)
        return None


def _get_commit_subject(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()[:80] if result.returncode == 0 else ""
    except Exception:
        return ""


def _finalize_iter_rec(iter_rec: IterationRecord, budget: AutoloopBudget) -> IterationRecord:
    if iter_rec.ts_end is None:
        iter_rec.ts_end = datetime.now(tz=timezone.utc).isoformat()
    if iter_rec.budget_after is None:
        iter_rec.budget_after = budget.snapshot(
            last_top3_run_ids=[], last_top3_metrics=[], iters_since_top3_change=0
        )
    return iter_rec
