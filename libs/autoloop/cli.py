from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from libs.autoloop.init import init_project
from libs.autoloop.prereqs import check_prereqs
from libs.autoloop.state import AutoloopConfig, IterationRecord, read_jsonl_iter
from libs.autoloop.supervisor import SupervisorArgs, run_supervisor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m libs.autoloop",
        description="Autonomous feature-engineering loop for ds3.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="ds3 repo root (default: cwd)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialize autoloop for a project.")
    p_init.add_argument("--project", required=True, help="Project name.")
    p_init.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Skip all prompts; write defaults.",
    )
    p_init.add_argument(
        "--defaults-only",
        action="store_true",
        default=False,
        help="Write defaults without prompting (no seed items).",
    )

    # check-prereqs
    p_prereqs = sub.add_parser("check-prereqs", help="Check project prerequisites.")
    p_prereqs.add_argument("--project", required=True, help="Project name.")

    # run
    p_run = sub.add_parser("run", help="Run the autoloop supervisor.")
    p_run.add_argument("--project", required=True, help="Project name.")
    p_run.add_argument(
        "--budget", type=int, default=None, metavar="N", help="Override iterations budget."
    )
    p_run.add_argument(
        "--dollars", type=float, default=None, metavar="X", help="Override dollar budget."
    )
    p_run.add_argument(
        "--wall-hours",
        type=float,
        default=None,
        metavar="H",
        help="Override wall-clock budget in hours.",
    )
    p_run.add_argument("--dry-run", action="store_true", default=False)
    p_run.add_argument("--resume", action="store_true", default=False)
    p_run.add_argument(
        "--once", action="store_true", default=False, help="Run a single iteration then stop."
    )
    p_run.add_argument("--no-gap-finder", action="store_true", default=False)
    p_run.add_argument(
        "--from-iter", type=int, default=None, metavar="N", help="Skip iterations before N."
    )
    p_run.add_argument("--verbose", action="store_true", default=False)

    # status
    p_status = sub.add_parser("status", help="Show autoloop status for a project.")
    p_status.add_argument("--project", required=True, help="Project name.")

    # stop
    p_stop = sub.add_parser("stop", help="Write STOP sentinel to halt a running autoloop.")
    p_stop.add_argument("--project", required=True, help="Project name.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "init":
        return _cmd_init(args)
    elif args.cmd == "check-prereqs":
        return _cmd_check_prereqs(args)
    elif args.cmd == "run":
        return _cmd_run(args)
    elif args.cmd == "status":
        return _cmd_status(args)
    elif args.cmd == "stop":
        return _cmd_stop(args)
    else:
        print(f"Unknown command: {args.cmd}", file=sys.stderr)
        return 2


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        init_project(
            args.project_root,
            args.project,
            non_interactive=args.non_interactive,
            defaults_only=args.defaults_only,
        )
        return 0
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except SystemExit as e:
        return int(e.code) if e.code is not None else 1


def _cmd_check_prereqs(args: argparse.Namespace) -> int:
    report = check_prereqs(args.project_root, args.project)
    print(report.format_table(use_color=sys.stdout.isatty()))
    return 0 if report.all_passed else 1


def _cmd_run(args: argparse.Namespace) -> int:
    supervisor_args = SupervisorArgs(
        project=args.project,
        project_root=args.project_root,
        budget_override_iters=args.budget,
        budget_override_dollars=args.dollars,
        wall_hours_override=args.wall_hours,
        dry_run=args.dry_run,
        resume=args.resume,
        once=args.once,
        no_gap_finder=args.no_gap_finder,
        from_iter=args.from_iter,
        verbose=args.verbose,
    )
    return asyncio.run(run_supervisor(supervisor_args))


def _cmd_status(args: argparse.Namespace) -> int:
    project_root: Path = args.project_root
    project: str = args.project
    state_dir = project_root / "projects" / project / "autoloop"
    config_path = state_dir / "config.yaml"

    if not config_path.exists():
        print(f"Autoloop not initialized for project '{project}'.")
        print(f"Run: python -m libs.autoloop init --project {project}")
        return 1

    cfg = AutoloopConfig.load(project_root / "projects" / project)

    iterations_used = 0
    all_iters: list[IterationRecord] = []
    iterations_path = state_dir / "iterations.jsonl"
    if iterations_path.exists():
        for row in read_jsonl_iter(iterations_path):
            if row.get("schema") == "iteration/v1":
                iterations_used += 1
                try:
                    all_iters.append(IterationRecord.model_validate(row))
                except Exception:
                    pass

    dollars_used: float = 0.0
    wall_seconds_used: float = 0.0
    top3_run_ids: list[str] = []
    top3_metrics: list[float] = []

    budget_path = state_dir / "budget.json"
    if budget_path.exists():
        try:
            snap = json.loads(budget_path.read_text())
            dollars_used = float(snap.get("dollars_used", 0.0))
            wall_seconds_used = float(snap.get("wall_seconds_used", 0.0))
            top3_run_ids = snap.get("last_top3_run_ids", [])
            top3_metrics = snap.get("last_top3_metrics", [])
        except (json.JSONDecodeError, OSError):
            pass

    wall_hours_used = wall_seconds_used / 3600.0
    wall_hours_max = cfg.budget.wall_seconds_max / 3600.0

    use_utf8 = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") in (
        "utf8",
        "utf-8",
    )
    ok_mark = "✓" if use_utf8 else "[ok]"
    fail_mark = "✗" if use_utf8 else "[fail]"

    top3_str = "  —"
    if top3_run_ids:
        pairs = []
        for rid, met in zip(top3_run_ids, top3_metrics):
            pairs.append(f"{rid} ({met:.3f})")
        for rid in top3_run_ids[len(top3_metrics) :]:
            pairs.append(rid)
        top3_str = ", ".join(pairs)

    print(f"\nAutoloop status — {project}\n")
    print(f"Iterations:  {iterations_used} / {cfg.budget.iterations_max}")
    print(f"Spent:       ${dollars_used:.2f} / ${cfg.budget.dollars_max:.2f}")
    print(f"Wall:        {wall_hours_used:.1f}h / {wall_hours_max:.0f}h")
    print(f"Last top-3:  {top3_str}")

    last3 = all_iters[-3:] if len(all_iters) >= 3 else all_iters
    if last3:
        print("\nLast 3 iterations:")
        for rec in last3:
            exc = rec.executor
            planner = rec.planner

            ok = exc is not None and exc.exit_code == 0 and exc.run_record_status == "completed"
            mark = ok_mark if ok else fail_mark

            if exc is not None and exc.run_id:
                metric_str = (
                    f"metric={exc.harness_metric:.3f}"
                    if exc.harness_metric is not None
                    else "metric=?"
                )
                if exc.lift is not None:
                    lift_sign = "+" if exc.lift >= 0 else "−"
                    lift_str = f"  lift={lift_sign}{abs(exc.lift):.3f}"
                else:
                    lift_str = ""
                iter_dollars = exc.dollars if exc else 0.0
                iter_dollars += planner.dollars if planner else 0.0
                print(
                    f"  iter {rec.iteration_id:>3}  {mark}  {exc.run_id}  {metric_str}{lift_str}  ${iter_dollars:.2f}"
                )
            else:
                kill = exc.kill_reason if exc else (planner.kill_reason if planner else None)
                reason_str = f"(executor crashed: {kill})" if kill else "(no run)"
                iter_dollars = (exc.dollars if exc else 0.0) + (planner.dollars if planner else 0.0)
                print(f"  iter {rec.iteration_id:>3}  {mark}  {reason_str}  ${iter_dollars:.2f}")

    print()
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    state_dir = args.project_root / "projects" / args.project / "autoloop"
    stop_path = state_dir / "STOP"
    state_dir.mkdir(parents=True, exist_ok=True)
    stop_path.touch()
    print(f"STOP sentinel written: {stop_path}")
    print("The running autoloop will halt at the next iteration boundary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
