from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from libs.autoloop.state import AutoloopConfig, BudgetSnapshot, IterationRecord

logger = logging.getLogger(__name__)


def _find_notify() -> str | None:
    found = shutil.which("slack-notify.sh")
    if found:
        return found
    candidate = (
        Path(os.environ.get("CLAUDE_SESSION_CWD", os.getcwd())) / ".claude/scripts/slack-notify.sh"
    )
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _send(message: str) -> None:
    notify = _find_notify()
    if notify is None:
        logger.info("slack-notify.sh not on PATH; skipping notification")
        return
    try:
        subprocess.run([notify, message], check=True, capture_output=True, text=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("slack-notify failed: %s", e)


def is_enabled(cfg: AutoloopConfig) -> bool:
    return _find_notify() is not None and bool(cfg.checkpoint.slack_milestone_iters)


def notify_start(cfg: AutoloopConfig) -> None:
    if not is_enabled(cfg):
        return
    msg = (
        f"Autoloop started: project={cfg.project}, "
        f"budget={cfg.budget.iterations_max} iters / ${cfg.budget.dollars_max:.0f}"
    )
    _send(msg)


def notify_milestone(
    cfg: AutoloopConfig,
    iter_id: int,
    rec: IterationRecord,
    snap: BudgetSnapshot,
) -> None:
    if not is_enabled(cfg):
        return
    if iter_id not in cfg.checkpoint.slack_milestone_iters:
        return
    top_metric = snap.last_top3_metrics[0] if snap.last_top3_metrics else None
    metric_str = f"{top_metric:.3f}" if top_metric is not None else "n/a"
    run_id = rec.executor.run_id if rec.executor else None
    lift = rec.executor.lift if rec.executor else None
    lift_str = (
        f"+{lift:.3f}"
        if lift is not None and lift >= 0
        else (f"{lift:.3f}" if lift is not None else "n/a")
    )
    last_run = f"last {run_id} lift {lift_str}" if run_id else "no run"
    msg = (
        f"[iter {iter_id}/{snap.iterations_max}] "
        f"top metric={metric_str}, "
        f"${snap.dollars_used:.2f}/${snap.dollars_max:.0f}, "
        f"{last_run}"
    )
    _send(msg)


def notify_termination(cfg: AutoloopConfig, reason: str, snap: BudgetSnapshot) -> None:
    if not is_enabled(cfg):
        return
    top_run = snap.last_top3_run_ids[0] if snap.last_top3_run_ids else "n/a"
    top_metric = snap.last_top3_metrics[0] if snap.last_top3_metrics else None
    metric_str = f"{top_metric:.3f}" if top_metric is not None else "n/a"
    msg = (
        f"Autoloop ended: {reason}. "
        f"{snap.iterations_used}/{snap.iterations_max} iters. "
        f"$ spent={snap.dollars_used:.2f}. "
        f"Top: {top_run} ({metric_str})"
    )
    _send(msg)


def notify_hard_failure(cfg: AutoloopConfig, reason: str, iter_id: int | None) -> None:
    if not is_enabled(cfg):
        return
    iter_str = str(iter_id) if iter_id is not None else "unknown"
    msg = f"Autoloop ABORTED at iter {iter_str}: {reason}"
    _send(msg)
