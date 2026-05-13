from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

from libs.autoloop.state import SessionResult
from libs.costs.tracker import CostTracker

logger = logging.getLogger(__name__)


def run_claude_session(
    *,
    role: Literal["planner", "executor"],
    prompt: str,
    cwd: Path,
    settings_path: Path,
    allowed_tools: list[str],
    denied_tools: list[str],
    max_wall_seconds: float,
    max_dollars: float,
    cost_tracker: CostTracker,
    cost_key: str,
    log_path: Path,
    cli_path: str,
    project: str,
    iteration_id: int,
    model: str,
    max_turns: int = 200,
) -> SessionResult:
    cmd = [
        cli_path,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--settings",
        str(settings_path),
        "--permission-mode",
        "bypassPermissions",  # see PreToolUse hook + settings deny for actual safety
        "--allowed-tools",
        ",".join(allowed_tools),
        "--disallowed-tools",
        ",".join(denied_tools),
        "--max-turns",
        str(max_turns),
        "--model",
        model,
    ]
    env = {
        **os.environ,
        "AUTOLOOP_COST_KEY": cost_key,
        "AUTOLOOP_PROJECT": project,
        "AUTOLOOP_ITER_ID": str(iteration_id),
    }

    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return _watch_session(
        proc=proc,
        role=role,
        log_path=log_path,
        max_wall_seconds=max_wall_seconds,
        max_dollars=max_dollars,
        cost_tracker=cost_tracker,
        cost_key=cost_key,
        model=model,
        start=start,
    )


def _kill(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _watch_session(
    *,
    proc: subprocess.Popen,  # type: ignore[type-arg]
    role: Literal["planner", "executor"],
    log_path: Path,
    max_wall_seconds: float,
    max_dollars: float,
    cost_tracker: CostTracker,
    cost_key: str,
    model: str,
    start: float,
) -> SessionResult:
    kill_reason: dict[str, str | None] = {"value": None}

    # Watchdog thread polls wall-clock and cost budget while the main thread
    # blocks on stdout readline. Without this, a session that produces no output
    # for many seconds would block budget enforcement indefinitely.
    def watchdog() -> None:
        while proc.poll() is None:
            time.sleep(5)
            elapsed = time.monotonic() - start
            if elapsed > max_wall_seconds:
                kill_reason["value"] = "wall_cap"
                _kill(proc)
                return
            try:
                if cost_tracker.get(cost_key).total > max_dollars:
                    kill_reason["value"] = "dollar_cap"
                    _kill(proc)
                    return
            except Exception:
                pass

    watch_thread = threading.Thread(target=watchdog, daemon=True)
    watch_thread.start()

    session_id: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    tools_used_counts: dict[str, int] = {}
    wrote_files: list[str] = []
    result_cost_usd: float | None = None

    try:
        with log_path.open("a") as log_f:
            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    try:
                        event = json.loads(raw_line)
                        log_f.write(json.dumps(event) + "\n")
                        log_f.flush()
                    except json.JSONDecodeError:
                        logger.warning("stream-json parse error — raw: %s", raw_line.rstrip())
                        try:
                            log_f.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                            log_f.flush()
                        except Exception:
                            pass
                        continue
                    except Exception as exc:
                        logger.warning("error writing log line: %s", exc)
                        continue

                    event_type = event.get("type")

                    if event_type == "system" and event.get("subtype") == "init":
                        session_id = event.get("session_id") or session_id

                    elif event_type == "assistant":
                        message = event.get("message") or {}
                        usage = message.get("usage") or {}
                        tokens_in += int(usage.get("input_tokens") or 0)
                        tokens_out += int(usage.get("output_tokens") or 0)
                        content = message.get("content") or []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                tool_name: str = block.get("name") or ""
                                if tool_name:
                                    tools_used_counts[tool_name] = (
                                        tools_used_counts.get(tool_name, 0) + 1
                                    )
                                if tool_name in {"Edit", "Write", "MultiEdit"}:
                                    inp = block.get("input") or {}
                                    file_path = inp.get("file_path")
                                    if file_path and isinstance(file_path, str):
                                        wrote_files.append(file_path)

                    elif event_type == "result":
                        raw_cost = event.get("total_cost_usd")
                        if raw_cost is not None:
                            try:
                                result_cost_usd = float(raw_cost)
                            except (TypeError, ValueError):
                                pass

    except Exception as exc:
        logger.warning("error reading session stdout: %s", exc)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    watch_thread.join(timeout=10)

    if result_cost_usd is not None:
        dollars = result_cost_usd
    else:
        try:
            dollars = cost_tracker.get(cost_key).total
        except Exception:
            dollars = 0.0

    raw_kill = kill_reason.get("value")
    if raw_kill in {"wall_cap", "dollar_cap"}:
        resolved_kill: Literal["normal", "wall_cap", "dollar_cap", "crashed"] = raw_kill  # type: ignore[assignment]
    elif (proc.returncode or 0) == 0:
        resolved_kill = "normal"
    else:
        resolved_kill = "crashed"

    return SessionResult(
        role=role,
        session_id=session_id,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        dollars=dollars,
        log_path=str(log_path),
        wall_seconds=time.monotonic() - start,
        tools_used_counts=tools_used_counts,
        wrote_files=wrote_files,
        kill_reason=resolved_kill,
    )
