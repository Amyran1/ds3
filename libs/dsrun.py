"""CLI wrapper for datascience scripts with timeout and progress.

Provides timeout enforcement, graceful shutdown via SIGUSR1, structured
summary output, and optional calibration tracking.

Usage:
    python -m lib.dsrun script.py [script-args...]
    python -m lib.dsrun --timeout 600 script.py --limit 50
    python -m lib.dsrun --resume script.py
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libs.progress_tracking.progress import read_progress
from libs.progress_tracking.slack import (
    post_message as slack_post,
)

if TYPE_CHECKING:
    import types

DSRUN_BASE_DIR = Path(tempfile.gettempdir()) / "dsrun"
GRACE_PERIOD_SEC = 10
DEFAULT_TIMEOUT_SEC = 300

_CALIBRATION_SAMPLE_LIMIT = 5
_CALIBRATION_FULL_LIMIT = 50
_CALIBRATION_LIMITS = {
    _CALIBRATION_SAMPLE_LIMIT,
    _CALIBRATION_FULL_LIMIT,
}

_GRACE_WAIT_SEC = 5
_NO_PROGRESS_HINT = (
    "Hint: Script did not report progress. Use ProgressReporter or tqdm for visibility."
)
_TIMEOUT_STATUS = "Status: timeout (no progress reported)"


def _stable_hash(script_path: str) -> str:
    """Compute a stable short hash of the script path.

    Args:
        script_path: Path to the script being run.

    Returns:
        Hex digest string (first 12 characters of SHA-256).
    """
    return hashlib.sha256(script_path.encode()).hexdigest()[:12]


def _parse_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse dsrun's own arguments from script arguments.

    Args:
        argv: Full argument list (excluding sys.argv[0]).

    Returns:
        Tuple of (dsrun namespace, remaining script + args).
    """
    parser = argparse.ArgumentParser(
        prog="dsrun",
        description=("Run datascience scripts with timeout and progress tracking."),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=(f"Timeout in seconds (default: {DEFAULT_TIMEOUT_SEC})"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume flag to child script",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration gate warning",
    )
    parser.add_argument(
        "--accept-slow",
        action="store_true",
        help="Override ETA warning for slow runs",
    )
    return parser.parse_known_args(argv)


def _detect_limit(script_args: list[str]) -> int | None:
    """Extract --limit value from script arguments if present.

    Args:
        script_args: The arguments being passed to the child.

    Returns:
        The limit value as int, or None if not found.
    """
    for i, arg in enumerate(script_args):
        if arg == "--limit" and i + 1 < len(script_args):
            try:
                return int(script_args[i + 1])
            except ValueError:
                return None
        if arg.startswith("--limit="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _load_calibration_state(
    script_path: str,
) -> dict[str, Any]:
    """Load the .dsrun_state.json file next to the script.

    Args:
        script_path: Path to the child script.

    Returns:
        Parsed state dict, or empty dict if missing/invalid.
    """
    state_file = Path(script_path).parent / ".dsrun_state.json"
    try:
        data: dict[str, Any] = json.loads(state_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data


def _save_calibration_state(
    script_path: str,
    state: dict[str, Any],
) -> None:
    """Save calibration state to .dsrun_state.json.

    Args:
        script_path: Path to the child script.
        state: State dict to persist.
    """
    state_file = Path(script_path).parent / ".dsrun_state.json"
    state_file.write_text(json.dumps(state, indent=2))


def _check_calibration(
    script_path: str,
    *,
    skip: bool,
) -> None:
    """Warn if calibration data is missing for a full run.

    Args:
        script_path: Path to the child script.
        skip: Whether the user passed --skip-calibration.
    """
    if skip:
        return
    state = _load_calibration_state(script_path)
    missing = []
    if "sample_5" not in state:
        missing.append("--limit 5")
    if "calibrate_50" not in state:
        missing.append("--limit 50")
    if missing:
        sys.stderr.write(
            "WARNING: No calibration data found for "
            f"{', '.join(missing)}. "
            "Consider running with --limit 5 then "
            "--limit 50 first.\n"
        )


def _save_calibration_result(
    script_path: str,
    limit: int,
    progress_name: str,
) -> None:
    """Save throughput from a calibration run to state file.

    Args:
        script_path: Path to the child script.
        limit: The --limit value used (5 or 50).
        progress_name: Progress directory name to read from.
    """
    if limit not in _CALIBRATION_LIMITS:
        return
    progress = read_progress(progress_name)
    if progress is None:
        return
    state = _load_calibration_state(script_path)
    key = "sample_5" if limit == _CALIBRATION_SAMPLE_LIMIT else "calibrate_50"
    state[key] = {
        "throughput_rec_s": progress.get("throughput_rec_s"),
        "elapsed_sec": progress.get("elapsed_sec"),
        "records_done": progress.get("records_done"),
    }
    _save_calibration_state(script_path, state)


def _print_summary_with_progress(
    script_path: str,
    status: str,
    progress: dict[str, Any],
) -> None:
    """Print summary when progress data is available.

    Args:
        script_path: Path to the child script.
        status: Final status string.
        progress: Parsed progress dict from the reporter.
    """
    done = progress.get("records_done", 0)
    total = progress.get("records_total", 0)
    pct = (done / total * 100) if total > 0 else 0.0
    throughput = progress.get("throughput_rec_s", 0)
    p_elapsed = progress.get("elapsed_sec", 0)
    eta = progress.get("eta_sec", 0)
    errors = progress.get("errors", 0)

    eta_min = eta / 60 if eta else 0
    lines = [
        "=== dsrun summary ===",
        f"Script: {script_path}",
        f"Status: {status}",
        f"Progress: {done}/{total} ({pct:.1f}%)",
        f"Throughput: {throughput} rec/s",
        f"Elapsed: {p_elapsed}s",
        f"ETA: {eta}s ({eta_min:.1f} min)",
        f"Errors: {errors}",
        "=== end dsrun ===",
    ]
    sys.stdout.write("\n".join(lines) + "\n")


def _print_summary_without_progress(
    script_path: str,
    status: str,
    elapsed: float,
) -> None:
    """Print summary when no progress data is available.

    Args:
        script_path: Path to the child script.
        status: Final status string.
        elapsed: Wall-clock seconds the child ran.
    """
    status_line = _TIMEOUT_STATUS if status == "timeout" else f"Status: {status}"
    lines = [
        "=== dsrun summary ===",
        f"Script: {script_path}",
        status_line,
        f"Elapsed: {elapsed:.1f}s",
        _NO_PROGRESS_HINT,
        "=== end dsrun ===",
    ]
    sys.stdout.write("\n".join(lines) + "\n")


def _print_summary(
    script_path: str,
    status: str,
    elapsed: float,
    progress_name: str,
) -> None:
    """Print a structured summary to stdout.

    Args:
        script_path: Path to the child script.
        status: Final status (complete, timeout, error).
        elapsed: Wall-clock seconds the child ran.
        progress_name: Progress dir name to read results from.
    """
    progress = read_progress(progress_name)
    if progress is not None:
        _print_summary_with_progress(script_path, status, progress)
    else:
        _print_summary_without_progress(script_path, status, elapsed)


def _forward_sigint(child_pid: int) -> None:
    """Install a SIGINT handler that forwards to the child.

    Args:
        child_pid: PID of the child process.
    """

    def _handler(
        _signum: int,
        _frame: types.FrameType | None,
    ) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGINT)

    signal.signal(signal.SIGINT, _handler)


def _launch_child(
    script_path: str,
    script_args: list[str],
    progress_dir: Path,
) -> int:
    """Launch the child script via posix_spawn.

    Args:
        script_path: Path to the Python script to run.
        script_args: Arguments to pass to the child script.
        progress_dir: Directory where progress JSON is written.

    Returns:
        The PID of the child process.
    """
    child_env = {
        **os.environ,
        "DSRUN_ACTIVE": "1",
        "DSRUN_PROGRESS_DIR": str(progress_dir),
    }
    argv = [sys.executable, script_path, *script_args]
    return os.posix_spawn(sys.executable, argv, child_env)


def _print_progress_tick(progress_name: str) -> None:
    """Read progress.json and print a one-line status to stdout.

    Args:
        progress_name: Progress directory name to read from.
    """
    progress = read_progress(progress_name)
    if progress is None:
        return
    done = progress.get("records_done", 0)
    total = progress.get("records_total", 0)
    pct = (done / total * 100) if total > 0 else 0.0
    throughput = progress.get("throughput_rec_s", 0)
    eta = progress.get("eta_sec", 0)
    eta_min = eta / 60 if eta else 0
    errors = progress.get("errors", 0)
    parts = [
        f"[dsrun] {done}/{total} ({pct:.1f}%)",
        f"{throughput} rec/s",
        f"ETA: {eta_min:.1f} min",
    ]
    if errors:
        parts.append(f"errors: {errors}")
    sys.stdout.write(" | ".join(parts) + "\n")
    sys.stdout.flush()


_PROGRESS_TICK_INTERVAL = 15.0


def _wait_for_child(
    child_pid: int,
    timeout: int,
    progress_name: str,
) -> tuple[str, int]:
    """Wait for child process, handling timeout.

    Uses polling with os.waitpid(WNOHANG) to implement timeout.
    Prints periodic progress ticks to stdout for visibility.

    Args:
        child_pid: PID of the child process.
        timeout: Maximum seconds to wait before timeout.
        progress_name: Progress directory name for reading progress.

    Returns:
        Tuple of (status string, exit code or -1).
    """
    deadline = time.monotonic() + timeout
    last_tick = time.monotonic()
    poll_interval = 0.1
    while time.monotonic() < deadline:
        pid, wait_status = os.waitpid(child_pid, os.WNOHANG)
        if pid != 0:
            code = _extract_exit_code(wait_status)
            if code != 0:
                return "error", code
            return "complete", 0
        now = time.monotonic()
        if now - last_tick >= _PROGRESS_TICK_INTERVAL:
            _print_progress_tick(progress_name)
            last_tick = now
        time.sleep(poll_interval)

    # Timeout -- attempt graceful shutdown
    _graceful_shutdown_pid(child_pid)
    return "timeout", -1


def _extract_exit_code(wait_status: int) -> int:
    """Extract exit code from os.waitpid status.

    Args:
        wait_status: Raw status from os.waitpid.

    Returns:
        The process exit code.
    """
    if os.WIFEXITED(wait_status):
        return os.WEXITSTATUS(wait_status)
    return 1


def _graceful_shutdown_pid(child_pid: int) -> None:
    """Attempt graceful shutdown of a child by PID.

    Sends SIGUSR1, waits a grace period, then SIGTERM/SIGKILL.

    Args:
        child_pid: PID of the child process.
    """
    with contextlib.suppress(ProcessLookupError, OSError):
        os.kill(child_pid, signal.SIGUSR1)

    # Wait grace period for child to exit
    deadline = time.monotonic() + GRACE_PERIOD_SEC
    while time.monotonic() < deadline:
        pid, _ = os.waitpid(child_pid, os.WNOHANG)
        if pid != 0:
            return
        time.sleep(0.1)

    # SIGTERM
    with contextlib.suppress(ProcessLookupError, OSError):
        os.kill(child_pid, signal.SIGTERM)

    term_deadline = time.monotonic() + _GRACE_WAIT_SEC
    while time.monotonic() < term_deadline:
        pid, _ = os.waitpid(child_pid, os.WNOHANG)
        if pid != 0:
            return
        time.sleep(0.1)

    # SIGKILL as last resort
    with contextlib.suppress(ProcessLookupError, OSError):
        os.kill(child_pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(child_pid, 0)


def _slack_notify(text: str, thread_ts: str | None = None) -> str | None:
    """Post a message to Slack, optionally as a thread reply.

    Args:
        text: Slack mrkdwn message text.
        thread_ts: If set, reply to this thread.

    Returns:
        The message ts for threading, or None.
    """
    return slack_post(text, thread_ts=thread_ts)


def _format_slack_message(
    script_path: str,
    status: str,
    elapsed: float,
    progress: dict[str, Any] | None,
    timeout: int,
) -> str:
    """Build a Slack mrkdwn message for the given status.

    Args:
        script_path: Path to the script being run.
        status: One of "started", "complete", "timeout", "error".
        elapsed: Wall-clock seconds.
        progress: Parsed progress dict, or None.
        timeout: The chunk timeout in seconds.

    Returns:
        Formatted Slack mrkdwn string.
    """
    name = Path(script_path).name

    if status == "started":
        return f"*dsrun* | `{name}` | timeout: {timeout}s"

    if progress is not None:
        return _format_slack_with_progress(status, elapsed, progress)

    # No progress data
    if status == "complete":
        return f"*Complete* | {elapsed:.0f}s | no progress data"
    if status == "timeout":
        return f"*Timeout* | {elapsed:.0f}s elapsed | no progress reported"
    return f"*Error* | {elapsed:.0f}s elapsed"


def _format_slack_with_progress(
    status: str,
    elapsed: float,
    progress: dict[str, Any],
) -> str:
    """Format Slack message when progress data is available.

    Args:
        status: One of "complete", "timeout", "error".
        elapsed: Wall-clock seconds.
        progress: Parsed progress dict.

    Returns:
        Formatted Slack mrkdwn string.
    """
    done = progress.get("records_done", 0)
    total = progress.get("records_total", 0)
    pct = (done / total * 100) if total > 0 else 0.0
    throughput = progress.get("throughput_rec_s", 0)
    errors = progress.get("errors", 0)

    if status == "complete":
        mins = elapsed / 60
        return (
            f"*Complete* | {done}/{total} | "
            f"{elapsed:.0f}s ({mins:.1f} min) | "
            f"{throughput} rec/s | errors: {errors}"
        )
    # timeout with progress = checkpoint
    eta = progress.get("eta_sec", 0)
    eta_min = eta / 60 if eta else 0
    return (
        f"*Checkpoint* | {done}/{total} ({pct:.1f}%) | "
        f"{throughput} rec/s | ETA: {eta_min:.1f} min | "
        f"errors: {errors}"
    )


def _clear_slack_thread_state(script_path: str) -> None:
    """Remove Slack thread state from .dsrun_state.json.

    Args:
        script_path: Path to the child script.
    """
    state = _load_calibration_state(script_path)
    changed = False
    if "slack_thread_ts" in state:
        del state["slack_thread_ts"]
        changed = True
    if "slack_run_started_at" in state:
        del state["slack_run_started_at"]
        changed = True
    if changed:
        _save_calibration_state(script_path, state)


def _init_slack_thread(
    script_path: str,
    *,
    is_resume: bool,
    timeout: int,
) -> str | None:
    """Initialize a Slack thread for a full run.

    Loads existing thread from state, resets if this is a fresh run,
    and posts an initial message if no thread exists yet.

    Args:
        script_path: Path to the child script.
        is_resume: Whether this is a --resume invocation.
        timeout: The chunk timeout in seconds.

    Returns:
        The Slack thread ts, or None if Slack is unavailable.
    """
    state = _load_calibration_state(script_path)
    thread_ts: str | None = state.get("slack_thread_ts")

    # Fresh full run (not resume) -- start new thread
    if not is_resume and thread_ts is not None:
        _clear_slack_thread_state(script_path)
        thread_ts = None

    # Post initial message if no thread exists
    if thread_ts is not None:
        return thread_ts

    msg = _format_slack_message(script_path, "started", 0.0, None, timeout)
    new_ts = _slack_notify(msg)
    if new_ts is None:
        return None

    state = _load_calibration_state(script_path)
    state["slack_thread_ts"] = new_ts
    state["slack_run_started_at"] = datetime.now(tz=timezone.utc).isoformat()
    _save_calibration_state(script_path, state)
    return new_ts


def _finalize_slack_thread(
    script_path: str,
    status: str,
    elapsed: float,
    progress: dict[str, Any] | None,
    thread_ts: str | None,
) -> None:
    """Post a status reply and clean up the Slack thread.

    Args:
        script_path: Path to the child script.
        status: Final status (complete, timeout, error).
        elapsed: Wall-clock seconds the child ran.
        progress: Parsed progress dict, or None.
        thread_ts: Slack thread ts, or None if no thread.
    """
    if thread_ts is not None:
        msg = _format_slack_message(
            script_path,
            status,
            elapsed,
            progress,
            0,
        )
        _slack_notify(msg, thread_ts=thread_ts)

    if status == "complete":
        _clear_slack_thread_state(script_path)


def run(argv: list[str] | None = None) -> int:
    """Main entry point for dsrun.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for timeout/error).
    """
    if argv is None:
        argv = sys.argv[1:]

    dsrun_args, remaining = _parse_args(argv)

    if not remaining:
        sys.stderr.write("Error: no script specified.\n")
        return 1

    script_path = remaining[0]
    script_args = remaining[1:]

    if dsrun_args.resume:
        script_args = ["--resume", *script_args]

    # Compute stable hash for the progress directory.
    resolved = str(Path(script_path).resolve())
    progress_name = _stable_hash(resolved)
    progress_dir = DSRUN_BASE_DIR / progress_name
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Calibration check for full (non-limited) runs.
    limit = _detect_limit(script_args)
    is_calibration = limit is not None
    if not is_calibration:
        _check_calibration(
            script_path,
            skip=dsrun_args.skip_calibration,
        )

    # --- Slack thread lifecycle ---
    thread_ts: str | None = None
    if not is_calibration:
        thread_ts = _init_slack_thread(
            script_path,
            is_resume=dsrun_args.resume,
            timeout=dsrun_args.timeout,
        )

    # Launch child and wait.
    child_pid = _launch_child(script_path, script_args, progress_dir)
    _forward_sigint(child_pid)

    t0 = time.monotonic()
    status, _exit_code = _wait_for_child(child_pid, dsrun_args.timeout, progress_name)
    elapsed = time.monotonic() - t0

    # Save calibration data if this was a limited run.
    if is_calibration:
        _save_calibration_result(script_path, limit, progress_name)

    _print_summary(script_path, status, elapsed, progress_name)

    # --- Slack status reply and cleanup ---
    if not is_calibration:
        progress = read_progress(progress_name)
        _finalize_slack_thread(
            script_path,
            status,
            elapsed,
            progress,
            thread_ts,
        )

    if status == "complete":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(run())
