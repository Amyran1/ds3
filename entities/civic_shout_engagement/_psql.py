"""Small shell-out wrapper for Redshift psql pulls.

Deliberately kept module-local (not promoted to libs/) because this is the
only entity that pulls from Redshift today. Promote to libs/clients/redshift.py
when a second entity needs it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import IO

from libs.settings import Settings

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 10
_SIGTERM_GRACE_S = 2
_STDERR_TAIL_LINES = 200


class _MissingPgPasswordError(RuntimeError):
    def __init__(self) -> None:
        msg = (
            "Redshift password missing. Set REDSHIFT_PASSWORD in .env, "
            "PGPASSWORD in the env, or configure ~/.pgpass."
        )
        super().__init__(msg)


class _MissingRedshiftConfigError(RuntimeError):
    def __init__(self, field: str) -> None:
        super().__init__(
            f"Missing Redshift config field '{field}' in .env. "
            "Required: REDSHIFT_HOST, REDSHIFT_PORT, REDSHIFT_USER, REDSHIFT_DATABASE.",
        )


class _PsqlJobFailedError(Exception):
    """Raised when a psql subprocess exits non-zero."""

    def __init__(
        self,
        job_index: int,
        returncode: int,
        stderr_tail: str,
    ) -> None:
        self.job_index = job_index
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        super().__init__(
            f"psql job {job_index} exited with code {returncode}.\n"
            f"Last {_STDERR_TAIL_LINES} lines of stderr:\n{stderr_tail}"
        )


def _load_redshift_config() -> Settings:
    """Return Settings() with Redshift fields validated for non-emptiness.

    Password may come from REDSHIFT_PASSWORD in .env OR an existing PGPASSWORD
    in the shell environment OR a ~/.pgpass line. Everything else must be in
    .env.
    """
    settings = Settings()
    for field in ("redshift_host", "redshift_user", "redshift_database"):
        if not getattr(settings, field):
            raise _MissingRedshiftConfigError(field)
    has_password = (
        settings.redshift_password.get_secret_value()
        or os.environ.get("PGPASSWORD")
        or (Path.home() / ".pgpass").exists()
    )
    if not has_password:
        raise _MissingPgPasswordError
    return settings


def _build_cmd(sql: str, settings: Settings) -> list[str]:
    return [
        "psql",
        "-h",
        settings.redshift_host,
        "-p",
        str(settings.redshift_port),
        "-U",
        settings.redshift_user,
        "-d",
        settings.redshift_database,
        "--csv",
        "-c",
        sql,
    ]


def _subprocess_env(settings: Settings) -> dict[str, str]:
    """Return a subprocess env mapping with PGPASSWORD injected from Settings."""
    env = dict(os.environ)
    pw = settings.redshift_password.get_secret_value()
    if pw:
        env["PGPASSWORD"] = pw
    return env


def run_psql_to_csv(sql: str, out_path: Path) -> None:
    """Run a single psql query and write its CSV output to *out_path*.

    Args:
        sql: The SQL query to execute (trusted caller — no shell injection risk
             since the command list form is used, never shell=True).
        out_path: Destination file. Parent directories are created automatically.

    Raises:
        _MissingPgPasswordError: Password missing in .env/PGPASSWORD/~/.pgpass.
        _MissingRedshiftConfigError: Required Redshift field empty in .env.
        subprocess.CalledProcessError: psql exits non-zero.
    """
    settings = _load_redshift_config()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(sql, settings)
    with out_path.open("wb") as fh:
        subprocess.run(  # noqa: S603
            cmd,
            check=True,
            stdout=fh,
            env=_subprocess_env(settings),
        )


def _sigterm_survivors(
    procs: list[subprocess.Popen[bytes]],
    skip_index: int | None = None,
) -> None:
    """SIGTERM all still-running procs except *skip_index*, then SIGKILL stragglers."""
    for j, proc in enumerate(procs):
        if j == skip_index:
            continue
        if proc.poll() is None:
            proc.terminate()

    deadline = time.monotonic() + _SIGTERM_GRACE_S
    for j, proc in enumerate(procs):
        if j == skip_index:
            continue
        if proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=remaining)
            if proc.poll() is None:
                proc.kill()


def _close_all(file_handles: list[IO[bytes]]) -> None:
    """Best-effort close all file handles; silently ignore errors."""
    for fh in file_handles:
        with contextlib.suppress(Exception):
            fh.close()


def _raise_job_failed(
    proc: subprocess.Popen[bytes],
    job_index: int,
) -> None:
    """Read stderr from *proc* and raise _PsqlJobFailedError."""
    if proc.stderr is None:
        msg = f"psql job {job_index}: stderr pipe was not captured"
        raise RuntimeError(msg)
    if proc.returncode is None:
        msg = f"psql job {job_index}: returncode is None after poll() reported exit"
        raise RuntimeError(msg)
    raw = proc.stderr.read().decode(errors="replace")
    tail = raw.splitlines()[-_STDERR_TAIL_LINES:]
    raise _PsqlJobFailedError(
        job_index=job_index,
        returncode=proc.returncode,
        stderr_tail="\n".join(tail),
    )


def run_psql_parallel(
    jobs: list[tuple[str, Path]],
    timeout_s: int = 3600,
) -> list[int]:
    """Launch one psql subprocess per job in parallel and wait for all to finish.

    Each job is a ``(sql, out_path)`` tuple.  Stdout is streamed directly to
    the CSV file handle.  Progress is logged every ``_POLL_INTERVAL_S`` seconds
    showing ``(job_index, bytes_written, elapsed_s)``.

    On first failure: surviving subprocesses are SIGTERMed (with a
    ``_SIGTERM_GRACE_S``-second grace before SIGKILL), the last
    ``_STDERR_TAIL_LINES`` lines of stderr from the failed job are captured,
    and ``_PsqlJobFailedError`` is raised.

    On overall timeout: same cleanup, then ``TimeoutError`` is raised.

    Args:
        jobs: List of ``(sql, out_path)`` pairs.
        timeout_s: Hard wall-clock limit in seconds (default 3600).

    Returns:
        List of exit codes (all 0 on success).

    Raises:
        _MissingPgPasswordError: Password missing in .env/PGPASSWORD/~/.pgpass.
        _MissingRedshiftConfigError: Required Redshift field empty in .env.
        _PsqlJobFailedError: Any subprocess exits non-zero.
        TimeoutError: Overall elapsed time exceeds *timeout_s*.
    """
    settings = _load_redshift_config()
    env = _subprocess_env(settings)

    file_handles: list[IO[bytes]] = []
    procs: list[subprocess.Popen[bytes]] = []
    out_paths: list[Path] = []

    for _sql, out_path in jobs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fh = out_path.open("wb")
        file_handles.append(fh)
        out_paths.append(out_path)
        proc = subprocess.Popen(  # noqa: S603
            _build_cmd(_sql, settings),
            stdout=fh,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        procs.append(proc)

    start = time.monotonic()

    try:
        _poll_loop(procs, out_paths, start, timeout_s, file_handles)
    finally:
        _close_all(file_handles)

    result: list[int] = []
    for i, proc in enumerate(procs):
        rc = proc.returncode
        if rc is None:
            msg = (
                f"run_psql_parallel: job {i} returncode is None after "
                "normal completion — this should not happen."
            )
            raise RuntimeError(msg)
        result.append(rc)
    return result


def _poll_loop(
    procs: list[subprocess.Popen[bytes]],
    out_paths: list[Path],
    start: float,
    timeout_s: int,
    file_handles: list[IO[bytes]],
) -> None:
    """Inner polling loop for run_psql_parallel.  Raises on timeout or failure."""
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            _sigterm_survivors(procs)
            _close_all(file_handles)
            msg = (
                f"run_psql_parallel timed out after {elapsed:.0f}s "
                f"(limit {timeout_s}s). {len(procs)} jobs were launched."
            )
            raise TimeoutError(msg)

        all_done = _check_procs(procs, out_paths, elapsed, file_handles)
        if all_done:
            break

        time.sleep(_POLL_INTERVAL_S)


def _check_procs(
    procs: list[subprocess.Popen[bytes]],
    out_paths: list[Path],
    elapsed: float,
    file_handles: list[IO[bytes]],
) -> bool:
    """Check each proc once; return True iff all finished successfully."""
    all_done = True
    for i, (proc, out_path) in enumerate(zip(procs, out_paths, strict=True)):
        rc = proc.poll()
        if rc is None:
            all_done = False
            csv_bytes = out_path.stat().st_size if out_path.exists() else 0
            logger.info(
                '{"job": %d, "csv_bytes": %d, "elapsed_s": %.1f}',
                i,
                csv_bytes,
                elapsed,
            )
        elif rc != 0:
            _sigterm_survivors(procs, skip_index=i)
            _close_all(file_handles)
            _raise_job_failed(proc, job_index=i)
    return all_done
