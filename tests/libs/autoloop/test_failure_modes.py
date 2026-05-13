from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from libs.autoloop.failure_modes import (
    FailureMode,
    append_failure_mode,
    extract_failure_fingerprint,
    format_for_prompt,
    recent_failures,
)


class FakeAnthropic:
    def __init__(self, response: str = "test summary") -> None:
        self._response = response

    async def complete(
        self,
        messages: list[dict],
        options: object = None,
    ) -> str:
        return self._response


class FakeContainer:
    def __init__(self, response: str = "test summary") -> None:
        self.anthropic = FakeAnthropic(response)


# ---------------------------------------------------------------------------
# FailureMode model tests
# ---------------------------------------------------------------------------


def test_failure_mode_round_trip_jsonl(tmp_path: Path) -> None:
    fm = FailureMode(
        iteration_id=1,
        role="planner",
        ts="2025-01-01T00:00:00+00:00",
        exit_code=1,
        kill_reason="crashed",
        crashing_tool="Edit",
        last_file_attempted="/foo/bar.py",
        stderr_tail="Traceback ...",
        summary="Something broke",
    )

    line = fm.model_dump_json(by_alias=True)
    parsed = json.loads(line)

    assert parsed["schema"] == "failure_mode/v1"
    assert parsed["iteration_id"] == 1
    assert parsed["role"] == "planner"

    reconstructed = FailureMode.model_validate(parsed)
    assert reconstructed.schema_version == "failure_mode/v1"
    assert reconstructed.crashing_tool == "Edit"
    assert reconstructed.summary == "Something broke"


def test_append_failure_mode_writes_one_line(tmp_path: Path) -> None:
    fm = FailureMode(
        iteration_id=2,
        role="executor",
        ts="2025-01-01T00:00:00+00:00",
        exit_code=0,
        summary="ok",
    )

    append_failure_mode(tmp_path, fm)

    lines = (tmp_path / "failure_modes.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["iteration_id"] == 2


# ---------------------------------------------------------------------------
# recent_failures tests
# ---------------------------------------------------------------------------


def _write_n_failures(state_dir: Path, n: int) -> list[FailureMode]:
    fms = []
    for i in range(n):
        fm = FailureMode(
            iteration_id=i + 1,
            role="planner",
            ts="2025-01-01T00:00:00+00:00",
            exit_code=1,
            summary=f"failure {i + 1}",
        )
        append_failure_mode(state_dir, fm)
        fms.append(fm)
    return fms


def test_recent_failures_returns_last_n(tmp_path: Path) -> None:
    _write_n_failures(tmp_path, 15)

    results = recent_failures(tmp_path, n=10)

    assert len(results) == 10
    assert results[0].iteration_id == 6
    assert results[-1].iteration_id == 15


def test_recent_failures_returns_empty_when_file_missing(tmp_path: Path) -> None:
    results = recent_failures(tmp_path, n=10)
    assert results == []


def test_recent_failures_skips_bad_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "failure_modes.jsonl"
    fm = FailureMode(
        iteration_id=1,
        role="planner",
        ts="2025-01-01T00:00:00+00:00",
        exit_code=1,
        summary="good",
    )
    with path.open("w") as f:
        f.write("{invalid json\n")
        f.write(fm.model_dump_json(by_alias=True) + "\n")

    with caplog.at_level(logging.WARNING):
        results = recent_failures(tmp_path, n=10)

    assert len(results) == 1
    assert results[0].iteration_id == 1


# ---------------------------------------------------------------------------
# extract_failure_fingerprint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_failure_fingerprint_with_tool_use_events(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    events = [
        {"type": "tool_use", "name": "Edit", "args": {"file_path": "/foo/bar.py"}},
        {"type": "tool_use", "name": "Write", "args": {"file_path": "/foo/baz.py"}},
        {"is_error": True, "message": "Traceback (most recent call last): ..."},
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    container = FakeContainer("The session crashed on Edit. Avoid editing /foo/bar.py.")

    fm = await extract_failure_fingerprint(
        container=container,
        iteration_id=3,
        role="executor",
        session_log_path=log_path,
        exit_code=1,
        kill_reason=None,
        project="myproj",
    )

    assert fm.crashing_tool == "Write"
    assert fm.last_file_attempted == "/foo/baz.py"
    assert "Traceback" in fm.stderr_tail
    assert fm.summary == "The session crashed on Edit. Avoid editing /foo/bar.py."
    assert fm.iteration_id == 3
    assert fm.role == "executor"


@pytest.mark.asyncio
async def test_extract_failure_fingerprint_no_tool_use_events(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text(json.dumps({"type": "assistant", "content": "Hello"}) + "\n")

    container = FakeContainer("No tool crashed.")

    fm = await extract_failure_fingerprint(
        container=container,
        iteration_id=1,
        role="planner",
        session_log_path=log_path,
        exit_code=2,
        kill_reason="wall_cap",
        project="myproj",
    )

    assert fm.crashing_tool is None
    assert fm.last_file_attempted is None
    assert fm.stderr_tail == ""
    assert fm.summary == "No tool crashed."


# ---------------------------------------------------------------------------
# format_for_prompt tests
# ---------------------------------------------------------------------------


def _make_fm(iteration_id: int, summary: str, crashing_tool: str | None = "Edit") -> FailureMode:
    return FailureMode(
        iteration_id=iteration_id,
        role="planner",
        ts="2025-01-01T00:00:00+00:00",
        exit_code=1,
        crashing_tool=crashing_tool,
        summary=summary,
    )


def test_format_for_prompt_produces_markdown_bullets() -> None:
    failures = [_make_fm(1, "Something went wrong")]
    result = format_for_prompt(failures)

    assert "## Past session failures" in result
    assert "**iter 1 / planner**" in result
    assert "Something went wrong" in result


def test_format_for_prompt_truncates_when_exceeding_max_chars() -> None:
    failures = [_make_fm(i, "x" * 300) for i in range(1, 20)]
    result = format_for_prompt(failures, max_chars=500)

    assert len(result) <= 500


def test_format_for_prompt_returns_empty_for_empty_list() -> None:
    result = format_for_prompt([])
    assert result == ""
