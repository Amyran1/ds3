from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch


from libs.autoloop import session
from libs.costs.tracker import CostTracker

# ---------------------------------------------------------------------------
# FakePopen
# ---------------------------------------------------------------------------


class FakePopen:
    """Minimal subprocess.Popen substitute for testing _watch_session."""

    def __init__(
        self,
        stdout_lines: list[str],
        exit_code: int = 0,
        hang_seconds: float = 0.0,
    ) -> None:
        self._exit_code = exit_code
        self._hang_seconds = hang_seconds
        self._lines = stdout_lines
        self._terminated = False
        self._killed = False

    @property
    def stdout(self) -> io.StringIO:
        if self._hang_seconds > 0:
            time.sleep(self._hang_seconds)
        return io.StringIO("\n".join(self._lines) + ("\n" if self._lines else ""))

    @property
    def returncode(self) -> int | None:
        return self._exit_code

    def poll(self) -> int | None:
        return self._exit_code

    def wait(self, timeout: float | None = None) -> int:
        return self._exit_code

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._killed = True


def _make_tracker(tmp_path: Path) -> CostTracker:
    return CostTracker(tmp_path / ".costs" / "ledger.jsonl")


def _make_system_init_event(session_id: str = "sess-123") -> str:
    return json.dumps({"type": "system", "subtype": "init", "session_id": session_id})


def _make_assistant_event(
    input_tokens: int = 10,
    output_tokens: int = 20,
    tool_uses: list[dict[str, Any]] | None = None,
) -> str:
    content: list[dict[str, Any]] = []
    for tu in tool_uses or []:
        content.append({"type": "tool_use", **tu})
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "content": content,
            },
        }
    )


def _make_result_event(total_cost_usd: float = 0.05) -> str:
    return json.dumps({"type": "result", "total_cost_usd": total_cost_usd})


# ---------------------------------------------------------------------------
# _watch_session tests via FakePopen
# ---------------------------------------------------------------------------


def test_watch_session_extracts_session_id(tmp_path: Path) -> None:
    lines = [_make_system_init_event("my-session-id")]
    proc = FakePopen(lines)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "test.jsonl"

    result = session._watch_session(
        proc=proc,  # type: ignore[arg-type]
        role="planner",
        log_path=log_path,
        max_wall_seconds=60.0,
        max_dollars=10.0,
        cost_tracker=tracker,
        cost_key="test-key",
        model="claude-test",
        start=time.monotonic(),
    )

    assert result.session_id == "my-session-id"


def test_watch_session_accumulates_tokens(tmp_path: Path) -> None:
    lines = [
        _make_assistant_event(input_tokens=100, output_tokens=200),
        _make_assistant_event(input_tokens=50, output_tokens=75),
    ]
    proc = FakePopen(lines)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "test.jsonl"

    result = session._watch_session(
        proc=proc,  # type: ignore[arg-type]
        role="planner",
        log_path=log_path,
        max_wall_seconds=60.0,
        max_dollars=10.0,
        cost_tracker=tracker,
        cost_key="test-key",
        model="claude-test",
        start=time.monotonic(),
    )

    assert result.tokens_in == 150
    assert result.tokens_out == 275


def test_watch_session_extracts_wrote_files(tmp_path: Path) -> None:
    edit_event = _make_assistant_event(
        tool_uses=[
            {"name": "Edit", "input": {"file_path": "/proj/foo.py"}},
            {"name": "Write", "input": {"file_path": "/proj/bar.py"}},
            {"name": "Read", "input": {"file_path": "/proj/baz.py"}},
        ]
    )
    proc = FakePopen([edit_event])
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "test.jsonl"

    result = session._watch_session(
        proc=proc,  # type: ignore[arg-type]
        role="executor",
        log_path=log_path,
        max_wall_seconds=60.0,
        max_dollars=10.0,
        cost_tracker=tracker,
        cost_key="test-key",
        model="claude-test",
        start=time.monotonic(),
    )

    assert "/proj/foo.py" in result.wrote_files
    assert "/proj/bar.py" in result.wrote_files
    assert "/proj/baz.py" not in result.wrote_files


def test_watch_session_counts_tools(tmp_path: Path) -> None:
    lines = [
        _make_assistant_event(
            tool_uses=[
                {"name": "Read", "input": {}},
                {"name": "Read", "input": {}},
                {"name": "Bash", "input": {}},
            ]
        ),
    ]
    proc = FakePopen(lines)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "test.jsonl"

    result = session._watch_session(
        proc=proc,  # type: ignore[arg-type]
        role="planner",
        log_path=log_path,
        max_wall_seconds=60.0,
        max_dollars=10.0,
        cost_tracker=tracker,
        cost_key="test-key",
        model="claude-test",
        start=time.monotonic(),
    )

    assert result.tools_used_counts.get("Read") == 2
    assert result.tools_used_counts.get("Bash") == 1


def test_watch_session_tolerates_invalid_json_lines(tmp_path: Path) -> None:
    lines = [
        "this is not json",
        _make_system_init_event("ok-session"),
        "{ broken",
    ]
    proc = FakePopen(lines)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "test.jsonl"

    result = session._watch_session(
        proc=proc,  # type: ignore[arg-type]
        role="planner",
        log_path=log_path,
        max_wall_seconds=60.0,
        max_dollars=10.0,
        cost_tracker=tracker,
        cost_key="test-key",
        model="claude-test",
        start=time.monotonic(),
    )

    assert result.session_id == "ok-session"
    assert result.kill_reason == "normal"


def test_watch_session_result_cost_takes_precedence_over_tracker(tmp_path: Path) -> None:
    lines = [_make_result_event(total_cost_usd=1.23)]
    proc = FakePopen(lines)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "test.jsonl"

    result = session._watch_session(
        proc=proc,  # type: ignore[arg-type]
        role="planner",
        log_path=log_path,
        max_wall_seconds=60.0,
        max_dollars=10.0,
        cost_tracker=tracker,
        cost_key="test-key",
        model="claude-test",
        start=time.monotonic(),
    )

    assert abs(result.dollars - 1.23) < 1e-9


# ---------------------------------------------------------------------------
# run_claude_session integration via FakePopen
# ---------------------------------------------------------------------------


def test_run_claude_session_normal_exit(tmp_path: Path) -> None:
    lines = [
        _make_system_init_event("sess-normal"),
        _make_assistant_event(input_tokens=5, output_tokens=10),
    ]
    fake_proc = FakePopen(lines, exit_code=0)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "session.jsonl"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("subprocess.Popen", return_value=fake_proc):
        result = session.run_claude_session(
            role="planner",
            prompt="hello",
            cwd=tmp_path,
            settings_path=settings_path,
            allowed_tools=["Read"],
            denied_tools=[],
            max_wall_seconds=60.0,
            max_dollars=5.0,
            cost_tracker=tracker,
            cost_key="test",
            log_path=log_path,
            cli_path="claude",
            project="demo",
            iteration_id=1,
            model="claude-test",
        )

    assert result.exit_code == 0
    assert result.kill_reason == "normal"
    assert result.session_id == "sess-normal"


def test_run_claude_session_writes_log_file(tmp_path: Path) -> None:
    lines = [_make_system_init_event("sess-log")]
    fake_proc = FakePopen(lines, exit_code=0)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "logs" / "session.jsonl"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("subprocess.Popen", return_value=fake_proc):
        session.run_claude_session(
            role="executor",
            prompt="hello",
            cwd=tmp_path,
            settings_path=settings_path,
            allowed_tools=["Read"],
            denied_tools=[],
            max_wall_seconds=60.0,
            max_dollars=5.0,
            cost_tracker=tracker,
            cost_key="test",
            log_path=log_path,
            cli_path="claude",
            project="demo",
            iteration_id=1,
            model="claude-test",
        )

    assert log_path.exists()
    assert log_path.stat().st_size > 0


def test_run_claude_session_crashed_exit_code(tmp_path: Path) -> None:
    fake_proc = FakePopen([], exit_code=1)
    tracker = _make_tracker(tmp_path)
    log_path = tmp_path / "session.jsonl"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("subprocess.Popen", return_value=fake_proc):
        result = session.run_claude_session(
            role="planner",
            prompt="crash me",
            cwd=tmp_path,
            settings_path=settings_path,
            allowed_tools=[],
            denied_tools=[],
            max_wall_seconds=60.0,
            max_dollars=5.0,
            cost_tracker=tracker,
            cost_key="test",
            log_path=log_path,
            cli_path="claude",
            project="demo",
            iteration_id=1,
            model="claude-test",
        )

    assert result.exit_code == 1
    assert result.kill_reason == "crashed"
