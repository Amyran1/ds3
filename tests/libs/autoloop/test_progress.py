from __future__ import annotations

import time

import pytest

from libs.autoloop.progress import Heartbeat, event, fmt_dur


def test_event_prints_timestamped_line(capsys: pytest.CaptureFixture[str]) -> None:
    event("hello world")
    captured = capsys.readouterr()
    out = captured.out
    assert out.startswith("[")
    assert "] hello world\n" in out
    # timestamp format [HH:MM:SS]
    assert len(out.split("]")[0]) == 9  # "[HH:MM:SS"


def test_event_flushes_immediately(capsys: pytest.CaptureFixture[str]) -> None:
    event("flush test")
    captured = capsys.readouterr()
    assert "flush test" in captured.out


def test_fmt_dur_seconds() -> None:
    assert fmt_dur(45) == "45s"
    assert fmt_dur(0) == "0s"
    assert fmt_dur(59) == "59s"


def test_fmt_dur_minutes() -> None:
    assert fmt_dur(125) == "2m05s"
    assert fmt_dur(60) == "1m00s"


def test_fmt_dur_hours() -> None:
    assert fmt_dur(3725) == "1h02m"
    assert fmt_dur(3600) == "1h00m"


def test_fmt_dur_none() -> None:
    assert fmt_dur(None) == "—"


def test_heartbeat_ticks_at_least_twice(capsys: pytest.CaptureFixture[str]) -> None:
    hb = Heartbeat("test · phase", interval_s=0.05)
    hb.start()
    time.sleep(0.15)
    hb.stop()
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) >= 2


def test_heartbeat_lines_contain_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    hb = Heartbeat("iter 1/5 · planner", interval_s=0.05)
    hb.start()
    time.sleep(0.12)
    hb.stop()
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) >= 1
    for line in lines:
        assert "iter 1/5 · planner" in line
        assert "running" in line


def test_heartbeat_stop_before_first_interval_yields_zero_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hb = Heartbeat("early stop", interval_s=10)
    hb.start()
    hb.stop()
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) == 0


def test_heartbeat_update_changes_next_tick(capsys: pytest.CaptureFixture[str]) -> None:
    hb = Heartbeat("iter 2/5 · executor", interval_s=0.05)
    hb.start()
    time.sleep(0.03)
    hb.update("$1.23")
    time.sleep(0.1)
    hb.stop()
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    suffix_lines = [ln for ln in lines if "$1.23" in ln]
    assert len(suffix_lines) >= 1
