from __future__ import annotations

import sys
import threading
import time
from datetime import datetime


def event(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt_dur(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class Heartbeat:
    def __init__(self, prefix: str, interval_s: float = 30) -> None:
        self._prefix = prefix
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._suffix = ""
        self._suffix_lock = threading.Lock()
        self._start_time: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def update(self, suffix: str) -> None:
        with self._suffix_lock:
            self._suffix = suffix

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._interval_s):
            self._tick()

    def _tick(self) -> None:
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        with self._suffix_lock:
            suffix = self._suffix
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {self._prefix} · running {fmt_dur(elapsed)}"
        if suffix:
            line = f"{line} · {suffix}"
        print(line, flush=True)
