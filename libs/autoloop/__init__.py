"""Autonomous feature-engineering loop for ds3."""

from __future__ import annotations

__all__ = [
    "AutoloopConfig",
    "State",
    "BrainstormItem",
    "IterationRecord",
    "run_supervisor",
    "SupervisorArgs",
    "SupervisorExitCode",
]


def __getattr__(name: str):  # PEP 562 lazy attribute resolution
    if name in {"AutoloopConfig", "BrainstormItem", "IterationRecord", "State"}:
        from libs.autoloop import state

        return getattr(state, name)
    if name in {"SupervisorArgs", "SupervisorExitCode", "run_supervisor"}:
        from libs.autoloop import supervisor

        return getattr(supervisor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
