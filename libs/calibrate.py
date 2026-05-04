"""Calibration helper for long-running ingest and compute scripts.

Generalizes the per-user-loop calibration pattern from
agent-tooling-overlay/modules/ds/rules/ds-planning.md to any script where a
single probe operation predicts total wall-clock. Callers run a small probe,
the helper measures throughput, projects total runtime, and hard-fails when
the projection is off — preventing the "kick off, discover mid-run that it's
3x slower than the plan" anti-pattern.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of a single calibration probe."""

    probe_items: int
    probe_seconds: float
    measured_rate: float  # items/sec
    total_items: int
    estimated_rate: float  # items/sec per the plan
    deviation_factor: float  # max(measured/estimated, estimated/measured)
    projected_minutes: float  # total_items / measured_rate / 60


async def calibrate(
    probe: Callable[[], Awaitable[int]],
    *,
    total_items: int,
    estimated_rate: float,
    max_deviation: float = 3.0,
    wall_clock_gate_min: float = 10.0,
    gate_strict: bool = True,
) -> CalibrationResult:
    """Run a probe once, measure throughput, and gate the full run.

    Args:
        probe: async callable returning the number of items it processed.
        total_items: full-run item count used for wall-clock projection.
        estimated_rate: the plan's expected items/sec.
        max_deviation: hard-fail if measured deviates by more than this
            factor from estimate (either direction). Default 3x matches
            the overlay's ds-planning.md calibration gate.
        wall_clock_gate_min: hard-fail if projected wall-clock exceeds
            this many minutes AND gate_strict is True. Default 10 min.
        gate_strict: if True (default), violations call sys.exit(1). If
            False, logs only and returns the result for caller inspection.

    Returns:
        CalibrationResult — caller should log it for auditability.
    """
    t0 = time.monotonic()
    n = await probe()
    elapsed = time.monotonic() - t0

    measured = n / elapsed if elapsed > 0 else 0.0
    if measured > 0 and estimated_rate > 0:
        deviation = max(measured / estimated_rate, estimated_rate / measured)
    else:
        deviation = float("inf")
    projected_min = total_items / measured / 60 if measured > 0 else float("inf")

    result = CalibrationResult(
        probe_items=n,
        probe_seconds=elapsed,
        measured_rate=measured,
        total_items=total_items,
        estimated_rate=estimated_rate,
        deviation_factor=deviation,
        projected_minutes=projected_min,
    )

    logger.info(
        "Calibration: probe %d items in %.1fs -> %.1f items/s "
        "(est. %.1f, dev %.1fx); projected full run: %.1f min",
        n,
        elapsed,
        measured,
        estimated_rate,
        deviation,
        projected_min,
    )

    if gate_strict:
        if deviation > max_deviation:
            logger.error(
                "CALIBRATION GATE: deviation %.1fx > %.1fx - aborting. "
                "Profile and update plan estimate before proceeding.",
                deviation,
                max_deviation,
            )
            sys.exit(1)
        if projected_min > wall_clock_gate_min:
            logger.error(
                "WALL-CLOCK GATE: %.1f min projected > %.1f min limit - "
                "aborting. Tune concurrency / batch_size / pre-filter "
                "before proceeding.",
                projected_min,
                wall_clock_gate_min,
            )
            sys.exit(1)

    return result
