"""Datascience2 library — flat `libs.*` namespace.

Re-exports the most common public entry points so callers can write
`from libs import calibrate` instead of reaching into submodules.
For narrower imports, prefer the submodule directly.
"""

from __future__ import annotations

from libs.budget import Budget
from libs.calibrate import CalibrationResult, calibrate
from libs.checkpoint import CheckpointConfig, CheckpointedPipeline, auto_concurrency
from libs.container import Container
from libs.parallel import GroupApplyConfig, parallel_group_apply
from libs.validation import ValidationReport, validate_texts

__all__ = [
    "Budget",
    "CalibrationResult",
    "CheckpointConfig",
    "CheckpointedPipeline",
    "Container",
    "GroupApplyConfig",
    "ValidationReport",
    "auto_concurrency",
    "calibrate",
    "parallel_group_apply",
    "validate_texts",
]
