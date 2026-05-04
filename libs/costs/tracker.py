"""JSONL-backed cost tracker with file persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from pathlib import Path

from libs.costs.models import CostEntry, CostSummary

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(".costs/ledger.jsonl")


@dataclass
class CostRecord:
    """Parameters for recording a cost entry."""

    key: str
    amount: float
    provider: str
    operation: str
    model: str = ""
    units: int = 0
    unit_type: str = ""
    source: str = "estimated"
    investigation: str = ""


class CostTracker:
    """Append-only JSONL ledger for tracking IO operation costs.

    Costs are persisted to a local file, independent of any external
    storage or memory system. Each ``record()`` call appends one line.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH

    def record(self, rec: CostRecord) -> None:
        """Append a cost entry to the ledger."""
        entry = CostEntry(
            key=rec.key,
            amount=rec.amount,
            provider=rec.provider,
            operation=rec.operation,
            model=rec.model,
            units=rec.units,
            unit_type=rec.unit_type,
            source=rec.source,
            investigation=rec.investigation,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get(self, key: str) -> CostSummary:
        """Get aggregated costs for a single key."""
        entries = [e for e in self._read_all() if e.key == key]
        return CostSummary(
            key=key,
            total=sum(e.amount for e in entries),
            count=len(entries),
            entries=entries,
        )

    def get_all(self) -> dict[str, CostSummary]:
        """Get aggregated costs for all keys."""
        all_entries = self._read_all()
        sorted_entries = sorted(all_entries, key=attrgetter("key"))
        result: dict[str, CostSummary] = {}
        for key, group in groupby(sorted_entries, key=attrgetter("key")):
            entries = list(group)
            result[key] = CostSummary(
                key=key,
                total=sum(e.amount for e in entries),
                count=len(entries),
                entries=entries,
            )
        return result

    def get_by_investigation(self, investigation: str) -> float:
        """Sum all costs attributed to an investigation."""
        return sum(
            e.amount for e in self._read_all()
            if e.investigation == investigation
        )

    def clear(self, key: str) -> int:
        """Remove all entries for a key. Returns number removed."""
        all_entries = self._read_all()
        kept = [e for e in all_entries if e.key != key]
        removed = len(all_entries) - len(kept)
        self._write_all(kept)
        return removed

    def clear_all(self) -> None:
        """Remove all cost entries."""
        if self._path.exists():
            self._path.unlink()

    def _read_all(self) -> list[CostEntry]:
        if not self._path.exists():
            return []
        entries: list[CostEntry] = []
        with self._path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entries.append(
                        CostEntry.model_validate_json(stripped),
                    )
                except (json.JSONDecodeError, Exception):
                    logger.warning(
                        "Skipping malformed cost entry: %s",
                        stripped[:80],
                    )
        return entries

    def _write_all(self, entries: list[CostEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            f.writelines(entry.model_dump_json() + "\n" for entry in entries)
