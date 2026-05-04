"""Cost tracking data models."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class CostEntry(BaseModel):
    """Single cost record from an IO operation."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    key: str
    amount: float
    provider: str
    operation: str
    model: str = ""
    units: int = 0
    unit_type: str = ""
    source: str = "estimated"
    investigation: str = ""


class CostSummary(BaseModel):
    """Aggregated cost view for a single key."""

    key: str
    total: float
    count: int
    entries: list[CostEntry] = Field(default_factory=list)

    @property
    def by_provider(self) -> dict[str, float]:
        """Break down total cost by provider."""
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry.provider] = totals.get(entry.provider, 0.0) + entry.amount
        return totals

    @property
    def by_operation(self) -> dict[str, float]:
        """Break down total cost by operation."""
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry.operation] = totals.get(entry.operation, 0.0) + entry.amount
        return totals

    @property
    def by_source(self) -> dict[str, float]:
        """Break down total cost by source (actual vs estimated)."""
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry.source] = totals.get(entry.source, 0.0) + entry.amount
        return totals
