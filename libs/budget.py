"""Budget enforcement wrapping CostTracker with a dollar ceiling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libs.costs.tracker import CostTracker


class BudgetExceededError(Exception):
    """Raised when an operation would exceed the project budget."""


class Budget:
    """Tracks spending against a dollar limit.

    Reads costs from a shared CostTracker instance and enforces
    a ceiling. Use ``check()`` before expensive operations to
    fail early rather than overspend.
    """

    def __init__(self, limit: float, tracker: CostTracker) -> None:
        self._limit = limit
        self._tracker = tracker

    @property
    def limit(self) -> float:
        """The total budget in dollars."""
        return self._limit

    def spent(self) -> float:
        """Total dollars spent across all cost keys."""
        return sum(s.total for s in self._tracker.get_all().values())

    def remaining(self) -> float:
        """Dollars remaining before hitting the budget limit."""
        return self._limit - self.spent()

    def check(self, estimated_cost: float) -> None:
        """Raise BudgetExceededError if estimated_cost would exceed budget.

        Args:
            estimated_cost: The estimated cost of the next operation in dollars.

        Raises:
            BudgetExceededError: If spent + estimated_cost > limit.
        """
        total_after = self.spent() + estimated_cost
        if total_after > self._limit:
            msg = (
                f"Budget exceeded: ${self.spent():.4f} spent + "
                f"${estimated_cost:.4f} estimated = ${total_after:.4f} "
                f"(limit: ${self._limit:.2f})"
            )
            raise BudgetExceededError(msg)

    def summary(self) -> dict[str, float]:
        """Return a dict with limit, spent, and remaining."""
        spent = self.spent()
        return {
            "limit": self._limit,
            "spent": spent,
            "remaining": self._limit - spent,
        }
