"""Cost estimation and tracking for IO operations.

Usage::

    from libs.costs import costs

    # IO operations record costs automatically via cost_key parameter
    embeddings = await client.embed(texts, cost_key="my-extraction")

    # Read costs back
    summary = costs.get("my-extraction")
    print(summary.total)

    # See all tracked keys
    all_costs = costs.get_all()
"""

from __future__ import annotations

from libs.costs.models import CostEntry, CostSummary
from libs.costs.tracker import CostRecord, CostTracker

__all__ = [
    "CostEntry",
    "CostRecord",
    "CostSummary",
    "CostTracker",
    "costs",
]

costs = CostTracker()
