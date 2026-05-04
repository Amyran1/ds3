"""Async MongoDB client wrapper using Motor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from libs.costs import costs as _default_costs
from libs.costs.pricing import estimate_mongo_read
from libs.costs.tracker import CostRecord, CostTracker

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class FindOptions:
    """Bundled query options for MongoDB find operations."""

    filter: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None
    limit: int = 0
    sort: list[tuple[str, int]] | None = None
    cost_key: str | None = None
    # Motor cursor network batch size. None leaves Motor's default (101 docs
    # per roundtrip). Raise to 5_000-10_000 for large pulls — the Python
    # iteration interface is unchanged, just fewer RPCs.
    batch_size: int | None = None


class MongoClient:
    """Thin async wrapper around Motor for MongoDB operations."""

    def __init__(
        self,
        uri: str,
        database: str,
        *,
        min_pool: int = 1,
        max_pool: int = 5,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._uri = uri
        self._database_name = database
        self._min_pool = min_pool
        self._max_pool = max_pool
        self._costs = cost_tracker or _default_costs
        self._client: AsyncIOMotorClient[dict[str, Any]] = AsyncIOMotorClient(
            uri,
            minPoolSize=min_pool,
            maxPoolSize=max_pool,
            maxIdleTimeMS=60000,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def _db(self) -> AsyncIOMotorDatabase[dict[str, Any]]:
        """Return the database handle."""
        return self._client[self._database_name]

    async def find(
        self,
        collection: str,
        options: FindOptions | None = None,
    ) -> list[dict[str, Any]]:
        """Find documents matching a filter."""
        opts = options or FindOptions()
        cursor = self._db[collection].find(
            filter=opts.filter or {},
            projection=opts.projection,
        )
        if opts.sort:
            cursor = cursor.sort(opts.sort)
        if opts.limit:
            cursor = cursor.limit(opts.limit)
        if opts.batch_size:
            cursor = cursor.batch_size(opts.batch_size)
        docs = await cursor.to_list(length=opts.limit or None)

        if opts.cost_key:
            self._costs.record(
                CostRecord(
                    key=opts.cost_key,
                    amount=estimate_mongo_read(max(len(docs), 1)),
                    provider="mongodb",
                    operation="find",
                    units=len(docs),
                    unit_type="documents",
                )
            )

        return docs

    async def aggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
        cost_key: str | None = None,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run an aggregation pipeline."""
        cursor = self._db[collection].aggregate(pipeline)
        if batch_size:
            cursor = cursor.batch_size(batch_size)
        docs = await cursor.to_list(length=None)

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_mongo_read(max(len(docs), 1)),
                    provider="mongodb",
                    operation="aggregate",
                    units=len(docs),
                    unit_type="documents",
                )
            )

        return docs

    async def count(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,  # noqa: A002
        cost_key: str | None = None,
    ) -> int:
        """Count documents matching a filter."""
        result: int = await self._db[collection].count_documents(
            filter or {},
        )

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_mongo_read(),
                    provider="mongodb",
                    operation="count",
                    units=1,
                    unit_type="operations",
                )
            )

        return result

    async def close(self) -> None:
        """Close the underlying Motor client."""
        self._client.close()
