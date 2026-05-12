"""Lazy DI container for runtime services.

Credentials load from ``.env`` via pydantic-settings.  For Mongo/Pinecone/S3,
enter the returned client when you need an active connection
(``async with c.mongo as mongo: ...``).  Cache loading still goes through
``cache.py`` interfaces — do not add cache properties here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from libs.clients.anthropic import AnthropicClient
    from libs.clients.cohere import CohereClient
    from libs.clients.mongo import MongoClient
    from libs.clients.openai import OpenAIClient
    from libs.clients.pinecone import PineconeClient
    from libs.clients.s3 import S3Client

from libs.budget import Budget
from libs.costs.tracker import CostTracker
from libs.settings import Settings


class Container:
    """Lazy async DI container that owns one Settings, one CostTracker, and one Budget.

    Clients are constructed on first property access and cached for the lifetime
    of the container.  Use as an async context manager so ``close()`` is called
    on exit.

    Example::

        async with Container() as c:
            results = await c.openai.embed(texts, cost_key="my-run")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or Settings()
        self._tracker = CostTracker(Path(self._settings.cost_ledger_path))
        self._budget = Budget(self._settings.budget_dollars, self._tracker)
        self._instances: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """The Settings instance used by this container."""
        return self._settings

    @property
    def costs(self) -> CostTracker:
        """The shared CostTracker for this container."""
        return self._tracker

    @property
    def budget(self) -> Budget:
        """The Budget backed by this container's CostTracker."""
        return self._budget

    # ------------------------------------------------------------------
    # Lazy client properties
    # ------------------------------------------------------------------

    @property
    def openai(self) -> OpenAIClient:
        """Lazy OpenAI client, shared for the container lifetime."""
        if "openai" not in self._instances:
            self._instances["openai"] = self._make_openai()
        return self._instances["openai"]  # type: ignore[return-value]

    @property
    def anthropic(self) -> AnthropicClient:
        """Lazy Anthropic client, shared for the container lifetime."""
        if "anthropic" not in self._instances:
            self._instances["anthropic"] = self._make_anthropic()
        return self._instances["anthropic"]  # type: ignore[return-value]

    @property
    def cohere(self) -> CohereClient:
        """Lazy Cohere client, shared for the container lifetime."""
        if "cohere" not in self._instances:
            self._instances["cohere"] = self._make_cohere()
        return self._instances["cohere"]  # type: ignore[return-value]

    @property
    def pinecone(self) -> PineconeClient:
        """Lazy Pinecone client, shared for the container lifetime.

        Enter via ``async with c.pinecone as pinecone:`` when an active
        connection is required.
        """
        if "pinecone" not in self._instances:
            self._instances["pinecone"] = self._make_pinecone()
        return self._instances["pinecone"]  # type: ignore[return-value]

    @property
    def mongo(self) -> MongoClient:
        """Lazy MongoDB client for the default database (``settings.mongo_database``).

        Enter via ``async with c.mongo as mongo:`` when an active
        connection is required.
        """
        return self.mongo_db(self._settings.mongo_database)

    def mongo_db(self, database: str) -> MongoClient:
        """Lazy MongoDB client for the given database, shared for container lifetime.

        Use when you need to target a database other than ``settings.mongo_database``
        (e.g. ``c.mongo_db("content_routing")``). The client is cached per database name.
        """
        key = f"mongo:{database}"
        if key not in self._instances:
            self._instances[key] = self._make_mongo_for(database)
        return self._instances[key]  # type: ignore[return-value]

    @property
    def s3(self) -> S3Client:
        """Lazy S3 client, shared for the container lifetime.

        Enter via ``async with c.s3 as s3:`` when an active
        connection is required.
        """
        if "s3" not in self._instances:
            self._instances["s3"] = self._make_s3()
        return self._instances["s3"]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _make_openai(self) -> OpenAIClient:
        from libs.clients.openai import OpenAIClient

        return OpenAIClient(
            api_key=self._settings.openai_api_key.get_secret_value(),
            cost_tracker=self._tracker,
        )

    def _make_anthropic(self) -> AnthropicClient:
        from libs.clients.anthropic import AnthropicClient

        return AnthropicClient(
            api_key=self._settings.anthropic_api_key.get_secret_value(),
            cost_tracker=self._tracker,
        )

    def _make_cohere(self) -> CohereClient:
        from libs.clients.cohere import CohereClient

        return CohereClient(
            api_key=self._settings.cohere_api_key.get_secret_value(),
            cost_tracker=self._tracker,
        )

    def _make_pinecone(self) -> PineconeClient:
        from libs.clients.pinecone import PineconeClient

        return PineconeClient(
            api_key=self._settings.pinecone_api_key.get_secret_value(),
            host=self._settings.pinecone_index_host,
            cost_tracker=self._tracker,
        )

    def _make_mongo_for(self, database: str) -> MongoClient:
        from libs.clients.mongo import MongoClient

        return MongoClient(
            uri=self._settings.mongo_uri.get_secret_value(),
            database=database,
            cost_tracker=self._tracker,
        )

    def _make_s3(self) -> S3Client:
        from libs.clients.s3 import S3Client

        return S3Client(
            bucket=self._settings.aws_s3_bucket,
            access_key_id=self._settings.aws_access_key_id.get_secret_value(),
            secret_access_key=self._settings.aws_secret_access_key.get_secret_value(),
            region=self._settings.aws_region,
            cost_tracker=self._tracker,
        )

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self  # type: ignore[return-value]

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close all instantiated clients and clear the instance cache.

        Idempotent: a second call is a no-op because the dict is already empty.
        """
        for client in list(self._instances.values()):
            if hasattr(client, "close"):
                await client.close()
        self._instances.clear()
