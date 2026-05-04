"""Async S3 client wrapper with session management."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import aioboto3
from botocore.config import Config

from libs.costs import costs as _default_costs
from libs.costs.pricing import (
    estimate_s3_get,
    estimate_s3_list,
    estimate_s3_put,
)
from libs.costs.tracker import CostRecord, CostTracker
from libs.resilience import RateLimiter, with_retry

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_TYPE = "application/octet-stream"

_RETRY_EXCEPTIONS = (Exception,)


class S3Client:
    """Thin async wrapper around S3 via aioboto3."""

    def __init__(
        self,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "us-east-1",
        max_concurrent: int = 25,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._bucket = bucket
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._region = region
        self._rate_limiter = RateLimiter(max_concurrent)
        self._costs = cost_tracker or _default_costs
        self._boto_config = Config(
            region_name=region,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=50,
        )
        self._session: aioboto3.Session | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aioboto3.Session:
        """Get or create a reusable aioboto3 session."""
        if self._session is None:
            async with self._session_lock:
                if self._session is None:
                    self._session = aioboto3.Session()
        return self._session

    def _client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for creating an S3 client context manager."""
        return {
            "service_name": "s3",
            "aws_access_key_id": self._access_key_id,
            "aws_secret_access_key": self._secret_access_key,
            "region_name": self._region,
            "config": self._boto_config,
        }

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
        cost_key: str | None = None,
    ) -> str:
        """Upload bytes to S3 and return the S3 URI."""
        session = await self._get_session()
        async with (
            self._rate_limiter,
            session.client(**self._client_kwargs()) as s3,
        ):
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type or _DEFAULT_CONTENT_TYPE,
            )

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_s3_put(len(data)),
                    provider="s3",
                    operation="upload",
                    units=len(data),
                    unit_type="bytes",
                )
            )

        return f"s3://{self._bucket}/{key}"

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def download(
        self,
        key: str,
        cost_key: str | None = None,
    ) -> bytes:
        """Download an object from S3 as bytes."""
        session = await self._get_session()
        async with (
            self._rate_limiter,
            session.client(**self._client_kwargs()) as s3,
        ):
            response = await s3.get_object(Bucket=self._bucket, Key=key)
            data = await response["Body"].read()

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_s3_get(len(data)),
                    provider="s3",
                    operation="download",
                    units=len(data),
                    unit_type="bytes",
                )
            )

        return data

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def get_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
    ) -> str:
        """Generate a presigned URL for an S3 object."""
        session = await self._get_session()
        async with (
            self._rate_limiter,
            session.client(**self._client_kwargs()) as s3,
        ):
            url: str = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expiration,
            )
        return url

    @with_retry(retry_on=_RETRY_EXCEPTIONS)
    async def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
        cost_key: str | None = None,
    ) -> list[str]:
        """List object keys matching a prefix."""
        session = await self._get_session()
        keys: list[str] = []
        async with (
            self._rate_limiter,
            session.client(**self._client_kwargs()) as s3,
        ):
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            ):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))

        if cost_key:
            self._costs.record(
                CostRecord(
                    key=cost_key,
                    amount=estimate_s3_list(),
                    provider="s3",
                    operation="list_objects",
                    units=len(keys),
                    unit_type="keys",
                )
            )

        return keys

    async def close(self) -> None:
        """Release the underlying session."""
        if self._session is not None:
            self._session = None
