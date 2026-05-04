"""Sync helpers for S3 download, upload, and existence checks."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

PARALLEL_THRESHOLD = 50 * 1024 * 1024  # 50 MB
PART_SIZE = 25 * 1024 * 1024  # 25 MB per part
MAX_WORKERS = 8


def download(bucket: str, key: str, local_path: Path) -> None:
    """Download an S3 object to a local file.

    Uses parallel byte-range downloads for files >= 50 MB.
    """
    client = boto3.client("s3")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    head = client.head_object(Bucket=bucket, Key=key)
    size = head["ContentLength"]

    if size < PARALLEL_THRESHOLD:
        logger.info("Downloading s3://%s/%s (%d bytes, single-part)", bucket, key, size)
        client.download_file(bucket, key, str(local_path))
        return

    logger.info("Downloading s3://%s/%s (%d bytes, parallel)", bucket, key, size)
    last = size - 1
    offsets = range(0, size, PART_SIZE)
    ranges = [(off, min(off + PART_SIZE - 1, last)) for off in offsets]

    def _fetch_range(byte_range: tuple[int, int]) -> tuple[int, bytes]:
        start, end = byte_range
        resp = client.get_object(
            Bucket=bucket,
            Key=key,
            Range=f"bytes={start}-{end}",
        )
        return start, resp["Body"].read()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        parts = sorted(pool.map(_fetch_range, ranges), key=lambda p: p[0])

    with local_path.open("wb") as f:
        for _offset, data in parts:
            f.write(data)


def upload(
    local_path: Path,
    bucket: str,
    key: str,
    *,
    expires_days: int | None = None,
) -> None:
    """Upload a local file to S3.

    Uses multipart upload with parallel parts for files >= 50 MB.
    If expires_days is set, the object will auto-expire after that many days.
    """
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    from boto3.s3.transfer import TransferConfig  # noqa: PLC0415

    size = local_path.stat().st_size
    client = boto3.client("s3")

    extra_args: dict[str, object] = {}
    if expires_days is not None:
        extra_args["Expires"] = datetime.now(timezone.utc) + timedelta(days=expires_days)

    if size >= PARALLEL_THRESHOLD:
        logger.info("Uploading %s (%d bytes, multipart) to s3://%s/%s", local_path, size, bucket, key)
        config = TransferConfig(
            multipart_threshold=PARALLEL_THRESHOLD,
            multipart_chunksize=PART_SIZE,
            max_concurrency=MAX_WORKERS,
        )
        client.upload_file(
            str(local_path), bucket, key, Config=config, ExtraArgs=extra_args or None,
        )
    else:
        logger.info("Uploading %s (%d bytes) to s3://%s/%s", local_path, size, bucket, key)
        client.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args or None)


def exists(bucket: str, key: str) -> bool:
    """Return True if the S3 object exists (HEAD check)."""
    client = boto3.client("s3")
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True
