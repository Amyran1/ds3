"""Reusable local/S3 cache helper for project-local feature artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import polars as pl

from libs.cache import s3
from libs.s3_bucket import default_bucket

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class S3Client(Protocol):
    """Minimal S3 helper protocol used by FeatureCache."""

    def upload(self, local_path: Path, bucket: str, key: str) -> None: ...

    def download(self, bucket: str, key: str, local_path: Path) -> None: ...

    def exists(self, bucket: str, key: str) -> bool: ...


@dataclass(frozen=True)
class FeatureVersionMeta:
    """Storage metadata for one feature-cache version."""

    local_path: Path
    s3_key: str
    required_columns: frozenset[str]


class FeatureCache:
    """Local-first feature cache with S3 upload/download support.

    Transformation logic belongs in feature build recipes. This class owns only
    storage paths, version validation, required-column validation, and S3 sync.
    """

    def __init__(
        self,
        *,
        name: str,
        versions: dict[int, FeatureVersionMeta],
        bucket: str | None = None,
        s3_client: S3Client = s3,
    ) -> None:
        self.name = name
        self.versions = versions
        self.bucket = bucket if bucket is not None else default_bucket()
        self.s3_client = s3_client

    def path(self, version: int) -> Path:
        """Resolve the local parquet path for ``version``."""
        return self._resolve(version).local_path

    def remote_uri(self, version: int) -> str:
        """Return the S3 URI for ``version``."""
        meta = self._resolve(version)
        return f"s3://{self.bucket}/{meta.s3_key}"

    def exists(self, version: int) -> bool:
        """Return True if the versioned parquet exists locally."""
        try:
            path = self.path(version)
        except ValueError:
            return False
        return path.exists() and path.stat().st_size > 0

    def remote_exists(self, version: int) -> bool:
        """Return True if the versioned parquet exists in S3."""
        meta = self._resolve(version)
        return self.s3_client.exists(self.bucket, meta.s3_key)

    def get(self, version: int) -> pl.DataFrame:
        """Load the versioned feature parquet, downloading from S3 on local miss."""
        meta = self._resolve(version)
        path = meta.local_path
        if not path.exists():
            logger.info(
                "Local cache miss for %s v%d; downloading %s",
                self.name,
                version,
                self.remote_uri(version),
            )
            try:
                self.s3_client.download(self.bucket, meta.s3_key, path)
            except Exception as exc:
                msg = f"Feature cache miss for {self.name} v{version}. Expected local path: {path}; remote: {self.remote_uri(version)}"
                raise FileNotFoundError(msg) from exc

        df = pl.scan_parquet(path).collect()
        self._validate_columns(df, version)
        logger.info("Loaded %s v%d: %d rows", self.name, version, len(df))
        return df

    def put(self, version: int, artifact: pl.DataFrame) -> Path:
        """Persist a Polars DataFrame locally and to S3."""
        meta = self._resolve(version)
        self._validate_columns(artifact, version)
        path = meta.local_path
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_parquet(path)
        self.s3_client.upload(path, self.bucket, meta.s3_key)
        logger.info(
            "Persisted %s v%d: %d rows -> %s and %s",
            self.name,
            version,
            len(artifact),
            path,
            self.remote_uri(version),
        )
        return path

    def upload_existing(self, version: int) -> str:
        """Upload an existing local versioned parquet to S3 without rewriting it."""
        meta = self._resolve(version)
        path = meta.local_path
        if not path.exists():
            msg = f"Cannot upload missing local feature cache: {path}"
            raise FileNotFoundError(msg)
        self.s3_client.upload(path, self.bucket, meta.s3_key)
        logger.info(
            "Uploaded %s v%d -> %s",
            self.name,
            version,
            self.remote_uri(version),
        )
        return self.remote_uri(version)

    def _resolve(self, version: int) -> FeatureVersionMeta:
        if version not in self.versions:
            msg = f"{self.name} cache: unsupported version {version!r}. Supported: {sorted(self.versions)}"
            raise ValueError(msg)
        return self.versions[version]

    def _validate_columns(self, df: pl.DataFrame, version: int) -> None:
        meta = self._resolve(version)
        missing = meta.required_columns - set(df.columns)
        if missing:
            msg = f"{self.name} v{version} cache: missing required columns: {sorted(missing)}"
            raise RuntimeError(msg)
