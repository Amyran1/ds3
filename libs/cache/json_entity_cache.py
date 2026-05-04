"""Entity cache for JSON dict data with local disk and S3 backing store."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from libs.cache import s3
from libs.cache.entity_cache import DEFAULT_BUCKET, VersionMeta

logger = logging.getLogger(__name__)


class JsonEntityCache:
    """Single-entity cache for JSON dicts backed by local disk and S3."""

    def __init__(
        self,
        entity: str,
        s3_prefix: str,
        cache_dir: Path,
        versions: dict[int, VersionMeta],
        bucket: str = DEFAULT_BUCKET,
    ) -> None:
        self.entity = entity
        self.s3_prefix = s3_prefix
        self.cache_dir = cache_dir
        self.versions = versions
        self.bucket = bucket
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, version: int) -> VersionMeta:
        """Look up version metadata or raise ValueError."""
        if version not in self.versions:
            registered = sorted(self.versions)
            msg = (
                f"Version {version} not registered "
                f"for '{self.entity}'. "
                f"Registered: {registered}"
            )
            raise ValueError(msg)
        return self.versions[version]

    def _local_path(self, meta: VersionMeta) -> Path:
        """Return the local file path for a version."""
        return self.cache_dir / f"{meta.key}.json"

    def _s3_key(self, meta: VersionMeta) -> str:
        """Return the S3 object key for a version."""
        return f"{self.s3_prefix}/{meta.key}.json"

    def get(self, version: int) -> dict:
        """Return cached dict. Local hit -> return. Miss -> pull S3 -> return."""
        meta = self._resolve(version)
        local_path = self._local_path(meta)

        if local_path.exists():
            logger.debug(
                "Cache hit: %s v%d at %s",
                self.entity, version, local_path,
            )
            return json.loads(local_path.read_text(encoding="utf-8"))

        # Pull from S3
        try:
            s3.download(
                self.bucket, self._s3_key(meta), local_path,
            )
        except Exception as exc:
            msg = f"Version {version} not found locally or in S3"
            raise FileNotFoundError(msg) from exc

        logger.info(
            "Downloaded %s v%d from S3", self.entity, version,
        )
        return json.loads(local_path.read_text(encoding="utf-8"))

    def put(self, version: int, data: dict) -> None:
        """Write JSON locally and upload to S3."""
        meta = self._resolve(version)
        local_path = self._local_path(meta)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        local_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

        s3.upload(local_path, self.bucket, self._s3_key(meta))
        logger.info(
            "Saved %s v%d locally and to S3",
            self.entity, version,
        )
