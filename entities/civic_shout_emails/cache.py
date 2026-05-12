from __future__ import annotations

from pathlib import Path

import entities.civic_shout_emails.data_dictionary as _data_dictionary
from libs.cache.entity_cache import EntityCache, VersionMeta

_S3_PREFIX = "autonomous-data-scientist/civic_shout_news_environment/entities/civic_shout_emails"
_CACHE_DIR = Path("data/entities/civic_shout_emails")


class CivicShoutEmailsCache(EntityCache):
    """One row per Civic Shout email with dense and sparse embeddings."""

    def __init__(self) -> None:
        super().__init__(
            entity="civic_shout_emails",
            s3_prefix=_S3_PREFIX,
            cache_dir=_CACHE_DIR,
            versions={
                1: VersionMeta(key="civic_shout_emails_v1", fmt="parquet"),
            },
        )

    def data_dictionary(self, version: int) -> dict[str, str]:
        self._resolve(version)
        return dict(_data_dictionary.VERSIONS[version])


cache = CivicShoutEmailsCache()
