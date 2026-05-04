"""Consumer API for Scout email performance data.

Ported from `datascience/projects/scout/extractions/emails/cache.py`.
S3 is the source of truth; `cache.get(v)` downloads on first access and
caches locally under `data/entities/scout_emails/`.

Usage:
    from entities.scout_emails import cache
    df = cache.get(2)  # 25 cols incl. client_organization_id for org filtering
"""

from __future__ import annotations

from pathlib import Path

from entities.scout_emails import data_dictionary
from libs.cache.entity_cache import EntityCache, VersionMeta


class ScoutEmailsCache(EntityCache):
    """Scout project email performance data, cached from S3."""

    def __init__(self) -> None:
        super().__init__(
            entity="scout_emails",
            s3_prefix="data-science/extractors/emails",
            cache_dir=Path("data/entities/scout_emails"),
            versions={
                1: VersionMeta(key="emails_all", fmt="parquet"),
                2: VersionMeta(key="emails_all_v2", fmt="parquet"),
            },
        )

    def data_dictionary(self, version: int) -> dict[str, str]:
        self._resolve(version)  # validates version is registered
        return dict(data_dictionary.VERSIONS[version])


cache = ScoutEmailsCache()
