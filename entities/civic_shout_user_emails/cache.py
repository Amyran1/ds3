"""EntityCache config for civic_shout_user_emails.

One row per (user_id, email_id) pair where a ``sent`` event exists in
email_activities v2.

v1 (legacy, broken baseline):
    Derived from email_activities v2 + attributed_actions v1 (7-day cap) +
    content_routing.subscription_state_events. attributed_actions v1 only has
    1 month of signatures, so the action rate is ~0% for the older 23 months.
    See V1_DICTIONARY in data_dictionary.py.

v2 (correct baseline):
    Derived from email_activities v2 + attributed_actions v2 (24-month, uncapped)
    + content_routing.subscription_state_events. Imputes ``opened`` from raw
    clicks (Apple-MPP-safe). Real action rate baseline ~9.76% (24-mo pooled).
    See V2_DICTIONARY in data_dictionary.py.

Shared schema (both versions):
    user_id:          i64
    email_id:         i64
    date_sent:        datetime[us, UTC]
    opened:           bool   -- imputation rule differs by version
    verified_opened:  bool
    clicked:          bool
    actioned:         bool   -- attribution-window differs by version
    unsubscribed:     bool

Usage::

    from entities.civic_shout_user_emails.cache import cache

    df = cache.get(2)  # use v2 for any modeling work
"""

from __future__ import annotations

from pathlib import Path

from entities.civic_shout_user_emails.data_dictionary import VERSIONS
from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/civic_shout_user_emails")
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities/civic_shout_user_emails"
)


class _UserEmailsCache(EntityCache):
    def data_dictionary(self, version: int) -> dict[str, str]:
        self._resolve(version)
        return VERSIONS[version]


cache = _UserEmailsCache(
    entity="civic_shout_user_emails",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="civic_shout_user_emails_v1", fmt="parquet"),
        2: VersionMeta(key="civic_shout_user_emails_v2", fmt="parquet"),
    },
)
