"""EntityCache config for civic_shout_engagement email_activities.

One row per email activity event (sent, open, click, verified_open, etc.) for
Action Network group 228691. The ``sent`` rows define the recipient universe
-- every ``(email_id, user_id)`` pair that received a send appears here even
if no subsequent engagement occurred.

Versions
--------
v1 (current): covers last 1 month + 14-day pre-window buffer (2x the 7-day
    attribution cap). Window is dynamic -- it is relative to the date the
    build script runs.

v2: anchored at ``BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z``. Covers the
    half-open window ``[BUILD_CUTOFF - 24 months, BUILD_CUTOFF)``. Schema
    is identical to v1; re-running the build script on any later date
    produces byte-identical output (modulo Redshift mirror state).

Schema (v1 and v2 identical):
    activity_id:  i64         -- PK from email_activities_16.id
    email_id:     i64         -- references the email that was sent
    user_id:      i64         -- renamed from recipient_id
    action_type:  Categorical -- {sent, open, click, verified_open, ...}
    created_at:   datetime[us, UTC]

Usage::

    from entities.civic_shout_engagement.email_activities_cache import cache

    df = cache.get(1)   # 1-month + 14d buffer (dynamic window)
    df = cache.get(2)   # 24-month anchored at 2026-04-21 (reproducible)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/civic_shout_engagement/email_activities")
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/civic_shout_engagement/email_activities"
)

cache = EntityCache(
    entity="civic_shout_engagement__email_activities",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="email_activities_v1", fmt="parquet"),
        2: VersionMeta(key="email_activities_v2", fmt="parquet"),
    },
)
