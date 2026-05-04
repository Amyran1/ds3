"""EntityCache config for civic_shout_engagement actions (signatures).

One row per petition signature for Action Network group 228691. Scoped to
petitions belonging to group 228691 via the
``group_228691_indexed.petitions`` sub-select.

Versions
--------
v1 (current): covers last 1 month. Window is dynamic -- it is relative to
    the date the build script runs.

v2: anchored at ``BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z``. Covers the
    half-open window ``[BUILD_CUTOFF - 24 months, BUILD_CUTOFF)``. Schema
    is identical to v1; re-running the build script on any later date
    produces byte-identical output (modulo Redshift mirror state).

Schema (v1 and v2 identical):
    signature_id: i64         -- PK from signatures.id
    user_id:      i64         -- signer
    petition_id:  i64         -- which petition was signed
    created_at:   datetime[us, UTC]

Usage::

    from entities.civic_shout_engagement.actions_cache import cache

    df = cache.get(1)   # 1-month window (dynamic)
    df = cache.get(2)   # 24-month anchored at 2026-04-21 (reproducible)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/civic_shout_engagement/actions")
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities/civic_shout_engagement/actions"
)

cache = EntityCache(
    entity="civic_shout_engagement__actions",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="actions_v1", fmt="parquet"),
        2: VersionMeta(key="actions_v2", fmt="parquet"),
    },
)
