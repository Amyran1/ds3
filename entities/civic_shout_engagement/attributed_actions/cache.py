"""EntityCache config for attributed_actions.

Derived entity: one row per petition signature, with each signature attributed
to the email send that most recently preceded it. Attribution semantics differ
by version — see the per-version notes below.

Schema (v1):
    signature_id:       i64               -- PK; every row from actions appears once
    user_id:            i64
    petition_id:        i64
    sign_time:          datetime[us, UTC]  -- from actions.created_at
    email_id:           i64 | null         -- null = unattributed
    send_time:          datetime[us, UTC] | null
    lag_seconds:        i64 | null         -- sign_time - send_time
    window_seconds:     i64 | null         -- min(next_send - send, 7d)
    is_first_in_window: bool               -- True = attributed; False = cascade
    is_attributed:      bool               -- email_id not null and is_first_in_window

    Attribution rule (v1): 7-day absolute window cap. User's last send per
    user excluded from attribution (exclude_last_send=True). Signatures
    outside any window have email_id=null and is_attributed=False.

Schema (v2, adds one column):
    signature_id:       i64               -- PK; every row from actions v2 appears once
    user_id:            i64
    petition_id:        i64
    sign_time:          datetime[us, UTC]
    email_id:           i64 | null
    send_time:          datetime[us, UTC] | null
    lag_seconds:        i64 | null
    window_seconds:     i64 | null         -- full inter-send interval; NOT capped
    is_first_in_window: bool
    is_attributed:      bool
    is_bot_lag:         bool               -- True when lag_seconds < 10 (NEW in v2)

    Attribution rule (v2): no absolute window cap. Every send participates.
    A user's most recent send (no later send in the 24-month window) uses
    ``BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z`` as ``next_send_time``.
    ``window_seconds = next_send_time - send_time`` (uncapped). Strict
    ``lag_seconds < window_seconds`` boundary preserved. ``is_bot_lag``
    flags signatures with lag < 10 s but does NOT drop them — Stage-3
    investigations decide filter/reweight policy. Built from
    email_activities v2 + actions v2 (24-month window each).

Upstream dependencies (must be built first):
    v1: civic_shout_engagement__email_activities v1 + actions v1
    v2: civic_shout_engagement__email_activities v2 + actions v2

Usage::

    from entities.civic_shout_engagement.attributed_actions.cache import cache

    df = cache.get(1)  # v1 schema (10 columns, 7-day cap)
    df = cache.get(2)  # v2 schema (11 columns, uncapped + is_bot_lag)
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/civic_shout_engagement/attributed_actions")

_S3_ROOT = "autonomous-data-scientist/civic_shout_news_environment/entities"
_S3_PREFIX = f"{_S3_ROOT}/civic_shout_engagement/attributed_actions"

cache = EntityCache(
    entity="civic_shout_engagement__attributed_actions",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="attributed_actions_v1", fmt="parquet"),
        2: VersionMeta(key="attributed_actions_v2", fmt="parquet"),
    },
)
