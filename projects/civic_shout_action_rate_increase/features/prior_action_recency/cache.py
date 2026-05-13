from __future__ import annotations

from pathlib import Path

from libs.cache.feature_cache import FeatureCache, FeatureVersionMeta

_V1_REQUIRED_COLUMNS = frozenset(
    [
        "user_id",
        "email_id",
        "date_sent",
        "actioned_last_1",
        "actioned_last_3",
        "actioned_last_5",
        "actioned_last_10",
        "sends_since_last_action",
        "action_in_last_5",
        "is_in_action_streak",
        "lifetime_actions_prior",
    ]
)

_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment"
    "/projects/civic_shout_action_rate_increase/features/prior_action_recency"
)

cache = FeatureCache(
    name="prior_action_recency",
    versions={
        1: FeatureVersionMeta(
            local_path=Path(
                "data/projects/civic_shout_action_rate_increase"
                "/features/prior_action_recency/v1.parquet"
            ),
            s3_key=f"{_S3_PREFIX}/v1.parquet",
            required_columns=_V1_REQUIRED_COLUMNS,
        ),
    },
)
