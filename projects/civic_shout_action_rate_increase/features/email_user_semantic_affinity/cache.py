from __future__ import annotations

from pathlib import Path

from libs.cache.feature_cache import FeatureCache, FeatureVersionMeta

_V1_REQUIRED_COLUMNS = frozenset(
    [
        "user_id",
        "email_id",
        "date_sent",
        "semantic_affinity_to_prior_actions",
        "n_prior_actions_for_affinity",
    ]
)

_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment"
    "/projects/civic_shout_action_rate_increase/features/email_user_semantic_affinity"
)

cache = FeatureCache(
    name="email_user_semantic_affinity",
    versions={
        1: FeatureVersionMeta(
            local_path=Path(
                "data/projects/civic_shout_action_rate_increase"
                "/features/email_user_semantic_affinity/v1.parquet"
            ),
            s3_key=f"{_S3_PREFIX}/v1.parquet",
            required_columns=_V1_REQUIRED_COLUMNS,
        ),
    },
)
