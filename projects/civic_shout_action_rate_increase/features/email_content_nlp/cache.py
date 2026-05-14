from __future__ import annotations

from pathlib import Path

from libs.cache.feature_cache import FeatureCache, FeatureVersionMeta

_V1_REQUIRED_COLUMNS = frozenset(
    [
        "email_id",
        "body_char_len",
        "body_word_count",
        "body_sentence_count",
        "body_avg_word_len_chars",
        "body_exclamation_count",
        "body_question_count",
        "body_uppercase_word_ratio",
        "body_urgency_keyword_count",
        "body_cta_keyword_count",
        "first_line_char_len",
        "first_line_word_count",
    ]
)

_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment"
    "/projects/civic_shout_action_rate_increase/features/email_content_nlp"
)

cache = FeatureCache(
    name="email_content_nlp",
    versions={
        1: FeatureVersionMeta(
            local_path=Path(
                "data/projects/civic_shout_action_rate_increase"
                "/features/email_content_nlp/v1.parquet"
            ),
            s3_key=f"{_S3_PREFIX}/v1.parquet",
            required_columns=_V1_REQUIRED_COLUMNS,
        ),
    },
)
