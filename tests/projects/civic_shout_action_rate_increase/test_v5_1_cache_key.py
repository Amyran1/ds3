from __future__ import annotations

import polars as pl
import pytest

from projects.civic_shout_action_rate_increase.harness_cache import (
    _CONFOUND_CACHE_SCHEMA_VERSION,
    confound_data_fingerprint,
)


def _base_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "email_id": [10, 20, 30],
            "date_sent": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True),
            "actioned_24h": [0, 1, 0],
        }
    )


def test_confound_fingerprint_invariant_under_feature_changes() -> None:
    base = _base_df()
    with_feat_a = base.with_columns(pl.lit(0.5).alias("feature_a"))
    with_feat_b = base.with_columns(pl.lit(0.5).alias("feature_b"))
    assert confound_data_fingerprint(with_feat_a) == confound_data_fingerprint(with_feat_b)


def test_confound_fingerprint_invariant_without_extra_columns() -> None:
    base = _base_df()
    with_extra = base.with_columns(
        pl.lit(1.0).alias("feat_x"),
        pl.lit(2.0).alias("feat_y"),
        pl.lit("foo").alias("category"),
    )
    assert confound_data_fingerprint(base) == confound_data_fingerprint(with_extra)


def test_confound_fingerprint_changes_when_outcome_changes() -> None:
    base = _base_df()
    other = base.with_columns(pl.Series("actioned_24h", [1, 1, 1]))
    assert confound_data_fingerprint(base) != confound_data_fingerprint(other)


def test_confound_fingerprint_changes_when_row_count_changes() -> None:
    base = _base_df()
    fewer = base.slice(0, 2)
    assert confound_data_fingerprint(base) != confound_data_fingerprint(fewer)


def test_confound_fingerprint_changes_when_dates_shift() -> None:
    base = _base_df()
    shifted = base.with_columns(
        pl.date_range(pl.date(2025, 1, 1), pl.date(2025, 1, 3), eager=True).alias("date_sent")
    )
    assert confound_data_fingerprint(base) != confound_data_fingerprint(shifted)


def test_confound_fingerprint_schema_version_in_raw_string() -> None:
    assert _CONFOUND_CACHE_SCHEMA_VERSION == 1


def test_confound_fingerprint_returns_16_char_hex() -> None:
    fp = confound_data_fingerprint(_base_df())
    assert len(fp) == 16
    assert all(c in "0123456789abcdef" for c in fp)
