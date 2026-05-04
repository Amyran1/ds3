"""Date-range filter helpers for the StoryGraphAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from datetime import datetime


def filter_metadata_by_date(
    metadata: pl.DataFrame,
    date_range: tuple[datetime, datetime] | None,
) -> pl.DataFrame:
    """Return metadata rows whose created_at falls within ``date_range``.

    If ``date_range`` is None, returns ``metadata`` unchanged.
    """
    if date_range is None:
        return metadata
    start, end = date_range
    return metadata.filter(
        pl.col("created_at").is_between(start, end, closed="both"),
    )


def build_date_mask(
    metadata: pl.DataFrame,
    story_ids: list[str],
    date_range: tuple[datetime, datetime] | None,
) -> np.ndarray:
    """Return a bool mask over ``story_ids`` (row order) in-window.

    If ``date_range`` is None, returns an all-True mask.
    """
    n = len(story_ids)
    if date_range is None:
        return np.ones(n, dtype=bool)

    eligible = filter_metadata_by_date(metadata, date_range)
    eligible_ids = set(eligible["story_id"].to_list())
    return np.fromiter(
        (sid in eligible_ids for sid in story_ids),
        count=n,
        dtype=bool,
    )


def join_metadata(
    result: pl.DataFrame,
    metadata: pl.DataFrame,
    on: str = "story_id",
) -> pl.DataFrame:
    """Left-join ``result`` with ``metadata`` preserving ``result`` row order."""
    return result.join(metadata, on=on, how="left")
