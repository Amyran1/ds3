"""bedrooms_per_room feature: AveBedrms / AveRooms per block group.

Feature dictionary: see `dictionary.yaml` in this directory.
Plan: see `PLAN.md` in this directory.
"""

from __future__ import annotations

import polars as pl

FEATURE_COLS: list[str] = ["bedrooms_per_room"]


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append `bedrooms_per_room = AveBedrms / AveRooms` to `df`.

    Both columns are per-household means in the sklearn California Housing
    dataset, so the ratio is interpretable as the share of rooms that are
    bedrooms — a denser proxy for housing quality than either raw column.
    A small upper tail above 1 is expected from public-dataset imputation
    artifacts and is tolerated raw (clipping is a downstream model decision).
    """
    if "AveBedrms" not in df.columns or "AveRooms" not in df.columns:
        raise ValueError(
            "bedrooms_per_room.add_features requires 'AveBedrms' and "
            f"'AveRooms' columns; got {df.columns!r}"
        )
    return df.with_columns(
        (pl.col("AveBedrms") / pl.col("AveRooms")).alias("bedrooms_per_room"),
    )
