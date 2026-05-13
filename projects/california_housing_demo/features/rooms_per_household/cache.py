"""rooms_per_household feature: AveRooms / AveOccup per block group.

Feature dictionary: see `dictionary.yaml` in this directory.
Plan: see `PLAN.md` in this directory.
"""

from __future__ import annotations

import polars as pl

FEATURE_COLS: list[str] = ["rooms_per_occupant"]


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append `rooms_per_occupant = AveRooms / AveOccup` to `df`.

    `AveRooms` in the sklearn California Housing dataset is already
    rooms-per-household; dividing by `AveOccup` (mean occupants per household)
    produces rooms-per-occupant. The feature family is named for brainstorm
    continuity; see PLAN.md §1 for the naming-vs-semantics note.
    """
    if "AveOccup" not in df.columns or "AveRooms" not in df.columns:
        raise ValueError(
            "rooms_per_household.add_features requires 'AveRooms' and "
            f"'AveOccup' columns; got {df.columns!r}"
        )
    return df.with_columns(
        (pl.col("AveRooms") / pl.col("AveOccup")).alias("rooms_per_occupant"),
    )
