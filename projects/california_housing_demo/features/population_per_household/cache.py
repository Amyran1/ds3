"""population_per_household feature: Population / AveOccup per block group.

Feature dictionary: see `dictionary.yaml` in this directory.
Plan: see `PLAN.md` in this directory.
"""

from __future__ import annotations

import polars as pl

FEATURE_COLS: list[str] = ["population_per_household"]


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append `population_per_household = Population / AveOccup` to `df`.

    `Population` is the total block-group population and `AveOccup` is the
    average household occupancy in the sklearn California Housing dataset,
    so the ratio approximates the household count per block group — a
    density proxy distinct from `AveOccup` itself (which is per-household
    occupancy, not per-block-group household count).
    """
    if "Population" not in df.columns or "AveOccup" not in df.columns:
        raise ValueError(
            "population_per_household.add_features requires 'Population' "
            f"and 'AveOccup' columns; got {df.columns!r}"
        )
    return df.with_columns(
        (pl.col("Population") / pl.col("AveOccup")).alias("population_per_household"),
    )
