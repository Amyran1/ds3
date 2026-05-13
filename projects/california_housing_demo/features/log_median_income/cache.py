"""log_median_income feature: log1p(MedInc) per block group.

Feature dictionary: see `dictionary.yaml` in this directory.
Plan: see `PLAN.md` in this directory.
"""

from __future__ import annotations

import polars as pl

FEATURE_COLS: list[str] = ["log_medinc"]


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append `log_medinc = log1p(MedInc)` to `df`.

    `log1p` is used rather than `log` to remain well-defined at zero in case
    future data drift introduces zero values; sklearn's California Housing
    `MedInc` is strictly positive in the shipped artifact.
    """
    if "MedInc" not in df.columns:
        raise ValueError(
            f"log_median_income.add_features requires 'MedInc' column; got {df.columns!r}"
        )
    return df.with_columns(
        pl.col("MedInc").log1p().alias("log_medinc"),
    )
