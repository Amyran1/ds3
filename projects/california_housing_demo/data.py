from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.datasets import fetch_california_housing


def load_data() -> pl.DataFrame:
    """Return the full California Housing dataset as a polars DataFrame.

    Columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
             Latitude, Longitude, median_house_value
    """
    bunch = fetch_california_housing(as_frame=False)
    columns = list(bunch.feature_names) + ["median_house_value"]
    arr = np.column_stack([bunch.data, bunch.target])
    return pl.DataFrame(arr, schema=columns)


RAW_FEATURE_COLS: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
OUTCOME_VARIABLE: str = "median_house_value"
