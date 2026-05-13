"""winsorize_heavy_tails feature family: 99th-percentile caps on heavy-tail raw cols.

Feature dictionary: see `dictionary.yaml` in this directory.
Plan: see `PLAN.md` in this directory.

`add_features` is pure with respect to its input: per-column 99th-percentile
thresholds are computed from the passed DataFrame (so a 10% smoke sample uses
the sample's quantile; a full run would use the full-frame quantile).
"""

from __future__ import annotations

import polars as pl

HEAVY_TAIL_RAW_COLS: list[str] = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
FEATURE_COLS: list[str] = [f"{c}_winz" for c in HEAVY_TAIL_RAW_COLS]
WINSORIZE_QUANTILE: float = 0.99


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append upper-tail winsorized variants of `HEAVY_TAIL_RAW_COLS`.

    For each column `c` in `HEAVY_TAIL_RAW_COLS`, computes the
    `WINSORIZE_QUANTILE`-quantile of `c` over `df` and appends
    `c_winz = min(c, p99)`. The lower tail is unchanged.
    """
    missing = [c for c in HEAVY_TAIL_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"winsorize_heavy_tails.add_features requires columns "
            f"{HEAVY_TAIL_RAW_COLS!r}; missing {missing!r}"
        )
    exprs: list[pl.Expr] = []
    for raw in HEAVY_TAIL_RAW_COLS:
        cap_value = df.select(pl.col(raw).quantile(WINSORIZE_QUANTILE)).item()
        exprs.append(pl.min_horizontal(pl.col(raw), pl.lit(cap_value)).alias(f"{raw}_winz"))
    return df.with_columns(exprs)
