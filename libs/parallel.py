"""Generic parallel group-apply utility for numpy-heavy per-group operations.

Applies a function to each group's rows in parallel using joblib,
with automatic fallback to sequential processing for small group counts.

Usage:
    from libs.parallel import parallel_group_apply, GroupApplyConfig

    config = GroupApplyConfig(group_col="user_id", n_jobs=-1)
    result = parallel_group_apply(df, compute_user_features, config)
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

logger = logging.getLogger(__name__)

GroupResult = dict[str, np.ndarray]

SEQUENTIAL_VERBOSE_LABEL = "groups"


@dataclasses.dataclass(frozen=True)
class GroupApplyConfig:
    """Configuration for parallel_group_apply."""

    group_col: str = "user_id"
    n_jobs: int = -1
    min_groups_for_parallel: int = 10_000


def _find_group_boundaries(
    sorted_col: np.ndarray,
) -> list[tuple[int, int]]:
    """Find (start, end) index pairs for each contiguous group in a sorted array."""
    n = len(sorted_col)
    if n == 0:
        return []

    change_points = np.where(sorted_col[1:] != sorted_col[:-1])[0] + 1
    starts = np.concatenate([[0], change_points])
    ends = np.concatenate([change_points, [n]])

    return list(zip(starts, ends, strict=True))


def _init_output_arrays(
    first_result: GroupResult,
    total_rows: int,
) -> GroupResult:
    """Initialize full-length output arrays based on the first group's result dtypes."""
    output: GroupResult = {}
    for col_name, arr in first_result.items():
        if np.issubdtype(arr.dtype, np.floating):
            output[col_name] = np.full(total_rows, np.nan, dtype=arr.dtype)
        elif np.issubdtype(arr.dtype, np.integer):
            output[col_name] = np.zeros(total_rows, dtype=arr.dtype)
        else:
            output[col_name] = np.full(total_rows, np.nan, dtype=np.float64)
    return output


def _process_group(
    df: pl.DataFrame,
    start: int,
    end: int,
    fn: Callable[[pl.DataFrame], GroupResult],
) -> tuple[int, int, GroupResult]:
    """Process a single group slice and return boundaries with results."""
    group_df = df[start:end]
    result = fn(group_df)
    return start, end, result


def parallel_group_apply(
    df: pl.DataFrame,
    fn: Callable[[pl.DataFrame], GroupResult],
    config: GroupApplyConfig | None = None,
) -> GroupResult:
    """Apply fn to each group's rows in parallel.

    fn receives a single group's polars DataFrame (rows for one group),
    returns dict of column_name -> numpy array (same length as input rows).
    Results are assembled into full-length arrays aligned to df's row order.

    Uses joblib for parallelization. Falls back to sequential if fewer than
    min_groups_for_parallel groups.

    Args:
        df: Input DataFrame to group and process.
        fn: Function applied to each group's DataFrame. Must return a dict
            mapping column names to numpy arrays with length equal to the
            group's row count.
        config: Parallelization configuration. Defaults to GroupApplyConfig().

    Returns:
        Dict mapping column names to full-length numpy arrays aligned to
        the input DataFrame's row order after sorting by group_col.

    """
    if config is None:
        config = GroupApplyConfig()

    t0 = time.perf_counter()

    # Sort by group column to ensure contiguous groups
    df = df.sort(config.group_col)
    total_rows = len(df)

    if total_rows == 0:
        logger.info("Empty DataFrame, nothing to process")
        return {}

    # Find group boundaries from the sorted column
    sorted_col = df[config.group_col].to_numpy()
    boundaries = _find_group_boundaries(sorted_col)
    n_groups = len(boundaries)

    if n_groups == 0:
        return {}

    use_parallel = n_groups >= config.min_groups_for_parallel

    if use_parallel:
        logger.info(
            "Processing %d groups (%d parallel jobs)",
            n_groups,
            config.n_jobs,
        )
        results = Parallel(n_jobs=config.n_jobs, backend="loky", verbose=10)(
            delayed(_process_group)(df, start, end, fn)
            for start, end in boundaries
        )
    else:
        logger.info("Processing %d groups (sequential)", n_groups)
        results = [
            _process_group(df, start, end, fn)
            for start, end in tqdm(
                boundaries,
                desc=SEQUENTIAL_VERBOSE_LABEL,
                unit="group",
            )
        ]

    # Assemble results into full-length output arrays
    _, _, first_result = results[0]
    output = _init_output_arrays(first_result, total_rows)

    for start, end, group_result in results:
        for col_name, arr in group_result.items():
            output[col_name][start:end] = arr

    elapsed = time.perf_counter() - t0
    logger.info(
        "Finished %d groups in %.1fs (%.0f groups/s)",
        n_groups,
        elapsed,
        n_groups / elapsed if elapsed > 0 else float("inf"),
    )

    return output
