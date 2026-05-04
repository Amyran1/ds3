"""LocalCluster wrapper for parallel feature computation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

if TYPE_CHECKING:
    import pandas as pd

_client: Client | None = None


def get_or_create_client(n_workers: int | None = None) -> Client:
    """Return a lazily-initialized Dask distributed client.

    Creates a ``LocalCluster`` on first call. Subsequent calls return
    the same client.

    Args:
        n_workers: Number of workers. Defaults to ``cpu_count - 1``.

    Returns:
        A connected Dask distributed client.
    """
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client

    workers = n_workers or max(os.cpu_count() or 2, 2) - 1
    cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
    _client = Client(cluster)
    return _client


def parallel_apply(
    df: pd.DataFrame,
    func: dd.map_partitions.__class__,
    meta: pd.DataFrame | dict[str, str],
    n_partitions: int | None = None,
) -> pd.DataFrame:
    """Apply a function to a DataFrame in parallel via Dask.

    Converts to a Dask DataFrame, applies ``func`` partition-wise,
    and converts back to pandas.

    Args:
        df: Input pandas DataFrame.
        func: Function to apply to each partition.
        meta: Dask meta (column types) for the output.
        n_partitions: Number of partitions. Defaults to worker count.

    Returns:
        Transformed pandas DataFrame.
    """
    client = get_or_create_client()
    n_parts = n_partitions or len(client.scheduler_info()["workers"])
    ddf = dd.from_pandas(df, npartitions=n_parts)
    result = ddf.map_partitions(func, meta=meta)
    return result.compute()


def shutdown() -> None:
    """Shut down the Dask client and cluster if running."""
    global _client  # noqa: PLW0603
    if _client is not None:
        _client.close()
        _client = None
