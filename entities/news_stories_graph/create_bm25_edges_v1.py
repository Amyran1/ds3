"""Create v1 of the news_stories_graph bm25_edges entity cache.

Computes undirected story/story edges from the full-corpus BM25 matrix at
the Q9 winning cosine threshold tau = 0.30 (see Addendum 5 of the
``eager-shimmying-quail`` construction plan). Edges are canonicalized as
``src < dst`` in story_id string order and persisted via
``Bm25EdgesCache.put`` as a parquet DataFrame with schema
``(src: Utf8, dst: Utf8, similarity: Float32)``.

Algorithm: chunked sparse cosine-similarity matmul. Because the BM25 matrix
is L2-normalized, rows dotted with rows equal cosine similarity. For each
row-block ``M[a:b]`` we compute ``M[a:b] @ M.T`` (scipy sparse matmul, BLAS-
backed), threshold at ``TAU``, then extract surviving (row, col) indices
with ``col > row`` (upper triangle) to avoid double-counting undirected
edges and self-loops.

Hard gates:
    - Wall-clock calibration gate at 50 chunks: projected total must be
      <= 120 min or the script exits.
    - Final edge-count band check logs a WARNING if outside [1M, 20M]
      (EDA 20K sample saw 11,635 edges at tau=0.30; 3.5M corpus expected
      1M-20M by linear extrapolation of quadratic edge density).

Usage::

    python -m entities.\
news_stories_graph.create_bm25_edges_v1
"""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

from entities.news_stories_graph import (
    bm25_edges_cache as _bec_module,
)
from entities.news_stories_graph import (
    bm25_vectors_cache as _bvc_module,
)

bm25_vectors_cache = _bvc_module.cache
bm25_edges_cache = _bec_module.cache

logger = logging.getLogger(__name__)

# ---- constants ------------------------------------------------------------

VERSION = 1
TAU = 0.30
CHUNK_SIZE = 5_000
CALIBRATION_CHUNKS = 50
WALL_CLOCK_CEILING_MIN = 120.0
EXPECTED_EDGES_MIN = 1_000_000
EXPECTED_EDGES_MAX = 20_000_000


def _load_matrix() -> tuple[csr_matrix, list[str]]:
    logger.info("Loading BM25 matrix v%d from cache", VERSION)
    matrix = bm25_vectors_cache.get(VERSION)
    if matrix.format != "csr":
        matrix = cast("csr_matrix", matrix.tocsr())
    story_ids = bm25_vectors_cache.get_story_ids(VERSION)
    shape = cast("tuple[int, int]", matrix.shape)
    if shape[0] != len(story_ids):
        n_rows = shape[0]
        n_ids = len(story_ids)
        msg = f"Row count mismatch: matrix has {n_rows} rows, story_ids has {n_ids}"
        raise RuntimeError(msg)
    logger.info(
        "Loaded matrix %s, nnz=%d, story_ids=%d",
        shape,
        matrix.nnz,
        len(story_ids),
    )
    return matrix, story_ids


def _compute_chunk_edges(
    matrix: csr_matrix,
    matrix_t: csr_matrix,
    start: int,
    end: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper-triangle edges from rows [start, end) against all rows.

    Returns (row_global, col_global, sim) int64/int64/float32 arrays.
    """
    block = matrix[start:end]
    sims_block = cast("csr_matrix", block @ matrix_t)
    # Threshold. scipy sparse masking via coo form is simplest.
    coo = sims_block.tocoo()
    mask = (coo.data >= tau) & ((coo.col) > (coo.row + start))
    # coo.row is local to the block; convert to global row index.
    rows_global = (coo.row[mask] + start).astype(np.int64)
    cols_global = coo.col[mask].astype(np.int64)
    sims = coo.data[mask].astype(np.float32)
    return rows_global, cols_global, sims


def _compute_edges(matrix: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = cast("tuple[int, int]", matrix.shape)
    n = shape[0]
    matrix_t = cast("csr_matrix", matrix.T.tocsr())
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(
        "Chunked matmul: n=%d, chunk_size=%d, n_chunks=%d, tau=%.2f",
        n,
        CHUNK_SIZE,
        n_chunks,
        TAU,
    )

    rows_acc: list[np.ndarray] = []
    cols_acc: list[np.ndarray] = []
    sims_acc: list[np.ndarray] = []

    t_start = time.monotonic()
    gate_checked = False
    for ci in range(n_chunks):
        start = ci * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        t_chunk = time.monotonic()
        r, c, s = _compute_chunk_edges(matrix, matrix_t, start, end, TAU)
        if r.size:
            rows_acc.append(r)
            cols_acc.append(c)
            sims_acc.append(s)
        chunk_elapsed = time.monotonic() - t_chunk

        total_elapsed = time.monotonic() - t_start
        chunks_done = ci + 1
        avg_chunk_s = total_elapsed / chunks_done
        remaining = n_chunks - chunks_done
        eta_min = (remaining * avg_chunk_s) / 60
        cumulative_edges = sum(a.size for a in rows_acc)
        if chunks_done % 10 == 0 or chunks_done == n_chunks:
            logger.info(
                "Chunk %d/%d done (%.1fs, nnz=%d, %.2fs/chunk, ETA %.1fm)",
                chunks_done,
                n_chunks,
                chunk_elapsed,
                cumulative_edges,
                avg_chunk_s,
                eta_min,
            )

        if not gate_checked and chunks_done >= CALIBRATION_CHUNKS:
            projected_total_min = (avg_chunk_s * n_chunks) / 60
            if projected_total_min > WALL_CLOCK_CEILING_MIN:
                logger.error(
                    "WALL-CLOCK GATE: projected %.0f min > %.0f min ceiling "
                    "(avg %.2fs/chunk over %d chunks). Aborting.",
                    projected_total_min,
                    WALL_CLOCK_CEILING_MIN,
                    avg_chunk_s,
                    chunks_done,
                )
                sys.exit(1)
            logger.info(
                "WALL-CLOCK GATE: PASS (projected %.0f min <= %.0f)",
                projected_total_min,
                WALL_CLOCK_CEILING_MIN,
            )
            gate_checked = True

    if not rows_acc:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.concatenate(rows_acc),
        np.concatenate(cols_acc),
        np.concatenate(sims_acc),
    )


def _canonicalize_string_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    sims: np.ndarray,
    story_ids: list[str],
) -> pl.DataFrame:
    """Map int indices -> story_id strings and canonicalize as src < dst."""
    logger.info("Mapping %d edges to story_id strings", len(rows))
    ids_arr = np.asarray(story_ids, dtype=object)
    src_ids = ids_arr[rows]
    dst_ids = ids_arr[cols]
    # Canonicalize: ensure src < dst as strings (upper-triangle from int
    # indices does not guarantee string order since story_ids are sorted
    # lexicographically in _load_corpus, so this is already canonical,
    # but enforce defensively).
    swap = src_ids > dst_ids
    if swap.any():
        tmp = src_ids[swap].copy()
        src_ids[swap] = dst_ids[swap]
        dst_ids[swap] = tmp
    return pl.DataFrame(
        {
            "src": pl.Series("src", src_ids.tolist(), dtype=pl.Utf8),
            "dst": pl.Series("dst", dst_ids.tolist(), dtype=pl.Utf8),
            "similarity": pl.Series("similarity", sims, dtype=pl.Float32),
        },
    )


def _log_degree_stats(df: pl.DataFrame, n_nodes: int) -> None:
    n_edges = len(df)
    if n_edges == 0:
        logger.warning("No edges produced")
        return
    avg_deg = (2.0 * n_edges) / n_nodes if n_nodes else 0.0
    # Build a per-node degree array using polars groupby for speed.
    degrees = (
        pl.concat(
            [
                df.select(pl.col("src").alias("node")),
                df.select(pl.col("dst").alias("node")),
            ],
        )
        .group_by("node")
        .len()
        .get_column("len")
        .to_numpy()
    )
    pct = np.percentile(degrees, [50, 90, 99, 99.9])
    logger.info(
        "Edge stats: n_edges=%d, n_nodes_with_edges=%d, avg_degree=%.3f, "
        "p50=%.0f, p90=%.0f, p99=%.0f, p99.9=%.0f, max=%d",
        n_edges,
        len(degrees),
        avg_deg,
        pct[0],
        pct[1],
        pct[2],
        pct[3],
        int(degrees.max()),
    )
    if n_edges < EXPECTED_EDGES_MIN or n_edges > EXPECTED_EDGES_MAX:
        logger.warning(
            "Edge count %d outside expected band [%d, %d]",
            n_edges,
            EXPECTED_EDGES_MIN,
            EXPECTED_EDGES_MAX,
        )
    else:
        logger.info(
            "Edge count %d within expected band [%d, %d]",
            n_edges,
            EXPECTED_EDGES_MIN,
            EXPECTED_EDGES_MAX,
        )


def create() -> None:
    """Build news_stories_graph bm25_edges v1 end-to-end."""
    t_total = time.monotonic()

    matrix, story_ids = _load_matrix()
    rows, cols, sims = _compute_edges(matrix)
    edge_df = _canonicalize_string_edges(rows, cols, sims, story_ids)

    _log_degree_stats(edge_df, len(story_ids))

    logger.info("Persisting to Bm25EdgesCache (local + S3)")
    bm25_edges_cache.put(VERSION, edge_df)

    total_elapsed = time.monotonic() - t_total
    logger.info("=" * 60)
    logger.info("news_stories_graph bm25_edges v1 build complete")
    logger.info("  Edges:   %d", len(edge_df))
    logger.info("  Tau:     %.2f", TAU)
    logger.info("  Elapsed: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    create()
