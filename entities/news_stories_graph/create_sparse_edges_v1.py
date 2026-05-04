"""Create v1 of the news_stories_graph sparse_edges entity cache.

Thresholded TF-IDF cosine similarity edges for the full ~3.5M news story
corpus. Loads the sparse TF-IDF matrix via ``SparseTfidfCache.get(1)`` and
computes cosine similarity in small chunks of 500 rows at a time using
``M_chunk @ M_T``. For each query chunk:

1. Sparse matmul ``(chunk_size, n_features) @ (n_features, n_stories)``
   produces a small CSR of similarities.
2. Convert to COO, apply the ``tau >= 0.25`` and upper-triangular
   ``src < dst`` filters numerically.
3. Append accepted triples to an in-memory buffer.
4. Every ``CHECKPOINT_EVERY`` chunks the buffer is flushed to a parquet
   fragment under ``.../sparse_edges/checkpoints/`` for resume safety.

Peak memory per chunk: ~500 rows x ~50K avg output nnz = ~25M nnz ~ 300 MB.
Stays well under system limits and avoids scipy's csr_matmat runaway
observed at larger chunk sizes.

After the matmul phase, fragments are reassembled, mapped to ``story_id``
strings, canonicalized, deduplicated, sorted, and written via
``SparseEdgesCache.put``.

A calibration gate runs after the first 20 chunks: if projected total
wall-clock > ``WALL_CLOCK_CEILING_MIN`` minutes the script ``sys.exit(1)``.

Usage::

    python -m entities.\
news_stories_graph.create_sparse_edges_v1
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from tqdm import tqdm

from entities.news_stories_graph import (
    sparse_edges_cache as _sec_module,
)
from entities.news_stories_graph import (
    sparse_tfidf_cache as _stc_module,
)

if TYPE_CHECKING:
    from scipy.sparse import coo_matrix, csr_matrix

sparse_edges_cache = _sec_module.cache
sparse_tfidf_cache = _stc_module.cache

logger = logging.getLogger(__name__)

# ---- constants ------------------------------------------------------------

VERSION = 1
TAU = 0.25
CHUNK_SIZE = 500
CHECKPOINT_EVERY = 200  # every ~100K rows processed
CALIBRATION_CHUNKS = 20
WALL_CLOCK_CEILING_MIN = 180.0
EDGE_BAND_MIN = 500_000
EDGE_BAND_MAX = 20_000_000

CHECKPOINT_DIR = Path(
    "data/entities/news_stories_graph/sparse_edges/checkpoints",
)
PROGRESS_PATH = CHECKPOINT_DIR / "progress.json"


@dataclass
class LoopState:
    """Mutable state threaded through the matmul loop."""

    buffer: list[np.ndarray]
    next_fragment: int
    chunks_done: int
    edges_seen: int
    t_run_start: float


def _fragment_path(idx: int) -> Path:
    return CHECKPOINT_DIR / f"edges_chunk_{idx:05d}.parquet"


def _load_progress() -> dict[str, int]:
    if PROGRESS_PATH.exists():
        data = json.loads(PROGRESS_PATH.read_text())
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
    return {}


def _save_progress(progress: dict[str, int]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress))


def _process_chunk(
    matrix: csr_matrix,
    matrix_t: csr_matrix,
    start: int,
    end: int,
) -> np.ndarray:
    """Return (K, 3) array [src_idx, dst_idx, similarity] for one chunk.

    Applies the tau threshold and the upper-triangular (src < dst) filter.
    """
    chunk = cast("csr_matrix", matrix[start:end])
    sims = cast("csr_matrix", chunk @ matrix_t)
    coo = cast("coo_matrix", sims.tocoo())

    data = np.asarray(coo.data, dtype=np.float32)
    rows = np.asarray(coo.row, dtype=np.int64) + start
    cols = np.asarray(coo.col, dtype=np.int64)

    mask = (data >= TAU) & (rows < cols)
    if not mask.any():
        return np.empty((0, 3), dtype=np.float64)

    out = np.empty((int(mask.sum()), 3), dtype=np.float64)
    out[:, 0] = rows[mask]
    out[:, 1] = cols[mask]
    out[:, 2] = data[mask]
    return out


def _flush_buffer(buffer: list[np.ndarray], fragment_idx: int) -> int:
    """Concat buffered triples and write a parquet fragment. Return row count."""
    if not buffer:
        return 0
    arr = np.concatenate(buffer, axis=0)
    df = pl.DataFrame(
        {
            "src_idx": arr[:, 0].astype(np.int64),
            "dst_idx": arr[:, 1].astype(np.int64),
            "similarity": arr[:, 2].astype(np.float32),
        }
    )
    path = _fragment_path(fragment_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info(
        "Flushed fragment %d: %d edges -> %s",
        fragment_idx,
        len(df),
        path.name,
    )
    return len(df)


def _check_calibration_gate(
    chunk_idx: int,
    n_chunks: int,
    state: LoopState,
) -> None:
    elapsed = time.monotonic() - state.t_run_start
    avg_chunk_sec = elapsed / state.chunks_done
    remaining_chunks = n_chunks - (chunk_idx + 1)
    projected_min = (avg_chunk_sec * remaining_chunks) / 60
    logger.info(
        "Calibration: %d chunks in %.1fs (%.2fs/chunk). "
        "Projected remaining: %.1f min (%d chunks left).",
        state.chunks_done,
        elapsed,
        avg_chunk_sec,
        projected_min,
        remaining_chunks,
    )
    if projected_min > WALL_CLOCK_CEILING_MIN:
        logger.error(
            "WALL-CLOCK GATE: projected %.0f min > %.0f min ceiling. "
            "Consider reducing chunk size or raising tau.",
            projected_min,
            WALL_CLOCK_CEILING_MIN,
        )
        sys.exit(1)
    logger.info("Calibration gate PASSED")


def _run_matmul_loop(
    matrix: csr_matrix,
    matrix_t: csr_matrix,
    n_total: int,
    n_chunks: int,
    last_done: int,
    next_fragment: int,
) -> LoopState:
    state = LoopState(
        buffer=[],
        next_fragment=next_fragment,
        chunks_done=0,
        edges_seen=0,
        t_run_start=time.monotonic(),
    )

    pbar = tqdm(
        range(last_done + 1, n_chunks),
        total=n_chunks - (last_done + 1),
        desc="sparse matmul",
    )
    for chunk_idx in pbar:
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_total)
        triples = _process_chunk(matrix, matrix_t, start, end)
        if triples.shape[0]:
            state.buffer.append(triples)
            state.edges_seen += triples.shape[0]
        state.chunks_done += 1

        if state.chunks_done == CALIBRATION_CHUNKS:
            _check_calibration_gate(chunk_idx, n_chunks, state)

        if (chunk_idx + 1) % CHECKPOINT_EVERY == 0:
            _flush_buffer(state.buffer, state.next_fragment)
            state.buffer = []
            state.next_fragment += 1
            _save_progress(
                {
                    "last_chunk_done": chunk_idx,
                    "next_fragment_idx": state.next_fragment,
                },
            )

    if state.buffer:
        _flush_buffer(state.buffer, state.next_fragment)
        state.next_fragment += 1
        state.buffer = []
    _save_progress(
        {
            "last_chunk_done": n_chunks - 1,
            "next_fragment_idx": state.next_fragment,
        },
    )
    return state


def _collect_fragments() -> pl.DataFrame:
    fragments = sorted(CHECKPOINT_DIR.glob("edges_chunk_*.parquet"))
    if not fragments:
        return pl.DataFrame(
            schema={
                "src_idx": pl.Int64,
                "dst_idx": pl.Int64,
                "similarity": pl.Float32,
            },
        )
    logger.info("Reading %d fragment files", len(fragments))
    frames = [pl.read_parquet(p) for p in fragments]
    combined = pl.concat(frames)
    logger.info("Combined fragments: %d rows", len(combined))
    return combined


def _finalize(df: pl.DataFrame, story_ids: list[str]) -> pl.DataFrame:
    """Map row indices -> story_id strings, canonicalize, dedupe, sort."""
    if df.is_empty():
        return pl.DataFrame(
            schema={"src": pl.Utf8, "dst": pl.Utf8, "similarity": pl.Float32},
        )

    ids_series = pl.Series("story_id", story_ids, dtype=pl.Utf8)
    id_lookup = pl.DataFrame(
        {
            "_idx": pl.arange(0, len(story_ids), eager=True).cast(pl.Int64),
            "story_id": ids_series,
        }
    )

    return (
        df.join(
            id_lookup.rename({"_idx": "src_idx", "story_id": "_src_id"}),
            on="src_idx",
            how="inner",
        )
        .join(
            id_lookup.rename({"_idx": "dst_idx", "story_id": "_dst_id"}),
            on="dst_idx",
            how="inner",
        )
        .with_columns(
            pl.when(pl.col("_src_id") < pl.col("_dst_id"))
            .then(pl.col("_src_id"))
            .otherwise(pl.col("_dst_id"))
            .alias("src"),
            pl.when(pl.col("_src_id") < pl.col("_dst_id"))
            .then(pl.col("_dst_id"))
            .otherwise(pl.col("_src_id"))
            .alias("dst"),
        )
        .select(["src", "dst", "similarity"])
        .filter(pl.col("src") != pl.col("dst"))
        .unique(subset=["src", "dst"], keep="first")
        .sort(["src", "dst"])
    )


def _log_degree_stats(df: pl.DataFrame) -> None:
    if df.is_empty():
        logger.warning("No edges to compute degree stats")
        return
    stacked = pl.concat(
        [
            df.select(pl.col("src").alias("node")),
            df.select(pl.col("dst").alias("node")),
        ]
    )
    degrees = stacked.group_by("node").agg(pl.len().alias("degree"))
    stats = degrees.select(
        pl.col("degree").mean().alias("mean"),
        pl.col("degree").quantile(0.5).alias("p50"),
        pl.col("degree").quantile(0.9).alias("p90"),
        pl.col("degree").quantile(0.99).alias("p99"),
        pl.col("degree").max().alias("max"),
        pl.len().alias("n_nodes"),
    ).row(0, named=True)
    logger.info(
        "Degree stats: nodes=%d avg=%.2f p50=%.0f p90=%.0f p99=%.0f max=%.0f",
        int(stats["n_nodes"]),
        float(stats["mean"]),
        float(stats["p50"]),
        float(stats["p90"]),
        float(stats["p99"]),
        float(stats["max"]),
    )


def _load_matrix_and_ids() -> tuple[csr_matrix, csr_matrix, list[str]]:
    logger.info("Loading sparse TF-IDF matrix")
    matrix = sparse_tfidf_cache.get(VERSION)
    if matrix.format != "csr":
        matrix = matrix.tocsr()
    shape = cast("tuple[int, int]", matrix.shape)
    logger.info("Matrix: shape=%s, nnz=%d", shape, matrix.nnz)
    logger.info("Loading story_ids")
    story_ids = sparse_tfidf_cache.get_story_ids(VERSION)
    if len(story_ids) != shape[0]:
        msg = f"story_ids length ({len(story_ids)}) != matrix rows ({shape[0]})"
        raise RuntimeError(msg)
    logger.info("Pre-transposing matrix for matmul")
    matrix_t = cast("csr_matrix", matrix.T.tocsr())
    return matrix, matrix_t, story_ids


def create() -> None:
    """Build news_stories_graph sparse_edges v1 end-to-end."""
    t_total = time.monotonic()
    matrix, matrix_t, story_ids = _load_matrix_and_ids()
    shape = cast("tuple[int, int]", matrix.shape)
    n_total = shape[0]

    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(
        "Chunk plan: %d chunks of up to %d rows (total rows=%d)",
        n_chunks,
        CHUNK_SIZE,
        n_total,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()
    last_done = int(progress.get("last_chunk_done", -1))
    next_fragment = int(progress.get("next_fragment_idx", 0))
    if last_done >= 0:
        logger.info(
            "Resume: last_chunk_done=%d, next_fragment_idx=%d",
            last_done,
            next_fragment,
        )

    state = _run_matmul_loop(
        matrix,
        matrix_t,
        n_total,
        n_chunks,
        last_done,
        next_fragment,
    )

    run_elapsed = time.monotonic() - state.t_run_start
    logger.info(
        "Matmul phase done: %d chunks, %d raw edges this run in %.1fs (%.1f min)",
        state.chunks_done,
        state.edges_seen,
        run_elapsed,
        run_elapsed / 60,
    )

    combined = _collect_fragments()
    finalized = _finalize(combined, story_ids)
    total_edges = len(finalized)
    density = total_edges / (n_total * (n_total - 1) / 2) if n_total > 1 else 0.0
    logger.info("Finalized edges: %d rows, graph density=%.2e", total_edges, density)
    _log_degree_stats(finalized)

    if total_edges < EDGE_BAND_MIN or total_edges > EDGE_BAND_MAX:
        logger.warning(
            "Edge count %d outside expected band [%d, %d]",
            total_edges,
            EDGE_BAND_MIN,
            EDGE_BAND_MAX,
        )

    logger.info("Persisting to SparseEdgesCache (local + S3)")
    sparse_edges_cache.put(VERSION, finalized)

    total_elapsed = time.monotonic() - t_total
    logger.info("=" * 60)
    logger.info("news_stories_graph sparse_edges v1 build complete")
    logger.info("  Nodes:   %d", n_total)
    logger.info("  Edges:   %d", total_edges)
    logger.info("  Density: %.2e", density)
    logger.info("  Elapsed: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    create()
