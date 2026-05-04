"""Train FAISS IVF index over dense embeddings and build dense_edges_v1.

Reads the local ``dense_embeddings/embeddings_v1.npy`` matrix (3.5M x 1536,
L2-normalized float32) plus its ``story_ids_v1.json`` sidecar, trains a
FAISS ``IndexIVFFlat`` with ``IndexFlatIP`` quantizer (cosine == inner
product on unit vectors), persists the trained index to
``DenseFaissIndexCache`` (local + S3), then runs batched kNN over the full
corpus, thresholds at ``TAU = 0.50``, canonicalizes edges so ``src < dst``,
and writes the result to ``DenseEdgesCache`` (local + S3).

If the trained index already exists locally (detected via
``DenseFaissIndexCache.local_path(1).exists()``), the train/add/persist
phases are skipped and the script resumes at the query phase. The
embedding matrix is memory-mapped (``mmap_mode="r"``) during the query
phase so concurrent processes (e.g. BM25 vector builders) are not
starved for RAM.

Usage:
    python -m entities.\
news_stories_graph.create_dense_faiss_index_v1
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import faiss
import numpy as np
import polars as pl

from entities.news_stories_graph import (
    dense_edges_cache as _de_module,
)
from entities.news_stories_graph import (
    dense_embeddings_cache as _dec_module,
)
from entities.news_stories_graph import (
    dense_faiss_index_cache as _dfi_module,
)

dense_edges_cache = _de_module.cache
dense_embeddings_cache = _dec_module.cache
dense_faiss_index_cache = _dfi_module.cache

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536
NLIST = 2048
NPROBE = 32
K_NEIGHBORS = 100
TAU = 0.50
QUERY_BATCH_SIZE = 50_000
TRAIN_SUBSAMPLE = 500_000

# Validation bands (logged warnings only -- do not gate).
MIN_EDGE_BAND = 1_000_000
MAX_EDGE_BAND = 100_000_000
MIN_AVG_DEGREE = 3
MAX_AVG_DEGREE = 30
MAX_NODE_DEGREE_WARN = 5000


def _load_vectors_mmap() -> tuple[np.ndarray, list[str]]:
    """Memory-map the embedding matrix and load row-ordered story_ids.

    Uses ``np.load(..., mmap_mode="r")`` so the full 20 GB matrix is not
    paged in. Queries copy slices into contiguous buffers as needed.
    """
    local_path = dense_embeddings_cache._local_path(1)  # noqa: SLF001
    if not local_path.exists():
        msg = (
            f"Expected embeddings at {local_path}; refusing to download a "
            "20 GB file when the query phase requires mmap access to an "
            "already-local file."
        )
        raise RuntimeError(msg)

    logger.info("mmap-loading dense embeddings from %s", local_path)
    arr = np.load(local_path, mmap_mode="r")
    story_ids = dense_embeddings_cache.get_story_ids(1)

    n_vecs = arr.shape[0]
    n_ids = len(story_ids)
    if n_vecs != n_ids:
        msg = f"Row count mismatch: vectors={n_vecs} vs story_ids={n_ids}"
        raise RuntimeError(msg)
    if arr.shape[1] != EMBEDDING_DIM:
        msg = f"Expected dim={EMBEDDING_DIM}, got {arr.shape[1]}"
        raise RuntimeError(msg)
    if arr.dtype != np.float32:
        msg = f"Expected float32, got {arr.dtype}"
        raise RuntimeError(msg)

    return arr, story_ids


def _load_vectors_full() -> tuple[np.ndarray, list[str]]:
    """Eagerly load the embedding matrix (only used during train/add)."""
    logger.info("Loading dense embeddings (full, non-mmap) via cache")
    arr = dense_embeddings_cache.get(1)
    story_ids = dense_embeddings_cache.get_story_ids(1)

    n_v = arr.shape[0]
    n_i = len(story_ids)
    if n_v != n_i:
        msg = f"Row count mismatch: vectors={n_v} vs story_ids={n_i}"
        raise RuntimeError(msg)
    return arr, story_ids


def _train_and_build_index(vectors: np.ndarray) -> faiss.Index:
    """Train an IVF-Flat (IP) index on a random subsample, then add all."""
    faiss_any = cast("Any", faiss)
    quantizer = faiss_any.IndexFlatIP(EMBEDDING_DIM)
    index = cast(
        "faiss.Index",
        faiss_any.IndexIVFFlat(quantizer, EMBEDDING_DIM, NLIST),
    )

    rng = np.random.default_rng(42)
    n_train = min(TRAIN_SUBSAMPLE, vectors.shape[0])
    train_idx = rng.choice(vectors.shape[0], size=n_train, replace=False)
    train_set = np.ascontiguousarray(vectors[train_idx])
    logger.info(
        "Training IVF on %d random vectors (nlist=%d, dim=%d)",
        n_train,
        NLIST,
        EMBEDDING_DIM,
    )

    t0 = time.monotonic()
    cast("Any", index).train(train_set)
    logger.info("train done in %.1fs", time.monotonic() - t0)

    t0 = time.monotonic()
    cast("Any", index).add(vectors)
    ntotal = int(cast("Any", index).ntotal)
    logger.info("add done in %.1fs (ntotal=%d)", time.monotonic() - t0, ntotal)

    return index


def _persist_index(index: faiss.Index) -> None:
    """Save the trained + populated index to local + S3 before querying."""
    t0 = time.monotonic()
    dense_faiss_index_cache.put(index, version=1)
    local = dense_faiss_index_cache.local_path(1)
    size_mb = local.stat().st_size / (1024 * 1024) if local.exists() else 0.0
    logger.info(
        "persisted index to %s (%.1f MB) in %.1fs",
        local,
        size_mb,
        time.monotonic() - t0,
    )


def _query_batches(
    index: faiss.Index,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched kNN search, return (src_idx, dst_idx, sims) as flat arrays.

    Applies self-match filter and tau threshold per batch. Canonicalizes
    by integer index (``src < dst``) to halve memory before string mapping.
    Works against a memory-mapped ``vectors`` array -- each batch is
    copied into a contiguous float32 buffer via ``np.ascontiguousarray``.
    """
    cast("Any", index).nprobe = NPROBE
    n = vectors.shape[0]
    n_batches = (n + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    sim_parts: list[np.ndarray] = []
    cum = 0

    t_start = time.monotonic()
    for b_i in range(n_batches):
        start = b_i * QUERY_BATCH_SIZE
        end = min(start + QUERY_BATCH_SIZE, n)
        # np.ascontiguousarray copies mmap slice into RAM as a contiguous
        # float32 buffer -- FAISS requires this.
        batch = np.ascontiguousarray(vectors[start:end])
        sims, neigh = cast("Any", index).search(batch, K_NEIGHBORS)

        row_idx = np.arange(start, end, dtype=np.int64)[:, None]
        src_full = np.broadcast_to(row_idx, neigh.shape)

        keep = (neigh >= 0) & (neigh > src_full) & (sims >= TAU)
        if keep.any():
            src_parts.append(src_full[keep].astype(np.int64, copy=False))
            dst_parts.append(neigh[keep].astype(np.int64, copy=False))
            sim_parts.append(sims[keep].astype(np.float32, copy=False))
            cum += int(keep.sum())

        if b_i % 5 == 0 or b_i == n_batches - 1:
            logger.info(
                "query batch %d/%d (rows %d..%d, cumulative edges=%d, %.1fs)",
                b_i + 1,
                n_batches,
                start,
                end,
                cum,
                time.monotonic() - t_start,
            )

    if not src_parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f
    return (
        np.concatenate(src_parts),
        np.concatenate(dst_parts),
        np.concatenate(sim_parts),
    )


def _edges_to_frame(
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    sims: np.ndarray,
    story_ids: list[str],
) -> pl.DataFrame:
    """Map integer indices to story_id strings, canonicalize, dedupe, sort."""
    ids_arr = np.asarray(story_ids, dtype=object)
    src_ids = ids_arr[src_idx]
    dst_ids = ids_arr[dst_idx]

    swap = src_ids > dst_ids
    if swap.any():
        src_ids = src_ids.copy()
        dst_ids = dst_ids.copy()
        tmp = src_ids[swap].copy()
        src_ids[swap] = dst_ids[swap]
        dst_ids[swap] = tmp

    df = pl.DataFrame(
        {
            "src": pl.Series(src_ids.tolist(), dtype=pl.Utf8),
            "dst": pl.Series(dst_ids.tolist(), dtype=pl.Utf8),
            "similarity": pl.Series(sims, dtype=pl.Float32),
        },
    )
    key = ["src", "dst"]
    return df.group_by(key).agg(pl.col("similarity").max()).sort(key)


def _query_and_threshold(
    index: faiss.Index,
    vectors: np.ndarray,
    story_ids: list[str],
) -> pl.DataFrame:
    """Run batched kNN, threshold, canonicalize to polars edge frame."""
    src_idx, dst_idx, sims = _query_batches(index, vectors)
    logger.info(
        "Raw thresholded edges (pre-dedup, pre-sort): %d",
        len(sims),
    )
    return _edges_to_frame(src_idx, dst_idx, sims, story_ids)


def _write_edges_cache(edges_df: pl.DataFrame) -> None:
    """Persist dense_edges_v1 via EntityCache (local + S3)."""
    t0 = time.monotonic()
    dense_edges_cache.put(1, edges_df)
    logger.info(
        "dense_edges_v1 written (%d rows) in %.1fs",
        len(edges_df),
        time.monotonic() - t0,
    )


def _log_validation(edges_df: pl.DataFrame, n_nodes: int) -> None:
    """Log edge-count + per-node degree stats and band checks."""
    n_edges = len(edges_df)
    avg_degree = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
    density = n_edges / (n_nodes * (n_nodes - 1) / 2.0) if n_nodes > 1 else 0.0

    logger.info("=" * 60)
    logger.info("Edge validation")
    logger.info("=" * 60)
    logger.info("n_nodes = %d", n_nodes)
    logger.info("n_edges = %d", n_edges)
    logger.info("avg_degree = %.3f", avg_degree)
    logger.info("density = %.2e", density)

    if n_edges == 0:
        logger.warning("No edges produced; skipping degree stats")
        return

    deg_df = (
        pl.concat(
            [
                edges_df.select(pl.col("src").alias("node")),
                edges_df.select(pl.col("dst").alias("node")),
            ],
        )
        .group_by("node")
        .agg(pl.len().alias("degree"))
    )
    degrees = deg_df["degree"].to_numpy()
    logger.info(
        "degree: min=%d p50=%.0f p90=%.0f p99=%.0f max=%d (n_touched=%d)",
        int(degrees.min()),
        float(np.percentile(degrees, 50)),
        float(np.percentile(degrees, 90)),
        float(np.percentile(degrees, 99)),
        int(degrees.max()),
        len(degrees),
    )

    logger.info(
        "band: edge_count in [%dM, %dM]? %s (expected ~10-30M)",
        MIN_EDGE_BAND // 1_000_000,
        MAX_EDGE_BAND // 1_000_000,
        MIN_EDGE_BAND <= n_edges <= MAX_EDGE_BAND,
    )
    logger.info(
        "band: avg_degree in [%d, %d]? %s",
        MIN_AVG_DEGREE,
        MAX_AVG_DEGREE,
        MIN_AVG_DEGREE <= avg_degree <= MAX_AVG_DEGREE,
    )

    if not (MIN_EDGE_BAND <= n_edges <= MAX_EDGE_BAND):
        logger.warning(
            "EDGE COUNT OUT OF BAND: %d not in [1M, 100M]",
            n_edges,
        )
    if degrees.max() > MAX_NODE_DEGREE_WARN:
        logger.warning(
            "MAX DEGREE HIGH: %d > %d",
            int(degrees.max()),
            MAX_NODE_DEGREE_WARN,
        )
    logger.info(
        "largest connected component fraction: skipped (expensive at scale)",
    )


def main() -> None:
    t_start = time.monotonic()

    index_local = dense_faiss_index_cache.local_path(1)
    resume = index_local.exists()

    if resume:
        logger.info(
            "Persisted FAISS index detected at %s -- skipping "
            "train/add/persist; loading index and using mmap embeddings",
            index_local,
        )
        t_load_0 = time.monotonic()
        vectors, story_ids = _load_vectors_mmap()
        t_load = time.monotonic() - t_load_0
        logger.info(
            "mmap-loaded dense embeddings: shape=%s (%.1fs)",
            vectors.shape,
            t_load,
        )

        t_idx_0 = time.monotonic()
        index = dense_faiss_index_cache.get(1)
        ntotal = int(cast("Any", index).ntotal)
        logger.info(
            "faiss.read_index done in %.1fs (ntotal=%d)",
            time.monotonic() - t_idx_0,
            ntotal,
        )
        n_v = vectors.shape[0]
        if ntotal != n_v:
            msg = f"Index ntotal ({ntotal}) != mmap matrix rows ({n_v})"
            raise RuntimeError(msg)

        t_build = 0.0
        t_persist = 0.0
    else:
        t_load_0 = time.monotonic()
        vectors, story_ids = _load_vectors_full()
        t_load = time.monotonic() - t_load_0
        logger.info(
            "Loaded dense embeddings: shape=%s (%.1fs)",
            vectors.shape,
            t_load,
        )

        t_build_0 = time.monotonic()
        index = _train_and_build_index(vectors)
        t_build = time.monotonic() - t_build_0

        t_persist_0 = time.monotonic()
        _persist_index(index)
        t_persist = time.monotonic() - t_persist_0

    t_query_0 = time.monotonic()
    edges_df = _query_and_threshold(index, vectors, story_ids)
    t_query = time.monotonic() - t_query_0
    logger.info("Total edges after dedupe/sort: %d", len(edges_df))

    t_write_0 = time.monotonic()
    _write_edges_cache(edges_df)
    t_write = time.monotonic() - t_write_0

    _log_validation(edges_df, n_nodes=len(story_ids))

    total = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info("Wall-clock breakdown")
    logger.info("  load:     %.1fs", t_load)
    logger.info("  train+add:%.1fs", t_build)
    logger.info("  persist:  %.1fs", t_persist)
    logger.info("  query:    %.1fs", t_query)
    logger.info("  write:    %.1fs", t_write)
    logger.info("  TOTAL:    %.1fs (%.1f min)", total, total / 60.0)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
