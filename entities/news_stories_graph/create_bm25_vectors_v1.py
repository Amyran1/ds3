r"""Create v1 of the news_stories_graph bm25_vectors entity cache.

Fits ``pinecone_text.sparse.BM25Encoder`` (defaults k1=1.2, b=0.75) on the
full ~3.5M news story corpus (``name + "\\n\\n" + summary``), encodes the
corpus in 100K-document chunks, compacts the hashed vocabulary to a dense
``[0, K)`` range, L2-normalizes rows, and persists three artifacts via
``Bm25VectorsCache.put``:

    bm25_v1.npz        - L2-normalized (N x K) CSR matrix
    params_v1.json     - encoder hyperparameters + corpus stats
    story_ids_v1.json  - row-ordered story_ids aligned to the matrix

Row order matches ``create_dense_embeddings_v1.py::_load_corpus`` and
``create_sparse_tfidf_v1.py::_load_corpus`` exactly:

    1. drop null story_ids
    2. strip + fill-null on name / summary
    3. drop rows whose joined ``name\\n\\nsummary`` is empty
    4. sort by ``story_id`` ascending

Memory pattern: the COO row/col/data lists are accumulated incrementally as
each 100K-doc chunk is encoded so the intermediate ``list[dict]`` for the
full 3.5M corpus never coexists in memory.

Usage::

    python -m entities.\
news_stories_graph.create_bm25_vectors_v1
"""

from __future__ import annotations

import importlib.metadata
import logging
import time
from typing import cast

import nltk
import polars as pl
from pinecone_text.sparse import BM25Encoder
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

from entities.news_stories.cache import (
    cache as news_stories_cache,
)
from entities.news_stories_graph import (
    bm25_vectors_cache as _bvc_module,
)

bm25_vectors_cache = _bvc_module.cache

logger = logging.getLogger(__name__)

# ---- constants ------------------------------------------------------------

VERSION = 1
CHUNK_SIZE = 100_000
NLTK_DATA_VERSION = "punkt+punkt_tab+stopwords"


def _load_corpus() -> tuple[list[str], list[str]]:
    """Return (texts, story_ids) aligned to the dense embeddings row order."""
    logger.info("Loading news_stories v1 from cache")
    stories_df = news_stories_cache.get(1)
    logger.info("news_stories raw shape: %s", stories_df.shape)

    prepped = (
        stories_df.select(["story_id", "name", "summary"])
        .with_columns(
            pl.col("name").fill_null(""),
            pl.col("summary").fill_null(""),
        )
        .with_columns(
            (
                pl.col("name").str.strip_chars()
                + pl.lit("\n\n")
                + pl.col("summary").str.strip_chars()
            )
            .str.strip_chars()
            .alias("_joined"),
        )
        .filter(pl.col("_joined").str.len_chars() > 0)
        .filter(pl.col("story_id").is_not_null())
        .sort("story_id")
    )
    logger.info(
        "After null-text filter + story_id sort: %d rows (dropped %d)",
        len(prepped),
        len(stories_df) - len(prepped),
    )

    texts: list[str] = prepped["_joined"].to_list()
    story_ids: list[str] = [str(sid) for sid in prepped["story_id"].to_list()]

    keep = [i for i, t in enumerate(texts) if t]
    if len(keep) != len(texts):
        dropped = len(texts) - len(keep)
        logger.warning("Dropping %d residual empty-text rows", dropped)
        texts = [texts[i] for i in keep]
        story_ids = [story_ids[i] for i in keep]

    logger.info("Final corpus: %d documents", len(texts))
    return texts, story_ids


def _download_nltk_data() -> None:
    logger.info("Downloading NLTK data (punkt, punkt_tab, stopwords)")
    t0 = time.monotonic()
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    logger.info("NLTK data ready in %.1fs", time.monotonic() - t0)


def _fit_encoder(texts: list[str]) -> BM25Encoder:
    logger.info("Fitting BM25Encoder on %d documents", len(texts))
    t0 = time.monotonic()
    encoder = BM25Encoder()
    encoder.fit(texts)
    logger.info("BM25Encoder.fit done in %.1fs", time.monotonic() - t0)
    return encoder


def _encode_to_coo(
    encoder: BM25Encoder,
    texts: list[str],
) -> tuple[coo_matrix, int]:
    """Encode ``texts`` in CHUNK_SIZE blocks, building COO incrementally.

    Returns (coo_matrix, n_cols) where the matrix uses a compacted hashed
    vocabulary (columns ``[0, n_cols)``).
    """
    n = len(texts)
    all_rows: list[int] = []
    all_cols: list[int] = []
    all_data: list[float] = []
    idx_map: dict[int, int] = {}

    t_start = time.monotonic()
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        t_chunk = time.monotonic()
        chunk_dicts = cast(
            "list[dict[str, list[int] | list[float]]]",
            encoder.encode_documents(texts[start:end]),
        )
        for row_i_local, d in enumerate(chunk_dicts):
            row_i = start + row_i_local
            d_indices = cast("list[int]", d["indices"])
            d_values = cast("list[float]", d["values"])
            for idx, val in zip(d_indices, d_values, strict=True):
                compact = idx_map.get(idx)
                if compact is None:
                    compact = len(idx_map)
                    idx_map[idx] = compact
                all_rows.append(row_i)
                all_cols.append(compact)
                all_data.append(float(val))
        chunk_elapsed = time.monotonic() - t_chunk
        total_elapsed = time.monotonic() - t_start
        done = end
        rate = done / total_elapsed if total_elapsed > 0 else 0.0
        eta_min = (n - done) / rate / 60 if rate > 0 else 0.0
        logger.info(
            (
                "Encoded %d/%d (chunk %.1fs, cumulative nnz=%d, vocab=%d, "
                "rate=%.0f docs/s, ETA %.1fm)"
            ),
            done,
            n,
            chunk_elapsed,
            len(all_data),
            len(idx_map),
            rate,
            eta_min,
        )

    n_cols = len(idx_map)
    logger.info(
        "Building COO matrix: shape=(%d, %d), nnz=%d",
        n,
        n_cols,
        len(all_data),
    )
    raw = coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n, n_cols),
    )
    return raw, n_cols


def _l2_normalize(raw: coo_matrix) -> csr_matrix:
    logger.info("Converting COO to CSR and L2-normalizing rows")
    t0 = time.monotonic()
    csr = raw.tocsr()
    normalized = cast(
        "csr_matrix",
        sk_normalize(csr, norm="l2", axis=1, copy=False),
    )
    if normalized.format != "csr":
        normalized = normalized.tocsr()
    logger.info("L2-normalize done in %.1fs", time.monotonic() - t0)
    return normalized


def create() -> None:
    """Build news_stories_graph bm25_vectors v1 end-to-end."""
    t_total = time.monotonic()

    _download_nltk_data()

    texts, story_ids = _load_corpus()
    n = len(texts)

    t_fit_start = time.monotonic()
    encoder = _fit_encoder(texts)
    fit_elapsed = time.monotonic() - t_fit_start

    raw, n_cols = _encode_to_coo(encoder, texts)
    matrix = _l2_normalize(raw)

    shape = cast("tuple[int, int]", matrix.shape)
    if shape[0] != n:
        msg = f"Row count mismatch: matrix has {shape[0]} rows, texts {n}"
        raise RuntimeError(msg)
    if shape[1] != n_cols:
        msg = f"Column count mismatch: matrix has {shape[1]} cols, vocab {n_cols}"
        raise RuntimeError(msg)
    if len(story_ids) != shape[0]:
        n_ids = len(story_ids)
        n_rows = shape[0]
        msg = f"story_ids length {n_ids} does not match matrix row count {n_rows}"
        raise RuntimeError(msg)

    try:
        pt_version = importlib.metadata.version("pinecone-text")
    except importlib.metadata.PackageNotFoundError:
        pt_version = "unknown"

    params: dict[str, object] = {
        "encoder_class": "pinecone_text.sparse.BM25Encoder",
        "k1": float(encoder.k1),
        "b": float(encoder.b),
        "n_docs": int(n),
        "n_cols": int(n_cols),
        "nnz": int(matrix.nnz),
        "nltk_data_version": NLTK_DATA_VERSION,
        "pinecone_text_version": pt_version,
        "fit_seconds": round(fit_elapsed, 2),
    }

    logger.info("Persisting to Bm25VectorsCache (local + S3)")
    bm25_vectors_cache.put(matrix, params, story_ids, version=VERSION)

    total_elapsed = time.monotonic() - t_total
    density = matrix.nnz / (shape[0] * shape[1]) if shape[0] and shape[1] else 0.0
    logger.info("=" * 60)
    logger.info("news_stories_graph bm25_vectors v1 build complete")
    logger.info("  Shape:    %s", shape)
    logger.info("  NNZ:      %d", matrix.nnz)
    logger.info("  Density:  %.4f%%", density * 100)
    logger.info("  Vocab K:  %d", n_cols)
    logger.info("  k1/b:     %.3f / %.3f", encoder.k1, encoder.b)
    logger.info("  Stories:  %d", len(story_ids))
    logger.info("  Fit:      %.1fs", fit_elapsed)
    logger.info("  Elapsed:  %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    create()
