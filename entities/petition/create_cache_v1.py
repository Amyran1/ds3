"""Build petition entity cache v1.

Pulls 1,219 petitions signed in the 24-month window [2024-04-21, 2026-04-21)
from Redshift group_228691_indexed.petitions, builds a text column from
title + HTML-stripped description_info, fits BM25 sparse vectors, calls
OpenAI text-embedding-3-small for dense vectors, and writes the result
to the entity cache.

Usage::

    ENVIRONMENT=PRODUCTION python -m entities.petition.create_cache_v1
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from pinecone_text.sparse import BM25Encoder
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

from entities.civic_shout_engagement._psql import run_psql_to_csv
from entities.petition.cache import cache as petition_cache
from libs.clients.openai import OpenAIClient
from libs.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = 1
_DENSE_MODEL = "text-embedding-3-small"
_DENSE_DIMS = 1536
_EMBED_BATCH_SIZE = 128
_MAX_CONCURRENT = 25
_BM25_K1 = 1.2
_BM25_B = 0.75
_COST_KEY = "petition_v1"

_HARD_MIN_ROWS = 1000
_HARD_MAX_ROWS = 1500
_DENSE_VECTOR_DIM = 1536

_RAW_CSV_DIR = Path("data/raw/petition")
_RAW_CSV = _RAW_CSV_DIR / "petitions_v1.csv"

_SQL = """
WITH signed AS (
  SELECT DISTINCT petition_id FROM group_228691_indexed.signatures
  WHERE created_at >= TIMESTAMP '2024-04-21 00:00:00'
    AND created_at <  TIMESTAMP '2026-04-21 00:00:00'
    AND petition_id IS NOT NULL
)
SELECT p.id AS petition_id, p.created_at, p.title,
       COALESCE(p.description_info, '') AS description_info
FROM group_228691_indexed.petitions p
JOIN signed s ON p.id = s.petition_id
"""


# ---------------------------------------------------------------------------
# Exception subclasses
# ---------------------------------------------------------------------------


class _DenseShapeError(ValueError):
    def __init__(self, got: tuple[int, ...], expected: tuple[int, int]) -> None:
        super().__init__(f"Dense shape mismatch: got {got}, expected {expected}")


class _ValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Phase B — Text helpers
# ---------------------------------------------------------------------------


def _strip_html(s: str) -> str:
    """Collapse HTML to plain text — strip tags, unescape entities, dedupe whitespace."""
    if not s:
        return ""
    text = re.sub(r"<[^>]+>", " ", s)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_text(title: str, description_info: str) -> str:
    body = _strip_html(description_info)
    if body:
        return f"{title} {body}"
    return title


# ---------------------------------------------------------------------------
# Phase C — BM25 sparse vectors
# ---------------------------------------------------------------------------


def _fit_bm25(texts: list[str]) -> csr_matrix:
    encoder = BM25Encoder(k1=_BM25_K1, b=_BM25_B)
    encoder.fit(texts)
    encoded = cast(
        "list[dict[str, list[int] | list[float]]]",
        encoder.encode_documents(texts),
    )
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    idx_map: dict[int, int] = {}
    for row_i, d in enumerate(encoded):
        d_indices = cast("list[int]", d["indices"])
        d_values = cast("list[float]", d["values"])
        for idx, val in zip(d_indices, d_values, strict=True):
            compact = idx_map.get(idx)
            if compact is None:
                compact = len(idx_map)
                idx_map[idx] = compact
            rows.append(row_i)
            cols.append(compact)
            data.append(float(val))
    n_cols = len(idx_map)
    raw = coo_matrix((data, (rows, cols)), shape=(len(texts), n_cols))
    csr = raw.tocsr()
    normalized = cast("csr_matrix", sk_normalize(csr, norm="l2", axis=1, copy=False))
    if normalized.format != "csr":
        normalized = normalized.tocsr()
    logger.info("BM25: shape=%s, nnz=%d, vocab=%d", normalized.shape, normalized.nnz, n_cols)
    return normalized


def _csr_to_struct_column(matrix: csr_matrix, name: str) -> pl.Series:
    records: list[dict[str, list[int] | list[float]]] = []
    n_rows = cast("tuple[int, int]", matrix.shape)[0]
    for i in range(n_rows):
        row = matrix.getrow(i)
        order = np.argsort(row.indices, kind="stable")
        records.append(
            {
                "indices": row.indices[order].astype(np.uint32).tolist(),
                "values": row.data[order].astype(np.float32).tolist(),
            },
        )
    return pl.Series(
        name,
        records,
        dtype=pl.Struct(
            [
                pl.Field("indices", pl.List(pl.UInt32)),
                pl.Field("values", pl.List(pl.Float32)),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Phase D — Dense embed (async)
# ---------------------------------------------------------------------------


async def _embed_dense(texts: list[str]) -> np.ndarray:
    """Embed texts with OpenAI text-embedding-3-small; return L2-normalized array."""
    settings = Settings()
    client = OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        max_concurrent=_MAX_CONCURRENT,
    )
    t0 = time.monotonic()
    async with client:
        embeddings = await client.embed(
            texts,
            model=_DENSE_MODEL,
            batch_size=_EMBED_BATCH_SIZE,
            cost_key=_COST_KEY,
        )
    elapsed = time.monotonic() - t0
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.shape != (len(texts), _DENSE_DIMS):
        raise _DenseShapeError(arr.shape, (len(texts), _DENSE_DIMS))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    logger.info(
        "Dense embed: %d texts in %.1fs (%.0f texts/s)",
        len(texts),
        elapsed,
        len(texts) / elapsed if elapsed > 0 else 0.0,
    )
    return arr


# ---------------------------------------------------------------------------
# Phase E — Validation
# ---------------------------------------------------------------------------


def _validate(df: pl.DataFrame) -> None:
    n = len(df)
    if n < _HARD_MIN_ROWS or n > _HARD_MAX_ROWS:
        msg = f"Row count {n} outside [{_HARD_MIN_ROWS}, {_HARD_MAX_ROWS}]"
        raise _ValidationError(msg)

    if df["petition_id"].null_count() != 0:
        msg = f"petition_id has {df['petition_id'].null_count()} nulls"
        raise _ValidationError(msg)

    empty_text = (df["text"] == "").sum()
    if empty_text != 0:
        msg = f"{empty_text} rows have empty text"
        raise _ValidationError(msg)

    dense_lens = df["dense_vector"].list.len()
    if dense_lens.min() != _DENSE_VECTOR_DIM or dense_lens.max() != _DENSE_VECTOR_DIM:
        msg = f"dense_vector length not uniformly {_DENSE_VECTOR_DIM}"
        raise _ValidationError(msg)

    sparse_indices = df["sparse_vector"].struct.field("indices").list.len()
    if (sparse_indices == 0).any():
        msg = "Some rows have empty sparse_vector"
        raise _ValidationError(msg)

    logger.info(
        "Validation passed: %d rows, dense_dim=%d, sparse_vocab from first row=%d",
        n,
        _DENSE_VECTOR_DIM,
        int(sparse_indices[0]),
    )


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def main() -> None:
    t_total = time.monotonic()

    # Phase A: Redshift pull
    logger.info("Phase A: Redshift pull -> %s", _RAW_CSV)
    _RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        run_psql_to_csv(_SQL, _RAW_CSV)
    except Exception:
        logger.exception("Redshift pull FAILED. Aborting.")
        sys.exit(1)
    logger.info(
        "Phase A: done in %.1fs. CSV size: %d bytes", time.monotonic() - t0, _RAW_CSV.stat().st_size
    )

    raw_df = pl.read_csv(_RAW_CSV, try_parse_dates=True).select(
        pl.col("petition_id").cast(pl.Int64),
        pl.col("created_at").cast(pl.Datetime("us", "UTC")),
        pl.col("title").cast(pl.Utf8),
        pl.col("description_info").fill_null("").cast(pl.Utf8),
    )
    logger.info("Phase A: %d rows loaded", len(raw_df))

    # Phase B: Build text column
    t0 = time.monotonic()
    logger.info("Phase B: building text column")
    titles = raw_df["title"].to_list()
    descs = raw_df["description_info"].to_list()
    texts = [_build_text(t or "", d or "") for t, d in zip(titles, descs, strict=True)]
    empty_count = sum(1 for t in texts if not t)
    if empty_count > 0:
        logger.error("Phase B: %d rows have empty text. Aborting.", empty_count)
        sys.exit(1)
    df = raw_df.drop("description_info").with_columns(pl.Series("text", texts, dtype=pl.Utf8))
    logger.info("Phase B: done in %.1fs", time.monotonic() - t0)

    # Phase C: BM25 sparse vectors
    t0 = time.monotonic()
    logger.info("Phase C: fitting BM25 + building sparse_vector column")
    bm25_matrix = _fit_bm25(texts)
    sparse_col = _csr_to_struct_column(bm25_matrix, "sparse_vector")
    logger.info("Phase C: done in %.1fs", time.monotonic() - t0)

    # Phase D: Dense embed
    t0 = time.monotonic()
    logger.info("Phase D: dense embed via OpenAI (%d texts)", len(texts))
    dense_arr = asyncio.run(_embed_dense(texts))
    logger.info("Phase D: done in %.1fs", time.monotonic() - t0)

    # Assemble final DataFrame (drop title — text column subsumes it)
    dense_rows = dense_arr.tolist()
    df = df.drop("title").with_columns(
        pl.Series("dense_vector", dense_rows, dtype=pl.List(pl.Float32)),
        sparse_col,
    )
    logger.info("Assembled: %d rows, columns=%s", len(df), df.columns)

    # Phase E: Validate
    t0 = time.monotonic()
    logger.info("Phase E: validation")
    try:
        _validate(df)
    except _ValidationError:
        logger.exception("Validation FAILED. Aborting.")
        sys.exit(1)
    logger.info("Phase E: done in %.1fs", time.monotonic() - t0)

    # Phase F: Write cache
    t0 = time.monotonic()
    logger.info("Phase F: writing parquet to cache")
    petition_cache.put(VERSION, df)
    logger.info("Phase F: done in %.1fs", time.monotonic() - t0)

    logger.info("petition v1 complete. Total wall-clock: %.1fs", time.monotonic() - t_total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
