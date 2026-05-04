r"""Build civic_shout_engagement__aggregate_emails v2.

Extends v1 with vector columns.

Reads v1 via the entity cache, HTML-strips each email body, concatenates
subject + plain-text body, fits TFIDF + BM25 vectorizers on the email corpus,
calls OpenAI text-embedding-3-small for dense embeddings, and adds four new
columns to the v1 schema: text_content, dense_vector, tfidf_vector,
sparse_vector.  Writes v2 to the entity cache and uploads fit artifacts
(TFIDF vocab + IDF, BM25 params, BM25 encoder, idx_map) as JSON sidecars.

Artifacts produced:
- tfidf_vocab.json   — TFIDF vocabulary + IDF weights + config
- bm25_params.json   — BM25 hyperparameters + corpus stats (DS-internal shape)
- bm25_encoder.json  — BM25Encoder native JSON (loadable via BM25Encoder.load())
- idx_map.json       — {str(raw_pinecone_index): compact_col_index} mapping;
                       consumed by the backend content_routing vectorize cron
                       (backend repo: .../cron/content_routing/vectorize_emails.py)

Cost: ~$0.18 at 4,575 emails x ~2K tokens avg.

Usage::

    OPENAI_API_KEY=<secret> python -m \
        entities.civic_shout_engagement.create_aggregate_emails_v2

    # Plumbing test on 100 rows:
    OPENAI_API_KEY=<secret> python -m \
        entities.civic_shout_engagement.create_aggregate_emails_v2 --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import time
from pathlib import Path
from typing import cast

import nltk
import numpy as np
import polars as pl
from bs4 import BeautifulSoup
from pinecone_text.sparse import BM25Encoder
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from entities.civic_shout_engagement.aggregate_emails_cache import (
    cache as aggregate_emails_cache,
)
from libs.cache import s3
from libs.cache.entity_cache import DEFAULT_BUCKET
from libs.clients.openai import OpenAIClient
from libs.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = 2
_DENSE_MODEL = "text-embedding-3-small"
_DENSE_DIMS = 1536
_EMBED_BATCH_SIZE = 500
_MAX_CONCURRENT = 25
_TEXT_MAX_CHARS = 32_000
_TFIDF_MAX_FEATURES = 20_000
_TFIDF_NGRAM = (1, 2)
_TFIDF_MIN_DF = 3
_TFIDF_MAX_DF = 0.5
_BM25_K1 = 1.2
_BM25_B = 0.75
_COST_KEY = "civic_shout_engagement__aggregate_emails_v2"
_NLTK_DATA_VERSION = "punkt+punkt_tab+stopwords"

_V2_ARTIFACTS_BASE = Path(
    "data/entities/civic_shout_engagement/aggregate_emails",
)
_V2_ARTIFACTS_DIR = _V2_ARTIFACTS_BASE / "v2_artifacts"
_S3_ARTIFACTS_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/civic_shout_engagement/aggregate_emails/v2_artifacts"
)
# Backend (content_routing) bucket + prefix consumed by the vectorize_emails cron
# (backend repo: entrypoints/external/cron/content_routing/vectorize_emails.py).
# Artifacts here must be loadable via BM25Encoder.load() and json.loads respectively.
_BE_ARTIFACT_BUCKET = "chorus-content-routing-artifacts"
_BE_ARTIFACT_PREFIX = "content_routing/vectorizers/232/v2"


# ---------------------------------------------------------------------------
# Exception subclasses (TRY003 / EM101 compliance)
# ---------------------------------------------------------------------------


class _DenseShapeError(ValueError):
    def __init__(self, got: tuple[int, ...], expected: tuple[int, int]) -> None:
        super().__init__(f"Dense shape mismatch: got {got}, expected {expected}")


# ---------------------------------------------------------------------------
# Phase 0 — Load + HTML-strip
# ---------------------------------------------------------------------------


def _html_to_text(html: str | None) -> str:
    """Convert email HTML body to whitespace-normalized plain text.

    Tolerant of None / empty / malformed input. Returns "" for all of those.
    """
    if not html:
        return ""
    # BeautifulSoup is tolerant of malformed HTML with "html.parser" backend.
    soup = BeautifulSoup(html, "html.parser")
    # Drop <script> / <style> content entirely.
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse runs of whitespace; .get_text already strips per-tag,
    # but multi-space runs can still appear from inline elements.
    return " ".join(text.split())


def _build_text_input(subject: str | None, body_html: str | None) -> str:
    """Concatenate subject + plain-text body. Truncate to _TEXT_MAX_CHARS."""
    subject_clean = (subject or "").strip()
    body_clean = _html_to_text(body_html)
    text = subject_clean if not body_clean else f"{subject_clean}\n\n{body_clean}"
    if len(text) > _TEXT_MAX_CHARS:
        text = text[:_TEXT_MAX_CHARS]
    return text


def _load_v1_with_text() -> pl.DataFrame:
    """Load aggregate_emails v1 and attach text_content column."""
    df = aggregate_emails_cache.get(1)
    logger.info("Loaded v1: %d rows", len(df))
    subjects = df["subject"].to_list()
    bodies = df["email_body"].to_list()
    texts = [_build_text_input(s, b) for s, b in zip(subjects, bodies, strict=False)]
    empty_count = sum(1 for t in texts if not t)
    if empty_count > 0:
        logger.warning(
            "Phase 0: %d/%d rows produced empty text_content (both subject and body "
            "were empty); a whitespace placeholder will be used to keep row alignment.",
            empty_count,
            len(texts),
        )
    return df.with_columns(pl.Series("text_content", texts, dtype=pl.Utf8))


# ---------------------------------------------------------------------------
# Phase A — TFIDF
# ---------------------------------------------------------------------------


def _fit_tfidf(texts: list[str]) -> tuple[csr_matrix, TfidfVectorizer]:
    """Fit TfidfVectorizer on email corpus, return L2-normalized CSR matrix."""
    vectorizer = TfidfVectorizer(
        max_features=_TFIDF_MAX_FEATURES,
        ngram_range=_TFIDF_NGRAM,
        min_df=_TFIDF_MIN_DF,
        max_df=_TFIDF_MAX_DF,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    raw = cast("csr_matrix", vectorizer.fit_transform(texts))
    normalized = cast(
        "csr_matrix",
        sk_normalize(raw, norm="l2", axis=1, copy=False),
    )
    if normalized.format != "csr":
        normalized = normalized.tocsr()
    logger.info(
        "TFIDF: shape=%s, nnz=%d, vocab=%d",
        normalized.shape,
        normalized.nnz,
        len(vectorizer.vocabulary_),
    )
    return normalized, vectorizer


# ---------------------------------------------------------------------------
# Phase B — BM25 with index compaction
# ---------------------------------------------------------------------------


def _download_nltk_data() -> None:
    logger.info("Downloading NLTK data (punkt, punkt_tab, stopwords)")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


def _fit_bm25(
    texts: list[str],
) -> tuple[csr_matrix, BM25Encoder, int, dict[int, int]]:
    """Fit BM25Encoder on email corpus, return L2-normalized CSR matrix + idx_map.

    Returns:
        (normalized_csr, encoder, n_cols, idx_map) where idx_map maps each raw
        Pinecone token index to the compact column index used in the CSR matrix.
    """
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
    logger.info(
        "BM25: shape=%s, nnz=%d, vocab=%d",
        normalized.shape,
        normalized.nnz,
        n_cols,
    )
    return normalized, encoder, n_cols, idx_map


# ---------------------------------------------------------------------------
# Phase C — Dense embed (async, context-managed)
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
    # OpenAI text-embedding-3-* is returned unit-normalized; re-normalize
    # defensively to match news_stories_graph convention.
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
# Phase D — Assemble (polars Struct columns)
# ---------------------------------------------------------------------------


def _csr_to_struct_column(matrix: csr_matrix, name: str) -> pl.Series:
    """Build a polars Struct<indices: List[UInt32], values: List[Float32]> series.

    Indices are sorted ascending per row so downstream consumers can assume
    monotonic order for set-operation-style merges.  sklearn's TfidfVectorizer
    and scipy's tocsr() do not guarantee sorted indices post-normalize.
    """
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


def _assemble(
    v1_with_text: pl.DataFrame,
    dense: np.ndarray,
    tfidf: csr_matrix,
    bm25: csr_matrix,
) -> pl.DataFrame:
    """Attach the four new vector columns to the v1 DataFrame."""
    dense_rows = dense.tolist()
    return v1_with_text.with_columns(
        pl.Series("dense_vector", dense_rows, dtype=pl.List(pl.Float32)),
        _csr_to_struct_column(tfidf, "tfidf_vector"),
        _csr_to_struct_column(bm25, "sparse_vector"),
    )


# ---------------------------------------------------------------------------
# Phase E — Put + sidecars
# ---------------------------------------------------------------------------


def _dump_tfidf_vocab(vectorizer: TfidfVectorizer, path: Path) -> None:
    """Write TFIDF vocabulary + IDF weights + config as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab": {tok: int(idx) for tok, idx in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.astype(np.float32).tolist(),
        "config": {
            "max_features": _TFIDF_MAX_FEATURES,
            "ngram_range": list(_TFIDF_NGRAM),
            "min_df": _TFIDF_MIN_DF,
            "max_df": _TFIDF_MAX_DF,
            "sublinear_tf": True,
            "stop_words": "english",
            "norm": "l2",
        },
    }
    with path.open("w") as f:
        json.dump(payload, f)
    logger.info("TFIDF vocab written to %s (%d tokens)", path, len(payload["vocab"]))


def _dump_bm25_params(
    encoder: BM25Encoder,
    n_docs: int,
    n_cols: int,
    path: Path,
) -> None:
    """Write BM25 encoder hyperparameters + corpus stats as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pt_version = importlib.metadata.version("pinecone-text")
    except importlib.metadata.PackageNotFoundError:
        pt_version = "unknown"
    payload: dict[str, object] = {
        "k1": float(encoder.k1),
        "b": float(encoder.b),
        "n_docs": int(n_docs),
        "n_cols": int(n_cols),
        "avg_doc_len": float(encoder.avgdl or 0.0),
        "doc_freq": {int(tok): int(df) for tok, df in (encoder.doc_freq or {}).items()},
        "pinecone_text_version": pt_version,
        "nltk_data_version": _NLTK_DATA_VERSION,
    }
    with path.open("w") as f:
        json.dump(payload, f)
    logger.info("BM25 params written to %s", path)


def _dump_bm25_encoder_and_idx_map(
    encoder: BM25Encoder,
    idx_map: dict[int, int],
    encoder_path: Path,
    idx_map_path: Path,
) -> None:
    """Write BM25Encoder native JSON + idx_map JSON for the backend cron.

    encoder_path: loadable via BM25Encoder.load(path)
    idx_map_path: {str(raw_pinecone_index): compact_col_index} — shape expected
                  by the content_routing vectorize cron in the backend repo.
    """
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(encoder_path))
    idx_map_path.write_text(json.dumps({str(k): v for k, v in idx_map.items()}))
    logger.info("BM25 encoder dumped to %s", encoder_path)
    logger.info(
        "BM25 idx_map dumped to %s (n_keys=%d)",
        idx_map_path,
        len(idx_map),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main(*, dry_run: bool = False, limit: int | None = None) -> None:
    """Build aggregate_emails v2 and write to the entity cache.

    Args:
        dry_run: When True, skip ``cache.put`` and S3 sidecar uploads.
        limit:   If set, truncate to the first N rows (for plumbing tests).
    """
    _download_nltk_data()

    df = _load_v1_with_text()
    if limit is not None:
        df = df.head(limit)
        logger.info("Limited to %d rows", limit)

    texts = df["text_content"].to_list()
    texts = [t or " " for t in texts]

    tfidf_matrix, tfidf_vec = _fit_tfidf(texts)
    bm25_matrix, bm25_enc, bm25_n_cols, idx_map = _fit_bm25(texts)
    dense = asyncio.run(_embed_dense(texts))

    out = _assemble(df, dense=dense, tfidf=tfidf_matrix, bm25=bm25_matrix)
    logger.info("aggregate_emails v2: %d rows, %d columns", len(out), len(out.columns))

    if dry_run:
        logger.info("Dry run: skipping cache.put + sidecar upload")
        return

    aggregate_emails_cache.put(VERSION, out)
    logger.info("aggregate_emails v2 written to cache.")

    _V2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tfidf_path = _V2_ARTIFACTS_DIR / "tfidf_vocab.json"
    bm25_path = _V2_ARTIFACTS_DIR / "bm25_params.json"
    _dump_tfidf_vocab(tfidf_vec, tfidf_path)
    _dump_bm25_params(bm25_enc, n_docs=len(texts), n_cols=bm25_n_cols, path=bm25_path)
    bm25_encoder_path = _V2_ARTIFACTS_DIR / "bm25_encoder.json"
    idx_map_path = _V2_ARTIFACTS_DIR / "idx_map.json"
    _dump_bm25_encoder_and_idx_map(bm25_enc, idx_map, bm25_encoder_path, idx_map_path)

    s3.upload(tfidf_path, DEFAULT_BUCKET, f"{_S3_ARTIFACTS_PREFIX}/tfidf_vocab.json")
    s3.upload(bm25_path, DEFAULT_BUCKET, f"{_S3_ARTIFACTS_PREFIX}/bm25_params.json")
    # DS bucket copies (symmetry with other sidecars)
    s3.upload(
        bm25_encoder_path,
        DEFAULT_BUCKET,
        f"{_S3_ARTIFACTS_PREFIX}/bm25_encoder.json",
    )
    s3.upload(idx_map_path, DEFAULT_BUCKET, f"{_S3_ARTIFACTS_PREFIX}/idx_map.json")
    # Backend bucket — consumed by content_routing vectorize_emails cron
    s3.upload(
        bm25_encoder_path,
        _BE_ARTIFACT_BUCKET,
        f"{_BE_ARTIFACT_PREFIX}/bm25_encoder.json",
    )
    s3.upload(
        idx_map_path,
        _BE_ARTIFACT_BUCKET,
        f"{_BE_ARTIFACT_PREFIX}/idx_map.json",
    )
    logger.info("Sidecars uploaded to S3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build civic_shout_engagement__aggregate_emails v2.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(dry_run=args.dry_run, limit=args.limit)
