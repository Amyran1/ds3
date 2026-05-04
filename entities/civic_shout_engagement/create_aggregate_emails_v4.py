r"""Build civic_shout_engagement__aggregate_emails v4.

Inherits all 23 columns from v3 and appends one new column:

- ``dense_vector_large`` — 3072-d OpenAI ``text-embedding-3-large`` vector,
  L2-normalized. Same source text (``text_content``) as v2's ``dense_vector``.

Derived-only build: no Redshift re-pull, no counter recompute. Allows direct
A/B of -small (``dense_vector``) vs -large (``dense_vector_large``) in a
single read of the v4 cache.

Build anchor: inherits v3 anchors directly (``BUILD_CUTOFF_UTC`` is implicit
in v3's content/counters; v4 adds nothing beyond a new embedding column).

Usage::

    # 200-email API smoke (cost gate before full build)
    OPENAI_API_KEY=<secret> python -m \
        entities.civic_shout_engagement.create_aggregate_emails_v4 \
        --limit 200 --no-write

    # Full corpus build (~4575 emails, ~$0.40)
    OPENAI_API_KEY=<secret> python -m \
        entities.civic_shout_engagement.create_aggregate_emails_v4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from entities.civic_shout_engagement.aggregate_emails_cache import (
    cache as aggregate_emails_cache,
)
from libs.clients.openai import OpenAIClient
from libs.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = 4
_DENSE_MODEL = "text-embedding-3-large"
_DENSE_DIMS = 3072
_EMBED_BATCH_SIZE = 500
_MAX_CONCURRENT = 25
_COST_KEY = "civic_shout_engagement__aggregate_emails_v4"
_DENSE_VECTOR_MIN_PRESENT = 0.99
_L2_NORM_TOLERANCE = 1e-4
_L2_NORM_MIN_FRACTION = 0.995

_TIMING_JSONL = Path(
    "entities/civic_shout_engagement/aggregate_emails_v4_timing_performance.jsonl",
)


# ---------------------------------------------------------------------------
# Exception subclasses (TRY003 / EM101 compliance)
# ---------------------------------------------------------------------------


class _DenseShapeError(ValueError):
    def __init__(self, got: tuple[int, ...], expected: tuple[int, int]) -> None:
        msg = f"Dense shape mismatch: got {got}, expected {expected}"
        super().__init__(msg)


class _DenseCoverageError(ValueError):
    def __init__(self, present_fraction: float, threshold: float) -> None:
        msg = (
            f"dense_vector_large coverage {present_fraction:.4f} below "
            f"threshold {threshold:.4f}"
        )
        super().__init__(msg)


class _DenseNormError(ValueError):
    def __init__(self, ok_fraction: float, threshold: float) -> None:
        msg = (
            f"dense_vector_large L2-norm pass-rate {ok_fraction:.4f} below "
            f"threshold {threshold:.4f}"
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


async def _embed_dense_large(texts: list[str]) -> tuple[np.ndarray, float]:
    """Embed texts with text-embedding-3-large.

    Returns (L2-normalized array, elapsed_s).
    """
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
        "Dense-large embed: %d texts in %.1fs (%.0f texts/s)",
        len(texts),
        elapsed,
        len(texts) / elapsed if elapsed > 0 else 0.0,
    )
    return arr, elapsed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_dense_large(df: pl.DataFrame) -> None:
    """Coverage + L2-norm gates on dense_vector_large column."""
    n_rows = len(df)
    n_present = df.select(
        pl.col("dense_vector_large").is_not_null().sum(),
    ).item()
    present_fraction = n_present / max(n_rows, 1)
    logger.info(
        "Coverage gate: dense_vector_large present in %d/%d rows (%.4f)",
        n_present,
        n_rows,
        present_fraction,
    )
    if present_fraction < _DENSE_VECTOR_MIN_PRESENT:
        raise _DenseCoverageError(present_fraction, _DENSE_VECTOR_MIN_PRESENT)

    sample = (
        df.filter(pl.col("dense_vector_large").is_not_null())
        .select("dense_vector_large")
        .head(min(2000, n_present))
    )
    arr = np.asarray(
        sample["dense_vector_large"].list.to_array(_DENSE_DIMS).to_numpy(),
        dtype=np.float32,
    )
    norms = np.linalg.norm(arr, axis=1)
    ok = np.abs(norms - 1.0) < _L2_NORM_TOLERANCE
    ok_fraction = float(ok.mean())
    logger.info(
        "L2-norm gate: %.4f of %d sampled vectors within ±%.0e of unit",
        ok_fraction,
        len(arr),
        _L2_NORM_TOLERANCE,
    )
    if ok_fraction < _L2_NORM_MIN_FRACTION:
        raise _DenseNormError(ok_fraction, _L2_NORM_MIN_FRACTION)


# ---------------------------------------------------------------------------
# Timing record
# ---------------------------------------------------------------------------


def _record_timing(
    *,
    n_emails: int,
    elapsed_s: float,
    note: str,
) -> None:
    """Append a row to aggregate_emails_v4_timing_performance.jsonl."""
    row = {
        "feature_family": "aggregate_emails_v4_dense_large",
        "version": "v4",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_emails": n_emails,
        "elapsed_s": round(elapsed_s, 2),
        "rows_per_s": (
            round(n_emails / elapsed_s, 1) if elapsed_s > 0 else 0.0
        ),
        "embed_model": _DENSE_MODEL,
        "embed_dims": _DENSE_DIMS,
        "batch_size": _EMBED_BATCH_SIZE,
        "max_concurrent": _MAX_CONCURRENT,
        "note": note,
    }
    _TIMING_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("Recorded timing row: %s", note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aggregate_emails v4 with dense_vector_large.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of emails embedded (smoke test).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip cache.put — use for API smoke / cost gate.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading aggregate_emails v3...")
    v3 = aggregate_emails_cache.get(3)
    n_v3_rows = len(v3)
    n_v3_cols = len(v3.columns)
    logger.info("aggregate_emails v3: %d rows, %d columns", n_v3_rows, n_v3_cols)

    if "dense_vector_large" in v3.columns:
        msg = "v3 unexpectedly already has dense_vector_large; aborting."
        raise RuntimeError(msg)

    if args.limit is not None:
        embed_target = v3.head(args.limit)
        logger.info("Limit=%d: embedding head(%d)", args.limit, len(embed_target))
    else:
        embed_target = v3

    embed_target = embed_target.with_columns(
        pl.col("text_content").fill_null("").alias("text_content"),
    )

    texts = embed_target["text_content"].to_list()
    logger.info(
        "Embedding %d texts via %s (dims=%d)...",
        len(texts),
        _DENSE_MODEL,
        _DENSE_DIMS,
    )

    arr, elapsed = asyncio.run(_embed_dense_large(texts))

    note = "smoke" if args.limit is not None else "full"
    _record_timing(n_emails=len(texts), elapsed_s=elapsed, note=note)

    if args.no_write:
        logger.info("--no-write: skipping cache.put. Smoke complete.")
        return

    if args.limit is not None:
        msg = (
            "Refusing to write v4 with --limit set (would orphan rows). "
            "Re-run without --limit."
        )
        raise RuntimeError(msg)

    dense_series = pl.Series(
        "dense_vector_large",
        arr.tolist(),
        dtype=pl.List(pl.Float32),
    )
    v4 = v3.with_columns(dense_series)
    logger.info(
        "aggregate_emails v4: %d rows, %d columns",
        len(v4),
        len(v4.columns),
    )

    _validate_dense_large(v4)

    aggregate_emails_cache.put(4, v4)
    logger.info("aggregate_emails v4 written to cache.")


if __name__ == "__main__":
    main()
