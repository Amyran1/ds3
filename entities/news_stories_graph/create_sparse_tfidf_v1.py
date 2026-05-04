r"""Create v1 of the news_stories_graph sparse_tfidf entity cache.

Fits a ``sklearn.feature_extraction.text.TfidfVectorizer`` on the full
~3.5M news story corpus (``name + "\\n\\n" + summary``) and writes three
artifacts via ``SparseTfidfCache.put``:

    tfidf_v1.npz       - L2-normalized (N x 50K) CSR matrix
    vocab_v1.txt       - row-ordered 50K vocabulary
    story_ids_v1.json  - row-ordered story_ids aligned to the matrix

Row order matches ``create_dense_embeddings_v1.py::_load_corpus``:

    1. drop null story_ids
    2. strip + fill-null on name / summary
    3. drop rows whose joined ``name\\n\\nsummary`` is empty
    4. sort by ``story_id`` ascending

TF-IDF config (locked; do not change without orchestrator approval):

    max_features = 50_000
    ngram_range  = (1, 2)
    min_df       = 5       # raised from EDA's 2 at full-corpus scale
    max_df       = 0.5
    stop_words   = "english"
    sublinear_tf = True
    lowercase    = True
    strip_accents = "unicode"

Usage::

    python -m entities.\
news_stories_graph.create_sparse_tfidf_v1
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from entities.news_stories.cache import (
    cache as news_stories_cache,
)
from entities.news_stories_graph import (
    sparse_tfidf_cache as _stc_module,
)

sparse_tfidf_cache = _stc_module.cache

logger = logging.getLogger(__name__)

# ---- constants ------------------------------------------------------------

MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)
MIN_DF = 5
MAX_DF = 0.5
VERSION = 1


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

    # Final belt-and-suspenders: drop any empty-text rows the expression missed.
    keep = [i for i, t in enumerate(texts) if t]
    if len(keep) != len(texts):
        dropped = len(texts) - len(keep)
        logger.warning("Dropping %d residual empty-text rows", dropped)
        texts = [texts[i] for i in keep]
        story_ids = [story_ids[i] for i in keep]

    logger.info("Final corpus: %d documents", len(texts))
    return texts, story_ids


def _fit_tfidf(texts: list[str]) -> tuple[csr_matrix, list[str]]:
    """Fit TfidfVectorizer, return (L2-normalized CSR matrix, vocab list)."""
    logger.info(
        "Fitting TfidfVectorizer: max_features=%d, ngram_range=%s, "
        "min_df=%d, max_df=%.2f, sublinear_tf=True",
        MAX_FEATURES,
        NGRAM_RANGE,
        MIN_DF,
        MAX_DF,
    )
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    t0 = time.monotonic()
    raw = cast("csr_matrix", vectorizer.fit_transform(texts))
    fit_elapsed = time.monotonic() - t0
    shape = cast("tuple[int, int]", raw.shape)
    density = raw.nnz / (shape[0] * shape[1]) if shape[0] and shape[1] else 0.0
    logger.info(
        "fit_transform done in %.1fs: shape=%s, nnz=%d, density=%.4f%%, vocab=%d",
        fit_elapsed,
        shape,
        raw.nnz,
        density * 100,
        len(vectorizer.vocabulary_),
    )

    logger.info("L2-normalizing rows (in-place)")
    t1 = time.monotonic()
    normalized = cast(
        "csr_matrix",
        sk_normalize(raw, norm="l2", axis=1, copy=False),
    )
    if normalized.format != "csr":
        normalized = normalized.tocsr()
    logger.info("normalize done in %.1fs", time.monotonic() - t1)

    vocab: list[str] = [str(tok) for tok in vectorizer.get_feature_names_out()]
    return normalized, vocab


def create() -> None:
    """Build news_stories_graph sparse_tfidf v1 end-to-end."""
    t_total = time.monotonic()

    texts, story_ids = _load_corpus()
    matrix, vocab = _fit_tfidf(texts)

    shape = cast("tuple[int, int]", matrix.shape)
    n_rows, n_cols = shape
    n_ids = len(story_ids)
    n_vocab = len(vocab)
    if n_rows != n_ids:
        msg = f"Row mismatch: {n_rows} matrix rows vs {n_ids} story_ids"
        raise RuntimeError(msg)
    if n_cols != n_vocab:
        msg = f"Col mismatch: {n_cols} matrix cols vs {n_vocab} vocab"
        raise RuntimeError(msg)

    logger.info("Persisting to SparseTfidfCache (local + S3)")
    sparse_tfidf_cache.put(matrix, vocab, story_ids, version=VERSION)

    total_elapsed = time.monotonic() - t_total
    logger.info("=" * 60)
    logger.info("news_stories_graph sparse_tfidf v1 build complete")
    logger.info("  Shape:   %s", shape)
    logger.info("  NNZ:     %d", matrix.nnz)
    logger.info("  Vocab:   %d", len(vocab))
    logger.info("  Stories: %d", len(story_ids))
    logger.info("  Elapsed: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    create()
