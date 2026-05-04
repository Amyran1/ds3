"""Create v1 of the news_stories_graph dense_embeddings entity cache.

Embeds the full ~3.5M news story corpus with OpenAI text-embedding-3-small
(1536 dims) and writes a (N, 1536) L2-normalized float32 matrix plus a
row-ordered story_ids side-car via ``DenseEmbeddingsCache``.

This is the only cost-bearing script in the news_stories_graph construction
plan (~$17.50, ~2 h wall-clock). The build is chunked externally into 70
chunks of 50K rows each so memory stays bounded (~300 MB per chunk) and any
mid-run crash can resume without re-spending on completed chunks.

Hard gates enforced at the calibration probe:
    1. Projected total wall-clock must be <= 240 min
    2. Extrapolated total cost must be <= $25
    3. Calibration deviation from plan estimate must be <= 3x

Any violation triggers ``sys.exit(1)`` before the full build begins.

Usage:
    python -m entities.news_stories_graph.\
create_dense_embeddings_v1
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from entities.news_stories.cache import (
    cache as news_stories_cache,
)
from entities.news_stories_graph import (
    dense_embeddings_cache as _dec_module,
)
from libs.budget import Budget
from libs.calibrate import calibrate
from libs.checkpoint import auto_concurrency
from libs.clients.openai import OpenAIClient
from libs.costs.tracker import CostTracker
from libs.settings import Settings
from libs.validation import preflight_check

dense_embeddings_cache = _dec_module.cache

logger = logging.getLogger(__name__)

# ---- constants ------------------------------------------------------------

MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# 500 rows/batch at ~250 tokens/story average ~125K tokens/request, well
# under OpenAI's 300K tokens/request embeddings cap. The calibration probe
# validates the token estimate and drops the batch to 300 if texts are
# heavier than expected.
BATCH_SIZE = 500
# Per-chunk row count for the outer (checkpointed) loop. 50K rows by 1536
# float32 ~ 300 MB in memory and produces a fragment file of the same size
# on disk. 70 chunks of 50K = 3.5M rows nominal.
CHUNK_SIZE = 50_000
# Initial client concurrency (inner OpenAI batches in flight). The
# calibration probe runs at this value; after the probe we optionally
# lower it via auto_concurrency. We never raise above this cap.
MAX_CONCURRENCY = 50
MIN_CONCURRENCY = 5

COST_KEY = "news-stories-graph-dense-embeddings-v1"
BUDGET_DOLLARS = 25.0
# Reservation for the full run pre-check. 3.5M texts at ~250 tok at
# $0.02/1M ~ $17.50. We check $18 (plan IO Budget headroom).
FULL_RUN_RESERVATION = 18.0

# Hard gates (non-negotiable; do NOT weaken these).
WALL_CLOCK_CEILING_MIN = 240.0
PROJECTED_COST_CEILING = 25.0
MAX_DEVIATION = 3.0
TARGET_MINUTES = 75.0
TOKEN_WARN_THRESHOLD = 500.0
CHARS_PER_TOKEN = 4.0

# Plan estimate from Addendum 3 Compute Budget: ~500 rows/sec via async
# 25-concurrent batches. We run at max_concurrent=50 (2x) so the expected
# rate is ~1000 rows/sec, but we calibrate anyway.
ESTIMATED_RATE = 1000.0

# 3.5M conservative cap for projection math; real count comes from the
# cache at runtime.
TOTAL_ITEMS_EST = 3_500_000

CHECKPOINT_DIR = Path(
    "data/entities/news_stories_graph/dense_embeddings/checkpoints",
)


# ---- small helpers --------------------------------------------------------


def _chunk_path(idx: int) -> Path:
    return CHECKPOINT_DIR / f"chunk_{idx:04d}.npy"


def _done_path(idx: int) -> Path:
    return CHECKPOINT_DIR / f"chunk_{idx:04d}_done.marker"


def _is_chunk_done(idx: int) -> bool:
    return _done_path(idx).exists() and _chunk_path(idx).exists()


def _build_text(name: str | None, summary: str | None) -> str:
    """Concatenate story name + summary with a blank line separator.

    The A/B finding (07_ab_metadata_augmentation.md) ruled that baseline
    text outperforms metadata-augmented text, so this is the one and only
    embedding input format.
    """
    n = (name or "").strip()
    s = (summary or "").strip()
    if n and s:
        return f"{n}\n\n{s}"
    return n or s


async def _embed_chunk(
    client: OpenAIClient,
    texts: list[str],
    idx: int,
) -> np.ndarray:
    """Embed one chunk of texts and return a float32 (len, 1536) array."""
    t0 = time.monotonic()
    embeddings = await client.embed(
        texts,
        model=MODEL,
        batch_size=BATCH_SIZE,
        cost_key=COST_KEY,
    )
    elapsed = time.monotonic() - t0

    arr = np.asarray(embeddings, dtype=np.float32)
    expected = (len(texts), EMBEDDING_DIM)
    if arr.shape != expected:
        msg = f"Chunk {idx}: expected shape {expected}, got {arr.shape}"
        raise RuntimeError(msg)

    rate = len(texts) / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Chunk %d embedded: %d rows in %.1fs (%.0f rows/s)",
        idx,
        len(texts),
        elapsed,
        rate,
    )
    return arr


def _save_chunk_fragment(arr: np.ndarray, idx: int) -> None:
    """Write chunk_{idx}.npy and then touch the done marker."""
    path = _chunk_path(idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    _done_path(idx).touch()


# ---- pipeline steps -------------------------------------------------------


@dataclass(frozen=True)
class CorpusData:
    """Prepared, sanitized embedding inputs with aligned story_ids."""

    texts: list[str]
    story_ids: list[str]

    @property
    def n_total(self) -> int:
        return len(self.texts)


def _load_corpus() -> CorpusData:
    """Load news_stories v1, build text column, sort, sanitize."""
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

    names = prepped["name"].to_list()
    summaries = prepped["summary"].to_list()
    story_ids: list[str] = [str(sid) for sid in prepped["story_id"].to_list()]
    _pairs = zip(names, summaries, strict=True)
    texts: list[str] = [_build_text(n, s) for n, s in _pairs]

    # Final safety: drop any remaining empty rows after _build_text.
    keep_idx = [i for i, t in enumerate(texts) if t]
    if len(keep_idx) != len(texts):
        dropped = len(texts) - len(keep_idx)
        logger.warning("Dropping %d rows with empty text after _build_text", dropped)
        texts = [texts[i] for i in keep_idx]
        story_ids = [story_ids[i] for i in keep_idx]

    logger.info("Final corpus: %d texts to embed", len(texts))

    # Preflight sanitize once (fills placeholders, truncates >8192-char).
    texts = preflight_check(texts, operation="embed")
    return CorpusData(texts=texts, story_ids=story_ids)


def _evaluate_gates(
    *,
    projected_minutes: float,
    projected_cost: float,
    deviation: float,
) -> list[str]:
    """Apply the three hard gates and return a list of failure messages."""
    failures: list[str] = []

    if projected_minutes > WALL_CLOCK_CEILING_MIN:
        msg = (
            f"GATE A (wall-clock): {projected_minutes:.1f} min > "
            f"{WALL_CLOCK_CEILING_MIN:.0f} min ceiling"
        )
        logger.error("FAIL %s", msg)
        failures.append(msg)
    else:
        logger.info(
            "GATE A (wall-clock): PASS -- %.1f min <= %.0f min ceiling",
            projected_minutes,
            WALL_CLOCK_CEILING_MIN,
        )

    if projected_cost > PROJECTED_COST_CEILING:
        msg = (
            f"GATE B (budget): projected ${projected_cost:.2f} > "
            f"${PROJECTED_COST_CEILING:.2f} ceiling"
        )
        logger.error("FAIL %s", msg)
        failures.append(msg)
    else:
        logger.info(
            "GATE B (budget): PASS -- projected $%.2f <= $%.2f ceiling",
            projected_cost,
            PROJECTED_COST_CEILING,
        )

    if deviation > MAX_DEVIATION:
        dev_str = f"{deviation:.1f}x"
        max_str = f"{MAX_DEVIATION:.1f}x"
        msg = f"GATE C (calibration deviation): {dev_str} > {max_str} ceiling"
        logger.error("FAIL %s", msg)
        failures.append(msg)
    else:
        logger.info(
            "GATE C (calibration deviation): PASS -- %.1fx <= %.1fx ceiling",
            deviation,
            MAX_DEVIATION,
        )

    return failures


async def _run_calibration_probe(
    corpus: CorpusData,
    settings: Settings,
    tracker: CostTracker,
) -> float:
    """Embed chunk 0, evaluate gates, return measured rows/sec.

    On gate failure, deletes the chunk_0 fragment and sys.exit(1). On
    success, marks chunk 0 done.
    """
    logger.info("=" * 60)
    logger.info(
        "Calibration probe: chunk 0 (%d rows)",
        min(CHUNK_SIZE, corpus.n_total),
    )
    logger.info("=" * 60)

    chunk0_texts = corpus.texts[:CHUNK_SIZE]
    avg_chars = sum(len(t) for t in chunk0_texts) / max(len(chunk0_texts), 1)
    approx_tokens_per_text = avg_chars / CHARS_PER_TOKEN
    logger.info(
        "Probe token estimate: %.0f chars/text ~= %.0f tokens/text",
        avg_chars,
        approx_tokens_per_text,
    )
    if approx_tokens_per_text > TOKEN_WARN_THRESHOLD:
        logger.warning(
            "Avg tokens/text (%.0f) > %.0f -- the inner batch_size=%d may "
            "approach the 300K tokens/request cap. Proceeding but watch "
            "for rate-limit errors.",
            approx_tokens_per_text,
            TOKEN_WARN_THRESHOLD,
            BATCH_SIZE,
        )

    probe_client = OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        max_concurrent=MAX_CONCURRENCY,
        cost_tracker=tracker,
    )
    cost_before_probe = tracker.get(COST_KEY).total

    async def _probe() -> int:
        async with probe_client:
            arr = await _embed_chunk(probe_client, chunk0_texts, idx=0)
        _chunk_path(0).parent.mkdir(parents=True, exist_ok=True)
        np.save(_chunk_path(0), arr)
        return arr.shape[0]

    cal = await calibrate(
        _probe,
        total_items=corpus.n_total,
        estimated_rate=ESTIMATED_RATE,
        max_deviation=MAX_DEVIATION,
        wall_clock_gate_min=WALL_CLOCK_CEILING_MIN,
        gate_strict=False,  # we apply project-specific gates below
    )
    probe_cost = tracker.get(COST_KEY).total - cost_before_probe
    projected_total_cost = probe_cost / max(cal.probe_items, 1) * corpus.n_total

    probe_prefix = "Probe results: %d rows in %.1fs -> %.0f rows/s, "
    probe_suffix = "$%.4f cost (projected full = $%.2f)"
    probe_line = probe_prefix + probe_suffix
    logger.info(
        probe_line,
        cal.probe_items,
        cal.probe_seconds,
        cal.measured_rate,
        probe_cost,
        projected_total_cost,
    )

    failures = _evaluate_gates(
        projected_minutes=cal.projected_minutes,
        projected_cost=projected_total_cost,
        deviation=cal.deviation_factor,
    )
    if failures:
        with contextlib.suppress(OSError):
            _chunk_path(0).unlink(missing_ok=True)
        logger.error(
            "Calibration gates FAILED (%d). Aborting before full build.",
            len(failures),
        )
        sys.exit(1)

    _done_path(0).touch()
    logger.info("Calibration gates PASSED. Chunk 0 committed.")
    return cal.measured_rate


async def _run_full_build(
    corpus: CorpusData,
    settings: Settings,
    tracker: CostTracker,
    concurrency: int,
    n_chunks: int,
) -> None:
    """Sequentially process remaining pending chunks."""
    pending = [i for i in range(n_chunks) if not _is_chunk_done(i)]
    logger.info(
        "Full build: %d/%d chunks pending (already done: %d)",
        len(pending),
        n_chunks,
        n_chunks - len(pending),
    )
    if not pending:
        logger.info("No chunks pending; all fragments already on disk")
        return

    full_client = OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        max_concurrent=concurrency,
        cost_tracker=tracker,
    )
    t_build_start = time.monotonic()
    async with full_client:
        for pos, idx in enumerate(pending, start=1):
            start = idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, corpus.n_total)
            chunk_texts = corpus.texts[start:end]

            t_chunk = time.monotonic()
            arr = await _embed_chunk(full_client, chunk_texts, idx=idx)
            _save_chunk_fragment(arr, idx)
            chunk_elapsed = time.monotonic() - t_chunk

            cumulative = tracker.get(COST_KEY).total
            avg_chunk_s = (time.monotonic() - t_build_start) / pos
            remaining = len(pending) - pos
            eta_min = (remaining * avg_chunk_s) / 60
            logger.info(
                "chunk %d/%d complete (%d rows, %.1fs, cumulative $%.2f, ETA %.1fm)",
                idx,
                n_chunks - 1,
                arr.shape[0],
                chunk_elapsed,
                cumulative,
                eta_min,
            )

            if cumulative > PROJECTED_COST_CEILING:
                logger.error(
                    "BUDGET BREACH mid-run: $%.2f > $%.2f. Aborting.",
                    cumulative,
                    PROJECTED_COST_CEILING,
                )
                sys.exit(1)


def _assemble_and_persist(corpus: CorpusData, n_chunks: int) -> np.ndarray:
    """Stack all chunk fragments, L2-normalize, write to cache."""
    logger.info("Assembling %d chunk fragments", n_chunks)
    missing = [i for i in range(n_chunks) if not _is_chunk_done(i)]
    if missing:
        msg = f"Missing {len(missing)} chunk marker(s) after build: {missing[:5]}..."
        raise RuntimeError(msg)

    fragments: list[np.ndarray] = [np.load(_chunk_path(i)) for i in range(n_chunks)]
    arr = np.vstack(fragments)
    del fragments

    if arr.shape[0] != corpus.n_total:
        msg = (
            f"Assembled shape {arr.shape} does not match expected "
            f"({corpus.n_total}, {EMBEDDING_DIM})"
        )
        raise RuntimeError(msg)
    logger.info("Assembled matrix shape: %s", arr.shape)

    logger.info("L2-normalizing rows")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = (arr / norms).astype(np.float32)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(arr.shape[0], size=min(10, arr.shape[0]), replace=False)
    sample_norms = np.linalg.norm(arr[sample_idx], axis=1)
    logger.info(
        "Sample norms after normalization: min=%.4f max=%.4f",
        float(sample_norms.min()),
        float(sample_norms.max()),
    )

    logger.info("Persisting to DenseEmbeddingsCache (local + S3)")
    dense_embeddings_cache.put(arr, corpus.story_ids, version=1)
    return arr


async def create() -> None:
    """Build news_stories_graph dense_embeddings v1 end-to-end."""
    settings = Settings()
    tracker = CostTracker(Path(settings.cost_ledger_path))
    budget = Budget(limit=BUDGET_DOLLARS, tracker=tracker)

    budget.check(FULL_RUN_RESERVATION)
    logger.info(
        "Pre-flight budget check passed: $%.2f remaining (limit $%.2f)",
        budget.remaining(),
        BUDGET_DOLLARS,
    )

    await asyncio.to_thread(CHECKPOINT_DIR.mkdir, parents=True, exist_ok=True)

    corpus = _load_corpus()
    n_total = corpus.n_total
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info("Chunk plan: %d chunks of up to %d rows each", n_chunks, CHUNK_SIZE)

    done_before = [i for i in range(n_chunks) if _is_chunk_done(i)]
    pending_before = [i for i in range(n_chunks) if i not in set(done_before)]
    if done_before:
        logger.info(
            "Resume: %d/%d chunks already complete, %d remaining (first pending = %d)",
            len(done_before),
            n_chunks,
            len(pending_before),
            pending_before[0] if pending_before else n_chunks,
        )
    else:
        logger.info("Fresh build: 0/%d chunks done", n_chunks)

    ran_probe = False
    if 0 not in set(done_before):
        measured_rate = await _run_calibration_probe(corpus, settings, tracker)
        ran_probe = True
    else:
        logger.info("Resume: skipping calibration probe (chunk 0 already done)")
        measured_rate = ESTIMATED_RATE

    concurrency = auto_concurrency(
        total_items=n_total,
        measured_rate=measured_rate,
        target_minutes=TARGET_MINUTES,
        max_concurrency=MAX_CONCURRENCY,
        min_concurrency=MIN_CONCURRENCY,
    )
    logger.info(
        "auto_concurrency -> %d (measured=%.0f rows/s, target=%.0f min, cap=%d)",
        concurrency,
        measured_rate,
        TARGET_MINUTES,
        MAX_CONCURRENCY,
    )

    await _run_full_build(corpus, settings, tracker, concurrency, n_chunks)
    arr = _assemble_and_persist(corpus, n_chunks)

    total_cost = tracker.get(COST_KEY).total
    if total_cost > PROJECTED_COST_CEILING:
        tc_str = f"${total_cost:.2f}"
        cc_str = f"${PROJECTED_COST_CEILING:.2f}"
        msg = f"Final cost {tc_str} exceeded ceiling {cc_str}"
        raise RuntimeError(msg)

    logger.info("=" * 60)
    logger.info("news_stories_graph dense_embeddings v1 build complete")
    logger.info("  Shape: %s", arr.shape)
    logger.info("  story_ids: %d", len(corpus.story_ids))
    logger.info("  Total cost: $%.4f", total_cost)
    logger.info("  Budget remaining: $%.2f", budget.remaining())
    logger.info("  Ran calibration probe this invocation: %s", ran_probe)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    asyncio.run(create())
