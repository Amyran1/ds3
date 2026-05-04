"""Create v1 of the news_stories entity cache.

Pulls the full documents.news_stories collection from production Mongo in
monthly buckets (resumable via CheckpointedPipeline), joins each story's
`unique_sources` list against the news_sources reference table, and writes
one combined parquet through EntityCache.

Usage:
    python -m entities.news_stories.create_cache_v1
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from entities.news_sources.cache import (
    cache as news_sources_cache,
)
from entities.news_stories.cache import cache
from libs.budget import Budget
from libs.calibrate import calibrate
from libs.checkpoint import CheckpointConfig, CheckpointedPipeline, auto_concurrency
from libs.clients.mongo import FindOptions, MongoClient
from libs.costs.tracker import CostTracker
from libs.settings import Settings

logger = logging.getLogger(__name__)

COLLECTION = "news_stories"
DATABASE = "documents"
COST_KEY = "news-stories-ingest-v1"
BUDGET_DOLLARS = 20.0

# Upper bound on concurrent monthly mongo fetches. The actual concurrency
# is chosen by auto_concurrency() after a calibration probe; this is just
# the cap.
MAX_CONCURRENCY = 16
MIN_CONCURRENCY = 2

# Motor cursor network batch size. Default (101) is ~3-5x slower than
# necessary for 80K-doc monthly chunks.
BATCH_SIZE = 5_000

# Approximate full-corpus size for wall-clock projection. Updated from the
# 2026-04-13 build (3,518,199 rows); bump if collection grows materially.
TOTAL_ITEMS_EST = 3_600_000

# Items/sec per single-worker probe. The 2026-04-13 build measured ~200-300
# docs/sec/worker at batch_size=101; raise once batch_size=5000 is measured.
ESTIMATED_RATE = 500.0

# Wall-clock ceiling for the full run. 30 min matches the soft ceiling we
# agreed on for one-shot entity builds.
WALL_CLOCK_TARGET_MIN = 15.0
WALL_CLOCK_GATE_MIN = 30.0

RANGE_START = datetime(2022, 9, 1, tzinfo=UTC)
# Stop at the start of next month relative to "now" so the last bucket is
# closed and deterministic across re-runs of the same calendar day.
_NOW = datetime.now(UTC)
RANGE_END = datetime(
    _NOW.year + (1 if _NOW.month == 12 else 0),
    1 if _NOW.month == 12 else _NOW.month + 1,
    1,
    tzinfo=UTC,
)

CHECKPOINT_DIR = Path("data/entities/news_stories/checkpoints")

PROJECTION: dict[str, int] = {
    "_id": 0,
    "story_id": 1,
    "name": 1,
    "summary": 1,
    "key_points": 1,
    "questions": 1,
    "article_count": 1,
    "unique_article_count": 1,
    "unique_source_count": 1,
    "reprint_count": 1,
    "sentiment_positive": 1,
    "sentiment_negative": 1,
    "sentiment_neutral": 1,
    "created_at": 1,
    "updated_at": 1,
    "topics": 1,
    "people": 1,
    "companies": 1,
    "countries": 1,
    "categories": 1,
    "top_topics": 1,
    "top_people": 1,
    "top_companies": 1,
    "top_countries": 1,
    "top_categories": 1,
    "locations": 1,
    "top_locations": 1,
    "unique_sources": 1,
}


def _month_buckets(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Return [(month_start, next_month_start), ...] covering [start, end)."""
    buckets: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt_year = cur.year + (1 if cur.month == 12 else 0)
        nxt_month = 1 if cur.month == 12 else cur.month + 1
        nxt = datetime(nxt_year, nxt_month, 1, tzinfo=UTC)
        buckets.append((cur, nxt))
        cur = nxt
    return buckets


def _docs_to_df(docs: list[dict]) -> pl.DataFrame:
    """Convert a list of mongo docs to a polars DataFrame."""
    if not docs:
        return pl.DataFrame()
    # infer_schema_length=None looks at every row, which is important for
    # list/struct columns (topics, people, ...) that may be empty in the
    # first few docs.
    return pl.from_dicts(docs, infer_schema_length=None)


async def _fetch_month(
    client: MongoClient,
    start: datetime,
    end: datetime,
    chunk_idx: int,
) -> pl.DataFrame:
    """Fetch one monthly bucket from mongo and return it as a polars DF."""
    options = FindOptions(
        filter={"created_at": {"$gte": start, "$lt": end}},
        projection=PROJECTION,
        cost_key=COST_KEY,
        batch_size=BATCH_SIZE,
    )
    t0 = time.monotonic()
    docs = await client.find(COLLECTION, options)
    elapsed = time.monotonic() - t0

    logger.info(
        "Chunk %d [%s → %s]: %d docs fetched in %.1fs",
        chunk_idx,
        start.date(),
        end.date(),
        len(docs),
        elapsed,
    )
    return _docs_to_df(docs)


def _save_chunk(df: pl.DataFrame, chunk_idx: int, directory: Path) -> None:
    path = directory / f"chunk_{chunk_idx:04d}.parquet"
    if df.is_empty():
        # Still write an empty marker file so the combine step can
        # distinguish "done but empty" from "missing".
        pl.DataFrame().write_parquet(path)
    else:
        df.write_parquet(path)


def _load_chunk(chunk_idx: int, directory: Path) -> pl.DataFrame:
    return pl.read_parquet(directory / f"chunk_{chunk_idx:04d}.parquet")


def _combine(chunks: list[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [c for c in chunks if not c.is_empty()]
    if not non_empty:
        return pl.DataFrame()
    # how="diagonal_relaxed" tolerates minor schema drift across chunks
    # (e.g., a nested struct column that's all-null in one month).
    return pl.concat(non_empty, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# News sources join
# ---------------------------------------------------------------------------

_SOURCE_COLS = [
    "domain",
    "global_rank",
    "monthly_visits",
    "avg_monthly_posts",
    "paywall",
    "bias",
    "top_category",
    "top_country",
]


def _join_news_sources(stories_df: pl.DataFrame) -> pl.DataFrame:
    """Join stories to news_sources and attach story-level source rollups."""
    logger.info("Loading news_sources reference table")
    sources_df = news_sources_cache.get(1)

    missing = [c for c in _SOURCE_COLS if c not in sources_df.columns]
    if missing:
        msg = f"news_sources is missing expected columns: {missing}"
        raise ValueError(msg)

    sources_df = sources_df.select(_SOURCE_COLS).with_columns(
        pl.lit(value=True).alias("_source_matched"),
    )

    exploded = (
        stories_df.select(["story_id", "unique_sources"])
        .explode("unique_sources")
        .rename({"unique_sources": "domain"})
        .filter(pl.col("domain").is_not_null())
    )

    joined = exploded.join(sources_df, on="domain", how="left").with_columns(
        pl.col("_source_matched").fill_null(value=False),
    )

    # Coverage log
    total_pairs = len(joined)
    matched_pairs = int(joined.select(pl.col("_source_matched").sum()).item() or 0)
    stories_with_match = (
        joined.filter(pl.col("_source_matched"))
        .select(pl.col("story_id").n_unique())
        .item()
        or 0
    )
    total_stories = len(stories_df)
    pair_pct = 100.0 * matched_pairs / total_pairs if total_pairs else 0.0
    story_pct = 100.0 * stories_with_match / total_stories if total_stories else 0.0
    logger.info(
        "news_sources coverage: %d/%d pairs matched (%.1f%%), "
        "%d/%d stories with ≥1 match (%.1f%%)",
        matched_pairs,
        total_pairs,
        pair_pct,
        stories_with_match,
        total_stories,
        story_pct,
    )

    rollups = joined.group_by("story_id").agg([
        pl.col("monthly_visits").sum().alias("source_reach_total"),
        pl.col("global_rank").mean().alias("source_rank_mean"),
        pl.col("global_rank").min().alias("source_rank_min"),
        pl.col("domain")
        .filter(pl.col("_source_matched"))
        .n_unique()
        .alias("source_diversity_n"),
        pl.col("top_country")
        .filter(pl.col("_source_matched"))
        .n_unique()
        .alias("source_diversity_country_n"),
        pl.col("top_category")
        .filter(pl.col("_source_matched"))
        .n_unique()
        .alias("source_diversity_category_n"),
        pl.col("paywall").cast(pl.Float64).mean().alias("source_paywall_frac"),
        pl.col("bias").drop_nulls().unique().alias("source_bias_set"),
        pl.col("bias").drop_nulls().n_unique().alias("source_bias_n"),
        pl.col("_source_matched").sum().cast(pl.Int64).alias("source_matched_count"),
        pl.struct([
            "domain",
            "global_rank",
            "monthly_visits",
            "paywall",
            "bias",
            "top_category",
            "top_country",
        ]).alias("source_details"),
    ])

    enriched = stories_df.join(rollups, on="story_id", how="left")

    # `source_unmatched_count` = list length − matched_count
    enriched = enriched.with_columns(
        (
            pl.col("unique_sources").list.len().cast(pl.Int64)
            - pl.col("source_matched_count").fill_null(0)
        ).alias("source_unmatched_count"),
    )

    return enriched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def create() -> None:
    """Build the news_stories v1 cache from production Mongo."""
    settings = Settings()
    tracker = CostTracker(Path(settings.cost_ledger_path))
    budget = Budget(limit=BUDGET_DOLLARS, tracker=tracker)

    # Pre-flight budget check (full pull is cheap per estimate_mongo_read
    # defaults, but we guard with a conservative reservation).
    budget.check(5.0)
    logger.info("Budget check passed: $%.2f remaining", budget.remaining())

    buckets = _month_buckets(RANGE_START, RANGE_END)
    logger.info(
        "Planning %d monthly buckets from %s to %s",
        len(buckets),
        RANGE_START.date(),
        RANGE_END.date(),
    )

    mongo_uri = settings.mongo_uri.get_secret_value()
    # Open with a conservative pool first; we re-open with the auto-tuned
    # pool size after calibration. Motor allows multiple clients per-process
    # so this is just a short-lived probe client.
    async with MongoClient(
        uri=mongo_uri,
        database=DATABASE,
        max_pool=2,
        cost_tracker=tracker,
    ) as probe_client:
        probe_start, probe_end = buckets[-1]

        async def _probe() -> int:
            opts = FindOptions(
                filter={"created_at": {"$gte": probe_start, "$lt": probe_end}},
                projection=PROJECTION,
                cost_key=COST_KEY,
                batch_size=BATCH_SIZE,
            )
            docs = await probe_client.find(COLLECTION, opts)
            return len(docs)

        cal = await calibrate(
            _probe,
            total_items=TOTAL_ITEMS_EST,
            estimated_rate=ESTIMATED_RATE,
            max_deviation=3.0,
            wall_clock_gate_min=WALL_CLOCK_GATE_MIN,
        )

    chosen_concurrency = auto_concurrency(
        total_items=TOTAL_ITEMS_EST,
        measured_rate=cal.measured_rate,
        target_minutes=WALL_CLOCK_TARGET_MIN,
        max_concurrency=MAX_CONCURRENCY,
        min_concurrency=MIN_CONCURRENCY,
    )
    mongo_pool = chosen_concurrency + 2
    logger.info(
        "auto_concurrency -> %d (cap=%d, target=%.1f min, "
        "measured=%.1f items/s); motor pool=%d",
        chosen_concurrency,
        MAX_CONCURRENCY,
        WALL_CLOCK_TARGET_MIN,
        cal.measured_rate,
        mongo_pool,
    )

    async with MongoClient(
        uri=mongo_uri,
        database=DATABASE,
        max_pool=mongo_pool,
        cost_tracker=tracker,
    ) as client:

        async def process_chunk(
            items: list[tuple[datetime, datetime]],
            chunk_idx: int,
        ) -> pl.DataFrame:
            start, end = items[0]
            return await _fetch_month(client, start, end, chunk_idx)

        pipeline: CheckpointedPipeline[tuple[datetime, datetime], pl.DataFrame] = (
            CheckpointedPipeline(
                config=CheckpointConfig(
                    name="civic_shout_news_environment_news_stories_v1",
                    checkpoint_dir=CHECKPOINT_DIR,
                    chunk_size=1,
                    concurrency=chosen_concurrency,
                ),
                process_chunk=process_chunk,
                save_chunk=_save_chunk,
                load_chunk=_load_chunk,
                combine=_combine,
            )
        )

        stories_df = await pipeline.run(buckets)

    logger.info("Combined stories shape: %s", stories_df.shape)
    if stories_df.is_empty():
        logger.error("No stories fetched — aborting cache write")
        return

    stories_df = stories_df.filter(pl.col("story_id").is_not_null())
    logger.info("After dropping null story_id: %d rows", len(stories_df))

    enriched = _join_news_sources(stories_df)
    logger.info("Enriched shape: %s", enriched.shape)

    cache.put(1, enriched)
    total_cost = tracker.get(COST_KEY).total
    logger.info("=" * 60)
    logger.info("news_stories v1 build complete")
    logger.info("  Final rows: %d", len(enriched))
    logger.info("  Cost: $%.4f", total_cost)
    logger.info("  Budget remaining: $%.2f", budget.remaining())
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(create())
