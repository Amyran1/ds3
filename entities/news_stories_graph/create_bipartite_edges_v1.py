"""Build bipartite (story, attribute_type, value) edges v1 for news_stories_graph.

Reads the full ``news_stories`` v1 cache, explodes the list-typed attribute
columns into long-form, applies the (FLOOR=2, CAP=5000) eligibility filter,
and writes the result via the ``bipartite_edges`` EntityCache (local parquet
plus S3 upload).

The ``locations`` attribute column is excluded because it was 0% populated
in news_stories v1 (per the news_stories_graph EDA Q1 finding).

Usage::

    python -m entities.news_stories_graph.create_bipartite_edges_v1
"""

from __future__ import annotations

import logging

import polars as pl

from entities.news_stories.cache import (
    cache as news_stories_cache,
)
from entities.news_stories_graph.bipartite_edges_cache import (
    cache as bipartite_cache,
)

logger = logging.getLogger(__name__)

ATTRIBUTE_COLS: tuple[str, ...] = (
    "topics",
    "people",
    "companies",
    "countries",
    "categories",
)
# locations excluded: 0% populated in news_stories v1 (per EDA Q1 finding)

ATTRIBUTE_FREQ_FLOOR = 2
ATTRIBUTE_FREQ_CAP = 5_000


def _explode_attribute(stories: pl.DataFrame, col: str) -> pl.DataFrame:
    """Explode one list-typed attribute column to ``(story_id, value)`` long-form.

    Handles both ``list[struct]`` (preferring the ``name`` field, with
    fallbacks) and ``list[str]`` inner types. Rows with null or empty
    string values are dropped. Values are cast to Utf8.
    """
    if col not in stories.columns:
        msg = f"Column {col!r} not present in stories DataFrame"
        raise KeyError(msg)

    dtype = stories.schema[col]
    inner = getattr(dtype, "inner", None)

    selected = stories.select(["story_id", col]).filter(pl.col(col).is_not_null())
    exploded = selected.explode(col)

    if isinstance(inner, pl.Struct):
        struct_fields = [f.name for f in inner.fields]
        label_field = next(
            (f for f in ("name", "label", "value", "text", "title") if f in struct_fields),
            struct_fields[0] if struct_fields else None,
        )
        if label_field is None:
            msg = f"Column {col!r} has struct inner type with no fields"
            raise ValueError(msg)
        long = exploded.with_columns(
            pl.col(col).struct.field(label_field).alias("value"),
        ).select(["story_id", "value"])
    else:
        long = exploded.rename({col: "value"}).select(["story_id", "value"])

    return (
        long.with_columns(pl.col("value").cast(pl.Utf8, strict=False))
        .filter(pl.col("value").is_not_null())
        .filter(pl.col("value").str.len_chars() > 0)
    )


def _all_attributes_long(stories: pl.DataFrame) -> pl.DataFrame:
    """Concatenate all five exploded attribute columns into one long DataFrame.

    Columns: ``story_id`` (Utf8), ``attribute_type`` (Utf8), ``value`` (Utf8).
    """
    frames: list[pl.DataFrame] = []
    for col in ATTRIBUTE_COLS:
        if col not in stories.columns:
            logger.warning("Attribute column %r not present in stories; skipping", col)
            continue
        exploded = _explode_attribute(stories, col).with_columns(
            pl.lit(col).alias("attribute_type"),
        )
        frames.append(
            exploded.select(
                pl.col("story_id").cast(pl.Utf8),
                pl.col("attribute_type").cast(pl.Utf8),
                pl.col("value").cast(pl.Utf8),
            ),
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "story_id": pl.Utf8,
                "attribute_type": pl.Utf8,
                "value": pl.Utf8,
            },
        )
    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    logger.info("Loading news_stories v1")
    stories = news_stories_cache.get(1)
    logger.info("Loaded %d stories", len(stories))

    logger.info("Exploding %d attribute columns", len(ATTRIBUTE_COLS))
    long = _all_attributes_long(stories)
    logger.info("Raw long-form rows: %d", len(long))

    freq = long.group_by(["attribute_type", "value"]).agg(
        pl.col("story_id").n_unique().alias("story_count"),
    )
    eligible = freq.filter(
        (pl.col("story_count") >= ATTRIBUTE_FREQ_FLOOR)
        & (pl.col("story_count") <= ATTRIBUTE_FREQ_CAP),
    )
    logger.info(
        "Eligible (attribute_type, value) pairs: %d / %d",
        len(eligible),
        len(freq),
    )

    filtered = (
        long.join(
            eligible.select(["attribute_type", "value"]),
            on=["attribute_type", "value"],
            how="inner",
        )
        .select(["story_id", "attribute_type", "value"])
        .unique()
    )
    logger.info(
        "Filtered long-form rows: %d (%.1f%% of raw)",
        len(filtered),
        100 * len(filtered) / max(len(long), 1),
    )

    per_type = (
        filtered.group_by("attribute_type")
        .agg(pl.len().alias("n_rows"))
        .sort("n_rows", descending=True)
    )
    logger.info("Per-attribute-type row counts:")
    for row in per_type.iter_rows(named=True):
        logger.info("  %s: %d", row["attribute_type"], row["n_rows"])

    logger.info("Writing bipartite_edges_v1 via EntityCache")
    bipartite_cache.put(1, filtered)
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
