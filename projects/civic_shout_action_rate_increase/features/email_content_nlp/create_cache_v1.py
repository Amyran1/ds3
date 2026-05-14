"""Build email_content_nlp v1 feature cache.

Per-email NLP features extracted from civic_shout_emails v1 body_plaintext:
length, urgency cues, CTA cues, exclamation / question density, uppercase
ratio, and a first-line proxy for subject. Pure Polars string ops — no LLM,
no embeddings. Output schema: email_id + 11 features.

Local-only write by default (skip_s3=True). Smoke autoloop sessions may run
without S3 credentials; set skip_s3=False for a canonical full build.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from entities.civic_shout_emails.cache import cache as emails_cache
from projects.civic_shout_action_rate_increase.features.email_content_nlp.cache import (
    cache as feature_cache,
)

logger = logging.getLogger(__name__)

_TIMING_JSONL = Path(
    "projects/civic_shout_action_rate_increase/features/email_content_nlp/timing_performance.jsonl"
)

_URGENCY_PATTERN = (
    r"(?i)\b(urgent|today|now|last chance|deadline|hurry|immediate|"
    r"ending|expires|don'?t miss|act fast|final)\b"
)
_CTA_PATTERN = r"(?i)\b(sign|click|donate|act|join|support|vote|contact|share|tell|stand)\b"
_SENTENCE_PATTERN = r"[.!?]+"
_UPPERCASE_WORD_PATTERN = r"\b[A-Z]{3,}\b"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/Users/aaronmyran/dev/ds3",
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_v1(emails_df: pl.DataFrame) -> pl.DataFrame:
    """Compute 11 per-email NLP features from civic_shout_emails v1."""
    body = pl.col("body_plaintext").fill_null("")
    first_line = body.str.split("\n").list.get(0, null_on_oob=True).fill_null("")

    df = emails_df.with_columns(
        body.str.len_chars().cast(pl.UInt32).alias("body_char_len"),
        body.str.count_matches(r"\S+").cast(pl.UInt32).alias("body_word_count"),
        body.str.count_matches(_SENTENCE_PATTERN).cast(pl.UInt32).alias("body_sentence_count"),
        body.str.count_matches(r"!").cast(pl.UInt32).alias("body_exclamation_count"),
        body.str.count_matches(r"\?").cast(pl.UInt32).alias("body_question_count"),
        body.str.count_matches(_UPPERCASE_WORD_PATTERN)
        .cast(pl.UInt32)
        .alias("_uppercase_word_count"),
        body.str.count_matches(_URGENCY_PATTERN)
        .cast(pl.UInt32)
        .alias("body_urgency_keyword_count"),
        body.str.count_matches(_CTA_PATTERN).cast(pl.UInt32).alias("body_cta_keyword_count"),
        first_line.str.len_chars().cast(pl.UInt32).alias("first_line_char_len"),
        first_line.str.count_matches(r"\S+").cast(pl.UInt32).alias("first_line_word_count"),
    )

    df = df.with_columns(
        (
            pl.col("body_char_len").cast(pl.Float32)
            / pl.max_horizontal(pl.col("body_word_count").cast(pl.Float32), pl.lit(1.0))
        ).alias("body_avg_word_len_chars"),
        (
            pl.col("_uppercase_word_count").cast(pl.Float32)
            / pl.max_horizontal(pl.col("body_word_count").cast(pl.Float32), pl.lit(1.0))
        ).alias("body_uppercase_word_ratio"),
    )

    return df.select(
        [
            "email_id",
            "body_char_len",
            "body_word_count",
            "body_sentence_count",
            "body_avg_word_len_chars",
            "body_exclamation_count",
            "body_question_count",
            "body_uppercase_word_ratio",
            "body_urgency_keyword_count",
            "body_cta_keyword_count",
            "first_line_char_len",
            "first_line_word_count",
        ]
    )


def main(skip_s3: bool = True) -> None:
    git_sha = _git_sha()
    t0 = time.time()

    logger.info("Loading civic_shout_emails v1 ...")
    emails_df = emails_cache.get(1)
    n_input = emails_df.height
    logger.info("Loaded %d email rows", n_input)

    logger.info("Building email_content_nlp v1 features ...")
    features_df = build_v1(emails_df)
    n_output = features_df.height
    logger.info("Feature frame: %d rows, %d columns", n_output, len(features_df.columns))

    if n_input != n_output:
        raise RuntimeError(
            f"Row count mismatch: input={n_input}, output={n_output}. Build is invalid."
        )

    meta = feature_cache.versions[1]
    if skip_s3:
        # Direct local write — avoids S3 round-trip in sandboxed autoloop sessions.
        # cache.get(1) reads the local path first and only fetches S3 on local miss.
        meta.local_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.write_parquet(meta.local_path)
        logger.info("Wrote local cache to %s (S3 upload skipped)", meta.local_path)
    else:
        feature_cache.put(1, features_df)
        logger.info("Persisted v1 cache to local + S3")

    wall = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    timing_row = {
        "feature_family": "email_content_nlp",
        "version": 1,
        "tier": "full",
        "rows": n_input,
        "wall_seconds": round(wall, 2),
        "ts": ts,
        "git": git_sha,
        "skip_s3": skip_s3,
    }
    _TIMING_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_JSONL.open("a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    logger.info("email_content_nlp v1: rows=%d wall=%.2fs", n_input, wall)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-s3", action="store_true", help="Also upload to S3")
    args = parser.parse_args()
    main(skip_s3=not args.upload_s3)
