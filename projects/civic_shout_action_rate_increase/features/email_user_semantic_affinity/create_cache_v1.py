"""Build email_user_semantic_affinity v1 feature cache.

Per (user_id, email_id, date_sent) row: cosine similarity between the
current email's 1536-d dense_vector and the running sum of the same user's
prior actioned-email vectors. Strict causality: a row's centroid uses only
rows whose date_sent is strictly less than the current row's date_sent.

Cache scope (smoke-tier, autoloop iter 3 budget):
    Builds for the same rows the iter-3 smoke run evaluates, by mirroring
    sample(fraction=0.001, seed=42) on the date-filtered user_emails frame.
    The per-user numpy loop runs over those users' full prior history (so
    affinity is computed correctly even for rows whose history was not
    sampled), but the cache only retains the sampled rows.

Output columns:
    user_id (i64), email_id (i64), date_sent (datetime[us, UTC]),
    semantic_affinity_to_prior_actions (f32),
    n_prior_actions_for_affinity (u32)

Usage:
    python -m projects.civic_shout_action_rate_increase.features.email_user_semantic_affinity.create_cache_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from entities.civic_shout_emails.cache import cache as emails_cache
from entities.civic_shout_user_emails.cache import cache as user_emails_cache
from projects.civic_shout_action_rate_increase.features.email_user_semantic_affinity.cache import (
    cache as feature_cache,
)

logger = logging.getLogger(__name__)

_TIMING_JSONL = Path(
    "projects/civic_shout_action_rate_increase/features/email_user_semantic_affinity"
    "/timing_performance.jsonl"
)

SAMPLE_FRAC = 0.001
SAMPLE_SEED = 42
EMBED_DIM = 1536


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


def _email_vec_matrix(emails_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Return (email_vecs[E,D], email_vec_norms[E], email_id_to_idx)."""
    email_ids = emails_df["email_id"].to_list()
    vec_lists = emails_df["dense_vector"].to_list()
    vecs = np.empty((len(email_ids), EMBED_DIM), dtype=np.float32)
    for i, v in enumerate(vec_lists):
        if v is None:
            vecs[i] = 0.0
        else:
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape[0] != EMBED_DIM:
                raise ValueError(
                    f"email_id={email_ids[i]} has dense_vector shape {arr.shape}; "
                    f"expected ({EMBED_DIM},)"
                )
            vecs[i] = arr
    norms = np.linalg.norm(vecs, axis=1).astype(np.float32)
    id_to_idx = {eid: i for i, eid in enumerate(email_ids)}
    return vecs, norms, id_to_idx


def _per_user_affinity(
    user_ids: np.ndarray,
    row_email_idx: np.ndarray,
    row_actioned: np.ndarray,
    email_vecs: np.ndarray,
    email_vec_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-user causal cosine-similarity-to-prior-actions pass.

    Arrays must be sorted by (user_id, date_sent). Returns (affinity[N],
    n_prior[N]) with affinity=0.0 when n_prior=0 (cold-start sentinel).
    """
    n = user_ids.shape[0]
    affinity = np.zeros(n, dtype=np.float32)
    n_prior = np.zeros(n, dtype=np.uint32)

    if n == 0:
        return affinity, n_prior

    # Group boundaries via diff on the sorted user_ids array.
    change_mask = np.empty(n, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = user_ids[1:] != user_ids[:-1]
    group_starts = np.flatnonzero(change_mask)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = n

    running = np.zeros(EMBED_DIM, dtype=np.float32)

    for g in range(group_starts.shape[0]):
        start = group_starts[g]
        end = group_ends[g]
        running.fill(0.0)
        action_count = 0
        for i in range(start, end):
            e_idx = row_email_idx[i]
            if action_count > 0:
                s_norm_sq = float(running @ running)
                if s_norm_sq > 1e-18:
                    dot_val = float(email_vecs[e_idx] @ running)
                    denom = email_vec_norms[e_idx] * np.sqrt(s_norm_sq)
                    if denom > 1e-12:
                        affinity[i] = dot_val / denom
                n_prior[i] = action_count
            if row_actioned[i]:
                running += email_vecs[e_idx]
                action_count += 1

    return affinity, n_prior


def build_v1(
    user_emails_history: pl.DataFrame,
    sample_rows: pl.DataFrame,
    emails_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute affinity for ``user_emails_history`` and restrict output to ``sample_rows``.

    ``user_emails_history`` MUST contain every row needed to causally derive
    affinity for the sampled rows (i.e., the full date-filtered history for
    every user appearing in ``sample_rows``). ``sample_rows`` is the
    sampled-subset whose feature values get persisted.
    """
    logger.info("Building email_id -> dense_vector lookup ...")
    email_vecs, email_vec_norms, id_to_idx = _email_vec_matrix(emails_df)
    logger.info(
        "Email vector matrix: %d emails x %d dims", email_vecs.shape[0], email_vecs.shape[1]
    )

    logger.info("Sorting history (%d rows) by (user_id, date_sent) ...", user_emails_history.height)
    history = user_emails_history.sort(["user_id", "date_sent"])

    user_ids = history["user_id"].to_numpy()
    email_ids = history["email_id"].to_numpy()
    actioned = history["actioned_24h"].to_numpy().astype(np.bool_)

    # Map email_ids to dense_vector row indices.
    logger.info("Mapping %d history rows -> email-vec indices ...", email_ids.shape[0])
    row_email_idx = np.empty(email_ids.shape[0], dtype=np.int32)
    miss = 0
    for i, eid in enumerate(email_ids):
        idx = id_to_idx.get(int(eid))
        if idx is None:
            row_email_idx[i] = -1
            miss += 1
        else:
            row_email_idx[i] = idx
    if miss > 0:
        # Defensive: any row with missing email vector cannot contribute; null it.
        logger.warning("%d history rows have no matching email in entity v1", miss)
        # Replace -1 indices with 0 and zero out their action contribution.
        actioned = actioned & (row_email_idx != -1)
        row_email_idx[row_email_idx == -1] = 0

    logger.info("Running per-user affinity pass over %d rows ...", user_ids.shape[0])
    affinity, n_prior = _per_user_affinity(
        user_ids=user_ids,
        row_email_idx=row_email_idx,
        row_actioned=actioned,
        email_vecs=email_vecs,
        email_vec_norms=email_vec_norms,
    )

    feature_df = history.select(["user_id", "email_id", "date_sent"]).with_columns(
        pl.Series("semantic_affinity_to_prior_actions", affinity, dtype=pl.Float32),
        pl.Series("n_prior_actions_for_affinity", n_prior, dtype=pl.UInt32),
    )

    # Restrict to the sampled (user_id, email_id, date_sent) tuples.
    output = sample_rows.select(["user_id", "email_id", "date_sent"]).join(
        feature_df, on=["user_id", "email_id", "date_sent"], how="inner"
    )

    if output.height != sample_rows.height:
        raise RuntimeError(
            f"Output row count {output.height} != sample row count {sample_rows.height}. "
            "Either history was incomplete or duplicate keys exist."
        )

    return output


def main(skip_s3: bool = True) -> None:
    git_sha = _git_sha()
    t0 = time.time()

    logger.info("Loading civic_shout_emails v1 ...")
    emails_df = emails_cache.get(1)
    logger.info("Loaded %d email rows", emails_df.height)

    logger.info("Loading civic_shout_user_emails v3 ...")
    user_emails_df = user_emails_cache.get(3)
    n_full = user_emails_df.height
    logger.info("Loaded %d user_emails rows", n_full)

    # Date filter + sample identical to run_03 to guarantee key alignment.
    max_date = user_emails_df["date_sent"].max()
    filtered = user_emails_df.filter(pl.col("date_sent") < (max_date - pl.duration(hours=24)))
    logger.info("After 24h-tail filter: %d rows", filtered.height)

    sample = filtered.sample(fraction=SAMPLE_FRAC, seed=SAMPLE_SEED, with_replacement=False)
    n_sample = sample.height
    logger.info("Sample (frac=%.4f seed=%d): %d rows", SAMPLE_FRAC, SAMPLE_SEED, n_sample)

    # History scope: ALL rows for users appearing in the sample, from the
    # unfiltered user_emails. We need pre-24h rows too so that the centroid
    # is accurate at the boundary; harness rows are restricted to filtered.
    sample_users = sample.select(pl.col("user_id").unique())
    history = user_emails_df.select(["user_id", "email_id", "date_sent", "actioned_24h"]).join(
        sample_users, on="user_id", how="inner"
    )
    n_history = history.height
    logger.info("History for sampled users: %d rows", n_history)

    feature_df = build_v1(history, sample, emails_df)
    n_output = feature_df.height
    logger.info("Feature frame: %d rows, %d columns", n_output, len(feature_df.columns))

    if n_output != n_sample:
        raise RuntimeError(f"Row count mismatch: sample={n_sample}, output={n_output}.")

    meta = feature_cache.versions[1]
    if skip_s3:
        meta.local_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.write_parquet(meta.local_path)
        logger.info("Wrote local cache to %s (S3 upload skipped)", meta.local_path)
    else:
        feature_cache.put(1, feature_df)
        logger.info("Persisted v1 cache to local + S3")

    wall = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    timing_row = {
        "feature_family": "email_user_semantic_affinity",
        "version": 1,
        "tier": f"smoke_sample_frac_{SAMPLE_FRAC}",
        "rows_sample": n_sample,
        "rows_history": n_history,
        "rows_full": n_full,
        "wall_seconds": round(wall, 2),
        "ts": ts,
        "git": git_sha,
        "skip_s3": skip_s3,
    }
    _TIMING_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_JSONL.open("a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    logger.info(
        "email_user_semantic_affinity v1: sample=%d history=%d wall=%.1fs",
        n_sample,
        n_history,
        wall,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-s3", action="store_true", help="Also upload to S3")
    args = parser.parse_args()
    main(skip_s3=not args.upload_s3)
