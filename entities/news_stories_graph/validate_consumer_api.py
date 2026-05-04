"""Sanity-check the StoryGraphAPI against an auto-generated eval set.

NOT a human-curated gold set — this is a v1 smoke/timing harness. A
human gold set is deferred to a v1.1 follow-up addendum. Writes
``findings/consumer_api_validation.md``.

Usage:
    python -m entities\
        .news_stories_graph.validate_consumer_api
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from entities.news_stories_graph.api import (
    StoryGraphAPI,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.sparse import csr_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_ENTITY_DIR = Path(
    "entities/news_stories_graph",
)
_FINDINGS_DIR = _ENTITY_DIR / "findings"
_REPORT_PATH = _FINDINGS_DIR / "consumer_api_validation.md"

_RNG_SEED = 42
_N_SELF_QUERIES = 20
_N_PER_METHOD_QUERIES = 5
_BATCH_M = 100
_BATCH_K = 10
_COST_KEY = "news-stories-graph-api-queries"
_MAX_COST_DOLLARS = 0.01
_MIN_ATTRIBUTE_DEGREE = 4
_BATCH_WALL_CLOCK_MAX_SEC = 60.0
_BATCH_RETURN_FRAC = 0.90


def _time_call(
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _self_query_sanity(api: StoryGraphAPI) -> dict[str, object]:
    dense_vectors = api._caches.dense_vectors  # noqa: SLF001
    story_ids = api._caches.dense_story_ids  # noqa: SLF001
    # Boost nprobe for the self-query sanity check (IVFFlat approximate search)
    index = api._caches.dense_index  # noqa: SLF001
    _any_idx = cast("Any", index)
    with contextlib.suppress(AttributeError):
        _any_idx.nprobe = 64
    rng = np.random.default_rng(_RNG_SEED)
    sample_idx = rng.choice(len(story_ids), size=_N_SELF_QUERIES, replace=False)

    hits_found = 0
    total_returned = 0
    timings: list[float] = []
    for i in sample_idx:
        q = dense_vectors[int(i)]
        out, dt = _time_call(
            api.related_stories_dense,
            q,
            k=5,
            min_similarity=0.30,
        )
        df = cast("pl.DataFrame", out)
        timings.append(dt)
        total_returned += df.height
        if df.height > 0 and story_ids[int(i)] in df["story_id"].to_list():
            hits_found += 1

    return {
        "method": "dense self-query",
        "n": _N_SELF_QUERIES,
        "wall_sec": sum(timings),
        "p50_sec": float(np.median(timings)),
        "hits_found": hits_found,
        "avg_returned": total_returned / max(_N_SELF_QUERIES, 1),
        "pass": total_returned >= _N_SELF_QUERIES,
    }


def _run_query_batch(
    method_name: str,
    queries: list[object],
    call_fn: Callable[[object], pl.DataFrame],
) -> dict[str, object]:
    """Run ``call_fn(q)`` for each query, collect timing + hit metrics."""
    nonempty = 0
    top1_sims: list[float] = []
    timings: list[float] = []
    for q in queries:
        out, dt = _time_call(call_fn, q)
        df = cast("pl.DataFrame", out)
        timings.append(dt)
        if df.height > 0:
            nonempty += 1
            top1_sims.append(float(df["similarity"][0]))
    return {
        "method": method_name,
        "n": len(queries),
        "wall_sec": sum(timings),
        "p50_sec": float(np.median(timings)) if timings else 0.0,
        "nonempty": nonempty,
        "top1_mean_sim": float(np.mean(top1_sims)) if top1_sims else 0.0,
        "pass": nonempty == len(queries),
    }


def _per_method_checks(api: StoryGraphAPI) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(_RNG_SEED + 1)

    dense_vectors = api._caches.dense_vectors  # noqa: SLF001
    d_idxs = rng.choice(len(dense_vectors), size=_N_PER_METHOD_QUERIES, replace=False)
    d_queries: list[object] = [dense_vectors[int(i)] for i in d_idxs]
    rows.append(
        _run_query_batch(
            "related_stories_dense",
            d_queries,
            lambda q: api.related_stories_dense(
                cast("np.ndarray", q),
                k=5,
                min_similarity=-1.0,
            ),
        ),
    )

    tfidf_m = api._caches.tfidf_matrix  # noqa: SLF001
    tfidf_shape = cast("tuple[int, int]", tfidf_m.shape)
    t_idxs = rng.choice(tfidf_shape[0], size=_N_PER_METHOD_QUERIES, replace=False)
    t_queries: list[object] = [cast("csr_matrix", tfidf_m[int(i)]) for i in t_idxs]
    rows.append(
        _run_query_batch(
            "related_stories_tfidf",
            t_queries,
            lambda q: api.related_stories_tfidf(
                cast("csr_matrix", q),
                k=5,
                min_similarity=0.0,
            ),
        ),
    )

    bm25_m = api._caches.bm25_matrix  # noqa: SLF001
    bm25_shape = cast("tuple[int, int]", bm25_m.shape)
    b_idxs = rng.choice(bm25_shape[0], size=_N_PER_METHOD_QUERIES, replace=False)
    b_queries: list[object] = [cast("csr_matrix", bm25_m[int(i)]) for i in b_idxs]
    rows.append(
        _run_query_batch(
            "related_stories_bm25",
            b_queries,
            lambda q: api.related_stories_bm25(
                cast("csr_matrix", q),
                k=5,
                min_similarity=0.0,
            ),
        ),
    )

    bip = api._caches.bipartite  # noqa: SLF001
    anchor_candidates = (
        bip.group_by("story_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= _MIN_ATTRIBUTE_DEGREE)
        .head(_N_PER_METHOD_QUERIES)
    )
    a_queries: list[object] = [
        str(row["story_id"]) for row in anchor_candidates.iter_rows(named=True)
    ]
    rows.append(
        _run_query_batch(
            "related_stories_by_attribute",
            a_queries,
            lambda q: api.related_stories_by_attribute(
                cast("str", q),
                min_shared=2,
            ),
        ),
    )

    return rows


def _batch_check(api: StoryGraphAPI) -> dict[str, object]:
    tfidf_m = api._caches.tfidf_matrix  # noqa: SLF001
    rng = np.random.default_rng(_RNG_SEED + 2)
    tfidf_shape = cast("tuple[int, int]", tfidf_m.shape)
    idxs = rng.choice(tfidf_shape[0], size=_BATCH_M, replace=False)
    query_matrix = cast("csr_matrix", tfidf_m[idxs])

    t0 = time.perf_counter()
    df = api.batch_related_stories_sparse(
        query_matrix,
        method="tfidf",
        k=_BATCH_K,
        min_similarity=0.0,
    )
    dt = time.perf_counter() - t0

    expected_rows = _BATCH_M * _BATCH_K
    enough_rows = df.height >= expected_rows * _BATCH_RETURN_FRAC
    fast_enough = dt < _BATCH_WALL_CLOCK_MAX_SEC
    return {
        "method": "batch_related_stories_sparse",
        "n": _BATCH_M,
        "wall_sec": dt,
        "p50_sec": dt / max(_BATCH_M, 1),
        "nonempty": df.height,
        "top1_mean_sim": 0.0,
        "pass": enough_rows and fast_enough,
    }


def _write_report(
    rows: list[dict[str, object]],
    total_cost: float,
) -> None:
    _FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# news_stories_graph v1 — Consumer API Validation",
        "",
        f"Run at: {datetime.now(UTC).isoformat()}",
        "",
        "This is an **auto-generated eval set** for smoke + timing checks.",
        "A human-curated gold set (20 hand-labeled email->story pairs) is",
        "deferred to a v1.1 follow-up — see Addendum 3 'Validation Section'.",
        "",
        f"Total OpenAI cost incurred: **${total_cost:.4f}** "
        f"(cost_key=`{_COST_KEY}`, gate < ${_MAX_COST_DOLLARS:.2f})",
        "",
        "| Method | n | wall (s) | p50 (s) | nonempty/hits | top1 sim | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    t_a = "| {method} | {n} | {wall:.2f} | {p50:.3f}"
    t_b = " | {hits} | {sim:.3f} | {p} |"
    row_template = t_a + t_b
    for r in rows:
        hits_field = r.get("hits_found", r.get("nonempty", 0))
        lines.append(
            row_template.format(
                method=r["method"],
                n=r["n"],
                wall=cast("float", r["wall_sec"]),
                p50=cast("float", r["p50_sec"]),
                hits=hits_field,
                sim=cast("float", r.get("top1_mean_sim", 0.0)),
                p="PASS" if r["pass"] else "FAIL",
            ),
        )
    _REPORT_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Wrote report: %s", _REPORT_PATH)


def main() -> int:
    logger.info("Instantiating StoryGraphAPI.from_cache(1)")
    api = StoryGraphAPI.from_cache(1)

    rows: list[dict[str, object]] = []
    logger.info("Running dense self-query sanity check")
    rows.append(_self_query_sanity(api))

    logger.info("Running per-method checks")
    rows.extend(_per_method_checks(api))

    logger.info("Running batch_related_stories_sparse check")
    rows.append(_batch_check(api))

    _write_report(rows, 0.0)

    any_fail = any(not r["pass"] for r in rows)
    for r in rows:
        logger.info(
            "  %-35s pass=%s wall=%.2fs",
            r["method"],
            r["pass"],
            cast("float", r["wall_sec"]),
        )
    if any_fail:
        logger.error("FAIL: one or more checks failed")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
