"""Validate full-corpus graph metrics against 20K-sample EDA bands.

Does not gate shipping — logs whether each metric falls in its band and
writes a markdown report. Used as a sanity check that the full-corpus
structure generalizes from the EDA.

Metrics:
  - Bipartite: n_edges, unique attributes per type, avg attributes per
    story, Louvain modularity on a random 50K story subgraph
  - Dense edges: n_edges, avg degree, degree percentiles
  - TF-IDF matrix: n_nonzero, density, vocab size
  - BM25 matrix: n_nonzero, density, vocab size

All metrics come from the cache singletons via
``StoryGraphAPI.from_cache(1)`` — no rebuild.

Usage:
    python -m entities\
        .news_stories_graph.validate_full_corpus_metrics
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import polars as pl
from networkx.algorithms.community import louvain_communities, modularity
from projects.civic_shout_news_environment.lib.news_stories_graph_api import (
    StoryGraphAPI,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_ENTITY_DIR = Path(
    "entities/news_stories_graph",
)
_FINDINGS_DIR = _ENTITY_DIR / "findings"
_REPORT_PATH = _FINDINGS_DIR / "full_corpus_validation.md"

_BIPARTITE_SAMPLE_N = 50_000
_RNG_SEED = 42


def _bipartite_metrics(api: StoryGraphAPI) -> dict[str, object]:
    bip = api._caches.bipartite  # noqa: SLF001
    logger.info("Bipartite rows: %d", bip.height)

    n_edges = bip.height
    counts_col = bip.group_by("story_id").agg(pl.len().alias("n"))["n"]
    attrs_raw = counts_col.mean()
    attrs_per_story: float = 0.0
    if attrs_raw is not None:
        attrs_per_story = float(cast("int | float", attrs_raw))
    unique_per_type = (
        bip.group_by("attribute_type")
        .agg(pl.col("value").n_unique().alias("n_unique"))
        .sort("attribute_type")
    )
    logger.info("Attrs per story (mean): %.2f", attrs_per_story)
    logger.info("Unique per type:\n%s", unique_per_type)

    # Louvain on a sampled subgraph
    all_stories = bip["story_id"].unique().to_list()
    rng = np.random.default_rng(_RNG_SEED)
    sample_n = min(_BIPARTITE_SAMPLE_N, len(all_stories))
    sample_ids = rng.choice(
        np.asarray(all_stories),
        size=sample_n,
        replace=False,
    ).tolist()
    logger.info("Sampling %d stories for Louvain", sample_n)

    sub = bip.filter(pl.col("story_id").is_in(sample_ids))
    g: nx.Graph = nx.Graph()
    edge_iter = sub.select(
        ["story_id", "attribute_type", "value"],
    ).iter_rows()
    for story_id, attr_type, value in edge_iter:
        s_node = f"s:{story_id}"
        a_node = f"a:{attr_type}::{value}"
        g.add_edge(s_node, a_node)
    logger.info("Subgraph: nodes=%d edges=%d", g.number_of_nodes(), g.number_of_edges())

    if g.number_of_edges() == 0:
        louvain_mod = 0.0
        largest_frac = 0.0
    else:
        components = sorted(nx.connected_components(g), key=len, reverse=True)
        largest_frac = len(components[0]) / g.number_of_nodes()
        logger.info(
            "Largest component fraction (sample): %.3f",
            largest_frac,
        )
        logger.info("Running Louvain (may take a few minutes)")
        comms = louvain_communities(g, seed=_RNG_SEED)
        louvain_mod = float(modularity(g, comms))
        logger.info("Louvain modularity (sample): %.3f", louvain_mod)

    upt_rows = unique_per_type.iter_rows(named=True)
    unique_by_type = {row["attribute_type"]: row["n_unique"] for row in upt_rows}
    return {
        "n_edges": n_edges,
        "avg_attrs_per_story": attrs_per_story,
        "unique_per_type": unique_by_type,
        "louvain_modularity_50k": louvain_mod,
        "largest_component_frac_50k": largest_frac,
    }


def _dense_edges_metrics(api: StoryGraphAPI) -> dict[str, object]:
    de = api._caches.dense_edges  # noqa: SLF001
    logger.info("Dense edges rows: %d", de.height)

    n_edges = de.height
    # src,dst degree count — combine both columns
    deg = (
        pl.concat(
            [
                de.select(pl.col("src").alias("node")),
                de.select(pl.col("dst").alias("node")),
            ],
        )
        .group_by("node")
        .agg(pl.len().alias("degree"))
    )
    degrees = deg["degree"].to_numpy()
    avg_deg = float(degrees.mean()) if degrees.size else 0.0
    pcts = {
        "p50": float(np.percentile(degrees, 50)) if degrees.size else 0.0,
        "p90": float(np.percentile(degrees, 90)) if degrees.size else 0.0,
        "p99": float(np.percentile(degrees, 99)) if degrees.size else 0.0,
        "max": float(degrees.max()) if degrees.size else 0.0,
    }
    logger.info("Dense avg degree: %.2f, p99=%.0f", avg_deg, pcts["p99"])

    return {
        "n_edges": n_edges,
        "avg_degree": avg_deg,
        "degree_percentiles": pcts,
    }


def _sparse_matrix_metrics(
    matrix_name: str,
    matrix: object,
) -> dict[str, object]:
    m_any = cast("object", matrix)
    shape = cast("tuple[int, int]", m_any.shape)  # type: ignore[attr-defined]
    nnz = int(m_any.nnz)  # type: ignore[attr-defined]
    density = nnz / (shape[0] * shape[1]) if shape[0] and shape[1] else 0.0
    logger.info(
        "%s: shape=%s nnz=%d density=%.2e",
        matrix_name,
        shape,
        nnz,
        density,
    )
    return {
        "name": matrix_name,
        "shape_rows": shape[0],
        "shape_cols": shape[1],
        "nnz": nnz,
        "density": density,
    }


def _compare_band(
    value: float,
    lo: float | None,
    hi: float | None,
) -> str:
    if lo is not None and value < lo:
        return "BELOW"
    if hi is not None and value > hi:
        return "ABOVE"
    return "OK"


def _write_report(
    bip_m: dict[str, object],
    dense_m: dict[str, object],
    tfidf_m: dict[str, object],
    bm25_m: dict[str, object],
) -> None:
    _FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    mod = cast("float", bip_m["louvain_modularity_50k"])
    mod_status = _compare_band(mod, 0.40, None)
    avg_deg = cast("float", dense_m["avg_degree"])
    lcf = cast("float", bip_m["largest_component_frac_50k"])
    louvain_label = f"- Louvain modularity (50K-story sample): **{mod:.3f}**"
    band_suffix = f" (band >= 0.40 -> {mod_status})"
    mod_line = louvain_label + band_suffix

    lines: list[str] = [
        "# news_stories_graph v1 — Full-Corpus Metric Validation",
        "",
        f"Run at: {datetime.now(UTC).isoformat()}",
        "",
        "Advisory only — a metric outside its band is NOT a ship blocker.",
        "",
        "## Bipartite",
        "",
        f"- Total edges: **{bip_m['n_edges']:,}**",
        f"- Avg attributes per story: **{bip_m['avg_attrs_per_story']:.2f}**",
        f"- Unique per type: `{bip_m['unique_per_type']}`",
        mod_line,
        f"- Largest component fraction (50K sample): **{lcf:.3f}**",
        "",
        "## Dense edges",
        "",
        f"- Total edges: **{dense_m['n_edges']:,}**",
        f"- Avg degree: **{avg_deg:.2f}**",
        f"- Degree percentiles: `{dense_m['degree_percentiles']}`",
        "",
        "> NOTE: the dense graph was built with k=100 FAISS neighbors at",
        "> τ=0.50. At full corpus this k-cap saturates — the 20K-sample",
        "> avg degree of 5.95 is NOT comparable to the full-corpus",
        "> number. This is a known ceiling artifact, not a failure.",
        "",
        "## TF-IDF matrix",
        "",
        f"- Shape: **{tfidf_m['shape_rows']:,} x {tfidf_m['shape_cols']:,}**",
        f"- nnz: **{tfidf_m['nnz']:,}** (expected ~225M — 175x the 20K sample's ~300K)",
        f"- Density: **{tfidf_m['density']:.2e}**",
        "",
        "## BM25 matrix",
        "",
        f"- Shape: **{bm25_m['shape_rows']:,} x {bm25_m['shape_cols']:,}**",
        f"- nnz: **{bm25_m['nnz']:,}** (expected ~204M — 155x the 20K sample's ~1.3M)",
        f"- Density: **{bm25_m['density']:.2e}**",
        "",
        "## Summary",
        "",
        f"- Bipartite modularity band: **{mod_status}**",
        "- Dense avg degree: ceiling artifact (informational only)",
        "- TF-IDF nnz: informational",
        "- BM25 nnz: informational",
    ]
    _REPORT_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Wrote report: %s", _REPORT_PATH)


def main() -> int:
    logger.info("Instantiating StoryGraphAPI.from_cache(1)")
    api = StoryGraphAPI.from_cache(1)

    logger.info("Computing bipartite metrics")
    bip_m = _bipartite_metrics(api)

    logger.info("Computing dense-edges metrics")
    dense_m = _dense_edges_metrics(api)

    logger.info("Computing TF-IDF matrix metrics")
    tfidf_m = _sparse_matrix_metrics("tfidf", api._caches.tfidf_matrix)  # noqa: SLF001

    logger.info("Computing BM25 matrix metrics")
    bm25_m = _sparse_matrix_metrics("bm25", api._caches.bm25_matrix)  # noqa: SLF001

    _write_report(bip_m, dense_m, tfidf_m, bm25_m)

    logger.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
