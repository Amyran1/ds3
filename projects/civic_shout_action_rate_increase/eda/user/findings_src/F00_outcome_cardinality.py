"""F00 Outcome cardinality check — real EDA (not mockup).

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F00_outcome_cardinality
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from entities.civic_shout_engagement.attributed_actions.cache import cache as aa_cache
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_GIT_SHA = "9083370"


def _compute_distribution(aa: pl.DataFrame) -> pl.DataFrame:
    per_pair = (
        aa.filter(pl.col("is_attributed"))
        .group_by(["user_id", "email_id"])
        .agg(pl.len().alias("n_attributed_actions"))
    )
    dist = (
        per_pair.group_by("n_attributed_actions")
        .agg(pl.len().alias("pair_count"))
        .sort("n_attributed_actions")
    )
    return per_pair, dist


def _chart(dist: pl.DataFrame, n_attributed_max: int) -> bytes:
    n_vals = dist["n_attributed_actions"].to_list()
    counts = dist["pair_count"].to_list()

    xs = list(range(1, n_attributed_max + 1))
    count_map = dict(zip(n_vals, counts))
    ys = [count_map.get(x, 0) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(xs, ys, color="#2956a3")
    ax.set_yscale("log")
    ax.set_xlabel("attributed actions per (user, email) pair")
    ax.set_ylabel("count of pairs (log)")
    ax.set_title("Distribution of n_attributed_actions | is_attributed=True")
    ax.set_xticks(xs)

    for i, (x, c) in enumerate(zip(xs, ys)):
        if c == 0:
            continue
        label = f"{c / 1e6:.2f}M" if c >= 1_000_000 else f"{c:,}"
        ax.text(x, c * 1.15, label, ha="center", fontsize=7)

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _decide_target(pct_pairs_ge2: float, pct_sigs_lost: float) -> str:
    if pct_pairs_ge2 < 5.0 and pct_sigs_lost < 10.0:
        return "BINARY"
    if pct_pairs_ge2 < 15.0 and pct_sigs_lost < 20.0:
        return "HYBRID"
    return "COUNT"


def _impact_md(
    pct_pairs_ge2: float,
    pct_sigs_lost: float,
    recommendation: str,
) -> str:
    if recommendation == "BINARY":
        target_text = (
            "**Recommendation: keep the binary target.**\n\n"
            f"Only {pct_pairs_ge2:.1f}% of attributed (user, email) pairs have ≥2 signatures, "
            f"and boolean collapse loses only {pct_sigs_lost:.1f}% of total signatures. "
            "The binary signal is cleaner, matches the dominant case (one signature per pair), "
            "and simplifies the harness. Poisson / hybrid models would add complexity without "
            "materially recovering lost signal."
        )
    elif recommendation == "HYBRID":
        target_text = (
            "**Recommendation: hybrid target (probability × conditional petition count).**\n\n"
            f"{pct_pairs_ge2:.1f}% of attributed (user, email) pairs have ≥2 signatures, "
            f"and boolean collapse loses {pct_sigs_lost:.1f}% of total signatures — "
            "too much to ignore but not dominant enough to switch entirely to count regression. "
            "A two-stage model (action probability × Poisson conditional count) or a "
            "zero-inflated Poisson is the recommended path."
        )
    else:
        target_text = (
            "**Recommendation: expected petition count (Poisson regression).**\n\n"
            f"{pct_pairs_ge2:.1f}% of attributed (user, email) pairs have ≥2 signatures, "
            f"and boolean collapse loses {pct_sigs_lost:.1f}% of total signatures. "
            "The count signal is too large to ignore; a binary target discards material information. "
            "Use Poisson regression or negative binomial with a count target (n_attributed_actions per pair)."
        )

    return (
        f"{target_text}\n\n"
        "Wave 3 features (F06, F07, F10) cannot ship until this decision is made — "
        "they target a single response variable. The harness target must be pinned "
        "before any feature evaluation begins."
    )


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    aa = aa_cache.get(2)

    per_pair, dist = _compute_distribution(aa)

    n_pairs = per_pair.height
    total_sigs = per_pair["n_attributed_actions"].sum()
    n_ge2 = per_pair.filter(pl.col("n_attributed_actions") >= 2).height
    sigs_lost = total_sigs - n_pairs

    pct_pairs_ge2 = 100.0 * n_ge2 / n_pairs if n_pairs > 0 else 0.0
    pct_sigs_lost = 100.0 * sigs_lost / total_sigs if total_sigs > 0 else 0.0

    n_attributed_max = min(dist["n_attributed_actions"].max(), 20)

    chart_bytes = _chart(dist, n_attributed_max)

    recommendation = _decide_target(pct_pairs_ge2, pct_sigs_lost)

    rec_label = {
        "BINARY": "BINARY target",
        "HYBRID": "HYBRID target",
        "COUNT": "COUNT (Poisson) target",
    }[recommendation]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tldr = (
        f"{pct_pairs_ge2:.1f}% of attributed pairs have ≥2 signatures; "
        f"boolean collapse loses {pct_sigs_lost:.1f}% of total signatures "
        f"→ recommend {rec_label}"
    )

    numbers_table = [
        ("attributed (user, email) pairs", f"{n_pairs:,}"),
        ("pairs with ≥2 attributed signatures", f"{n_ge2:,} ({pct_pairs_ge2:.2f}%)"),
        ("total attributed signatures", f"{total_sigs:,}"),
        ("signatures lost by boolean collapse", f"{sigs_lost:,} ({pct_sigs_lost:.2f}%)"),
        ("recommended harness target", rec_label),
    ]

    impact = _impact_md(pct_pairs_ge2, pct_sigs_lost, recommendation)

    method_src = """
        aa = aa_cache.get(2)  # attributed_actions v2, ~13M rows
        per_pair = (
            aa.filter(pl.col("is_attributed"))
              .group_by(["user_id", "email_id"])
              .agg(pl.len().alias("n_attributed_actions"))
        )
        dist = per_pair.group_by("n_attributed_actions").agg(pl.len().alias("pair_count")).sort("n_attributed_actions")
        n_pairs = per_pair.height
        total_sigs = per_pair["n_attributed_actions"].sum()
        n_ge2 = per_pair.filter(pl.col("n_attributed_actions") >= 2).height
        sigs_lost = total_sigs - n_pairs
        pct_pairs_ge2 = 100.0 * n_ge2 / n_pairs
        pct_sigs_lost = 100.0 * sigs_lost / total_sigs
    """

    html = render_finding(
        finding_id="F00",
        title="Outcome cardinality check",
        tldr=tldr,
        wave=1,
        severity="BLOCKING",
        classification="confounder",
        question_md=(
            "GOAL.md says 'petitions signed per email.' civic_shout_user_emails.actioned is boolean. "
            "What is the distribution of n_attributed_actions per (user_id, email_id) for attributed pairs, "
            "and how much signal does the boolean collapse cost?"
        ),
        plain_english_md=(
            "Alice opens one email about climate. She clicks the link, lands on a page that "
            "lists 3 related petitions, and signs all 3. Our table records this as one row "
            "with `actioned=True` — a single yes/no. Did Alice sign 1 petition or 3?\n\n"
            "Right now we only know 'at least 1.' If lots of users sign multiple petitions per "
            "email, the boolean undercounts their real impact. F00 asks: how much signal does "
            "that yes/no flattening cost? If it's tiny (<5%), keep the boolean. If it's big, "
            "the modeling target has to change to expected petition COUNT, not probability."
        ),
        method_py_source=method_src,
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact,
        provenance={
            "data": "attributed_actions v2",
            "sample": "full",
            "ts": ts,
            "git": _GIT_SHA,
        },
        is_mockup=False,
    )

    out_path = _OUT_DIR / "F00_outcome_cardinality.html"
    out_path.write_text(html, encoding="utf-8")

    summary = (
        f"F00: {n_pairs:,} pairs, {pct_pairs_ge2:.1f}% with ≥2 signatures, "
        f"{pct_sigs_lost:.1f}% signatures lost by boolean collapse "
        f"→ recommend {rec_label}"
    )
    print(summary)


if __name__ == "__main__":
    main()
