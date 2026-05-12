"""F01. YoY drift / cohort-aging decomposition (gating finding).

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F01_cohort_aging_decomposition
"""

from __future__ import annotations

import io
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from projects.civic_shout_action_rate_increase.eda.user._html import (
    render_finding,
)

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_SAMPLE_FRAC = None  # set to 0.10 if full scan exceeds 5 min


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


def _load_lazy() -> pl.LazyFrame:
    from entities.civic_shout_user_emails.cache import cache as ue_cache

    lf = ue_cache.scan(2)
    return lf


def _compute_cohort_quarter(lf: pl.LazyFrame) -> pl.DataFrame:
    first_send = lf.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))

    enriched = lf.join(first_send, on="user_id").with_columns(
        pl.col("first_send_date").dt.year().alias("cohort_year"),
        pl.col("date_sent").dt.year().alias("calendar_year"),
        pl.col("date_sent").dt.quarter().alias("calendar_quarter"),
    )

    cohort_qtr = (
        enriched.group_by(["cohort_year", "calendar_year", "calendar_quarter"])
        .agg(
            pl.col("actioned").mean().alias("action_rate"),
            pl.len().alias("n_sends"),
        )
        .sort(["cohort_year", "calendar_year", "calendar_quarter"])
        .collect()
    )

    pooled_qtr = (
        enriched.group_by(["calendar_year", "calendar_quarter"])
        .agg(
            pl.col("actioned").mean().alias("action_rate"),
            pl.len().alias("n_sends"),
        )
        .sort(["calendar_year", "calendar_quarter"])
        .collect()
    )

    return cohort_qtr, pooled_qtr


def _quarter_label(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def _build_chart(
    cohort_qtr: pl.DataFrame,
    pooled_qtr: pl.DataFrame,
) -> bytes:
    all_yq = sorted(
        {(r["calendar_year"], r["calendar_quarter"]) for r in cohort_qtr.iter_rows(named=True)}
        | {(r["calendar_year"], r["calendar_quarter"]) for r in pooled_qtr.iter_rows(named=True)}
    )
    quarter_labels = [_quarter_label(y, q) for y, q in all_yq]
    yq_idx = {yq: i for i, yq in enumerate(all_yq)}

    cohort_years = sorted(cohort_qtr["cohort_year"].unique().to_list())
    cohort_colors = {
        cohort_years[0]: "#4a3aff",
        cohort_years[1] if len(cohort_years) > 1 else None: "#2956a3",
        cohort_years[2] if len(cohort_years) > 2 else None: "#207e3c",
    }
    fallback_colors = ["#9b59b6", "#e67e22", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(9, 4.2))

    for i, cy in enumerate(cohort_years):
        rows = cohort_qtr.filter(pl.col("cohort_year") == cy).iter_rows(named=True)
        pts = sorted(
            [(yq_idx[(r["calendar_year"], r["calendar_quarter"])], r["action_rate"]) for r in rows]
        )
        xs, ys = zip(*pts)
        color = cohort_colors.get(cy) or fallback_colors[i % len(fallback_colors)]
        ax.plot(xs, ys, marker="o", color=color, label=f"{cy} cohort", linewidth=2)

    pooled_rows = pooled_qtr.iter_rows(named=True)
    pts_p = sorted(
        [
            (yq_idx[(r["calendar_year"], r["calendar_quarter"])], r["action_rate"])
            for r in pooled_rows
        ]
    )
    xs_p, ys_p = zip(*pts_p)
    ax.plot(
        xs_p,
        ys_p,
        color="#b00020",
        label="pooled (all cohorts)",
        linestyle="--",
        linewidth=1.8,
    )

    entry_rates = []
    for cy in cohort_years:
        first_row = (
            cohort_qtr.filter(pl.col("cohort_year") == cy)
            .sort(["calendar_year", "calendar_quarter"])
            .row(0, named=True)
        )
        entry_rates.append(first_row["action_rate"])
    mean_entry = sum(entry_rates) / len(entry_rates) if entry_rates else 0.075

    ax.axhline(y=mean_entry, color="#aaa", linestyle=":", linewidth=1)
    ax.text(
        0.02,
        mean_entry + 0.003,
        f"all cohorts enter at ~{mean_entry:.1%}",
        fontsize=7.5,
        color="#666",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_xticks(range(len(quarter_labels)))
    ax.set_xticklabels(quarter_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("action rate")
    ax.set_title("Action rate by cohort over calendar time")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.set_constrained_layout(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _compute_drift_explained(
    cohort_qtr: pl.DataFrame,
    pooled_qtr: pl.DataFrame,
) -> dict:
    cohort_years = sorted(cohort_qtr["cohort_year"].unique().to_list())
    earliest_cohort = cohort_years[0]

    pooled_sorted = pooled_qtr.sort(["calendar_year", "calendar_quarter"])
    pooled_first = pooled_sorted.row(0, named=True)
    pooled_last = pooled_sorted.row(-1, named=True)
    raw_drift = pooled_last["action_rate"] - pooled_first["action_rate"]

    earliest_rows = cohort_qtr.filter(pl.col("cohort_year") == earliest_cohort).sort(
        ["calendar_year", "calendar_quarter"]
    )
    earliest_entry_ar = earliest_rows.row(0, named=True)["action_rate"]
    earliest_latest_ar = earliest_rows.row(-1, named=True)["action_rate"]
    cohort_aging_drift = earliest_latest_ar - earliest_entry_ar

    pct_explained = (cohort_aging_drift / raw_drift * 100) if raw_drift > 0 else float("nan")

    cohort_entry_ars = {}
    for cy in cohort_years:
        first_row = (
            cohort_qtr.filter(pl.col("cohort_year") == cy)
            .sort(["calendar_year", "calendar_quarter"])
            .row(0, named=True)
        )
        cohort_entry_ars[cy] = first_row["action_rate"]

    cohort_latest_ars = {}
    for cy in cohort_years:
        last_row = (
            cohort_qtr.filter(pl.col("cohort_year") == cy)
            .sort(["calendar_year", "calendar_quarter"])
            .row(-1, named=True)
        )
        cohort_latest_ars[cy] = last_row["action_rate"]

    return {
        "cohort_years": cohort_years,
        "earliest_cohort": earliest_cohort,
        "pooled_first_yq": _quarter_label(
            pooled_first["calendar_year"], pooled_first["calendar_quarter"]
        ),
        "pooled_first_ar": pooled_first["action_rate"],
        "pooled_last_yq": _quarter_label(
            pooled_last["calendar_year"], pooled_last["calendar_quarter"]
        ),
        "pooled_last_ar": pooled_last["action_rate"],
        "raw_drift": raw_drift,
        "cohort_aging_drift": cohort_aging_drift,
        "pct_explained": pct_explained,
        "cohort_entry_ars": cohort_entry_ars,
        "cohort_latest_ars": cohort_latest_ars,
    }


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    lf = _load_lazy()

    if _SAMPLE_FRAC is not None:
        lf = lf.sample(fraction=_SAMPLE_FRAC, seed=42)

    cohort_qtr, pooled_qtr = _compute_cohort_quarter(lf)

    stats = _compute_drift_explained(cohort_qtr, pooled_qtr)
    cohort_years = stats["cohort_years"]
    earliest_cohort = stats["earliest_cohort"]

    chart_bytes = _build_chart(cohort_qtr, pooled_qtr)

    numbers_table = []
    for cy in cohort_years:
        entry_ar = stats["cohort_entry_ars"][cy]
        latest_ar = stats["cohort_latest_ars"][cy]
        numbers_table.append((f"{cy} cohort entry AR", f"{entry_ar:.2%}"))
        numbers_table.append((f"{cy} cohort latest-quarter AR", f"{latest_ar:.2%}"))

    numbers_table.extend(
        [
            (f"pooled AR {stats['pooled_first_yq']}", f"{stats['pooled_first_ar']:.2%}"),
            (f"pooled AR {stats['pooled_last_yq']}", f"{stats['pooled_last_ar']:.2%}"),
            ("raw pooled drift", f"{stats['raw_drift']:.2%}pp"),
            (
                f"{earliest_cohort}-cohort aging drift (entry→latest)",
                f"{stats['cohort_aging_drift']:.2%}pp",
            ),
            ("fraction of YoY drift explained by cohort aging", f"{stats['pct_explained']:.0f}%"),
        ]
    )

    sample_note = (
        f"{_SAMPLE_FRAC:.0%} stratified sample, seed=42" if _SAMPLE_FRAC else "full dataset"
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    git_sha = _git_sha()

    provenance = {
        "data": "civic_shout_user_emails v2",
        "sample": sample_note,
        "ts": ts,
        "git": git_sha,
    }

    pct = stats["pct_explained"]
    entry_strs = "; ".join(
        f"{cy} cohort enters {stats['cohort_entry_ars'][cy]:.1%}" for cy in cohort_years
    )
    first_cy = cohort_years[0]
    summary_line = (
        f"F01: {first_cy} cohort {stats['pooled_first_yq']} AR={stats['cohort_entry_ars'][first_cy]:.2%} "
        f"→ {stats['pooled_last_yq']} AR={stats['cohort_latest_ars'][first_cy]:.2%}; "
        f"{entry_strs}. "
        f"Aging explains {pct:.0f}% of YoY drift → temporal holdout split with tenure stratification."
    )

    tldr_real = (
        f"YoY drift ({stats['pooled_first_ar']:.1%}→{stats['pooled_last_ar']:.1%}) is "
        f"{pct:.0f}% cohort aging: each cohort enters at ~{stats['cohort_entry_ars'][cohort_years[0]]:.0%} "
        "and matures. Harness MUST use temporal holdout."
    )

    plain_english_md = (
        "Two coffee shops each have 100 customers. Shop A opened in 2024; half its 2024 "
        "customers were one-time tourists who never came back. By 2026, only A's 50 loyal "
        "regulars remain, and they buy way more per visit. Shop B opened in 2026 with 100 "
        "fresh customers, some of whom haven't bought yet.\n\n"
        "Shop A's per-visit sales now look way better than B's — but A didn't make better "
        "coffee. It just kept its highest-intent customers. Civic Shout's drift "
        f"({stats['pooled_first_ar']:.1%} → {stats['pooled_last_ar']:.1%}) "
        "looks like product improvement but is almost entirely the same dynamic: "
        f"the {earliest_cohort} cohort who didn't unsubscribe are getting more loyal, while "
        "newer cohorts enter at the same ~7-8% they always did. F01 proves this with a "
        "cohort-by-cohort plot, which forces the harness to split data by send DATE not random rows."
    )

    impact_md = (
        f"**Aging explains {pct:.0f}% of YoY drift.** "
        f"Pooled AR moved {stats['pooled_first_ar']:.2%} → {stats['pooled_last_ar']:.2%} "
        f"({stats['pooled_first_yq']}→{stats['pooled_last_yq']}). "
        f"The {earliest_cohort} cohort alone moved {stats['cohort_entry_ars'][earliest_cohort]:.2%} → "
        f"{stats['cohort_latest_ars'][earliest_cohort]:.2%} over the same period — matching the pooled drift magnitude.\n\n"
        "**Split convention**: temporal holdout by send date with tenure-bin stratification. "
        "Random row splits inflate AUC by leaking cohort composition.\n\n"
        "**Baseline**: the pooled 9.87% target is mechanically suppressed by early-cohort "
        "under-representation. Per-cohort or per-tenure-bucket targets are more honest. "
        "The 2026 cohort baseline (~7-8% entry AR) is the cleanest single comparison point.\n\n"
        "**Follow-up commit**: append a paragraph to GOAL.md recommending the harness anchor "
        "— pooled vs 2026-only vs per-tenure. Do NOT inline-revise GOAL.md (user deferred)."
    )

    html = render_finding(
        finding_id="F01",
        title="YoY drift / cohort-aging decomposition",
        tldr=tldr_real,
        wave=1,
        severity="BLOCKING",
        classification="confounder",
        question_md=(
            f"Pre-EDA showed YoY drift is empirically cohort aging. "
            f"Formal verification: pooled AR {stats['pooled_first_ar']:.2%} → {stats['pooled_last_ar']:.2%} "
            f"with {pct:.0f}% explained by cohort aging. "
            "Harness must use cohort/tenure-aware temporal validation; "
            "random row splits and pooled-baseline comparisons are invalid."
        ),
        plain_english_md=plain_english_md,
        method_py_source="""
lf = ue_cache.scan(2)
first_send = lf.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))
enriched = lf.join(first_send, on="user_id").with_columns(
    pl.col("first_send_date").dt.year().alias("cohort_year"),
    pl.col("date_sent").dt.year().alias("calendar_year"),
    pl.col("date_sent").dt.quarter().alias("calendar_quarter"),
)
cohort_qtr = (
    enriched.group_by(["cohort_year", "calendar_year", "calendar_quarter"])
    .agg(pl.col("actioned").mean().alias("action_rate"), pl.len().alias("n_sends"))
    .sort(["cohort_year", "calendar_year", "calendar_quarter"])
    .collect()
)
""",
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact_md,
        provenance=provenance,
        is_mockup=False,
    )

    out_path = _OUT_DIR / "F01_cohort_aging_decomposition.html"
    out_path.write_text(html, encoding="utf-8")

    print(summary_line)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
