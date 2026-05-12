"""F04. Email / date opportunity audit — real EDA (not mockup).

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F04_email_date_opportunity_audit

Sampling cadence: 1% probe → 5% probe → full run (escalate if 5%*20 > 30min).
"""

from __future__ import annotations

import io
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_LEDGER = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)

_TENURE_BREAKS = [7, 30, 90, 180, 365, 730]
_TENURE_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr", "2yr+"]


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


def _peak_memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _append_timing(
    tier: str,
    fraction: float,
    rows_in: int,
    wall_seconds: float,
    peak_memory_mb: float,
    git_sha: str,
    notes: str = "",
) -> None:
    row = {
        "finding_id": "F04",
        "tier": tier,
        "sample_fraction": fraction,
        "rows_in": rows_in,
        "wall_seconds": round(wall_seconds, 2),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "notes": notes,
    }
    with _TIMING_LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _load_sample(fraction: float | None, seed: int = 42) -> pl.DataFrame:
    from entities.civic_shout_user_emails.cache import cache as ue_cache

    if fraction is None:
        df = ue_cache.get(2)
    else:
        df = ue_cache.get(2).sample(fraction=fraction, seed=seed, shuffle=False)
    return df


def _add_tenure(df: pl.DataFrame) -> pl.DataFrame:
    first_send = df.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))
    df = df.join(first_send, on="user_id")
    df = df.with_columns(
        ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days()).alias("tenure_days")
    )
    df = df.with_columns(
        pl.col("tenure_days")
        .cut(
            _TENURE_BREAKS,
            labels=_TENURE_LABELS,
        )
        .alias("tenure_bucket")
    )
    return df


def _variance_component(
    df: pl.DataFrame,
    group_cols: list[str],
    overall_mean: float,
    sst: float,
) -> float:
    gb = df.group_by(group_cols).agg(
        pl.col("actioned").mean().alias("g_mean"),
        pl.col("actioned").var(ddof=0).fill_null(0.0).alias("g_var"),
        pl.len().alias("g_n"),
    )
    sse = (gb["g_n"] * gb["g_var"]).sum()
    r2 = float(1.0 - sse / sst) if sst > 0 else 0.0
    return max(0.0, r2)


def _compute_variance_components(df: pl.DataFrame) -> dict[str, float]:
    df_actioned = df.with_columns(pl.col("actioned").cast(pl.Float64))
    df_actioned = df_actioned.with_columns(
        pl.col("date_sent").cast(pl.Utf8).str.slice(0, 10).alias("date_sent_d")
    )

    overall_var = float(df_actioned["actioned"].var(ddof=0) or 0.0)
    n = df_actioned.height
    sst = n * overall_var

    specs: dict[str, list[str]] = {
        "user only": ["user_id"],
        "email FE only": ["email_id"],
        "date FE only": ["date_sent_d"],
        "user + email": ["user_id", "email_id"],
        "user + date": ["user_id", "date_sent_d"],
        "user + email + date": ["user_id", "email_id", "date_sent_d"],
    }

    results: dict[str, float] = {}
    for name, cols in specs.items():
        results[name] = _variance_component(df_actioned, cols, 0.0, sst)

    return results


def _compute_tenure_residualized(df: pl.DataFrame) -> pl.DataFrame:
    df2 = df.with_columns(pl.col("actioned").cast(pl.Float64))

    email_means = df2.group_by("email_id").agg(pl.col("actioned").mean().alias("email_mean_ar"))
    df2 = df2.join(email_means, on="email_id")
    df2 = df2.with_columns(
        (pl.col("actioned") - pl.col("email_mean_ar")).alias("actioned_residualized")
    )

    tenure_stats = (
        df2.group_by("tenure_bucket")
        .agg(
            pl.col("actioned").mean().alias("raw_ar"),
            pl.col("actioned_residualized").mean().alias("residualized_ar"),
            pl.len().alias("n_sends"),
        )
        .sort("tenure_bucket")
    )
    return tenure_stats


def _build_chart(
    r2_results: dict[str, float],
    tenure_stats: pl.DataFrame,
) -> bytes:
    specs = list(r2_results.keys())
    r2_vals = [r2_results[s] for s in specs]

    colors = []
    for s in specs:
        if s == "email FE only":
            colors.append("#c5500f")
        elif s == "user + email + date":
            colors.append("#207e3c")
        else:
            colors.append("#2956a3")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.bar(range(len(specs)), r2_vals, color=colors)
    ax1.set_xticks(range(len(specs)))
    ax1.set_xticklabels(specs, rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("variance explained (R²) of actioned")
    ax1.set_title("ANOVA decomposition: where does user-level variance live?")
    for i, v in enumerate(r2_vals):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax1.set_ylim(0, max(r2_vals) * 1.2 + 0.05)

    buckets = tenure_stats["tenure_bucket"].to_list()
    raw_ar = tenure_stats["raw_ar"].to_list()
    resid_ar = tenure_stats["residualized_ar"].to_list()

    x = np.arange(len(buckets))
    ax2.plot(x, raw_ar, marker="o", color="#2956a3", linewidth=2, label="raw AR")
    ax2.plot(
        x,
        resid_ar,
        marker="s",
        color="#c5500f",
        linewidth=2,
        linestyle="--",
        label="email-FE residualized AR",
    )
    ax2.axhline(y=0, color="#aaa", linestyle=":", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("action rate (or residual)")
    ax2.set_title("Raw vs email-FE residualized AR by tenure bucket")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _run_tier(
    fraction: float | None,
    tier_name: str,
    tier_label: str,
    git_sha: str,
) -> tuple[dict[str, float], pl.DataFrame, int, float]:
    t0 = time.perf_counter()
    df = _load_sample(fraction)
    rows_in = df.height
    df = _add_tenure(df)
    r2_results = _compute_variance_components(df)
    tenure_stats = _compute_tenure_residualized(df)
    wall = time.perf_counter() - t0
    mem = _peak_memory_mb()

    user_r2 = r2_results.get("user only", 0.0)
    email_r2 = r2_results.get("email FE only", 0.0)
    date_r2 = r2_results.get("date FE only", 0.0)
    joint_r2 = r2_results.get("user + email + date", 0.0)

    print(
        f"F04 [{tier_label}] rows_in={rows_in:,} wall={wall:.1f}s "
        f"user_R2={user_r2:.4f} email_R2={email_r2:.4f} date_R2={date_r2:.4f} joint_R2={joint_r2:.4f}"
    )

    _append_timing(
        tier=tier_name,
        fraction=fraction if fraction is not None else 1.0,
        rows_in=rows_in,
        wall_seconds=wall,
        peak_memory_mb=mem,
        git_sha=git_sha,
        notes=f"user_R2={user_r2:.4f} email_R2={email_r2:.4f} date_R2={date_r2:.4f} joint_R2={joint_r2:.4f}",
    )

    return r2_results, tenure_stats, rows_in, wall


def _write_html(
    r2_results: dict[str, float],
    tenure_stats: pl.DataFrame,
    rows_in: int,
    wall: float,
    tier_label: str,
    fraction: float | None,
    git_sha: str,
) -> None:
    from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

    chart_bytes = _build_chart(r2_results, tenure_stats)

    user_r2 = r2_results.get("user only", 0.0)
    email_r2 = r2_results.get("email FE only", 0.0)
    date_r2 = r2_results.get("date FE only", 0.0)
    user_email_r2 = r2_results.get("user + email", 0.0)
    user_date_r2 = r2_results.get("user + date", 0.0)
    joint_r2 = r2_results.get("user + email + date", 0.0)

    incremental_user_over_email = user_email_r2 - email_r2
    incremental_user_over_date = user_date_r2 - date_r2

    numbers_table = [
        ("rows analysed", f"{rows_in:,}"),
        ("user-only R²", f"{user_r2:.4f}"),
        ("email-FE-only R²", f"{email_r2:.4f}"),
        ("date-FE-only R²", f"{date_r2:.4f}"),
        ("user + email R²", f"{user_email_r2:.4f}"),
        ("user + date R²", f"{user_date_r2:.4f}"),
        ("joint R² (user + email + date)", f"{joint_r2:.4f}"),
        ("incremental user above email FE", f"{incremental_user_over_email:.4f}"),
        ("incremental user above date FE", f"{incremental_user_over_date:.4f}"),
        ("email FE dominates user FE?", "YES" if email_r2 > user_r2 else "NO"),
    ]

    if email_r2 > user_r2:
        routing_dominant = (
            "Email-FE explains more variance than user-FE alone — Civic Shout's content routing "
            "is the primary driver of action-rate heterogeneity. User features must be evaluated "
            "against same-email or same-day baselines to separate real propensity from routing exposure."
        )
        intervention_note = (
            "Consider reframing the harness intervention from 'score users for the daily email' "
            "to 'select among candidate petitions for the day.'"
        )
    else:
        routing_dominant = (
            "User-FE explains more variance than email-FE — genuine user-level propensity differences "
            "exist above and beyond content routing. User-history features have meaningful scope."
        )
        intervention_note = (
            "The current intervention frame ('score users for the daily email') is reasonable. "
            "User-history features can add real lift."
        )

    impact_md = (
        f"{routing_dominant}\n\n"
        f"{intervention_note}\n\n"
        "**Mandatory for F06/F07**: if email-FE R² exceeds user R², evaluate against same-email "
        "or same-day baselines wherever possible. A user feature that adds lift against pooled "
        "baselines but not against email-FE residuals is recovering routing exposure, not propensity.\n\n"
        "**Flag for harness design**: same-email residualized metrics should be part of F07's "
        "ablation ladder to verify that embedding centroid adds real lift above content routing."
    )

    sample_note = f"{fraction:.0%} sample, seed=42" if fraction is not None else "full dataset"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    provenance = {
        "data": "civic_shout_user_emails v2",
        "sample": sample_note,
        "tier": tier_label,
        "wall_seconds": f"{wall:.1f}s",
        "ts": ts,
        "git": git_sha,
    }

    tldr = (
        f"email-FE R²={email_r2:.3f} vs user-only R²={user_r2:.3f}; "
        f"joint R²={joint_r2:.3f}. "
        + (
            "Email routing dominates user propensity."
            if email_r2 > user_r2
            else "User propensity dominates email routing."
        )
    )

    html = render_finding(
        finding_id="F04",
        title="Email / date opportunity audit",
        tldr=tldr,
        wave=2,
        severity="IMPORTANT",
        classification="confounder",
        question_md=(
            "Users who appear high-AR may simply have been exposed to easier-to-convert emails. "
            "Decompose the variance of actioned among user / email / date fixed effects using "
            "analytical variance components (no design matrix): "
            "R² = 1 - SSE_within / SST for each grouping."
        ),
        plain_english_md=(
            "Alice always gets the easy crossword puzzles. Bob always gets the hard ones. "
            "Alice solves 90% of hers; Bob solves 60% of his. Looking at the people, "
            "'Alice is the smarter student.' Looking at the puzzles, 'Alice's puzzles are just "
            "easier.' The honest answer requires looking at both at once.\n\n"
            "Civic Shout already routes emails: maybe high-engagement users get the most "
            "compelling petitions, and low-engagement users get the leftovers. Then any "
            "'user feature' looks predictive — but it's really recovering 'which user gets "
            "which petition,' i.e. Civic Shout's existing routing policy. A model built on "
            "that won't IMPROVE routing; it'll just reproduce it.\n\n"
            "F04 decomposes variance into user vs email vs date effects. If email-FE alone "
            "explains more than user-FE alone, our user-level features must be evaluated "
            "against same-email or same-day baselines to know if they add real lift."
        ),
        method_py_source="""
# Analytical variance decomposition — no design matrix, O(n) per grouping
df = ue.sample(fraction=0.05, seed=42)  # or full
df = df.with_columns(pl.col("date_sent").cast(pl.Utf8).str.slice(0, 10).alias("date_sent_d"))
overall_var = df["actioned"].cast(pl.Float64).var(ddof=0)
sst = df.height * overall_var

def r2_for_groups(df, group_cols):
    gb = df.group_by(group_cols).agg(
        pl.col("actioned").var(ddof=0).fill_null(0.0).alias("g_var"),
        pl.len().alias("g_n"),
    )
    sse = (gb["g_n"] * gb["g_var"]).sum()
    return max(0.0, 1 - sse / sst)

r2_user   = r2_for_groups(df, ["user_id"])
r2_email  = r2_for_groups(df, ["email_id"])
r2_date   = r2_for_groups(df, ["date_sent_d"])
r2_ue     = r2_for_groups(df, ["user_id", "email_id"])
r2_ud     = r2_for_groups(df, ["user_id", "date_sent_d"])
r2_joint  = r2_for_groups(df, ["user_id", "email_id", "date_sent_d"])

# Per-tenure residualization (email FE proxy)
email_means = df.group_by("email_id").agg(pl.col("actioned").mean().alias("email_mean_ar"))
df2 = df.join(email_means, on="email_id").with_columns(
    (pl.col("actioned") - pl.col("email_mean_ar")).alias("actioned_residualized")
)
tenure_stats = df2.group_by("tenure_bucket").agg(
    pl.col("actioned").mean().alias("raw_ar"),
    pl.col("actioned_residualized").mean().alias("residualized_ar"),
)
""",
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact_md,
        provenance=provenance,
        is_mockup=False,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "F04_email_date_opportunity_audit.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()

    # ── Tier 1: 1% probe ─────────────────────────────────────────────────────
    r2_results, tenure_stats, rows_in_1pct, wall_1pct = _run_tier(
        fraction=0.01, tier_name="1pct", tier_label="1%", git_sha=git_sha
    )
    _write_html(r2_results, tenure_stats, rows_in_1pct, wall_1pct, "1%", 0.01, git_sha)

    if wall_1pct > 30.0:
        print(
            f"F04 STOP: 1% tier took {wall_1pct:.1f}s > 30s budget. "
            "Diagnose bottleneck before scaling."
        )
        return

    # ── Tier 2: 5% probe ─────────────────────────────────────────────────────
    r2_results, tenure_stats, rows_in_5pct, wall_5pct = _run_tier(
        fraction=0.05, tier_name="5pct", tier_label="5%", git_sha=git_sha
    )
    _write_html(r2_results, tenure_stats, rows_in_5pct, wall_5pct, "5%", 0.05, git_sha)

    if wall_5pct > 300.0:
        projected_full = wall_5pct * 20
        print(
            f"F04 STOP: 5% tier took {wall_5pct:.1f}s > 5min budget. "
            f"Projected full={projected_full / 60:.1f}min."
        )
        return

    projected_full_min = (wall_5pct * 20) / 60.0
    if projected_full_min > 30.0:
        print(
            f"F04 ESCALATE: 5%_wall={wall_5pct:.1f}s → projected full={projected_full_min:.1f}min > 30min. "
            "Stopping at 5% tier. Use tier=5pct results."
        )
        return

    # ── Tier 3: Full run ──────────────────────────────────────────────────────
    r2_results, tenure_stats, rows_in_full, wall_full = _run_tier(
        fraction=None, tier_name="full", tier_label="full", git_sha=git_sha
    )
    _write_html(r2_results, tenure_stats, rows_in_full, wall_full, "full", None, git_sha)


if __name__ == "__main__":
    main()
