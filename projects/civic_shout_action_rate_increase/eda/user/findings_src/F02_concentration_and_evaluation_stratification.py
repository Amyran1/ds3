"""F02. Action concentration → evaluation stratification.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F02_concentration_and_evaluation_stratification
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

from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_JSONL = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)
_STOP_AFTER_1PCT_SECONDS = 30.0


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


def _get_data(tier: str) -> tuple[pl.DataFrame, float]:
    from projects.civic_shout_action_rate_increase.data import user_emails

    if tier == "1pct":
        return user_emails.sample_percentage(0.01, seed=42), 0.01
    if tier == "5pct":
        return user_emails.sample_percentage(0.05, seed=42), 0.05
    if tier == "full":
        return user_emails.full(), 1.0
    raise ValueError(f"unknown tier: {tier}")


def _compute_lorenz(df: pl.DataFrame) -> dict:
    per_user = (
        df.group_by("user_id").agg(pl.col("actioned").sum().alias("n_actions")).sort("n_actions")
    )

    n_users = per_user.height
    users_with_zero = per_user.filter(pl.col("n_actions") == 0).height
    total_actions = per_user["n_actions"].sum()

    n_actions_arr = per_user["n_actions"].to_numpy().astype(np.float64)
    cumulative_actions = np.cumsum(n_actions_arr)

    if total_actions == 0:
        cum_share = np.zeros_like(cumulative_actions)
    else:
        cum_share = cumulative_actions / total_actions

    user_rank_frac = np.arange(1, n_users + 1) / n_users

    def _top_pct_share(pct: float) -> float:
        if total_actions == 0:
            return 0.0
        cutoff_idx = int(np.ceil((1.0 - pct) * n_users))
        top_actions = n_actions_arr[cutoff_idx:].sum()
        return float(top_actions / total_actions)

    top_1 = _top_pct_share(0.01)
    top_5 = _top_pct_share(0.05)
    top_10 = _top_pct_share(0.10)
    top_20 = _top_pct_share(0.20)

    if total_actions > 0:
        n = n_users
        gini_num = np.sum((2 * np.arange(1, n + 1) - n - 1) * n_actions_arr)
        gini = float(gini_num / (n * n_actions_arr.sum()))
    else:
        gini = 0.0

    return {
        "n_users": n_users,
        "users_with_zero": users_with_zero,
        "pct_zero": users_with_zero / n_users if n_users > 0 else 0.0,
        "total_actions": int(total_actions),
        "top_1_share": top_1,
        "top_5_share": top_5,
        "top_10_share": top_10,
        "top_20_share": top_20,
        "gini": gini,
        "user_rank_frac": user_rank_frac,
        "cum_share": cum_share,
    }


def _build_chart(stats: dict, tier: str) -> bytes:
    user_rank_frac = stats["user_rank_frac"]
    cum_share = stats["cum_share"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(user_rank_frac, cum_share, color="#4a3aff", linewidth=2, label="actual")
    ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", label="equality")

    annots = [
        (0.99, stats["top_1_share"], "top 1%"),
        (0.95, stats["top_5_share"], "top 5%"),
        (0.90, stats["top_10_share"], "top 10%"),
        (0.80, stats["top_20_share"], "top 20%"),
    ]
    for user_pct, action_share, label in annots:
        x_pt = user_pct
        y_pt = 1.0 - action_share
        offset_x = -0.14
        offset_y = -0.07
        ax.annotate(
            f"{label} → {action_share:.1%}",
            xy=(x_pt, y_pt),
            xytext=(x_pt + offset_x, y_pt + offset_y),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.6),
        )

    ax.set_xlabel("cumulative share of users (sorted by action count, ascending)")
    ax.set_ylabel("cumulative share of actions")
    ax.set_title(f"Lorenz curve of attributed actions per user ({tier})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.set_constrained_layout(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _append_timing_row(
    tier: str,
    sample_fraction: float,
    rows_in: int,
    wall_seconds: float,
    peak_memory_mb: float,
    git_sha: str,
    notes: str = "",
) -> None:
    row = {
        "finding_id": "F02",
        "tier": tier,
        "sample_fraction": sample_fraction,
        "rows_in": rows_in,
        "wall_seconds": round(wall_seconds, 3),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "notes": notes,
    }
    with _TIMING_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _render_and_write(
    stats: dict,
    tier: str,
    sample_fraction: float,
    wall_seconds: float,
    chart_bytes: bytes,
    git_sha: str,
) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    tier_label_map = {"1pct": "1%", "5pct": "5%", "full": "100%"}
    tier_label = tier_label_map.get(tier, tier)

    numbers_table = [
        ("distinct users", f"{stats['n_users']:,}"),
        (
            "users with 0 actions",
            f"{stats['users_with_zero']:,} ({stats['pct_zero']:.1%})",
        ),
        ("top 1% share of actions", f"{stats['top_1_share']:.1%}"),
        ("top 5% share of actions", f"{stats['top_5_share']:.1%}"),
        ("top 10% share of actions", f"{stats['top_10_share']:.1%}"),
        ("top 20% share of actions", f"{stats['top_20_share']:.1%}"),
        ("Gini coefficient", f"{stats['gini']:.3f}"),
        ("total attributed actions", f"{stats['total_actions']:,}"),
    ]

    tldr = (
        f"[{tier_label} sample] Top 1% of users → {stats['top_1_share']:.1%} of actions; "
        f"top 10% → {stats['top_10_share']:.1%}. Unstratified metrics are invalid. "
        f"Gini={stats['gini']:.3f}."
    )

    provenance = {
        "data": "civic_shout_user_emails v2",
        "sample": f"{tier_label} sample (fraction={sample_fraction}, seed=42)",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git": git_sha,
        "tier": tier,
        "wall_seconds": str(round(wall_seconds, 2)),
    }

    html = render_finding(
        finding_id="F02",
        title="Action concentration → evaluation stratification",
        tldr=tldr,
        wave=2,
        severity="IMPORTANT",
        classification="confounder",
        question_md=(
            f"Pre-EDA showed top 1% → 26.5% of actions, top 10% → 84%. "
            f"Reproduce the Lorenz curve cleanly and recommend an evaluation regime "
            f"that prevents power users from dominating the metric. "
            f"[This rendering: {tier_label} sample, {stats['n_users']:,} distinct users]"
        ),
        plain_english_md=(
            "Imagine measuring 'average household income' in a room that happens to include "
            "Jeff Bezos. The 'average' is millions of dollars — useless for understanding "
            "the median family. Our pooled action rate is exactly that average.\n\n"
            f"Top 1% of users drive {stats['top_1_share']:.1%} of all signatures. A model "
            "that scores 'this user is likely to sign' is mostly identifying the top 1% — "
            "but we already know who they are and we already email them. The 10% lift has "
            "to come from the MIDDLE of the distribution, not the saturated top.\n\n"
            "F02's job is to prove this with a Lorenz curve and pin the consequence: every "
            "evaluation metric must be stratified (by tenure, by prior-action count, by "
            "cohort), because a single pooled AUC will be dominated by 'did the model find "
            "the rich users?' — which is unactionable."
        ),
        method_py_source="""
per_user = (
    df.group_by("user_id")
      .agg(pl.col("actioned").sum().alias("n_actions"))
      .sort("n_actions")
)
n_actions_arr = per_user["n_actions"].to_numpy().astype(np.float64)
cum_share = np.cumsum(n_actions_arr) / n_actions_arr.sum()
user_rank_frac = np.arange(1, len(n_actions_arr) + 1) / len(n_actions_arr)
# Top k% share: n_actions_arr[ceil((1-k)*N):].sum() / total
""",
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=(
            "Unstratified global metrics are invalid. Pick the right alternative based on intervention shape:\n\n"
            "- If intervention = 'choose which users receive which email' → **within-cell ranking** (per-email or per-day).\n"
            "- If intervention = 'score all active users for the daily email' → **stratified evaluation** by tenure bucket, prior-action count, recent activity, cohort, send frequency.\n"
            "- Always report send-weighted AND user-weighted metrics in parallel.\n\n"
            "Default to stratified evaluation + group-normalized AUC/log-loss until F04's variance decomposition tells us "
            "how much of apparent user-variance is content-mix."
        ),
        provenance=provenance,
        is_mockup=False,
    )

    out_path = _OUT_DIR / "F02_concentration_and_evaluation_stratification.html"
    out_path.write_text(html, encoding="utf-8")


def _run_tier(tier: str, prev_stats: dict | None, git_sha: str) -> dict:
    t0 = time.perf_counter()
    df, sample_fraction = _get_data(tier)
    rows_in = df.height

    stats = _compute_lorenz(df)
    chart_bytes = _build_chart(stats, tier)
    wall_seconds = time.perf_counter() - t0

    peak_memory_mb = psutil.Process().memory_info().rss / 1e6

    tier_label_map = {"1pct": "1pct", "5pct": "5pct", "full": "full"}
    tier_label = tier_label_map.get(tier, tier)

    print(
        f"F02 [{tier_label}] rows_in={rows_in:,} wall={wall_seconds:.1f}s "
        f"peak_mem={peak_memory_mb:.0f}mb "
        f"top_1pct={stats['top_1_share']:.1%} top_10pct={stats['top_10_share']:.1%}"
    )

    notes = ""
    if prev_stats is not None:
        drift_1pct = abs(stats["top_1_share"] - prev_stats["top_1_share"]) / max(
            prev_stats["top_1_share"], 1e-9
        )
        drift_10pct = abs(stats["top_10_share"] - prev_stats["top_10_share"]) / max(
            prev_stats["top_10_share"], 1e-9
        )
        flags = []
        if drift_1pct > 0.20:
            flags.append(f"top_1pct drifted {drift_1pct:.1%} vs prev tier")
        if drift_10pct > 0.20:
            flags.append(f"top_10pct drifted {drift_10pct:.1%} vs prev tier")
        if flags:
            notes = "DRIFT_FLAG: " + "; ".join(flags)

    _render_and_write(stats, tier, sample_fraction, wall_seconds, chart_bytes, git_sha)
    _append_timing_row(tier, sample_fraction, rows_in, wall_seconds, peak_memory_mb, git_sha, notes)

    return stats


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()

    tiers = ["1pct", "5pct", "full"]
    prev_stats: dict | None = None

    for tier in tiers:
        t_start = time.perf_counter()

        if tier == "1pct":
            stats = _run_tier(tier, prev_stats, git_sha)
            elapsed = time.perf_counter() - t_start
            if elapsed > _STOP_AFTER_1PCT_SECONDS:
                print(
                    f"F02 1pct probe exceeded {_STOP_AFTER_1PCT_SECONDS}s "
                    f"(actual={elapsed:.1f}s). STOPPING — investigate bottleneck."
                )
                return
        else:
            stats = _run_tier(tier, prev_stats, git_sha)

        prev_stats = stats

    final = prev_stats
    if final is not None:
        print(
            f"\nFinal Lorenz numbers ({tiers[-1]}):\n"
            f"  distinct users:   {final['n_users']:,}\n"
            f"  users 0 actions:  {final['users_with_zero']:,} ({final['pct_zero']:.1%})\n"
            f"  top  1% share:    {final['top_1_share']:.1%}\n"
            f"  top  5% share:    {final['top_5_share']:.1%}\n"
            f"  top 10% share:    {final['top_10_share']:.1%}\n"
            f"  top 20% share:    {final['top_20_share']:.1%}\n"
            f"  Gini coefficient: {final['gini']:.3f}\n"
        )
        print(
            "Recommended evaluation regime:\n"
            "  Stratified evaluation by tenure bucket + prior-action-count bin. "
            "Report send-weighted AND user-weighted metrics in parallel. "
            "Within-cell ranking (per-email or per-day) is the preferred form if "
            "the intervention is email selection rather than user scoring."
        )


if __name__ == "__main__":
    main()
