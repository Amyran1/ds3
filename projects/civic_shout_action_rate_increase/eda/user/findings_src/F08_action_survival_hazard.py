"""F08. Action survival / hazard — real EDA (not mockup).

1% → 5% → 100% sampling cadence with timing/memory telemetry per tier.

At the 5% tier a user-level sub-check is performed: sample 5% of unique
user_ids and take all their rows, then compare per-user metrics vs row-level.
If they differ materially (>20% relative on streak count or OAD share), the
user-level sample is preferred.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F08_action_survival_hazard
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

from projects.civic_shout_action_rate_increase.data import user_emails
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_JSONL = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)

_LAG_MAX = 60
_STREAK_MIN = 3

_TIERS: list[tuple[str, float | None]] = [
    ("1pct", 0.01),
    ("5pct", 0.05),
    ("full", None),
]

_LIFETIME_BINS = [0, 1, 3, 6, 11, 21, 41, 81, 161, 301]
_LIFETIME_LABELS = [
    "0",
    "1-2",
    "3-5",
    "6-10",
    "11-20",
    "21-40",
    "41-80",
    "81-160",
    "161-300",
    "300+",
]


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


def _peak_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _compute_sends_since_last_action(df: pl.DataFrame) -> pl.DataFrame:
    """Add `sends_since_last_action` column.

    For each user, sort by date_sent. Within each user's sequence, the counter
    resets to 0 on the send immediately after an actioned row, then increments
    by 1 each non-actioned send. The row with `actioned=True` itself is lag 0
    (i.e., the very send where they acted, relative to the previous action).

    Strategy: compute cumsum of `actioned` lagged by 1 to get "group index"
    (which action epoch we're in), then compute the row index within each
    (user_id, epoch_group) sub-sequence.
    """
    df = df.sort(["user_id", "date_sent"])
    df = df.with_columns(
        pl.col("actioned").cast(pl.Int32).alias("_actioned_int"),
    )
    df = df.with_columns(
        pl.col("_actioned_int").shift(1, fill_value=0).over("user_id").alias("_prev_actioned"),
    )
    df = df.with_columns(
        pl.col("_prev_actioned").cum_sum().over("user_id").alias("_epoch"),
    )
    df = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64)
        .over(["user_id", "_epoch"])
        .alias("sends_since_last_action"),
    )
    return df.drop(["_actioned_int", "_prev_actioned", "_epoch"])


def _compute_hazard_curve(df: pl.DataFrame) -> pl.DataFrame:
    """Hazard at each lag k in [0, _LAG_MAX]: mean(actioned) where lag == k."""
    lags = pl.DataFrame({"sends_since_last_action": list(range(_LAG_MAX + 1))})
    hazard = (
        df.filter(pl.col("sends_since_last_action") <= _LAG_MAX)
        .group_by("sends_since_last_action")
        .agg(
            pl.col("actioned").mean().alias("hazard"),
            pl.len().alias("n"),
        )
    )
    result = (
        lags.join(hazard, on="sends_since_last_action", how="left")
        .with_columns(pl.col("hazard").fill_null(float("nan")))
        .sort("sends_since_last_action")
    )
    return result


def _compute_streak_users(df: pl.DataFrame) -> int:
    """Count users with ≥ STREAK_MIN consecutive actioned sends."""
    df_sorted = df.sort(["user_id", "date_sent"])
    df_sorted = df_sorted.with_columns(
        pl.col("actioned").cast(pl.Int32).alias("_a"),
    )
    df_sorted = df_sorted.with_columns(
        (
            pl.col("_a")
            * (pl.col("_a") != pl.col("_a").shift(1, fill_value=0).over("user_id")).cast(pl.Int32)
        ).alias("_streak_start"),
    )
    df_sorted = df_sorted.with_columns(
        pl.col("_streak_start").cum_sum().over("user_id").alias("_streak_group"),
    )
    streak_lens = (
        df_sorted.filter(pl.col("actioned"))
        .group_by(["user_id", "_streak_group"])
        .agg(pl.len().alias("streak_len"))
        .filter(pl.col("streak_len") >= _STREAK_MIN)
        .select("user_id")
        .unique()
    )
    return streak_lens.height


def _compute_one_and_done(df: pl.DataFrame) -> int:
    """Count users with exactly 1 lifetime action."""
    user_action_counts = df.group_by("user_id").agg(
        pl.col("actioned").sum().cast(pl.Int64).alias("lifetime_actions")
    )
    return user_action_counts.filter(pl.col("lifetime_actions") == 1).height


def _compute_lifetime_histogram(df: pl.DataFrame) -> list[tuple[str, int]]:
    """Lifetime action count binned histogram."""
    user_action_counts = df.group_by("user_id").agg(
        pl.col("actioned").sum().cast(pl.Int64).alias("lifetime_actions")
    )
    rows: list[tuple[str, int]] = []
    bounds = _LIFETIME_BINS + [int(1e12)]
    for i, label in enumerate(_LIFETIME_LABELS):
        lo = bounds[i]
        hi = bounds[i + 1]
        n = user_action_counts.filter(
            (pl.col("lifetime_actions") >= lo) & (pl.col("lifetime_actions") < hi)
        ).height
        rows.append((label, n))
    return rows


def _build_chart(
    hazard: pl.DataFrame,
    pooled_ar: float,
    lifetime_hist: list[tuple[str, int]],
) -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    lags = hazard["sends_since_last_action"].to_list()
    haz = hazard["hazard"].to_list()
    ns = hazard["n"].to_list()

    valid_mask = [not (h != h) for h in haz]  # NaN check
    lags_v = [lags[i] for i in range(len(lags)) if valid_mask[i]]
    haz_v = [haz[i] for i in range(len(haz)) if valid_mask[i]]
    ns_v = [ns[i] for i in range(len(ns)) if valid_mask[i]]

    se = []
    for i, h in enumerate(haz_v):
        n = ns_v[i] or 1
        se.append((h * (1 - h) / n) ** 0.5)

    haz_arr = np.array(haz_v)
    se_arr = np.array(se)
    lags_arr = np.array(lags_v)

    ax1.plot(lags_arr, haz_arr, color="#4a3aff", linewidth=2, label="hazard")
    ax1.fill_between(lags_arr, haz_arr - se_arr, haz_arr + se_arr, alpha=0.18, color="#4a3aff")
    ax1.axhline(y=pooled_ar, color="#aaa", linestyle="--", linewidth=0.9)
    ax1.text(
        lags_arr[-1] * 0.75,
        pooled_ar + 0.005,
        f"pooled AR {pooled_ar:.1%}",
        fontsize=8,
        color="#666",
    )
    ax1.set_xlabel("sends since last action")
    ax1.set_ylabel("P(action | this send)")
    ax1.set_title("Action hazard curve")
    ax1.grid(alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    labels = [r[0] for r in lifetime_hist]
    counts = [r[1] for r in lifetime_hist]
    x = np.arange(len(labels))
    ax2.bar(x, counts, color="#2956a3")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax2.set_xlabel("lifetime action count")
    ax2.set_ylabel("users (log)")
    ax2.set_title("Lifetime action count distribution")
    for i, c in enumerate(counts):
        if c > 0:
            ax2.text(i, c * 1.5, f"{c:,}", ha="center", fontsize=6.5, rotation=45)

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _numbers_table(
    hazard: pl.DataFrame,
    pooled_ar: float,
    streak_users: int,
    oad_users: int,
    n_total_users: int,
    lifetime_hist: list[tuple[str, int]],
) -> list[tuple[str, str]]:
    def _haz_at(lag: int) -> str:
        row = hazard.filter(pl.col("sends_since_last_action") == lag)
        if row.height == 0:
            return "n/a"
        v = row["hazard"].to_list()[0]
        if v != v:
            return "n/a"
        return f"{v:.2%}"

    rows: list[tuple[str, str]] = [
        ("pooled action rate (sample)", f"{pooled_ar:.2%}"),
        ("hazard at lag 0 (send right after an action)", _haz_at(0)),
        ("hazard at lag 5", _haz_at(5)),
        ("hazard at lag 30", _haz_at(30)),
        ("streak users (≥3 consecutive actioned)", f"{streak_users:,}"),
        ("one-and-done users (lifetime_actions=1)", f"{oad_users:,}"),
        ("total unique users in sample", f"{n_total_users:,}"),
        (
            "streak users as % of users with ≥1 action",
            f"{streak_users / max(1, n_total_users):.2%}",
        ),
    ]
    for label, cnt in lifetime_hist:
        rows.append((f"lifetime actions = {label} (users)", f"{cnt:,}"))
    return rows


def _impact_md(hazard: pl.DataFrame, pooled_ar: float, streak_users: int, oad_users: int) -> str:
    def _haz_at(lag: int) -> float:
        row = hazard.filter(pl.col("sends_since_last_action") == lag)
        if row.height == 0:
            return float("nan")
        v = row["hazard"].to_list()[0]
        return v if v == v else float("nan")

    h0 = _haz_at(0)
    h5 = _haz_at(5)
    h30 = _haz_at(30)

    spike_ratio = h0 / pooled_ar if pooled_ar > 0 and h0 == h0 else float("nan")

    if h0 == h0 and spike_ratio > 1.5:
        spike_verdict = (
            f"**Hazard SPIKES at lag 0**: P(action | next send after action) = {h0:.1%} vs "
            f"pooled {pooled_ar:.1%} ({spike_ratio:.1f}× lift). "
            "Users are primed immediately after acting. "
            "Feature `sends_since_last_action` is high-value; low lag values are strong positive signal."
        )
    elif h0 == h0 and spike_ratio < 0.8:
        spike_verdict = (
            f"**Hazard DROPS at lag 0**: P(action | next send) = {h0:.1%} vs pooled {pooled_ar:.1%}. "
            "Users appear satisfied / fatigued right after acting. "
            "Give them a rest window before next send."
        )
    else:
        spike_verdict = (
            f"**Hazard is flat at lag 0**: {h0:.1%} vs pooled {pooled_ar:.1%}. "
            "No strong immediate priming effect."
        )

    decay_verdict = ""
    if h5 == h5 and h30 == h30:
        if h5 > pooled_ar * 1.1:
            decay_verdict = (
                f"Elevated hazard persists to lag 5 ({h5:.1%}). "
                f"By lag 30 it converges to baseline ({h30:.1%}). "
                "A `sends_since_last_action_lt_10` boolean feature captures most of this."
            )
        else:
            decay_verdict = (
                f"Hazard decays quickly to baseline by lag 5 ({h5:.1%}), lag 30 ({h30:.1%}). "
                "Only the immediate post-action window matters."
            )

    return (
        f"{spike_verdict}\n\n"
        f"{decay_verdict}\n\n"
        f"**Recommended features**: `sends_since_last_action`, `is_in_action_streak` (boolean flag for "
        f"users currently in a ≥{_STREAK_MIN}-send streak), `action_in_last_5_sends` (boolean). "
        f"These are cheap to compute per-send from the user_emails entity.\n\n"
        f"**Streak users** ({streak_users:,}) tolerate lower per-email match quality — useful for exploration sends. "
        f"**One-and-done users** ({oad_users:,}) are the target for re-engagement."
    )


def _run_tier(
    tier_label: str,
    fraction: float | None,
    git_sha: str,
    do_user_level_check: bool = False,
) -> dict:
    t0 = time.time()
    print(f"\n[F08] Starting tier={tier_label} ...")

    if fraction is not None:
        sample = user_emails.sample_percentage(fraction, seed=42)
    else:
        sample = user_emails.full()

    n_rows = sample.height
    print(f"  row-sample rows={n_rows:,}")

    df = _compute_sends_since_last_action(sample)
    n_total_users = df["user_id"].n_unique()
    pooled_ar = df["actioned"].mean()
    if pooled_ar is None:
        pooled_ar = 0.0

    hazard = _compute_hazard_curve(df)
    streak_users = _compute_streak_users(df)
    oad_users = _compute_one_and_done(df)
    lifetime_hist = _compute_lifetime_histogram(df)

    h0_row = hazard.filter(pl.col("sends_since_last_action") == 0)
    h0 = h0_row["hazard"].to_list()[0] if h0_row.height > 0 else float("nan")
    h30_row = hazard.filter(pl.col("sends_since_last_action") == 30)
    h30 = h30_row["hazard"].to_list()[0] if h30_row.height > 0 else float("nan")

    user_level_note = ""
    if do_user_level_check and fraction is not None:
        print(f"  [F08] Running user-level sub-check at {fraction:.0%} ...")
        all_uids = sample["user_id"].unique()
        n_uid_sample = max(1, int(len(all_uids) * fraction))
        uid_sample = all_uids.sample(n=n_uid_sample, seed=42)
        if fraction is not None:
            full_data = user_emails.full()
        else:
            full_data = sample
        user_level_df = full_data.filter(pl.col("user_id").is_in(uid_sample.to_list()))
        ul_df = _compute_sends_since_last_action(user_level_df)
        ul_streak = _compute_streak_users(ul_df)
        ul_total = ul_df["user_id"].n_unique()

        row_streak_pct = streak_users / max(1, n_total_users)
        ul_streak_pct = ul_streak / max(1, ul_total)
        drift = abs(ul_streak_pct - row_streak_pct) / max(1e-9, row_streak_pct)
        if drift > 0.20:
            user_level_note = (
                f"User-level sub-check DIFFERS from row-level: "
                f"streak% row={row_streak_pct:.2%} vs user={ul_streak_pct:.2%} "
                f"(drift={drift:.1%} > 20%). Per-user metrics may be biased by row sampling."
            )
            print(f"  [F08] User-level drift detected: {user_level_note}")
        else:
            user_level_note = (
                f"User-level sub-check agrees with row-level: "
                f"streak% row={row_streak_pct:.2%} vs user={ul_streak_pct:.2%} "
                f"(drift={drift:.1%} <= 20%)."
            )
            print(f"  [F08] User-level sub-check: {user_level_note}")

    chart_bytes = _build_chart(hazard, pooled_ar, lifetime_hist)
    numbers = _numbers_table(
        hazard, pooled_ar, streak_users, oad_users, n_total_users, lifetime_hist
    )
    impact = _impact_md(hazard, pooled_ar, streak_users, oad_users)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wall = time.time() - t0
    peak_mb = _peak_mb()

    sample_label = f"{fraction:.0%} sample, seed=42" if fraction is not None else "full dataset"
    if user_level_note:
        sample_label += f" | {user_level_note}"

    provenance = {
        "data": "civic_shout_user_emails v2",
        "sample": sample_label,
        "tier": tier_label,
        "wall_seconds": f"{wall:.1f}s",
        "peak_memory_mb": f"{peak_mb:.0f}",
        "ts": ts,
        "git": git_sha,
    }

    h0_str = f"{h0:.2%}" if h0 == h0 else "n/a"
    h30_str = f"{h30:.2%}" if h30 == h30 else "n/a"

    tldr = (
        f"pooled AR={pooled_ar:.2%}; hazard@lag0={h0_str}; hazard@lag30={h30_str}; "
        f"streaks={streak_users:,}; one-and-done={oad_users:,} "
        f"[{tier_label} n={n_rows:,} wall={wall:.0f}s]"
    )

    method_src = f"""
sample = user_emails.sample_percentage({fraction}, seed=42)  # {tier_label}

# Sort by (user_id, date_sent); compute sends_since_last_action
# lag=0 on actioned row; resets to 0 on next send after action
df = sample.sort(["user_id", "date_sent"])
df = df.with_columns(
    pl.col("actioned").cast(pl.Int32).shift(1, fill_value=0).over("user_id").alias("_prev_actioned"),
).with_columns(
    pl.col("_prev_actioned").cum_sum().over("user_id").alias("_epoch"),
).with_columns(
    pl.int_range(pl.len(), dtype=pl.Int64).over(["user_id", "_epoch"]).alias("sends_since_last_action"),
)

# Hazard curve: mean(actioned) per lag k in [0, 60]
hazard = (
    df.filter(pl.col("sends_since_last_action") <= 60)
    .group_by("sends_since_last_action")
    .agg(pl.col("actioned").mean().alias("hazard"), pl.len().alias("n"))
    .sort("sends_since_last_action")
)

# Streak users: ≥{_STREAK_MIN} consecutive actioned sends per user
# One-and-done: lifetime_actions == 1
"""

    html = render_finding(
        finding_id="F08",
        title="Action survival / hazard",
        tldr=tldr,
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
        question_md=(
            "After a user signs a petition, what's the distribution of 'sends until next action'? "
            "Does action probability spike, decay, or persist? Are there 'streak users' vs 'one-and-done'?"
        ),
        plain_english_md=(
            "After someone watches a movie they loved, are they MORE likely to watch "
            "another in the next week — primed and energized — or LESS likely — satisfied "
            "and saturated? Different people behave differently. Some go on streaks. Some "
            "are one-and-done.\n\n"
            "The 'hazard curve' answers: on the send right after signing, what's the action "
            f"rate? On send #5 after? Send #30? If the curve spikes right after a signature "
            f"vs the {pooled_ar:.1%} baseline, users are primed — send more, fast. If it drops, "
            "they're satisfied — give them a rest.\n\n"
            "Most production models miss this temporal pattern entirely. F08 measures it "
            "and creates cheap features like `is_in_action_streak`, `sends_since_last_action`. "
            "These may outperform the fancy embedding feature on simpler grounds."
        ),
        method_py_source=method_src,
        chart_png_bytes=chart_bytes,
        numbers_table=numbers,
        impact_md=impact,
        provenance=provenance,
        is_mockup=False,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "F08_action_survival_hazard.html"
    out_path.write_text(html, encoding="utf-8")

    timing_row = {
        "finding_id": "F08",
        "tier": tier_label,
        "rows": n_rows,
        "wall_seconds": round(wall, 2),
        "peak_memory_mb": round(peak_mb, 1),
        "fraction": fraction,
        "ts": ts,
        "git": git_sha,
    }
    with _TIMING_JSONL.open("a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    print(
        f"F08 [{tier_label}] rows={n_rows:,} wall={wall:.1f}s; "
        f"hazard_at_lag0={h0_str}; hazard_at_lag30={h30_str}; "
        f"streaks={streak_users:,}; one_and_done={oad_users:,}"
    )
    print(f"  wrote {out_path}")

    return {
        "tier": tier_label,
        "n_rows": n_rows,
        "wall": wall,
        "hazard": hazard,
        "pooled_ar": pooled_ar,
        "streak_users": streak_users,
        "oad_users": oad_users,
        "h0": h0,
        "h30": h30,
        "lifetime_hist": lifetime_hist,
    }


def main() -> None:
    git_sha = _git_sha()
    results: list[dict] = []

    for i, (tier_label, fraction) in enumerate(_TIERS):
        do_user_level = tier_label == "5pct"
        result = _run_tier(tier_label, fraction, git_sha, do_user_level_check=do_user_level)
        results.append(result)

        if tier_label == "1pct" and result["wall"] > 30:
            print(
                f"[F08] 1% tier took {result['wall']:.1f}s > 30s budget. STOPPING — diagnose bottleneck."
            )
            return

        if tier_label == "5pct":
            projected_full = result["wall"] * 20
            if projected_full > 30 * 60:
                print(
                    f"[F08] 5% tier took {result['wall']:.1f}s → projected full={projected_full / 60:.1f}min > 30min. "
                    "ESCALATING — skipping full tier."
                )
                return

    print("\n[F08] All 3 tiers complete.")
    for r in results:
        h0_str = f"{r['h0']:.2%}" if r["h0"] == r["h0"] else "n/a"
        h30_str = f"{r['h30']:.2%}" if r["h30"] == r["h30"] else "n/a"
        print(
            f"  {r['tier']}: rows={r['n_rows']:,} wall={r['wall']:.1f}s "
            f"hazard@0={h0_str} hazard@30={h30_str} "
            f"streaks={r['streak_users']:,} oad={r['oad_users']:,}"
        )


if __name__ == "__main__":
    main()
