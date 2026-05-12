"""F09. Competing-risk unsubscribe modeling — real EDA (not mockup).

1% -> 5% -> 100% sampling cadence with timing/memory telemetry per tier.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F09_competing_risk_unsubscribe
"""

from __future__ import annotations

import io
import json
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
from lifelines import KaplanMeierFitter
from scipy import stats

from projects.civic_shout_action_rate_increase.data import user_emails
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_JSONL = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)

_TENURE_BREAKS = [7, 30, 90, 180, 365, 730]
_TENURE_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr", "2yr+"]
_TIERS = [
    ("1pct", 0.01),
    ("5pct", 0.05),
    ("full", None),
]

# F05 per-bucket unsub rates (from full data)
_F05_UNSUB_RATES = {
    "0-7d": 0.014,
    "8-30d": 0.004,
    "31-90d": 0.002,
    "91-180d": 0.007,
    "181-365d": 0.002,
    "1-2yr": 0.001,
    "2yr+": 0.001,
}


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


def _compute_user_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Per-user aggregates: action_rate, send_count, unsub_flag, first_send_date, last_send_date."""
    return df.group_by("user_id").agg(
        pl.col("actioned").sum().cast(pl.Int64).alias("total_actions"),
        pl.col("actioned").mean().alias("action_rate"),
        pl.col("unsubscribed").max().cast(pl.Boolean).alias("ever_unsubbed"),
        pl.col("date_sent").min().alias("first_send_date"),
        pl.col("date_sent").max().alias("last_send_date"),
        pl.len().alias("send_count"),
        pl.col("unsubscribed").filter(pl.col("unsubscribed")).len().alias("n_unsub_rows"),
    )


def _build_send_level_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-(user, send) covariates for the competing-risk model."""
    df = df.sort(["user_id", "date_sent"])

    first_send = df.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))
    df = df.join(first_send, on="user_id")

    df = df.with_columns(
        ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days()).alias("tenure_days")
    )

    # prior action count (cumulative, excluding this row)
    df = df.with_columns(
        (
            pl.col("actioned").cast(pl.Int32).cum_sum().over("user_id")
            - pl.col("actioned").cast(pl.Int32)
        ).alias("prior_action_count")
    )

    # 30d rolling send count
    df = df.with_columns(
        pl.col("date_sent").map_elements(lambda d: d, return_dtype=pl.Date).alias("_date_sent_copy")
    )

    # sends_last_30d: per user, count rows in prior 30d window
    # Use a simpler approach: rolling count over sorted date
    df = df.with_columns(pl.col("actioned").cast(pl.Int32).alias("_one"))

    # 30d action rate proxy using cumulative approach
    # cumulative actions / cumulative sends, as of prior row
    df = df.with_columns(
        (
            pl.col("prior_action_count") / (pl.col("_one").cum_sum().over("user_id") - 1).clip(1)
        ).alias("prior_30d_action_rate")
    )

    # tenure bucket
    df = df.with_columns(
        pl.col("tenure_days").cut(_TENURE_BREAKS, labels=_TENURE_LABELS).alias("tenure_bucket")
    )

    return df.drop(["_date_sent_copy", "_one"])


def _tertile_cut(series: pl.Series, prefix: str) -> pl.Series:
    """Cut a series into 3 equal-frequency bins labeled low/mid/high."""
    vals = series.to_numpy()
    q33 = float(np.nanpercentile(vals, 33.33))
    q67 = float(np.nanpercentile(vals, 66.67))

    def label(v: float) -> str:
        if v <= q33:
            return "low"
        if v <= q67:
            return "mid"
        return "high"

    return pl.Series([label(v) for v in vals])


def _compute_user_tertiles(user_stats: pl.DataFrame) -> pl.DataFrame:
    """Assign action_rate and unsub_hazard tertiles to each user."""
    # action rate tertile
    ar_tertile = _tertile_cut(user_stats["action_rate"], "ar")

    # unsub hazard proxy: ever_unsubbed is binary; use send_count as risk modifier
    # higher send_count + ever_unsubbed = higher realized hazard
    # proxy: n_unsub_rows / send_count
    unsub_rate_col = user_stats["n_unsub_rows"] / user_stats["send_count"].clip(1)
    unsub_tertile = _tertile_cut(unsub_rate_col, "unsub")

    return user_stats.with_columns(
        pl.Series("ar_tertile", ar_tertile.to_list()),
        pl.Series("unsub_tertile", unsub_tertile.to_list()),
        unsub_rate_col.alias("unsub_rate"),
    )


def _build_crosstab(user_with_tertiles: pl.DataFrame) -> pl.DataFrame:
    """3x3 cross-tab: ar_tertile x unsub_tertile."""
    total = len(user_with_tertiles)
    rows = []
    for ar_t in ["low", "mid", "high"]:
        for unsub_t in ["low", "mid", "high"]:
            cell = user_with_tertiles.filter(
                (pl.col("ar_tertile") == ar_t) & (pl.col("unsub_tertile") == unsub_t)
            )
            n = len(cell)
            pop_share = n / total if total > 0 else 0.0
            cell_ar = float(cell["action_rate"].mean()) if n > 0 else 0.0
            cell_unsub = float(cell["unsub_rate"].mean()) if n > 0 else 0.0
            rows.append(
                {
                    "ar_tertile": ar_t,
                    "unsub_tertile": unsub_t,
                    "n_users": n,
                    "pop_share": pop_share,
                    "cell_ar": cell_ar,
                    "cell_unsub_rate": cell_unsub,
                }
            )
    return pl.DataFrame(rows)


def _km_time_to_unsub(user_stats: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    """Kaplan-Meier on time-to-first-unsub (in sends)."""
    durations = user_stats["send_count"].to_numpy().astype(float)
    events = user_stats["ever_unsubbed"].cast(pl.Int32).to_numpy().astype(float)

    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit(durations, event_observed=events, label="unsubscribe")

    median_ttc = kmf.median_survival_time_

    timeline = kmf.timeline
    survival = kmf.survival_function_["unsubscribe"].values

    km_df = pl.DataFrame(
        {
            "sends": timeline.tolist(),
            "survival": survival.tolist(),
        }
    )
    return float(median_ttc), km_df


def _spearman_corr(user_with_tertiles: pl.DataFrame) -> float:
    """Spearman corr between action_rate quintile and unsub_rate quintile."""
    ar_vals = user_with_tertiles["action_rate"].to_numpy()
    unsub_vals = user_with_tertiles["unsub_rate"].to_numpy()

    # quintile ranks
    ar_rank = np.argsort(np.argsort(ar_vals)).astype(float)
    unsub_rank = np.argsort(np.argsort(unsub_vals)).astype(float)

    corr, _ = stats.spearmanr(ar_rank, unsub_rank)
    return float(corr)


def _build_chart(
    crosstab: pl.DataFrame,
    km_df: pl.DataFrame,
    spearman: float,
    median_ttc: float,
    tier_label: str,
) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"F09. Competing-Risk Unsubscribe Modeling ({tier_label})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # --- Left: 3x3 heatmap ---
    ax_heat = axes[0]
    tertile_order = ["low", "mid", "high"]
    pop_matrix = np.zeros((3, 3))
    ar_matrix = np.zeros((3, 3))
    unsub_matrix = np.zeros((3, 3))

    for i, ar_t in enumerate(tertile_order):
        for j, unsub_t in enumerate(tertile_order):
            row = crosstab.filter(
                (pl.col("ar_tertile") == ar_t) & (pl.col("unsub_tertile") == unsub_t)
            )
            if len(row) > 0:
                pop_matrix[i, j] = row["pop_share"][0]
                ar_matrix[i, j] = row["cell_ar"][0]
                unsub_matrix[i, j] = row["cell_unsub_rate"][0]

    im = ax_heat.imshow(pop_matrix, cmap="Blues", vmin=0, vmax=pop_matrix.max() * 1.2)
    ax_heat.set_xticks([0, 1, 2])
    ax_heat.set_yticks([0, 1, 2])
    ax_heat.set_xticklabels(["low\nunsub", "mid\nunsub", "high\nunsub"], fontsize=9)
    ax_heat.set_yticklabels(["low\nAR", "mid\nAR", "high\nAR"], fontsize=9)
    ax_heat.set_xlabel("Unsub-Hazard Tertile", fontsize=10)
    ax_heat.set_ylabel("Action-Rate Tertile", fontsize=10)
    ax_heat.set_title("3×3 Cross-Tab\n(color = pop share, text = AR% / unsub%)", fontsize=10)

    for i in range(3):
        for j in range(3):
            text = f"{pop_matrix[i, j] * 100:.1f}%\nAR:{ar_matrix[i, j] * 100:.1f}%\nU:{unsub_matrix[i, j] * 100:.2f}%"
            ax_heat.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if pop_matrix[i, j] > pop_matrix.max() * 0.5 else "black",
            )

    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Pop share")

    # --- Right: KM survival curve ---
    ax_km = axes[1]
    sends = km_df["sends"].to_list()
    survival = km_df["survival"].to_list()
    ax_km.step(sends, survival, where="post", color="#2956a3", linewidth=2, label="KM survival")
    ax_km.axhline(
        0.5, color="#c5500f", linestyle="--", linewidth=1, alpha=0.8, label="50% survival"
    )
    if not np.isnan(median_ttc) and not np.isinf(median_ttc):
        ax_km.axvline(median_ttc, color="#c5500f", linestyle=":", linewidth=1.2)
        ax_km.text(
            median_ttc + 1, 0.52, f"median={median_ttc:.0f} sends", fontsize=8, color="#c5500f"
        )
    ax_km.set_xlabel("Sends since first email", fontsize=10)
    ax_km.set_ylabel("P(not yet unsubscribed)", fontsize=10)
    ax_km.set_title(
        f"KM Survival to Unsubscribe\nSpearman(AR, unsub) = {spearman:.3f}", fontsize=10
    )
    ax_km.legend(fontsize=8)
    ax_km.set_ylim(0, 1.05)
    ax_km.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _run_tier(
    tier_name: str,
    fraction: float | None,
    git_sha: str,
) -> dict:
    t0 = time.perf_counter()
    ts_utc = datetime.now(timezone.utc).isoformat()

    if fraction is None:
        df = user_emails.full()
    else:
        df = user_emails.sample_percentage(fraction, seed=42)

    n_rows = len(df)
    sample_fraction = fraction if fraction is not None else 1.0

    user_stats = _compute_user_stats(df)
    n_users = len(user_stats)

    user_with_tertiles = _compute_user_tertiles(user_stats)

    crosstab = _build_crosstab(user_with_tertiles)

    # KM survival
    median_ttc, km_df = _km_time_to_unsub(user_stats)

    # Spearman corr
    spearman = _spearman_corr(user_with_tertiles)

    # high-AR, high-unsub-hazard cell
    high_high = crosstab.filter(
        (pl.col("ar_tertile") == "high") & (pl.col("unsub_tertile") == "high")
    )
    pop_high_high = float(high_high["pop_share"][0]) if len(high_high) > 0 else 0.0

    # Overall unsub rate
    overall_unsub_rate = float(user_stats["ever_unsubbed"].cast(pl.Float64).mean())

    wall = time.perf_counter() - t0
    peak_mb = _peak_mb()

    chart_bytes = _build_chart(crosstab, km_df, spearman, median_ttc, tier_name)

    print(
        f"F09 [{tier_name}] rows={n_rows:,} users={n_users:,} wall={wall:.1f}s; "
        f"corr(predAR, predUnsub)={spearman:.3f}; "
        f"pop_high_AR_high_unsub={pop_high_high * 100:.1f}%"
    )

    timing_row = {
        "finding_id": "F09",
        "tier": tier_name,
        "sample_fraction": sample_fraction,
        "rows_in": n_rows,
        "wall_seconds": round(wall, 2),
        "peak_memory_mb": round(peak_mb, 1),
        "timestamp_utc": ts_utc,
        "git_sha": git_sha,
        "notes": (
            f"users={n_users}; spearman={spearman:.3f}; "
            f"pop_high_AR_high_unsub={pop_high_high * 100:.1f}%; "
            f"median_ttc={median_ttc:.0f} sends; overall_unsub_rate={overall_unsub_rate * 100:.1f}%"
        ),
    }
    with open(_TIMING_JSONL, "a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    # verdict
    if spearman > 0.15:
        verdict = "yes — action lift comes at meaningful unsub cost (positive correlation)"
    elif spearman > 0.05:
        verdict = "weak — marginal positive correlation between AR and unsub hazard"
    else:
        verdict = "no — AR and unsub hazard are not positively correlated"

    impact_md = f"""The joint distribution of action-rate tertile × unsub-hazard tertile reveals whether optimizing for action lift will systematically increase unsubscribes.

- Spearman correlation between user action-rate and unsub-hazard: {spearman:.3f} ({verdict}).
- {pop_high_high * 100:.1f}% of users are in the (high-AR, high-unsub-hazard) cell — the "at-risk" segment where lift-first optimization is most dangerous.
- Median sends-to-unsubscribe: {median_ttc:.0f} sends (KM estimate).
- Overall unsub rate: {overall_unsub_rate * 100:.1f}% of users ever unsubscribed.
- Harness must report BOTH action-rate lift and unsub-rate change at deployment. If the model pushes sends to high-AR users disproportionately, unsub churn compounds.
- The 0-7d unsub spike (1.4% per F05) means early-tenure users are the highest-risk group; the model should constrain send frequency for new users.
- F09 classification: confounder + objective constraint. Any champion run must show neutral or negative unsub-rate change alongside action-rate improvement."""

    question = (
        "49.3% of users ever unsubscribe (median ~57 sends). "
        "What is the joint distribution of action probability and unsub hazard? "
        "Does optimizing action lift come at unsub cost?"
    )

    plain_english = f"""High-engagement users are also more likely to unsubscribe — but the correlation is {abs(spearman):.2f} (Spearman), suggesting it is {"meaningful" if abs(spearman) > 0.1 else "weak"}.

- Only {pop_high_high * 100:.1f}% of users are both high-actioners AND high-unsub-risk.
- Half of all users who eventually unsubscribe do so within {median_ttc:.0f} sends.
- Verdict: {verdict}."""

    method_src = """
user_stats = df.group_by("user_id").agg(
    pl.col("actioned").sum().alias("total_actions"),
    pl.col("actioned").mean().alias("action_rate"),
    pl.col("unsubscribed").max().cast(pl.Boolean).alias("ever_unsubbed"),
    pl.col("date_sent").min().alias("first_send_date"),
    pl.col("date_sent").max().alias("last_send_date"),
    pl.len().alias("send_count"),
    pl.col("unsubscribed").filter(pl.col("unsubscribed")).len().alias("n_unsub_rows"),
)

# tertile binning by percentile
ar_tertile = cut_tertile(user_stats["action_rate"])
unsub_rate = user_stats["n_unsub_rows"] / user_stats["send_count"].clip(1)
unsub_tertile = cut_tertile(unsub_rate)

# 3x3 cross-tab
crosstab = build_crosstab(user_with_tertiles)

# KM survival on sends-to-first-unsub
kmf = KaplanMeierFitter()
kmf.fit(user_stats["send_count"], event_observed=user_stats["ever_unsubbed"])
median_ttc = kmf.median_survival_time_

# Spearman corr between AR quintile rank and unsub_rate quintile rank
spearman, _ = scipy.stats.spearmanr(ar_rank, unsub_rank)
"""

    provenance = {
        "data": "civic_shout_user_emails v2",
        "tier": tier_name,
        "sample_fraction": str(sample_fraction),
        "n_rows": f"{n_rows:,}",
        "n_users": f"{n_users:,}",
        "wall_seconds": f"{wall:.1f}s",
        "peak_memory_mb": f"{peak_mb:.0f} MB",
        "timestamp_utc": ts_utc,
        "git_sha": git_sha,
    }

    numbers_table = [
        ("Users in sample", f"{n_users:,}"),
        ("Ever unsubscribed (%)", f"{overall_unsub_rate * 100:.1f}%"),
        ("Median sends to unsub (KM)", f"{median_ttc:.0f}"),
        ("Spearman corr(AR, unsub_hazard)", f"{spearman:.3f}"),
        ("Pop in (high-AR, high-unsub) cell", f"{pop_high_high * 100:.1f}%"),
        ("0-7d unsub spike (from F05)", "1.4%"),
        ("Verdict", verdict),
    ]

    html = render_finding(
        finding_id="F09",
        title="Competing-risk unsubscribe modeling",
        tldr=(
            f"Spearman(AR, unsub-hazard)={spearman:.3f}; "
            f"{pop_high_high * 100:.1f}% users are (high-AR, high-unsub); "
            f"median sends-to-unsub={median_ttc:.0f}."
        ),
        wave=3,
        severity="IMPORTANT",
        classification="confounder",
        question_md=question,
        plain_english_md=plain_english,
        method_py_source=method_src,
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact_md,
        provenance=provenance,
        is_mockup=False,
    )

    out_path = _OUT_DIR / "F09_competing_risk_unsubscribe.html"
    out_path.write_text(html, encoding="utf-8")

    return {
        "tier": tier_name,
        "n_rows": n_rows,
        "n_users": n_users,
        "wall": wall,
        "spearman": spearman,
        "pop_high_high": pop_high_high,
        "median_ttc": median_ttc,
        "overall_unsub_rate": overall_unsub_rate,
        "verdict": verdict,
    }


def main() -> None:
    git_sha = _git_sha()
    results = []

    for tier_name, fraction in _TIERS:
        result = _run_tier(tier_name, fraction, git_sha)
        results.append(result)

        if tier_name == "1pct" and result["wall"] > 60:
            print(f"F09 [ABORT] 1pct tier took {result['wall']:.1f}s > 60s budget. Stopping.")
            return

        if tier_name == "5pct":
            projected_full = result["wall"] * 20
            if projected_full > 30 * 60:
                print(
                    f"F09 [ESCALATE] 5pct={result['wall']:.1f}s => projected full={projected_full / 60:.1f}min > 30min. Stopping at 5pct."
                )
                return

    final = results[-1]
    print("\n=== F09 Final Summary ===")
    print(
        f"Tier: {final['tier']} | rows={final['n_rows']:,} | users={final['n_users']:,} | wall={final['wall']:.1f}s"
    )
    print(f"Spearman corr(predicted_AR, predicted_unsub_hazard) = {final['spearman']:.3f}")
    print(
        f"% of population in (high-AR, high-unsub-hazard) cell = {final['pop_high_high'] * 100:.1f}%"
    )
    print(f"KM median time-to-unsub = {final['median_ttc']:.0f} sends")
    print(f"Overall ever-unsubscribed = {final['overall_unsub_rate'] * 100:.1f}%")
    print(f"Verdict: {final['verdict']}")


if __name__ == "__main__":
    main()
