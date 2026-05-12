"""F05. 91-180d tenure valley decomposition — real EDA (not mockup).

1% → 5% → 100% sampling cadence with timing/memory telemetry per tier.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F05_valley_decomposition_91_180d
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

from entities.civic_shout_engagement.email_activities_cache import cache as ea_cache
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
_TOP_N_EMAILS = 20
_TIERS = [
    ("1pct", 0.01),
    ("5pct", 0.05),
    ("full", None),
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


def _load_ea_opens_clicks() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load raw opens and clicks from email_activities v2 (full — not sampled)."""
    ea_oc = (
        ea_cache.scan(2)
        .filter(pl.col("action_type").is_in(["open", "click"]))
        .group_by(["user_id", "email_id", "action_type"])
        .agg(pl.len().alias("n_events"))
        .collect()
    )
    opens = (
        ea_oc.filter(pl.col("action_type") == "open")
        .select(["user_id", "email_id"])
        .with_columns(pl.lit(True).alias("raw_opened"))
    )
    clicks = (
        ea_oc.filter(pl.col("action_type") == "click")
        .select(["user_id", "email_id"])
        .with_columns(pl.lit(True).alias("raw_clicked"))
    )
    return opens, clicks


def _build_df(sample: pl.DataFrame, opens: pl.DataFrame, clicks: pl.DataFrame) -> pl.DataFrame:
    """Join sample with raw opens/clicks and compute tenure_days + tenure_bucket."""
    first_send = sample.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))
    df = (
        sample.join(first_send, on="user_id")
        .join(opens, on=["user_id", "email_id"], how="left")
        .join(clicks, on=["user_id", "email_id"], how="left")
        .with_columns(
            pl.col("raw_opened").fill_null(False),
            pl.col("raw_clicked").fill_null(False),
        )
        .with_columns(
            ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days()).alias("tenure_days")
        )
        .with_columns(
            pl.col("tenure_days").cut(_TENURE_BREAKS, labels=_TENURE_LABELS).alias("tenure_bucket")
        )
    )
    return df


_BUCKET_ORDER: dict[str, int] = {b: i for i, b in enumerate(_TENURE_LABELS)}


def _sort_by_bucket(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("tenure_bucket")
            .map_elements(lambda b: _BUCKET_ORDER.get(b, 99), return_dtype=pl.Int32)
            .alias("_sort_key")
        )
        .sort("_sort_key")
        .drop("_sort_key")
    )


def _compute_funnel(df: pl.DataFrame) -> pl.DataFrame:
    """Per-bucket funnel: open / click / action / unsub rates."""
    return _sort_by_bucket(
        df.group_by("tenure_bucket").agg(
            pl.col("raw_opened").mean().alias("open_rate"),
            pl.col("raw_clicked").mean().alias("click_rate"),
            pl.col("actioned").mean().alias("action_rate"),
            pl.col("unsubscribed").mean().alias("unsub_rate"),
            pl.len().alias("n_sends"),
        )
    )


def _compute_same_email(df: pl.DataFrame) -> pl.DataFrame:
    """AR by tenure bucket for top-N email_ids."""
    top_emails = (
        df.group_by("email_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(_TOP_N_EMAILS)["email_id"]
    )
    within = (
        df.filter(pl.col("email_id").is_in(top_emails.to_list()))
        .group_by(["email_id", "tenure_bucket"])
        .agg(
            pl.col("actioned").mean().alias("action_rate"),
            pl.len().alias("n_sends"),
        )
    )
    return within


def _valley_persists(same_email: pl.DataFrame) -> bool:
    """True if 91-180d is below adjacent buckets for majority of top emails."""
    valley_bucket = "91-180d"
    adjacent = {"31-90d", "181-365d"}
    emails_with_valley = 0
    emails_checked = 0
    for email_id in same_email["email_id"].unique().to_list():
        sub = same_email.filter(pl.col("email_id") == email_id)
        buckets_present = set(sub["tenure_bucket"].to_list())
        if valley_bucket not in buckets_present or not (adjacent & buckets_present):
            continue
        valley_ar = sub.filter(pl.col("tenure_bucket") == valley_bucket)["action_rate"].to_list()
        if not valley_ar:
            continue
        valley_ar_val = valley_ar[0]
        adj_ars = sub.filter(pl.col("tenure_bucket").is_in(list(adjacent)))["action_rate"].to_list()
        emails_checked += 1
        if valley_ar_val < min(adj_ars):
            emails_with_valley += 1
    if emails_checked == 0:
        return False
    return emails_with_valley >= emails_checked / 2


def _compute_cna_share(funnel: pl.DataFrame) -> pl.DataFrame:
    """Clicked-not-actioned share per bucket."""
    return funnel.with_columns(
        (
            (pl.col("click_rate") - pl.col("action_rate")).clip(0.0)
            / pl.col("click_rate").clip(0.001)
        ).alias("cna_share")
    ).select(["tenure_bucket", "cna_share", "click_rate", "action_rate"])


def _build_chart(
    funnel: pl.DataFrame,
    cna: pl.DataFrame,
) -> bytes:
    labels = [row["tenure_bucket"] for row in funnel.iter_rows(named=True)]
    open_rates = [row["open_rate"] for row in funnel.iter_rows(named=True)]
    click_rates = [row["click_rate"] for row in funnel.iter_rows(named=True)]
    action_rates = [row["action_rate"] for row in funnel.iter_rows(named=True)]

    cna_sorted = _sort_by_bucket(cna)
    cna_shares = [row["cna_share"] for row in cna_sorted.iter_rows(named=True)]
    cna_labels = [row["tenure_bucket"] for row in cna_sorted.iter_rows(named=True)]

    x = np.arange(len(labels))
    x_cna = np.arange(len(cna_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(x, open_rates, marker="o", label="open rate (raw)", color="#2956a3", linewidth=2)
    ax1.plot(x, click_rates, marker="o", label="click rate (raw)", color="#c5500f", linewidth=2)
    ax1.plot(x, action_rates, marker="o", label="action rate", color="#b00020", linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("rate")
    ax1.set_title("Funnel rates by tenure bucket\n(opens/clicks from raw email_activities)")
    ax1.legend(loc="upper left", fontsize=8)
    valley_idx = labels.index("91-180d") if "91-180d" in labels else None
    if valley_idx is not None:
        ax1.axvspan(valley_idx - 0.45, valley_idx + 0.45, color="#fff3cf", alpha=0.55)
        ax1.text(
            valley_idx, max(open_rates) * 0.97, "valley", fontsize=8, ha="center", color="#7a6700"
        )
    ax1.grid(alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax2.bar(x_cna, cna_shares, color="#c5500f")
    ax2.set_xticks(x_cna)
    ax2.set_xticklabels(cna_labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("(click - action) / click")
    ax2.set_title("Clicked-not-actioned share by tenure")
    for i, v in enumerate(cna_shares):
        ax2.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5)
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _numbers_table(
    funnel: pl.DataFrame, cna: pl.DataFrame, valley_persists: bool
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    cna_sorted = _sort_by_bucket(cna)
    for row in funnel.iter_rows(named=True):
        b = row["tenure_bucket"]
        rows.append((f"{b} sends", f"{row['n_sends']:,}"))
        rows.append((f"{b} open rate (raw)", f"{row['open_rate']:.2%}"))
        rows.append((f"{b} click rate (raw)", f"{row['click_rate']:.2%}"))
        rows.append((f"{b} action rate", f"{row['action_rate']:.2%}"))
    valley_row = cna_sorted.filter(pl.col("tenure_bucket") == "91-180d")
    if valley_row.height > 0:
        cna_val = valley_row["cna_share"].to_list()[0]
        rows.append(("91-180d clicked-not-actioned share", f"{cna_val:.1%}"))
    rows.append(("same-email valley persists", "YES" if valley_persists else "NO"))
    return rows


def _impact_md(funnel: pl.DataFrame, cna: pl.DataFrame, valley_persists: bool) -> str:
    valley_row = funnel.filter(pl.col("tenure_bucket") == "91-180d")
    ar_valley = valley_row["action_rate"].to_list()[0] if valley_row.height > 0 else float("nan")
    open_valley = valley_row["open_rate"].to_list()[0] if valley_row.height > 0 else float("nan")
    click_valley = valley_row["click_rate"].to_list()[0] if valley_row.height > 0 else float("nan")

    cna_valley_row = cna.filter(pl.col("tenure_bucket") == "91-180d")
    cna_val = (
        cna_valley_row["cna_share"].to_list()[0] if cna_valley_row.height > 0 else float("nan")
    )

    if valley_persists:
        verdict_lifecycle = (
            "**Valley persists within same email_id** → the dip is a real LIFECYCLE effect. "
            f"91-180d AR is {ar_valley:.2%} even controlling for email quality. "
            "Tenure bucket is a non-linear feature that MUST be included (bucketed, not raw days). "
            "This is the primary signal: user fatigue / habituation, not content routing."
        )
    else:
        verdict_lifecycle = (
            "**Valley disappears within same email_id** → the dip is a CONTENT-ALLOCATION effect. "
            f"91-180d AR ({ar_valley:.2%}) is largely recovering Civic Shout's routing policy — "
            "worse petitions are sent to this tenure cohort. Tenure feature is mostly proxy for routing."
        )

    if cna_val > 0.5:
        funnel_verdict = (
            f"**Post-click leak** ({cna_val:.1%} clicked-not-actioned): "
            "the failure point is the PETITION PAGE, not the email itself. "
            "A subject-line or email-content fix alone won't recover this valley. "
            "Flag a side-quest plan for petition-page optimisation for month-4 users."
        )
    elif click_valley < open_valley * 0.2:
        funnel_verdict = (
            f"**Pre-click leak** (open={open_valley:.2%} but click={click_valley:.2%}): "
            "the failure is EMAIL CONTENT or subject line. An email-level model tuned on "
            "91-180d users can address this."
        )
    else:
        funnel_verdict = (
            f"**Mixed leak**: open={open_valley:.2%}, click={click_valley:.2%}, "
            f"action={ar_valley:.2%}, CNA share={cna_val:.1%}. "
            "Both email content and petition-page contribute; check both."
        )

    return f"{verdict_lifecycle}\n\n{funnel_verdict}"


def _run_tier(
    tier_label: str,
    fraction: float | None,
    opens: pl.DataFrame,
    clicks: pl.DataFrame,
    git_sha: str,
) -> dict:
    t0 = time.time()
    print(f"\n[F05] Starting tier={tier_label} ...")

    if fraction is not None:
        sample = user_emails.sample_percentage(fraction, seed=42)
    else:
        sample = user_emails.full()

    n_rows = sample.height
    print(f"  sample rows={n_rows:,}")

    df = _build_df(sample, opens, clicks)
    funnel = _compute_funnel(df)
    same_email = _compute_same_email(df)
    persists = _valley_persists(same_email)
    cna = _compute_cna_share(funnel)

    valley_row = funnel.filter(pl.col("tenure_bucket") == "91-180d")
    ar_valley = valley_row["action_rate"].to_list()[0] if valley_row.height > 0 else float("nan")

    chart_bytes = _build_chart(funnel, cna)
    numbers_table = _numbers_table(funnel, cna, persists)
    impact = _impact_md(funnel, cna, persists)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wall = time.time() - t0
    peak_mb = _peak_mb()

    sample_label = f"{fraction:.0%} sample, seed=42" if fraction is not None else "full dataset"
    provenance = {
        "data": "user_emails v2 + email_activities v2 (opens/clicks from raw events per F03)",
        "sample": sample_label,
        "tier": tier_label,
        "wall_seconds": f"{wall:.1f}s",
        "peak_memory_mb": f"{peak_mb:.0f}",
        "ts": ts,
        "git": git_sha,
    }

    valley_ar_str = f"{ar_valley:.2%}"
    tldr = (
        (
            f"91-180d open={valley_row['open_rate'].to_list()[0]:.1%}, "
            f"click={valley_row['click_rate'].to_list()[0]:.1%}, "
            f"action={valley_ar_str}. "
            f"Valley within same email: {'YES (lifecycle)' if persists else 'NO (content-alloc)'}."
            f" [{tier_label} n={n_rows:,} wall={wall:.0f}s]"
        )
        if valley_row.height > 0
        else f"91-180d row not found. [{tier_label} n={n_rows:,}]"
    )

    html = render_finding(
        finding_id="F05",
        title="91-180d tenure valley decomposition",
        tldr=tldr,
        wave=2,
        severity="IMPORTANT",
        classification="both",
        question_md=(
            "91-180d has ~34% open rate but ~2.1% action rate. Is it user fatigue (user-side) or "
            "content-allocation bias (system-side)? Where in the funnel does the valley happen? "
            "Opens and clicks are sourced from raw email_activities per F03 (user_emails engagement "
            "flags are imputed and contaminated)."
        ),
        plain_english_md=(
            "Three users at month 4 on the list: Carol opens every email but never clicks. "
            "Dave clicks every email but never signs. Eve doesn't even open. The 91-180d "
            "'valley' is mostly Carols and Daves: still engaged enough to open or click "
            "but not crossing the finish line.\n\n"
            "Two stories for why. Story A: Civic Shout's routing sends WORSE petitions to "
            "month-4 users (content allocation). Story B: something about being 4 months in "
            "personally erodes willingness (user lifecycle). These have different fixes.\n\n"
            "F05 looks at the SAME emails across tenure buckets. If the valley shows up even "
            "when we hold the petition constant, it's Story B (lifecycle). If it disappears, "
            "it's Story A (allocation). We also check pre-click vs post-click leak."
        ),
        method_py_source=f"""
sample = user_emails.sample_percentage({fraction}, seed=42)  # {tier_label}

# Raw opens / clicks from email_activities v2 (per F03 — no imputation)
ea_oc = (
    ea_cache.scan(2)
    .filter(pl.col("action_type").is_in(["open", "click"]))
    .group_by(["user_id", "email_id", "action_type"])
    .agg(pl.len().alias("n_events"))
    .collect()
)
opens = ea_oc.filter(pl.col("action_type") == "open").select(["user_id", "email_id"]).with_columns(pl.lit(True).alias("raw_opened"))
clicks = ea_oc.filter(pl.col("action_type") == "click").select(["user_id", "email_id"]).with_columns(pl.lit(True).alias("raw_clicked"))

df = (
    sample
    .join(first_send_per_user, on="user_id")
    .join(opens, on=["user_id", "email_id"], how="left")
    .join(clicks, on=["user_id", "email_id"], how="left")
    .with_columns(
        pl.col("raw_opened").fill_null(False),
        pl.col("raw_clicked").fill_null(False),
        ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days()).alias("tenure_days"),
    )
    .with_columns(
        pl.col("tenure_days").cut([0,7,30,90,180,365,730],
            labels=["0-7d","8-30d","31-90d","91-180d","181-365d","1-2yr"]).alias("tenure_bucket")
    )
)

funnel = (
    df.group_by("tenure_bucket")
    .agg(
        pl.col("raw_opened").mean().alias("open_rate"),
        pl.col("raw_clicked").mean().alias("click_rate"),
        pl.col("actioned").mean().alias("action_rate"),
        pl.col("unsubscribed").mean().alias("unsub_rate"),
        pl.len().alias("n_sends"),
    )
)
# Same-email comparison: top-{_TOP_N_EMAILS} emails by send count; AR by tenure within each
top_emails = df.group_by("email_id").agg(pl.len().alias("n")).sort("n", descending=True).head({_TOP_N_EMAILS})["email_id"]
within_email = (
    df.filter(pl.col("email_id").is_in(top_emails))
    .group_by(["email_id", "tenure_bucket"])
    .agg(pl.col("actioned").mean().alias("action_rate"), pl.len().alias("n_sends"))
)
""",
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact,
        provenance=provenance,
        is_mockup=False,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "F05_valley_decomposition_91_180d.html"
    out_path.write_text(html, encoding="utf-8")

    timing_row = {
        "finding_id": "F05",
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
        f"F05 [{tier_label}] rows={n_rows:,} wall={wall:.1f}s; "
        f"91-180d AR={valley_ar_str}; "
        f"same-email valley persists={'YES' if persists else 'NO'}"
    )
    print(f"  wrote {out_path}")

    return {
        "tier": tier_label,
        "n_rows": n_rows,
        "wall": wall,
        "funnel": funnel,
        "cna": cna,
        "persists": persists,
        "ar_valley": ar_valley,
    }


def main() -> None:
    git_sha = _git_sha()
    print("[F05] Loading raw email_activities opens/clicks (full entity — not sampled)...")
    t_ea = time.time()
    opens, clicks = _load_ea_opens_clicks()
    print(
        f"  ea opens={opens.height:,} clicks={clicks.height:,} loaded in {time.time() - t_ea:.1f}s"
    )

    last_result: dict = {}
    for tier_label, fraction in _TIERS:
        last_result = _run_tier(tier_label, fraction, opens, clicks, git_sha)

    print("\n=== F05 FINAL SUMMARY ===")
    funnel = last_result["funnel"]
    for row in funnel.iter_rows(named=True):
        print(
            f"  {row['tenure_bucket']:12s}  open={row['open_rate']:.2%}  "
            f"click={row['click_rate']:.2%}  AR={row['action_rate']:.2%}  "
            f"unsub={row['unsub_rate']:.2%}  n={row['n_sends']:,}"
        )
    print(
        f"\nValley persists within same email: {'YES → lifecycle effect' if last_result['persists'] else 'NO → content-allocation bias'}"
    )
    valley_row = funnel.filter(pl.col("tenure_bucket") == "91-180d")
    if valley_row.height > 0:
        open_v = valley_row["open_rate"].to_list()[0]
        click_v = valley_row["click_rate"].to_list()[0]
        cna_row = last_result["cna"].filter(pl.col("tenure_bucket") == "91-180d")
        cna_v = cna_row["cna_share"].to_list()[0] if cna_row.height > 0 else float("nan")
        if cna_v > 0.5:
            funnel_loc = f"POST-CLICK (petition-page failure; CNA share={cna_v:.1%})"
        elif click_v < open_v * 0.2:
            funnel_loc = f"PRE-CLICK (email content/subject; open={open_v:.1%} click={click_v:.1%})"
        else:
            funnel_loc = f"MIXED (open={open_v:.1%} click={click_v:.1%} CNA={cna_v:.1%})"
        print(f"Funnel failure location for 91-180d: {funnel_loc}")


if __name__ == "__main__":
    main()
