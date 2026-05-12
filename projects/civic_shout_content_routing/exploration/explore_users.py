"""User-level exploration of the wide engagement frame.

Reads the derived wide parquet built by ``build_wide_engagement.py`` and
writes a single self-contained HTML file with one section per question:
prose explanation + base64-embedded PNG.

The HTML is incrementally extensible -- add more ``Section`` builders to
``SECTIONS`` and re-run.

Usage
-----
    python -m projects.civic_shout_content_routing.exploration.explore_users
    python ... --in path/to/wide.parquet --out path/to/exploration.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from entities.civic_shout_engagement.aggregate_emails_cache import (
    cache as agg_cache,
)

# 2026-04-21 build cutoff matches email_activities v2 / attributed_actions v2.
ANCHOR_UTC = datetime(2026, 4, 21, tzinfo=timezone.utc)
CHURN_DAYS = 90


@dataclass
class Section:
    slug: str
    heading: str
    prose: str
    png_b64: str


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Per-user aggregate frame -- computed once, reused across sections.
# ---------------------------------------------------------------------------

def build_action_matrix(
    wide: pl.DataFrame,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Sparse user x email action matrix (actioners only, 1.0 = action present).

    Returns (M_csr, email_ids_sorted, user_ids_sorted) where M[u, e] == 1
    iff user_ids[u] took an attributed action on email_ids[e].
    """
    actioned = (
        wide.filter(pl.col("actioned"))
        .select("email_id", "user_id")
        .unique()
    )
    email_ids = np.sort(actioned["email_id"].unique().to_numpy())
    user_ids = np.sort(actioned["user_id"].unique().to_numpy())
    rows = np.searchsorted(user_ids, actioned["user_id"].to_numpy())
    cols = np.searchsorted(email_ids, actioned["email_id"].to_numpy())
    data = np.ones(len(rows), dtype=np.float32)
    M = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(email_ids))
    )
    return M, email_ids, user_ids


def build_user_frame(wide: pl.DataFrame) -> pl.DataFrame:
    """One row per user with engagement counts + first/last timestamps."""
    return (
        wide.group_by("user_id")
        .agg(
            pl.len().alias("n_sent"),
            pl.col("opened").sum().alias("n_opened"),
            pl.col("verified_opened").sum().alias("n_verified_opened"),
            pl.col("clicked").sum().alias("n_clicked"),
            pl.col("actioned").sum().alias("n_actioned"),
            pl.col("sent_at").min().alias("first_sent_at"),
            pl.col("sent_at").max().alias("last_sent_at"),
            pl.when(pl.col("actioned"))
            .then(pl.col("sent_at"))
            .otherwise(None)
            .min()
            .alias("first_actioned_at"),
            pl.when(pl.col("actioned"))
            .then(pl.col("sent_at"))
            .otherwise(None)
            .max()
            .alias("last_actioned_at"),
        )
        .with_columns(
            (pl.col("n_actioned") > 0).alias("has_actioned"),
            (pl.col("n_clicked") > 0).alias("has_clicked"),
            (pl.col("n_opened") > 0).alias("has_opened"),
            (pl.col("n_verified_opened") > 0).alias("has_verified_opened"),
        )
    )


# ---------------------------------------------------------------------------
# Section builders.
# ---------------------------------------------------------------------------

def _section_tenure_proxy(users: pl.DataFrame) -> Section:
    days = (
        (pl.lit(ANCHOR_UTC) - pl.col("first_sent_at")).dt.total_days()
    )
    df = users.select(days.alias("days_since_first_send"))
    arr = df["days_since_first_send"].drop_nulls().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(arr, bins=60, color="#3b82f6", edgecolor="white")
    ax.set_xlabel("Days between first observed send and 2026-04-21 cutoff")
    ax.set_ylabel("Users")
    ax.set_title("Tenure proxy: days since first observed send (24-mo window)")
    ax.grid(axis="y", alpha=0.3)

    median = int(np.median(arr))
    p90 = int(np.percentile(arr, 90))
    capped = int((arr >= 700).sum())  # users present from window start
    prose = (
        "We do <b>not</b> have user-level signup date in the cached entities -- "
        "<code>created_at</code> in <code>actions_cache</code> is the signature "
        "timestamp, not user creation. This chart uses a <b>proxy</b>: days "
        "between the user's first observed send (within the 24-month "
        "<code>email_activities</code> v2 window) and the 2026-04-21 anchor.<br><br>"
        f"Median proxy tenure = <b>{median} days</b>, p90 = <b>{p90} days</b>. "
        f"<b>{capped:,}</b> users were already on the list at window start "
        "(>=700 days), so their true tenure is right-censored; we'd need an "
        "<code>actionnetwork_users</code> entity pull to read real signup dates."
    )
    return Section("tenure", "Q1. How long have users been members?", prose, _fig_to_b64(fig))


def _section_actions_distribution(users: pl.DataFrame) -> Section:
    arr = users["n_actioned"].to_numpy()
    actioners = arr[arr > 0]
    max_act = int(actioners.max()) if len(actioners) else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: discrete buckets so the zero bar doesn't crush everything else.
    edges = [0, 1, 2, 6, 21, 101, max(max_act + 1, 102)]
    labels = ["0", "1", "2-5", "6-20", "21-100", "100+"]
    counts = np.histogram(arr, bins=edges)[0]
    bars = axes[0].bar(labels, counts, color="#10b981", edgecolor="white")
    for b, c in zip(bars, counts):
        pct = 100 * c / len(arr)
        axes[0].text(
            b.get_x() + b.get_width() / 2, c, f"{c:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Attributed actions per user (bucket)")
    axes[0].set_ylabel("Users (log)")
    axes[0].set_title("All users (incl. zeros) by bucket")
    axes[0].set_ylim(top=counts.max() * 2)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: actioners only with log-spaced bins -> visible across the tail.
    if len(actioners):
        log_bins = np.unique(
            np.round(np.logspace(0, np.log10(max_act + 1), 35)).astype(int)
        )
        if log_bins[0] > 1:
            log_bins = np.insert(log_bins, 0, 1)
        axes[1].hist(actioners, bins=log_bins, color="#059669", edgecolor="white")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Attributed actions per user (log)")
        axes[1].set_ylabel("Users (log)")
        axes[1].set_title(f"Actioners only (n={len(actioners):,}); max={max_act}")
        axes[1].grid(axis="both", alpha=0.3)
    fig.tight_layout()

    pct_any = 100 * (arr > 0).mean()
    median_act = int(np.median(actioners)) if len(actioners) else 0
    p95 = int(np.percentile(actioners, 95)) if len(actioners) else 0
    p99 = int(np.percentile(actioners, 99)) if len(actioners) else 0
    prose = (
        f"<b>{pct_any:.1f}%</b> of users in the send universe ever take an "
        f"attributed action ({len(actioners):,} of {len(arr):,}). Among "
        f"actioners, median = <b>{median_act}</b>, p95 = <b>{p95}</b>, "
        f"p99 = <b>{p99}</b>, max = <b>{max_act}</b>. The right panel uses "
        "log-spaced bins on log-log axes so the heavy tail is legible -- "
        "a thin tail of users with hundreds of attributed actions sits "
        "alongside a mass at 1-2 actions."
    )
    return Section("actions_dist", "Q2. Distribution of attributed actions per user", prose, _fig_to_b64(fig))


def _section_lifetime_and_churn(users: pl.DataFrame) -> Section:
    actioners = users.filter(pl.col("n_actioned") > 0)
    lifetimes_d = (
        (pl.col("last_actioned_at") - pl.col("first_actioned_at")).dt.total_days()
    )
    days_since_last = (
        (pl.lit(ANCHOR_UTC) - pl.col("last_actioned_at")).dt.total_days()
    )
    df = actioners.select(
        lifetimes_d.alias("lifetime_days"),
        days_since_last.alias("days_since_last_action"),
    ).with_columns(
        (pl.col("days_since_last_action") > CHURN_DAYS).alias("churned"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    lt = df["lifetime_days"].to_numpy()
    axes[0].hist(lt, bins=60, color="#8b5cf6", edgecolor="white")
    axes[0].set_xlabel("Lifetime: last action - first action (days)")
    axes[0].set_ylabel("Actioners")
    axes[0].set_title("Action lifetime distribution")
    axes[0].grid(axis="y", alpha=0.3)

    dsla = df["days_since_last_action"].to_numpy()
    axes[1].hist(dsla, bins=60, color="#ef4444", edgecolor="white")
    axes[1].axvline(CHURN_DAYS, color="black", linestyle="--", label=f"{CHURN_DAYS}-day churn cutoff")
    axes[1].set_xlabel("Days since last action")
    axes[1].set_ylabel("Actioners")
    axes[1].set_title("Recency of last action")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()

    n_actioners = df.height
    n_churned = int(df["churned"].sum())
    pct_churned = 100 * n_churned / n_actioners if n_actioners else 0.0
    median_lt = int(np.median(lt))
    one_shot = int((lt == 0).sum())
    pct_one_shot = 100 * one_shot / n_actioners if n_actioners else 0.0
    prose = (
        f"Of <b>{n_actioners:,}</b> users who ever took an attributed action, "
        f"<b>{pct_churned:.1f}%</b> ({n_churned:,}) are churned (>{CHURN_DAYS} "
        f"days since last action as of 2026-04-21). Median lifetime is "
        f"<b>{median_lt} days</b>; <b>{pct_one_shot:.1f}%</b> are one-shot "
        f"actioners (lifetime = 0 days, only ever acted once). Lifetimes are "
        "right-censored by the 24-month window -- users who first acted near "
        "window start could appear longer-lived than they truly are."
    )
    return Section("lifetime", "Q3. User lifetime and churn (>90d since last action)", prose, _fig_to_b64(fig))


def _section_persona_buckets(users: pl.DataFrame) -> Section:
    """KMeans (k=4) on log-engagement vectors -> persona segmentation."""
    actioners = users.filter(pl.col("n_actioned") > 0)
    feats = (
        actioners.select(
            (pl.col("n_sent") + 1).log().alias("log_sent"),
            (pl.col("n_opened") + 1).log().alias("log_opened"),
            (pl.col("n_clicked") + 1).log().alias("log_clicked"),
            (pl.col("n_actioned") + 1).log().alias("log_actioned"),
        )
        .to_numpy()
    )
    if feats.shape[0] < 100:
        # Degenerate case -- not enough actioners to cluster.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Too few actioners to cluster.", ha="center", va="center")
        ax.set_axis_off()
        return Section("personas", "Q4. Distinct user buckets", "Insufficient data.", _fig_to_b64(fig))

    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=4, random_state=0, n_init=10)
    labels = km.fit_predict(X)

    # Per-cluster summary.
    summary = (
        actioners.with_columns(pl.Series("cluster", labels))
        .group_by("cluster")
        .agg(
            pl.len().alias("n_users"),
            pl.col("n_sent").mean().alias("avg_sent"),
            pl.col("n_opened").mean().alias("avg_opened"),
            pl.col("n_clicked").mean().alias("avg_clicked"),
            pl.col("n_actioned").mean().alias("avg_actioned"),
        )
        .sort("avg_actioned")
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["avg_sent", "avg_opened", "avg_clicked", "avg_actioned"]
    metric_labels = ["sent", "opened", "clicked", "actioned"]
    bar_w = 0.2
    x = np.arange(summary.height)
    for i, (m, lab) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + (i - 1.5) * bar_w, summary[m].to_numpy(), width=bar_w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{int(c)}\nn={int(n):,}" for c, n in zip(summary["cluster"], summary["n_users"])]
    )
    ax.set_ylabel("Mean events per user")
    ax.set_yscale("log")
    ax.set_title("4 personas via KMeans on log-engagement (actioners only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    rows = []
    for r in summary.iter_rows(named=True):
        rows.append(
            f"<li>Cluster {int(r['cluster'])} (<b>{int(r['n_users']):,}</b> users): "
            f"avg sent={r['avg_sent']:.1f}, opened={r['avg_opened']:.1f}, "
            f"clicked={r['avg_clicked']:.1f}, actioned={r['avg_actioned']:.1f}</li>"
        )
    bullets = "<ul>" + "".join(rows) + "</ul>"
    prose = (
        "Full pairwise user-overlap (Jaccard on the 2,183-D email vector) is "
        "infeasible at 982K users (~10<sup>11</sup> pairs). Instead we cluster "
        "actioners on log-transformed (sent, opened, clicked, actioned) with "
        "KMeans (k=4) -- a fast persona segmentation. Clusters are sorted by "
        "average action volume." + bullets +
        "Caveat: KMeans on 4 dims with rough log scaling is a first cut, not a "
        "definitive segmentation. A follow-up could use sparse user-email "
        "co-action matrices and matrix-factorization or graph clustering."
    )
    return Section("personas", "Q4. Are there distinct user buckets?", prose, _fig_to_b64(fig))


def _section_user_funnel(users: pl.DataFrame) -> Section:
    n = users.height
    counts = users.select(
        pl.lit(n).alias("sent"),
        pl.col("has_opened").sum().alias("opened"),
        pl.col("has_verified_opened").sum().alias("verified_opened"),
        pl.col("has_clicked").sum().alias("clicked"),
        pl.col("has_actioned").sum().alias("actioned"),
    ).row(0, named=True)

    stages = ["sent", "opened", "verified_opened", "clicked", "actioned"]
    vals = [counts[s] for s in stages]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(stages, vals, color=["#94a3b8", "#3b82f6", "#0ea5e9", "#f59e0b", "#10b981"])
    for b, v in zip(bars, vals):
        pct = 100 * v / n
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Users with >=1 event of stage")
    ax.set_title("User-level funnel (denominator = users with >=1 send)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3)

    open_rate = 100 * counts["opened"] / n
    click_rate_giv_open = 100 * counts["clicked"] / counts["opened"] if counts["opened"] else 0
    action_rate_giv_click = 100 * counts["actioned"] / counts["clicked"] if counts["clicked"] else 0
    prose = (
        f"<b>{open_rate:.1f}%</b> of recipients ever open. Conditional rates: "
        f"<b>P(click | opened)={click_rate_giv_open:.1f}%</b>, "
        f"<b>P(action | clicked)={action_rate_giv_click:.1f}%</b>. The bottom-"
        "of-funnel conditional is the leverage point -- moving more clickers "
        "into actioners is fewer users to influence than getting non-openers "
        "to open."
    )
    return Section("funnel", "Q5. User-level conversion funnel", prose, _fig_to_b64(fig))


def _section_inbox_volume(users: pl.DataFrame) -> Section:
    arr = users["n_sent"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(arr, bins=60, color="#0ea5e9", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Emails received (24 months)")
    ax.set_ylabel("Users (log)")
    ax.set_title("Inbox volume per user")
    ax.grid(axis="y", alpha=0.3)

    median = int(np.median(arr))
    p90 = int(np.percentile(arr, 90))
    p99 = int(np.percentile(arr, 99))
    prose = (
        f"Median user receives <b>{median}</b> emails over 24 months; "
        f"p90 = <b>{p90}</b>, p99 = <b>{p99}</b>. The send rate is non-uniform "
        "across users -- worth checking whether send volume itself predicts "
        "engagement (over-mailed users may be unsubscribing or going inactive)."
    )
    return Section("inbox", "Q6. Inbox volume per user", prose, _fig_to_b64(fig))


def _section_pareto(users: pl.DataFrame) -> Section:
    arr = np.sort(users["n_actioned"].to_numpy())[::-1]  # descending
    total = arr.sum()
    if total == 0:
        return Section("pareto", "Q7. Pareto check", "No actions in dataset.", "")
    cum = np.cumsum(arr) / total
    user_pct = np.arange(1, len(arr) + 1) / len(arr) * 100

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(user_pct, cum * 100, color="#dc2626", linewidth=2)
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", label="equality")
    for marker in [1, 10, 50]:
        idx = max(int(len(arr) * marker / 100) - 1, 0)
        ax.scatter([marker], [cum[idx] * 100], color="black", zorder=3)
        ax.annotate(f"top {marker}% -> {cum[idx]*100:.1f}% of actions",
                    (marker, cum[idx] * 100), textcoords="offset points",
                    xytext=(8, -8), fontsize=9)
    ax.set_xlabel("Top X% of users (sorted by action count)")
    ax.set_ylabel("Cumulative % of all attributed actions")
    ax.set_title("Pareto: are actions concentrated in a power-user minority?")
    ax.legend()
    ax.grid(alpha=0.3)

    def share(pct: float) -> float:
        idx = max(int(len(arr) * pct / 100) - 1, 0)
        return cum[idx] * 100

    prose = (
        f"Top <b>1%</b> of users carry <b>{share(1):.1f}%</b> of all "
        f"attributed actions; top <b>10%</b> carry <b>{share(10):.1f}%</b>; "
        f"top <b>50%</b> carry <b>{share(50):.1f}%</b>. The closer the curve "
        "bows toward the top-left, the more action volume is concentrated. "
        "This shapes content-routing strategy -- a small power-user cohort "
        "may be best served differently from the long tail."
    )
    return Section("pareto", "Q7. Pareto: where do actions concentrate?", prose, _fig_to_b64(fig))


def _section_rfm(users: pl.DataFrame) -> Section:
    actioners = users.filter(pl.col("n_actioned") > 0).select(
        (pl.lit(ANCHOR_UTC) - pl.col("last_actioned_at")).dt.total_days().alias("recency_days"),
        pl.col("n_actioned").alias("frequency"),
    )
    R = actioners["recency_days"].to_numpy()
    F = actioners["frequency"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    h = ax.hexbin(R, F, gridsize=40, mincnt=1, bins="log", cmap="viridis")
    ax.axvline(CHURN_DAYS, color="red", linestyle="--", label=f"{CHURN_DAYS}d churn")
    ax.set_xlabel("Recency: days since last action")
    ax.set_ylabel("Frequency: total attributed actions")
    ax.set_yscale("log")
    ax.set_title("RFM: recency x frequency (actioners only)")
    cb = fig.colorbar(h, ax=ax, label="users (log)")
    ax.legend()
    fig.tight_layout()

    n_total = len(R)
    n_active = int((R <= CHURN_DAYS).sum())
    n_active_freq = int(((R <= CHURN_DAYS) & (F >= 5)).sum())
    prose = (
        f"Of <b>{n_total:,}</b> actioners, <b>{n_active:,}</b> ({100*n_active/n_total:.1f}%) "
        f"are active (acted within {CHURN_DAYS} days). Of those, "
        f"<b>{n_active_freq:,}</b> are also high-frequency (>=5 actions). "
        "These four corners of the RFM grid are natural targeting buckets: "
        "recent-frequent (champions), recent-rare (new/casual), lapsed-frequent "
        "(reactivation candidates), lapsed-rare (likely lost)."
    )
    return Section("rfm", "Q8. RFM: recency x frequency", prose, _fig_to_b64(fig))


def _section_action_timing(wide: pl.DataFrame) -> Section:
    """When do actions happen? Use sent_at as proxy timing for actioned rows."""
    actioned = wide.filter(pl.col("actioned"))
    df = actioned.with_columns(
        pl.col("sent_at").dt.weekday().alias("dow"),       # 1=Mon..7=Sun
        pl.col("sent_at").dt.hour().alias("hour"),
    )
    dow_counts = (
        df.group_by("dow").agg(pl.len().alias("n")).sort("dow")
    )
    hour_counts = (
        df.group_by("hour").agg(pl.len().alias("n")).sort("hour")
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[0].bar(dow_labels, dow_counts["n"].to_numpy(), color="#7c3aed")
    axes[0].set_title("Actioned sends by day-of-week (UTC)")
    axes[0].set_ylabel("Actioned (email,user) pairs")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(hour_counts["hour"].to_numpy(), hour_counts["n"].to_numpy(), color="#7c3aed")
    axes[1].set_title("Actioned sends by hour-of-day (UTC)")
    axes[1].set_xlabel("Hour (UTC)")
    axes[1].set_ylabel("Actioned (email,user) pairs")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()

    prose = (
        "Caveat: <b>send time, not action time</b>. The wide frame's "
        "<code>sent_at</code> is the email send timestamp -- so this chart "
        "shows when emails that <i>get acted on</i> are typically sent, not "
        "when users click. To get true action-time, we'd join "
        "<code>attributed_actions.sign_time</code>; doable next pass."
    )
    return Section("timing", "Q9. Action timing (caveat: send-time proxy)", prose, _fig_to_b64(fig))


# ---------------------------------------------------------------------------
# "Are users distinct in what they action?" -- 5 complementary views.
# ---------------------------------------------------------------------------

def _section_heterogeneity_at_fixed_exposure(users: pl.DataFrame) -> Section:
    """Q10. Boxplot of per-user action_rate within fixed-exposure bins."""
    df = users.filter((pl.col("n_actioned") > 0) & (pl.col("n_sent") > 0))
    n_sent = df["n_sent"].to_numpy().astype(float)
    n_act = df["n_actioned"].to_numpy().astype(float)
    rate = n_act / n_sent

    edges = np.unique(np.quantile(n_sent, np.linspace(0, 1, 7)).astype(int))
    boxes, labels = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (n_sent >= lo) & (n_sent <= hi)
        if mask.sum() > 200:
            boxes.append(rate[mask])
            labels.append(f"{lo}-{hi}\nn={mask.sum():,}")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    bp = ax.boxplot(
        boxes, labels=labels, showfliers=False, patch_artist=True,
        medianprops=dict(color="black"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#a78bfa")
    ax.set_ylabel("Per-user action rate (n_actioned / n_sent)")
    ax.set_xlabel("Exposure bucket: emails received (24-mo window)")
    ax.set_title(
        "Q10. Among actioners, do users with the same inbox volume "
        "act at the same rate?"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    cvs = [(np.std(b) / np.mean(b)) if np.mean(b) > 0 else 0 for b in boxes]
    iqr_to_med = [
        (np.percentile(b, 75) - np.percentile(b, 25)) / max(np.median(b), 1e-9)
        for b in boxes
    ]
    prose = (
        "Among <b>actioners</b>, we bin users by emails-received and box-plot "
        "their per-user action rate within each bin. If users were "
        "interchangeable given exposure, each box would collapse to a thin "
        f"line. Median CV (std/mean) across bins = <b>{np.median(cvs):.2f}</b>; "
        f"median IQR/median = <b>{np.median(iqr_to_med):.2f}</b> -- a value "
        "around 1 means the middle 50% of users span a range comparable to "
        "the bin's median rate. This is the <b>necessary</b> condition for "
        "content-based routing: users must differ in appetite at fixed "
        "exposure. Pair with the email-email block structure (Q12) to test "
        "the <b>sufficient</b> condition."
    )
    return Section(
        "heterogeneity",
        "Q10. Action-rate heterogeneity at fixed exposure",
        prose,
        _fig_to_b64(fig),
    )


def _section_pairwise_jaccard(
    M: sp.csr_matrix, email_ids: np.ndarray
) -> Section:
    """Q11. Sampled user-pair Jaccard vs popularity-weighted null."""
    rng = np.random.default_rng(0)
    counts = np.asarray(M.sum(axis=1)).flatten()
    eligible = np.where(counts >= 2)[0]
    sample_size = min(5000, len(eligible))
    sample = rng.choice(eligible, size=sample_size, replace=False)

    Ms = M[sample]
    inter = (Ms @ Ms.T).toarray().astype(np.float32)
    sums = counts[sample]
    union = sums[:, None] + sums[None, :] - inter
    J_obs = np.where(union > 0, inter / union, 0)
    iu = np.triu_indices(sample_size, k=1)
    obs = J_obs[iu]

    pop = np.asarray(M.sum(axis=0)).flatten()
    p = pop / pop.sum()
    n_emails = M.shape[1]
    null_rows: list[int] = []
    null_cols: list[int] = []
    for i, u in enumerate(sample):
        k = int(counts[u])
        chosen = rng.choice(n_emails, size=k, replace=False, p=p)
        null_rows.extend([i] * k)
        null_cols.extend(chosen.tolist())
    Mn = sp.csr_matrix(
        (np.ones(len(null_rows), dtype=np.float32), (null_rows, null_cols)),
        shape=(sample_size, n_emails),
    )
    inter_n = (Mn @ Mn.T).toarray().astype(np.float32)
    union_n = sums[:, None] + sums[None, :] - inter_n
    J_null = np.where(union_n > 0, inter_n / union_n, 0)
    null = J_null[iu]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    upper = max(np.percentile(obs, 99.5), np.percentile(null, 99.5))
    bins = np.linspace(0, upper, 60)
    ax.hist(
        null, bins=bins, alpha=0.55, color="#94a3b8", density=True,
        label=f"Null (popularity-weighted, mean={null.mean():.4f})",
    )
    ax.hist(
        obs, bins=bins, alpha=0.6, color="#dc2626", density=True,
        label=f"Observed (mean={obs.mean():.4f})",
    )
    ax.set_xlabel("Pairwise Jaccard between user action vectors")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Q11. User-pair similarity vs popularity-weighted null "
        f"(n={sample_size:,} actioners, {len(obs):,} pairs)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lift = obs.mean() / null.mean() if null.mean() > 0 else float("nan")
    prose = (
        f"Sampled <b>{sample_size:,}</b> users with >=2 actions; computed "
        f"pairwise Jaccard on their action vectors over emails. The <b>null</b> "
        "preserves each user's action count but draws emails weighted by "
        "global popularity -- so any deviation from null reflects "
        "content-aligned co-action <i>beyond</i> raw popularity. "
        f"Observed mean = <b>{obs.mean():.4f}</b>, null mean = "
        f"<b>{null.mean():.4f}</b> (lift = <b>{lift:.2f}x</b>). A "
        "right-shifted observed distribution = users co-act on the same "
        "content; tightly-overlapping observed and null = users are "
        "interchangeable given email popularity."
    )
    return Section(
        "user_pair_jaccard",
        "Q11. Pairwise user Jaccard vs popularity-weighted null",
        prose,
        _fig_to_b64(fig),
    )


def _section_email_email_jaccard(
    M: sp.csr_matrix, email_ids: np.ndarray
) -> Section:
    """Q12. Email x email Jaccard heatmap, hierarchically reordered."""
    co = (M.T @ M).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, 0)

    dist = 1.0 - J
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    cdist = squareform(dist, checks=False)
    Z = linkage(cdist, method="average")
    order = leaves_list(Z)
    J_ord = J[np.ix_(order, order)]

    iu = np.triu_indices(J.shape[0], k=1)
    off_diag = J[iu]
    cap = max(np.percentile(off_diag, 99), 0.05)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(J_ord, cmap="magma", vmin=0, vmax=cap, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Q12. Email x email Jaccard ({len(email_ids):,} emails, "
        f"avg-linkage reorder; vmax=p99={cap:.3f})"
    )
    fig.colorbar(im, ax=ax, label="Jaccard (audience overlap)")
    fig.tight_layout()

    high_pairs = int((off_diag >= 0.30).sum())
    mod_pairs = int((off_diag >= 0.10).sum())
    prose = (
        "Each cell is the Jaccard between two emails computed on their "
        "actioner sets (users who actioned both / users who actioned either). "
        "Hierarchically reordered (average linkage on 1-Jaccard). "
        f"Off-diagonal mean = <b>{off_diag.mean():.4f}</b>, p99 = "
        f"<b>{np.percentile(off_diag, 99):.4f}</b>. Pairs with "
        f"Jaccard >= 0.10: <b>{mod_pairs:,}</b>; >= 0.30: "
        f"<b>{high_pairs:,}</b>. Visible block-diagonal patches = thematic "
        "email clusters that share an audience -- those blocks are the "
        "natural targets for content-based routing. A uniformly dim heatmap "
        "= little audience-sharing structure across emails."
    )
    return Section(
        "email_email_jaccard",
        "Q12. Email x email audience-overlap heatmap",
        prose,
        _fig_to_b64(fig),
    )


def _section_user_svd_scatter(M: sp.csr_matrix) -> Section:
    """Q13. TruncatedSVD k=3, scatter on volume axis vs content axes."""
    counts = np.asarray(M.sum(axis=1)).flatten()
    svd = TruncatedSVD(n_components=3, random_state=0)
    U = svd.fit_transform(M)

    rng = np.random.default_rng(0)
    n = U.shape[0]
    sample_size = min(50_000, n)
    idx = rng.choice(n, size=sample_size, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    norm = mcolors.LogNorm(vmin=max(counts[idx].min(), 1), vmax=counts[idx].max())
    sc1 = axes[0].scatter(
        U[idx, 0], U[idx, 1], s=2, c=counts[idx], cmap="viridis",
        alpha=0.4, norm=norm,
    )
    axes[0].set_xlabel("SVD-0 (typically aligned with action volume)")
    axes[0].set_ylabel("SVD-1")
    axes[0].set_title("Components 0 vs 1, color = n_actioned (log)")
    axes[0].grid(alpha=0.3)
    fig.colorbar(sc1, ax=axes[0], label="n_actioned")

    sc2 = axes[1].scatter(
        U[idx, 1], U[idx, 2], s=2, c=counts[idx], cmap="viridis",
        alpha=0.4, norm=norm,
    )
    axes[1].set_xlabel("SVD-1 (content axis)")
    axes[1].set_ylabel("SVD-2 (content axis)")
    axes[1].set_title("Components 1 vs 2: content-only (no volume axis)")
    axes[1].grid(alpha=0.3)
    fig.colorbar(sc2, ax=axes[1], label="n_actioned")
    fig.tight_layout()

    ve = svd.explained_variance_ratio_
    prose = (
        f"TruncatedSVD with k=3 on the actioner x email matrix "
        f"({M.shape[0]:,} actioners x {M.shape[1]:,} emails). Variance ratio "
        f"per component: {ve[0]:.3f}, {ve[1]:.3f}, {ve[2]:.3f}. Component 0 "
        "typically correlates with action volume (its color stripe should "
        "be visible in the left panel). The right panel uses components 1 "
        "and 2 -- the <b>content-only</b> axes. Distinct clouds in the right "
        "panel = users segment by which emails they action, not just how "
        "many. A single blob = no content-based segmentation."
    )
    return Section(
        "user_svd",
        "Q13. SVD scatter: do user clouds separate on content axes?",
        prose,
        _fig_to_b64(fig),
    )


def _section_topk_cohort_retention(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5
) -> Section:
    """Q14. Cohort retention to top-K nearest emails vs random baseline."""
    co = (M.T @ M).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -1.0)

    n_emails = J.shape[0]
    topk_idx = np.argpartition(-J, k, axis=1)[:, :k]

    adj = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj[e, topk_idx[e]] = 1.0
    adj = adj.tocsr()

    G = M @ adj.T
    G_bool = (G > 0).astype(np.float32)
    denom = np.asarray(M.sum(axis=0)).flatten()
    numer = np.asarray(M.multiply(G_bool).sum(axis=0)).flatten()
    retention = np.where(denom > 0, numer / denom, np.nan)

    rng = np.random.default_rng(0)
    rand_neighbors = rng.integers(0, n_emails - 1, size=(n_emails, k))
    self_mask = rand_neighbors >= np.arange(n_emails)[:, None]
    rand_neighbors = rand_neighbors + self_mask.astype(int)

    adj_r = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj_r[e, rand_neighbors[e]] = 1.0
    adj_r = adj_r.tocsr()
    G_r = M @ adj_r.T
    G_r_bool = (G_r > 0).astype(np.float32)
    numer_r = np.asarray(M.multiply(G_r_bool).sum(axis=0)).flatten()
    rand_retention = np.where(denom > 0, numer_r / denom, np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0, 1, 50)
    ax.hist(
        rand_retention[~np.isnan(rand_retention)], bins=bins, alpha=0.55,
        color="#94a3b8",
        label=f"Random K={k} emails (mean={np.nanmean(rand_retention):.3f})",
    )
    ax.hist(
        retention[~np.isnan(retention)], bins=bins, alpha=0.6, color="#0ea5e9",
        label=f"Top K={k} nearest (mean={np.nanmean(retention):.3f})",
    )
    ax.set_xlabel(
        f"% of email's actioners who also act on any of its top-{k} neighbors"
    )
    ax.set_ylabel("Emails")
    ax.set_title(
        "Q14. Cohort retention: do email audiences carry over to similar emails?"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lift = np.nanmean(retention) / max(np.nanmean(rand_retention), 1e-9)
    prose = (
        f"For each email, found the top-{k} most similar emails by user-action "
        "Jaccard. Cohort retention = % of the email's actioners who also "
        f"actioned at least one of those {k} neighbors. Compared against a "
        f"baseline of {k} <i>random</i> emails. Mean retention (top-K) = "
        f"<b>{np.nanmean(retention):.3f}</b> vs random = "
        f"<b>{np.nanmean(rand_retention):.3f}</b> (lift = <b>{lift:.2f}x</b>). "
        "High lift = audiences carry over to content-similar emails (so "
        "routing on content similarity should retain the audience). Lift "
        "near 1 = audiences are essentially random across emails -- "
        "content-based routing wouldn't help."
    )
    return Section(
        "cohort_retention",
        "Q14. Top-K cohort retention vs random baseline",
        prose,
        _fig_to_b64(fig),
    )


def _section_topk_retention_by_tier(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5
) -> Section:
    """Q15. Same metric as Q14, stratified by user volume tier.

    Users with n_actioned == 1 cannot have any neighbor co-action by
    construction, so we restrict to actioners with >=2 actions before
    splitting into tiers. Top-K neighbors are still computed on the FULL
    matrix (content similarity is a property of the emails, not the tier).
    """
    co = (M.T @ M).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -1.0)
    n_emails = J.shape[0]

    topk = np.argpartition(-J, k, axis=1)[:, :k]
    adj = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj[e, topk[e]] = 1.0
    adj = adj.tocsr()

    rng = np.random.default_rng(0)
    rand_n = rng.integers(0, n_emails - 1, size=(n_emails, k))
    self_mask = rand_n >= np.arange(n_emails)[:, None]
    rand_n = rand_n + self_mask.astype(int)
    adj_r = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj_r[e, rand_n[e]] = 1.0
    adj_r = adj_r.tocsr()

    counts = np.asarray(M.sum(axis=1)).flatten()
    eligible_mask = counts >= 2
    M_e = M[eligible_mask]
    counts_e = counts[eligible_mask]
    qs = np.quantile(counts_e, [0.25, 0.50, 0.75])
    tier_edges = [2, int(qs[0]) + 1, int(qs[1]) + 1, int(qs[2]) + 1, int(counts_e.max()) + 1]
    # Make edges strictly increasing in case quartiles collide.
    for i in range(1, len(tier_edges)):
        if tier_edges[i] <= tier_edges[i - 1]:
            tier_edges[i] = tier_edges[i - 1] + 1

    labels: list[str] = []
    means_topk: list[float] = []
    means_rand: list[float] = []
    lifts: list[float] = []
    n_per_tier: list[int] = []

    for i in range(4):
        lo, hi = tier_edges[i], tier_edges[i + 1]
        m = (counts_e >= lo) & (counts_e < hi)
        if m.sum() == 0:
            continue
        Mt = M_e[m]
        denom = np.asarray(Mt.sum(axis=0)).flatten()
        Gt = Mt @ adj.T
        numer = np.asarray(Mt.multiply((Gt > 0).astype(np.float32)).sum(axis=0)).flatten()
        retention = np.where(denom > 0, numer / denom, np.nan)

        Gt_r = Mt @ adj_r.T
        numer_r = np.asarray(
            Mt.multiply((Gt_r > 0).astype(np.float32)).sum(axis=0)
        ).flatten()
        rand_ret = np.where(denom > 0, numer_r / denom, np.nan)

        m_top = float(np.nanmean(retention))
        m_rnd = float(np.nanmean(rand_ret))
        labels.append(f"Q{i+1}\nn_act in [{lo}-{hi-1}]\nusers={int(m.sum()):,}")
        means_topk.append(m_top)
        means_rand.append(m_rnd)
        lifts.append(m_top / max(m_rnd, 1e-9))
        n_per_tier.append(int(m.sum()))
        print(
            f"  tier Q{i+1} n_act in [{lo}-{hi-1}]: users={m.sum():,}  "
            f"top-K={m_top:.4f}  random={m_rnd:.4f}  lift={m_top/max(m_rnd,1e-9):.2f}x"
        )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, means_rand, width=w, color="#94a3b8", label=f"random K={k}")
    ax.bar(x + w / 2, means_topk, width=w, color="#0ea5e9", label=f"top K={k}")
    for i in range(len(labels)):
        ax.text(
            i + w / 2, means_topk[i],
            f"{means_topk[i]:.2f}\n({lifts[i]:.2f}x)",
            ha="center", va="bottom", fontsize=9,
        )
        ax.text(
            i - w / 2, means_rand[i], f"{means_rand[i]:.2f}",
            ha="center", va="bottom", fontsize=9, color="#475569",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean cohort retention (per email, averaged across emails)")
    ax.set_title(
        f"Q15. Top-{k} retention by user volume tier (actioners with >=2 actions)"
    )
    ax.set_ylim(0, max(means_topk) * 1.18)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    rows_html = "".join(
        f"<li>Q{i+1} (n_actioned in [{tier_edges[i]}-{tier_edges[i+1]-1}], "
        f"<b>{n_per_tier[i]:,}</b> users): "
        f"top-K={means_topk[i]:.3f}, random={means_rand[i]:.3f}, "
        f"<b>lift={lifts[i]:.2f}x</b></li>"
        for i in range(len(labels))
    )
    prose = (
        "Stratifies Q14 by user-volume quartile (computed over actioners with "
        ">=2 actions; one-shot actioners are excluded because they cannot "
        f"co-act on a neighbor by construction). The top-{k} neighbor structure "
        "is still computed on the full matrix.<ul>" + rows_html + "</ul>"
        "If lift is roughly constant across tiers, the content-similarity "
        "signal works at every engagement level. If lift is concentrated in "
        "high-volume tiers, the 2.6x global lift from Q14 is power-user "
        "driven and content-routing is less actionable for casual users."
    )
    return Section(
        "retention_by_tier",
        "Q15. Top-K retention by user volume tier",
        prose,
        _fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Cosine-similarity helpers + Q16-Q18 sections.
# ---------------------------------------------------------------------------

def _build_topk_adj(scores: np.ndarray, k: int) -> sp.csr_matrix:
    """Top-K-neighbor adjacency from a similarity matrix (diag <= 0)."""
    n = scores.shape[0]
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    adj = sp.lil_matrix((n, n), dtype=np.float32)
    for e in range(n):
        adj[e, topk[e]] = 1.0
    return adj.tocsr()


def _cohort_retention(M: sp.csr_matrix, adj: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per-email retention, per-email actioner count)."""
    G = M @ adj.T
    denom = np.asarray(M.sum(axis=0)).flatten()
    numer = np.asarray(M.multiply((G > 0).astype(np.float32)).sum(axis=0)).flatten()
    return np.where(denom > 0, numer / denom, np.nan), denom


def _random_topk_adj(n_emails: int, k: int, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, n_emails - 1, size=(n_emails, k))
    self_mask = rand >= np.arange(n_emails)[:, None]
    rand = rand + self_mask.astype(int)
    adj = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj[e, rand[e]] = 1.0
    return adj.tocsr()


def _jaccard_topk_adj(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    co = (M.T @ M).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -1.0)
    return _build_topk_adj(J, k)


def _aligned_cosine_setup(
    M: sp.csr_matrix, email_ids: np.ndarray, agg_version: int = 3
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Filter M and email_ids to those with embeddings; return cosine top-K input.

    Returns (M_aligned, email_ids_aligned, S_cos) where S_cos[i,j] = cosine
    similarity (vectors are pre-L2-normalized in aggregate_emails v2/v3/v4)
    with diagonal forced to -inf.
    """
    df = (
        agg_cache.get(agg_version)
        .filter(pl.col("email_id").is_in(email_ids.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id = {row["email_id"]: row["dense_vector"] for row in df.iter_rows(named=True)}
    keep_mask = np.array([e in by_id for e in email_ids])
    n_drop = int((~keep_mask).sum())
    if n_drop:
        print(f"  cosine: dropping {n_drop} emails missing embeddings (of {len(email_ids)})")
    keep_idx = np.where(keep_mask)[0]
    email_ids_a = email_ids[keep_idx]
    V = np.asarray(
        [by_id[int(email_ids[i])] for i in keep_idx], dtype=np.float32
    )
    M_a = M[:, keep_idx].tocsr()
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    return M_a, email_ids_a, S


def _section_topk_cohort_retention_cosine(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5
) -> Section:
    """Q16. Same as Q14 but neighbors picked by cosine similarity of embeddings."""
    M_a, email_ids_a, S = _aligned_cosine_setup(M, email_ids)
    n = S.shape[0]
    adj_cos = _build_topk_adj(S, k)
    adj_rand = _random_topk_adj(n, k)
    ret_cos, denom = _cohort_retention(M_a, adj_cos)
    ret_rand, _ = _cohort_retention(M_a, adj_rand)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0, 1, 50)
    ax.hist(
        ret_rand[~np.isnan(ret_rand)], bins=bins, alpha=0.55, color="#94a3b8",
        label=f"Random K={k} (mean={np.nanmean(ret_rand):.3f})",
    )
    ax.hist(
        ret_cos[~np.isnan(ret_cos)], bins=bins, alpha=0.6, color="#16a34a",
        label=f"Top-K cosine (mean={np.nanmean(ret_cos):.3f})",
    )
    ax.set_xlabel(f"% of email's actioners who also act on any of its top-{k} cosine-nearest emails")
    ax.set_ylabel("Emails")
    ax.set_title(
        "Q16. Cohort retention with cosine-similarity neighbors (content embeddings)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lift = np.nanmean(ret_cos) / max(np.nanmean(ret_rand), 1e-9)
    prose = (
        f"Identical retention metric to Q14, but neighbors are picked by "
        f"<b>cosine similarity of OpenAI <code>text-embedding-3-small</code> "
        f"vectors</b> (1536-d, L2-normalized; from <code>aggregate_emails</code> "
        f"v3) instead of behavior-Jaccard. "
        f"Mean retention (top-{k} cosine) = "
        f"<b>{np.nanmean(ret_cos):.3f}</b> vs random = "
        f"<b>{np.nanmean(ret_rand):.3f}</b> (lift = <b>{lift:.2f}x</b>). "
        f"Pure-content nearest-neighbor: only the email's body + subject is "
        f"used, no behavior data leaks into the neighbor selection. "
        f"Compared to Q14's behavior-Jaccard neighbors -- if cosine matches "
        f"or beats Jaccard, content-only routing is viable. If cosine is "
        f"much weaker, you need behavior signal to predict audience overlap."
    )
    return Section(
        "cohort_retention_cosine",
        "Q16. Top-K cohort retention (cosine neighbors)",
        prose,
        _fig_to_b64(fig),
    )


def _section_topk_retention_by_tier_cosine(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5
) -> Section:
    """Q17. Cosine version of Q15: per-tier retention with cosine top-K."""
    M_a, email_ids_a, S = _aligned_cosine_setup(M, email_ids)
    n = S.shape[0]
    adj_cos = _build_topk_adj(S, k)
    adj_rand = _random_topk_adj(n, k)

    counts = np.asarray(M_a.sum(axis=1)).flatten()
    eligible = counts >= 2
    M_e = M_a[eligible]
    counts_e = counts[eligible]
    qs = np.quantile(counts_e, [0.25, 0.50, 0.75])
    edges = [2, int(qs[0]) + 1, int(qs[1]) + 1, int(qs[2]) + 1, int(counts_e.max()) + 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1

    labels: list[str] = []
    means_cos: list[float] = []
    means_rnd: list[float] = []
    lifts: list[float] = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (counts_e >= lo) & (counts_e < hi)
        if m.sum() == 0:
            continue
        Mt = M_e[m]
        ret_c, _ = _cohort_retention(Mt, adj_cos)
        ret_r, _ = _cohort_retention(Mt, adj_rand)
        m_c = float(np.nanmean(ret_c))
        m_r = float(np.nanmean(ret_r))
        labels.append(f"Q{i+1}\nn_act in [{lo}-{hi-1}]\nusers={int(m.sum()):,}")
        means_cos.append(m_c)
        means_rnd.append(m_r)
        lifts.append(m_c / max(m_r, 1e-9))
        print(
            f"  cosine tier Q{i+1} n_act in [{lo}-{hi-1}]: users={m.sum():,}  "
            f"top-K={m_c:.4f}  random={m_r:.4f}  lift={m_c/max(m_r,1e-9):.2f}x"
        )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, means_rnd, width=w, color="#94a3b8", label=f"random K={k}")
    ax.bar(x + w / 2, means_cos, width=w, color="#16a34a", label=f"top-K cosine")
    for i in range(len(labels)):
        ax.text(
            i + w / 2, means_cos[i],
            f"{means_cos[i]:.2f}\n({lifts[i]:.2f}x)",
            ha="center", va="bottom", fontsize=9,
        )
        ax.text(
            i - w / 2, means_rnd[i], f"{means_rnd[i]:.2f}",
            ha="center", va="bottom", fontsize=9, color="#475569",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean cohort retention (per email, averaged across emails)")
    ax.set_title(f"Q17. Cosine top-{k} retention by user volume tier")
    ax.set_ylim(0, max(means_cos) * 1.18)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    rows_html = "".join(
        f"<li>Q{i+1}: cosine top-K={means_cos[i]:.3f}, random={means_rnd[i]:.3f}, "
        f"<b>lift={lifts[i]:.2f}x</b></li>"
        for i in range(len(labels))
    )
    prose = (
        "Cosine-neighbor version of Q15. Side-by-side with Q15 (behavior-Jaccard) "
        "in Q18 below. Per-tier breakdown:<ul>" + rows_html + "</ul>"
    )
    return Section(
        "retention_by_tier_cosine",
        "Q17. Cosine top-K retention by user volume tier",
        prose,
        _fig_to_b64(fig),
    )


def _section_jaccard_vs_cosine_compare(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5
) -> Section:
    """Q18. Head-to-head: Jaccard top-K vs cosine top-K, global + per tier."""
    M_a, email_ids_a, S = _aligned_cosine_setup(M, email_ids)
    n = S.shape[0]
    adj_cos = _build_topk_adj(S, k)
    adj_jac = _jaccard_topk_adj(M_a, k)
    adj_rand = _random_topk_adj(n, k)

    counts = np.asarray(M_a.sum(axis=1)).flatten()
    eligible = counts >= 2
    M_e = M_a[eligible]
    counts_e = counts[eligible]
    qs = np.quantile(counts_e, [0.25, 0.50, 0.75])
    edges = [2, int(qs[0]) + 1, int(qs[1]) + 1, int(qs[2]) + 1, int(counts_e.max()) + 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1

    groups: list[tuple[str, sp.csr_matrix]] = [("All (>=2)", M_e)]
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (counts_e >= lo) & (counts_e < hi)
        if m.sum() > 0:
            groups.append((f"Q{i+1} [{lo}-{hi-1}]", M_e[m]))

    rets_jac: list[float] = []
    rets_cos: list[float] = []
    rets_rnd: list[float] = []
    labels: list[str] = []
    for label, sub in groups:
        rj, _ = _cohort_retention(sub, adj_jac)
        rc, _ = _cohort_retention(sub, adj_cos)
        rr, _ = _cohort_retention(sub, adj_rand)
        labels.append(label)
        rets_jac.append(float(np.nanmean(rj)))
        rets_cos.append(float(np.nanmean(rc)))
        rets_rnd.append(float(np.nanmean(rr)))

    # Compute neighbor-set overlap between Jaccard and cosine.
    jac_arr = adj_jac.toarray().astype(bool)
    cos_arr = adj_cos.toarray().astype(bool)
    overlap = (jac_arr & cos_arr).sum(axis=1) / k  # fraction of common neighbors per email
    mean_overlap = float(overlap.mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))
    w = 0.27
    axes[0].bar(x - w, rets_rnd, width=w, color="#94a3b8", label=f"random K={k}")
    axes[0].bar(x,     rets_jac, width=w, color="#0ea5e9", label="Jaccard top-K")
    axes[0].bar(x + w, rets_cos, width=w, color="#16a34a", label="cosine top-K")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean cohort retention")
    axes[0].set_title("Absolute retention: Jaccard vs cosine vs random")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    lifts_jac = [j / max(r, 1e-9) for j, r in zip(rets_jac, rets_rnd)]
    lifts_cos = [c / max(r, 1e-9) for c, r in zip(rets_cos, rets_rnd)]
    axes[1].bar(x - w/2, lifts_jac, width=w, color="#0ea5e9", label="Jaccard lift")
    axes[1].bar(x + w/2, lifts_cos, width=w, color="#16a34a", label="cosine lift")
    for i in range(len(labels)):
        axes[1].text(
            i - w/2, lifts_jac[i], f"{lifts_jac[i]:.1f}x",
            ha="center", va="bottom", fontsize=8.5,
        )
        axes[1].text(
            i + w/2, lifts_cos[i], f"{lifts_cos[i]:.1f}x",
            ha="center", va="bottom", fontsize=8.5,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Lift over random K")
    axes[1].set_title("Lift vs random baseline (higher = more informative)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Q18. Behavior-Jaccard vs content-cosine top-{k} neighbors")
    fig.tight_layout()

    print(f"  Q18 neighbor-set overlap (mean fraction of shared top-{k} per email) = {mean_overlap:.3f}")

    rows_html = "".join(
        f"<li>{labels[i]}: Jaccard={rets_jac[i]:.3f} ({lifts_jac[i]:.2f}x), "
        f"cosine={rets_cos[i]:.3f} ({lifts_cos[i]:.2f}x)</li>"
        for i in range(len(labels))
    )
    prose = (
        f"Same retention metric as Q14/Q16; both neighbor sources evaluated on "
        f"the same email set ({n:,} emails with embeddings). Mean fraction of "
        f"top-{k} neighbors that both methods agree on per email: "
        f"<b>{mean_overlap:.3f}</b> -- closer to 0 = methods disagree about "
        f"who's similar to whom; closer to 1 = behavior and content embeddings "
        f"see the same structure.<ul>" + rows_html + "</ul>"
        "Reading this:"
        "<ul>"
        "<li>If <b>Jaccard >> cosine</b>: behavioral co-action captures "
        "structure that text content alone misses (e.g., timing, sender "
        "trust, off-topic recipients). Routing needs behavior data.</li>"
        "<li>If <b>cosine >= Jaccard</b>: content embeddings carry the "
        "audience-overlap signal. Pure content routing (no behavior data) "
        "is viable -- great for cold-start emails.</li>"
        "<li>If <b>both lifts collapse for power users</b> (Q4 column): the "
        "high-volume tier is naturally easy to retain via any neighbor set; "
        "the differentiator is the long tail.</li>"
        "</ul>"
    )
    return Section(
        "jaccard_vs_cosine",
        "Q18. Head-to-head: behavior-Jaccard vs content-cosine neighbors",
        prose,
        _fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Sparse-embedding helpers (TF-IDF + BM25).
# ---------------------------------------------------------------------------

def _load_sparse_vectors(
    email_ids: np.ndarray, col: str, agg_version: int = 3,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Load a sparse vector column as CSR (n_emails x vocab).

    Returns (V_csr, email_ids_aligned, keep_idx_into_input).
    """
    df = (
        agg_cache.get(agg_version)
        .filter(pl.col("email_id").is_in(email_ids.tolist()))
        .select("email_id", col)
    )
    by_id = {row["email_id"]: row[col] for row in df.iter_rows(named=True)}
    keep_mask = np.array([int(e) in by_id for e in email_ids])
    n_drop = int((~keep_mask).sum())
    if n_drop:
        print(f"  {col}: dropping {n_drop} emails missing vector (of {len(email_ids)})")
    keep_idx = np.where(keep_mask)[0]

    rows_chunks: list[np.ndarray] = []
    cols_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    max_col = 0
    for new_row, i in enumerate(keep_idx):
        s = by_id[int(email_ids[i])]
        if not isinstance(s, dict):
            continue
        idxs = s.get("indices")
        vals = s.get("values")
        if idxs is None or len(idxs) == 0:
            continue
        idxs_a = np.asarray(idxs, dtype=np.int64)
        vals_a = np.asarray(vals, dtype=np.float32)
        rows_chunks.append(np.full(len(idxs_a), new_row, dtype=np.int64))
        cols_chunks.append(idxs_a)
        data_chunks.append(vals_a)
        if len(idxs_a):
            max_col = max(max_col, int(idxs_a.max()))
    rows_arr = np.concatenate(rows_chunks) if rows_chunks else np.array([], dtype=np.int64)
    cols_arr = np.concatenate(cols_chunks) if cols_chunks else np.array([], dtype=np.int64)
    data_arr = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)
    n_rows = len(keep_idx)
    n_cols = max_col + 1
    V = sp.csr_matrix(
        (data_arr, (rows_arr, cols_arr)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )
    return V, email_ids[keep_idx], keep_idx


def _aligned_sparse_setup(
    M: sp.csr_matrix, email_ids: np.ndarray, col: str, agg_version: int = 3,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Same shape as _aligned_cosine_setup but for a sparse-vector column."""
    V, email_ids_a, keep_idx = _load_sparse_vectors(email_ids, col, agg_version)
    M_a = M[:, keep_idx].tocsr()
    S = (V @ V.T).toarray().astype(np.float32)
    np.fill_diagonal(S, -np.inf)
    return M_a, email_ids_a, S


def _retention_overall_and_tiers(
    M_a: sp.csr_matrix, S: np.ndarray, k: int,
) -> dict:
    n = S.shape[0]
    adj = _build_topk_adj(S, k)
    adj_rand = _random_topk_adj(n, k)
    ret_top, _ = _cohort_retention(M_a, adj)
    ret_rnd, _ = _cohort_retention(M_a, adj_rand)

    counts = np.asarray(M_a.sum(axis=1)).flatten()
    eligible = counts >= 2
    M_e = M_a[eligible]
    counts_e = counts[eligible]
    qs = np.quantile(counts_e, [0.25, 0.50, 0.75])
    edges = [2, int(qs[0]) + 1, int(qs[1]) + 1, int(qs[2]) + 1, int(counts_e.max()) + 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1

    tiers: list[dict] = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (counts_e >= lo) & (counts_e < hi)
        if m.sum() == 0:
            continue
        Mt = M_e[m]
        r_t, _ = _cohort_retention(Mt, adj)
        r_r, _ = _cohort_retention(Mt, adj_rand)
        tiers.append({
            "lo": lo, "hi": hi - 1, "n": int(m.sum()),
            "top": float(np.nanmean(r_t)),
            "rnd": float(np.nanmean(r_r)),
        })

    return {
        "overall_top": float(np.nanmean(ret_top)),
        "overall_rnd": float(np.nanmean(ret_rnd)),
        "ret_top": ret_top,
        "ret_rnd": ret_rnd,
        "tiers": tiers,
        "adj": adj,
    }


def _make_method_section(
    method_label: str, method_color: str, slug: str, qnum: int,
    M_a: sp.csr_matrix, S: np.ndarray, k: int, extra_prose: str = "",
) -> Section:
    res = _retention_overall_and_tiers(M_a, S, k)
    overall_lift = res["overall_top"] / max(res["overall_rnd"], 1e-9)
    print(f"  {method_label}: overall top-K={res['overall_top']:.4f} "
          f"random={res['overall_rnd']:.4f} lift={overall_lift:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 50)
    axes[0].hist(
        res["ret_rnd"][~np.isnan(res["ret_rnd"])], bins=bins, alpha=0.55,
        color="#94a3b8",
        label=f"Random K={k} (mean={res['overall_rnd']:.3f})",
    )
    axes[0].hist(
        res["ret_top"][~np.isnan(res["ret_top"])], bins=bins, alpha=0.6,
        color=method_color,
        label=f"{method_label} top-K (mean={res['overall_top']:.3f})",
    )
    axes[0].set_xlabel(
        f"% of email's actioners who also act on top-{k} {method_label} neighbors"
    )
    axes[0].set_ylabel("Emails")
    axes[0].set_title(f"Overall (lift = {overall_lift:.2f}x)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    tiers = res["tiers"]
    x = np.arange(len(tiers))
    w = 0.4
    means_top = [t["top"] for t in tiers]
    means_rnd = [t["rnd"] for t in tiers]
    lifts = [t["top"] / max(t["rnd"], 1e-9) for t in tiers]
    labels_t = [f"Q{i+1}\nn_act in [{t['lo']}-{t['hi']}]\nusers={t['n']:,}"
                for i, t in enumerate(tiers)]
    axes[1].bar(x - w / 2, means_rnd, width=w, color="#94a3b8", label="random")
    axes[1].bar(x + w / 2, means_top, width=w, color=method_color,
                label=f"{method_label} top-K")
    for i, t in enumerate(tiers):
        axes[1].text(i + w / 2, t["top"], f"{t['top']:.2f}\n({lifts[i]:.2f}x)",
                     ha="center", va="bottom", fontsize=8.5)
        axes[1].text(i - w / 2, t["rnd"], f"{t['rnd']:.2f}",
                     ha="center", va="bottom", fontsize=8.5, color="#475569")
        print(f"    tier Q{i+1} [{t['lo']}-{t['hi']}]: "
              f"top-K={t['top']:.4f} random={t['rnd']:.4f} lift={lifts[i]:.2f}x")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_t, fontsize=8.5)
    axes[1].set_ylabel("Mean retention")
    axes[1].set_title("Per user-volume tier")
    axes[1].set_ylim(0, max(max(means_top), max(means_rnd)) * 1.18)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Q{qnum}. {method_label} top-{k} cohort retention")
    fig.tight_layout()

    rows_html = "".join(
        f"<li>Q{i+1} (n_act in [{t['lo']}-{t['hi']}], <b>{t['n']:,}</b> users): "
        f"top-K={t['top']:.3f}, random={t['rnd']:.3f}, "
        f"<b>lift={t['top']/max(t['rnd'], 1e-9):.2f}x</b></li>"
        for i, t in enumerate(tiers)
    )
    prose = (
        f"<b>{method_label}</b> cosine top-{k} cohort retention. Overall mean = "
        f"<b>{res['overall_top']:.3f}</b> vs random = "
        f"<b>{res['overall_rnd']:.3f}</b> (lift = <b>{overall_lift:.2f}x</b>). "
        f"Per tier:<ul>{rows_html}</ul>" + extra_prose
    )
    return Section(slug, f"Q{qnum}. {method_label} top-K cohort retention",
                   prose, _fig_to_b64(fig))


def _section_topk_cohort_retention_tfidf(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5,
) -> Section:
    """Q19. TF-IDF cosine top-K cohort retention (overall + by tier)."""
    M_a, _, S = _aligned_sparse_setup(M, email_ids, "tfidf_vector")
    return _make_method_section(
        "TF-IDF (sparse, 20K vocab)", "#f59e0b", "cohort_retention_tfidf", 19,
        M_a, S, k,
        extra_prose=(
            "<br>Sklearn <code>TfidfVectorizer</code>: ngram_range=(1,2), "
            "max_features=20000, sublinear_tf=True, English stopwords; "
            "L2-normalized. Captures lexical overlap including bigram phrases. "
            "Sparse vectors -- a few hundred non-zero entries per email."
        ),
    )


def _section_topk_cohort_retention_bm25(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5,
) -> Section:
    """Q20. BM25 cosine top-K cohort retention (overall + by tier)."""
    M_a, _, S = _aligned_sparse_setup(M, email_ids, "sparse_vector")
    return _make_method_section(
        "BM25 (sparse)", "#a855f7", "cohort_retention_bm25", 20,
        M_a, S, k,
        extra_prose=(
            "<br>Pinecone <code>BM25Encoder</code> (k1=1.2, b=0.75), "
            "L2-normalized. Same bag-of-words family as TF-IDF but with "
            "document-length normalization and saturating term frequency."
        ),
    )


def _section_all_methods_compare(
    M: sp.csr_matrix, email_ids: np.ndarray, k: int = 5,
) -> Section:
    """Q21. Random / Jaccard / dense-cosine / TF-IDF / BM25 head-to-head."""
    M_a, email_ids_a, S_dense = _aligned_cosine_setup(M, email_ids)
    n_a = S_dense.shape[0]

    V_tfidf, _, keep_t = _load_sparse_vectors(email_ids_a, "tfidf_vector")
    V_bm25, _, keep_b = _load_sparse_vectors(email_ids_a, "sparse_vector")

    common = np.intersect1d(keep_t, keep_b)
    print(f"  4-way intersect: {len(common)}/{n_a} emails with all 3 embeddings")
    M_c = M_a[:, common].tocsr()
    n = len(common)
    S_dense_c = S_dense[np.ix_(common, common)].copy()
    np.fill_diagonal(S_dense_c, -np.inf)

    pos_t = {a: j for j, a in enumerate(keep_t)}
    pos_b = {a: j for j, a in enumerate(keep_b)}
    rows_t = np.array([pos_t[a] for a in common])
    rows_b = np.array([pos_b[a] for a in common])
    V_tfidf_c = V_tfidf[rows_t]
    V_bm25_c = V_bm25[rows_b]
    S_tfidf = (V_tfidf_c @ V_tfidf_c.T).toarray().astype(np.float32)
    np.fill_diagonal(S_tfidf, -np.inf)
    S_bm25 = (V_bm25_c @ V_bm25_c.T).toarray().astype(np.float32)
    np.fill_diagonal(S_bm25, -np.inf)

    methods = [
        ("random", "Random",        "#94a3b8", _random_topk_adj(n, k)),
        ("jaccard", "Jaccard",      "#0ea5e9", _jaccard_topk_adj(M_c, k)),
        ("dense", "Dense cosine",   "#16a34a", _build_topk_adj(S_dense_c, k)),
        ("tfidf", "TF-IDF cosine",  "#f59e0b", _build_topk_adj(S_tfidf, k)),
        ("bm25", "BM25 cosine",     "#a855f7", _build_topk_adj(S_bm25, k)),
    ]

    counts = np.asarray(M_c.sum(axis=1)).flatten()
    eligible = counts >= 2
    M_e = M_c[eligible]
    counts_e = counts[eligible]
    qs = np.quantile(counts_e, [0.25, 0.50, 0.75])
    edges = [2, int(qs[0]) + 1, int(qs[1]) + 1, int(qs[2]) + 1, int(counts_e.max()) + 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1

    groups: list[tuple[str, sp.csr_matrix]] = [("All (>=2)", M_e)]
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (counts_e >= lo) & (counts_e < hi)
        if m.sum() > 0:
            groups.append((f"Q{i+1}\n[{lo}-{hi-1}]", M_e[m]))

    grid: dict[tuple[str, str], float] = {}
    for key, label, color, adj in methods:
        for g_label, sub in groups:
            r, _ = _cohort_retention(sub, adj)
            grid[(key, g_label)] = float(np.nanmean(r))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    x = np.arange(len(groups))
    n_methods = len(methods)
    w = 0.85 / n_methods
    for j, (key, label, color, _) in enumerate(methods):
        offset = (j - (n_methods - 1) / 2) * w
        vals = [grid[(key, g_label)] for g_label, _ in groups]
        axes[0].bar(x + offset, vals, width=w, color=color, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g for g, _ in groups], rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Mean cohort retention")
    axes[0].set_title("Absolute retention")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    rand_vals = {g_label: grid[("random", g_label)] for g_label, _ in groups}
    methods_nr = [m for m in methods if m[0] != "random"]
    n_nr = len(methods_nr)
    w_l = 0.85 / n_nr
    for j, (key, label, color, _) in enumerate(methods_nr):
        offset = (j - (n_nr - 1) / 2) * w_l
        vals = [grid[(key, g_label)] / max(rand_vals[g_label], 1e-9) for g_label, _ in groups]
        axes[1].bar(x + offset, vals, width=w_l, color=color, label=label)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([g for g, _ in groups], rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Lift over random K")
    axes[1].set_title("Lift vs random baseline")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Q21. All methods head-to-head (top-{k}, n={n} emails)")
    fig.tight_layout()

    method_adjs_dense = {
        key: adj.toarray().astype(bool) for key, _, _, adj in methods
    }
    keys = ["jaccard", "dense", "tfidf", "bm25"]
    agreements: list[tuple[str, str, float]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = method_adjs_dense[keys[i]] & method_adjs_dense[keys[j]]
            agreement = float(a.sum(axis=1).mean() / k)
            agreements.append((keys[i], keys[j], agreement))
            print(f"  agreement {keys[i]} vs {keys[j]}: {agreement:.3f}")

    for key, label, _, _ in methods:
        all_v = grid[(key, "All (>=2)")]
        print(f"  {label}: all-tier retention = {all_v:.4f}  "
              f"(lift = {all_v/max(grid[('random','All (>=2)')], 1e-9):.2f}x)")

    rows_html = "".join(
        f"<li><b>{label}</b>: All-tier = {grid[(key, 'All (>=2)')]:.3f}, "
        f"lift = {grid[(key, 'All (>=2)')]/max(grid[('random','All (>=2)')], 1e-9):.2f}x</li>"
        for key, label, _, _ in methods if key != "random"
    )
    agree_html = "".join(
        f"<li>{a} &harr; {b}: <b>{v:.3f}</b></li>" for a, b, v in agreements
    )
    prose = (
        f"All four neighbor methods evaluated on the same {n:,}-email subset "
        f"(emails with dense cosine + TF-IDF + BM25 vectors). Same retention "
        f"metric as Q14/Q16/Q19/Q20.<br><br><b>All-tier retention + lift over random:</b>"
        f"<ul>{rows_html}</ul>"
        f"<b>Top-{k} neighbor-set agreement (mean fraction shared per email):</b>"
        f"<ul>{agree_html}</ul>"
        "<b>How to read this:</b><ul>"
        "<li>If the three content methods (dense / TF-IDF / BM25) cluster "
        "tightly together far below Jaccard: content choice is not the "
        "bottleneck -- audience overlap simply isn't a content property.</li>"
        "<li>If sparse methods (TF-IDF / BM25) beat dense: the audience signal "
        "lives in surface lexical overlap (shared phrases, named entities) "
        "more than deep semantics.</li>"
        "<li>If dense beats sparse: the opposite -- semantic similarity carries "
        "more audience signal than lexical match.</li>"
        "<li>If neighbor-set agreements between content methods are high "
        "(&gt;0.5) but agreement with Jaccard is low: content methods see the "
        "same content structure but it doesn't predict audience.</li>"
        "</ul>"
    )
    return Section(
        "all_methods_compare",
        "Q21. All-method head-to-head: random / Jaccard / dense / TF-IDF / BM25",
        prose, _fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Q22. Held-out Jaccard (fair head-to-head with content methods).
# ---------------------------------------------------------------------------

def _section_held_out_jaccard_compare(
    wide: pl.DataFrame,
    M: sp.csr_matrix,
    email_ids: np.ndarray,
    user_ids: np.ndarray,
    k: int = 5,
) -> Section:
    """Q22. Per-user chronological split. Train Jaccard on first-half acts,
    evaluate retention on second-half acts. Content methods play on the same
    test surface so comparison is apples-to-apples.
    """
    actioned = (
        wide.filter(pl.col("actioned"))
        .select("email_id", "user_id", "sent_at")
        .unique()
    )
    n_per_user = actioned.group_by("user_id").agg(pl.len().alias("n"))
    keep_users = set(n_per_user.filter(pl.col("n") >= 3)["user_id"].to_list())
    actioned_f = actioned.filter(pl.col("user_id").is_in(list(keep_users)))
    print(
        f"  held-out: users with n_full>=3: {len(keep_users):,} of "
        f"{len(user_ids):,}"
    )

    actioned_f = actioned_f.with_columns(
        pl.col("sent_at").rank("ordinal").over("user_id").alias("rk"),
        pl.len().over("user_id").alias("n_user"),
    ).with_columns(
        (pl.col("rk") <= (pl.col("n_user") / 2).floor().cast(pl.Int64))
        .alias("is_train")
    )

    def _make_matrix(sub: pl.DataFrame) -> sp.csr_matrix:
        rows = np.searchsorted(user_ids, sub["user_id"].to_numpy())
        cols = np.searchsorted(email_ids, sub["email_id"].to_numpy())
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(email_ids)),
        )

    M_train = _make_matrix(actioned_f.filter(pl.col("is_train")))
    M_test = _make_matrix(actioned_f.filter(~pl.col("is_train")))
    print(f"  M_train.nnz={M_train.nnz:,}  M_test.nnz={M_test.nnz:,}")

    train_pop = np.asarray(M_train.sum(axis=0)).flatten()
    test_pop = np.asarray(M_test.sum(axis=0)).flatten()
    valid = (train_pop > 0) & (test_pop > 0)
    print(
        f"  emails with >=1 train AND >=1 test action: "
        f"{valid.sum()}/{len(email_ids)}"
    )

    valid_email_ids = email_ids[valid]
    M_train_v = M_train[:, valid].tocsr()
    M_test_v = M_test[:, valid].tocsr()
    M_full_v = M[:, valid].tocsr()

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(valid_email_ids.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id_dense = {
        r["email_id"]: r["dense_vector"] for r in df.iter_rows(named=True)
    }
    has_dense = np.array([int(e) in by_id_dense for e in valid_email_ids])
    keep_dense_idx = np.where(has_dense)[0]
    n_drop = int((~has_dense).sum())
    if n_drop:
        print(f"  dense: dropping {n_drop} valid emails missing embeddings")
    email_ids_a = valid_email_ids[keep_dense_idx]
    M_train_a = M_train_v[:, keep_dense_idx].tocsr()
    M_test_a = M_test_v[:, keep_dense_idx].tocsr()
    M_full_a = M_full_v[:, keep_dense_idx].tocsr()

    V_dense = np.asarray(
        [by_id_dense[int(e)] for e in email_ids_a], dtype=np.float32
    )
    S_dense = V_dense @ V_dense.T
    np.fill_diagonal(S_dense, -np.inf)

    V_tfidf, _, keep_t = _load_sparse_vectors(email_ids_a, "tfidf_vector")
    V_bm25, _, keep_b = _load_sparse_vectors(email_ids_a, "sparse_vector")
    common = np.intersect1d(keep_t, keep_b)
    print(f"  4-way intersect: {len(common)}/{len(email_ids_a)}")

    pos_t = {a: j for j, a in enumerate(keep_t)}
    pos_b = {a: j for j, a in enumerate(keep_b)}
    rows_t = np.array([pos_t[a] for a in common])
    rows_b = np.array([pos_b[a] for a in common])
    V_tfidf_c = V_tfidf[rows_t]
    V_bm25_c = V_bm25[rows_b]

    M_train_c = M_train_a[:, common].tocsr()
    M_test_c = M_test_a[:, common].tocsr()
    M_full_c = M_full_a[:, common].tocsr()

    S_dense_c = S_dense[np.ix_(common, common)].copy()
    np.fill_diagonal(S_dense_c, -np.inf)
    S_tfidf = (V_tfidf_c @ V_tfidf_c.T).toarray().astype(np.float32)
    np.fill_diagonal(S_tfidf, -np.inf)
    S_bm25 = (V_bm25_c @ V_bm25_c.T).toarray().astype(np.float32)
    np.fill_diagonal(S_bm25, -np.inf)

    n = len(common)
    methods = [
        ("random",    "Random",            "#94a3b8", _random_topk_adj(n, k)),
        ("jac_full",  "Jaccard in-sample", "#60a5fa", _jaccard_topk_adj(M_full_c, k)),
        ("jac_train", "Jaccard held-out",  "#1e3a8a", _jaccard_topk_adj(M_train_c, k)),
        ("dense",     "Dense cosine",      "#16a34a", _build_topk_adj(S_dense_c, k)),
        ("tfidf",     "TF-IDF cosine",     "#f59e0b", _build_topk_adj(S_tfidf, k)),
        ("bm25",      "BM25 cosine",       "#a855f7", _build_topk_adj(S_bm25, k)),
    ]

    counts_test = np.asarray(M_test_c.sum(axis=1)).flatten()
    eligible = counts_test >= 2
    M_test_e = M_test_c[eligible]
    counts_e = counts_test[eligible]
    print(f"  eligible users (test n>=2): {int(eligible.sum()):,}")
    if counts_e.size < 100:
        return Section(
            "held_out_jaccard", "Q22. Held-out Jaccard",
            "Insufficient data after split.", "",
        )

    qs = np.quantile(counts_e, [0.25, 0.50, 0.75])
    edges = [2, int(qs[0]) + 1, int(qs[1]) + 1, int(qs[2]) + 1, int(counts_e.max()) + 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1

    groups: list[tuple[str, sp.csr_matrix]] = [("All (test>=2)", M_test_e)]
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (counts_e >= lo) & (counts_e < hi)
        if m.sum() > 0:
            groups.append((f"Q{i+1}\n[{lo}-{hi-1}]", M_test_e[m]))

    grid: dict[tuple[str, str], float] = {}
    for key, label, color, adj in methods:
        for g_label, sub in groups:
            r, _ = _cohort_retention(sub, adj)
            grid[(key, g_label)] = float(np.nanmean(r))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    x = np.arange(len(groups))
    n_methods = len(methods)
    w = 0.85 / n_methods
    for j, (key, label, color, _) in enumerate(methods):
        offset = (j - (n_methods - 1) / 2) * w
        vals = [grid[(key, g_label)] for g_label, _ in groups]
        axes[0].bar(x + offset, vals, width=w, color=color, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g for g, _ in groups], rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Mean cohort retention (on test)")
    axes[0].set_title("Absolute retention (evaluated on M_test)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    rand_vals = {g_label: grid[("random", g_label)] for g_label, _ in groups}
    methods_nr = [m for m in methods if m[0] != "random"]
    n_nr = len(methods_nr)
    w_l = 0.85 / n_nr
    for j, (key, label, color, _) in enumerate(methods_nr):
        offset = (j - (n_nr - 1) / 2) * w_l
        vals = [grid[(key, g_label)] / max(rand_vals[g_label], 1e-9) for g_label, _ in groups]
        axes[1].bar(x + offset, vals, width=w_l, color=color, label=label)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([g for g, _ in groups], rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Lift over random K")
    axes[1].set_title("Lift vs random baseline")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Q22. Fair head-to-head: held-out Jaccard vs content "
        f"(n={n} emails, {int(eligible.sum()):,} eligible users)"
    )
    fig.tight_layout()

    for key, label, _, _ in methods:
        all_v = grid[(key, "All (test>=2)")]
        all_lift = all_v / max(grid[("random", "All (test>=2)")], 1e-9)
        print(f"  {label}: all-tier retention = {all_v:.4f}  (lift = {all_lift:.2f}x)")

    rows_html = "".join(
        f"<li><b>{label}</b>: All-tier = {grid[(key, 'All (test>=2)')]:.3f}, "
        f"lift = {grid[(key, 'All (test>=2)')] / max(grid[('random','All (test>=2)')], 1e-9):.2f}x</li>"
        for key, label, _, _ in methods if key != "random"
    )
    prose = (
        "<b>The fair comparison.</b> Q21's Jaccard advantage was partly an "
        "over-fitting artifact: top-K Jaccard was chosen using the same "
        "actions we then evaluated retention on. To fix this, we split each "
        "user's actions chronologically by <code>sent_at</code> -- first half "
        "= train, second half = test. We build Jaccard neighbors on TRAIN "
        "actions only and evaluate retention on TEST actions. Content methods "
        "don't use behavior, so the split doesn't change them.<br><br>"
        f"Same eval surface for all methods: {n:,}-email subset with both "
        "train and test actions plus all three embeddings. Eligibility: "
        "users with test n_actioned >= 2.<br><br>"
        "<b>All-tier retention + lift over random:</b>"
        f"<ul>{rows_html}</ul>"
        "<b>Reading the chart:</b><ul>"
        "<li><b>Jaccard in-sample</b> (light blue) = Q21's tautological "
        "number for reference -- the upper bound when neighbors are picked "
        "using the same data we evaluate on.</li>"
        "<li><b>Jaccard held-out</b> (dark blue) = the honest behavior "
        "number. The gap between them is the over-fit gap.</li>"
        "<li><b>Content methods</b> (green / amber / purple) play on the "
        "same eval surface and have no over-fit advantage. The honest "
        "comparison is held-out Jaccard vs content method.</li>"
        "</ul>"
    )
    return Section(
        "held_out_jaccard",
        "Q22. Fair head-to-head: held-out Jaccard vs content methods",
        prose,
        _fig_to_b64(fig),
    )


# ---------------------------------------------------------------------------
# Q23-Q24. Hybrid model + per-email distribution (built on shared held-out setup).
# ---------------------------------------------------------------------------

def _heldout_setup(
    wide: pl.DataFrame, M: sp.csr_matrix,
    email_ids: np.ndarray, user_ids: np.ndarray,
    agg_version: int = 3, min_n: int = 3,
) -> dict:
    """Shared held-out alignment used by Q22-Q24.

    Returns a dict with M_train_c, M_test_c, M_full_c, S_dense, S_tfidf,
    S_bm25, subjects, senders, all aligned to the same n-email subset.
    """
    actioned = (
        wide.filter(pl.col("actioned"))
        .select("email_id", "user_id", "sent_at")
        .unique()
    )
    n_per_user = actioned.group_by("user_id").agg(pl.len().alias("n"))
    keep_users = set(
        n_per_user.filter(pl.col("n") >= min_n)["user_id"].to_list()
    )
    af = (
        actioned.filter(pl.col("user_id").is_in(list(keep_users)))
        .with_columns(
            pl.col("sent_at").rank("ordinal").over("user_id").alias("rk"),
            pl.len().over("user_id").alias("n_user"),
        )
        .with_columns(
            (pl.col("rk") <= (pl.col("n_user") / 2).floor().cast(pl.Int64))
            .alias("is_train")
        )
    )

    def _mat(sub: pl.DataFrame) -> sp.csr_matrix:
        rows = np.searchsorted(user_ids, sub["user_id"].to_numpy())
        cols = np.searchsorted(email_ids, sub["email_id"].to_numpy())
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(email_ids)),
        )

    M_train = _mat(af.filter(pl.col("is_train")))
    M_test = _mat(af.filter(~pl.col("is_train")))

    train_pop = np.asarray(M_train.sum(axis=0)).flatten()
    test_pop = np.asarray(M_test.sum(axis=0)).flatten()
    valid = (train_pop > 0) & (test_pop > 0)
    valid_email_ids = email_ids[valid]

    df = (
        agg_cache.get(agg_version)
        .filter(pl.col("email_id").is_in(valid_email_ids.tolist()))
        .select("email_id", "subject", "sender", "dense_vector")
    )
    by_id_meta = {row["email_id"]: row for row in df.iter_rows(named=True)}
    has_dense = np.array([
        int(e) in by_id_meta and by_id_meta[int(e)]["dense_vector"] is not None
        for e in valid_email_ids
    ])
    keep_dense_idx = np.where(has_dense)[0]
    email_ids_a = valid_email_ids[keep_dense_idx]

    M_train_a = M_train[:, valid][:, keep_dense_idx].tocsr()
    M_test_a = M_test[:, valid][:, keep_dense_idx].tocsr()
    M_full_a = M[:, valid][:, keep_dense_idx].tocsr()

    V_dense = np.asarray(
        [by_id_meta[int(e)]["dense_vector"] for e in email_ids_a],
        dtype=np.float32,
    )
    S_dense = V_dense @ V_dense.T
    np.fill_diagonal(S_dense, -np.inf)

    V_tfidf, _, keep_t = _load_sparse_vectors(email_ids_a, "tfidf_vector")
    V_bm25, _, keep_b = _load_sparse_vectors(email_ids_a, "sparse_vector")
    common = np.intersect1d(keep_t, keep_b)
    pos_t = {a: j for j, a in enumerate(keep_t)}
    pos_b = {a: j for j, a in enumerate(keep_b)}
    rows_t = np.array([pos_t[a] for a in common])
    rows_b = np.array([pos_b[a] for a in common])
    V_tfidf_c = V_tfidf[rows_t]
    V_bm25_c = V_bm25[rows_b]

    M_train_c = M_train_a[:, common].tocsr()
    M_test_c = M_test_a[:, common].tocsr()
    M_full_c = M_full_a[:, common].tocsr()
    email_ids_c = email_ids_a[common]

    S_dense_c = S_dense[np.ix_(common, common)].copy()
    np.fill_diagonal(S_dense_c, -np.inf)
    S_tfidf = (V_tfidf_c @ V_tfidf_c.T).toarray().astype(np.float32)
    np.fill_diagonal(S_tfidf, -np.inf)
    S_bm25 = (V_bm25_c @ V_bm25_c.T).toarray().astype(np.float32)
    np.fill_diagonal(S_bm25, -np.inf)

    subjects = [
        (by_id_meta[int(e)]["subject"] or "") for e in email_ids_c
    ]
    senders = [
        (by_id_meta[int(e)]["sender"] or "") for e in email_ids_c
    ]

    return {
        "M_train_c": M_train_c,
        "M_test_c": M_test_c,
        "M_full_c": M_full_c,
        "email_ids_c": email_ids_c,
        "subjects": subjects,
        "senders": senders,
        "S_dense": S_dense_c,
        "S_tfidf": S_tfidf,
        "S_bm25": S_bm25,
        "n": len(common),
    }


def _row_zscore(M: np.ndarray) -> np.ndarray:
    """Per-row z-score, ignoring -inf diagonal entries."""
    safe = M.copy()
    np.fill_diagonal(safe, 0.0)
    means = safe.mean(axis=1, keepdims=True)
    stds = safe.std(axis=1, keepdims=True) + 1e-9
    Z = (safe - means) / stds
    np.fill_diagonal(Z, -np.inf)
    return Z.astype(np.float32)


def _section_hybrid_ranking(
    wide: pl.DataFrame, M: sp.csr_matrix,
    email_ids: np.ndarray, user_ids: np.ndarray, k: int = 5,
) -> Section:
    """Q23. Sweep α in [0, 1] for hybrid score = α · Jaccard + (1-α) · BM25.

    The headline question: does combining behavior + content beat either alone?
    """
    setup = _heldout_setup(wide, M, email_ids, user_ids)
    M_train_c = setup["M_train_c"]
    M_test_c = setup["M_test_c"]
    n = setup["n"]
    S_bm25 = setup["S_bm25"]

    co = (M_train_c.T @ M_train_c).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -np.inf)

    J_z = _row_zscore(J)
    S_z = _row_zscore(S_bm25)

    counts_test = np.asarray(M_test_c.sum(axis=1)).flatten()
    eligible = counts_test >= 2
    M_test_e = M_test_c[eligible]
    print(f"  Q23: n={n} emails, eligible users={int(eligible.sum()):,}")

    rand_seeds = list(range(5))
    rand_rets_per_seed = []
    for s in rand_seeds:
        adj_r = _random_topk_adj(n, k, seed=s)
        r, _ = _cohort_retention(M_test_e, adj_r)
        rand_rets_per_seed.append(np.nanmean(r))
    rand_mean = float(np.mean(rand_rets_per_seed))
    print(f"  Q23: random baseline (mean of {len(rand_seeds)} seeds) = {rand_mean:.4f}")

    alphas = np.linspace(0, 1, 11)
    rets: list[float] = []
    lifts: list[float] = []
    for alpha in alphas:
        score = alpha * J_z + (1 - alpha) * S_z
        adj = _build_topk_adj(score, k)
        r, _ = _cohort_retention(M_test_e, adj)
        m = float(np.nanmean(r))
        rets.append(m)
        lifts.append(m / max(rand_mean, 1e-9))
        print(f"    alpha={alpha:.2f}: retention={m:.4f}  lift={lifts[-1]:.2f}x")
    best_idx = int(np.argmax(lifts))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(alphas, rets, marker="o", linewidth=2, color="#0ea5e9")
    axes[0].axhline(rand_mean, color="black", linestyle="--", linewidth=0.7,
                    label=f"random ({rand_mean:.3f})")
    axes[0].set_xlabel("α (weight on behavior; (1-α) on content)")
    axes[0].set_ylabel("Mean cohort retention (test)")
    axes[0].set_title("Absolute retention")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(alphas, lifts, marker="o", linewidth=2, color="#0ea5e9")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].scatter([0], [lifts[0]], color="#a855f7", s=120, zorder=5,
                    label=f"100% content: {lifts[0]:.2f}x")
    axes[1].scatter([1], [lifts[-1]], color="#1e3a8a", s=120, zorder=5,
                    label=f"100% behavior: {lifts[-1]:.2f}x")
    axes[1].scatter([alphas[best_idx]], [lifts[best_idx]], color="red",
                    marker="*", s=260, zorder=6,
                    label=f"best α={alphas[best_idx]:.2f}: {lifts[best_idx]:.2f}x")
    axes[1].set_xlabel("α (weight on behavior; (1-α) on content)")
    axes[1].set_ylabel("Lift over random K=5")
    axes[1].set_title("Lift")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Q23. Hybrid: α · Jaccard(held-out) + (1-α) · BM25 cosine — "
        f"does combining beat either alone?"
    )
    fig.tight_layout()

    if 0 < alphas[best_idx] < 1:
        gain_vs_behavior = (lifts[best_idx] - lifts[-1]) / max(lifts[-1], 1e-9) * 100
        verdict = (
            f"Best α = <b>{alphas[best_idx]:.2f}</b> sits between 0 and 1, "
            f"meaning the hybrid <b>beats either signal alone</b>. Combined "
            f"lift = <b>{lifts[best_idx]:.2f}x</b> vs pure-behavior "
            f"<b>{lifts[-1]:.2f}x</b> -- a <b>+{gain_vs_behavior:.1f}%</b> "
            f"relative improvement from adding content. This confirms "
            f"content carries signal that's <b>independent of behavior</b>."
        )
    elif alphas[best_idx] >= 1.0:
        verdict = (
            f"Best α = 1.0 -- behavior alone is optimal in this sweep. "
            f"Hybrid reduces to held-out Jaccard. Doesn't rule out content "
            f"adding value on cold-start emails (see Q24 distribution)."
        )
    else:
        verdict = (
            f"Best α = 0.0 -- content alone won this sweep. Behavior added "
            f"no incremental signal at K={k}."
        )

    prose = (
        "We rank candidate emails by a weighted combination of "
        "<b>behavior-Jaccard (held-out, from M_train)</b> and "
        "<b>BM25 content cosine</b>, with both scores row-z-scored so they're "
        "on the same scale. We sweep α from 0 (pure content) to 1 (pure "
        "behavior) and measure top-K cohort retention on M_test.<br><br>"
        f"{verdict}"
    )
    return Section(
        "hybrid_ranking",
        "Q23. Hybrid behavior + content ranking",
        prose, _fig_to_b64(fig),
    )


def _section_per_email_content_lift(
    wide: pl.DataFrame, M: sp.csr_matrix,
    email_ids: np.ndarray, user_ids: np.ndarray, k: int = 5,
) -> Section:
    """Q24. Per-email BM25 lift distribution + concrete examples."""
    setup = _heldout_setup(wide, M, email_ids, user_ids)
    M_test_c = setup["M_test_c"]
    n = setup["n"]
    S_bm25 = setup["S_bm25"]
    subjects = setup["subjects"]
    senders = setup["senders"]

    adj_bm25 = _build_topk_adj(S_bm25, k)
    bm25_ret, denom = _cohort_retention(M_test_c, adj_bm25)

    rand_seeds = list(range(5))
    rand_per_seed = []
    for s in rand_seeds:
        adj_r = _random_topk_adj(n, k, seed=s)
        r, _ = _cohort_retention(M_test_c, adj_r)
        rand_per_seed.append(r)
    rand_ret = np.nanmean(np.stack(rand_per_seed), axis=0)

    valid = (~np.isnan(bm25_ret)) & (~np.isnan(rand_ret)) & (rand_ret > 0) & (denom >= 30)
    lifts = np.where(valid, bm25_ret / np.where(rand_ret > 0, rand_ret, 1.0), np.nan)
    lifts_v = lifts[valid]
    print(
        f"  Q24: {valid.sum()} emails with >=30 test actioners; "
        f"mean lift={np.nanmean(lifts_v):.2f}x  median={np.nanmedian(lifts_v):.2f}x  "
        f">=2x: {(lifts_v >= 2).sum()} ({100*(lifts_v >= 2).mean():.1f}%)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.logspace(np.log10(max(lifts_v.min(), 0.1)), np.log10(lifts_v.max() * 1.05), 50)
    axes[0].hist(lifts_v, bins=bins, color="#a855f7", edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=0.6, label="random")
    axes[0].axvline(np.nanmean(lifts_v), color="red", linestyle="--", linewidth=1.2,
                    label=f"mean = {np.nanmean(lifts_v):.2f}x")
    axes[0].axvline(np.nanmedian(lifts_v), color="orange", linestyle="--", linewidth=1.2,
                    label=f"median = {np.nanmedian(lifts_v):.2f}x")
    axes[0].set_xlabel("Per-email content lift = BM25 retention / random retention (log)")
    axes[0].set_ylabel("Number of emails")
    axes[0].set_title("Distribution of per-email content lift")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(rand_ret[valid], bm25_ret[valid], s=14, alpha=0.45,
                    c=np.log10(denom[valid]), cmap="viridis")
    lim = max(np.nanmax(rand_ret[valid]), np.nanmax(bm25_ret[valid])) * 1.05
    axes[1].plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=0.7,
                 label="parity (no lift)")
    axes[1].set_xlabel("Random K=5 retention (per email, mean of 5 seeds)")
    axes[1].set_ylabel("BM25 top-K retention (per email)")
    axes[1].set_title("Per-email: BM25 vs random (color = log10 actioners)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Q24. Where does content routing work? Per-email lift distribution")
    fig.tight_layout()

    valid_idx = np.where(valid)[0]
    top_idx = valid_idx[np.argsort(-lifts[valid_idx])[:10]]
    bot_idx = valid_idx[np.argsort(lifts[valid_idx])[:10]]

    def _trim(s: str, n_chars: int) -> str:
        s = s.replace("\r", " ").replace("\n", " ")
        return (s[:n_chars] + "...") if len(s) > n_chars else s

    def _row(idx: int) -> str:
        return (
            f"<tr>"
            f"<td style='text-align:right'><b>{lifts[idx]:.2f}x</b></td>"
            f"<td style='text-align:right'>{bm25_ret[idx]:.3f}</td>"
            f"<td style='text-align:right'>{rand_ret[idx]:.3f}</td>"
            f"<td style='text-align:right'>{int(denom[idx]):,}</td>"
            f"<td>{html.escape(_trim(senders[idx], 28))}</td>"
            f"<td>{html.escape(_trim(subjects[idx], 90))}</td>"
            f"</tr>"
        )

    table_header = (
        "<table style='font-size:12px; border-collapse:collapse; width:100%;'>"
        "<tr><th>lift</th><th>bm25</th><th>random</th>"
        "<th>actioners</th><th>sender</th><th>subject</th></tr>"
    )
    table_top = table_header + "".join(_row(i) for i in top_idx) + "</table>"
    table_bot = table_header + "".join(_row(i) for i in bot_idx) + "</table>"

    pct_above_2x = float((lifts_v >= 2).sum()) / valid.sum() * 100
    pct_above_5x = float((lifts_v >= 5).sum()) / valid.sum() * 100

    prose = (
        f"Per-email content lift = (BM25 top-{k} retention on test) / "
        f"(mean random K={k} retention from 5 seeds), per email, "
        f"filtered to <b>{int(valid.sum())}</b> emails with at least 30 test "
        f"actioners.<br><br>"
        f"<b>Distribution stats:</b> mean lift = "
        f"<b>{np.nanmean(lifts_v):.2f}x</b>, median = "
        f"<b>{np.nanmedian(lifts_v):.2f}x</b>. "
        f"<b>{pct_above_2x:.1f}%</b> of emails see >=2x content lift; "
        f"<b>{pct_above_5x:.1f}%</b> see >=5x.<br><br>"
        "Content routing isn't uniform -- some emails benefit a lot, others "
        "barely. The right scatter plot shows per-email BM25 retention vs "
        "random retention; points above the parity line are emails where "
        "content routing helps. Color = log10(test actioners), so darker "
        "points are emails with smaller actioner pools.<br><br>"
        "<b>Top 10 emails by content lift</b> (high content signal):"
        f"<br>{table_top}<br>"
        "<b>Bottom 10 emails</b> (content gives no boost or hurts):"
        f"<br>{table_bot}"
    )
    return Section(
        "per_email_content_lift",
        "Q24. Where does content routing work? Per-email distribution + examples",
        prose, _fig_to_b64(fig),
    )


def _section_outlier_check(
    wide: pl.DataFrame, M: sp.csr_matrix,
    email_ids: np.ndarray, user_ids: np.ndarray, k: int = 5,
) -> Section:
    """Q25. Are low-volume outlier emails distorting the headline numbers?

    Distribution of per-email sends + actioners. Re-runs held-out Jaccard
    and BM25 lift at progressively higher send thresholds. If lifts stay flat
    or rise, outliers were not driving the result. If they collapse, the
    headline number was carried by tiny emails.
    """
    setup = _heldout_setup(wide, M, email_ids, user_ids)
    M_train_c = setup["M_train_c"]
    M_test_c = setup["M_test_c"]
    M_full_c = setup["M_full_c"]
    n = setup["n"]
    S_bm25 = setup["S_bm25"]
    email_ids_c = setup["email_ids_c"]
    subjects = setup["subjects"]
    senders = setup["senders"]

    sends_per_email = (
        wide.group_by("email_id").len().rename({"len": "n_sends"})
    )
    sends_dict = dict(zip(
        sends_per_email["email_id"].to_list(),
        sends_per_email["n_sends"].to_list(),
    ))
    n_sends = np.array([sends_dict.get(int(e), 0) for e in email_ids_c])
    n_actioners = np.asarray(M_full_c.sum(axis=0)).flatten()
    n_train_act = np.asarray(M_train_c.sum(axis=0)).flatten()
    n_test_act = np.asarray(M_test_c.sum(axis=0)).flatten()

    print(
        f"  Q25: {n} emails. "
        f"n_sends p10={int(np.percentile(n_sends, 10)):,} "
        f"median={int(np.median(n_sends)):,} "
        f"p90={int(np.percentile(n_sends, 90)):,} "
        f"max={int(n_sends.max()):,}"
    )
    print(
        f"       n_actioners p10={int(np.percentile(n_actioners, 10)):,} "
        f"median={int(np.median(n_actioners)):,} "
        f"p90={int(np.percentile(n_actioners, 90)):,} "
        f"max={int(n_actioners.max()):,}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    bins_s = np.logspace(0, np.log10(max(n_sends.max(), 2) + 1), 50)
    axes[0, 0].hist(n_sends, bins=bins_s, color="#0ea5e9", edgecolor="white")
    axes[0, 0].set_xscale("log")
    axes[0, 0].axvline(np.median(n_sends), color="red", linestyle="--",
                       label=f"median={int(np.median(n_sends)):,}")
    axes[0, 0].axvline(np.percentile(n_sends, 10), color="orange",
                       linestyle="--",
                       label=f"p10={int(np.percentile(n_sends, 10)):,}")
    axes[0, 0].set_xlabel("Sends per email (log)")
    axes[0, 0].set_ylabel("Emails")
    axes[0, 0].set_title("Send-volume distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    bins_a = np.logspace(0, np.log10(max(n_actioners.max(), 2) + 1), 50)
    axes[0, 1].hist(n_actioners, bins=bins_a, color="#a855f7", edgecolor="white")
    axes[0, 1].set_xscale("log")
    axes[0, 1].axvline(np.median(n_actioners), color="red", linestyle="--",
                       label=f"median={int(np.median(n_actioners)):,}")
    axes[0, 1].axvline(np.percentile(n_actioners, 10), color="orange",
                       linestyle="--",
                       label=f"p10={int(np.percentile(n_actioners, 10)):,}")
    axes[0, 1].set_xlabel("Actioners per email (log)")
    axes[0, 1].set_ylabel("Emails")
    axes[0, 1].set_title("Actioner-count distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    thresholds = [0, 100, 500, 1000, 5000, 10000]
    results: list[dict] = []
    for thr in thresholds:
        keep = n_sends >= thr
        if keep.sum() < 50:
            continue
        keep_idx = np.where(keep)[0]
        M_train_f = M_train_c[:, keep_idx].tocsr()
        M_test_f = M_test_c[:, keep_idx].tocsr()
        S_bm25_f = S_bm25[np.ix_(keep_idx, keep_idx)].copy()
        np.fill_diagonal(S_bm25_f, -np.inf)

        co = (M_train_f.T @ M_train_f).toarray().astype(np.float32)
        pop = np.diag(co).copy()
        union = pop[:, None] + pop[None, :] - co
        J = np.where(union > 0, co / union, 0).astype(np.float32)
        np.fill_diagonal(J, -np.inf)

        n_f = int(keep.sum())
        adj_jac = _build_topk_adj(J, k)
        adj_bm25 = _build_topk_adj(S_bm25_f, k)

        counts_test_f = np.asarray(M_test_f.sum(axis=1)).flatten()
        eligible_f = counts_test_f >= 2
        if int(eligible_f.sum()) < 100:
            continue
        M_test_e_f = M_test_f[eligible_f]

        rand_means: list[float] = []
        for s in range(3):
            adj_r = _random_topk_adj(n_f, k, seed=s)
            r, _ = _cohort_retention(M_test_e_f, adj_r)
            rand_means.append(float(np.nanmean(r)))
        rand_mean = float(np.mean(rand_means))

        r_jac, _ = _cohort_retention(M_test_e_f, adj_jac)
        r_bm25, _ = _cohort_retention(M_test_e_f, adj_bm25)
        m_jac = float(np.nanmean(r_jac))
        m_bm25 = float(np.nanmean(r_bm25))
        results.append({
            "thr": thr, "n_emails": n_f,
            "eligible_users": int(eligible_f.sum()),
            "rand": rand_mean,
            "jac": m_jac, "bm25": m_bm25,
            "jac_lift": m_jac / max(rand_mean, 1e-9),
            "bm25_lift": m_bm25 / max(rand_mean, 1e-9),
        })
        print(
            f"    thr>={thr}: emails={n_f}  users={eligible_f.sum():,}  "
            f"random={rand_mean:.3f}  "
            f"Jac_lift={m_jac/max(rand_mean,1e-9):.2f}x  "
            f"BM25_lift={m_bm25/max(rand_mean,1e-9):.2f}x"
        )

    if results:
        thrs = [r["thr"] for r in results]
        n_em = [r["n_emails"] for r in results]
        x_pos = np.arange(len(results))
        axes[1, 0].plot(x_pos, [r["jac_lift"] for r in results],
                        marker="o", color="#1e3a8a", linewidth=2,
                        label="Jaccard held-out")
        axes[1, 0].plot(x_pos, [r["bm25_lift"] for r in results],
                        marker="o", color="#a855f7", linewidth=2, label="BM25")
        axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=0.6)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(
            [f">={t:,}\n(n={ne:,})" for t, ne in zip(thrs, n_em)], fontsize=8.5,
        )
        axes[1, 0].set_xlabel("Filter: emails with >= n_sends")
        axes[1, 0].set_ylabel("Lift over random")
        axes[1, 0].set_title("Lift as we drop low-volume emails")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].bar(x_pos, n_em, color="#94a3b8")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f">={t:,}" for t in thrs], fontsize=8.5)
        axes[1, 1].set_xlabel("Filter: emails with >= n_sends")
        axes[1, 1].set_ylabel("Emails remaining")
        axes[1, 1].set_title("Email count at each threshold")
        axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("Q25. Outlier check: do low-volume emails distort the lift?")
    fig.tight_layout()

    def _trim(s: str, n_chars: int) -> str:
        s = s.replace("\r", " ").replace("\n", " ")
        return (s[:n_chars] + "...") if len(s) > n_chars else s

    low_vol_idx = np.argsort(n_sends)[:12]
    table_low = (
        "<table style='font-size:12px; border-collapse:collapse; width:100%;'>"
        "<tr><th>sends</th><th>actioners</th><th>train_act</th><th>test_act</th>"
        "<th>sender</th><th>subject</th></tr>"
    )
    for i in low_vol_idx:
        table_low += (
            f"<tr>"
            f"<td style='text-align:right'>{int(n_sends[i]):,}</td>"
            f"<td style='text-align:right'>{int(n_actioners[i]):,}</td>"
            f"<td style='text-align:right'>{int(n_train_act[i]):,}</td>"
            f"<td style='text-align:right'>{int(n_test_act[i]):,}</td>"
            f"<td>{html.escape(_trim(senders[i], 30))}</td>"
            f"<td>{html.escape(_trim(subjects[i], 90))}</td>"
            f"</tr>"
        )
    table_low += "</table>"

    rows_html = "".join(
        f"<tr><td>>={r['thr']:,}</td>"
        f"<td>{r['n_emails']:,}</td>"
        f"<td>{r['eligible_users']:,}</td>"
        f"<td>{r['rand']:.3f}</td>"
        f"<td>{r['jac']:.3f} <b>({r['jac_lift']:.2f}x)</b></td>"
        f"<td>{r['bm25']:.3f} <b>({r['bm25_lift']:.2f}x)</b></td></tr>"
        for r in results
    )

    pct_below_100 = 100 * float((n_sends < 100).mean())
    pct_below_1000 = 100 * float((n_sends < 1000).mean())

    prose = (
        f"Are low-volume outlier emails distorting the lift?<br><br>"
        f"<b>Per-email send distribution</b> ({n:,} aligned emails): "
        f"p10 = <b>{int(np.percentile(n_sends, 10)):,}</b>, "
        f"median = <b>{int(np.median(n_sends)):,}</b>, "
        f"p90 = <b>{int(np.percentile(n_sends, 90)):,}</b>, "
        f"max = <b>{int(n_sends.max()):,}</b>. "
        f"<b>{pct_below_100:.1f}%</b> of emails have <100 sends; "
        f"<b>{pct_below_1000:.1f}%</b> have <1,000 sends.<br><br>"
        f"<b>Lift at progressively stricter send thresholds</b> "
        f"(held-out, eligibility test n_actioned >= 2):"
        f"<table style='font-size:12px; border-collapse:collapse; margin-top:6px;'>"
        f"<tr><th>filter</th><th>emails</th><th>users</th>"
        f"<th>random</th><th>Jaccard held-out</th><th>BM25</th></tr>"
        f"{rows_html}</table><br>"
        "<b>How to read this:</b><ul>"
        "<li>If lifts stay roughly the same as we filter -- low-volume "
        "emails weren't driving the result; the signal is robust.</li>"
        "<li>If lifts collapse -- our headline numbers were carried by "
        "tiny outlier emails, and the conclusion needs caveats.</li>"
        "</ul>"
        "<b>12 lowest-volume emails (likely test sends, niche segments, or "
        "very recent / very old):</b>"
        f"<br>{table_low}"
    )
    return Section(
        "outlier_check",
        "Q25. Outlier check: are low-volume emails distorting the lift?",
        prose, _fig_to_b64(fig),
    )


def _section_top_quartile_deep_dive(users: pl.DataFrame) -> Section:
    """Q26. Top quartile of actioners — what % of received emails do they action?"""
    actioners = users.filter(pl.col("n_actioned") > 0)
    counts = actioners["n_actioned"].to_numpy().astype(float)
    qs = np.quantile(counts, [0.25, 0.50, 0.75])

    df = (
        actioners
        .with_columns(
            pl.when(pl.col("n_actioned") <= int(qs[0])).then(pl.lit("Q1"))
            .when(pl.col("n_actioned") <= int(qs[1])).then(pl.lit("Q2"))
            .when(pl.col("n_actioned") <= int(qs[2])).then(pl.lit("Q3"))
            .otherwise(pl.lit("Q4"))
            .alias("tier")
        )
        .with_columns(
            (pl.col("n_actioned") / pl.col("n_sent")).alias("action_rate"),
            (pl.col("n_opened") / pl.col("n_sent")).alias("open_rate"),
            (pl.col("n_verified_opened") / pl.col("n_sent")).alias("vo_rate"),
            (pl.col("n_clicked") / pl.col("n_sent")).alias("click_rate"),
        )
    )

    tier_summary = (
        df.group_by("tier")
        .agg(
            pl.len().alias("users"),
            pl.col("n_actioned").min().alias("n_act_min"),
            pl.col("n_actioned").max().alias("n_act_max"),
            pl.col("n_sent").mean().alias("avg_sent"),
            pl.col("n_actioned").mean().alias("avg_actioned"),
            pl.col("action_rate").mean().alias("mean_act_rate"),
            pl.col("action_rate").median().alias("median_act_rate"),
            pl.col("action_rate").quantile(0.10).alias("p10_act_rate"),
            pl.col("action_rate").quantile(0.90).alias("p90_act_rate"),
            pl.col("open_rate").mean().alias("mean_open_rate"),
            pl.col("vo_rate").mean().alias("mean_vo_rate"),
            pl.col("click_rate").mean().alias("mean_click_rate"),
        )
        .sort("tier")
    )

    q4 = df.filter(pl.col("tier") == "Q4")
    rate_q4 = q4["action_rate"].to_numpy()
    sent_q4 = q4["n_sent"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    upper_x = min(float(np.percentile(rate_q4, 99)) * 1.05, 1.0)
    bins = np.linspace(0, upper_x, 50)
    axes[0, 0].hist(rate_q4, bins=bins, color="#16a34a", edgecolor="white")
    axes[0, 0].axvline(np.median(rate_q4), color="red", linestyle="--", linewidth=1.5,
                       label=f"median = {np.median(rate_q4)*100:.2f}%")
    axes[0, 0].axvline(np.mean(rate_q4), color="orange", linestyle="--", linewidth=1.5,
                       label=f"mean = {np.mean(rate_q4)*100:.2f}%")
    axes[0, 0].set_xlabel("Action rate (n_actioned / n_sent)")
    axes[0, 0].set_ylabel(f"Q4 users (n={len(rate_q4):,})")
    axes[0, 0].set_title("Q4 action-rate distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    boxes = []
    labels = []
    for tier_val in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df.filter(pl.col("tier") == tier_val)
        if sub.height > 0:
            r = sub["action_rate"].to_numpy()
            boxes.append(r)
            n_act_lo = int(sub["n_actioned"].min())
            n_act_hi = int(sub["n_actioned"].max())
            labels.append(f"{tier_val}\nn_act [{n_act_lo}-{n_act_hi}]\nusers={sub.height:,}")
    bp = axes[0, 1].boxplot(
        boxes, tick_labels=labels, showfliers=False, patch_artist=True,
        medianprops=dict(color="black"),
    )
    tier_colors = ["#a78bfa", "#60a5fa", "#fbbf24", "#16a34a"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(tier_colors[i])
    axes[0, 1].set_ylabel("Action rate")
    axes[0, 1].set_title("Action rate by tier")
    axes[0, 1].grid(axis="y", alpha=0.3)

    rows = list(tier_summary.iter_rows(named=True))
    metrics = ["mean_open_rate", "mean_vo_rate", "mean_click_rate", "mean_act_rate"]
    metric_labels = ["open", "verified open", "click", "action"]
    metric_colors = ["#3b82f6", "#0ea5e9", "#f59e0b", "#10b981"]
    bar_w = 0.2
    x = np.arange(len(rows))
    for i, (m, lab, col) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [r[m] for r in rows]
        axes[1, 0].bar(x + (i - 1.5) * bar_w, vals, width=bar_w, color=col, label=lab)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([r["tier"] for r in rows])
    axes[1, 0].set_ylabel("Mean per-user rate")
    axes[1, 0].set_title("Funnel rates by tier (open → VO → click → action)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].scatter(sent_q4, rate_q4, s=4, alpha=0.3, color="#16a34a")
    axes[1, 1].set_xlabel("Emails received (log)")
    axes[1, 1].set_ylabel("Action rate")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(f"Q4 users: action rate vs inbox size (n={len(sent_q4):,})")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Q26. Top quartile user deep dive — what % of received emails do they action?")
    fig.tight_layout()

    print("  Q26 tier summary:")
    for r in rows:
        print(
            f"    {r['tier']}: users={int(r['users']):,}  "
            f"n_act=[{int(r['n_act_min'])}-{int(r['n_act_max'])}]  "
            f"avg_sent={r['avg_sent']:.0f}  avg_actioned={r['avg_actioned']:.1f}  "
            f"mean_act_rate={r['mean_act_rate']*100:.2f}%  "
            f"median_act_rate={r['median_act_rate']*100:.2f}%"
        )

    table_rows = "".join(
        f"<tr><td><b>{r['tier']}</b></td>"
        f"<td>{int(r['users']):,}</td>"
        f"<td>{int(r['n_act_min'])}–{int(r['n_act_max'])}</td>"
        f"<td>{r['avg_sent']:.0f}</td>"
        f"<td>{r['avg_actioned']:.1f}</td>"
        f"<td>{r['mean_open_rate']*100:.1f}%</td>"
        f"<td>{r['mean_vo_rate']*100:.1f}%</td>"
        f"<td>{r['mean_click_rate']*100:.1f}%</td>"
        f"<td><b>{r['mean_act_rate']*100:.2f}%</b></td>"
        f"<td>{r['median_act_rate']*100:.2f}%</td>"
        f"<td>{r['p10_act_rate']*100:.2f}%</td>"
        f"<td>{r['p90_act_rate']*100:.2f}%</td></tr>"
        for r in rows
    )

    p10_q4 = float(np.percentile(rate_q4, 10)) * 100
    p25_q4 = float(np.percentile(rate_q4, 25)) * 100
    p90_q4 = float(np.percentile(rate_q4, 90)) * 100
    avg_sent_q4 = float(q4["n_sent"].mean())
    avg_act_q4 = float(q4["n_actioned"].mean())

    prose = (
        f"<b>Top quartile (Q4) — answer:</b><br>"
        f"<ul>"
        f"<li>Mean action rate = <b>{np.mean(rate_q4)*100:.2f}%</b> "
        f"of received emails</li>"
        f"<li>Median = <b>{np.median(rate_q4)*100:.2f}%</b></li>"
        f"<li>p10 = <b>{p10_q4:.2f}%</b>, p25 = <b>{p25_q4:.2f}%</b>, "
        f"p90 = <b>{p90_q4:.2f}%</b></li>"
        f"<li>{q4.height:,} users in Q4; on average they receive "
        f"<b>{avg_sent_q4:.0f}</b> emails and action <b>{avg_act_q4:.1f}</b> "
        f"of them</li></ul>"
        f"Tier-summary table (quartiles of <code>n_actioned</code> among "
        f"the {actioners.height:,} actioners):"
        "<table style='font-size:12px; border-collapse:collapse; margin-top:6px;'>"
        "<tr><th>tier</th><th>users</th><th>n_actioned</th>"
        "<th>avg sent</th><th>avg actioned</th>"
        "<th>open</th><th>VO</th><th>click</th>"
        "<th>mean action %</th><th>median</th><th>p10</th><th>p90</th></tr>"
        f"{table_rows}</table><br>"
        "<b>Reading the chart</b>:<ul>"
        "<li><b>Top-left</b>: Q4 action-rate histogram. Most Q4 users "
        "action a single-digit percent of their inbox; a small fraction "
        "action much more.</li>"
        "<li><b>Top-right</b>: action rates across all four tiers. Note "
        "the median doesn't grow monotonically with tier -- mid tiers can "
        "have higher rates because they receive fewer emails relative to "
        "their action count.</li>"
        "<li><b>Bottom-left</b>: full funnel rates per tier. Open and "
        "click rates trend with engagement; action rates are tighter.</li>"
        "<li><b>Bottom-right</b>: Q4 action rate vs inbox size. The "
        "scatter shows whether high-volume Q4 users action at the same "
        "rate as low-volume Q4 users (slope tells you).</li>"
        "</ul>"
    )
    return Section(
        "top_quartile",
        "Q26. Top quartile user deep dive",
        prose, _fig_to_b64(fig),
    )


_Q4_SETUP_CACHE: dict = {}


def _q4_setup(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
) -> dict:
    """Build Q4-only held-out alignment. Cached across calls."""
    if "data" in _Q4_SETUP_CACHE:
        return _Q4_SETUP_CACHE["data"]

    actioners = users.filter(pl.col("n_actioned") > 0)
    counts = actioners["n_actioned"].to_numpy().astype(float)
    q3_thr = int(np.quantile(counts, 0.75))
    q4_user_ids = (
        actioners.filter(pl.col("n_actioned") > q3_thr)["user_id"].to_numpy()
    )
    pos = {int(u): i for i, u in enumerate(user_ids)}
    keep = np.array([int(u) for u in q4_user_ids if int(u) in pos])
    user_ids_q4 = np.sort(keep)
    row_idx = np.array([pos[int(u)] for u in user_ids_q4])
    M_q4 = M[row_idx].tocsr()

    wide_q4 = wide.filter(pl.col("user_id").is_in(user_ids_q4.tolist()))
    setup = _heldout_setup(wide_q4, M_q4, email_ids, user_ids_q4, min_n=3)
    setup["q3_thr"] = q3_thr
    setup["n_q4_users"] = int(user_ids_q4.size)
    setup["user_ids_q4"] = user_ids_q4
    setup["M_q4"] = M_q4
    print(
        f"  Q4 setup: q3_thr={q3_thr}; Q4 users={user_ids_q4.size:,}; "
        f"n_emails={setup['n']}; M_train.nnz={setup['M_train_c'].nnz:,}; "
        f"M_test.nnz={setup['M_test_c'].nnz:,}"
    )
    _Q4_SETUP_CACHE["data"] = setup
    return setup


# Reference numbers from Q22 (population-level).
_POP_REF = {
    "random_retention": 0.0893,
    "lifts": {
        "random": 1.00,
        "jac_full": 5.07,
        "jac_train": 4.76,
        "dense": 1.42,
        "tfidf": 1.49,
        "bm25": 1.64,
    },
}


def _section_q4_held_out_compare(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    k: int = 5,
) -> Section:
    """Q28. Q4-only held-out comparison: behavior vs content lifts."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_train_c = setup["M_train_c"]
    M_test_c = setup["M_test_c"]
    M_full_c = setup["M_full_c"]
    n = setup["n"]
    S_dense = setup["S_dense"]
    S_tfidf = setup["S_tfidf"]
    S_bm25 = setup["S_bm25"]
    n_q4_users = setup["n_q4_users"]
    q3_thr = setup["q3_thr"]

    methods = [
        ("random",    "Random",            "#94a3b8", _random_topk_adj(n, k)),
        ("jac_full",  "Jaccard in-sample", "#60a5fa", _jaccard_topk_adj(M_full_c, k)),
        ("jac_train", "Jaccard held-out",  "#1e3a8a", _jaccard_topk_adj(M_train_c, k)),
        ("dense",     "Dense cosine",      "#16a34a", _build_topk_adj(S_dense, k)),
        ("tfidf",     "TF-IDF cosine",     "#f59e0b", _build_topk_adj(S_tfidf, k)),
        ("bm25",      "BM25 cosine",       "#a855f7", _build_topk_adj(S_bm25, k)),
    ]

    counts_test = np.asarray(M_test_c.sum(axis=1)).flatten()
    eligible = counts_test >= 2
    M_test_e = M_test_c[eligible]
    n_eligible = int(eligible.sum())
    print(f"  Q28: eligible test users (test n>=2): {n_eligible:,}")

    rets: dict[str, float] = {}
    for key, label, _color, adj in methods:
        r, _ = _cohort_retention(M_test_e, adj)
        rets[key] = float(np.nanmean(r))
        print(f"    {label:<22s} retention={rets[key]:.4f}")

    rand_q4 = max(rets["random"], 1e-9)
    pop_lifts = _POP_REF["lifts"]
    pop_random = _POP_REF["random_retention"]
    pop_retention = {k: pop_lifts[k] * pop_random for k in pop_lifts}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    keys = [m[0] for m in methods]
    labels = [m[1] for m in methods]
    colors = [m[2] for m in methods]

    x = np.arange(len(methods))
    bw = 0.36
    pop_vals = [pop_retention[k] for k in keys]
    q4_vals = [rets[k] for k in keys]
    axes[0].bar(x - bw / 2, pop_vals, width=bw, color="#cbd5e1",
                edgecolor="white", label="population")
    axes[0].bar(x + bw / 2, q4_vals, width=bw, color=colors,
                edgecolor="white", label="Q4 only")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean cohort retention")
    axes[0].set_title(
        f"Absolute retention (random: pop={pop_random:.3f}, Q4={rand_q4:.3f})"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    nr_keys = [k for k in keys if k != "random"]
    nr_labels = [labels[i] for i, k in enumerate(keys) if k != "random"]
    nr_colors = [colors[i] for i, k in enumerate(keys) if k != "random"]
    pop_l = [pop_lifts[k] for k in nr_keys]
    q4_l = [rets[k] / rand_q4 for k in nr_keys]
    x2 = np.arange(len(nr_keys))
    axes[1].bar(x2 - bw / 2, pop_l, width=bw, color="#cbd5e1",
                edgecolor="white", label="population")
    axes[1].bar(x2 + bw / 2, q4_l, width=bw, color=nr_colors,
                edgecolor="white", label="Q4 only")
    for i, (pv, qv) in enumerate(zip(pop_l, q4_l)):
        axes[1].text(x2[i] - bw / 2, pv + 0.05, f"{pv:.2f}×",
                     ha="center", fontsize=8, color="#475569")
        axes[1].text(x2[i] + bw / 2, qv + 0.05, f"{qv:.2f}×",
                     ha="center", fontsize=8, color="black")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(nr_labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Lift over random K=5")
    axes[1].set_title("Lift over random — does behavior's lead shrink in Q4?")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Q28. Q4-only head-to-head — behavior vs content "
        f"(n={n_q4_users:,} Q4 users, {n_eligible:,} eligible test users, "
        f"{n} emails)"
    )
    fig.tight_layout()

    behv_lift = rets["jac_train"] / rand_q4
    bm25_lift = rets["bm25"] / rand_q4
    dense_lift = rets["dense"] / rand_q4
    tfidf_lift = rets["tfidf"] / rand_q4
    pop_behv = pop_lifts["jac_train"]
    pop_bm25 = pop_lifts["bm25"]
    behv_drop = (behv_lift - pop_behv) / pop_behv * 100
    bm25_drop = (bm25_lift - pop_bm25) / pop_bm25 * 100

    table_rows = "".join(
        f"<tr><td>{l}</td>"
        f"<td>{pop_lifts[k]:.2f}×</td>"
        f"<td>{rets[k] / rand_q4:.2f}×</td>"
        f"<td>{(rets[k] / rand_q4 - pop_lifts[k]) / pop_lifts[k] * 100:+.1f}%</td></tr>"
        for k, l, _c in [(m[0], m[1], m[2]) for m in methods]
    )

    if behv_lift > bm25_lift * 1.1:
        verdict = (
            f"Behavior still leads inside Q4: held-out Jaccard = "
            f"<b>{behv_lift:.2f}×</b> vs BM25 <b>{bm25_lift:.2f}×</b>. "
            f"But behavior dropped <b>{behv_drop:+.1f}%</b> from population, "
            f"vs content's <b>{bm25_drop:+.1f}%</b> — the gap is closing."
        )
    elif bm25_lift > behv_lift * 1.05:
        verdict = (
            f"<b>Content overtakes behavior inside Q4.</b> "
            f"BM25 <b>{bm25_lift:.2f}×</b> &gt; held-out Jaccard "
            f"<b>{behv_lift:.2f}×</b>. This is the strongest argument for "
            f"content routing inside the dense-engagement segment."
        )
    else:
        verdict = (
            f"Behavior and content are <b>roughly tied</b> inside Q4: "
            f"Jaccard {behv_lift:.2f}× vs BM25 {bm25_lift:.2f}×. "
            f"Hybrid (Q29) should give the cleanest answer."
        )

    prose = (
        f"<b>The setup.</b> Restrict everything to Q4 users — action matrix, "
        f"train/test split, neighbor scoring, and evaluation all run within "
        f"the n={n_q4_users:,} Q4 cohort (n_actioned &gt; {q3_thr}). "
        f"Same chronological per-user split as Q22.<br><br>"
        f"<b>Q4 random baseline = {rand_q4:.3f}</b> "
        f"(population was {pop_random:.3f}). Q4 users action many more "
        f"emails, so even a random K=5 catches a much higher fraction of "
        f"their test set — the headroom for any method is smaller, which "
        f"compresses lifts.<br><br>"
        "<b>Lift comparison: population vs Q4-only</b>"
        "<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        "<tr><th>method</th><th>population lift</th><th>Q4-only lift</th>"
        "<th>Δ</th></tr>"
        f"{table_rows}</table>"
        f"{verdict}"
        "<br><br><b>Reading the chart:</b><ul>"
        "<li><b>Left</b>: absolute retention. The Q4 random baseline rises "
        "dramatically; behavior numbers stay roughly flat in absolute terms "
        "but lose their multiplier.</li>"
        "<li><b>Right</b>: side-by-side lifts. The grey bar = population "
        "lift, colored bar = Q4-only lift. The shape of the change tells "
        "you whether content routing is unique to Q4 or just smaller.</li>"
        "</ul>"
        f"<b>Numbers</b>: dense {dense_lift:.2f}×, TF-IDF {tfidf_lift:.2f}×, "
        f"BM25 {bm25_lift:.2f}×, behavior {behv_lift:.2f}×."
    )
    return Section(
        "q4_held_out_compare",
        "Q28. Q4-only head-to-head: behavior vs content",
        prose, _fig_to_b64(fig),
    )


def _section_q4_hybrid_ranking(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    k: int = 5,
) -> Section:
    """Q29. Q4-only hybrid sweep: α · Jaccard + (1-α) · BM25."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_train_c = setup["M_train_c"]
    M_test_c = setup["M_test_c"]
    n = setup["n"]
    S_bm25 = setup["S_bm25"]
    n_q4_users = setup["n_q4_users"]

    co = (M_train_c.T @ M_train_c).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -np.inf)

    J_z = _row_zscore(J)
    S_z = _row_zscore(S_bm25)

    counts_test = np.asarray(M_test_c.sum(axis=1)).flatten()
    eligible = counts_test >= 2
    M_test_e = M_test_c[eligible]
    n_eligible = int(eligible.sum())
    print(f"  Q29: n={n} emails, eligible Q4 test users={n_eligible:,}")

    rand_seeds = list(range(5))
    rand_rets = []
    for s in rand_seeds:
        adj_r = _random_topk_adj(n, k, seed=s)
        r, _ = _cohort_retention(M_test_e, adj_r)
        rand_rets.append(np.nanmean(r))
    rand_mean = float(np.mean(rand_rets))
    print(f"  Q29: Q4 random baseline (mean of {len(rand_seeds)} seeds) = {rand_mean:.4f}")

    alphas = np.linspace(0, 1, 11)
    rets: list[float] = []
    lifts: list[float] = []
    for alpha in alphas:
        score = alpha * J_z + (1 - alpha) * S_z
        adj = _build_topk_adj(score, k)
        r, _ = _cohort_retention(M_test_e, adj)
        m = float(np.nanmean(r))
        rets.append(m)
        lifts.append(m / max(rand_mean, 1e-9))
        print(f"    alpha={alpha:.2f}: retention={m:.4f}  lift={lifts[-1]:.2f}x")
    best_idx = int(np.argmax(lifts))

    pop_alpha_lifts = [
        1.63, 2.46, 3.05, 3.47, 3.76, 3.96, 4.33, 4.65, 4.74, 4.75, 4.74,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].plot(alphas, rets, marker="o", linewidth=2, color="#0ea5e9", label="Q4")
    axes[0].axhline(rand_mean, color="black", linestyle="--", linewidth=0.7,
                    label=f"Q4 random ({rand_mean:.3f})")
    axes[0].set_xlabel("α (weight on behavior; (1-α) on BM25 content)")
    axes[0].set_ylabel("Mean cohort retention (Q4 test)")
    axes[0].set_title("Q4 hybrid retention")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(alphas, lifts, marker="o", linewidth=2, color="#0ea5e9",
                 label="Q4 only")
    axes[1].plot(alphas, pop_alpha_lifts, marker="s", linewidth=2,
                 color="#cbd5e1", label="population (Q23)")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].scatter([alphas[best_idx]], [lifts[best_idx]], color="red",
                    marker="*", s=260, zorder=6,
                    label=f"Q4 best α={alphas[best_idx]:.2f}: {lifts[best_idx]:.2f}×")
    axes[1].set_xlabel("α (weight on behavior; (1-α) on BM25 content)")
    axes[1].set_ylabel("Lift over random K=5")
    axes[1].set_title("Lift comparison: Q4 vs population α-sweep")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Q29. Q4 hybrid — α · Jaccard(held-out) + (1-α) · BM25 (n={n_q4_users:,} Q4 users)"
    )
    fig.tight_layout()

    pure_content = lifts[0]
    pure_behavior = lifts[-1]
    gain_vs_behavior = (lifts[best_idx] - pure_behavior) / max(pure_behavior, 1e-9) * 100
    gain_vs_content = (lifts[best_idx] - pure_content) / max(pure_content, 1e-9) * 100

    if 0 < alphas[best_idx] < 1 and gain_vs_behavior > 1.0:
        verdict = (
            f"<b>Hybrid wins inside Q4.</b> Best α = {alphas[best_idx]:.2f} "
            f"sits between pure behavior (α=1: {pure_behavior:.2f}×) and pure "
            f"content (α=0: {pure_content:.2f}×), with hybrid lift = "
            f"<b>{lifts[best_idx]:.2f}×</b> — a <b>+{gain_vs_behavior:.1f}%</b> "
            f"relative improvement over behavior alone."
        )
    elif alphas[best_idx] >= 0.9:
        verdict = (
            f"Best α = {alphas[best_idx]:.2f}. Behavior dominates inside Q4 "
            f"too. Pure-behavior {pure_behavior:.2f}× vs pure-content "
            f"{pure_content:.2f}× — content adds little incremental signal "
            f"on top of behavior at K=5 even in the dense segment."
        )
    else:
        verdict = (
            f"Best α = {alphas[best_idx]:.2f}. Content has unusual lift inside "
            f"Q4 — pure content {pure_content:.2f}× vs pure behavior "
            f"{pure_behavior:.2f}×."
        )

    prose = (
        f"<b>What we did.</b> Same hybrid sweep as Q23, restricted to Q4 "
        f"({n_q4_users:,} users). Score(i,j) = α · Jaccard(M_train) + (1-α) · "
        f"BM25 cosine, both row z-scored. Top-K=5 cohort retention on "
        f"M_test_Q4.<br><br>"
        f"<b>Q4 random baseline = {rand_mean:.3f}</b> "
        f"(population was 0.090).<br><br>"
        f"{verdict}<br><br>"
        f"<b>Sweep:</b><ul>"
        f"<li>α=0.0 (pure BM25 content) → {pure_content:.2f}×</li>"
        f"<li>α=1.0 (pure behavior) → {pure_behavior:.2f}×</li>"
        f"<li>best α={alphas[best_idx]:.2f} → "
        f"{lifts[best_idx]:.2f}× ({gain_vs_behavior:+.1f}% vs behavior; "
        f"{gain_vs_content:+.1f}% vs content)</li>"
        f"</ul>"
        f"<b>Reading:</b> the lift curve shape is the answer. If α=1 still "
        f"wins by a wide margin, content adds nothing. If best-α moves "
        f"meaningfully off 1.0, content carries unique signal even within "
        f"the dense Q4 cohort. Compare to the population Q23 curve overlaid "
        f"in grey — the population curve plateaus near α=0.8."
    )
    return Section(
        "q4_hybrid_ranking",
        "Q29. Q4-only hybrid behavior + content ranking",
        prose, _fig_to_b64(fig),
    )


def _section_q4_volume_fatigue(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    n_clusters: int = 12,
) -> Section:
    """Q37. Action rate vs inbox volume, stratified by cohort.
    Does fatigue compound the routing gain?"""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_full = setup["M_full_c"]
    email_ids_c = setup["email_ids_c"]
    user_ids_q4 = setup["user_ids_q4"]

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id = {r["email_id"]: r["dense_vector"] for r in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)] for e in email_ids_c], dtype=np.float32,
    )
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    user_action_cluster = (M_full @ C).toarray().astype(np.float32)
    n_act_per_user = user_action_cluster.sum(axis=1)
    valid = n_act_per_user > 0
    p = np.zeros_like(user_action_cluster)
    p[valid] = user_action_cluster[valid] / n_act_per_user[valid, None]
    top_share = p.max(axis=1)
    top_share[~valid] = 0.0
    cohort = np.full(len(user_ids_q4), "absent", dtype=object)
    cohort[top_share >= 0.50] = "niche"
    cohort[(top_share >= 0.25) & (top_share < 0.50)] = "mixed"
    cohort[(top_share < 0.25) & valid] = "omni"

    cohort_df = pl.DataFrame({
        "user_id": user_ids_q4.tolist(),
        "cohort": cohort.tolist(),
    })
    profile = (
        users.filter(pl.col("user_id").is_in(user_ids_q4.tolist()))
        .join(cohort_df, on="user_id", how="inner")
        .with_columns(
            (pl.col("n_actioned") / pl.col("n_sent")).alias("action_rate"),
            (pl.col("n_clicked") / pl.col("n_sent")).alias("click_rate"),
            (pl.col("n_opened") / pl.col("n_sent")).alias("open_rate"),
        )
    )

    bin_edges = np.array([0, 50, 100, 200, 400, 800, 1600, 5000])
    bin_labels = [
        "≤50", "51-100", "101-200", "201-400", "401-800", "801-1600", "1600+",
    ]

    rows: list[dict] = []
    for ck in ["niche", "mixed", "omni"]:
        sub = profile.filter(pl.col("cohort") == ck)
        if sub.height == 0:
            continue
        n_sent = sub["n_sent"].to_numpy().astype(float)
        rate = sub["action_rate"].to_numpy().astype(float)
        click_rate = sub["click_rate"].to_numpy().astype(float)
        open_rate = sub["open_rate"].to_numpy().astype(float)
        bin_ix = np.digitize(n_sent, bin_edges) - 1
        for b in range(len(bin_labels)):
            mask = bin_ix == b
            if mask.sum() < 30:
                continue
            rows.append({
                "cohort": ck,
                "bin": bin_labels[b],
                "bin_lo": bin_edges[b],
                "bin_hi": bin_edges[b + 1],
                "n": int(mask.sum()),
                "median_sent": float(np.median(n_sent[mask])),
                "mean_rate": float(rate[mask].mean()),
                "median_rate": float(np.median(rate[mask])),
                "mean_click_rate": float(click_rate[mask].mean()),
                "mean_open_rate": float(open_rate[mask].mean()),
            })

    print("  Q37 inbox-volume vs action-rate by cohort:")
    last_cohort = None
    for r in rows:
        if r["cohort"] != last_cohort:
            print(f"    --- {r['cohort']} ---")
            last_cohort = r["cohort"]
        print(
            f"      sent {r['bin']:>9s}: n={r['n']:>6,}  median_sent={r['median_sent']:.0f}  "
            f"mean_rate={r['mean_rate']*100:.2f}%  median_rate={r['median_rate']*100:.2f}%  "
            f"open={r['mean_open_rate']*100:.1f}%  click={r['mean_click_rate']*100:.1f}%"
        )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    color_map = {"niche": "#16a34a", "mixed": "#fbbf24", "omni": "#dc2626"}

    # 1. Action rate vs median n_sent (log x).
    for ck in ["niche", "mixed", "omni"]:
        cohort_rows = [r for r in rows if r["cohort"] == ck]
        if not cohort_rows:
            continue
        xs = [r["median_sent"] for r in cohort_rows]
        ys = [r["mean_rate"] * 100 for r in cohort_rows]
        ns = [r["n"] for r in cohort_rows]
        axes[0, 0].plot(xs, ys, marker="o", linewidth=2,
                        color=color_map[ck],
                        label=f"{ck} (n={sum(ns):,})")
        for x, y, n in zip(xs, ys, ns):
            axes[0, 0].text(x, y + 0.5, f"n={n//1000}k" if n >= 1000 else f"n={n}",
                            ha="center", fontsize=8, color=color_map[ck])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Inbox volume (n_sent, log scale)")
    axes[0, 0].set_ylabel("Mean action rate %")
    axes[0, 0].set_title("Action rate vs inbox volume — fatigue check")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # 2. Open + click + action funnel by inbox bin (omni cohort).
    for ck in ["niche", "mixed", "omni"]:
        cohort_rows = [r for r in rows if r["cohort"] == ck]
        if not cohort_rows:
            continue
        xs = [r["median_sent"] for r in cohort_rows]
        opens = [r["mean_open_rate"] * 100 for r in cohort_rows]
        clicks = [r["mean_click_rate"] * 100 for r in cohort_rows]
        axes[0, 1].plot(xs, opens, marker="o", linestyle="--",
                        color=color_map[ck], alpha=0.6,
                        label=f"{ck} open")
        axes[0, 1].plot(xs, clicks, marker="s", linestyle=":",
                        color=color_map[ck],
                        label=f"{ck} click")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Inbox volume (log scale)")
    axes[0, 1].set_ylabel("Rate %")
    axes[0, 1].set_title("Open vs click rate by inbox volume")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # 3. Median rate (less skew-prone) vs n_sent.
    for ck in ["niche", "mixed", "omni"]:
        cohort_rows = [r for r in rows if r["cohort"] == ck]
        if not cohort_rows:
            continue
        xs = [r["median_sent"] for r in cohort_rows]
        ys = [r["median_rate"] * 100 for r in cohort_rows]
        axes[1, 0].plot(xs, ys, marker="o", linewidth=2,
                        color=color_map[ck], label=ck)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Inbox volume (log scale)")
    axes[1, 0].set_ylabel("Median action rate %")
    axes[1, 0].set_title("Median action rate by inbox volume (skew-robust)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # 4. Action count vs send count scatter (sample).
    for ck in ["niche", "mixed", "omni"]:
        sub = profile.filter(pl.col("cohort") == ck)
        if sub.height == 0:
            continue
        ns = sub["n_sent"].to_numpy().astype(float)
        na = sub["n_actioned"].to_numpy().astype(float)
        sample = np.random.RandomState(0).choice(
            len(ns), size=min(3000, len(ns)), replace=False,
        )
        axes[1, 1].scatter(ns[sample], na[sample], s=4, alpha=0.18,
                          color=color_map[ck], label=ck)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("n_sent (log)")
    axes[1, 1].set_ylabel("n_actioned (log)")
    axes[1, 1].set_title("n_actioned vs n_sent scatter")
    axes[1, 1].legend(fontsize=9, markerscale=2)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Q37. Inbox-volume → action-rate (fatigue check) by cohort"
    )
    fig.tight_layout()

    rows_by_cohort = {ck: [r for r in rows if r["cohort"] == ck]
                      for ck in ["niche", "mixed", "omni"]}

    def _trend(rs: list[dict]) -> str:
        if len(rs) < 3:
            return "n/a"
        first_rate = rs[0]["mean_rate"]
        last_rate = rs[-1]["mean_rate"]
        delta = (last_rate - first_rate) * 100
        return f"{first_rate*100:.1f}% → {last_rate*100:.1f}% ({delta:+.1f} pp)"

    table_html = (
        "<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        "<tr><th>cohort</th><th>inbox bin</th><th>n</th>"
        "<th>median sent</th><th>mean rate</th><th>median rate</th>"
        "<th>open</th><th>click</th></tr>"
    )
    for r in rows:
        table_html += (
            f"<tr><td>{r['cohort']}</td><td>{r['bin']}</td>"
            f"<td>{r['n']:,}</td>"
            f"<td>{r['median_sent']:.0f}</td>"
            f"<td>{r['mean_rate']*100:.2f}%</td>"
            f"<td>{r['median_rate']*100:.2f}%</td>"
            f"<td>{r['mean_open_rate']*100:.1f}%</td>"
            f"<td>{r['mean_click_rate']*100:.1f}%</td></tr>"
        )
    table_html += "</table>"

    prose = (
        f"<b>The fatigue question.</b> Q36 assumes user behavior is "
        f"independent of inbox volume. If high-volume users are getting "
        f"fatigued (lower per-email rate), then routing's inbox cut adds "
        f"a behavioral lift on top of the topic-fit lift. We test this "
        f"directly by binning Q4 users by inbox volume and measuring per-"
        f"email rates within each cohort.<br><br>"
        f"<b>Cohort trends across inbox-volume bins (smallest → largest):</b><ul>"
        f"<li><b>Niche</b>: action-rate {_trend(rows_by_cohort['niche'])}</li>"
        f"<li><b>Mixed</b>: action-rate {_trend(rows_by_cohort['mixed'])}</li>"
        f"<li><b>Omnivore</b>: action-rate {_trend(rows_by_cohort['omni'])}</li>"
        f"</ul>"
        f"<b>Bin-level breakdown:</b>"
        f"{table_html}"
        f"<b>Reading:</b><ul>"
        f"<li><b>Top-left</b>: action rate vs inbox volume (log x). A "
        f"downward slope = fatigue. A flat or rising line = no fatigue effect "
        f"(more sends = more actions, not lower rate).</li>"
        f"<li><b>Top-right</b>: open + click rates. If opens drop with "
        f"volume but clicks/actions don't, it's selective engagement, not "
        f"fatigue.</li>"
        f"<li><b>Bottom-left</b>: median (skew-robust) rate. Confirms whether "
        f"the trend in mean is driven by outliers.</li>"
        f"<li><b>Bottom-right</b>: scatter of n_actioned vs n_sent. Slope &lt; 1 "
        f"on log-log = sublinear (heavy users action proportionally less).</li>"
        f"</ul>"
        f"<b>Interpretation.</b> If fatigue is real (downward slope in "
        f"top-left), the Q36 routing gain is a <i>lower bound</i>: cutting "
        f"inbox volume should also recover some per-email rate. If the "
        f"slope is flat, inbox volume is an admissions filter (heavier "
        f"engagers get more emails because they engage more, not in spite "
        f"of it) and Q36's gains are pure topical-fit gains."
    )
    return Section(
        "q4_volume_fatigue",
        "Q37. Inbox-volume → action-rate (fatigue check)",
        prose, _fig_to_b64(fig),
    )


def _section_q4_routing_summary(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
) -> Section:
    """Q38. Final synthesis: routing playbook from Q28-Q37 findings."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    cohorts = ["niche", "mixed", "omnivore"]
    colors = ["#16a34a", "#fbbf24", "#dc2626"]
    n_users = [3809, 46194, 88589]
    rate_lift_top2 = [1.87, 1.65, 1.31]
    action_capture_top2 = [91.0, 48.7, 34.7]
    inbox_cut_top2 = [100 - 61.8, 100 - 31.5, 100 - 27.8]
    policy = ["ROUTE (top-2)", "ROUTE (top-3)", "DO NOT ROUTE"]

    y = np.arange(len(cohorts))
    ax.barh(y, n_users, color=colors, alpha=0.5, edgecolor="black")
    for i, (yi, n, lift, cap, cut, pol) in enumerate(zip(
        y, n_users, rate_lift_top2, action_capture_top2, inbox_cut_top2, policy,
    )):
        ax.text(n + 1500, yi,
                f"{cohorts[i].upper()}: {n:,} users — {pol}\n"
                f"top-2 routing → {cap:.0f}% actions kept, {cut:.0f}% inbox cut, "
                f"{lift:.2f}× rate lift",
                va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([c.upper() for c in cohorts], fontsize=11, fontweight="bold")
    ax.set_xlabel("Cohort size (Q4 users)")
    ax.set_title("Q4 routing playbook — who, what, expected lift",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(n_users) * 1.55)
    fig.tight_layout()

    prose = (
        f"<b>The Q4 routing playbook</b> distilled from Q28-Q37:"
        f"<table style='font-size:12px; border-collapse:collapse; "
        f"margin:8px 0; width:100%;'>"
        f"<tr><th>cohort</th><th>n</th><th>top_share</th>"
        f"<th>policy</th><th>captured actions</th>"
        f"<th>inbox cut</th><th>rate lift</th></tr>"
        f"<tr style='background:#dcfce7;'>"
        f"<td><b>NICHE</b></td><td>3,809</td><td>≥ 50%</td>"
        f"<td><b>top-2 clusters</b></td><td>91.0%</td>"
        f"<td>38.2%</td><td><b>1.87×</b></td></tr>"
        f"<tr style='background:#fef9c3;'>"
        f"<td><b>MIXED</b></td><td>46,194</td><td>25–50%</td>"
        f"<td><b>top-3 clusters</b></td><td>62.3%</td>"
        f"<td>58.0%</td><td><b>1.55×</b></td></tr>"
        f"<tr style='background:#fee2e2;'>"
        f"<td><b>OMNIVORE</b></td><td>88,589</td><td>&lt; 25%</td>"
        f"<td><b>do not route</b></td><td>—</td>"
        f"<td>—</td><td>—</td></tr>"
        f"</table>"
        f"<b>What we learned</b><ul>"
        f"<li><b>Q28</b>: behavior's lead over content shrinks 9% inside "
        f"Q4 (4.76× → 4.35×) but doesn't collapse. Cohort-level lift is "
        f"the wrong metric for routing.</li>"
        f"<li><b>Q29</b>: hybrid still wants pure behavior at the cohort "
        f"level — content adds nothing on top of behavior in aggregate.</li>"
        f"<li><b>Q30</b>: high-Jaccard email pairs (≥ 0.30) jump <b>+61%</b> "
        f"inside Q4 — sub-tribes exist, just hidden in mean statistics.</li>"
        f"<li><b>Q31</b>: <b>3,809 niche users</b> action ≥ 50% in one "
        f"topical cluster (median action rate 93%).</li>"
        f"<li><b>Q32</b>: per-user content lift is <b>2.26× for niches, "
        f"0.21× for omnivores</b>. Content actively HURTS omnivores.</li>"
        f"<li><b>Q33</b>: top 3 of 12 clusters host 47% of niche users — "
        f"the routing prize is concentrated in a few topics.</li>"
        f"<li><b>Q34</b>: optimal hybrid α differs by cohort. Niches: "
        f"<b>α=0.75 (mostly behavior)</b>. Mixed: 0.50. Omni: 0.25 — but "
        f"omni still loses to random.</li>"
        f"<li><b>Q35</b>: niche users are long-tenured (~669d) and engage "
        f"less often than omnivores (45.8% active-30d vs 65%) but action "
        f"93% (median) when they do.</li>"
        f"<li><b>Q36</b>: top-2 cluster routing captures 91% of niche "
        f"actions, lifts rate <b>1.87×</b> while cutting inbox 38%.</li>"
        f"<li><b>Q37</b>: fatigue check — see chart above for the slope.</li>"
        f"</ul>"
        f"<b>Implementation order</b><ol>"
        f"<li><b>Niche pilot</b>: route the 3,809 niche users to top-2 "
        f"clusters from their last 30d action history. Measure lift A/B.</li>"
        f"<li><b>Mixed expansion</b>: extend to mixed cohort (46K users) "
        f"with top-3 cluster routing. Smaller per-user lift, much larger "
        f"volume.</li>"
        f"<li><b>Omnivore exemption</b>: continue sending omnivores "
        f"everything. Routing them only loses actions.</li>"
        f"</ol>"
        f"<b>Open questions for next iteration</b><ul>"
        f"<li>How quickly can we classify a user as niche from their first "
        f"N actions? (predictive cohort detection)</li>"
        f"<li>Are these clusters stable over time, or do they drift "
        f"week-to-week? (cluster recency)</li>"
        f"<li>Does the same cohort split exist in Q1-Q3 (low-action "
        f"users)? Niche-routing might also help newer users find their "
        f"topic faster.</li>"
        f"</ul>"
    )
    return Section(
        "q4_routing_summary",
        "Q38. Routing playbook — synthesis of Q28-Q37",
        prose, _fig_to_b64(fig),
    )


def _section_q4_routing_simulation(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    n_clusters: int = 12,
) -> Section:
    """Q36. Counterfactual routing simulation: send each user only their
    top-K clusters. Trade off captured actions vs reduced inbox volume."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_full = setup["M_full_c"]
    email_ids_c = setup["email_ids_c"]
    user_ids_q4 = setup["user_ids_q4"]

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id = {r["email_id"]: r["dense_vector"] for r in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)] for e in email_ids_c], dtype=np.float32,
    )
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )

    # User x cluster ACTION counts.
    user_action_cluster = (M_full @ C).toarray().astype(np.float32)

    # User x cluster SEND counts. Build send matrix from wide.
    sends_q4 = (
        wide.filter(pl.col("user_id").is_in(user_ids_q4.tolist()))
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("user_id", "email_id")
        .unique()
    )
    pos_user = {int(u): i for i, u in enumerate(user_ids_q4)}
    pos_email = {int(e): i for i, e in enumerate(email_ids_c)}
    rows = np.array([pos_user[int(u)] for u in sends_q4["user_id"].to_list()])
    cols = np.array([pos_email[int(e)] for e in sends_q4["email_id"].to_list()])
    M_sends = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids_q4), len(email_ids_c)),
    )
    user_send_cluster = (M_sends @ C).toarray().astype(np.float32)

    n_act_per_user = user_action_cluster.sum(axis=1)
    n_send_per_user = user_send_cluster.sum(axis=1)
    valid = n_act_per_user > 0

    # Cohort split (matches Q31).
    p = np.zeros_like(user_action_cluster)
    p[valid] = user_action_cluster[valid] / n_act_per_user[valid, None]
    top_share = p.max(axis=1)
    top_share[~valid] = 0.0
    cohort = np.full(len(user_ids_q4), "absent", dtype=object)
    cohort[top_share >= 0.50] = "niche"
    cohort[(top_share >= 0.25) & (top_share < 0.50)] = "mixed"
    cohort[(top_share < 0.25) & valid] = "omni"

    # Per-user, sort clusters by action count descending.
    order = np.argsort(-user_action_cluster, axis=1)
    rows_ix = np.arange(len(user_ids_q4))[:, None]
    sorted_actions = user_action_cluster[rows_ix, order]
    sorted_sends = user_send_cluster[rows_ix, order]
    cum_actions = np.cumsum(sorted_actions, axis=1)
    cum_sends = np.cumsum(sorted_sends, axis=1)

    Ks = np.arange(1, n_clusters + 1)
    capture_action = np.where(
        n_act_per_user[:, None] > 0,
        cum_actions / n_act_per_user[:, None], 0.0,
    )
    capture_send = np.where(
        n_send_per_user[:, None] > 0,
        cum_sends / n_send_per_user[:, None], 0.0,
    )
    # Per-user new action rate at top-K policy = cum_actions / cum_sends.
    new_rate = np.where(cum_sends > 0, cum_actions / cum_sends, 0.0)
    cur_rate = np.where(
        n_send_per_user > 0,
        n_act_per_user / np.maximum(n_send_per_user, 1), 0.0,
    )
    rate_lift = np.where(
        cur_rate[:, None] > 0,
        new_rate / np.maximum(cur_rate[:, None], 1e-9), 0.0,
    )

    cohort_masks = {
        "niche": (cohort == "niche"),
        "mixed": (cohort == "mixed"),
        "omni": (cohort == "omni"),
    }

    cohort_curves = {}
    for ck, mask in cohort_masks.items():
        if mask.sum() == 0:
            continue
        cohort_curves[ck] = {
            "n": int(mask.sum()),
            "capture_action": capture_action[mask].mean(axis=0),
            "capture_send": capture_send[mask].mean(axis=0),
            "new_rate": new_rate[mask].mean(axis=0),
            "cur_rate": float(cur_rate[mask].mean()),
            "rate_lift": rate_lift[mask].mean(axis=0),
        }

    print("  Q36 routing simulation (top-K clusters per user):")
    for ck, cur in cohort_curves.items():
        print(f"    {ck} (n={cur['n']:,}, baseline rate={cur['cur_rate']*100:.2f}%):")
        for K in [1, 2, 3, 5]:
            i = K - 1
            print(
                f"      top-{K}: capture_actions={cur['capture_action'][i]*100:.1f}%, "
                f"keep_inbox={cur['capture_send'][i]*100:.1f}%, "
                f"new_rate={cur['new_rate'][i]*100:.2f}%, "
                f"rate_lift={cur['rate_lift'][i]:.2f}×"
            )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    color_map = {"niche": "#16a34a", "mixed": "#fbbf24", "omni": "#dc2626"}
    label_map = {
        "niche": f"niche (n={cohort_curves['niche']['n']:,})",
        "mixed": f"mixed (n={cohort_curves['mixed']['n']:,})",
        "omni":  f"omni (n={cohort_curves['omni']['n']:,})",
    }

    # 1. Pareto curve: actions captured vs inbox kept.
    for ck in ["niche", "mixed", "omni"]:
        if ck not in cohort_curves:
            continue
        cur = cohort_curves[ck]
        axes[0, 0].plot(
            cur["capture_send"] * 100, cur["capture_action"] * 100,
            marker="o", linewidth=2, color=color_map[ck], label=label_map[ck],
        )
        for K in [1, 2, 3]:
            i = K - 1
            axes[0, 0].text(
                cur["capture_send"][i] * 100 + 1,
                cur["capture_action"][i] * 100 - 1,
                f"K={K}", fontsize=9, color=color_map[ck],
            )
    axes[0, 0].plot([0, 100], [0, 100], color="black", linestyle=":", linewidth=0.7,
                    label="no-routing diagonal (proportional)")
    axes[0, 0].set_xlabel("% of inbox kept")
    axes[0, 0].set_ylabel("% of actions captured")
    axes[0, 0].set_title("Pareto: actions captured vs inbox kept (top-K clusters)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # 2. New action rate at top-K.
    for ck in ["niche", "mixed", "omni"]:
        if ck not in cohort_curves:
            continue
        cur = cohort_curves[ck]
        axes[0, 1].plot(
            Ks, cur["new_rate"] * 100,
            marker="o", linewidth=2, color=color_map[ck], label=label_map[ck],
        )
        axes[0, 1].axhline(cur["cur_rate"] * 100, color=color_map[ck],
                          linestyle="--", linewidth=0.7, alpha=0.7)
    axes[0, 1].set_xlabel("Top-K clusters per user")
    axes[0, 1].set_ylabel("Action rate %")
    axes[0, 1].set_title("Action rate at top-K policy (dashed = current rate)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xticks(Ks)

    # 3. Rate lift at top-K.
    for ck in ["niche", "mixed", "omni"]:
        if ck not in cohort_curves:
            continue
        cur = cohort_curves[ck]
        axes[1, 0].plot(
            Ks, cur["rate_lift"],
            marker="o", linewidth=2, color=color_map[ck], label=label_map[ck],
        )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1, 0].set_xlabel("Top-K clusters per user")
    axes[1, 0].set_ylabel("Action rate lift (top-K rate / current rate)")
    axes[1, 0].set_title("Rate lift over current (dashed = parity)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xticks(Ks)

    # 4. Inbox reduction vs action loss (efficiency frontier).
    for ck in ["niche", "mixed", "omni"]:
        if ck not in cohort_curves:
            continue
        cur = cohort_curves[ck]
        # Inbox reduction = 1 - capture_send; action loss = 1 - capture_action.
        axes[1, 1].plot(
            (1 - cur["capture_send"]) * 100,
            (1 - cur["capture_action"]) * 100,
            marker="o", linewidth=2, color=color_map[ck], label=label_map[ck],
        )
        for K in [1, 2, 3]:
            i = K - 1
            axes[1, 1].text(
                (1 - cur["capture_send"][i]) * 100 + 1,
                (1 - cur["capture_action"][i]) * 100,
                f"K={K}", fontsize=9, color=color_map[ck],
            )
    axes[1, 1].set_xlabel("% inbox reduction (emails NOT sent)")
    axes[1, 1].set_ylabel("% actions lost")
    axes[1, 1].set_title("Cost frontier: inbox cut vs actions lost (lower is better)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Q36. Counterfactual routing: top-K clusters per user "
        f"(K=1..{n_clusters}, K-means on dense embeddings)"
    )
    fig.tight_layout()

    table_html = (
        "<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        "<tr><th>cohort</th><th>K</th><th>actions kept</th>"
        "<th>inbox kept</th><th>new rate</th><th>rate lift</th></tr>"
    )
    for ck in ["niche", "mixed", "omni"]:
        if ck not in cohort_curves:
            continue
        cur = cohort_curves[ck]
        for K in [1, 2, 3]:
            i = K - 1
            table_html += (
                f"<tr><td><b>{ck}</b></td><td>{K}</td>"
                f"<td>{cur['capture_action'][i]*100:.1f}%</td>"
                f"<td>{cur['capture_send'][i]*100:.1f}%</td>"
                f"<td>{cur['new_rate'][i]*100:.2f}%</td>"
                f"<td><b>{cur['rate_lift'][i]:.2f}×</b></td></tr>"
            )
    table_html += "</table>"

    if "niche" in cohort_curves:
        n_cur = cohort_curves["niche"]
        verdict = (
            f"<b>Niche routing (top-1 cluster only)</b>: keeps "
            f"<b>{n_cur['capture_action'][0]*100:.1f}%</b> of their actions "
            f"while cutting inbox to <b>{n_cur['capture_send'][0]*100:.1f}%</b> — "
            f"action rate jumps from <b>{n_cur['cur_rate']*100:.1f}%</b> to "
            f"<b>{n_cur['new_rate'][0]*100:.1f}%</b> "
            f"(<b>{n_cur['rate_lift'][0]:.2f}×</b>) for a population of "
            f"{n_cur['n']:,} users."
        )
    else:
        verdict = ""

    prose = (
        f"<b>The counterfactual.</b> Take each Q4 user's actual sends and "
        f"actions. Sort clusters by their action count for that user. The "
        f"\"top-K\" policy keeps only emails in the user's top-K clusters; "
        f"the rest are not sent. We then ask:<ul>"
        f"<li>How many of their actual actions did we still capture?</li>"
        f"<li>How much did we cut their inbox volume?</li>"
        f"<li>What's their new action rate?</li></ul>"
        f"This is an <i>upper bound</i> assumption (it presumes their "
        f"actions don't change when we send less) but gives a clean read "
        f"on the routing trade-off — how much volume can we cut without "
        f"losing engagement?<br><br>"
        f"<b>Cohort outcomes at K=1, 2, 3:</b>"
        f"{table_html}"
        f"{verdict}<br><br>"
        f"<b>Reading:</b><ul>"
        f"<li><b>Top-left</b>: Pareto curve. The further above the diagonal, "
        f"the better the routing efficiency (more actions captured per "
        f"inbox unit kept).</li>"
        f"<li><b>Top-right</b>: action rate vs K. Niche dashed line is "
        f"the baseline; the curve at K=1 shows the routing prize.</li>"
        f"<li><b>Bottom-left</b>: rate lift. K=1 = how much routing improves "
        f"action rate per email sent.</li>"
        f"<li><b>Bottom-right</b>: cost frontier. Niches sit close to the "
        f"x-axis (cut a lot of inbox, lose few actions); omnivores rise "
        f"steeply (you can't cut inbox without losing actions).</li>"
        f"</ul>"
        f"<b>Caveat.</b> This assumes their action behavior is independent "
        f"of inbox volume. If sending less also reduces fatigue and lifts "
        f"per-email engagement, the gains compound. If sending less reduces "
        f"surface-area for them to discover new topics, gains discount."
    )
    return Section(
        "q4_routing_simulation",
        "Q36. Counterfactual routing — top-K clusters per user",
        prose, _fig_to_b64(fig),
    )


def _section_q4_niche_tenure_recency(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    n_clusters: int = 12,
) -> Section:
    """Q35. Niche/mixed/omnivore profiles on tenure + recency + cadence."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_full = setup["M_full_c"]
    email_ids_c = setup["email_ids_c"]
    user_ids_q4 = setup["user_ids_q4"]

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id = {r["email_id"]: r["dense_vector"] for r in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)] for e in email_ids_c], dtype=np.float32,
    )
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    user_cluster = (M_full @ C).toarray().astype(np.float32)
    n_act_per_user = user_cluster.sum(axis=1)
    valid = n_act_per_user > 0
    p = np.zeros_like(user_cluster)
    p[valid] = user_cluster[valid] / n_act_per_user[valid, None]
    top_share = p.max(axis=1)
    top_share[~valid] = 0.0

    cohort = np.full(len(user_ids_q4), "absent", dtype=object)
    cohort[top_share >= 0.50] = "niche"
    cohort[(top_share >= 0.25) & (top_share < 0.50)] = "mixed"
    cohort[(top_share < 0.25) & valid] = "omni"

    cohort_df = pl.DataFrame({
        "user_id": user_ids_q4.tolist(),
        "top_share": top_share.tolist(),
        "cohort": cohort.tolist(),
    })

    # Last engagement (click OR action) per user.
    engaged = (
        wide.filter(pl.col("user_id").is_in(user_ids_q4.tolist()))
        .filter(pl.col("clicked") | pl.col("actioned"))
        .group_by("user_id")
        .agg(pl.col("sent_at").max().alias("last_engaged_at"))
    )
    profile = (
        users.filter(pl.col("user_id").is_in(user_ids_q4.tolist()))
        .join(cohort_df, on="user_id", how="inner")
        .join(engaged, on="user_id", how="left")
        .with_columns(
            ((pl.lit(ANCHOR_UTC) - pl.col("first_sent_at"))
                .dt.total_days().alias("tenure_days")),
            ((pl.lit(ANCHOR_UTC) - pl.col("last_engaged_at"))
                .dt.total_days().alias("days_since_engaged")),
            ((pl.col("last_actioned_at") - pl.col("first_actioned_at"))
                .dt.total_days().alias("act_span_days")),
        )
        .with_columns(
            pl.when(pl.col("n_actioned") >= 2)
            .then(pl.col("act_span_days") / (pl.col("n_actioned") - 1))
            .otherwise(None)
            .alias("avg_gap_days"),
            (pl.col("n_actioned") / pl.col("n_sent")).alias("action_rate"),
        )
    )

    rows = []
    for ck in ["niche", "mixed", "omni"]:
        sub = profile.filter(pl.col("cohort") == ck)
        if sub.height == 0:
            continue
        days_eng = sub["days_since_engaged"].to_numpy().astype(float)
        days_eng = np.where(np.isnan(days_eng), 9999, days_eng)
        rows.append({
            "cohort": ck,
            "n": sub.height,
            "tenure_med": float(np.nanmedian(sub["tenure_days"].to_numpy().astype(float))),
            "tenure_p25": float(np.nanpercentile(sub["tenure_days"].to_numpy().astype(float), 25)),
            "tenure_p75": float(np.nanpercentile(sub["tenure_days"].to_numpy().astype(float), 75)),
            "recency_med": float(np.nanmedian(days_eng[days_eng < 9999])),
            "active7": float((days_eng <= 7).mean()),
            "active30": float((days_eng <= 30).mean()),
            "active90": float((days_eng <= 90).mean()),
            "n_act_med": float(np.nanmedian(sub["n_actioned"].to_numpy().astype(float))),
            "rate_med": float(np.nanmedian(sub["action_rate"].to_numpy().astype(float))),
            "gap_med": float(np.nanmedian(
                sub["avg_gap_days"].drop_nulls().to_numpy().astype(float)
            )) if sub["avg_gap_days"].drop_nulls().shape[0] else float("nan"),
        })

    print("  Q35 cohort profiles:")
    for r in rows:
        print(
            f"    {r['cohort']:<6s} n={r['n']:>6,} "
            f"tenure_med={r['tenure_med']:.0f}d "
            f"recency_med={r['recency_med']:.0f}d "
            f"active30={r['active30']*100:.1f}% "
            f"n_act_med={r['n_act_med']:.0f} "
            f"rate_med={r['rate_med']*100:.1f}% "
            f"gap_med={r['gap_med']:.1f}d"
        )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    color_map = {"niche": "#16a34a", "mixed": "#fbbf24", "omni": "#dc2626"}
    label_map = {
        "niche": f"niche ≥50% (n={rows[0]['n']:,})" if rows else "",
        "mixed": f"mixed 25-50% (n={rows[1]['n']:,})" if len(rows) > 1 else "",
        "omni": f"omnivore <25% (n={rows[2]['n']:,})" if len(rows) > 2 else "",
    }

    # 1. Tenure histogram.
    bins = np.linspace(0, 1500, 60)
    for ck in ["niche", "mixed", "omni"]:
        sub = profile.filter(pl.col("cohort") == ck)
        if sub.height == 0:
            continue
        t = sub["tenure_days"].drop_nulls().to_numpy().astype(float)
        t = np.clip(t, 0, 1500)
        axes[0, 0].hist(
            t, bins=bins, density=True, alpha=0.55,
            color=color_map[ck], label=label_map[ck],
        )
    axes[0, 0].set_xlabel("Tenure (days since first send)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Tenure distribution by cohort")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # 2. Recency histogram.
    rbins = np.linspace(0, 365, 60)
    for ck in ["niche", "mixed", "omni"]:
        sub = profile.filter(pl.col("cohort") == ck)
        if sub.height == 0:
            continue
        d = sub["days_since_engaged"].drop_nulls().to_numpy().astype(float)
        d = d[d < 9999]
        d = np.clip(d, 0, 365)
        axes[0, 1].hist(
            d, bins=rbins, density=True, alpha=0.55,
            color=color_map[ck], label=label_map[ck],
        )
    for w, c, lbl in [(7, "#000000", "7d"), (30, "#666666", "30d")]:
        axes[0, 1].axvline(w, color=c, linestyle="--", linewidth=0.9)
    axes[0, 1].set_xlabel("Days since last click or action")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Recency distribution by cohort (7d / 30d markers)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # 3. Active fractions per cohort.
    cohorts = [r["cohort"] for r in rows]
    x = np.arange(len(cohorts))
    bw = 0.25
    a7 = [r["active7"] * 100 for r in rows]
    a30 = [r["active30"] * 100 for r in rows]
    a90 = [r["active90"] * 100 for r in rows]
    axes[1, 0].bar(x - bw, a7, bw, color="#dc2626", label="active ≤ 7d")
    axes[1, 0].bar(x,      a30, bw, color="#f59e0b", label="active ≤ 30d")
    axes[1, 0].bar(x + bw, a90, bw, color="#1e3a8a", label="active ≤ 90d")
    for i, (v7, v30, v90) in enumerate(zip(a7, a30, a90)):
        axes[1, 0].text(x[i] - bw, v7 + 1, f"{v7:.0f}%", ha="center", fontsize=9)
        axes[1, 0].text(x[i],      v30 + 1, f"{v30:.0f}%", ha="center", fontsize=9)
        axes[1, 0].text(x[i] + bw, v90 + 1, f"{v90:.0f}%", ha="center", fontsize=9)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(cohorts)
    axes[1, 0].set_ylabel("% active in window")
    axes[1, 0].set_title("Active-window penetration by cohort")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Cadence + action rate scatter (per-cohort medians + spread).
    for ck in ["niche", "mixed", "omni"]:
        sub = profile.filter(pl.col("cohort") == ck)
        if sub.height == 0:
            continue
        gap = sub["avg_gap_days"].drop_nulls().to_numpy().astype(float)
        rate = (
            sub.filter(pl.col("avg_gap_days").is_not_null())["action_rate"]
            .to_numpy().astype(float)
        )
        sample = np.random.RandomState(0).choice(
            len(gap), size=min(3000, len(gap)), replace=False,
        )
        axes[1, 1].scatter(
            np.clip(gap[sample], 0, 60), rate[sample] * 100,
            s=4, alpha=0.25, color=color_map[ck], label=label_map[ck],
        )
    axes[1, 1].set_xlabel("Avg days between actions per user")
    axes[1, 1].set_ylabel("Action rate %")
    axes[1, 1].set_xlim(0, 60)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_title("Cadence vs action-rate by cohort")
    axes[1, 1].legend(fontsize=9, markerscale=2)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Q35. Niche / mixed / omnivore profiles — tenure, recency, cadence"
    )
    fig.tight_layout()

    table_html = (
        "<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        "<tr><th>cohort</th><th>n</th><th>tenure med</th>"
        "<th>recency med</th><th>active 7d</th><th>active 30d</th>"
        "<th>active 90d</th><th>n_act med</th><th>rate med</th>"
        "<th>cadence med</th></tr>"
    )
    for r in rows:
        table_html += (
            f"<tr><td><b>{r['cohort']}</b></td>"
            f"<td>{r['n']:,}</td>"
            f"<td>{r['tenure_med']:.0f}d</td>"
            f"<td>{r['recency_med']:.0f}d</td>"
            f"<td>{r['active7']*100:.1f}%</td>"
            f"<td>{r['active30']*100:.1f}%</td>"
            f"<td>{r['active90']*100:.1f}%</td>"
            f"<td>{r['n_act_med']:.0f}</td>"
            f"<td>{r['rate_med']*100:.1f}%</td>"
            f"<td>{r['gap_med']:.1f}d</td></tr>"
        )
    table_html += "</table>"

    if rows:
        niche_r = next((r for r in rows if r["cohort"] == "niche"), None)
        omni_r = next((r for r in rows if r["cohort"] == "omni"), None)
        if niche_r and omni_r:
            tenure_diff = niche_r["tenure_med"] - omni_r["tenure_med"]
            recency_ratio = niche_r["active30"] / max(omni_r["active30"], 1e-9)
            verdict = (
                f"<b>Niche vs omnivore profile</b>:<ul>"
                f"<li><b>Tenure</b>: niches {niche_r['tenure_med']:.0f}d vs "
                f"omnivores {omni_r['tenure_med']:.0f}d "
                f"({tenure_diff:+.0f}d). "
                f"{'Niches are NEWER' if tenure_diff < 0 else 'Niches are LONGER tenured'}.</li>"
                f"<li><b>Active-30d penetration</b>: niches "
                f"{niche_r['active30']*100:.1f}% vs omnivores "
                f"{omni_r['active30']*100:.1f}% ({recency_ratio:.2f}× ratio). "
                f"{'Niches are MORE recently active.' if recency_ratio > 1.05 else 'Niches are LESS or equally recent.'}</li>"
                f"<li><b>Cadence</b>: niches "
                f"{niche_r['gap_med']:.1f}d between actions, omnivores "
                f"{omni_r['gap_med']:.1f}d.</li>"
                f"<li><b>Action rate (median)</b>: niches "
                f"{niche_r['rate_med']*100:.1f}%, omnivores "
                f"{omni_r['rate_med']*100:.1f}%.</li>"
                f"</ul>"
            )
        else:
            verdict = ""
    else:
        verdict = ""

    prose = (
        f"<b>Question.</b> Are niche users a particular kind of person — "
        f"long-tenured topical experts, or recently-activated specialists "
        f"following a single issue? Comparing tenure, recency, cadence, and "
        f"action rate across the cohorts answers that.<br><br>"
        f"<b>Cohort profiles (median values):</b>"
        f"{table_html}"
        f"{verdict}<br>"
        f"<b>Reading:</b><ul>"
        f"<li><b>Top-left</b>: tenure histograms by cohort. If niches "
        f"are recent activations, the green distribution is left-shifted.</li>"
        f"<li><b>Top-right</b>: recency. The dotted lines mark 7-day and "
        f"30-day windows. A right-shifted distribution = many users haven't "
        f"engaged in months (likely churned).</li>"
        f"<li><b>Bottom-left</b>: active-window penetration. A high green "
        f"7d/30d bar means niches are concentrated in recent engagement.</li>"
        f"<li><b>Bottom-right</b>: cadence (days between actions) vs action "
        f"rate. Niches with tight cadence + high action rate cluster in "
        f"the upper-left.</li>"
        f"</ul>"
    )
    return Section(
        "q4_niche_tenure_recency",
        "Q35. Niche / mixed / omnivore tenure + recency profile",
        prose, _fig_to_b64(fig),
    )


def _section_q4_hybrid_per_cohort(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    k: int = 5, n_clusters: int = 12,
) -> Section:
    """Q34. Per-user hybrid α sweep stratified by niche/mixed/omnivore.
    Does optimal α differ across cohorts?"""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_train = setup["M_train_c"]
    M_test = setup["M_test_c"]
    M_full = setup["M_full_c"]
    email_ids_c = setup["email_ids_c"]

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id = {r["email_id"]: r["dense_vector"] for r in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)] for e in email_ids_c], dtype=np.float32,
    )
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    n_train = np.asarray(M_train.sum(axis=1)).flatten()
    n_test = np.asarray(M_test.sum(axis=1)).flatten()
    eligible = (n_train >= 2) & (n_test >= 2)
    M_train_e = M_train[eligible].tocsr()
    M_test_e = M_test[eligible].tocsr()
    M_full_e = M_full[eligible].tocsr()
    n_users_e = M_train_e.shape[0]

    # Cohort split (matches Q31/Q32).
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    user_cluster = (M_full_e @ C).toarray().astype(np.float32)
    n_act = user_cluster.sum(axis=1)
    valid = n_act > 0
    p = np.zeros_like(user_cluster)
    p[valid] = user_cluster[valid] / n_act[valid, None]
    top_share = p.max(axis=1)
    niche = top_share >= 0.50
    mixed = (top_share >= 0.25) & (top_share < 0.50)
    omni = top_share < 0.25

    # Item-item Jaccard from M_train.
    co = (M_train_e.T @ M_train_e).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, 0)

    # Per-user content centroid.
    centroid = (M_train_e @ V) / n_train[eligible][:, None]
    centroid /= (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9)

    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cohort_masks: dict[str, np.ndarray] = {
        "all": np.ones(n_users_e, dtype=bool),
        "niche": niche,
        "mixed": mixed,
        "omni": omni,
    }
    n_per: dict[str, int] = {k_: int(v.sum()) for k_, v in cohort_masks.items()}
    hits = {k_: np.zeros(len(alphas), dtype=np.float64) for k_ in cohort_masks}

    chunk = 4000
    print(f"  Q34: per-user hybrid sweep over {n_users_e:,} users x {len(alphas)} alphas")

    for start in range(0, n_users_e, chunk):
        end = min(start + chunk, n_users_e)
        idx = slice(start, end)

        b_block = (M_train_e[idx] @ J).astype(np.float32)
        c_block = (centroid[idx] @ V.T).astype(np.float32)
        train_block = M_train_e[idx].toarray() > 0
        test_block = M_test_e[idx].toarray() > 0

        # Row z-score with train masked out (replaced by row mean = 0 effectively).
        b_safe = np.where(train_block, np.nan, b_block)
        c_safe = np.where(train_block, np.nan, c_block)
        b_mean = np.nanmean(b_safe, axis=1, keepdims=True)
        b_std = np.nanstd(b_safe, axis=1, keepdims=True) + 1e-9
        c_mean = np.nanmean(c_safe, axis=1, keepdims=True)
        c_std = np.nanstd(c_safe, axis=1, keepdims=True) + 1e-9
        b_z = (b_block - b_mean) / b_std
        c_z = (c_block - c_mean) / c_std
        b_z[train_block] = -np.inf
        c_z[train_block] = -np.inf

        rows = np.arange(end - start)[:, None]
        for ai, alpha in enumerate(alphas):
            score = alpha * b_z + (1 - alpha) * c_z
            topk = np.argpartition(-score, k, axis=1)[:, :k]
            row_hits = test_block[rows, topk].sum(axis=1)
            for ck, mask in cohort_masks.items():
                m_chunk = mask[start:end]
                hits[ck][ai] += int(row_hits[m_chunk].sum())
        if start % 20000 == 0:
            print(f"    Q34: processed {end:,}/{n_users_e:,}")

    p_at_k = {
        ck: hits[ck] / max(n_per[ck] * k, 1) for ck in cohort_masks
    }
    print("  Q34 precision@5 by cohort and alpha:")
    print(f"    alpha:       " + "  ".join(f"{a:.2f}" for a in alphas))
    for ck in ["all", "niche", "mixed", "omni"]:
        print(
            f"    {ck:<6s}  n={n_per[ck]:>6,}: "
            + "  ".join(f"{v*100:.2f}%" for v in p_at_k[ck])
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    cohort_colors = {
        "all": "#1e3a8a", "niche": "#16a34a",
        "mixed": "#fbbf24", "omni": "#dc2626",
    }
    cohort_labels = {
        "all": f"all (n={n_per['all']:,})",
        "niche": f"niche (n={n_per['niche']:,})",
        "mixed": f"mixed (n={n_per['mixed']:,})",
        "omni": f"omnivore (n={n_per['omni']:,})",
    }
    for ck in ["all", "niche", "mixed", "omni"]:
        axes[0].plot(alphas, p_at_k[ck] * 100,
                     marker="o", linewidth=2,
                     color=cohort_colors[ck], label=cohort_labels[ck])
    axes[0].set_xlabel("α (weight on behavior; (1-α) on content)")
    axes[0].set_ylabel("Per-user precision@5 %")
    axes[0].set_title("Absolute precision@5 by α and cohort")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right panel: lift over each cohort's α=0 baseline (relative shape).
    for ck in ["all", "niche", "mixed", "omni"]:
        baseline = max(p_at_k[ck][0], 1e-9)
        axes[1].plot(alphas, p_at_k[ck] / baseline,
                     marker="o", linewidth=2,
                     color=cohort_colors[ck], label=cohort_labels[ck])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7,
                    label="α=0 baseline (pure content)")
    axes[1].set_xlabel("α (weight on behavior; (1-α) on content)")
    axes[1].set_ylabel("p@5(α) / p@5(α=0)")
    axes[1].set_title("Shape of α curve by cohort")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Q34. Per-user hybrid α sweep stratified by niche/mixed/omnivore"
    )
    fig.tight_layout()

    best_alpha = {ck: float(alphas[int(np.argmax(p_at_k[ck]))])
                  for ck in cohort_masks}
    table_rows = []
    for ck in ["all", "niche", "mixed", "omni"]:
        cells = "".join(f"<td>{p_at_k[ck][i]*100:.2f}%</td>" for i in range(len(alphas)))
        table_rows.append(
            f"<tr><td><b>{ck}</b></td><td>{n_per[ck]:,}</td>{cells}"
            f"<td><b>α={best_alpha[ck]:.2f}</b></td></tr>"
        )

    if best_alpha["niche"] < best_alpha["omni"]:
        verdict = (
            f"<b>Niche prefers more content; omnivores prefer more behavior.</b> "
            f"Best α: niche = {best_alpha['niche']:.2f} (more content-weighted) "
            f"vs omnivore = {best_alpha['omni']:.2f} (more behavior-weighted). "
            f"This validates segment-aware blending: route niches by content, "
            f"route omnivores by collaborative-filtering."
        )
    elif best_alpha["niche"] == best_alpha["omni"]:
        verdict = (
            f"Best α is the same across cohorts ({best_alpha['niche']:.2f}). "
            f"Either signal mix is universal or the granularity here is too coarse."
        )
    else:
        verdict = (
            f"<b>Counter-intuitive</b>: niche best α = {best_alpha['niche']:.2f}, "
            f"omnivore best α = {best_alpha['omni']:.2f}. Niches prefer more "
            f"behavior than omnivores — suggests their behavior history "
            f"alone is highly predictive."
        )

    prose = (
        f"<b>Question.</b> If niche users want more content and omnivores "
        f"want more behavior (or vice versa), the optimal hybrid α should "
        f"differ across cohorts. We test that by running a per-user "
        f"hybrid sweep <code>α · behavior(item-item Jaccard) + (1-α) · "
        f"content(centroid cosine)</code>, both row z-scored, and "
        f"measuring precision@K=5 per cohort.<br><br>"
        f"<b>Precision@5 by cohort and α:</b>"
        f"<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        f"<tr><th>cohort</th><th>n</th>"
        + "".join(f"<th>α={a:.2f}</th>" for a in alphas)
        + "<th>best α</th></tr>"
        + "".join(table_rows)
        + "</table>"
        f"{verdict}<br><br>"
        f"<b>Reading:</b><ul>"
        f"<li><b>Left</b>: absolute precision@5 vs α. Niche curve "
        f"shape vs omnivore curve shape tells you whether they want "
        f"different rankers.</li>"
        f"<li><b>Right</b>: each cohort normalized by its α=0 (pure "
        f"content) value. The shape difference is the routing-policy "
        f"signal independent of absolute precision.</li>"
        f"</ul>"
    )

    return Section(
        "q4_hybrid_per_cohort",
        "Q34. Per-user hybrid α sweep by cohort — segment-aware routing",
        prose, _fig_to_b64(fig),
    )


def _section_q4_niche_cluster_topics(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    n_clusters: int = 12, n_examples: int = 4,
) -> Section:
    """Q33. What ARE the niche topical clusters? Top subjects + senders per cluster,
    action volume, niche-cohort assignment, action rate."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_full_c = setup["M_full_c"]
    email_ids_c = setup["email_ids_c"]
    user_ids_q4 = setup["user_ids_q4"]
    n_q4_users = setup["n_q4_users"]

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector", "subject", "sender")
    )
    by_id = {row["email_id"]: row for row in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)]["dense_vector"] for e in email_ids_c],
        dtype=np.float32,
    )
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    subjects = [by_id[int(e)]["subject"] or "" for e in email_ids_c]
    senders = [by_id[int(e)]["sender"] or "" for e in email_ids_c]

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)

    # Per-email Q4 action volume.
    pop_per_email = np.asarray(M_full_c.sum(axis=0)).flatten()

    # Per-user cluster distribution.
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    user_cluster = (M_full_c @ C).toarray().astype(np.float32)
    n_act_per_user = user_cluster.sum(axis=1)
    valid_user = n_act_per_user > 0
    p = np.zeros_like(user_cluster)
    p[valid_user] = user_cluster[valid_user] / n_act_per_user[valid_user, None]
    top_share = p.max(axis=1)
    top_cluster = p.argmax(axis=1)
    top_share[~valid_user] = 0.0
    top_cluster[~valid_user] = -1

    niche_mask = top_share >= 0.50

    # Build per-cluster summary.
    rate_map = {
        int(r["user_id"]): (float(r["n_actioned"]) / max(int(r["n_sent"]), 1))
        for r in users.filter(
            pl.col("user_id").is_in(user_ids_q4.tolist())
        ).iter_rows(named=True)
    }
    rates_q4 = np.array([rate_map.get(int(u), 0.0) for u in user_ids_q4])

    cluster_info: list[dict] = []
    for c in range(n_clusters):
        idx_emails = np.where(labels == c)[0]
        n_emails = int(idx_emails.size)
        n_actions = int(pop_per_email[idx_emails].sum())
        n_users_top = int((top_cluster == c).sum())
        n_niche_top = int(((top_cluster == c) & niche_mask).sum())
        # Niche cohort with this cluster as #1: their action rate.
        niche_in_cluster = (top_cluster == c) & niche_mask
        rate_niche = (
            float(rates_q4[niche_in_cluster].mean())
            if niche_in_cluster.sum() else 0.0
        )
        # Top subjects by within-cluster popularity.
        sorted_idx = idx_emails[np.argsort(-pop_per_email[idx_emails])]
        top_subjects = []
        seen = set()
        for ei in sorted_idx:
            s = (subjects[ei] or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            top_subjects.append((s, int(pop_per_email[ei])))
            if len(top_subjects) >= n_examples:
                break
        # Top senders.
        sender_pop: dict[str, int] = {}
        for ei in idx_emails:
            sd = (senders[ei] or "").strip()
            if not sd:
                continue
            sender_pop[sd] = sender_pop.get(sd, 0) + int(pop_per_email[ei])
        top_senders = sorted(sender_pop.items(), key=lambda kv: -kv[1])[:3]

        cluster_info.append({
            "cluster": c,
            "n_emails": n_emails,
            "n_actions": n_actions,
            "n_users_top": n_users_top,
            "n_niche_top": n_niche_top,
            "rate_niche_pct": rate_niche * 100,
            "top_subjects": top_subjects,
            "top_senders": top_senders,
        })

    cluster_info.sort(key=lambda r: -r["n_niche_top"])

    print(f"  Q33: n_clusters={n_clusters}; emails={V.shape[0]}; niche users with cluster #1 by cluster:")
    for ci in cluster_info:
        print(
            f"    c={ci['cluster']:>2d}  emails={ci['n_emails']:>3d}  "
            f"actions={ci['n_actions']:>9,d}  niche_top={ci['n_niche_top']:>5,}  "
            f"rate_niche={ci['rate_niche_pct']:.1f}%"
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cs = [ci["cluster"] for ci in cluster_info]
    n_niche_arr = [ci["n_niche_top"] for ci in cluster_info]
    n_users_arr = [ci["n_users_top"] for ci in cluster_info]
    rate_arr = [ci["rate_niche_pct"] for ci in cluster_info]
    x = np.arange(len(cs))

    axes[0].bar(x, n_users_arr, color="#cbd5e1", edgecolor="white",
                label="Q4 users with cluster as #1")
    axes[0].bar(x, n_niche_arr, color="#16a34a", edgecolor="white",
                label="niche subset (top ≥50%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"c{c}" for c in cs])
    axes[0].set_ylabel("Q4 users")
    axes[0].set_title("Cluster size in user-space (sorted by niche count)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, rate_arr, color="#16a34a", edgecolor="white")
    for xi, v, n in zip(x, rate_arr, n_niche_arr):
        if n > 0:
            axes[1].text(xi, v + 1, f"{v:.0f}%\nn={n:,}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"c{c}" for c in cs])
    axes[1].set_ylabel("Mean action rate %")
    axes[1].set_title("Niche cohort action rate per cluster")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Q33. What ARE the niche topical clusters? K={n_clusters} from dense embeddings"
    )
    fig.tight_layout()

    def _esc(s: str) -> str:
        return html.escape(s[:120])

    cluster_blocks = []
    for ci in cluster_info:
        if ci["n_niche_top"] == 0 and ci["n_users_top"] == 0:
            continue
        subjects_html = (
            "<ul style='margin:4px 0 6px 16px; padding:0;'>"
            + "".join(
                f"<li><i>{_esc(s)}</i> — {n:,} Q4 actions</li>"
                for s, n in ci["top_subjects"]
            )
            + "</ul>"
        )
        senders_html = ", ".join(
            f"<code>{_esc(sd)}</code> ({n:,})" for sd, n in ci["top_senders"]
        ) or "—"
        cluster_blocks.append(
            f"<div style='border:1px solid #cbd5e1; border-radius:6px; "
            f"padding:8px 12px; margin:6px 0; font-size:12px;'>"
            f"<b>Cluster {ci['cluster']}</b> — "
            f"{ci['n_emails']} emails, "
            f"<b>{ci['n_actions']:,}</b> Q4 actions, "
            f"<b>{ci['n_niche_top']:,}</b> niche users (out of "
            f"{ci['n_users_top']:,} total Q4 users with this as #1) — "
            f"niche action rate <b>{ci['rate_niche_pct']:.1f}%</b>"
            f"<br><b>Top subjects:</b>{subjects_html}"
            f"<br><b>Top senders:</b> {senders_html}"
            f"</div>"
        )
    clusters_html = "\n".join(cluster_blocks)

    total_niche = sum(ci["n_niche_top"] for ci in cluster_info)
    top3_niche = sum(ci["n_niche_top"] for ci in cluster_info[:3])
    top3_share = top3_niche / max(total_niche, 1) * 100

    prose = (
        f"<b>What are the niche topics?</b> We label each of the K={n_clusters} "
        f"KMeans clusters using its top subjects and senders, and report the "
        f"size + action rate of the niche cohort that lives in each cluster.<br><br>"
        f"<b>Concentration:</b> top 3 clusters host "
        f"<b>{top3_niche:,} of {total_niche:,} niche users ({top3_share:.0f}%)</b> — "
        f"the niche prize is concentrated in a few topics, not spread uniformly.<br><br>"
        f"<b>Reading the chart:</b><ul>"
        f"<li><b>Left</b>: cluster size in user-space, sorted descending by "
        f"niche-user count. Grey = all Q4 users with cluster as their #1; "
        f"green = niche subset (top ≥ 50%).</li>"
        f"<li><b>Right</b>: action rate of the niche cohort per cluster. "
        f"Differences across clusters tell you which topics produce the "
        f"highest-engagement specialists.</li></ul>"
        f"<b>Cluster cards</b> (sorted by niche-user count):"
        f"{clusters_html}"
    )

    return Section(
        "q4_niche_cluster_topics",
        "Q33. Niche cluster characterization — what topics ARE these niches?",
        prose, _fig_to_b64(fig),
    )


def _section_q4_personalized_content_lift(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    k: int = 5, n_clusters: int = 12,
) -> Section:
    """Q32. Per-user content lift: precision@K from a content centroid,
    stratified by niche / mixed / omnivore (Q31 split)."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_train = setup["M_train_c"]
    M_test = setup["M_test_c"]
    M_full = setup["M_full_c"]
    email_ids_c = setup["email_ids_c"]
    n_q4_users = setup["n_q4_users"]

    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector")
    )
    by_id = {r["email_id"]: r["dense_vector"] for r in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)] for e in email_ids_c], dtype=np.float32,
    )
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n_emails = V.shape[0]

    # Eligibility for per-user simulation.
    n_train = np.asarray(M_train.sum(axis=1)).flatten()
    n_test = np.asarray(M_test.sum(axis=1)).flatten()
    eligible = (n_train >= 2) & (n_test >= 2)
    M_train_e = M_train[eligible].tocsr()
    M_test_e = M_test[eligible].tocsr()
    M_full_e = M_full[eligible].tocsr()
    n_test_e = n_test[eligible]
    n_users_e = M_train_e.shape[0]
    print(f"  Q32: eligible per-user simulation users = {n_users_e:,}")

    user_centroid = (M_train_e @ V) / n_train[eligible][:, None]
    user_centroid /= (np.linalg.norm(user_centroid, axis=1, keepdims=True) + 1e-9)

    # Niche/mixed/omnivore split using full Q4 actions (matches Q31).
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    user_cluster = (M_full_e @ C).toarray().astype(np.float32)
    n_act_e = user_cluster.sum(axis=1)
    valid = n_act_e > 0
    p = np.zeros_like(user_cluster)
    p[valid] = user_cluster[valid] / n_act_e[valid, None]
    top_share = p.max(axis=1)
    top_share[~valid] = 0.0
    niche = top_share >= 0.50
    mixed = (top_share >= 0.25) & (top_share < 0.50)
    omni = top_share < 0.25

    # Precision@K with content vs random; chunked.
    chunk = 4000
    topk_hits = np.zeros(n_users_e, dtype=np.float32)
    rand_hits = np.zeros(n_users_e, dtype=np.float32)

    M_train_csr_csc = M_train_e.tocsc()
    M_test_csc = M_test_e.tocsc()

    rng = np.random.RandomState(0)
    for start in range(0, n_users_e, chunk):
        end = min(start + chunk, n_users_e)
        scores = user_centroid[start:end] @ V.T
        # Block-row train mask.
        train_block = M_train_e[start:end].toarray() > 0
        test_block = M_test_e[start:end].toarray() > 0
        scores[train_block] = -np.inf
        # Top-K indices.
        topk_idx = np.argpartition(-scores, k, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        topk_hits[start:end] = test_block[rows, topk_idx].sum(axis=1) / k

        # Random K from non-train emails.
        for i in range(end - start):
            avail_mask = ~train_block[i]
            avail = np.where(avail_mask)[0]
            if avail.size < k:
                continue
            chosen = rng.choice(avail, size=k, replace=False)
            rand_hits[start + i] = test_block[i, chosen].sum() / k
        if start % 20000 == 0:
            print(f"    Q32: processed {end:,}/{n_users_e:,}")

    def _stat(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return dict(n=0, content=0.0, rand=0.0, lift=0.0,
                        n_test_med=0.0)
        return dict(
            n=int(mask.sum()),
            content=float(topk_hits[mask].mean()),
            rand=float(rand_hits[mask].mean()),
            lift=(float(topk_hits[mask].mean()) /
                  max(float(rand_hits[mask].mean()), 1e-9)),
            n_test_med=float(np.median(n_test_e[mask])),
        )

    stats = {
        "all":   _stat(np.ones(n_users_e, dtype=bool)),
        "niche": _stat(niche),
        "mixed": _stat(mixed),
        "omni":  _stat(omni),
    }
    for k_, s in stats.items():
        print(
            f"  Q32 {k_:<6s}: n={s['n']:>7,}  content_p@K={s['content']*100:.2f}%  "
            f"random_p@K={s['rand']*100:.2f}%  lift={s['lift']:.2f}×  "
            f"median_n_test={s['n_test_med']:.0f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    groups = ["all", "niche", "mixed", "omni"]
    g_labels = [
        f"all\n(n={stats['all']['n']:,})",
        f"niche ≥50%\n(n={stats['niche']['n']:,})",
        f"mixed 25-50%\n(n={stats['mixed']['n']:,})",
        f"omnivore <25%\n(n={stats['omni']['n']:,})",
    ]
    g_colors = ["#1e3a8a", "#16a34a", "#fbbf24", "#dc2626"]
    x = np.arange(len(groups))
    bw = 0.36
    cont_vals = [stats[g]["content"] * 100 for g in groups]
    rand_vals = [stats[g]["rand"] * 100 for g in groups]
    axes[0].bar(x - bw / 2, cont_vals, width=bw, color=g_colors,
                edgecolor="white", label="content top-K")
    axes[0].bar(x + bw / 2, rand_vals, width=bw, color="#cbd5e1",
                edgecolor="white", label="random K")
    for i, (c, r) in enumerate(zip(cont_vals, rand_vals)):
        axes[0].text(x[i] - bw / 2, c + 0.4, f"{c:.1f}%", ha="center", fontsize=9)
        axes[0].text(x[i] + bw / 2, r + 0.4, f"{r:.1f}%", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(g_labels, fontsize=9)
    axes[0].set_ylabel("Per-user precision@5 %")
    axes[0].set_title("Personalized content vs random per-user lift")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    lift_vals = [stats[g]["lift"] for g in groups]
    bars = axes[1].bar(x, lift_vals, color=g_colors, edgecolor="white")
    for bar, v in zip(bars, lift_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{v:.2f}×", ha="center", fontsize=10, fontweight="bold")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g_labels, fontsize=9)
    axes[1].set_ylabel("Lift over random K=5")
    axes[1].set_title("Per-user content lift (the routing prize)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Q32. Per-user personalized content lift "
        f"(eligible n={n_users_e:,} of {n_q4_users:,} Q4 users)"
    )
    fig.tight_layout()

    rows_html = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{stats[g]['n']:,}</td>"
        f"<td>{stats[g]['content']*100:.2f}%</td>"
        f"<td>{stats[g]['rand']*100:.2f}%</td>"
        f"<td><b>{stats[g]['lift']:.2f}×</b></td></tr>"
        for g, name in [
            ("all", "All eligible"),
            ("niche", "Niche ≥50%"),
            ("mixed", "Mixed 25-50%"),
            ("omni", "Omnivore <25%"),
        ]
    )

    headline = (
        f"Niche users see <b>{stats['niche']['lift']:.2f}× lift</b> "
        f"from personalized content ranking — vs omnivores' "
        f"<b>{stats['omni']['lift']:.2f}×</b>. "
        f"This is the per-user payoff Q28's cohort number can't see."
    )

    prose = (
        f"<b>The routing simulation.</b> For each Q4 user with at least 2 "
        f"train and 2 test actions:"
        f"<ol>"
        f"<li>Build their content centroid: average of their TRAIN-action "
        f"email embeddings (OpenAI dense, L2-normalized).</li>"
        f"<li>Score every email by cosine to that centroid; mask the user's "
        f"train actions to -∞.</li>"
        f"<li>Take their top-K=5 candidates → measure precision@5 against "
        f"their test actions.</li>"
        f"<li>Compare to a random K=5 baseline drawn from non-train emails.</li>"
        f"</ol>"
        f"This is the actual routing question: <i>if we ranked their next "
        f"emails by content fit to what they've already actioned, would "
        f"they action our picks?</i><br><br>"
        f"<b>Per-user precision@5 (mean over the segment):</b>"
        f"<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        f"<tr><th>segment</th><th>users</th><th>content top-K</th>"
        f"<th>random K</th><th>lift</th></tr>"
        f"{rows_html}</table>"
        f"<b>{headline}</b><br><br>"
        f"<b>Reading:</b><ul>"
        f"<li><b>All eligible Q4</b>: averages over the cohort. The cohort-"
        f"level lift looks modest because it's pulled down by omnivores.</li>"
        f"<li><b>Niche ≥ 50%</b>: users whose actions concentrate in one "
        f"topic. These are the users routing was made for. Even a small "
        f"absolute precision@5 = an enormous lift relative to random.</li>"
        f"<li><b>Mixed 25-50%</b>: weaker but still routable signal.</li>"
        f"<li><b>Omnivore &lt; 25%</b>: action everywhere; content "
        f"ranking offers little leverage because random K already picks up "
        f"their next actions.</li>"
        f"</ul>"
        f"<b>Why Q28's cohort lift was small but Q32's per-user lift is "
        f"large.</b> Q28 measured \"if we routed every email to its 5 most-"
        f"behaviorally-similar emails, how often did the same Q4 users "
        f"action both?\" That's a population-level audience-overlap test. "
        f"Q32 measures \"if we picked the top 5 emails for THIS user "
        f"based on what they've already actioned, do they action them?\" "
        f"The two metrics answer different questions; the per-user metric "
        f"is the one routing actually delivers."
    )

    return Section(
        "q4_personalized_content_lift",
        "Q32. Per-user personalized content lift (the routing prize)",
        prose, _fig_to_b64(fig),
    )


def _section_q4_topical_entropy(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
    n_clusters: int = 12,
) -> Section:
    """Q31. Per-user topical entropy inside Q4 -- niche vs omnivore split."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_full_c = setup["M_full_c"]  # Q4 users x aligned emails
    email_ids_c = setup["email_ids_c"]
    n_q4_users = setup["n_q4_users"]

    # Cluster emails on dense embeddings.
    df = (
        agg_cache.get(3)
        .filter(pl.col("email_id").is_in(email_ids_c.tolist()))
        .select("email_id", "dense_vector", "subject", "sender")
    )
    by_id = {row["email_id"]: row for row in df.iter_rows(named=True)}
    V = np.asarray(
        [by_id[int(e)]["dense_vector"] for e in email_ids_c],
        dtype=np.float32,
    )
    print(f"  Q31: clustering {V.shape[0]} emails into K={n_clusters} (KMeans on dense)")
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(V)
    cluster_size = np.bincount(labels, minlength=n_clusters)
    print(f"  Q31: cluster sizes: {cluster_size.tolist()}")

    # Per-user counts per cluster: M_full_c (n_users x n_emails) -> n_users x K
    # Build C: n_emails x K one-hot.
    C = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32),
         (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    user_cluster = (M_full_c @ C).toarray().astype(np.float32)
    n_act_per_user = user_cluster.sum(axis=1)
    keep = n_act_per_user > 0
    user_cluster = user_cluster[keep]
    n_act_per_user = n_act_per_user[keep]
    n_eval = user_cluster.shape[0]
    print(f"  Q31: evaluating {n_eval:,} Q4 users with >=1 action in aligned email set")

    p = user_cluster / n_act_per_user[:, None]
    log_p = np.log(p + 1e-12)
    H = -np.sum(p * log_p, axis=1)
    H_max = float(np.log(n_clusters))
    H_norm = H / H_max  # 0 = single-cluster, 1 = uniform.
    top_share = p.max(axis=1)
    top_cluster = p.argmax(axis=1)

    # Sub-cohorts.
    niche_mask = top_share >= 0.50
    mixed_mask = (top_share >= 0.25) & (top_share < 0.50)
    omni_mask = top_share < 0.25
    n_niche = int(niche_mask.sum())
    n_mixed = int(mixed_mask.sum())
    n_omni = int(omni_mask.sum())

    # Action rates per sub-cohort. Use Q4 users frame.
    q4_users_df = users.filter(
        pl.col("user_id").is_in(setup["user_ids_q4"].tolist())
    )
    user_ids_q4 = setup["user_ids_q4"]
    # Re-derive per-user action_rate aligned to user_cluster rows.
    pos_q4 = {int(u): i for i, u in enumerate(user_ids_q4)}
    user_ids_eval = user_ids_q4[keep]
    rate_map = {
        int(r["user_id"]): float(r["n_actioned"]) / max(int(r["n_sent"]), 1)
        for r in q4_users_df.iter_rows(named=True)
    }
    n_act_lifetime = np.array([
        int(r["n_actioned"]) for r in q4_users_df.iter_rows(named=True)
    ])
    user_id_lifetime = np.array([
        int(r["user_id"]) for r in q4_users_df.iter_rows(named=True)
    ])
    lifetime_map = dict(zip(user_id_lifetime, n_act_lifetime))

    rates_eval = np.array([rate_map.get(int(u), 0.0) for u in user_ids_eval])
    n_act_eval = np.array([lifetime_map.get(int(u), 0) for u in user_ids_eval])

    def _stats(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return dict(n=0, mean_rate=0.0, median_rate=0.0,
                        mean_n_act=0.0, median_n_act=0.0)
        return dict(
            n=int(mask.sum()),
            mean_rate=float(rates_eval[mask].mean()),
            median_rate=float(np.median(rates_eval[mask])),
            mean_n_act=float(n_act_eval[mask].mean()),
            median_n_act=float(np.median(n_act_eval[mask])),
        )

    s_niche = _stats(niche_mask)
    s_mixed = _stats(mixed_mask)
    s_omni = _stats(omni_mask)
    print(
        f"  Q31: niche  (top>=50%): n={s_niche['n']:,}  "
        f"mean_rate={s_niche['mean_rate']*100:.2f}%  "
        f"median_n_act={s_niche['median_n_act']:.0f}"
    )
    print(
        f"  Q31: mixed  (25-50%): n={s_mixed['n']:,}  "
        f"mean_rate={s_mixed['mean_rate']*100:.2f}%  "
        f"median_n_act={s_mixed['median_n_act']:.0f}"
    )
    print(
        f"  Q31: omni   (top<25%): n={s_omni['n']:,}  "
        f"mean_rate={s_omni['mean_rate']*100:.2f}%  "
        f"median_n_act={s_omni['median_n_act']:.0f}"
    )

    # Top-cluster preview (top subjects per cluster by within-cluster popularity).
    pop_in_cluster = np.zeros(n_clusters, dtype=np.float32)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        pop_in_cluster[c] = float(np.asarray(M_full_c[:, idx].sum()).flatten().sum())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Histogram of normalized entropy with sub-cohort thresholds.
    bins = np.linspace(0, 1, 50)
    axes[0, 0].hist(H_norm, bins=bins, color="#0ea5e9", edgecolor="white")
    for thr, c, lab in [
        (np.median(H_norm), "red", f"median={np.median(H_norm):.2f}"),
        (np.percentile(H_norm, 25), "orange", f"p25={np.percentile(H_norm, 25):.2f}"),
    ]:
        axes[0, 0].axvline(thr, color=c, linestyle="--", linewidth=1.2, label=lab)
    axes[0, 0].set_xlabel(f"Normalized entropy H / log({n_clusters})")
    axes[0, 0].set_ylabel(f"Q4 users (n={n_eval:,})")
    axes[0, 0].set_title("Per-user topical entropy (0 = niche, 1 = omnivore)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # 2. Top-cluster share histogram with sub-cohort coloring.
    bins2 = np.linspace(0, 1, 50)
    axes[0, 1].hist(
        [top_share[niche_mask], top_share[mixed_mask], top_share[omni_mask]],
        bins=bins2,
        color=["#16a34a", "#fbbf24", "#dc2626"],
        label=[
            f"niche ≥50% (n={n_niche:,})",
            f"mixed 25-50% (n={n_mixed:,})",
            f"omnivore <25% (n={n_omni:,})",
        ],
        stacked=True,
    )
    axes[0, 1].set_xlabel("Top-cluster share (fraction of actions in #1 topic)")
    axes[0, 1].set_ylabel("Q4 users")
    axes[0, 1].set_title("Concentration: top-cluster share")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # 3. Sub-cohort comparison: action rate.
    sub_labels = ["niche\n≥50%", "mixed\n25-50%", "omnivore\n<25%"]
    sub_n = [s_niche["n"], s_mixed["n"], s_omni["n"]]
    sub_mean_rate = [
        s_niche["mean_rate"] * 100,
        s_mixed["mean_rate"] * 100,
        s_omni["mean_rate"] * 100,
    ]
    sub_median_n_act = [
        s_niche["median_n_act"],
        s_mixed["median_n_act"],
        s_omni["median_n_act"],
    ]
    sub_colors = ["#16a34a", "#fbbf24", "#dc2626"]
    x = np.arange(3)
    bars = axes[1, 0].bar(x, sub_mean_rate, color=sub_colors, edgecolor="white")
    for bar, n, mn in zip(bars, sub_n, sub_median_n_act):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"n={n:,}\nmedian n_act={mn:.0f}",
            ha="center", fontsize=9,
        )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sub_labels)
    axes[1, 0].set_ylabel("Mean action rate %")
    axes[1, 0].set_title("Are concentrated users more engaged?")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Action rate vs entropy scatter.
    sample = np.random.RandomState(0).choice(
        n_eval, size=min(20000, n_eval), replace=False,
    )
    axes[1, 1].scatter(
        H_norm[sample], rates_eval[sample] * 100, s=3, alpha=0.18,
        color="#1e3a8a",
    )
    # Bin medians.
    bin_edges = np.linspace(0, 1, 21)
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    medians = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (H_norm >= lo) & (H_norm < hi)
        medians.append(float(np.median(rates_eval[mask]) * 100) if mask.sum() else np.nan)
    axes[1, 1].plot(bin_mid, medians, color="red", linewidth=2,
                    marker="o", markersize=5, label="binned median")
    axes[1, 1].set_xlabel(f"Normalized entropy H / log({n_clusters})")
    axes[1, 1].set_ylabel("Action rate %")
    axes[1, 1].set_ylim(0, min(float(np.percentile(rates_eval * 100, 99)), 100))
    axes[1, 1].set_title("Action rate vs concentration")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Q31. Per-user topical entropy in Q4 — who's routable? "
        f"(K={n_clusters} dense-embedding clusters, n={n_eval:,} Q4 users)"
    )
    fig.tight_layout()

    cluster_action_volume = pop_in_cluster.tolist()
    cluster_rows = "".join(
        f"<tr><td>{c}</td><td>{int(cluster_size[c])}</td>"
        f"<td>{int(cluster_action_volume[c]):,}</td>"
        f"<td>{int((top_cluster == c).sum()):,}</td></tr>"
        for c in range(n_clusters)
    )

    routable_pct = (n_niche + n_mixed) / n_eval * 100
    routable_total = n_niche + n_mixed
    rate_lift_niche = (s_niche["mean_rate"] - s_omni["mean_rate"]) / max(s_omni["mean_rate"], 1e-9) * 100

    prose = (
        f"<b>Setup.</b> Cluster the {V.shape[0]:,} aligned emails into K={n_clusters} "
        f"topical groups via KMeans on OpenAI dense embeddings. For each Q4 "
        f"user, compute the distribution of their actions over these K "
        f"clusters. Niche users action mostly one topic; omnivores spread "
        f"evenly. Two metrics:<ul>"
        f"<li><b>Normalized entropy</b> H / log(K) ∈ [0, 1]: 0 = all "
        f"actions in one cluster; 1 = uniform spread.</li>"
        f"<li><b>Top-cluster share</b>: fraction of actions in the user's "
        f"#1 topic.</li></ul>"
        f"<b>Sub-cohort split</b> (by top-cluster share):<ul>"
        f"<li><b>Niche</b> (top ≥ 50%): "
        f"{n_niche:,} users ({n_niche/n_eval*100:.1f}%) — mean action rate "
        f"<b>{s_niche['mean_rate']*100:.2f}%</b>, median lifetime "
        f"n_actioned = {s_niche['median_n_act']:.0f}</li>"
        f"<li><b>Mixed</b> (top 25–50%): "
        f"{n_mixed:,} users ({n_mixed/n_eval*100:.1f}%) — mean action rate "
        f"<b>{s_mixed['mean_rate']*100:.2f}%</b></li>"
        f"<li><b>Omnivore</b> (top &lt; 25%): "
        f"{n_omni:,} users ({n_omni/n_eval*100:.1f}%) — mean action rate "
        f"<b>{s_omni['mean_rate']*100:.2f}%</b>, median lifetime "
        f"n_actioned = {s_omni['median_n_act']:.0f}</li></ul>"
        f"<b>Routable fraction</b> (niche + mixed): "
        f"<b>{routable_total:,} of {n_eval:,} = {routable_pct:.1f}%</b> of Q4 "
        f"users have meaningful topical concentration.<br><br>"
        f"<b>Cluster sizes (emails per cluster + Q4 actions per cluster):</b>"
        f"<table style='font-size:11px; border-collapse:collapse; margin:6px 0;'>"
        f"<tr><th>cluster</th><th>n_emails</th><th>n_Q4_actions</th>"
        f"<th>users with this as #1</th></tr>"
        f"{cluster_rows}</table>"
        f"<b>Reading the chart:</b><ul>"
        f"<li><b>Top-left</b>: entropy histogram. A right-shifted distribution "
        f"means most Q4 users are omnivores; left-shifted = many specialists.</li>"
        f"<li><b>Top-right</b>: top-cluster share, stacked by sub-cohort.</li>"
        f"<li><b>Bottom-left</b>: action rate by sub-cohort. Niche users "
        f"action {s_niche['mean_rate']*100:.2f}% vs omnivores' "
        f"{s_omni['mean_rate']*100:.2f}% "
        f"({rate_lift_niche:+.1f}% lift). If positive: concentrated users "
        f"are also more engaged — routing earns its keep on them.</li>"
        f"<li><b>Bottom-right</b>: rate vs entropy scatter with binned "
        f"median (red). A downward-sloping median = niche users have "
        f"higher action rates.</li></ul>"
        f"<b>Implication for routing.</b> "
        f"If <b>{routable_pct:.0f}%</b> of Q4 are routable (niche + mixed), "
        f"that's <b>~{routable_total:,} users</b> for whom content-aware "
        f"ranking should produce real per-user lifts even though population-"
        f"level lifts compress. The remaining ~{n_omni:,} omnivores see "
        f"every email anyway — routing offers no leverage on them, and "
        f"holding back content from them costs nothing."
    )
    return Section(
        "q4_topical_entropy",
        "Q31. Per-user topical entropy in Q4 — niche vs omnivore",
        prose, _fig_to_b64(fig),
    )


def _section_q4_email_email_jaccard(
    users: pl.DataFrame, wide: pl.DataFrame,
    M: sp.csr_matrix, email_ids: np.ndarray, user_ids: np.ndarray,
) -> Section:
    """Q30. Email x email Jaccard restricted to Q4 actioners only."""
    setup = _q4_setup(users, wide, M, email_ids, user_ids)
    M_q4 = setup["M_q4"]
    n_q4_users = setup["n_q4_users"]

    def _jaccard(M_in: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        co = (M_in.T @ M_in).toarray().astype(np.float32)
        pop = np.diag(co).copy()
        union = pop[:, None] + pop[None, :] - co
        J = np.where(union > 0, co / union, 0).astype(np.float32)
        np.fill_diagonal(J, 0)
        return J, pop

    # Population matrix uses the full M (all users).
    J_pop, _ = _jaccard(M)
    J_q4, _ = _jaccard(M_q4)

    # Hierarchically reorder using Q4 distances; apply same order to both.
    dist_q4 = 1.0 - J_q4
    np.fill_diagonal(dist_q4, 0)
    dist_q4 = (dist_q4 + dist_q4.T) / 2
    Z = linkage(squareform(dist_q4, checks=False), method="average")
    order = leaves_list(Z)
    J_pop_ord = J_pop[np.ix_(order, order)]
    J_q4_ord = J_q4[np.ix_(order, order)]

    iu = np.triu_indices(J_pop.shape[0], k=1)
    pop_off = J_pop[iu]
    q4_off = J_q4[iu]
    cap_pop = max(float(np.percentile(pop_off, 99)), 0.05)
    cap_q4 = max(float(np.percentile(q4_off, 99)), 0.05)
    cap = max(cap_pop, cap_q4)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    im0 = axes[0].imshow(J_pop_ord, cmap="magma", vmin=0, vmax=cap, aspect="auto")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(
        f"Population — all users\n"
        f"off-diag mean={pop_off.mean():.4f}, p99={np.percentile(pop_off, 99):.4f}"
    )
    fig.colorbar(im0, ax=axes[0], label="Jaccard")

    im1 = axes[1].imshow(J_q4_ord, cmap="magma", vmin=0, vmax=cap, aspect="auto")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(
        f"Q4 only — n={n_q4_users:,}\n"
        f"off-diag mean={q4_off.mean():.4f}, p99={np.percentile(q4_off, 99):.4f}"
    )
    fig.colorbar(im1, ax=axes[1], label="Jaccard")
    fig.suptitle(
        f"Q30. Email × email Jaccard: population vs Q4-only (n={J_pop.shape[0]:,} emails, same ordering)"
    )
    fig.tight_layout()

    pop_high = int((pop_off >= 0.30).sum())
    q4_high = int((q4_off >= 0.30).sum())
    pop_mod = int((pop_off >= 0.10).sum())
    q4_mod = int((q4_off >= 0.10).sum())
    pop_low = int((pop_off >= 0.05).sum())
    q4_low = int((q4_off >= 0.05).sum())

    pop_p90 = float(np.percentile(pop_off, 90))
    q4_p90 = float(np.percentile(q4_off, 90))
    pop_p99 = float(np.percentile(pop_off, 99))
    q4_p99 = float(np.percentile(q4_off, 99))

    print(f"  Q30: population off-diag mean={pop_off.mean():.4f}, "
          f"p90={pop_p90:.3f}, p99={pop_p99:.3f}, max={pop_off.max():.3f}")
    print(f"  Q30: Q4-only    off-diag mean={q4_off.mean():.4f}, "
          f"p90={q4_p90:.3f}, p99={q4_p99:.3f}, max={q4_off.max():.3f}")
    print(f"  Q30: pairs >= 0.30 (high overlap): pop={pop_high:,}  Q4={q4_high:,}  "
          f"({(q4_high/max(pop_high,1)):.2f}x)")
    print(f"  Q30: pairs >= 0.10                 : pop={pop_mod:,}  Q4={q4_mod:,}  "
          f"({(q4_mod/max(pop_mod,1)):.2f}x)")
    print(f"  Q30: pairs >= 0.05                 : pop={pop_low:,}  Q4={q4_low:,}  "
          f"({(q4_low/max(pop_low,1)):.2f}x)")

    if q4_off.mean() > pop_off.mean() * 1.10:
        verdict = (
            f"<b>Q4 audiences are more cohesive.</b> The off-diagonal mean "
            f"jumps from {pop_off.mean():.4f} (population) to "
            f"{q4_off.mean():.4f} (Q4) — a "
            f"{(q4_off.mean()/pop_off.mean()-1)*100:+.0f}% increase in average "
            f"audience overlap. Tighter blocks emerge in the right panel: "
            f"Q4 users are more likely than the population to converge on "
            f"the same emails. This is good news for routing — the natural "
            f"sub-tribes are sharper."
        )
    elif q4_off.mean() < pop_off.mean() * 0.90:
        verdict = (
            f"<b>Q4 audiences are LESS cohesive.</b> Off-diag mean drops "
            f"from {pop_off.mean():.4f} → {q4_off.mean():.4f} "
            f"({(q4_off.mean()/pop_off.mean()-1)*100:+.0f}%). Q4 users span "
            f"more topics individually, so the email-email overlap is "
            f"lower in this dense cohort — they really do read everything."
        )
    else:
        verdict = (
            f"<b>Block structure is similar.</b> Off-diag mean: "
            f"{pop_off.mean():.4f} (pop) vs {q4_off.mean():.4f} (Q4) — "
            f"~{(q4_off.mean()/pop_off.mean()-1)*100:+.0f}%. Q4 doesn't "
            f"show meaningfully tighter sub-tribes than the broader population."
        )

    prose = (
        f"<b>What we're testing.</b> If Q4 users cluster into 4–8 stable "
        f"topical sub-tribes (climate / immigration / voting / ...), the "
        f"email-email Jaccard heatmap restricted to Q4 actioners should be "
        f"more <i>block-diagonal</i> than the same heatmap on the full "
        f"population. Tighter blocks = sharper routing targets.<br><br>"
        f"<b>Method.</b> Compute Jaccard between every pair of emails "
        f"using only Q4 actioners (right panel) vs all actioners (left "
        f"panel). Hierarchically reorder both heatmaps using the Q4-only "
        f"distance matrix so blocks are aligned visually. Same color scale "
        f"on both panels.<br><br>"
        f"{verdict}<br><br>"
        f"<b>High-overlap pair counts</b> (number of email pairs with "
        f"Jaccard ≥ threshold; threshold = pairs sharing that fraction of audience):"
        f"<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        f"<tr><th>threshold</th><th>population</th><th>Q4 only</th><th>Q4 / pop</th></tr>"
        f"<tr><td>≥ 0.30</td><td>{pop_high:,}</td><td>{q4_high:,}</td>"
        f"<td>{q4_high/max(pop_high,1):.2f}×</td></tr>"
        f"<tr><td>≥ 0.10</td><td>{pop_mod:,}</td><td>{q4_mod:,}</td>"
        f"<td>{q4_mod/max(pop_mod,1):.2f}×</td></tr>"
        f"<tr><td>≥ 0.05</td><td>{pop_low:,}</td><td>{q4_low:,}</td>"
        f"<td>{q4_low/max(pop_low,1):.2f}×</td></tr>"
        f"</table>"
        f"<b>Reading the chart:</b><ul>"
        f"<li>Both panels share the same email ordering (computed from "
        f"Q4 distances) and the same color scale, so the comparison is "
        f"apples-to-apples.</li>"
        f"<li>Look for <b>brighter / larger</b> block-diagonal regions on "
        f"the right panel — those are sub-tribes that exist in Q4 but get "
        f"diluted in the broader audience.</li>"
        f"<li>If both panels look similar, Q4 isn't structurally different "
        f"from the population in terms of audience overlap — the routing "
        f"target is the same overall topic structure, just observed at "
        f"higher density.</li>"
        f"</ul>"
    )
    return Section(
        "q4_email_email_jaccard",
        "Q30. Email × email Jaccard: population vs Q4 only",
        prose, _fig_to_b64(fig),
    )


def _section_q4_recency_and_patterns(
    users: pl.DataFrame, wide: pl.DataFrame
) -> Section:
    """Q27. Q4 cohort -- recency (active 30d) + behavioral patterns."""
    actioners = users.filter(pl.col("n_actioned") > 0)
    counts = actioners["n_actioned"].to_numpy().astype(float)
    q3_thr = int(np.quantile(counts, 0.75))
    q4_users = actioners.filter(pl.col("n_actioned") > q3_thr)
    q4_ids = q4_users["user_id"]

    # Last *engagement* (click OR action) per user from wide.
    engaged = (
        wide.filter(pl.col("user_id").is_in(q4_ids))
        .filter(pl.col("clicked") | pl.col("actioned"))
        .group_by("user_id")
        .agg(pl.col("sent_at").max().alias("last_engaged_at"))
    )
    q4 = (
        q4_users.join(engaged, on="user_id", how="left")
        .with_columns(
            (pl.col("n_actioned") / pl.col("n_sent")).alias("action_rate"),
            ((pl.lit(ANCHOR_UTC) - pl.col("last_engaged_at"))
                .dt.total_days().alias("days_since_engaged")),
            ((pl.lit(ANCHOR_UTC) - pl.col("first_sent_at"))
                .dt.total_days().alias("tenure_days")),
            ((pl.lit(ANCHOR_UTC) - pl.col("last_actioned_at"))
                .dt.total_days().alias("days_since_action")),
        )
    )

    n_q4 = q4.height
    days = q4["days_since_engaged"].to_numpy().astype(float)
    days = np.where(np.isnan(days), 9999, days)

    windows = [7, 14, 30, 60, 90, 180]
    win_counts = {w: int((days <= w).sum()) for w in windows}
    n_active30 = win_counts[30]
    n_dormant30 = n_q4 - n_active30

    # Cadence: among Q4 users with >=2 actions, median gap between actions.
    # Approximate using (last_action - first_action) / (n_actioned - 1).
    cadence = (
        q4.filter(pl.col("n_actioned") >= 2)
        .with_columns(
            ((pl.col("last_actioned_at") - pl.col("first_actioned_at"))
                .dt.total_days() / (pl.col("n_actioned") - 1)
            ).alias("avg_gap_days")
        )
    )
    gap_active = cadence.filter(pl.col("days_since_engaged") <= 30)["avg_gap_days"].to_numpy()
    gap_dormant = cadence.filter(pl.col("days_since_engaged") > 30)["avg_gap_days"].to_numpy()
    gap_active = gap_active[np.isfinite(gap_active)]
    gap_dormant = gap_dormant[np.isfinite(gap_dormant)]

    # Active vs dormant comparison stats.
    active = q4.filter(pl.col("days_since_engaged") <= 30)
    dormant = q4.filter(pl.col("days_since_engaged") > 30)

    def _stats(df: pl.DataFrame) -> dict:
        if df.height == 0:
            return {k: 0.0 for k in [
                "users", "n_sent", "n_actioned", "action_rate",
                "tenure_days", "n_clicked", "n_opened",
            ]}
        return {
            "users": df.height,
            "n_sent": float(df["n_sent"].mean()),
            "n_actioned": float(df["n_actioned"].mean()),
            "action_rate": float(df["action_rate"].mean()),
            "tenure_days": float(df["tenure_days"].mean()),
            "n_clicked": float(df["n_clicked"].mean()),
            "n_opened": float(df["n_opened"].mean()),
        }

    a_stats = _stats(active)
    d_stats = _stats(dormant)

    # Power-user split inside Q4: top decile by n_actioned.
    p90_act = float(np.percentile(q4["n_actioned"].to_numpy(), 90))
    power = q4.filter(pl.col("n_actioned") >= p90_act)
    rest = q4.filter(pl.col("n_actioned") < p90_act)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # 1. Recency histogram with cumulative-active overlay.
    finite_days = days[days < 9999]
    bins = np.linspace(0, min(float(np.percentile(finite_days, 99)), 730), 60)
    axes[0, 0].hist(finite_days, bins=bins, color="#16a34a", edgecolor="white")
    for w, c in zip([7, 30, 90], ["#dc2626", "#f59e0b", "#6b7280"]):
        axes[0, 0].axvline(w, color=c, linestyle="--", linewidth=1.2,
                           label=f"{w}d ({win_counts[w]:,} = {win_counts[w]/n_q4*100:.1f}%)")
    axes[0, 0].set_xlabel("Days since last click or action")
    axes[0, 0].set_ylabel(f"Q4 users (n={n_q4:,})")
    axes[0, 0].set_title("Q4 recency: how recently were they engaged?")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # 2. Active-30d vs dormant comparison (normalized bars).
    metrics = [
        ("avg n_sent", a_stats["n_sent"], d_stats["n_sent"]),
        ("avg n_actioned", a_stats["n_actioned"], d_stats["n_actioned"]),
        ("avg action rate %", a_stats["action_rate"] * 100, d_stats["action_rate"] * 100),
        ("avg tenure (days)", a_stats["tenure_days"], d_stats["tenure_days"]),
    ]
    x = np.arange(len(metrics))
    bw = 0.35
    a_vals = [m[1] for m in metrics]
    d_vals = [m[2] for m in metrics]
    # Normalize by max so bars fit on one axis.
    norms = [max(a, d, 1e-9) for a, d in zip(a_vals, d_vals)]
    a_norm = [v / n for v, n in zip(a_vals, norms)]
    d_norm = [v / n for v, n in zip(d_vals, norms)]
    axes[0, 1].bar(x - bw / 2, a_norm, bw, color="#16a34a",
                   label=f"active ≤30d (n={a_stats['users']:,})")
    axes[0, 1].bar(x + bw / 2, d_norm, bw, color="#9ca3af",
                   label=f"dormant >30d (n={d_stats['users']:,})")
    for i, (lab, av, dv) in enumerate(metrics):
        axes[0, 1].text(i - bw / 2, a_norm[i] + 0.02, f"{av:.1f}",
                        ha="center", fontsize=8)
        axes[0, 1].text(i + bw / 2, d_norm[i] + 0.02, f"{dv:.1f}",
                        ha="center", fontsize=8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([m[0] for m in metrics], fontsize=9)
    axes[0, 1].set_ylabel("normalized to active vs dormant max")
    axes[0, 1].set_ylim(0, 1.18)
    axes[0, 1].set_title("Q4 active-30d vs dormant: how do they differ?")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Cadence: avg days between actions, active vs dormant.
    if gap_active.size and gap_dormant.size:
        max_gap = float(np.percentile(np.concatenate([gap_active, gap_dormant]), 95))
        gap_bins = np.linspace(0, max(max_gap, 30), 50)
        axes[1, 0].hist(
            [np.clip(gap_active, 0, max_gap), np.clip(gap_dormant, 0, max_gap)],
            bins=gap_bins,
            color=["#16a34a", "#9ca3af"],
            label=[
                f"active (median={np.median(gap_active):.1f}d)",
                f"dormant (median={np.median(gap_dormant):.1f}d)",
            ],
            density=True,
            histtype="stepfilled",
            alpha=0.6,
        )
    axes[1, 0].set_xlabel("Avg days between actions per user")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Q4 cadence: how often do they take actions?")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # 4. Power vs rest within Q4: action rate distribution.
    rest_rate = rest["action_rate"].to_numpy()
    power_rate = power["action_rate"].to_numpy()
    upper = float(np.percentile(np.concatenate([rest_rate, power_rate]), 99))
    rate_bins = np.linspace(0, min(upper * 1.05, 1.0), 50)
    axes[1, 1].hist(
        rest_rate, bins=rate_bins, color="#60a5fa", edgecolor="white",
        label=f"Q4 rest (n={rest.height:,}, n_act<{int(p90_act)})", alpha=0.75,
    )
    axes[1, 1].hist(
        power_rate, bins=rate_bins, color="#dc2626", edgecolor="white",
        label=f"Q4 top decile (n={power.height:,}, n_act≥{int(p90_act)})", alpha=0.75,
    )
    axes[1, 1].axvline(np.median(rest_rate), color="#1e3a8a", linestyle="--",
                       linewidth=1.2, label=f"rest median={np.median(rest_rate)*100:.1f}%")
    axes[1, 1].axvline(np.median(power_rate), color="#7f1d1d", linestyle="--",
                       linewidth=1.2, label=f"power median={np.median(power_rate)*100:.1f}%")
    axes[1, 1].set_xlabel("Action rate (n_actioned / n_sent)")
    axes[1, 1].set_ylabel("Q4 users")
    axes[1, 1].set_title("Q4 power-user split (top decile vs rest)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Q27. Q4 cohort recency + patterns (n={n_q4:,}, threshold n_actioned>{q3_thr})"
    )
    fig.tight_layout()

    print(f"  Q27: Q4 size = {n_q4:,}  (threshold n_actioned > {q3_thr})")
    print(f"  Q27 active-window counts (days since click-or-action):")
    for w in windows:
        print(f"    <= {w:>3}d: {win_counts[w]:>7,}  ({win_counts[w]/n_q4*100:5.1f}%)")
    print(f"  Q27 active(30d) vs dormant comparison:")
    print(f"    active : users={a_stats['users']:,}  avg_sent={a_stats['n_sent']:.0f}  "
          f"avg_act={a_stats['n_actioned']:.1f}  rate={a_stats['action_rate']*100:.2f}%  "
          f"tenure={a_stats['tenure_days']:.0f}d")
    print(f"    dormant: users={d_stats['users']:,}  avg_sent={d_stats['n_sent']:.0f}  "
          f"avg_act={d_stats['n_actioned']:.1f}  rate={d_stats['action_rate']*100:.2f}%  "
          f"tenure={d_stats['tenure_days']:.0f}d")
    print(f"  Q27 power vs rest: power n={power.height:,} (n_act>={int(p90_act)}) "
          f"median_rate={np.median(power_rate)*100:.2f}%; "
          f"rest n={rest.height:,} median_rate={np.median(rest_rate)*100:.2f}%")
    if gap_active.size:
        print(f"  Q27 cadence: active median gap={np.median(gap_active):.1f}d, "
              f"dormant median gap={np.median(gap_dormant):.1f}d")

    win_rows = "".join(
        f"<tr><td>≤ {w}d</td><td>{win_counts[w]:,}</td>"
        f"<td>{win_counts[w]/n_q4*100:.1f}%</td></tr>"
        for w in windows
    )

    def _row(name: str, a: float, d: float, fmt: str = ".1f", suffix: str = "") -> str:
        ratio = (a / d) if d else float("inf")
        return (
            f"<tr><td>{name}</td>"
            f"<td>{a:{fmt}}{suffix}</td>"
            f"<td>{d:{fmt}}{suffix}</td>"
            f"<td>{ratio:.2f}×</td></tr>"
        )

    prose = (
        f"<b>Q4 size:</b> <b>{n_q4:,}</b> users (top quartile of actioners, "
        f"<code>n_actioned &gt; {q3_thr}</code>).<br><br>"
        f"<b>Active in last 30 days</b> (clicked OR actioned within 30d of "
        f"the {ANCHOR_UTC.date()} cutoff): "
        f"<b>{n_active30:,} users ({n_active30/n_q4*100:.1f}%)</b>; "
        f"dormant = {n_dormant30:,} ({n_dormant30/n_q4*100:.1f}%).<br><br>"
        "<b>Recency windows:</b>"
        "<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        "<tr><th>window</th><th>users</th><th>% of Q4</th></tr>"
        f"{win_rows}</table>"
        "<b>Active-30d vs dormant patterns</b> (mean per user):"
        "<table style='font-size:12px; border-collapse:collapse; margin:6px 0;'>"
        "<tr><th>metric</th><th>active ≤30d</th><th>dormant &gt;30d</th><th>active / dormant</th></tr>"
        f"{_row('users', a_stats['users'], d_stats['users'], ',.0f')}"
        f"{_row('avg n_sent', a_stats['n_sent'], d_stats['n_sent'])}"
        f"{_row('avg n_opened', a_stats['n_opened'], d_stats['n_opened'])}"
        f"{_row('avg n_clicked', a_stats['n_clicked'], d_stats['n_clicked'])}"
        f"{_row('avg n_actioned', a_stats['n_actioned'], d_stats['n_actioned'])}"
        f"{_row('avg action rate', a_stats['action_rate']*100, d_stats['action_rate']*100, '.2f', '%')}"
        f"{_row('avg tenure', a_stats['tenure_days'], d_stats['tenure_days'], '.0f', 'd')}"
        "</table>"
        "<b>Patterns within Q4</b><ul>"
        f"<li><b>Recency is heavy at the front</b>: "
        f"{win_counts[7]/n_q4*100:.1f}% engaged in last 7d, "
        f"{win_counts[30]/n_q4*100:.1f}% in last 30d, "
        f"{win_counts[90]/n_q4*100:.1f}% in last 90d. "
        f"That means <b>{(1 - win_counts[90]/n_q4)*100:.1f}%</b> of Q4 went dark "
        f"more than 90 days ago and may be effectively churned despite their lifetime action count.</li>"
        f"<li><b>Active users are bigger inboxes AND higher rates</b>: "
        f"active-30d Q4 receive ~{a_stats['n_sent']/max(d_stats['n_sent'],1):.1f}× "
        f"more emails than dormant Q4 yet still action at "
        f"{a_stats['action_rate']*100:.2f}% vs dormant's "
        f"{d_stats['action_rate']*100:.2f}% — engagement is not just retention bias.</li>"
        f"<li><b>Tenure ratio</b>: active = {a_stats['tenure_days']:.0f}d, "
        f"dormant = {d_stats['tenure_days']:.0f}d. "
        f"{'Active users are NEWER' if a_stats['tenure_days'] < d_stats['tenure_days'] else 'Active users are LONGER tenured'} "
        f"on average.</li>"
        f"<li><b>Cadence</b>: active-30d Q4 take an action every "
        f"~<b>{np.median(gap_active) if gap_active.size else float('nan'):.0f} days</b> "
        f"on average; dormant Q4 averaged "
        f"~<b>{np.median(gap_dormant) if gap_dormant.size else float('nan'):.0f} days</b> "
        f"between actions. Active users have a tighter, more regular rhythm.</li>"
        f"<li><b>Power-user tail</b>: top decile of Q4 "
        f"(n_actioned ≥ {int(p90_act)}, {power.height:,} users) action at median "
        f"<b>{np.median(power_rate)*100:.2f}%</b> vs the rest of Q4 at "
        f"<b>{np.median(rest_rate)*100:.2f}%</b> — a "
        f"~{np.median(power_rate)/max(np.median(rest_rate),1e-9):.1f}× difference.</li>"
        "</ul>"
        "<b>Reading the chart</b>:<ul>"
        "<li><b>Top-left</b>: recency histogram with 7d/30d/90d markers shows "
        "the dropoff curve from \"engaged this week\" to \"likely churned.\"</li>"
        "<li><b>Top-right</b>: active vs dormant on four metrics, normalized "
        "to whichever side is larger so bar height = relative magnitude. "
        "Raw values printed above each bar.</li>"
        "<li><b>Bottom-left</b>: action cadence (avg days between actions). "
        "Active users cluster on shorter intervals.</li>"
        "<li><b>Bottom-right</b>: within Q4, top-decile power users action at "
        "much higher rates than the rest of Q4 — Q4 itself has a long tail.</li>"
        "</ul>"
    )

    return Section(
        "q4_recency_patterns",
        "Q27. Q4 cohort recency &amp; patterns",
        prose, _fig_to_b64(fig),
    )


SECTIONS: list[Callable] = [
    _section_tenure_proxy,
    _section_actions_distribution,
    _section_lifetime_and_churn,
    _section_persona_buckets,
    _section_user_funnel,
    _section_inbox_volume,
    _section_pareto,
    _section_rfm,
    _section_heterogeneity_at_fixed_exposure,  # Q10
]
WIDE_SECTIONS: list[Callable] = [_section_action_timing]
MATRIX_SECTIONS: list[Callable] = [
    _section_pairwise_jaccard,                    # Q11
    _section_email_email_jaccard,                 # Q12
    _section_user_svd_scatter,                    # Q13
    _section_topk_cohort_retention,               # Q14
    _section_topk_retention_by_tier,              # Q15
    _section_topk_cohort_retention_cosine,        # Q16
    _section_topk_retention_by_tier_cosine,       # Q17
    _section_jaccard_vs_cosine_compare,           # Q18
    _section_topk_cohort_retention_tfidf,         # Q19
    _section_topk_cohort_retention_bm25,          # Q20
    _section_all_methods_compare,                 # Q21
    _section_held_out_jaccard_compare,            # Q22
    _section_hybrid_ranking,                      # Q23
    _section_per_email_content_lift,              # Q24
    _section_outlier_check,                       # Q25
]


# ---------------------------------------------------------------------------
# HTML rendering.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Civic Shout content routing -- user exploration</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 980px;
         margin: 32px auto; padding: 0 16px; color: #0f172a; line-height: 1.5; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  .meta {{ color: #64748b; font-size: 13px; margin-bottom: 32px; }}
  .toc {{ background: #f1f5f9; border-radius: 8px; padding: 14px 18px;
          margin-bottom: 28px; }}
  .toc ul {{ margin: 4px 0 0 0; padding-left: 22px; }}
  section {{ margin-bottom: 44px; padding-bottom: 24px;
             border-bottom: 1px solid #e2e8f0; }}
  section:last-child {{ border-bottom: 0; }}
  h2 {{ font-size: 19px; color: #0f172a; }}
  .prose {{ color: #1e293b; }}
  .prose code {{ background: #e2e8f0; padding: 1px 5px; border-radius: 3px;
                 font-size: 0.92em; }}
  img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px;
         margin-top: 12px; }}
</style>
</head>
<body>
<h1>Civic Shout content routing -- user exploration</h1>
<div class="meta">
  Source: <code>{src}</code> &middot; Rows: {rows} &middot; Users: {users}
  &middot; Emails: {emails} &middot; Anchor: 2026-04-21 UTC
</div>
<div class="toc">
  <b>Sections:</b>
  <ul>{toc}</ul>
</div>
{body}
</body>
</html>
"""


def render_html(sections: list[Section], src: Path, rows: int, users: int, emails: int) -> str:
    toc = "".join(
        f'<li><a href="#{s.slug}">{html.escape(s.heading)}</a></li>' for s in sections
    )
    body_parts = []
    for s in sections:
        body_parts.append(
            f'<section id="{s.slug}">'
            f"<h2>{html.escape(s.heading)}</h2>"
            f'<div class="prose">{s.prose}</div>'
            f'<img alt="{html.escape(s.heading)}" '
            f'src="data:image/png;base64,{s.png_b64}">'
            f"</section>"
        )
    return _HTML_TEMPLATE.format(
        src=html.escape(str(src)),
        rows=f"{rows:,}",
        users=f"{users:,}",
        emails=f"{emails:,}",
        toc=toc,
        body="\n".join(body_parts),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in",
        dest="src",
        type=Path,
        default=Path(
            "data/projects/civic_shout_content_routing/exploration/wide_engagement.parquet"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "projects/civic_shout_content_routing/exploration/exploration.html"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"loading {args.src}")
    wide = pl.read_parquet(args.src)
    print(f"  rows={wide.height:,}  emails={wide['email_id'].n_unique():,}  users={wide['user_id'].n_unique():,}")

    print("building per-user aggregate frame")
    users = build_user_frame(wide)

    sections: list[Section] = []
    for builder in SECTIONS:
        print(f"  -> {builder.__name__}")
        sections.append(builder(users))
    for builder in WIDE_SECTIONS:
        print(f"  -> {builder.__name__}")
        sections.append(builder(wide))

    print("building sparse user x email action matrix")
    M, email_ids, user_ids = build_action_matrix(wide)
    print(f"  M.shape={M.shape}  nnz={M.nnz:,}")
    if M.shape[1] >= 2:
        for builder in MATRIX_SECTIONS:
            print(f"  -> {builder.__name__}")
            if builder is _section_user_svd_scatter:
                sections.append(builder(M))
            elif builder in (
                _section_held_out_jaccard_compare,
                _section_hybrid_ranking,
                _section_per_email_content_lift,
                _section_outlier_check,
            ):
                sections.append(builder(wide, M, email_ids, user_ids))
            else:
                sections.append(builder(M, email_ids))

    # Q26 -- top-quartile deep dive renders LAST so it appears at the end of the HTML.
    print("  -> _section_top_quartile_deep_dive (Q26)")
    sections.append(_section_top_quartile_deep_dive(users))

    print("  -> _section_q4_recency_and_patterns (Q27)")
    sections.append(_section_q4_recency_and_patterns(users, wide))

    if M.shape[1] >= 2:
        print("  -> _section_q4_held_out_compare (Q28)")
        sections.append(_section_q4_held_out_compare(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_hybrid_ranking (Q29)")
        sections.append(_section_q4_hybrid_ranking(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_email_email_jaccard (Q30)")
        sections.append(_section_q4_email_email_jaccard(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_topical_entropy (Q31)")
        sections.append(_section_q4_topical_entropy(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_personalized_content_lift (Q32)")
        sections.append(_section_q4_personalized_content_lift(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_niche_cluster_topics (Q33)")
        sections.append(_section_q4_niche_cluster_topics(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_hybrid_per_cohort (Q34)")
        sections.append(_section_q4_hybrid_per_cohort(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_niche_tenure_recency (Q35)")
        sections.append(_section_q4_niche_tenure_recency(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_routing_simulation (Q36)")
        sections.append(_section_q4_routing_simulation(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_volume_fatigue (Q37)")
        sections.append(_section_q4_volume_fatigue(
            users, wide, M, email_ids, user_ids,
        ))

        print("  -> _section_q4_routing_summary (Q38)")
        sections.append(_section_q4_routing_summary(
            users, wide, M, email_ids, user_ids,
        ))

    html_doc = render_html(
        sections,
        src=args.src,
        rows=wide.height,
        users=wide["user_id"].n_unique(),
        emails=wide["email_id"].n_unique(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_doc)
    print(f"wrote {args.out}  ({len(html_doc):,} chars, {len(sections)} sections)")


if __name__ == "__main__":
    main()
