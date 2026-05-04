"""EDA: aggregate_emails v3 Rate Sanity Report.

Loads aggregate_emails_cache v3, computes six per-email engagement rates,
and writes a self-contained HTML report to:
  entities/civic_shout_engagement/analysis/aggregate-emails-rates.html

Usage:
    PYTHONPATH=$(pwd) python \
        entities/civic_shout_engagement/analysis/run_aggregate_emails_rates.py
"""

from __future__ import annotations

import base64
import io
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.figure

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from entities.civic_shout_engagement.aggregate_emails_cache import (
    cache as _agg_cache,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VO_ERA_START: datetime = datetime(2024, 3, 1, tzinfo=timezone.utc)
BUILD_CUTOFF_UTC = "2026-04-21T00:00:00Z"

OUT_DIR = Path("entities/civic_shout_engagement/analysis")
OUT_HTML = OUT_DIR / "aggregate-emails-rates.html"

# Minimum denominators for drilldown top/bottom tables
MIN_SENT_FOR_RATE = 100  # open_rate, vo_rate, click_rate, action_rate
MIN_OPENS_FOR_CPO = 50  # click_per_open
MIN_CLICKS_FOR_APC = 25  # action_per_click

# Minimum rows to draw a hexbin panel (avoids empty-axis errors)
_MIN_HEXBIN_ROWS = 10

# Anchor tolerances (expected values from prior EDA v3 build 2026-04-21)
ANCHOR_N_TOTAL = 4575
ANCHOR_N_VO_ERA = 2330
ANCHOR_N_VO_VIOLATIONS = 115
ANCHOR_TOL_TOTAL = 50
ANCHOR_TOL_VO_ERA = 100
ANCHOR_TOL_VIOLATIONS = 5

# Industry-norm plausible median ranges for advocacy email
EXPECTED_BANDS: dict[str, tuple[float, float]] = {
    "open_rate": (0.15, 0.50),
    "vo_rate": (0.10, 0.40),
    "click_rate": (0.005, 0.15),
    "action_rate": (0.001, 0.05),
    "click_per_open": (0.05, 0.40),
    "action_per_click": (0.05, 0.50),
}

RATE_LABELS: dict[str, str] = {
    "open_rate": "Open Rate",
    "vo_rate": "Verified-Open Rate",
    "click_rate": "Click Rate",
    "action_rate": "Action Rate",
    "click_per_open": "Click-per-Open",
    "action_per_click": "Action-per-Click",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Base64 helper (lifted verbatim from ds-report SKILL.md:91)
# ---------------------------------------------------------------------------


def fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# NaN helper (avoids PLR0124 "compared with itself")
# ---------------------------------------------------------------------------


def _is_nan(v: float) -> bool:
    """Return True when v is NaN."""
    return math.isnan(v)


def _as_float(v: object) -> float:
    """Narrow a polars aggregate result to float; 0.0 on None."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _as_int(v: object) -> int:
    """Narrow a polars aggregate result to int; 0 on None."""
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    return 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_compute() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load aggregate_emails v3, add rate cols, vo_era, and volume bucket."""
    df = _agg_cache.get(3)
    logger.info("Loaded %d rows from aggregate_emails v3", len(df))

    # Three-way guard for vo_rate:
    #   total_sent>0, total_vo IS NOT NULL, date>=VO_ERA_START
    vo_guard = (
        (pl.col("total_sent") > 0)
        & pl.col("total_vo").is_not_null()
        & (pl.col("date_sent") >= pl.lit(VO_ERA_START))
    )

    df = df.with_columns(
        pl.when(pl.col("total_sent") > 0)
        .then(pl.col("total_opens") / pl.col("total_sent"))
        .otherwise(None)
        .alias("open_rate"),
        pl.when(vo_guard)
        .then(pl.col("total_vo") / pl.col("total_sent"))
        .otherwise(None)
        .alias("vo_rate"),
        pl.when(pl.col("total_sent") > 0)
        .then(pl.col("total_clicks") / pl.col("total_sent"))
        .otherwise(None)
        .alias("click_rate"),
        pl.when(pl.col("total_sent") > 0)
        .then(pl.col("total_actions") / pl.col("total_sent"))
        .otherwise(None)
        .alias("action_rate"),
        pl.when(pl.col("total_opens") > 0)
        .then(pl.col("total_clicks") / pl.col("total_opens"))
        .otherwise(None)
        .alias("click_per_open"),
        pl.when(pl.col("total_clicks") > 0)
        .then(pl.col("total_actions") / pl.col("total_clicks"))
        .otherwise(None)
        .alias("action_per_click"),
        (pl.col("date_sent") >= pl.lit(VO_ERA_START)).alias("vo_era"),
    )

    # Quintile volume buckets; allow_duplicates handles heavy right tail
    sent_series = df["total_sent"]
    qcut_result = sent_series.qcut(
        quantiles=[0.2, 0.4, 0.6, 0.8],
        allow_duplicates=True,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        include_breaks=True,
    )
    # qcut with include_breaks returns struct[{breakpoint, category}]
    category_ser = qcut_result.struct.field("category")
    df = df.with_columns(category_ser.alias("send_volume_bucket"))
    breakpoints = (
        qcut_result.to_frame(name="_qcut")
        .select(
            [
                pl.col("_qcut").struct.field("breakpoint").alias("breakpoint"),
                pl.col("_qcut").struct.field("category").alias("category"),
            ]
        )
        .unique(maintain_order=False)
        .sort("breakpoint")
    )

    logger.info("send_volume_bucket breakpoints: %s", breakpoints)
    return df, breakpoints


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summary_stats(df: pl.DataFrame) -> dict[str, Any]:
    """Compute per-rate medians, P10/P90, drop counts, and invariant counts."""
    rates = list(RATE_LABELS.keys())
    stats: dict[str, Any] = {}

    stats["n_total"] = len(df)
    stats["date_min"] = df["date_sent"].min()
    stats["date_max"] = df["date_sent"].max()
    stats["n_vo_era"] = int(df["vo_era"].sum())
    stats["pct_vo_era"] = stats["n_vo_era"] / stats["n_total"] * 100
    sent_vals = df["total_sent"].drop_nulls()
    stats["median_total_sent"] = _as_float(sent_vals.median())
    stats["max_total_sent"] = _as_int(sent_vals.max())

    rate_stats: dict[str, dict[str, Any]] = {}
    for r in rates:
        col = df[r].drop_nulls()
        n_kept = len(col)
        n_dropped = stats["n_total"] - n_kept
        rate_stats[r] = {
            "median": _as_float(col.median()) if n_kept > 0 else float("nan"),
            "p10": (_as_float(col.quantile(0.10)) if n_kept > 0 else float("nan")),
            "p90": (_as_float(col.quantile(0.90)) if n_kept > 0 else float("nan")),
            "n_kept": n_kept,
            "n_dropped": n_dropped,
        }
    stats["rates"] = rate_stats

    # Hard invariant: total_vo <= total_opens
    vo_nn = pl.col("total_vo").is_not_null()
    vo_gt = pl.col("total_vo") > pl.col("total_opens")
    stats["inv_vo_gt_opens"] = int(df.filter(vo_nn & vo_gt).height)

    # Report-only incidence (not invariant violations)
    stats["inc_clicks_gt_opens"] = int(
        df.filter(pl.col("total_clicks") > pl.col("total_opens")).height
    )
    stats["inc_actions_gt_sent"] = int(
        df.filter(pl.col("total_actions") > pl.col("total_sent")).height
    )

    # Hard invariant: probability rates <= 100%
    prob_rates = ["open_rate", "vo_rate", "click_rate", "click_per_open"]
    gt100: dict[str, int] = {}
    for r in prob_rates:
        nn = pl.col(r).is_not_null()
        gt100[r] = int(df.filter(nn & (pl.col(r) > 1.0)).height)
    stats["prob_rates_gt100"] = gt100

    # Distribution tail for excluded rates
    excluded_tail: dict[str, int] = {}
    for r in ["action_rate", "action_per_click"]:
        nn = pl.col(r).is_not_null()
        excluded_tail[r] = int(df.filter(nn & (pl.col(r) > 1.0)).height)
    stats["excluded_gt100"] = excluded_tail

    # Zero-rate incidence for Q5 (high-volume) sends
    zero_inc: dict[str, Any] = {}
    for r in ["open_rate", "click_rate", "action_rate"]:
        q5 = pl.col("send_volume_bucket") == "Q5_high"
        zero_inc[r] = int(df.filter(q5 & (pl.col(r) == 0)).height)
    stats["zero_incidence_q5"] = zero_inc

    return stats


# ---------------------------------------------------------------------------
# Top / bottom drilldown tables
# ---------------------------------------------------------------------------


def top_bottom_tables(df: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return top/bottom-10 rows per rate with minimum-denominator filters."""
    result: dict[str, list[dict[str, Any]]] = {}

    def _rows(
        sub: pl.DataFrame,
        rate_col: str,
        ascending: bool,
    ) -> list[dict[str, Any]]:
        sorted_df = sub.sort(rate_col, descending=not ascending).head(10)
        return [
            {
                "email_id": row["email_id"],
                "date_sent": (str(row["date_sent"])[:10] if row["date_sent"] else ""),
                "subject": (row.get("subject") or "")[:60],
                "total_sent": row["total_sent"],
                "total_opens": row["total_opens"],
                "total_vo": row["total_vo"],
                "total_clicks": row["total_clicks"],
                "total_actions": row["total_actions"],
                "rate": round(float(row[rate_col]), 4),
            }
            for row in sorted_df.iter_rows(named=True)
        ]

    # open/vo/click/action_rate — gate on total_sent >= MIN_SENT_FOR_RATE
    for r in ["open_rate", "vo_rate", "click_rate", "action_rate"]:
        nn = pl.col(r).is_not_null()
        ge_min = pl.col("total_sent") >= MIN_SENT_FOR_RATE
        sub = df.filter(nn & ge_min)
        result[f"{r}_top"] = _rows(sub, r, ascending=False)
        result[f"{r}_bottom"] = _rows(sub, r, ascending=True)

    # click_per_open — gate on total_opens >= MIN_OPENS_FOR_CPO
    nn_cpo = pl.col("click_per_open").is_not_null()
    ge_cpo = pl.col("total_opens") >= MIN_OPENS_FOR_CPO
    sub_cpo = df.filter(nn_cpo & ge_cpo)
    result["click_per_open_top"] = _rows(sub_cpo, "click_per_open", ascending=False)
    result["click_per_open_bottom"] = _rows(sub_cpo, "click_per_open", ascending=True)

    # action_per_click — gate on total_clicks >= MIN_CLICKS_FOR_APC
    nn_apc = pl.col("action_per_click").is_not_null()
    ge_apc = pl.col("total_clicks") >= MIN_CLICKS_FOR_APC
    sub_apc = df.filter(nn_apc & ge_apc)
    result["action_per_click_top"] = _rows(sub_apc, "action_per_click", ascending=False)
    result["action_per_click_bottom"] = _rows(
        sub_apc,
        "action_per_click",
        ascending=True,
    )

    return result


# ---------------------------------------------------------------------------
# Chart A: send histogram
# ---------------------------------------------------------------------------


def chart_send_histogram(df: pl.DataFrame) -> str:
    """Chart A: histogram of total_sent on log-x with P10/P50/P90 markers."""
    s = df["total_sent"].drop_nulls()
    sent = s.filter(s > 0).to_numpy()
    p10 = float(np.percentile(sent, 10))
    p50 = float(np.percentile(sent, 50))
    p90 = float(np.percentile(sent, 90))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(
        sent,
        bins=60,
        color="#3498db",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xscale("log")
    for pct, val, color in [
        (10, p10, "#e74c3c"),
        (50, p50, "#2c3e50"),
        (90, p90, "#27ae60"),
    ]:
        ax.axvline(
            val,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"P{pct}={val:,.0f}",
        )
    ax.set_xlabel("total_sent (log scale)")
    ax.set_ylabel("Number of emails")
    ax.set_title("Chart A - Distribution of total_sent per email")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Chart B: rate panels
# ---------------------------------------------------------------------------


def chart_rate_panels(df: pl.DataFrame) -> str:
    """Chart B: 2x3 panel of rate distributions with median + expected band."""
    rates = list(RATE_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, r in zip(axes_flat, rates, strict=False):
        vals = df[r].drop_nulls().to_numpy()
        if len(vals) == 0:
            ax.set_title(RATE_LABELS[r])
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        lo, hi = EXPECTED_BANDS[r]
        band_label = f"Expected {lo * 100:.0f}-{hi * 100:.0f}%"
        ax.axvspan(lo, hi, alpha=0.15, color="#27ae60", label=band_label)
        ax.hist(
            vals,
            bins=50,
            color="#3498db",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.2,
        )
        median_val = float(np.median(vals))
        ax.axvline(
            median_val,
            color="#e74c3c",
            linestyle="--",
            linewidth=1.4,
            label=f"Median={median_val * 100:.1f}%",
        )
        ax.set_title(RATE_LABELS[r], fontsize=10)
        ax.set_xlabel("Rate")
        ax.set_ylabel("Emails")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Chart B - Per-email rate distributions (post-guard data)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Chart C: invariant stacked bar
# ---------------------------------------------------------------------------


def chart_invariants(df: pl.DataFrame, stats: dict[str, Any]) -> str:
    """Chart C: stacked bar of invariant pass/fail per check."""
    n = stats["n_total"]
    n_vo_nn = int(df.filter(pl.col("total_vo").is_not_null()).height)

    checks: list[tuple[str, int, int]] = [
        (
            "total_vo <= total_opens\n(hard invariant)",
            n_vo_nn - stats["inv_vo_gt_opens"],
            stats["inv_vo_gt_opens"],
        ),
        (
            "clicks > opens\n(report-only incidence)",
            n - stats["inc_clicks_gt_opens"],
            stats["inc_clicks_gt_opens"],
        ),
        (
            "actions > sent\n(report-only incidence)",
            n - stats["inc_actions_gt_sent"],
            stats["inc_actions_gt_sent"],
        ),
    ]
    for r in ["open_rate", "vo_rate", "click_rate", "click_per_open"]:
        n_valid = stats["rates"][r]["n_kept"]
        n_bad = stats["prob_rates_gt100"][r]
        checks.append(
            (
                f"{r} <= 100%\n(hard invariant)",
                n_valid - n_bad,
                n_bad,
            )
        )

    labels = [c[0] for c in checks]
    pass_counts = [c[1] for c in checks]
    fail_counts = [c[2] for c in checks]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = list(range(len(labels)))
    ax.barh(y, pass_counts, color="#27ae60", alpha=0.8, label="Pass / normal")
    ax.barh(
        y,
        fail_counts,
        left=pass_counts,
        color="#e74c3c",
        alpha=0.8,
        label="Violation / incidence",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of emails")
    ax.set_title("Chart C - Invariant / incidence check summary")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Chart D: monthly medians
# ---------------------------------------------------------------------------


def _fmt_month_ts(x: float, _pos: int) -> str:
    """Format a POSIX timestamp as YYYY-MM for matplotlib axis labels."""
    return datetime.fromtimestamp(x, tz=timezone.utc).strftime("%Y-%m")


def chart_monthly_medians(df: pl.DataFrame) -> str:
    """Chart D: monthly median of each rate, pre-VO-era shaded."""
    df_m = df.with_columns(pl.col("date_sent").dt.truncate("1mo").alias("month"))
    rates = list(RATE_LABELS.keys())
    agg_exprs = [pl.col(r).median().alias(r) for r in rates]
    monthly = df_m.group_by("month").agg(agg_exprs).sort("month")

    months_dt = monthly["month"].to_list()
    months_np = np.array([m.timestamp() if m else 0.0 for m in months_dt])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()
    vo_ts = VO_ERA_START.timestamp()

    for ax, r in zip(axes_flat, rates, strict=False):
        vals = monthly[r].to_numpy()
        ax.plot(
            months_np,
            vals,
            color="#3498db",
            linewidth=1.4,
            marker="o",
            markersize=3,
        )
        x_start = months_np[0] if len(months_np) > 0 else vo_ts
        ax.axvspan(x_start, vo_ts, alpha=0.1, color="#e74c3c", label="pre-VO-era")
        ax.axvline(
            vo_ts,
            color="#e74c3c",
            linestyle="--",
            linewidth=1.0,
            label="VO start 2024-03",
        )
        lo, hi = EXPECTED_BANDS[r]
        ax.axhspan(lo, hi, alpha=0.1, color="#27ae60")
        ax.set_title(RATE_LABELS[r], fontsize=10)
        ax.set_ylabel("Median rate")
        ax.set_xlabel("")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_month_ts))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        if r == "open_rate":
            ax.legend(fontsize=7)

    fig.suptitle("Chart D - Monthly median rates over time", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Chart E: rate vs volume hexbin
# ---------------------------------------------------------------------------


def chart_rate_vs_volume(df: pl.DataFrame) -> str:
    """Chart E: rate vs total_sent hexbin (one panel per rate)."""
    rates = list(RATE_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, r in zip(axes_flat, rates, strict=False):
        sub = df.filter(pl.col(r).is_not_null() & (pl.col("total_sent") > 0))
        if len(sub) < _MIN_HEXBIN_ROWS:
            ax.set_title(RATE_LABELS[r])
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        x = np.log10(sub["total_sent"].to_numpy() + 1)
        y = sub[r].to_numpy()
        hb = ax.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel("log10(total_sent)")
        ax.set_ylabel("Rate")
        ax.set_title(RATE_LABELS[r], fontsize=10)

    fig.suptitle("Chart E - Rate vs send volume (hexbin, log-x)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Chart F: funnel shape per email, faceted by send-volume quartile
# ---------------------------------------------------------------------------

_FUNNEL_RATE_COLS: tuple[str, ...] = (
    "open_rate",
    "vo_rate",
    "click_rate",
    "action_rate",
)
_FUNNEL_STAGE_COLS: tuple[str, ...] = (
    "total_sent",
    "total_opens",
    "total_vo",
    "total_clicks",
    "total_actions",
)
_FUNNEL_STAGE_LABELS: tuple[str, ...] = (
    "sent",
    "opens",
    "VO",
    "clicks",
    "actions",
)
_STD_EPSILON = 1e-6
_N_OUTLIERS_PER_QUARTILE = 3
_N_PANELS = 4


def _score_unusualness(sub: pl.DataFrame) -> pl.DataFrame:
    """Add an `unusualness` column: sum of abs z-score across the 4 rates.

    Z-scores are computed within each email's `vol_q` cohort so a high-volume
    email with a weird profile isn't punished for being in a different
    cohort. Null rates contribute 0 (not a penalty).
    """
    z_stats = sub.group_by("vol_q", maintain_order=True).agg(
        [pl.col(c).median().alias(f"{c}_q_med") for c in _FUNNEL_RATE_COLS]
        + [pl.col(c).std().alias(f"{c}_q_std") for c in _FUNNEL_RATE_COLS],
    )
    sub = sub.join(z_stats, on="vol_q", how="left")
    z_exprs: list[pl.Expr] = []
    for c in _FUNNEL_RATE_COLS:
        std_safe = (
            pl.when(pl.col(f"{c}_q_std") > _STD_EPSILON)
            .then(pl.col(f"{c}_q_std"))
            .otherwise(_STD_EPSILON)
        )
        z_raw = (pl.col(c) - pl.col(f"{c}_q_med")) / std_safe
        z_exprs.append(z_raw.abs().fill_null(0.0).alias(f"{c}_z"))
    sub = sub.with_columns(z_exprs)
    return sub.with_columns(
        pl.sum_horizontal([pl.col(f"{c}_z") for c in _FUNNEL_RATE_COLS]).alias(
            "unusualness",
        ),
    )


def _pick_outliers(
    sub: pl.DataFrame,
    quartiles: list[str],
) -> tuple[list[dict[str, Any]], set[int]]:
    """Return (outlier-detail-rows, set-of-outlier-email-ids).

    Top-N most-unusual emails per quartile by `unusualness` score.
    """
    rows: list[dict[str, Any]] = []
    ids: set[int] = set()
    for q in quartiles:
        q_top = (
            sub.filter(pl.col("vol_q") == q)
            .sort("unusualness", descending=True)
            .head(_N_OUTLIERS_PER_QUARTILE)
        )
        for row in q_top.iter_rows(named=True):
            ds = str(row["date_sent"])[:10] if row["date_sent"] else ""
            z_scores = {c: _as_float(row[f"{c}_z"]) for c in _FUNNEL_RATE_COLS}
            driver = max(z_scores, key=lambda k: z_scores[k])
            rows.append(
                {
                    "quartile": q,
                    "email_id": row["email_id"],
                    "subject": (row.get("subject") or "")[:60],
                    "date_sent": ds,
                    "total_sent": row["total_sent"],
                    "total_opens": row["total_opens"],
                    "total_vo": row["total_vo"],
                    "total_clicks": row["total_clicks"],
                    "total_actions": row["total_actions"],
                    "z_scores": z_scores,
                    "driver": driver,
                    "unusualness": round(_as_float(row["unusualness"]), 2),
                },
            )
            ids.add(row["email_id"])
    return rows, ids


def _annotate_outlier(
    ax: Axes,
    email_id: int,
    values: list[float],
) -> None:
    last: tuple[int, float] | None = None
    for j, v in enumerate(values):
        if not np.isnan(v):
            last = (j, v)
    if last is not None:
        ax.annotate(
            str(email_id),
            xy=last,
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7,
            color="#d62728",
            zorder=6,
        )


def _draw_funnel_panel(
    ax: Axes,
    q_df: pl.DataFrame,
    frac_cols: list[str],
    outlier_ids: set[int],
    x_pos: np.ndarray,
) -> None:
    """Draw one quartile panel: all-email lines + median profile + P10-P90 band."""
    for row in q_df.iter_rows(named=True):
        values = [np.nan if row[c] is None else float(row[c]) for c in frac_cols]
        is_out = row["email_id"] in outlier_ids
        ax.plot(
            x_pos,
            values,
            color="#d62728" if is_out else "#666",
            alpha=0.9 if is_out else 0.04,
            lw=1.6 if is_out else 0.6,
            zorder=5 if is_out else 1,
        )
        if is_out:
            _annotate_outlier(ax, row["email_id"], values)

    medians: list[float] = [1.0]
    for col in frac_cols[1:]:
        med = q_df[col].drop_nulls().median()
        medians.append(_as_float(med) if med is not None else float("nan"))
    ax.plot(
        x_pos,
        medians,
        color="#1f77b4",
        lw=3,
        label="quartile median",
        zorder=4,
    )

    p10: list[float] = [1.0]
    p90: list[float] = [1.0]
    for col in frac_cols[1:]:
        v = q_df[col].drop_nulls()
        if v.is_empty():
            p10.append(float("nan"))
            p90.append(float("nan"))
        else:
            lo = v.quantile(0.1)
            hi = v.quantile(0.9)
            p10.append(float(lo) if lo is not None else float("nan"))
            p90.append(float(hi) if hi is not None else float("nan"))
    ax.fill_between(
        x_pos,
        p10,
        p90,
        color="#1f77b4",
        alpha=0.15,
        label="P10-P90",
        zorder=2,
    )


def _prep_funnel_df(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to meaningful funnels, bucket by volume, compute fractions + z."""
    sub = df.filter(pl.col("total_sent") >= MIN_SENT_FOR_RATE)
    if sub.is_empty():
        return sub
    sent_series = sub["total_sent"].cast(pl.Float64)
    vol_q = sent_series.qcut(
        quantiles=[0.25, 0.5, 0.75],
        labels=["Q1", "Q2", "Q3", "Q4"],
        allow_duplicates=True,
    )
    sub = sub.with_columns(vol_q.alias("vol_q"))
    frac_exprs = [
        (pl.col(c).cast(pl.Float64) / pl.col("total_sent").cast(pl.Float64)).alias(
            f"frac_{label}",
        )
        for c, label in zip(_FUNNEL_STAGE_COLS, _FUNNEL_STAGE_LABELS, strict=True)
    ]
    sub = sub.with_columns(frac_exprs)
    return _score_unusualness(sub)


def chart_funnel_by_quartile(
    df: pl.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    """Per-email funnel profiles faceted 2x2 by send-volume quartile.

    The y-axis normalizes each email to fraction of `total_sent` so funnel
    shape is comparable across sends of any size. Gaps on the VO stage
    indicate pre-VO-era emails with null `total_vo`. Top-3 most-unusual
    emails per quartile are highlighted in red and returned as table rows.

    Returns (base64 chart, outlier detail rows).
    """
    sub = _prep_funnel_df(df)
    if sub.is_empty():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return fig_to_base64(fig), []

    frac_cols = [f"frac_{label}" for label in _FUNNEL_STAGE_LABELS]
    quartiles = sorted(
        [q for q in sub["vol_q"].unique().to_list() if q is not None],
    )
    outlier_rows, outlier_ids = _pick_outliers(sub, quartiles)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes_flat = axes.flatten()
    x_pos = np.arange(len(_FUNNEL_STAGE_LABELS))

    for i in range(min(len(quartiles), _N_PANELS)):
        q = quartiles[i]
        ax = axes_flat[i]
        q_df = sub.filter(pl.col("vol_q") == q)
        sent_min = _as_int(q_df["total_sent"].min())
        sent_max = _as_int(q_df["total_sent"].max())

        _draw_funnel_panel(ax, q_df, frac_cols, outlier_ids, x_pos)

        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(_FUNNEL_STAGE_LABELS))
        ax.set_title(
            f"{q} (N={q_df.height}; total_sent {sent_min:,}-{sent_max:,})",
            fontsize=10,
        )
        if i in (0, 2):
            ax.set_ylabel("fraction of total_sent (log)")
        ax.grid(visible=True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    for j in range(len(quartiles), _N_PANELS):
        axes_flat[j].set_axis_off()

    fig.suptitle(
        "Chart F - Funnel shape per email (fraction of sent), "
        "faceted by send-volume quartile. "
        "Red = top-3 most-unusual vs quartile peers.",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    return fig_to_base64(fig), outlier_rows


# ---------------------------------------------------------------------------
# Chart G: distribution of |z-score| per rate
# ---------------------------------------------------------------------------

_Z_REF_UNCOMMON = 2.0
_Z_REF_UNUSUAL = 3.0


def _draw_z_panel(
    ax: Axes,
    z_vals: np.ndarray,
    rate_label: str,
    is_leftmost: bool,
) -> None:
    """Draw one |z-score| histogram panel with reference lines + tail counts."""
    if z_vals.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.hist(z_vals, bins=40, color="#5a8cb5", edgecolor="white")
    ax.axvline(
        _Z_REF_UNCOMMON,
        color="#f2a900",
        linestyle="--",
        lw=1.2,
        label=f"|z|={_Z_REF_UNCOMMON:.0f} (uncommon)",
    )
    ax.axvline(
        _Z_REF_UNUSUAL,
        color="#d62728",
        linestyle="--",
        lw=1.2,
        label=f"|z|={_Z_REF_UNUSUAL:.0f} (unusual)",
    )

    n_over_2 = int((z_vals >= _Z_REF_UNCOMMON).sum())
    n_over_3 = int((z_vals >= _Z_REF_UNUSUAL).sum())
    line1 = f"|z|>={_Z_REF_UNCOMMON:.0f}: {n_over_2:,}"
    line2 = f"|z|>={_Z_REF_UNUSUAL:.0f}: {n_over_3:,}"
    tail_text = line1 + "\n" + line2
    ax.text(
        0.98,
        0.97,
        tail_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_yscale("log")
    ax.set_xlabel(f"|z-score| for {rate_label}")
    if is_leftmost:
        ax.set_ylabel("count (log)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(visible=True, alpha=0.3)


def chart_zscore_distributions(df: pl.DataFrame) -> str:
    """Distribution of absolute z-scores per rate (within-quartile cohort).

    Shows the empirical distribution of |z-score| across all post-guard
    emails for each of the 4 rates. Reference lines at |z|=2 (uncommon)
    and |z|=3 (unusual) help calibrate the tails. Log-y makes the right
    tail visible.
    """
    sub = _prep_funnel_df(df)
    if sub.is_empty():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return fig_to_base64(fig)

    fig, axes = plt.subplots(1, len(_FUNNEL_RATE_COLS), figsize=(18, 4), sharey=True)
    for i, rate in enumerate(_FUNNEL_RATE_COLS):
        z_vals = sub[f"{rate}_z"].drop_nulls().to_numpy()
        _draw_z_panel(axes[i], z_vals, RATE_LABELS[rate], is_leftmost=(i == 0))

    fig.suptitle(
        "Chart G - |z-score| distribution per rate (within-quartile cohort). "
        "Tails beyond |z|=3 are the drivers of outlier unusualness.",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _fmt_pct(v: float) -> str:
    """Format float rate as percentage string."""
    if _is_nan(v):
        return "N/A"
    return f"{v * 100:.1f}%"


def _flag(rate: str, median: float) -> str:
    """Return OK or FLAG based on whether median is within expected band."""
    if _is_nan(median):
        return "N/A"
    lo, hi = EXPECTED_BANDS[rate]
    return "OK" if lo <= median <= hi else "FLAG"


def _fmt_count(v: int | None) -> str:
    if v is None:
        return "&mdash;"
    return f"{v:,}"


def _table_rows(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"<tr><td>{row['email_id']}</td>"
        f"<td>{row['date_sent']}</td>"
        f"<td>{row['subject'][:55]}...</td>"
        f"<td>{row['total_sent']:,}</td>"
        f"<td>{_fmt_count(row['total_opens'])}</td>"
        f"<td>{_fmt_count(row['total_vo'])}</td>"
        f"<td>{_fmt_count(row['total_clicks'])}</td>"
        f"<td>{_fmt_count(row['total_actions'])}</td>"
        f"<td>{row['rate'] * 100:.2f}%</td></tr>"
        for row in rows
    )


def _drilldown_section(rate: str, tables: dict[str, list[dict[str, Any]]]) -> str:
    label = RATE_LABELS[rate]
    top = tables.get(f"{rate}_top", [])
    bot = tables.get(f"{rate}_bottom", [])
    hdr = (
        f"<tr><th>email_id</th><th>date_sent</th><th>subject</th>"
        f"<th>sent</th><th>opens</th><th>VO</th>"
        f"<th>clicks</th><th>actions</th><th>{label}</th></tr>"
    )
    return (
        f"\n<h3>{label} drilldown</h3>\n"
        f"<p><strong>Top 10 (highest {label})</strong></p>\n"
        f"<table>\n  {hdr}\n  {_table_rows(top)}\n</table>\n"
        f"<p><strong>Bottom 10 (lowest {label})</strong></p>\n"
        f"<table>\n  {hdr}\n  {_table_rows(bot)}\n</table>\n"
    )


def _stat_box(number: str, label: str) -> str:
    return (
        '  <div class="stat-box">\n'
        f'    <div class="stat-number">{number}</div>\n'
        f'    <div class="stat-label">{label}</div>\n'
        "  </div>\n"
    )


# ---------------------------------------------------------------------------
# CSS block (verbatim from ds-report SKILL.md:159-255)
# ---------------------------------------------------------------------------

_CSS = """\
body {
  font-family: 'Georgia', serif;
  max-width: 900px;
  margin: 40px auto;
  color: #333;
  line-height: 1.6;
}
h1 {
  font-size: 28px;
  border-bottom: 3px solid #2c3e50;
  padding-bottom: 10px;
}
h2 {
  font-size: 22px;
  color: #2c3e50;
  margin-top: 40px;
  border-bottom: 1px solid #bdc3c7;
  padding-bottom: 5px;
}
h3 {
  font-size: 18px;
  color: #34495e;
}
.stat-box {
  display: inline-block;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 15px 25px;
  margin: 10px;
  text-align: center;
}
.stat-number {
  font-size: 36px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 14px;
  color: #6c757d;
}
.chart {
  text-align: center;
  margin: 20px 0;
}
.chart img {
  max-width: 100%;
  border: 1px solid #eee;
  border-radius: 4px;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 20px 0;
  font-size: 14px;
}
th {
  background: #2c3e50;
  color: white;
  padding: 10px 12px;
  text-align: left;
}
td {
  padding: 8px 12px;
  border-bottom: 1px solid #eee;
}
tr:nth-child(even) {
  background: #f8f9fa;
}
.abstract {
  background: #f0f4f8;
  padding: 20px;
  border-left: 4px solid #2c3e50;
  margin: 20px 0;
  font-style: italic;
}
.callout {
  background: #e8f5e9;
  padding: 15px;
  border-radius: 8px;
  margin: 15px 0;
}
.callout-warn {
  background: #fff3e0;
  padding: 15px;
  border-radius: 8px;
  margin: 15px 0;
}
.methodology {
  font-size: 14px;
  color: #555;
}
@media print {
  body { margin: 20px; }
  .stat-box { border: 1px solid #ccc; }
}
"""


# ---------------------------------------------------------------------------
# HTML render — split into section helpers to stay under PLR0915 limit
# ---------------------------------------------------------------------------


def _section_intro(abstract: str, now_str: str) -> str:
    """Sections: header + abstract + introduction."""
    w = io.StringIO()
    w.write("<h1>aggregate_emails v3 - Rate Sanity Report</h1>\n")
    w.write('<p style="color: #6c757d;">')
    w.write(f"Entity-level EDA - generated {now_str}</p>\n\n")
    w.write('<div class="abstract">\n')
    w.write(f"<strong>Abstract.</strong> {abstract}\n")
    w.write("</div>\n\n")
    w.write("<h2>1. Introduction</h2>\n<p>\n")
    w.write("This report answers one question: <em>are the per-email rates in\n")
    w.write("<code>aggregate_emails v3</code> plausible for advocacy email?</em>\n")
    w.write("It does NOT re-examine field-level quality (null rates, range flags,\n")
    w.write("embedding coverage) - those were covered in\n")
    w.write("<code>projects/civic_shout_actions_routing/eda/")
    w.write("aggregate-emails-quality.md</code>\n")
    w.write("(commit 3ef5200). Instead, it treats the six derived rates as the\n")
    w.write("primary subject: open rate, verified-open rate, click rate,\n")
    w.write("action rate, click-per-open, and action-per-click.\n")
    w.write("</p>\n<p>\n")
    w.write("v3 upgrades the engagement counters (<code>total_*</code>) to a\n")
    w.write("24-month attribution window (vs. ~1 month in v1/v2), making rates\n")
    w.write("higher than v1/v2. All rate computations use denominator guards\n")
    w.write("that null out rates where the denominator is zero or null.\n")
    w.write("</p>\n\n")
    return w.getvalue()


def _section_scope(stat_boxes_scope: str, chart_a: str) -> str:
    """Section 2: Data Scope and Denominators."""
    w = io.StringIO()
    w.write("<h2>2. Data Scope &amp; Denominators</h2>\n\n")
    w.write('<div style="text-align: center;">\n')
    w.write(stat_boxes_scope)
    w.write("</div>\n\n")
    w.write("<p>\n<strong>Chart A</strong> shows the distribution of ")
    w.write("<code>total_sent</code> per email on a log-x scale. The heavy right\n")
    w.write("tail motivates variance-stabilising transforms or Bayesian smoothing\n")
    w.write("for any feature built from these rates.\n</p>\n")
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_a}"')
    w.write(' alt="Chart A - total_sent histogram" />\n</div>\n\n')
    return w.getvalue()


def _section_rates(
    stat_boxes_rates: str,
    chart_b: str,
    rate_agg_rows: str,
) -> str:
    """Section 3: Per-Email Rate Aggregates."""
    w = io.StringIO()
    w.write("<h2>3. Per-Email Rate Aggregates</h2>\n\n")
    w.write('<div style="text-align: center;">\n')
    w.write(stat_boxes_rates)
    w.write("</div>\n\n")
    w.write("<p>\n<strong>Chart B</strong> shows rate distributions for all six\n")
    w.write("metrics. The green band is the expected industry-norm range.\n")
    w.write("OK = median inside the band; FLAG = median falls outside.\n</p>\n")
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_b}"')
    w.write(' alt="Chart B - rate distributions" />\n</div>\n\n')
    w.write("<table>\n  <tr>\n")
    w.write("    <th>Rate</th><th>Median</th><th>P10</th><th>P90</th>")
    w.write("<th>Expected band</th><th>N kept</th><th>N dropped</th>\n  </tr>\n")
    w.write(f"  {rate_agg_rows}")
    w.write("</table>\n\n")
    return w.getvalue()


def _section_invariants(inv_rows: str, chart_c: str) -> str:
    """Section 4: Rate Sanity Invariants."""
    w = io.StringIO()
    w.write("<h2>4. Rate Sanity Invariants</h2>\n\n<p>\n")
    w.write("Hard invariants are treated as bugs if violated. Report-only\n")
    w.write("incidence counts are diagnostic - legitimate data patterns that\n")
    w.write("are worth noting but do not indicate bad data.\n</p>\n\n")
    w.write("<table>\n")
    w.write("  <tr><th>Check</th><th>Type</th>")
    w.write("<th>Violation count</th><th>Notes</th></tr>\n")
    w.write(f"  {inv_rows}")
    w.write("</table>\n\n")
    w.write("<p>\n<strong>Chart C</strong> shows a pass/fail breakdown for each\n")
    w.write("check. Green = passing rows; red = violations or incidence.\n</p>\n")
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_c}"')
    w.write(' alt="Chart C - invariant summary" />\n</div>\n\n')
    return w.getvalue()


def _section_temporal(chart_d: str, chart_e: str) -> str:
    """Section 5: Temporal Stability."""
    w = io.StringIO()
    w.write("<h2>5. Temporal Stability</h2>\n\n<p>\n")
    w.write("<strong>Chart D</strong> shows the monthly median of each rate over\n")
    w.write("time. The red-shaded region covers pre-VO-era (before 2024-03-01)\n")
    w.write("; the green band is the expected advocacy range.\n</p>\n")
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_d}"')
    w.write(' alt="Chart D - monthly median rates" />\n</div>\n\n')
    w.write("<p>\n<strong>Chart E</strong> shows rate vs. <code>total_sent</code>\n")
    w.write("as a hexbin plot (log-x). High variance at low send volumes confirms\n")
    w.write("that raw rates are noisy for small sends and motivates a\n")
    w.write("minimum-denominator gate or Bayesian smoothing.\n</p>\n")
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_e}"')
    w.write(' alt="Chart E - rate vs send volume" />\n</div>\n\n')
    return w.getvalue()


def _fmt_z(v: float, is_driver: bool) -> str:
    s = f"{v:.1f}"
    return f"<strong>{s}</strong>" if is_driver else s


_DRIVER_LABEL = {
    "open_rate": "open",
    "vo_rate": "VO",
    "click_rate": "click",
    "action_rate": "action",
}


def _outlier_row_html(row: dict[str, Any]) -> str:
    """One <tr> for the outlier table, with the driving z-score cell bolded."""
    z = row["z_scores"]
    driver = row["driver"]
    return (
        f"<tr><td>{row['quartile']}</td>"
        f"<td>{row['email_id']}</td>"
        f"<td>{row['date_sent']}</td>"
        f"<td>{row['subject'][:50]}...</td>"
        f"<td>{row['total_sent']:,}</td>"
        f"<td>{_fmt_count(row['total_opens'])}</td>"
        f"<td>{_fmt_count(row['total_vo'])}</td>"
        f"<td>{_fmt_count(row['total_clicks'])}</td>"
        f"<td>{_fmt_count(row['total_actions'])}</td>"
        f"<td>{_fmt_z(z['open_rate'], driver == 'open_rate')}</td>"
        f"<td>{_fmt_z(z['vo_rate'], driver == 'vo_rate')}</td>"
        f"<td>{_fmt_z(z['click_rate'], driver == 'click_rate')}</td>"
        f"<td>{_fmt_z(z['action_rate'], driver == 'action_rate')}</td>"
        f"<td><strong>{_DRIVER_LABEL[driver]}</strong></td>"
        f"<td>{row['unusualness']:.1f}</td></tr>\n"
    )


def _section_funnel(
    chart_f: str,
    chart_g: str,
    outliers: list[dict[str, Any]],
) -> str:
    """Section 6: Funnel shape by volume quartile."""
    w = io.StringIO()
    w.write("<h2>6. Funnel shape by volume quartile</h2>\n\n<p>\n")
    w.write(
        "<strong>Chart F</strong> normalizes every email to fraction of "
        "<code>total_sent</code> so the funnel shape (opens &rarr; VO "
        "&rarr; clicks &rarr; actions) is comparable across sends of any "
        "size. Panels split emails by send-volume quartile (recomputed "
        "on emails with <code>total_sent &gt;= "
        f"{MIN_SENT_FOR_RATE}</code>). The blue line is the quartile "
        "median profile with P10-P90 shaded. Red lines are the three "
        "emails in each quartile with the most-unusual rate profile vs "
        "their peers. A break in a line on the VO stage indicates a "
        "pre-VO-era email.\n</p>\n",
    )
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_f}"')
    w.write(' alt="Chart F - funnel by quartile" />\n</div>\n\n')

    w.write("<h3>How unusual is unusual? |z-score| distribution per rate</h3>\n<p>\n")
    w.write(
        "<strong>Chart G</strong> shows the empirical distribution of "
        "absolute z-scores (within each email's volume quartile cohort) "
        "for the four base rates. Most emails are within |z|&lt;2 of "
        "their quartile median; tails beyond |z|=3 are the drivers that "
        "push specific emails to the top of the unusualness table below.\n</p>\n",
    )
    w.write('<div class="chart">\n')
    w.write(f'  <img src="data:image/png;base64,{chart_g}"')
    w.write(' alt="Chart G - z-score distributions" />\n</div>\n\n')

    w.write("<h3>Most-unusual emails per volume quartile</h3>\n<p>\n")
    w.write(
        "Unusualness = sum of |z-score| across the 4 rates, vs. the email's "
        "quartile cohort. Null rates contribute 0. The bolded z-score is "
        "the <strong>driver</strong> &mdash; the rate pulling this email "
        "away from its peers the hardest. Inspect these before trusting "
        "rate features on the full corpus.\n</p>\n",
    )
    w.write("<table>\n  ")
    w.write(
        "<tr><th>Q</th><th>email_id</th><th>date</th><th>subject</th>"
        "<th>sent</th><th>opens</th><th>VO</th><th>clicks</th><th>actions</th>"
        "<th>z(open)</th><th>z(VO)</th><th>z(click)</th><th>z(action)</th>"
        "<th>driver</th><th>&Sigma;|z|</th></tr>\n",
    )
    for row in outliers:
        w.write(_outlier_row_html(row))
    w.write("</table>\n\n")
    return w.getvalue()


def _section_nextsteps(min_sent: int) -> str:
    """Section 7: Next Steps."""
    w = io.StringIO()
    w.write("<h2>7. Next Steps</h2>\n<ol>\n")
    w.write("  <li><strong>Choose a send-volume gate.</strong> Chart E shows\n")
    w.write("  variance inflation below ~500 sends. The minimum-denominator\n")
    w.write(f"  thresholds in this report (>= {min_sent} sent for\n")
    w.write("  open/vo/click/action rates) are a starting point; a smoothed\n")
    w.write("  (Empirical Bayes) rate may be more powerful for small sends.\n")
    w.write("  Owner: next investigation that builds rate-based features.\n")
    w.write("  </li>\n")
    w.write("  <li><strong>Investigate any out-of-band medians.</strong>\n")
    w.write("  Review the distribution tail and confirm the outlier emails\n")
    w.write("  are not data errors (test sends, seed accounts, etc.).\n")
    w.write("  Owner: EDA follow-up.\n  </li>\n")
    w.write("  <li><strong>Confirm VO-era stability.</strong> Chart D shows\n")
    w.write("  whether <code>vo_rate</code> stabilised after 2024-03.\n")
    w.write("  If trending up, restrict analyses to a stable sub-window.\n")
    w.write("  Owner: next investigation using VO signal.\n  </li>\n")
    w.write("  <li><strong>Decide on leaky-column exclusion policy.</strong>\n")
    w.write("  The prior EDA flagged <code>sent_in_range</code> /\n")
    w.write("  <code>opens_in_range</code> / <code>clicks_in_range</code> as\n")
    w.write("  misleading (R5). Do not include these flags in modeling matrices\n")
    w.write("  without a design decision on what 'in range' means post-v3.\n")
    w.write("  </li>\n")
    w.write("</ol>\n\n")
    return w.getvalue()


def _section_appendix_a(
    guard_rows: str,
    bp_rows: str,
    now_str: str,
) -> str:
    """Section 7 tables: cache details, guards, breakpoints."""
    w = io.StringIO()
    w.write("<h2>8. Appendix - Methods &amp; Open Questions</h2>\n")
    w.write('<div class="methodology">\n\n')
    w.write("<h3>Cache details</h3>\n<ul>\n")
    w.write("  <li>Cache version: <code>aggregate_emails v3</code></li>\n")
    w.write(f"  <li>Build cutoff: <code>BUILD_CUTOFF_UTC = {BUILD_CUTOFF_UTC}")
    w.write("</code></li>\n")
    w.write(f"  <li>Report generated: <code>{now_str}</code></li>\n")
    w.write("</ul>\n\n")
    w.write("<h3>Denominator guards applied</h3>\n<table>\n")
    w.write("  <tr><th>Rate</th><th>Guard condition</th></tr>\n")
    w.write("  <tr><td>open_rate</td><td>total_sent &gt; 0</td></tr>\n")
    w.write("  <tr><td>vo_rate</td><td>total_sent &gt; 0 AND total_vo IS NOT NULL")
    w.write(" AND date_sent &gt;= 2024-03-01</td></tr>\n")
    w.write("  <tr><td>click_rate</td><td>total_sent &gt; 0</td></tr>\n")
    w.write("  <tr><td>action_rate</td><td>total_sent &gt; 0</td></tr>\n")
    w.write("  <tr><td>click_per_open</td><td>total_opens &gt; 0</td></tr>\n")
    w.write("  <tr><td>action_per_click</td><td>total_clicks &gt; 0</td></tr>\n")
    w.write("</table>\n\n")
    w.write("<h3>N_kept / N_dropped per rate (post-guard)</h3>\n<table>\n")
    w.write("  <tr><th>Rate</th><th>N kept</th><th>N dropped</th></tr>\n")
    w.write(f"  {guard_rows}")
    w.write("</table>\n\n")
    w.write("<h3>Drilldown minimum-denominator thresholds</h3>\n<table>\n")
    w.write("  <tr><th>Rate</th><th>Minimum denominator</th></tr>\n")
    w.write("  <tr><td>open_rate, vo_rate, click_rate, action_rate</td>")
    w.write(f"<td>total_sent &gt;= {MIN_SENT_FOR_RATE}</td></tr>\n")
    w.write("  <tr><td>click_per_open</td>")
    w.write(f"<td>total_opens &gt;= {MIN_OPENS_FOR_CPO}</td></tr>\n")
    w.write("  <tr><td>action_per_click</td>")
    w.write(f"<td>total_clicks &gt;= {MIN_CLICKS_FOR_APC}</td></tr>\n")
    w.write("</table>\n\n")
    w.write("<h3>send_volume_bucket quintile breakpoints</h3>\n<table>\n")
    w.write("  <tr><th>Bucket</th><th>Break-point (total_sent)</th></tr>\n")
    w.write(f"  {bp_rows}")
    w.write("</table>\n\n")
    return w.getvalue()


def _section_appendix_b(drilldown_html: str) -> str:
    """Open questions + Appendix B drilldown tables."""
    w = io.StringIO()
    w.write("<h3>Expected advocacy-email bands (industry norms)</h3>\n<p>\n")
    w.write("The bands are opinionated estimates from industry benchmarks.\n")
    w.write("If a median falls outside its band, the flag means 'worth\n")
    w.write("understanding why', NOT 'data is broken'.\n")
    w.write("</p>\n\n")
    w.write("<h3>Open questions carried from prior EDA</h3>\n<ul>\n")
    w.write("  <li><strong>Leaky-column exclusion policy</strong>: ")
    w.write("<code>sent_in_range</code>,\n")
    w.write("  <code>opens_in_range</code>, <code>clicks_in_range</code> are\n")
    w.write("  known to be misleading (prior EDA R5). Deferred.\n  </li>\n")
    w.write("  <li><strong>Bayesian smoothing prior choice</strong>: which Beta\n")
    w.write("  prior parameters suit Civic Shout? Not answered here.\n  </li>\n")
    w.write("  <li><strong>VO-era hard filter vs. null-handling</strong>:\n")
    w.write("  downstream features may hard-filter to VO-era rows or treat\n")
    w.write("  pre-era rows as a separate regime. This report uses null-gating.\n")
    w.write("  </li>\n")
    w.write("</ul>\n\n</div>\n\n")
    w.write("<h2>Appendix B - Top/Bottom Drilldown Tables</h2>\n<p>\n")
    w.write("All drilldown tables apply minimum-denominator thresholds\n")
    w.write("(see Appendix A) to exclude 1/1 artifacts from rankings.\n</p>\n\n")
    w.write(drilldown_html)
    return w.getvalue()


def _build_html_body(
    abstract: str,
    now_str: str,
    stat_boxes_scope: str,
    stat_boxes_rates: str,
    chart_a: str,
    chart_b: str,
    chart_c: str,
    chart_d: str,
    chart_e: str,
    chart_f: str,
    chart_g: str,
    funnel_outliers: list[dict[str, Any]],
    rate_agg_rows: str,
    inv_rows: str,
    guard_rows: str,
    bp_rows: str,
    drilldown_html: str,
) -> str:
    """Assemble HTML body from section helpers."""
    return (
        _section_intro(abstract, now_str)
        + _section_scope(stat_boxes_scope, chart_a)
        + _section_rates(stat_boxes_rates, chart_b, rate_agg_rows)
        + _section_invariants(inv_rows, chart_c)
        + _section_temporal(chart_d, chart_e)
        + _section_funnel(chart_f, chart_g, funnel_outliers)
        + _section_nextsteps(MIN_SENT_FOR_RATE)
        + _section_appendix_a(guard_rows, bp_rows, now_str)
        + _section_appendix_b(drilldown_html)
    )


def _build_rate_rows(
    rs: dict[str, Any],
    flags: dict[str, str],
) -> str:
    """Build HTML table rows for the rate aggregates table."""
    rows = ""
    for r, label in RATE_LABELS.items():
        ri = rs[r]
        band_lo, band_hi = EXPECTED_BANDS[r]
        band_str = f"{band_lo * 100:.0f}%-{band_hi * 100:.0f}%"
        flag_sym = flags[r]
        rows += (
            f"<tr><td>{label}</td>"
            f"<td>{_fmt_pct(ri['median'])} {flag_sym}</td>"
            f"<td>{_fmt_pct(ri['p10'])}</td>"
            f"<td>{_fmt_pct(ri['p90'])}</td>"
            f"<td>{band_str}</td>"
            f"<td>{ri['n_kept']:,}</td>"
            f"<td>{ri['n_dropped']:,}</td></tr>\n"
        )
    return rows


def _build_inv_rows(stats: dict[str, Any]) -> str:
    """Build HTML table rows for the invariants table."""
    viol_ct = stats["inv_vo_gt_opens"]
    rows = (
        "<tr><td>total_vo &lt;= total_opens</td>"
        "<td>HARD INVARIANT</td>"
        f"<td>{viol_ct}</td>"
        "<td>Expected ~115 (MPP dedup artifact per prior EDA R2)</td></tr>\n"
        "<tr><td>total_clicks &gt; total_opens</td>"
        "<td>REPORT-ONLY (incidence)</td>"
        f"<td>{stats['inc_clicks_gt_opens']}</td>"
        "<td>Legitimate - images-disabled clicks fire without open pixel"
        "</td></tr>\n"
        "<tr><td>total_actions &gt; total_sent</td>"
        "<td>REPORT-ONLY (incidence)</td>"
        f"<td>{stats['inc_actions_gt_sent']}</td>"
        "<td>Legitimate - total_sent is unique users; "
        "total_actions is row count</td></tr>\n"
    )
    for r in ["open_rate", "vo_rate", "click_rate", "click_per_open"]:
        cnt = stats["prob_rates_gt100"][r]
        status = "PASS" if cnt == 0 else f"FAIL ({cnt} emails)"
        rows += (
            f"<tr><td>{r} &lt;= 100%</td>"
            "<td>HARD INVARIANT</td>"
            f"<td>{cnt}</td><td>{status}</td></tr>\n"
        )
    for r in ["action_rate", "action_per_click"]:
        cnt = stats["excluded_gt100"][r]
        rows += (
            f"<tr><td>{r} &gt; 100%</td>"
            "<td>REPORT-ONLY tail (not a bug)</td>"
            f"<td>{cnt}</td>"
            "<td>Attribution semantics allow &gt;100%; "
            "inspect distribution tail</td></tr>\n"
        )
    return rows


def render_html(
    stats: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    breakpoints: pl.DataFrame,
    chart_a: str,
    chart_b: str,
    chart_c: str,
    chart_d: str,
    chart_e: str,
    chart_f: str,
    chart_g: str,
    funnel_outliers: list[dict[str, Any]],
) -> str:
    """Render the full self-contained HTML report."""
    rs = stats["rates"]
    medians = {r: rs[r]["median"] for r in RATE_LABELS}
    flags = {r: _flag(r, medians[r]) for r in RATE_LABELS}
    out_of_norm = [r for r in RATE_LABELS if flags[r] == "FLAG"]

    if out_of_norm:
        flagged = ", ".join(out_of_norm)
        verdict = f"Flag: {flagged} median(s) outside expected bands - see section 4."
    else:
        verdict = "Rates within advocacy norms - v3 looks healthy."

    date_min_str = str(stats["date_min"])[:7] if stats["date_min"] else "?"
    date_max_str = str(stats["date_max"])[:7] if stats["date_max"] else "?"

    abstract = (
        f"aggregate_emails v3 covers {stats['n_total']:,} emails spanning "
        f"{date_min_str} through {date_max_str}. "
        f"{stats['n_vo_era']:,} emails ({stats['pct_vo_era']:.1f}%) fall "
        "in the VO-tracking era (>= 2024-03-01). "
        f"Headline medians: open {_fmt_pct(medians['open_rate'])}, "
        f"VO {_fmt_pct(medians['vo_rate'])}, "
        f"click {_fmt_pct(medians['click_rate'])}, "
        f"action {_fmt_pct(medians['action_rate'])}, "
        f"click-per-open {_fmt_pct(medians['click_per_open'])}, "
        f"action-per-click {_fmt_pct(medians['action_per_click'])}. "
        f"Verdict: {verdict}"
    )

    rate_agg_rows = _build_rate_rows(rs, flags)
    inv_rows = _build_inv_rows(stats)

    drilldown_html = "".join(_drilldown_section(r, tables) for r in RATE_LABELS)

    def _bp_row(row: dict[str, object]) -> str:
        cat = row.get("category", "")
        bp = row.get("breakpoint", "")
        return f"<tr><td>{cat}</td><td>{bp}</td></tr>\n"

    bp_rows = "".join(_bp_row(row) for row in breakpoints.iter_rows(named=True))

    def _gr_row(r: str, lbl: str) -> str:
        nk = rs[r]["n_kept"]
        nd = rs[r]["n_dropped"]
        return f"<tr><td>{lbl}</td><td>{nk:,}</td><td>{nd:,}</td></tr>\n"

    guard_rows = "".join(_gr_row(r, lbl) for r, lbl in RATE_LABELS.items())

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Scope stat boxes
    sb_n = _stat_box(f"{stats['n_total']:,}", "Total emails<br>(v3 rows)")
    sb_dr = _stat_box(f"{date_min_str}-{date_max_str}", "Date range<br>(send_date)")
    sb_vo = _stat_box(
        f"{stats['n_vo_era']:,}",
        f"VO-era emails<br>({stats['pct_vo_era']:.1f}%)",
    )
    m_sent = f"{stats['median_total_sent']:,.0f}"
    sb_med = _stat_box(m_sent, "Median total_sent<br>per email")
    sb_max = _stat_box(f"{stats['max_total_sent']:,}", "Max total_sent<br>(largest)")

    # Rate stat boxes
    sb_open = _stat_box(
        f"{_fmt_pct(medians['open_rate'])} {flags['open_rate']}",
        "Median open rate",
    )
    sb_vo_r = _stat_box(
        f"{_fmt_pct(medians['vo_rate'])} {flags['vo_rate']}",
        "Median VO rate",
    )
    sb_click = _stat_box(
        f"{_fmt_pct(medians['click_rate'])} {flags['click_rate']}",
        "Median click rate",
    )
    sb_act = _stat_box(
        f"{_fmt_pct(medians['action_rate'])} {flags['action_rate']}",
        "Median action rate",
    )
    sb_cpo = _stat_box(
        f"{_fmt_pct(medians['click_per_open'])} {flags['click_per_open']}",
        "Median click-per-open",
    )
    sb_apc = _stat_box(
        f"{_fmt_pct(medians['action_per_click'])} {flags['action_per_click']}",
        "Median action-per-click",
    )

    stat_boxes_scope = sb_n + sb_dr + sb_vo + sb_med + sb_max
    stat_boxes_rates = sb_open + sb_vo_r + sb_click + sb_act + sb_cpo + sb_apc

    body = _build_html_body(
        abstract=abstract,
        now_str=now_str,
        stat_boxes_scope=stat_boxes_scope,
        stat_boxes_rates=stat_boxes_rates,
        chart_a=chart_a,
        chart_b=chart_b,
        chart_c=chart_c,
        chart_d=chart_d,
        chart_e=chart_e,
        chart_f=chart_f,
        chart_g=chart_g,
        funnel_outliers=funnel_outliers,
        rate_agg_rows=rate_agg_rows,
        inv_rows=inv_rows,
        guard_rows=guard_rows,
        bp_rows=bp_rows,
        drilldown_html=drilldown_html,
    )

    title = "aggregate_emails v3 - Rate Sanity Report"
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_CSS}</style>\n"
        "</head>\n<body>\n\n" + body + "\n</body>\n</html>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _log_summary(stats: dict[str, Any]) -> None:
    """Write a one-line summary to stdout."""
    rs = stats["rates"]
    medians = {r: rs[r]["median"] for r in RATE_LABELS}
    flags = {r: _flag(r, medians[r]) for r in RATE_LABELS}
    parts = [f"{r}={_fmt_pct(medians[r])}({flags[r]})" for r in RATE_LABELS]
    d_min = str(stats["date_min"])[:7] if stats["date_min"] else "?"
    d_max = str(stats["date_max"])[:7] if stats["date_max"] else "?"
    segments = [
        f"SUMMARY: n={stats['n_total']:,}",
        f"range={d_min}-{d_max}",
        f"VO-era={stats['n_vo_era']:,} ({stats['pct_vo_era']:.1f}%)",
    ]
    segments.extend(parts)
    sys.stdout.write(" | ".join(segments) + "\n")


def _check_anchors(stats: dict[str, Any]) -> None:
    """Verify expected anchor counts from prior EDA and log results."""
    n_total = stats["n_total"]
    if abs(n_total - ANCHOR_N_TOTAL) > ANCHOR_TOL_TOTAL:
        logger.warning(
            "ANCHOR MISMATCH: expected ~%d total rows, got %d",
            ANCHOR_N_TOTAL,
            n_total,
        )
    n_vo = stats["n_vo_era"]
    if abs(n_vo - ANCHOR_N_VO_ERA) > ANCHOR_TOL_VO_ERA:
        logger.warning(
            "ANCHOR MISMATCH: expected ~%d VO-era rows, got %d",
            ANCHOR_N_VO_ERA,
            n_vo,
        )
    n_viol = stats["inv_vo_gt_opens"]
    if abs(n_viol - ANCHOR_N_VO_VIOLATIONS) > ANCHOR_TOL_VIOLATIONS:
        logger.warning(
            "ANCHOR MISMATCH: expected ~%d vo>opens violations, got %d",
            ANCHOR_N_VO_VIOLATIONS,
            n_viol,
        )
    else:
        logger.info(
            "ANCHOR OK: total_vo>total_opens violations = %d (expected ~%d)",
            n_viol,
            ANCHOR_N_VO_VIOLATIONS,
        )


def main() -> None:
    """Load data, compute stats, render HTML, write output."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading aggregate_emails v3...")
    df, breakpoints = load_and_compute()

    logger.info("Computing summary statistics...")
    stats = summary_stats(df)

    logger.info("Building drilldown tables...")
    tables = top_bottom_tables(df)

    logger.info("Generating charts...")
    chart_a = chart_send_histogram(df)
    chart_b = chart_rate_panels(df)
    chart_c = chart_invariants(df, stats)
    chart_d = chart_monthly_medians(df)
    chart_e = chart_rate_vs_volume(df)
    chart_f, funnel_outliers = chart_funnel_by_quartile(df)
    chart_g = chart_zscore_distributions(df)

    logger.info("Rendering HTML...")
    html = render_html(
        stats=stats,
        tables=tables,
        breakpoints=breakpoints,
        chart_a=chart_a,
        chart_b=chart_b,
        chart_c=chart_c,
        chart_d=chart_d,
        chart_e=chart_e,
        chart_f=chart_f,
        chart_g=chart_g,
        funnel_outliers=funnel_outliers,
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", OUT_HTML, len(html))

    _log_summary(stats)
    _check_anchors(stats)


if __name__ == "__main__":
    main()
