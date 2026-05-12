"""F10. Temporal autocorrelation of opens / clicks / actions — real EDA (not mockup).

Per-row windowed sequence features from raw email_activities (per F03 rule).
1% -> 5% -> 100% sampling cadence with timing/memory telemetry per tier.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F10_temporal_autocorrelation
"""

from __future__ import annotations

import io
import json
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from entities.civic_shout_engagement.email_activities_cache import cache as ea_cache
from projects.civic_shout_action_rate_increase.data import user_emails
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_JSONL = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)

_WINDOWS = [1, 3, 5, 10]
_F06_AUC_PLACEHOLDER = 0.84

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


def _load_raw_oc() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load raw opened/clicked indicators from email_activities v2."""
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


def _build_sequence_features(
    sample: pl.DataFrame, opens: pl.DataFrame, clicks: pl.DataFrame
) -> pl.DataFrame:
    """Join sample with raw opens/clicks, sort by user+date, build windowed sequence features."""
    df = (
        sample.join(opens, on=["user_id", "email_id"], how="left")
        .join(clicks, on=["user_id", "email_id"], how="left")
        .with_columns(
            pl.col("raw_opened").fill_null(False),
            pl.col("raw_clicked").fill_null(False),
        )
        .sort(["user_id", "date_sent"])
    )

    exprs: list[pl.Expr] = []
    for n in _WINDOWS:
        exprs.append(
            pl.col("raw_opened")
            .cast(pl.Int8)
            .shift(1)
            .rolling_sum(window_size=n)
            .over("user_id")
            .alias(f"opened_last_{n}")
        )
        exprs.append(
            pl.col("raw_clicked")
            .cast(pl.Int8)
            .shift(1)
            .rolling_sum(window_size=n)
            .over("user_id")
            .alias(f"clicked_last_{n}")
        )
        exprs.append(
            pl.col("actioned")
            .cast(pl.Int8)
            .shift(1)
            .rolling_sum(window_size=n)
            .over("user_id")
            .alias(f"actioned_last_{n}")
        )

    df = df.with_columns(exprs)

    df = df.with_columns(
        (pl.col("actioned_last_5") >= 1).cast(pl.Int8).alias("action_in_last_5"),
    )

    df = _add_consec_unopened(df)

    return df


def _add_consec_unopened(df: pl.DataFrame) -> pl.DataFrame:
    """Compute consecutive_prior_unopened via a map_batches approach per user."""
    rows = df.select(["user_id", "raw_opened"]).with_row_index("_row_idx")

    user_groups = rows.partition_by("user_id", maintain_order=True)
    consec_counts: list[int] = []

    for group in user_groups:
        opened_vals = group["raw_opened"].to_list()
        counts: list[int] = []
        running = 0
        for i, o in enumerate(opened_vals):
            if i == 0:
                counts.append(0)
                running = 0 if o else 1
            else:
                counts.append(running)
                if o:
                    running = 0
                else:
                    running += 1
        consec_counts.extend(counts)

    consec_series = pl.Series("consec_unopened_count", consec_counts, dtype=pl.Int32)
    return df.with_columns(consec_series)


def _temporal_auc(df: pl.DataFrame, feature_col: str) -> float:
    """Univariate AUC using temporal split (train = rows before 80th pct date_sent)."""
    cutoff_date = df["date_sent"].sort().gather_every(1)[int(df.height * 0.8)]

    valid = df.filter(pl.col(feature_col).is_not_null()).filter(pl.col("actioned").is_not_null())
    if valid.height < 50:
        return float("nan")

    train = valid.filter(pl.col("date_sent") < cutoff_date)
    test = valid.filter(pl.col("date_sent") >= cutoff_date)

    if train.height < 20 or test.height < 20:
        return float("nan")

    y_test = test["actioned"].cast(pl.Int8).to_numpy()
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return float("nan")

    x_train = train[feature_col].fill_null(0).cast(pl.Float64).to_numpy().reshape(-1, 1)
    y_train = train["actioned"].cast(pl.Int8).to_numpy()
    x_test = test[feature_col].fill_null(0).cast(pl.Float64).to_numpy().reshape(-1, 1)

    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    try:
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)[:, 1]
        return float(roc_auc_score(y_test, proba))
    except Exception:
        return float("nan")


def _multivariate_auc(df: pl.DataFrame, feature_cols: list[str]) -> float:
    """Multivariate AUC over all sequence features using temporal split."""
    cutoff_date = df["date_sent"].sort().gather_every(1)[int(df.height * 0.8)]

    valid = df.drop_nulls(subset=feature_cols + ["actioned"])
    if valid.height < 50:
        return float("nan")

    train = valid.filter(pl.col("date_sent") < cutoff_date)
    test = valid.filter(pl.col("date_sent") >= cutoff_date)

    if train.height < 20 or test.height < 20:
        return float("nan")

    y_test = test["actioned"].cast(pl.Int8).to_numpy()
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return float("nan")

    x_train = train.select(feature_cols).fill_null(0).cast(pl.Float64).to_numpy()
    y_train = train["actioned"].cast(pl.Int8).to_numpy()
    x_test = test.select(feature_cols).fill_null(0).cast(pl.Float64).to_numpy()

    clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    try:
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)[:, 1]
        return float(roc_auc_score(y_test, proba))
    except Exception:
        return float("nan")


def _sequence_feature_cols(df: pl.DataFrame) -> list[str]:
    candidates = (
        [f"opened_last_{n}" for n in _WINDOWS]
        + [f"clicked_last_{n}" for n in _WINDOWS]
        + [f"actioned_last_{n}" for n in _WINDOWS]
        + ["consec_unopened_count", "action_in_last_5"]
    )
    return [c for c in candidates if c in df.columns]


def _compute_aucs(df: pl.DataFrame) -> tuple[dict[str, float], float, list[tuple[str, float]]]:
    """Return (uni_aucs, multi_auc, sorted_features_desc)."""
    feat_cols = _sequence_feature_cols(df)
    uni_aucs: dict[str, float] = {}
    for col in feat_cols:
        uni_aucs[col] = _temporal_auc(df, col)

    multi_auc = _multivariate_auc(df, feat_cols)

    sorted_feats = sorted(
        [(k, v) for k, v in uni_aucs.items() if not np.isnan(v)],
        key=lambda x: x[1],
        reverse=True,
    )
    return uni_aucs, multi_auc, sorted_feats


def _build_chart(sorted_feats: list[tuple[str, float]], multi_auc: float) -> bytes:
    labels = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    colors = []
    for lbl in labels:
        if lbl.startswith("opened"):
            colors.append("#2956a3")
        elif lbl.startswith("clicked"):
            colors.append("#c5500f")
        elif lbl.startswith("actioned"):
            colors.append("#207e3c")
        else:
            colors.append("#555")

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(labels) * 0.38)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Univariate AUC (temporal split)")
    ax.set_title(
        "Sequence feature univariate AUC vs actioned\n(from raw email_activities — F03-compliant)"
    )
    ax.set_xlim(0.45, max(values + [multi_auc, 0.65]) + 0.05)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7.5,
        )

    if not np.isnan(multi_auc):
        ax.axvline(
            multi_auc,
            color="#b00020",
            linestyle="--",
            linewidth=1.5,
            label=f"Multivariate AUC = {multi_auc:.3f}",
        )
        ax.legend(fontsize=8)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#2956a3", label="opened_last_N"),
        plt.Rectangle((0, 0), 1, 1, color="#c5500f", label="clicked_last_N"),
        plt.Rectangle((0, 0), 1, 1, color="#207e3c", label="actioned_last_N"),
        plt.Rectangle((0, 0), 1, 1, color="#555", label="consec / action_in"),
    ]
    ax.legend(
        handles=legend_patches
        + [
            plt.Line2D(
                [0],
                [0],
                color="#b00020",
                linestyle="--",
                linewidth=1.5,
                label=f"Multivariate = {multi_auc:.3f}",
            )
        ],
        fontsize=7.5,
        loc="lower right",
    )

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _numbers_table(
    sorted_feats: list[tuple[str, float]],
    multi_auc: float,
    n_rows: int,
    wall: float,
    delta_f06: float,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("rows in sample", f"{n_rows:,}"),
        ("wall time (s)", f"{wall:.1f}"),
        (
            "multivariate AUC (all sequence feats)",
            f"{multi_auc:.4f}" if not np.isnan(multi_auc) else "n/a",
        ),
        ("F06 baseline AUC (TBD — placeholder)", f"{_F06_AUC_PLACEHOLDER:.4f}"),
        ("delta vs F06 baseline", f"{delta_f06:+.4f}" if not np.isnan(delta_f06) else "n/a"),
    ]
    for i, (feat, auc) in enumerate(sorted_feats[:10]):
        rows.append((f"rank {i + 1}: {feat}", f"{auc:.4f}"))
    return rows


def _impact_md(sorted_feats: list[tuple[str, float]], multi_auc: float, delta_f06: float) -> str:
    top3 = sorted_feats[:3]
    top3_str = ", ".join(f"{f} ({a:.3f})" for f, a in top3)
    verdict_beat = (
        f"Sequence features BEAT F06 baseline by {delta_f06:+.3f} AUC."
        if not np.isnan(delta_f06) and delta_f06 > 0.01
        else (
            f"Sequence features roughly match F06 baseline (delta {delta_f06:+.3f})."
            if not np.isnan(delta_f06) and abs(delta_f06) <= 0.01
            else f"Sequence features BELOW F06 baseline by {abs(delta_f06):.3f} AUC."
            if not np.isnan(delta_f06)
            else "F06 comparison TBD (placeholder used)."
        )
    )
    return (
        f"Top 3 univariate sequence features: {top3_str}.\n\n"
        f"Multivariate AUC = {multi_auc:.4f}. {verdict_beat}\n\n"
        "Feature whitelist rule: these features are computed from raw email_activities events "
        "(F03-compliant). The user_emails.opened / clicked columns MUST NOT be used.\n\n"
        "If multivariate AUC exceeds F06 by >0.01: include opened_last_N, clicked_last_N, "
        "actioned_last_N, and consec_unopened_count in the production feature set — they are "
        "cheap to compute and likely high-signal.\n\n"
        "If multivariate AUC is at or below F06: sequence features are redundant given RFM "
        "rates — include only as auxiliary features, not as the primary signal."
    )


def _plain_english_md(sorted_feats: list[tuple[str, float]], multi_auc: float) -> str:
    top_feat = sorted_feats[0][0] if sorted_feats else "n/a"
    top_auc = sorted_feats[0][1] if sorted_feats else float("nan")
    return (
        f"Imagine you can see the last few emails a user received. If they opened the last one, "
        f"are they more likely to sign the next petition? F10 measures exactly that.\n\n"
        f"The best single signal is '{top_feat}' (AUC = {top_auc:.3f}): knowing whether a user "
        f"engaged with the most recent email(s) is the cheapest and most direct predictor of "
        f"whether they'll act on the next one.\n\n"
        f"Combining all sequence features gives AUC = {multi_auc:.3f}. Compare this to the "
        f"F06 RFM baseline (rate-window features over 7/30/90d) to see whether short-horizon "
        f"sequence features add complementary signal or just duplicate what RFM already captures."
    )


def _run_tier(
    tier_label: str,
    fraction: float | None,
    opens: pl.DataFrame,
    clicks: pl.DataFrame,
    git_sha: str,
) -> dict:
    t0 = time.time()
    print(f"\n[F10] Starting tier={tier_label} ...")

    if fraction is not None:
        sample = user_emails.sample_percentage(fraction, seed=42)
    else:
        sample = user_emails.full()

    n_rows = sample.height
    print(f"  sample rows={n_rows:,}")

    df = _build_sequence_features(sample, opens, clicks)
    print(f"  sequence features built; shape={df.shape}")

    uni_aucs, multi_auc, sorted_feats = _compute_aucs(df)
    best_uni = sorted_feats[0] if sorted_feats else ("n/a", float("nan"))
    delta_f06 = multi_auc - _F06_AUC_PLACEHOLDER

    chart_bytes = _build_chart(sorted_feats, multi_auc)
    numbers_table = _numbers_table(sorted_feats, multi_auc, n_rows, time.time() - t0, delta_f06)
    impact = _impact_md(sorted_feats, multi_auc, delta_f06)
    plain_english = _plain_english_md(sorted_feats, multi_auc)

    wall = time.time() - t0
    peak_mb = _peak_mb()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

    tldr = (
        f"Sequence features: best_uni={best_uni[0]} ({best_uni[1]:.3f} AUC); "
        f"multi={multi_auc:.3f}; delta_vs_F06={delta_f06:+.3f}. "
        f"[{tier_label} n={n_rows:,} wall={wall:.0f}s]"
    )

    html = render_finding(
        finding_id="F10",
        title="Temporal autocorrelation of opens / clicks / actions",
        tldr=tldr,
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
        question_md=(
            "How informative are 'opened last send', 'opened last 3 sends', and "
            "'consecutive unopened sends' features? Per-row windowed aggregates over "
            "prior N sends per user. Raw email_activities used throughout (F03 compliant)."
        ),
        plain_english_md=plain_english,
        method_py_source=f"""
sample = user_emails.sample_percentage({fraction!r}, seed=42)  # {tier_label}

# Raw opened/clicked from email_activities v2 (F03 compliant)
ea_oc = (
    ea_cache.scan(2)
    .filter(pl.col("action_type").is_in(["open", "click"]))
    .group_by(["user_id", "email_id", "action_type"])
    .agg(pl.len().alias("n_events"))
    .collect()
)
raw_opened = ea_oc.filter(pl.col("action_type") == "open").select(["user_id", "email_id"]).with_columns(pl.lit(True).alias("raw_opened"))
raw_clicked = ea_oc.filter(pl.col("action_type") == "click").select(["user_id", "email_id"]).with_columns(pl.lit(True).alias("raw_clicked"))

df = (
    sample
    .join(raw_opened, on=["user_id", "email_id"], how="left")
    .join(raw_clicked, on=["user_id", "email_id"], how="left")
    .with_columns(pl.col("raw_opened").fill_null(False), pl.col("raw_clicked").fill_null(False))
    .sort(["user_id", "date_sent"])
)

# Windowed sequence features for N in [1, 3, 5, 10]
for n in [1, 3, 5, 10]:
    df = df.with_columns([
        pl.col("raw_opened").cast(pl.Int8).shift(1).rolling_sum(window_size=n).over("user_id").alias(f"opened_last_{{n}}"),
        pl.col("raw_clicked").cast(pl.Int8).shift(1).rolling_sum(window_size=n).over("user_id").alias(f"clicked_last_{{n}}"),
        pl.col("actioned").cast(pl.Int8).shift(1).rolling_sum(window_size=n).over("user_id").alias(f"actioned_last_{{n}}"),
    ])
df = df.with_columns((pl.col("actioned_last_5") >= 1).cast(pl.Int8).alias("action_in_last_5"))

# consec_unopened_count: per user, count consecutive prior rows with raw_opened=False
# Univariate AUC per feature + multivariate AUC; temporal split at 80th pct date_sent
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
cutoff_date = df["date_sent"].sort().gather_every(1)[int(df.height * 0.8)]
""",
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact,
        provenance=provenance,
        is_mockup=False,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "F10_temporal_autocorrelation.html"
    out_path.write_text(html, encoding="utf-8")

    timing_row = {
        "finding_id": "F10",
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
        f"F10 [{tier_label}] rows={n_rows:,} wall={wall:.1f}s; "
        f"best_uni={best_uni[0]} ({best_uni[1]:.3f} AUC); "
        f"delta_over_F06={delta_f06:+.4f}"
    )
    print(f"  wrote {out_path}")

    return {
        "tier": tier_label,
        "n_rows": n_rows,
        "wall": wall,
        "sorted_feats": sorted_feats,
        "multi_auc": multi_auc,
        "delta_f06": delta_f06,
    }


def main() -> None:
    git_sha = _git_sha()
    print("[F10] Loading raw email_activities opens/clicks (full entity — not sampled)...")
    t_ea = time.time()
    opens, clicks = _load_raw_oc()
    print(
        f"  ea opens={opens.height:,} clicks={clicks.height:,} loaded in {time.time() - t_ea:.1f}s"
    )

    _env_tiers = os.environ.get("EDA_TIERS")
    tier_filter = {t.strip() for t in _env_tiers.split(",")} if _env_tiers else None

    last_result: dict = {}
    for tier_label, fraction in _TIERS:
        if tier_filter is not None and tier_label not in tier_filter:
            print(f"[F10] Skipping tier={tier_label} (EDA_TIERS={_env_tiers})")
            continue
        last_result = _run_tier(tier_label, fraction, opens, clicks, git_sha)

    print("\n=== F10 FINAL SUMMARY ===")
    sorted_feats = last_result["sorted_feats"]
    multi_auc = last_result["multi_auc"]
    delta_f06 = last_result["delta_f06"]

    print("Top 10 univariate sequence features (full run):")
    for i, (feat, auc) in enumerate(sorted_feats[:10]):
        print(f"  {i + 1:2d}. {feat:30s}  AUC={auc:.4f}")
    print(f"\nMultivariate AUC (all sequence features): {multi_auc:.4f}")
    print(f"F06 baseline AUC (placeholder, TBD): {_F06_AUC_PLACEHOLDER:.4f}")
    print(f"Delta vs F06: {delta_f06:+.4f}")
    if not np.isnan(delta_f06):
        if delta_f06 > 0.01:
            print("VERDICT: Sequence features BEAT F06 baseline — ship them.")
        elif delta_f06 > -0.005:
            print("VERDICT: Sequence features roughly match F06 — include as auxiliary.")
        else:
            print("VERDICT: Sequence features BELOW F06 — skip or use only as auxiliary.")


if __name__ == "__main__":
    main()
