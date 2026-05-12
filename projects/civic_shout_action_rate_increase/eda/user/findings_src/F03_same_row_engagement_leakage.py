"""F03 Same-row engagement leakage diagnostic — real EDA (not mockup).

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F03_same_row_engagement_leakage
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from entities.civic_shout_engagement.attributed_actions.cache import cache as aa_cache
from entities.civic_shout_engagement.email_activities_cache import cache as ea_cache
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_GIT_SHA = "9083370"
_SAMPLE_FRACTION = 0.10


def _chart(shares: dict[str, dict[str, float]]) -> bytes:
    flags = ["opened", "clicked", "verified_opened"]
    raw_pre = [shares[f]["raw_pre"] for f in flags]
    raw_post = [shares[f]["raw_post"] for f in flags]
    imputed = [shares[f]["imputed"] for f in flags]

    x = np.arange(len(flags))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x, raw_pre, color="#207e3c", label="raw event pre-action (usable)")
    ax.bar(
        x, raw_post, bottom=raw_pre, color="#c5500f", label="raw event post-action (still leaks)"
    )
    ax.bar(
        x,
        imputed,
        bottom=np.array(raw_pre) + np.array(raw_post),
        color="#b00020",
        label="no raw event — purely imputed",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(flags)
    ax.set_ylabel("share of actioned rows")
    ax.set_title("Source of `opened/clicked/verified_opened` for actioned=True rows")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8)

    for i, flag in enumerate(flags):
        pre = raw_pre[i]
        post = raw_post[i]
        imp = imputed[i]
        ax.text(
            i,
            pre / 2,
            f"{pre:.0%}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )
        if post > 0.04:
            ax.text(
                i,
                pre + post / 2,
                f"{post:.0%}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
            )
        ax.text(
            i,
            pre + post + imp / 2,
            f"{imp:.0%}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _compute_shares(
    aa: pl.DataFrame,
    ea: pl.LazyFrame,
) -> tuple[dict[str, dict[str, float]], int]:
    sample = aa.filter(pl.col("is_attributed")).sample(fraction=_SAMPLE_FRACTION, seed=42)
    n_sample = sample.height

    key_cols = ["user_id", "email_id"]
    sample_keys = sample.select(key_cols + ["sign_time"])

    ea_open = (
        ea.filter(pl.col("action_type") == "open")
        .select(["user_id", "email_id", "created_at"])
        .rename({"created_at": "open_at"})
        .group_by(["user_id", "email_id"])
        .agg(pl.col("open_at").min().alias("first_open_at"))
        .collect()
    )

    ea_click = (
        ea.filter(pl.col("action_type") == "click")
        .select(["user_id", "email_id", "created_at"])
        .rename({"created_at": "click_at"})
        .group_by(["user_id", "email_id"])
        .agg(pl.col("click_at").min().alias("first_click_at"))
        .collect()
    )

    joined = sample_keys.join(ea_open, on=key_cols, how="left").join(
        ea_click, on=key_cols, how="left"
    )

    joined = joined.with_columns(
        pl.when(pl.col("first_open_at").is_null())
        .then(pl.lit("imputed"))
        .when(pl.col("first_open_at") < pl.col("sign_time"))
        .then(pl.lit("raw_pre"))
        .otherwise(pl.lit("raw_post"))
        .alias("open_source"),
        pl.when(pl.col("first_click_at").is_null())
        .then(pl.lit("imputed"))
        .when(pl.col("first_click_at") < pl.col("sign_time"))
        .then(pl.lit("raw_pre"))
        .otherwise(pl.lit("raw_post"))
        .alias("click_source"),
    )

    def _shares(col: str) -> dict[str, float]:
        counts = (
            joined.group_by(col)
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / n_sample).alias("share"))
        )
        result = {row[col]: row["share"] for row in counts.iter_rows(named=True)}
        return {
            "raw_pre": result.get("raw_pre", 0.0),
            "raw_post": result.get("raw_post", 0.0),
            "imputed": result.get("imputed", 0.0),
        }

    open_shares = _shares("open_source")
    click_shares = _shares("click_source")

    shares = {
        "opened": open_shares,
        "clicked": click_shares,
        "verified_opened": open_shares,
    }
    return shares, n_sample


def _impact_md(shares: dict[str, dict[str, float]], n_sample: int) -> str:
    open_imp = shares["opened"]["imputed"]
    click_imp = shares["clicked"]["imputed"]
    open_pre = shares["opened"]["raw_pre"]
    click_pre = shares["clicked"]["raw_pre"]

    open_verdict = "DROP (label proxy)" if open_imp > 0.20 else "usable from raw events only"
    click_verdict = "DROP (label proxy)" if click_imp > 0.20 else "usable from raw events only"

    return (
        f"**Sample**: {n_sample:,} actioned rows ({_SAMPLE_FRACTION:.0%} of attributed).\n\n"
        "**Same-row `opened/clicked/verified_opened` are NEVER features** for predicting "
        "same-row `actioned` — they are imputed from actioned by v2's create_cache rule.\n\n"
        "**Historical aggregates** (`opened_last_30d`, `clicked_last_90d`) MUST be computed "
        "from RAW `email_activities` events, NOT from the `user_emails` table. Post-imputation "
        "aggregates are contaminated by past-actioned rows having forced `opened=clicked=True`.\n\n"
        "**Current-email features** may include email content, send date, and user history "
        "strictly prior to `date_sent`.\n\n"
        f"**Purely-imputed shares** (no independent raw event at all):\n"
        f"- `opened`: **{open_imp:.1%}** → {open_verdict}\n"
        f"- `clicked`: **{click_imp:.1%}** → {click_verdict}\n"
        f"- `verified_opened`: **{open_imp:.1%}** (identical to opened by v2 rule)\n\n"
        f"**Raw-pre-action shares** (independent corroborating event exists before sign time):\n"
        f"- `opened`: {open_pre:.1%} · `clicked`: {click_pre:.1%}\n\n"
        "**Feature whitelist for Wave 3** (F06, F07, F10): build `opened_last_Nd` and "
        "`clicked_last_Nd` features exclusively from `email_activities` raw events. "
        "Any feature sourced from `user_emails.opened` or `user_emails.clicked` is contaminated "
        "and must be rebuilt or dropped before harness evaluation."
    )


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    aa = aa_cache.get(2)
    ea = ea_cache.scan(2).filter(pl.col("action_type").is_in(["open", "click"]))

    shares, n_sample = _compute_shares(aa, ea)

    open_imp = shares["opened"]["imputed"]
    click_imp = shares["clicked"]["imputed"]

    open_verdict = "drop" if open_imp > 0.20 else "usable"
    click_verdict = "drop" if click_imp > 0.20 else "usable"

    chart_bytes = _chart(shares)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tldr = (
        f"Of actioned rows: opened purely-imputed={open_imp:.1%}, "
        f"clicked purely-imputed={click_imp:.1%}. "
        f"Feature whitelist: {open_verdict} opened, {click_verdict} clicked (from raw events only)."
    )

    numbers_table = [
        ("actioned rows sampled", f"{n_sample:,} ({_SAMPLE_FRACTION:.0%} of attributed)"),
        ("opened: raw event pre-action", f"{shares['opened']['raw_pre']:.1%}"),
        ("opened: raw event post-action", f"{shares['opened']['raw_post']:.1%}"),
        ("opened: purely imputed (no raw event)", f"{open_imp:.1%}"),
        ("clicked: raw event pre-action", f"{shares['clicked']['raw_pre']:.1%}"),
        ("clicked: raw event post-action", f"{shares['clicked']['raw_post']:.1%}"),
        ("clicked: purely imputed (no raw event)", f"{click_imp:.1%}"),
        ("verified_opened: purely imputed", f"{open_imp:.1%} (same as opened by v2 rule)"),
        ("opened feature verdict", open_verdict),
        ("clicked feature verdict", click_verdict),
    ]

    impact = _impact_md(shares, n_sample)

    method_src = f"""
        aa = aa_cache.get(2)   # attributed_actions v2, ~13M rows
        ea = ea_cache.scan(2).filter(pl.col("action_type").is_in(["open", "click"]))

        sample = aa.filter(pl.col("is_attributed")).sample(fraction={_SAMPLE_FRACTION}, seed=42)

        ea_open = (
            ea.filter(pl.col("action_type") == "open")
              .select(["user_id", "email_id", "created_at"])
              .group_by(["user_id", "email_id"])
              .agg(pl.col("created_at").min().alias("first_open_at"))
              .collect()
        )
        ea_click = (
            ea.filter(pl.col("action_type") == "click")
              .select(["user_id", "email_id", "created_at"])
              .group_by(["user_id", "email_id"])
              .agg(pl.col("created_at").min().alias("first_click_at"))
              .collect()
        )

        joined = (
            sample.select(["user_id", "email_id", "sign_time"])
                  .join(ea_open, on=["user_id", "email_id"], how="left")
                  .join(ea_click, on=["user_id", "email_id"], how="left")
        )

        # Classify: raw_pre (event before sign_time), raw_post, or imputed (no event)
        joined = joined.with_columns(
            pl.when(pl.col("first_open_at").is_null()).then("imputed")
              .when(pl.col("first_open_at") < pl.col("sign_time")).then("raw_pre")
              .otherwise("raw_post").alias("open_source"),
            pl.when(pl.col("first_click_at").is_null()).then("imputed")
              .when(pl.col("first_click_at") < pl.col("sign_time")).then("raw_pre")
              .otherwise("raw_post").alias("click_source"),
        )
    """

    html = render_finding(
        finding_id="F03",
        title="Same-row engagement leakage diagnostic",
        tldr=tldr,
        wave=1,
        severity="BLOCKING",
        classification="confounder",
        question_md=(
            "v2 forces `actioned ⇒ opened=clicked=verified_opened=True` "
            "(see `create_cache_v2.py:155-159`). "
            "For each engagement flag, what fraction of `actioned=True` rows have an "
            "independent raw event in `email_activities`?"
        ),
        plain_english_md=(
            "Imagine a hospital that records 'patient drank water' only when a nurse gave them "
            "water — and writes 'drank water = yes' for every patient given water, even ones "
            "who refused the cup. Now try to predict 'will this patient drink water tomorrow' "
            "using 'drank water yesterday' as a feature. The feature is contaminated: it "
            "partly records GIVING, not drinking.\n\n"
            "v2 of our user_emails table does this: every actioned email gets `clicked=True` "
            "and `opened=True` forced on, whether or not the user actually clicked or opened. "
            "If we use 'clicked last 30 days' as a feature later, it's partly just 'actioned "
            "last 30 days' wearing a costume. The model 'learns' a tautology.\n\n"
            "F03 looks at the RAW email_activities table (which has actual click/open events) "
            "and asks: of the actioned rows we have, what fraction have an INDEPENDENT raw "
            "click/open event vs. were purely imputed by the rule? That fraction sets the "
            "feature whitelist for Wave 3."
        ),
        method_py_source=method_src,
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact,
        provenance={
            "data": "user_emails v2 + email_activities v2 + attributed_actions v2",
            "sample": f"{_SAMPLE_FRACTION:.0%} of is_attributed=True rows, seed=42",
            "ts": ts,
            "git": _GIT_SHA,
        },
        is_mockup=False,
    )

    out_path = _OUT_DIR / "F03_same_row_engagement_leakage.html"
    out_path.write_text(html, encoding="utf-8")

    summary = (
        f"F03: actioned rows N={n_sample:,}; "
        f"opened purely-imputed {open_imp:.1%}, "
        f"clicked purely-imputed {click_imp:.1%}, "
        f"verified_opened purely-imputed {open_imp:.1%}. "
        f"Feature whitelist: {open_verdict} opened, {click_verdict} clicked."
    )
    print(summary)


if __name__ == "__main__":
    main()
