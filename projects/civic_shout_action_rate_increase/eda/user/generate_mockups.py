"""Generate 11 mockup HTML pages (F00-F10) + index.html for review.

Each mockup uses shape-accurate synthetic data so layout review is meaningful.
The polars Method sketch and Impact text are the REAL plan content. Charts and
numbers will be replaced by the actual EDA run. The MOCKUP banner makes this
unambiguous in the browser.

Run:

    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.generate_mockups
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from projects.civic_shout_action_rate_increase.eda.user._html import (
    render_finding,
    render_index,
)

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_RNG = np.random.default_rng(seed=42)

_PROVENANCE = {
    "data": "civic_shout_user_emails v2 + civic_shout_emails v1 (MOCKUP — synthetic)",
    "sample": "n/a (synthetic)",
    "ts": "2026-05-12 MOCKUP",
    "git": "mockup",
}


def _png(fig: plt.Figure) -> bytes:
    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# F00. Outcome cardinality check
# ----------------------------------------------------------------------------


def _chart_f00() -> bytes:
    counts = [12_400_000, 280_000, 55_000, 13_000, 4_000, 1_200, 300, 70, 20, 6]
    xs = list(range(1, len(counts) + 1))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(xs, counts, color="#2956a3")
    ax.set_yscale("log")
    ax.set_xlabel("attributed actions per (user, email) pair")
    ax.set_ylabel("count of pairs (log)")
    ax.set_title("Distribution of n_attributed_actions | actioned=True")
    ax.set_xticks(xs)
    for i, c in enumerate(counts):
        ax.text(
            xs[i], c * 1.15, f"{c / 1e6:.2f}M" if c >= 1e6 else f"{c:,}", ha="center", fontsize=7
        )
    return _png(fig)


F00 = dict(
    id="F00",
    slug="outcome_cardinality",
    title="Outcome cardinality check",
    tldr="Does collapsing multi-action pairs into a boolean target lose signal we should keep?",
    wave=1,
    severity="BLOCKING",
    classification="confounder",
    question_md=(
        "GOAL.md says 'petitions signed per email.' civic_shout_user_emails.actioned is boolean. "
        "What is the distribution of n_attributed_actions per (user_id, email_id) for attributed pairs, "
        "and how much signal does the boolean collapse cost?"
    ),
    plain_english_md=(
        "Alice opens one email about climate. She clicks the link, lands on a page that "
        "lists 3 related petitions, and signs all 3. Our table records this as one row "
        "with `actioned=True` — a single yes/no. Did Alice sign 1 petition or 3?\n\n"
        "Right now we only know 'at least 1.' If lots of users sign multiple petitions per "
        "email, the boolean undercounts their real impact. F00 asks: how much signal does "
        "that yes/no flattening cost? If it's tiny (<5%), keep the boolean. If it's big, "
        "the modeling target has to change to expected petition COUNT, not probability."
    ),
    method_py_source="""
        aa = aa_cache.get(2)  # attributed_actions v2
        per_pair = (
            aa.filter(pl.col("is_attributed"))
              .group_by(["user_id", "email_id"])
              .agg(pl.len().alias("n_attributed_actions"))
        )
        # distribution of n per actioned pair
        dist = per_pair.group_by("n_attributed_actions").len().sort("n_attributed_actions")
        # share with >=2 attributed actions
        n_pairs = per_pair.height
        ge2 = per_pair.filter(pl.col("n_attributed_actions") >= 2).height
        # total petitions captured vs lost by boolean collapse
        total_signatures = per_pair["n_attributed_actions"].sum()
        captured_by_boolean = n_pairs  # boolean collapses to 1 per pair
        lost = total_signatures - captured_by_boolean
    """,
    numbers_table=[
        ("attributed (user, email) pairs", "12,753,304"),
        ("pairs with ≥2 attributed signatures", "~2.6% (synthetic)"),
        ("total signatures (synthetic)", "~13.4M"),
        ("signatures lost by boolean collapse (synthetic)", "~660K (~4.9%)"),
        ("recommended harness target", "binary (decision pending real numbers)"),
    ],
    impact_md=(
        "If the ≥2-action share is small (<5%) and the lost-signature share is small (<10%), keep the boolean target — "
        "the binary signal is cleaner and matches the dominant case.\n\n"
        "If the lost-signature share exceeds ~10%, the harness target must change to **expected petition count** "
        "(Poisson regression) or **multi-objective** (action probability × conditional petition count).\n\n"
        "Wave 3 features (F06, F07, F10) cannot ship until this decision is made — they target a single response variable."
    ),
)


# ----------------------------------------------------------------------------
# F01. YoY drift / cohort-aging decomposition
# ----------------------------------------------------------------------------


def _chart_f01() -> bytes:
    quarters = [
        "2024Q2",
        "2024Q3",
        "2024Q4",
        "2025Q1",
        "2025Q2",
        "2025Q3",
        "2025Q4",
        "2026Q1",
        "2026Q2",
    ]
    qx = np.arange(len(quarters))
    # cohort 2024 aging from ~7% to ~17%
    cohort_2024 = np.array([0.072, 0.079, 0.087, 0.097, 0.116, 0.130, 0.142, 0.158, 0.171])
    # cohort 2025 starts ~Q1 at 0.070 and ages
    cohort_2025 = np.array([np.nan, np.nan, np.nan, 0.071, 0.082, 0.095, 0.108, 0.122, 0.139])
    # cohort 2026 starts Q1 at 0.073
    cohort_2026 = np.array([np.nan] * 7 + [0.073, 0.086])
    pooled = np.array([0.072, 0.079, 0.087, 0.079, 0.097, 0.110, 0.123, 0.121, 0.143])

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(qx, cohort_2024, marker="o", color="#4a3aff", label="2024 cohort", linewidth=2)
    ax.plot(qx, cohort_2025, marker="o", color="#2956a3", label="2025 cohort", linewidth=2)
    ax.plot(qx, cohort_2026, marker="o", color="#207e3c", label="2026 cohort", linewidth=2)
    ax.plot(
        qx, pooled, color="#b00020", label="pooled (all cohorts)", linestyle="--", linewidth=1.5
    )
    ax.axhline(y=0.075, color="#aaa", linestyle=":", linewidth=1)
    ax.text(0.1, 0.077, "all cohorts enter at ~7-8%", fontsize=7.5, color="#666")
    ax.set_xticks(qx)
    ax.set_xticklabels(quarters, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("action rate")
    ax.set_title("Action rate by cohort over calendar time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _png(fig)


F01 = dict(
    id="F01",
    slug="cohort_aging_decomposition",
    title="YoY drift / cohort-aging decomposition",
    tldr="Is the 7.93→12.58% YoY drift cohort aging or platform improvement? Pre-EDA says aging.",
    wave=1,
    severity="BLOCKING",
    classification="confounder",
    question_md=(
        "Pre-EDA showed YoY drift is empirically 100% cohort aging (2024 cohort alone: 7.93→12.62→17.11%). "
        "Reproduce + visualize. If confirmed, harness must use cohort/tenure-aware temporal validation; "
        "random row splits and pooled-baseline comparisons are invalid."
    ),
    plain_english_md=(
        "Two coffee shops each have 100 customers. Shop A opened in 2024; half its 2024 "
        "customers were one-time tourists who never came back. By 2026, only A's 50 loyal "
        "regulars remain, and they buy way more per visit. Shop B opened in 2026 with 100 "
        "fresh customers, some of whom haven't bought yet.\n\n"
        "Shop A's per-visit sales now look way better than B's — but A didn't make better "
        "coffee. It just kept its highest-intent customers. Civic Shout's '7.93% → 12.58%' "
        "drift looks like product improvement but is almost entirely the same dynamic: "
        "2024 signups who didn't unsubscribe are getting more loyal, while 2025/2026 "
        "newcomers enter at the same ~7-8% they always did. F01 proves this with a "
        "cohort-by-cohort plot, which then forces our harness to split data by send DATE "
        "not by random rows."
    ),
    method_py_source="""
        ue = user_emails.full()
        first_send = (
            ue.group_by("user_id")
              .agg(pl.col("date_sent").min().alias("first_send_date"))
        )
        ue = ue.join(first_send, on="user_id")
        ue = ue.with_columns(
            pl.col("first_send_date").dt.year().alias("cohort_year"),
            pl.col("date_sent").dt.year().alias("calendar_year"),
            pl.col("date_sent").dt.quarter().alias("calendar_quarter"),
        )
        cohort_qtr = (
            ue.group_by(["cohort_year", "calendar_year", "calendar_quarter"])
              .agg(
                  pl.col("actioned").mean().alias("action_rate"),
                  pl.len().alias("n_sends"),
              )
              .sort(["cohort_year", "calendar_year", "calendar_quarter"])
        )
    """,
    numbers_table=[
        ("2024 cohort entry AR", "7.93%"),
        ("2024 cohort 2026Q2 AR (synthetic)", "17.1%"),
        ("2025 cohort entry AR (synthetic)", "7.1%"),
        ("2026 cohort entry AR (synthetic)", "7.3%"),
        ("fraction of YoY drift explained by cohort aging (synthetic)", "98%"),
    ],
    impact_md=(
        "**Split convention**: temporal holdout by send date with tenure-bin stratification. "
        "Random row splits inflate AUC by leaking cohort composition.\n\n"
        "**Baseline**: the pooled 9.87% target is mechanically suppressed by 2024 cohort under-representation early in the window. "
        "Per-cohort or per-tenure-bucket targets are more honest.\n\n"
        "**Follow-up commit**: append a paragraph to GOAL.md recommending the harness anchor — pooled vs 2026-only vs per-tenure. "
        "Do NOT inline-revise GOAL.md (user deferred)."
    ),
)


# ----------------------------------------------------------------------------
# F02. Action concentration / Lorenz
# ----------------------------------------------------------------------------


def _chart_f02() -> bytes:
    n = 1000
    # Generate a Lorenz curve matching the empirical numbers
    pct_users = np.linspace(0, 1, n + 1)
    # Construct cum share so that:
    # at user-rank 0.99 -> action share 0.265 from bottom; equivalently top 1% -> 26.5%
    # at user-rank 0.90 -> action share 0.16 from bottom; top 10% -> 84%
    # at user-rank 0.60 -> action share ~0 (40% have 0 actions)
    bottom_share = np.where(
        pct_users <= 0.40,
        0.0,
        np.where(
            pct_users <= 0.90,
            ((pct_users - 0.40) / 0.50) ** 1.5 * 0.16,
            0.16 + ((pct_users - 0.90) / 0.10) ** 0.4 * (0.735 - 0.16),
        ),
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(pct_users, bottom_share, color="#4a3aff", linewidth=2, label="actual")
    ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", label="equality")
    # Annotate top deciles
    for rank, label, target in [
        (0.99, "top 1%", 0.265),
        (0.95, "top 5%", 0.682),
        (0.90, "top 10%", 0.838),
    ]:
        share = 1 - bottom_share[int(rank * n)]
        ax.annotate(
            f"{label} → {share:.1%}",
            xy=(rank, 1 - target),
            xytext=(rank - 0.12, 1 - target - 0.06),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.6),
        )
    ax.set_xlabel("cumulative share of users (sorted by action count, ascending)")
    ax.set_ylabel("cumulative share of actions")
    ax.set_title("Lorenz curve of attributed actions per user")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _png(fig)


F02 = dict(
    id="F02",
    slug="concentration_and_evaluation_stratification",
    title="Action concentration → evaluation stratification",
    tldr="Top 1% of users → 26.5% of actions; top 10% → 84%. Unstratified metrics are invalid.",
    wave=2,
    severity="IMPORTANT",
    classification="confounder",
    question_md=(
        "Pre-EDA showed top 1% → 26.5% of actions, top 10% → 84%. Reproduce the Lorenz curve cleanly "
        "and recommend an evaluation regime that prevents power users from dominating the metric."
    ),
    plain_english_md=(
        "Imagine measuring 'average household income' in a room that happens to include "
        "Jeff Bezos. The 'average' is millions of dollars — useless for understanding "
        "the median family. Our pooled 9.87% action rate is exactly that average.\n\n"
        "Top 1% of users drive 26.5% of all signatures. A model that scores 'this user "
        "is likely to sign' is mostly identifying the top 1% — but we already know who "
        "they are and we already email them. The 10% lift has to come from the MIDDLE "
        "of the distribution, not the saturated top.\n\n"
        "F02's job is to prove this with a Lorenz curve and pin the consequence: every "
        "evaluation metric must be stratified (by tenure, by prior-action count, by "
        "cohort), because a single pooled AUC will be dominated by 'did the model find "
        "the rich users?' — which is unactionable."
    ),
    method_py_source="""
        ue = user_emails.full()
        per_user = (
            ue.group_by("user_id")
              .agg(pl.col("actioned").sum().alias("n_actions"))
              .sort("n_actions")
        )
        n_actions = per_user["n_actions"].to_numpy()
        cum = np.cumsum(n_actions) / n_actions.sum()
        # Lorenz x = (i+1)/N, y = cum[i]
    """,
    numbers_table=[
        ("distinct users", "982,014"),
        ("users with 0 actions", "393,003 (40.0%)"),
        ("top 1% share of actions", "26.5%"),
        ("top 5% share of actions", "68.2%"),
        ("top 10% share of actions", "83.8%"),
        ("Gini coefficient (synthetic)", "0.84"),
    ],
    impact_md=(
        "Unstratified global metrics are invalid. Pick the right alternative based on intervention shape:\n\n"
        "- If intervention = 'choose which users receive which email' → **within-cell ranking** (per-email or per-day).\n"
        "- If intervention = 'score all active users for the daily email' → **stratified evaluation** by tenure bucket, prior-action count, recent activity, cohort, send frequency.\n"
        "- Always report send-weighted AND user-weighted metrics in parallel.\n\n"
        "Default to stratified evaluation + group-normalized AUC/log-loss until F04's variance decomposition tells us how much of apparent user-variance is content-mix."
    ),
)


# ----------------------------------------------------------------------------
# F03. Same-row engagement leakage diagnostic
# ----------------------------------------------------------------------------


def _chart_f03() -> bytes:
    flags = ["opened", "clicked", "verified_opened"]
    raw_pre = [0.62, 0.71, 0.62]  # has independent raw event before actioned_at
    raw_post = [0.18, 0.14, 0.18]  # raw event exists but only after actioned_at
    imputed = [0.20, 0.15, 0.20]  # no raw event at all — purely imputed
    x = np.arange(len(flags))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x, raw_pre, color="#207e3c", label="raw event pre-actioned (usable)")
    ax.bar(
        x, raw_post, bottom=raw_pre, color="#c5500f", label="raw event post-actioned (still leaks)"
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
    return _png(fig)


F03 = dict(
    id="F03",
    slug="same_row_engagement_leakage",
    title="Same-row engagement leakage diagnostic",
    tldr="v2 imputes opened/clicked/verified_opened from actioned. Quantify how much.",
    wave=1,
    severity="BLOCKING",
    classification="confounder",
    question_md=(
        "v2 forces actioned ⇒ opened=clicked=verified_opened=True (see create_cache_v2.py:155-159). "
        "For each engagement flag, what fraction of actioned=True rows have an independent raw event in email_activities?"
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
    method_py_source="""
        # Bypass project data.py — needs raw email_activities
        from entities.civic_shout_engagement.email_activities_cache import cache as ea_cache
        ea = ea_cache.get(2)
        ue = user_emails.full().filter(pl.col("actioned"))
        # for each (user_id, email_id, date_sent), check raw events
        ea_pre = (
            ea.filter(pl.col("action_type").is_in(["open", "click"]))
              .join(ue, on=["user_id", "email_id"])
              .group_by(["user_id", "email_id", "action_type"])
              .agg(pl.col("created_at").min().alias("first_raw_at"))
        )
        # For each actioned row, fraction with raw_open / raw_click / purely imputed
        diag = (
            ue.join(ea_pre, on=["user_id", "email_id"], how="left")
              .group_by("action_type")
              .agg([
                  pl.col("first_raw_at").is_not_null().mean().alias("share_with_raw_event"),
                  # split pre/post actioned_at if available
              ])
        )
    """,
    numbers_table=[
        ("actioned rows with raw `open` event pre-action (synthetic)", "62%"),
        ("actioned rows with raw `click` event pre-action (synthetic)", "71%"),
        ("actioned rows with NO raw `click` event (purely imputed)", "15%"),
        ("actioned rows with NO raw `open` event (purely imputed)", "20%"),
    ],
    impact_md=(
        "**Feature whitelist** (enforce in F06/F07/F10):\n\n"
        "- Same-row opened/clicked/verified_opened are NEVER features for predicting same-row actioned.\n"
        "- Historical aggregates (opened_last_30d, clicked_last_90d) must be computed from raw email_activities events, not from user_emails — otherwise past-actioned imputation contaminates the feature.\n"
        "- Current-email features may include email content, send date, and user history strictly prior to date_sent.\n\n"
        "If purely-imputed share is high (>20%) on any flag, that flag is essentially a label proxy and must be dropped or built from raw events only."
    ),
)


# ----------------------------------------------------------------------------
# F04. Email/date opportunity audit
# ----------------------------------------------------------------------------


def _chart_f04() -> bytes:
    specs = [
        "user only",
        "email FE only",
        "date FE only",
        "user + email",
        "user + date",
        "user + email + date",
    ]
    r2 = [0.12, 0.31, 0.05, 0.39, 0.16, 0.44]
    x = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(x, r2, color="#2956a3")
    bars[1].set_color("#c5500f")  # highlight email FE
    bars[5].set_color("#207e3c")  # highlight joint
    ax.set_xticks(x)
    ax.set_xticklabels(specs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("variance share (R²) of actioned")
    ax.set_title("ANOVA decomposition: where does user-level variance actually live?")
    for i, v in enumerate(r2):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    return _png(fig)


F04 = dict(
    id="F04",
    slug="email_date_opportunity_audit",
    title="Email / date opportunity audit",
    tldr="How much apparent user-variance is actually email-mix or date-mix?",
    wave=2,
    severity="IMPORTANT",
    classification="confounder",
    question_md=(
        "Users who appear high-AR may simply have been exposed to easier-to-convert emails. "
        "Decompose the variance of actioned among user / email / date fixed effects."
    ),
    plain_english_md=(
        "Alice always gets the easy crossword puzzles. Bob always gets the hard ones. "
        "Alice solves 90% of hers; Bob solves 60% of his. Looking at the people, 'Alice "
        "is the smarter student.' Looking at the puzzles, 'Alice's puzzles are just "
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
        # Stratified sample for tractability on 130M rows
        df = user_emails.sample_percentage(0.05, seed=42)
        # Variance components via OLS dummy-encoded fixed effects
        import statsmodels.formula.api as smf
        specs = {
            "user": "actioned ~ C(user_id)",
            "email": "actioned ~ C(email_id)",
            "date": "actioned ~ C(date_sent_d)",
            "user_email": "actioned ~ C(user_id) + C(email_id)",
            "user_date": "actioned ~ C(user_id) + C(date_sent_d)",
            "joint": "actioned ~ C(user_id) + C(email_id) + C(date_sent_d)",
        }
        results = {name: smf.ols(f, data=df).fit().rsquared for name, f in specs.items()}
        # For each tenure bucket, compute raw AR vs FE-residualized AR
    """,
    numbers_table=[
        ("user-only R² (synthetic)", "0.12"),
        ("email-FE-only R² (synthetic)", "0.31"),
        ("date-FE-only R² (synthetic)", "0.05"),
        ("joint R² (synthetic)", "0.44"),
        ("variance attributable to user above email/date", "~13pp"),
    ],
    impact_md=(
        "If email-FE-only R² exceeds user-only R² (as the synthetic shows), apparent user-history features may largely be recovering Civic Shout's existing content-routing policy, not real user propensity.\n\n"
        "**Mandatory for F06/F07**: evaluate against same-email or same-day baselines wherever possible. A user feature that only adds lift against pooled baselines but not against email-FE residuals is recovering exposure, not propensity.\n\n"
        "**Intervention reframe**: if joint R² is dominated by email FE, the harness intervention shape should arguably move from 'score users for the daily email' to 'select among candidate petitions for the day.'"
    ),
)


# ----------------------------------------------------------------------------
# F05. 91-180d tenure valley decomposition
# ----------------------------------------------------------------------------


def _chart_f05() -> bytes:
    buckets = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr"]
    ar = np.array([0.127, 0.031, 0.039, 0.021, 0.062, 0.141])
    open_rate = np.array([0.426, 0.218, 0.338, 0.342, 0.374, 0.435])
    click_rate = np.array([0.155, 0.060, 0.065, 0.045, 0.092, 0.181])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(buckets))
    ax1.plot(x, open_rate, marker="o", label="open rate", color="#2956a3", linewidth=2)
    ax1.plot(x, click_rate, marker="o", label="click rate", color="#c5500f", linewidth=2)
    ax1.plot(x, ar, marker="o", label="action rate", color="#b00020", linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("rate")
    ax1.set_title("Funnel rates by tenure bucket")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.axvspan(2.5, 3.5, color="#fff3cf", alpha=0.5)
    ax1.text(3, 0.45, "valley", fontsize=8, ha="center", color="#7a6700")
    ax1.grid(alpha=0.25)

    # Funnel ratios — clicked-not-actioned share by tenure bucket
    cna_rate = click_rate - ar
    cna_share = cna_rate / click_rate
    ax2.bar(x, cna_share, color="#c5500f")
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("(click - action) / click")
    ax2.set_title("Clicked-not-actioned share by tenure")
    for i, v in enumerate(cna_share):
        ax2.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5)
    ax2.set_ylim(0, 1)
    return _png(fig)


F05 = dict(
    id="F05",
    slug="valley_decomposition_91_180d",
    title="91-180d tenure valley decomposition",
    tldr="Is the engagement valley user fatigue or content allocation? Where does it leak?",
    wave=2,
    severity="IMPORTANT",
    classification="both",
    question_md=(
        "91-180d has 34% open rate but 2.1% action rate. Is it user fatigue (user-side) or "
        "content-allocation bias (system-side)? Where in the funnel does the valley happen?"
    ),
    plain_english_md=(
        "Three users at month 4 on the list: Carol opens every email but never clicks. "
        "Dave clicks every email but never signs. Eve doesn't even open. The 91-180d "
        "'valley' (2.1% action rate despite a 34% open rate) is mostly Carols and Daves: "
        "they're still engaged enough to open or click but they're not crossing the "
        "finish line.\n\n"
        "Two stories for why. Story A: Civic Shout's routing happens to send WORSE "
        "petitions to month-4 users (content allocation). Story B: there's something "
        "about being 4 months in that personally erodes willingness (user lifecycle). "
        "These have different fixes — fix the routing vs. fix the messaging cadence.\n\n"
        "F05 looks at the SAME emails across tenure buckets. If the valley shows up "
        "even when we hold the petition constant, it's Story B (lifecycle). If it "
        "disappears, it's Story A (allocation). We also check whether the valley is "
        "pre-click or post-click — post-click leak points at the petition page, which "
        "is downstream of our email-level model."
    ),
    method_py_source="""
        ue = user_emails.full()
        ue = ue.with_columns(
            ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days()).alias("tenure_days")
        )
        bins = [0, 7, 30, 90, 180, 365, 730]
        ue = ue.with_columns(
            pl.col("tenure_days")
              .cut(bins, labels=["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "1-2yr"])
              .alias("tenure_bucket")
        )
        funnel = (
            ue.group_by("tenure_bucket")
              .agg(
                  pl.col("opened").mean().alias("open_rate"),
                  pl.col("clicked").mean().alias("click_rate"),
                  pl.col("actioned").mean().alias("action_rate"),
                  pl.col("unsubscribed").mean().alias("unsub_rate"),
                  pl.len().alias("n_sends"),
              )
        )
        # Same-email comparison: top-100 emails by send volume; AR by tenure within each
        top_emails = ue.group_by("email_id").agg(pl.len().alias("n")).top_k(100, by="n")["email_id"]
        within_email = (
            ue.filter(pl.col("email_id").is_in(top_emails))
              .group_by(["email_id", "tenure_bucket"])
              .agg(pl.col("actioned").mean().alias("action_rate"))
        )
    """,
    numbers_table=[
        ("91-180d users", "236,296"),
        ("91-180d sends", "22.0M"),
        ("91-180d open rate", "34.2%"),
        ("91-180d action rate", "2.13%"),
        ("91-180d clicked-not-actioned share (synthetic)", "55%"),
        ("does valley persist within same email? (synthetic)", "yes (mostly)"),
    ],
    impact_md=(
        "If valley persists within same email_id (same-email tenure comparison shows AR dip), it's a real lifecycle effect → 'days since first send' is a non-linear feature requiring bucketing.\n\n"
        "If valley disappears within same email_id, it's content-allocation bias → tenure feature is mostly recovering routing policy.\n\n"
        "**Funnel split** (per right panel): if 91-180d is mostly clicked-not-actioned, the petition page or petition-ask is the failure point, not the email. That's downstream of any email-level model and flags a side-quest plan.\n\n"
        "If 91-180d is mostly opened-not-clicked, the email content or subject is the failure — email-level model can address it."
    ),
)


# ----------------------------------------------------------------------------
# F06. Recency-weighted RFM baseline
# ----------------------------------------------------------------------------


def _chart_f06() -> bytes:
    features = [
        "AR last 7d",
        "AR last 30d",
        "AR last 90d",
        "Click rate last 30d",
        "Open rate last 30d",
        "days since last action",
        "log(prior actions)",
        "sends last 30d",
        "Multivariate (all)",
    ]
    auc = [0.71, 0.78, 0.81, 0.66, 0.62, 0.74, 0.79, 0.59, 0.84]
    colors = ["#2956a3"] * 8 + ["#207e3c"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(features, auc, color=colors)
    for bar, v in zip(bars, auc):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.5, 0.9)
    ax.set_title("Univariate + multivariate AUC of RFM baseline features")
    ax.axvline(x=0.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.invert_yaxis()
    return _png(fig)


F06 = dict(
    id="F06",
    slug="recency_rfm_baseline",
    title="Recency-weighted RFM baseline + send-frequency heterogeneity",
    tldr="What's the AUC of a model with prior-window AR/click/open rates only? This is the floor.",
    wave=3,
    severity="IMPORTANT",
    classification="feature-eng",
    question_md=(
        "What's the AUC of a model with ONLY prior-window action/click/open rates as features? "
        "This is the baseline every richer feature (F07 embedding centroid, F10 sequence) must beat."
    ),
    plain_english_md=(
        "Predicting whether someone will buy ice cream today. The dumbest useful "
        "predictor is: 'how often did they buy ice cream in the past 30 days?' Recency "
        "+ frequency. It's not deep, it doesn't use weather, doesn't use demographics. "
        "But it's a strong baseline.\n\n"
        "Any fancier model — embeddings, sequence features, content matching — has to "
        "beat that dumb predictor by enough to justify its complexity, or it's not "
        "worth shipping. F06 builds the dumb predictor for petition signing using only "
        "prior-window action rates, click rates, open rates, days-since-last-action, "
        "and prior action count. Everything else in Wave 3 has to clear this floor.\n\n"
        "Subtle catch (from F03): the click/open features have to come from RAW email "
        "activity events, not from our user_emails table — otherwise past-imputed clicks "
        "leak into the feature."
    ),
    method_py_source="""
        # Build features from RAW email_activities (per F03 rule — no imputation contamination)
        ea = ea_cache.get(2)
        ue = user_emails.sample_percentage(0.05, seed=42)
        # For each (user_id, date_sent), compute prior-window aggregates strictly < date_sent
        features = compute_prior_window_features(
            ue, ea,
            windows_days=[7, 30, 90],
            metrics=["action", "click", "open"],
        )
        features = features.with_columns(
            (pl.col("date_sent") - pl.col("last_action_at")).dt.total_days().alias("days_since_last_action"),
            pl.col("prior_action_count").log1p().alias("log_prior_action_count"),
        )
        # Simple LR fit + AUC per feature univariate + multivariate
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        # Stratified test split by tenure bucket
    """,
    numbers_table=[
        ("multivariate AUC (synthetic)", "0.84"),
        ("best univariate (log prior actions, synthetic)", "0.79"),
        ("AUC in 0-7d tenure bucket (synthetic)", "0.62"),
        ("AUC in 91-180d tenure bucket (synthetic)", "0.69"),
        ("AUC in 1-2yr tenure bucket (synthetic)", "0.88"),
    ],
    impact_md=(
        "This baseline is the floor that F07's embedding centroid and F10's sequence features must beat materially after history-depth controls — otherwise those richer features are not pulling weight.\n\n"
        "Per-tenure-bucket AUC drop in early-tenure buckets is expected (sparse history); flag as a deployment risk for new users.\n\n"
        "Send-frequency heterogeneity (sends last 30d) is the F06 sub-feature for the original 'active-30d filter is unstable' concern. If sends_last_30d shows up in feature-importance, the harness population definition is leaking into the features."
    ),
)


# ----------------------------------------------------------------------------
# F07. Embedding affinity with ablation ladder
# ----------------------------------------------------------------------------


def _chart_f07() -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Ablation ladder
    rungs = ["A: tenure+freq", "B: +prior actions", "C: +recency (F06)", "D: +centroid"]
    auc = [0.65, 0.78, 0.84, 0.851]
    ax1.bar(rungs, auc, color=["#aaa", "#888", "#666", "#207e3c"])
    for i, v in enumerate(auc):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax1.set_ylim(0.6, 0.9)
    ax1.set_ylabel("AUC")
    ax1.set_title("Ablation ladder — does centroid add lift over F06?")
    ax1.tick_params(axis="x", rotation=15, labelsize=8)

    # Centroid stability scatter — cosine vs prior action count
    prior_actions = np.concatenate(
        [
            _RNG.integers(0, 1, size=600),
            _RNG.integers(1, 5, size=250),
            _RNG.integers(5, 20, size=100),
            _RNG.integers(20, 250, size=50),
        ]
    )
    # cosine increases & stabilizes with prior actions
    cosines = np.tanh(np.log1p(prior_actions) / 3.0) * 0.7 + _RNG.normal(
        0, 0.05, size=len(prior_actions)
    )
    ax2.scatter(prior_actions + 0.5, cosines, alpha=0.35, s=8, color="#2956a3")
    ax2.set_xscale("log")
    ax2.set_xlabel("prior action count (log)")
    ax2.set_ylabel("cos(centroid, today_email)")
    ax2.set_title("Centroid stability — is centroid quality a power-user proxy?")
    ax2.grid(alpha=0.25)
    return _png(fig)


F07 = dict(
    id="F07",
    slug="embedding_affinity_with_ablations",
    title="Embedding affinity (with full ablation ladder)",
    tldr="Does cosine(user_centroid, today_email) add lift beyond F06? Or is it a power-user detector?",
    wave=3,
    severity="IMPORTANT",
    classification="feature-eng",
    question_md=(
        "Does cosine(user_centroid, today_email_dense_vec) add predictive lift beyond the F06 baseline, "
        "after controlling for history depth? Or is centroid quality just a proxy for being a power user?"
    ),
    plain_english_md=(
        "Imagine each user has a 'taste profile' — for movies, it'd be a point in "
        "'movie space' where their favorites cluster (sci-fi fan vs. rom-com fan vs. "
        "documentary watcher). For Civic Shout, the profile is a point in 'petition "
        "topic space' near the petitions they've signed (climate cluster vs. healthcare "
        "cluster vs. immigration cluster).\n\n"
        "Today we want to send Alice an email. We measure: 'how close is today's "
        "petition to Alice's profile?' If close, she's more likely to sign. That's "
        "the feature.\n\n"
        "But there's a trap. Power users (who've signed 50 petitions) have TIGHT, "
        "accurate profiles. New users (who've signed 1) have wobbly profiles. So "
        "'close to profile' may really just be 'this user has lots of past signatures' "
        "— which means 'this is a power user' — which is UNACTIONABLE: we can't make "
        "new users into power users by sending them better emails. F07's ablation "
        "ladder + permutation test asks: does the closeness add real signal once you "
        "control for how many signatures the user has? If no, the feature is dead."
    ),
    method_py_source="""
        em = emails.full()  # 4634 rows, dense_vector list[f32] (1536d)
        ue = user_emails.sample_percentage(0.05, seed=42)
        # Per-user centroid from actioned emails' dense vectors, with shrinkage toward global mean
        actioned_pairs = ue.filter(pl.col("actioned")).select(["user_id", "email_id"])
        centroids = (
            actioned_pairs.join(em.select(["email_id", "dense_vector"]), on="email_id")
                          .group_by("user_id")
                          .agg(pl.col("dense_vector").list.mean().alias("centroid"))
        )
        # Funnel-aware centroids: clicked-not-actioned, opened-not-clicked centroids
        # Ablation ladder: A (tenure+freq) → B (+prior actions) → C (+recency) → D (+centroid sim)
        # Permutation control: shuffle centroids within prior-action-count bucket
        # Within-history-bucket AUC
        # Cold-start fallback: zero-history users get population centroid
    """,
    numbers_table=[
        ("ablation D AUC over C (synthetic)", "+0.011"),
        ("centroid stability — corr(cosine, log prior actions) (synthetic)", "0.62"),
        ("permutation control AUC (synthetic)", "0.847"),
        ("permutation gap (true centroid - shuffled)", "0.004 (suspicious)"),
        ("AUC within 5-19 prior-action bucket (synthetic)", "0.83"),
        ("AUC within 0 prior-action bucket — cold start", "centroid not applicable"),
    ],
    impact_md=(
        "If permutation gap is small (synthetic shows 0.004), the centroid is mostly a power-user detector, not a topic-affinity signal. The 'rich' embedding feature would be deprecated for routing.\n\n"
        "If ablation D adds substantial lift AND survives the permutation test (gap > 0.02) AND the within-bucket lift holds for moderate-history users, the centroid is a real feature.\n\n"
        "**Boilerplate diagnostic**: cluster emails by dense vector and eyeball nearest neighbors. If clusters track templates/organizations rather than topics, centroid is misleading even when stable."
    ),
)


# ----------------------------------------------------------------------------
# F08. Action survival / hazard
# ----------------------------------------------------------------------------


def _chart_f08() -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # Hazard curve — action probability vs sends-since-last-action
    sends_since = np.arange(0, 60)
    hazard = 0.32 * np.exp(-sends_since / 5) + 0.04 + _RNG.normal(0, 0.005, size=len(sends_since))
    hazard = np.clip(hazard, 0, None)
    ax1.plot(sends_since, hazard, color="#4a3aff", linewidth=2)
    ax1.fill_between(sends_since, hazard, alpha=0.15, color="#4a3aff")
    ax1.set_xlabel("sends since last action")
    ax1.set_ylabel("P(action | this send)")
    ax1.set_title("Action hazard — spike after action, then decay")
    ax1.grid(alpha=0.25)
    ax1.axhline(y=0.099, color="#aaa", linestyle="--", linewidth=0.8)
    ax1.text(45, 0.105, "pooled AR", fontsize=8, color="#666")

    # Lifetime action count
    counts = [393_003, 230_000, 130_000, 95_000, 65_000, 35_000, 18_000, 9_000, 4_500, 2_500]
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21-40", "41-80", "81-160", "161-300", "300+"]
    ax2.bar(labels, counts, color="#2956a3")
    ax2.set_yscale("log")
    ax2.set_xlabel("lifetime action count")
    ax2.set_ylabel("users (log)")
    ax2.set_title("Lifetime action count — power-law")
    ax2.tick_params(axis="x", rotation=30, labelsize=7.5)
    return _png(fig)


F08 = dict(
    id="F08",
    slug="action_survival_hazard",
    title="Action survival / hazard",
    tldr="After signing, does the next-action probability spike, decay, or persist? Streak users?",
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
        "The 'hazard curve' answers: on the day right after signing, what's the action "
        "rate? On day 2 after? Day 30? If the curve spikes right after a signature "
        "(e.g., 32% AR the next day vs. the 9.87% baseline), users are primed — we "
        "should send them more, fast. If it drops, they're satisfied — give them rest.\n\n"
        "Most production models miss this temporal pattern entirely. F08 measures it "
        "and creates cheap features like `is_in_action_streak`, `sends_since_last_action`. "
        "These may outperform the fancy embedding feature on simpler grounds."
    ),
    method_py_source="""
        ue = user_emails.full().sort(["user_id", "date_sent"])
        # For each row, compute sends_since_last_action and days_since_last_action
        ue = ue.with_columns(
            pl.col("actioned").cumsum().over("user_id").alias("action_seq"),
        )
        # Group by action_seq to find inter-action gaps
        gaps = (
            ue.filter(pl.col("actioned"))
              .group_by("user_id")
              .agg(
                  pl.col("date_sent").alias("action_dates"),
              )
              .with_columns(
                  pl.col("action_dates").list.diff().alias("inter_action_gaps")
              )
        )
        # Hazard at lag k: action_rate among rows where sends_since_last_action == k
        hazard = (
            ue.group_by("sends_since_last_action")
              .agg(pl.col("actioned").mean().alias("hazard_at_lag"))
        )
        # Streak users: ≥3 consecutive actioned sends
        # One-and-done: lifetime_actions == 1
    """,
    numbers_table=[
        ("hazard at lag 0 (immediately after action, synthetic)", "32%"),
        ("hazard at lag 30 sends (synthetic)", "8.5%"),
        ("streak users (≥3 consecutive actioned, synthetic)", "~4,200"),
        ("one-and-done users (lifetime=1)", "~70,000"),
        ("hazard half-life (sends)", "~5"),
    ],
    impact_md=(
        "**Cheap high-signal features** for production: `is_in_action_streak`, `sends_since_last_action`, `hazard_at_current_lag`. Likely beat embedding affinity on simpler grounds.\n\n"
        "**Streak-user detection** is operationally valuable — a sustained-action user can tolerate lower per-email match quality, opening room for exploration sends.\n\n"
        "If the hazard spike at lag 0-3 is large (synthetic shows 32%→18% at lag 3), the model should explicitly include a recent-action-recency feature. Without it, the model under-scores users right after their last action — exactly when they're most likely to act again."
    ),
)


# ----------------------------------------------------------------------------
# F09. Competing-risk unsubscribe modeling
# ----------------------------------------------------------------------------


def _chart_f09() -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # 3x3 cross-tab heatmap: action probability tertile × unsub hazard tertile
    rates = np.array(
        [
            # cell content: AR (top), unsub rate (bottom)
            [(0.012, 0.0018), (0.024, 0.0038), (0.042, 0.0055)],  # low AR
            [(0.041, 0.0021), (0.082, 0.0044), (0.125, 0.0064)],  # mid AR
            [(0.118, 0.0019), (0.245, 0.0050), (0.486, 0.0098)],  # high AR
        ]
    )
    # Show population share text + ar text
    pop_share = np.array(
        [
            [0.11, 0.09, 0.06],
            [0.09, 0.11, 0.08],
            [0.06, 0.10, 0.30],
        ]
    )
    ax1.imshow(pop_share, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            ar = rates[i, j, 0]
            us = rates[i, j, 1]
            ax1.text(
                j,
                i,
                f"pop {pop_share[i, j]:.0%}\nAR {ar:.1%}\nunsub {us:.2%}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if pop_share[i, j] > 0.18 else "black",
            )
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["low", "mid", "high"], fontsize=8)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["low", "mid", "high"], fontsize=8)
    ax1.set_xlabel("predicted unsub hazard tertile")
    ax1.set_ylabel("predicted action probability tertile")
    ax1.set_title("Cross-tab: action × unsub joint distribution")

    # Hazard curve over tenure
    tenure_d = np.arange(0, 730)
    survival = np.exp(-tenure_d / 350)
    ax2.plot(tenure_d, survival, color="#b00020", linewidth=2)
    ax2.fill_between(tenure_d, survival, alpha=0.15, color="#b00020")
    ax2.set_xlabel("tenure (days)")
    ax2.set_ylabel("P(not yet unsubscribed)")
    ax2.set_title("Unsub survival curve — KM estimate")
    ax2.grid(alpha=0.25)
    ax2.axvline(x=57, color="#666", linestyle=":")
    ax2.text(62, 0.92, "median 57 emails ≈ ?d", fontsize=8, color="#666")
    return _png(fig)


F09 = dict(
    id="F09",
    slug="competing_risk_unsubscribe",
    title="Competing-risk unsubscribe modeling",
    tldr="High-AR users may also be high-unsub-risk. Naive lift overstates if so.",
    wave=3,
    severity="IMPORTANT",
    classification="confounder",
    question_md=(
        "49.3% of users ever unsubscribe (median 57 emails). Any per-email action-lift optimization "
        "must trade off against unsub hazard. What's the joint distribution of (action probability, unsub hazard)?"
    ),
    plain_english_md=(
        "A gym wants more visits per member. The easy win: target your most-frequent "
        "visitors with extra promotions. But what if those frequent visitors are also "
        "the ones closest to cancelling (because they're saturating)? Pushing them "
        "harder makes them quit.\n\n"
        "You win the visits-per-member battle this month but lose the audience-size "
        "war by Q4. Total visits crater because the audience shrunk faster than "
        "per-member intensity grew.\n\n"
        "Same risk for Civic Shout: 49% of users unsubscribe; the median user unsubs "
        "around their 57th email. If high-AR users are also high-unsub-hazard users, "
        "a model that optimizes purely for action probability ships short-term lift "
        "and long-term audience collapse. F09 maps the joint distribution so the "
        "harness CAN'T silently make this trade — it has to report both metrics."
    ),
    method_py_source="""
        from lifelines import KaplanMeierFitter
        ue = user_emails.full().sort(["user_id", "date_sent"])
        # Per (user, send), compute hazard of unsub in next 30 sends
        ue = ue.with_columns(
            pl.col("unsubscribed").shift(-30).over("user_id").alias("unsub_within_30s_forward")
        )
        # Bin users by predicted action probability (from F06 model) and predicted unsub hazard
        # 3x3 cross-tab: tertile × tertile
        cross_tab = (
            ue.group_by(["ar_tertile", "unsub_haz_tertile"])
              .agg(
                  pl.col("actioned").mean().alias("ar"),
                  pl.col("unsub_within_30s_forward").mean().alias("unsub_rate"),
                  pl.len().alias("n"),
              )
        )
        # KM unsub survival curve by tenure
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed)
    """,
    numbers_table=[
        ("users with high AR AND high unsub hazard (synthetic)", "30%"),
        ("AR in high/high cell", "48.6%"),
        ("unsub rate in high/high cell", "0.98%"),
        ("AR in low/low cell", "1.2%"),
        ("correlation(predicted AR, predicted unsub hazard)", "+0.41 (synthetic)"),
    ],
    impact_md=(
        "If AR and unsub-hazard are positively correlated (synthetic shows +0.41), a model optimized purely for action probability will systematically over-target users about to unsubscribe → the active-30d population shrinks → metric improvement is short-lived.\n\n"
        "**Harness rule**: report BOTH action-rate lift AND unsub-rate change at deployment. If unsub change > X% threshold, the lift number is overstated; multi-objective optimization is required.\n\n"
        "**Operational lever**: send-frequency throttling for high-unsub-hazard users — protect the active cohort even at the cost of short-term lift."
    ),
)


# ----------------------------------------------------------------------------
# F10. Temporal autocorrelation
# ----------------------------------------------------------------------------


def _chart_f10() -> bytes:
    features = [
        "opened_last_1",
        "opened_last_3",
        "opened_last_5",
        "clicked_last_1",
        "clicked_last_3",
        "clicked_last_5",
        "consec_unopened_count",
        "action_in_last_5",
        "actioned_last_1",
    ]
    auc = [0.69, 0.74, 0.76, 0.72, 0.79, 0.81, 0.67, 0.83, 0.78]
    colors = ["#2956a3"] * 6 + ["#c5500f"] + ["#207e3c"] + ["#207e3c"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(features, auc, color=colors)
    for bar, v in zip(bars, auc):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("univariate AUC")
    ax.set_xlim(0.5, 0.9)
    ax.axvline(x=0.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.set_title("Univariate AUC of prior-send sequence features")
    ax.invert_yaxis()
    return _png(fig)


F10 = dict(
    id="F10",
    slug="temporal_autocorrelation",
    title="Temporal autocorrelation of opens / clicks / actions",
    tldr="Cheap windowed-aggregation features — may beat F06's RFM baseline.",
    wave=3,
    severity="IMPORTANT",
    classification="feature-eng",
    question_md=(
        "How informative are 'opened last send,' 'opened last 3 sends,' 'consecutive unopened sends' features? "
        "Is there a recent-action chronotype signal?"
    ),
    plain_english_md=(
        "Predicting whether you'll open today's email. The single strongest predictor "
        "is rarely 'who you are' (demographics) or 'when we sent it' (time of day). "
        "It's: did you open YESTERDAY's email? Recent behavior beats deep biography "
        "in engagement-prediction problems, almost always.\n\n"
        "F10 tests the same pattern for petition SIGNING — features like 'opened 3 "
        "of the last 5 sends,' 'actioned in the last 5 sends,' 'consecutive unopened "
        "streak.' These are cheap to compute (just windowed aggregations) and "
        "historically punch above their weight.\n\n"
        "F06's RFM baseline doesn't include them (it uses rolling rates, not "
        "sequences), so F10 is the natural follow-up: does sequence add lift over "
        "rate? Subtle: per F03, the underlying opened/clicked signals have to come "
        "from raw events, or we re-import the imputation leakage."
    ),
    method_py_source="""
        ue = user_emails.full().sort(["user_id", "date_sent"])
        # Per-row windowed aggregates over previous N sends per user
        for n in [1, 3, 5, 10]:
            for flag in ["opened", "clicked", "actioned"]:
                ue = ue.with_columns(
                    pl.col(flag).shift(1)
                      .rolling_sum(window_size=n)
                      .over("user_id")
                      .alias(f"{flag}_last_{n}")
                )
        # Consecutive-unopened streak count (resets on any open)
        ue = ue.with_columns(
            (pl.col("opened").shift(1).over("user_id").cum_count_when(False))
              .alias("consec_unopened_count")
        )
        # Univariate AUC per sequence feature; multivariate add-on to F06 baseline
    """,
    numbers_table=[
        ("best univariate (action_in_last_5, synthetic)", "0.83"),
        ("clicked_last_5 AUC (synthetic)", "0.81"),
        ("opened_last_3 AUC (synthetic)", "0.74"),
        ("delta vs F06 baseline (synthetic)", "+0.03 multivariate"),
        ("compute cost relative to F06", "1.2× (cheap)"),
    ],
    impact_md=(
        "Sequence features are CHEAP — single windowed-aggregation per user per row. If they beat F06's recency baseline (synthetic shows +0.03 over F06), they belong in the production feature set.\n\n"
        "**Per F03 rule**: features must be computed strictly from prior-to-this-send events, and the underlying engagement signals (opened/clicked) must come from raw email_activities if F03 found large imputation contamination.\n\n"
        "**Watch out for the F03 contamination cascade**: if `opened_last_N` is computed from user_emails (post-imputation), it inherits the leakage. Use raw events when in doubt."
    ),
)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


FINDINGS = [
    (F00, _chart_f00),
    (F01, _chart_f01),
    (F02, _chart_f02),
    (F03, _chart_f03),
    (F04, _chart_f04),
    (F05, _chart_f05),
    (F06, _chart_f06),
    (F07, _chart_f07),
    (F08, _chart_f08),
    (F09, _chart_f09),
    (F10, _chart_f10),
]


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for finding, chart_fn in FINDINGS:
        png_bytes = chart_fn()
        html = render_finding(
            **{k: v for k, v in finding.items() if k not in {"slug", "id"}},
            finding_id=finding["id"],
            chart_png_bytes=png_bytes,
            provenance=_PROVENANCE,
            is_mockup=True,
        )
        out_path = _OUT_DIR / f"{finding['id']}_{finding['slug']}.html"
        out_path.write_text(html, encoding="utf-8")
        index_rows.append({**finding, "is_mockup": True})
        print(f"wrote {out_path}")

    index_html = render_index(index_rows)
    (_OUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(f"wrote {_OUT_DIR / 'index.html'}")
    print(f"\nopen in browser:\n  open {_OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
