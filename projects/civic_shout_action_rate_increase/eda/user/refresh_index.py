"""Regenerate findings/index.html introspecting which HTMLs are real vs mockup.

A "real" finding is one whose HTML does NOT contain the mockup banner div. The
index renders MOCKUP badges only for findings that have not yet been replaced
by their real EDA run.

Run:

    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.refresh_index
"""

from __future__ import annotations

from pathlib import Path

from projects.civic_shout_action_rate_increase.eda.user._html import render_index

_FINDINGS_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)


_FINDING_META: list[dict] = [
    dict(
        id="F00",
        slug="outcome_cardinality",
        title="Outcome cardinality check",
        tldr="Binary target is lossless — `is_first_in_window` enforces 1 sig/pair by construction.",
        wave=1,
        severity="BLOCKING",
        classification="confounder",
    ),
    dict(
        id="F01",
        slug="cohort_aging_decomposition",
        title="YoY drift / cohort-aging decomposition",
        tldr="Cohort aging explains 198% of pooled drift (>100% because new low-AR cohorts dilute the pooled line).",
        wave=1,
        severity="BLOCKING",
        classification="confounder",
    ),
    dict(
        id="F02",
        slug="concentration_and_evaluation_stratification",
        title="Action concentration → evaluation stratification",
        tldr="Top 1% → 26.5% of actions; top 10% → 84%. Unstratified metrics are invalid.",
        wave=2,
        severity="IMPORTANT",
        classification="confounder",
    ),
    dict(
        id="F03",
        slug="same_row_engagement_leakage",
        title="Same-row engagement leakage diagnostic",
        tldr="71.6% of `opened` is purely imputed. Drop all three engagement flags from feature whitelist.",
        wave=1,
        severity="BLOCKING",
        classification="confounder",
    ),
    dict(
        id="F04",
        slug="email_date_opportunity_audit",
        title="Email / date opportunity audit",
        tldr="How much apparent user-variance is actually email-mix or date-mix?",
        wave=2,
        severity="IMPORTANT",
        classification="confounder",
    ),
    dict(
        id="F05",
        slug="valley_decomposition_91_180d",
        title="91-180d tenure valley decomposition",
        tldr="Is the engagement valley user fatigue or content allocation? Where does it leak?",
        wave=2,
        severity="IMPORTANT",
        classification="both",
    ),
    dict(
        id="F06",
        slug="recency_rfm_baseline",
        title="Recency-weighted RFM baseline + send-frequency heterogeneity",
        tldr="What's the AUC of a model with prior-window AR/click/open rates only? This is the floor.",
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
    ),
    dict(
        id="F07",
        slug="embedding_affinity_with_ablations",
        title="Embedding affinity (with full ablation ladder)",
        tldr="Does cosine(user_centroid, today_email) add lift beyond F06? Or is it a power-user detector?",
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
    ),
    dict(
        id="F08",
        slug="action_survival_hazard",
        title="Action survival / hazard",
        tldr="After signing, does the next-action probability spike, decay, or persist? Streak users?",
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
    ),
    dict(
        id="F09",
        slug="competing_risk_unsubscribe",
        title="Competing-risk unsubscribe modeling",
        tldr="High-AR users may also be high-unsub-risk. Naive lift overstates if so.",
        wave=3,
        severity="IMPORTANT",
        classification="confounder",
    ),
    dict(
        id="F10",
        slug="temporal_autocorrelation",
        title="Temporal autocorrelation of opens / clicks / actions",
        tldr="Cheap windowed-aggregation features — may beat F06's RFM baseline.",
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
    ),
]


def _is_mockup(html_path: Path) -> bool:
    """Heuristic: a real HTML has no `mockup-banner` div in its body."""
    if not html_path.exists():
        return True  # treat missing as still-mockup
    text = html_path.read_text(encoding="utf-8")
    # The mockup banner div appears in the body when is_mockup=True is passed.
    # Don't confuse with the `.mockup-banner` CSS class declaration in <style>.
    return '<div class="mockup-banner">' in text


def main() -> None:
    rows = []
    for meta in _FINDING_META:
        html_path = _FINDINGS_DIR / f"{meta['id']}_{meta['slug']}.html"
        rows.append({**meta, "is_mockup": _is_mockup(html_path)})

    index_html = render_index(sorted(rows, key=lambda r: r["id"]))
    (_FINDINGS_DIR / "index.html").write_text(index_html, encoding="utf-8")

    real = [r["id"] for r in rows if not r["is_mockup"]]
    mockup = [r["id"] for r in rows if r["is_mockup"]]
    print(f"index refreshed: real={len(real)} ({', '.join(real) or '-'})")
    print(f"                 mockup={len(mockup)} ({', '.join(mockup) or '-'})")


if __name__ == "__main__":
    main()
