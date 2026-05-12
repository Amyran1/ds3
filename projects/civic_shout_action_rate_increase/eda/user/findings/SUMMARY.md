# User-level EDA — Summary

**Project**: `civic_shout_action_rate_increase`
**Goal** (per `GOAL.md`): lift petitions-signed-per-email by ~10%
**Pooled baseline**: 9.87% across 129,190,540 (user, email) sends, 2024-04-21 → 2026-04-20
**Plan**: `/Users/aaronmyran/.claude/plans/velvety-swimming-spindle.md`
**Executed**: 2026-05-12, 11 findings (F00–F10), three waves, three-tier 1%→5%→full sampling cadence

---

## TL;DR

1. **Harness target = binary `actioned`.** Lossless by construction in attributed_actions v2 (F00).
2. **Train/test split = temporal holdout by `send_date`, stratified by tenure bucket.** YoY drift is 198% cohort aging — random row splits leak future maturity (F01).
3. **DROP `opened` / `clicked` / `verified_opened` from the feature whitelist.** 28-72% purely imputed by v2's rule; severe label leakage. Use raw `email_activities` events only for engagement aggregates (F03).
4. **User-level signal is genuine, not routing-policy recovery.** User R² = 0.27 vs email R² = 0.07 (F04). 4× dominance.
5. **The strongest single feature in the EDA is `actioned_last_10` (AUC 0.8564).** Lag-0 post-action hazard is 39.18%, ~4× pooled (F08, F10). Recency of last action is the dominant signal — "strike while the iron is hot."
6. **No action-lift vs unsub trade-off** (F09: Spearman = -0.185, weakly negative). High-AR users are not high-unsub-risk.
7. **The pre-EDA's "91-180d valley" story is overturned.** Real tenure-AR curve is monotone-positive after a small 8-30d dip; the dip is content-allocation, not lifecycle (F05).
8. **Embedding-centroid affinity does NOT pay for itself.** F07 full-tier ablation: cosine adds only +0.0012 AUC over recency features, and the permutation control (shuffle centroids within prior-action-count buckets) gives zero gap. Centroid is a power-user detector, not a topic-affinity signal. Save the compute.
9. **Project lift target should be re-baselined.** Pooled 9.87% is mechanically suppressed by new cohort dilution; the 2024 cohort alone runs 7.80% → 17.13%. Recommend a separate GOAL.md revision.

---

## What we now know — harness design table

| Decision | Pre-EDA assumption | Post-EDA answer | Source |
|---|---|---|---|
| Target shape | Count or binary, TBD | **Binary** (attribution layer enforces 1 sig/pair) | F00 |
| Split convention | Pooled or random | **Temporal + tenure-stratified** | F01 |
| Pooled baseline | 9.87% | Mechanically suppressed; 2024 cohort alone 7.80→17.13% | F01 |
| User concentration | Top 1% ≈ 27% of actions | Top 1% = **26.5%**, top 10% = **83.8%**, Gini = **0.885** | F02 |
| Engagement-flag features | Use opened/clicked/verified_opened | **DROP all three from user_emails.** Use raw events. | F03 |
| User-vs-routing variance | Unknown | User R² = **0.27** vs email = 0.07 vs date = 0.015 → user dominates | F04 |
| 91-180d "valley" | 2.1% AR fatigue dip | **Misread.** Real dip is 8-30d (4.5%); curve is monotone-positive after. Same-email comparison: dip is allocation, not lifecycle. | F05 |
| Best feature family | Embedding centroid (guess) | **Prior-action recency.** `actioned_last_10` alone = AUC 0.8564 | F08, F10 |
| Lift-vs-unsub trade-off | Likely real | **Empirically absent.** Spearman = -0.185 | F09 |
| RFM baseline (floor) | TBD | AUC = 0.8266 (5pct only — full requires refactor) | F06 |
| Embedding affinity (cosine) | Likely positive lift | **MARGINAL → effectively zero.** Cosine +0.0012 over recency; permutation gap = 0.0000. Centroid is a power-user proxy. Drop. | F07 |

---

## Per-finding one-liners

### Wave 1 — Gating (BLOCKING)

| ID | Finding | Verdict |
|---|---|---|
| F00 | Outcome cardinality check | Binary target is lossless. `is_first_in_window=True` enforces 1 signature per pair by construction. The "petitions per email" question lives at the attribution layer, not modeling. |
| F01 | Cohort aging decomposition | Aging explains 198% of pooled YoY drift (>100% because new low-AR cohorts dilute the pooled line). 2024 cohort 7.80→17.13%. **2026 cohort entry = 4.4% — anomalous, may be Q1 immaturity. Follow-up flagged.** |
| F03 | Same-row engagement leakage | `opened` 71.6% purely imputed, `clicked` 28%, `verified_opened` 71.6%. **Drop all three.** Historical engagement aggregates must come from raw `email_activities`. |

### Wave 2 — Evaluation + content confounds (IMPORTANT)

| ID | Finding | Verdict |
|---|---|---|
| F02 | Action concentration + evaluation stratification | Top 1% = 26.5% of actions, top 10% = 83.8%, Gini = 0.885. 40% of users have zero actions. Stratify evaluation by tenure × prior-action count; report send-weighted AND user-weighted in parallel. |
| F04 | Email/date opportunity audit | User R² = 0.27 vs email = 0.07 vs date = 0.015. **User propensity dominates email-mix by ~4×.** User features are not primarily recovering routing policy. Same-email residualization stays as a robustness check only. |
| F05 | 91-180d valley decomposition | Real tenure-AR: 7.2% / 4.5% / 6.1% / 8.4% / 12.4% / 15.8%. **Monotone-positive after a small 8-30d dip — no U-shape.** Same-email comparison: dip does NOT persist within email_id → it's content-allocation, not lifecycle. Pre-EDA's "91-180d at 2.1%" was likely a conditional rate misread as marginal. |

### Wave 3 — Feature hypotheses (IMPORTANT)

| ID | Finding | Verdict |
|---|---|---|
| F06 | RFM baseline + send-freq heterogeneity | **5pct only.** Full tier OOM'd (16 `join_asof` ops + 17GB at 5pct). Best AUC = 0.8266; top univariate = `log1p_prior_action_count` (AUC 0.7917). Refactor to `rolling_sum().over("user_id")` lazy streaming is the recommended F06_v2. |
| F07 | Embedding affinity ablation | **Full-tier MARGINAL.** Ablation A→D: 0.6740 / 0.8141 / 0.8417 / **0.8429**. Cosine adds only +0.0012 over recency features. **Permutation gap = 0.0000** — shuffling centroids within prior-action-count buckets gives the same AUC, so cosine carries no information beyond what prior-history already captures. Within-bucket AUCs (after controlling for prior_action_count) stay 0.69-0.77 — the apparent 0.84 is between-bucket variance, not within-user topic match. Centroid stability r = 0.070. **Verdict: drop embedding centroid from feature set. Save the 33-min/run compute + 33-GB memory cost.** Wall = 2021s = 33.7 min, peak RSS = 33 GB. |
| F08 | Action survival / hazard | **Lag-0 hazard = 39.18%** (~4× pooled). 115,327 streak users, 202,787 one-and-done. Strong "strike while iron is hot" pattern. Recommended features: `sends_since_last_action`, `is_in_action_streak`, `action_in_last_5`, `hazard_at_current_lag`, `lifetime_actions`. |
| F09 | Competing-risk unsubscribe | Spearman corr(predicted AR, predicted unsub-hazard) = **-0.185** at full (weakly negative). **No action-lift vs unsub trade-off.** KM median time-to-unsub = 125 sends. Real unsub spike is the 0-7d cohort (1.4%), early-tenure volume management — separate concern. |
| F10 | Temporal autocorrelation | **Multivariate AUC = 0.8655.** Top univariate: `actioned_last_10` = 0.8564 (beats F06's full multivariate). Sequence features ship. F03-compliant (built from raw events). |

---

## Methodology surprises

### Row-vs-user sampling bias (caught 4 times by the cadence)

Polars `sample_percentage` is **row-level**, which systematically biases user-aggregate metrics on small samples:

| Finding | Metric | 1pct | 5pct | Full | Bias direction |
|---|---|---|---|---|---|
| F02 | Top-1% action share | 20.0% | 22.6% | 26.5% | Undercounts concentration |
| F04 | User R² | 0.4585 | 0.3162 | 0.2656 | Overstates user explanatory power |
| F08 | Lag-0 action hazard | 12.03% | 21.11% | 39.18% | Undercounts streak/recency signal |
| F09 | Spearman(AR, unsub) | +0.458 | +0.270 | -0.185 | Sign-flips — sample inverts the conclusion |

**Rule for future EDAs:** for any per-user metric, the full-tier number is authoritative. The cadence's 1% / 5% probes are for *wall-time validation*, not statistical reads. When a metric is user-aggregate, prefer a *user-level* seeded sample (sample unique user_ids, take all their rows) over row-level sampling.

### Scaling-rule calibration

The plan's "escalate if 5%×20 > 30min" rule assumes linear scaling. Two findings violated it:

- F10: 5pct = 17s → full = 1094s (**64× ratio**, not 20×). Rolling-window features need denominator depth that grows super-linearly.
- F06: 5pct = 29s → full = OOM at 30 min. 16 `join_asof` operations are not the right pattern for windowed features at 130M-row scale.

Recommend updating the plan's cadence section: for windowed-aggregation / per-user-state features, project full-tier as 5pct × 60-100, not × 20.

### Pre-EDA brainstorm reliability

The pre-EDA brainstorm produced numbers that turned out to be:
- ✓ F01 directional (aging dominates) — quantitatively even stronger (100% → 198%)
- ✓ F02 directional (top 1% concentration) — exact (26.5% in both)
- ✓ F03 directional (engagement contamination) — quantitatively dramatic (20% synth → 71.6% real)
- ✗ **F05 fundamentally wrong** (91-180d valley shape and location)

The brainstorm's 1% sample with a non-canonical tenure derivation produced a misleading shape. **Lesson**: brainstorm-stage numbers without the cadence should be marked "directional, unverified" and flagged for re-derivation in Wave 1.

---

## Open items / follow-ups (NOT in this EDA's scope)

1. **F06 v2** — refactor to streaming `rolling_sum().over("user_id")` instead of 16 `join_asof`s. Needed to ship F06 at full scale with current memory budget. Until then, treat 0.8266 as a *floor*, not the canonical RFM AUC.
2. ~~**F07 full** — running now (background, orchestrator-owned). Embedding ablation result will resolve whether topic affinity is a production lever or a power-user detector.~~ **RESOLVED: power-user detector. Drop from feature set.**
3. **GOAL.md revision** — pooled 9.87% target is artifically depressed. Anchor on cohort-aware baselines (e.g., per-cohort current-quarter rate). Separate commit.
4. **2026 cohort 4.4% entry** — anomalous vs 2024/2025's ~7-8%. Could be Q1 immaturity, real activation drop, or data recency artifact. Worth ~30 min of follow-up before Wave 3 baseline assumptions harden.
5. **Multi-signature per window** — F00 surfaced that the "petitions per email" framing is collapsed at the attribution layer (`is_first_in_window=True`). If the project goal is genuinely count-of-signatures rather than at-least-one, look at all attributed_actions rows including `is_first_in_window=False`.
6. **8-30d engagement-dip routing investigation** — F05 says the dip is content-allocation. Which petitions are being routed to early-tenure users? Out of scope here but a high-value follow-up for the content-routing project.

---

## Recommended Phase 2 work

In priority order:

1. **Feature family: `prior_action_recency`** — `sends_since_last_action`, `action_in_last_5`, `actioned_last_10`, `is_in_action_streak`, `lifetime_actions`. F08 + F10 prove these dominate. Cheap, F03-compliant, projects past F06 baseline.
2. **Feature family: `raw_engagement_history`** — opens/clicks from raw email_activities, rolling 7d/30d/90d. Replace the dropped user_emails columns. F03-required.
3. **Harness scaffold** — binary target, temporal split by send_date, tenure × prior-action count stratification, send-weighted + user-weighted metrics in parallel. Use `runs/run_01.py` to wire pinned outcome + feature dictionaries.
4. **F06 v2 gating** — before locking the production feature set, F06's refactored full-tier AUC needs to land to anchor the recency-features floor. F07's ablation conclusion is final (embedding centroid drops out).
5. **Discovery harness** — after first champion run, gap-finder identifies residual slices. Likely candidates given the EDA: petitions-routed-to-early-tenure (F05 follow-up), 2026-cohort entry weirdness (F01 follow-up).

---

## Artifacts

- 11 self-contained HTML finding pages: `projects/civic_shout_action_rate_increase/eda/user/findings/F00*.html` through `F10*.html`
- Index page (sortable by wave / severity): `projects/civic_shout_action_rate_increase/eda/user/findings/index.html`
- Reproducible scripts: `projects/civic_shout_action_rate_increase/eda/user/findings_src/F00_outcome_cardinality.py` through `F10_temporal_autocorrelation.py`
- Timing ledger: `projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl` (30+ rows, 3 tiers per finding for findings with full coverage)
- Mockup generator (synthetic shape, for layout review only): `projects/civic_shout_action_rate_increase/eda/user/generate_mockups.py`
- Index refresher: `projects/civic_shout_action_rate_increase/eda/user/refresh_index.py`

---

_Generated 2026-05-12 by the user-level EDA execution. Re-run `refresh_index.py` to regenerate the index. Re-run individual finding scripts via `python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F<NN>_<slug>` (set `EDA_TIERS=full` to skip probe tiers)._
