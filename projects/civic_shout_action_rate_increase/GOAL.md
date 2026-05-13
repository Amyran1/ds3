# Goal

Our goal is to increase the number of total petitions signed per email sent by ~10%. So if the total petitions signed per email is 10,000, a 1,000 increase is a slam dunk.

Civic Shout sends one email every day to their users who are active in the last 30 days.

## Baseline (v3 cache — 24 h attribution window)

Source: `entities/civic_shout_user_emails` v3 — pins `actioned_24h` (24-hour attribution window). See memory note `~/.claude/projects/-Users-aaronmyran-dev-ds3/memory/project_civic_shout_24h_attribution.md`.

| Metric | Value |
|---|---|
| Pooled action rate | **9.26%** |
| Attribution window | 24 hours |
| Cache version | v3 (`actioned_24h`) |

The year-over-year action rate is **monotone-increasing** across 2024–2026 cohorts. Whether that reflects real audience-engagement growth, tracking improvements, or a Civic Shout product change is an **open question** — it determines whether the pooled or recent baseline is the right one to chase.

## Target translation

A 10% relative lift on the v3 baseline:

| Anchor | Target | Absolute add |
|---|---|---|
| Pooled 9.26% | **10.19%** (+0.93pp) | ~10% more attributed signatures over the evaluation window |

### AUC scale target

The current 5% ship-run (smoke scope) reports a residualized AUC of **0.6755**. The harness primary-metric target is a **10% relative lift**: **0.6755 × 1.10 = 0.7430**.

## Open data questions affecting the target

1. **Year-over-year drift**: confirm whether action-rate growth across 2024–2026 cohorts reflects real growth or tracking changes. If tracking-driven, the recent cohort rate is a better anchor than the pooled 9.26%.
2. **`exclude_last_send=True` in attributed_actions**: each user's most-recent send is excluded from attribution until their *next* send arrives. For a daily sender, this slightly suppresses the latest-day rate. Worth a one-off check to confirm impact.
3. **The 67% email-attribution rate**: across 24 months, 67% of *all* signatures attribute to a send within the window. That makes email the dominant signature channel and validates the project premise — but the other 33% (organic, shares, off-platform) aren't influenced by changes to email content/targeting.
