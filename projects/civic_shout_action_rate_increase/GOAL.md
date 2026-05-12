# Goal

Our goal is to increase the number of total petitions signed per email sent by ~10%. So if the total petitions signed per email is 10,000, a 1,000 increase is a slam dunk.

Civic Shout sends one email every day to their users who are active in the last 30 days.

## Baseline (24 months, 2024-04-21 → 2026-04-20)

Source: `entities/civic_shout_user_emails` v2 — pins `attributed_actions` v2 (uncapped, 24-month window) + the raw-click → opened imputation.

| Period | Sends | Actioned | **Action rate** | Unsub rate |
|---|---:|---:|---:|---:|
| 2024 (Apr–Dec) | 39,852,143 | 3,161,165 | **7.93%** | 0.362% |
| 2025 (full) | 70,164,233 | 7,179,107 | **10.23%** | 0.375% |
| 2026 (Jan–Apr) | 19,174,164 | 2,413,032 | **12.58%** | 0.400% |
| **Pooled** | **129,190,540** | **12,753,304** | **9.87%** | **0.375%** |

`actioned` = the (user, email) pair was attributed as the most-recent-prior send before that user signed a petition (via `attributed_actions` v2, last-touch attribution, uncapped because Civic Shout's daily cadence makes the cap a no-op for ~99% of sends).

The year-over-year action rate is **monotone-increasing** (7.93% → 10.23% → 12.58%). Whether that's real audience-engagement growth, tracking improvements, or a Civic Shout product change (e.g., signing-flow simplification) is an **open question** — it determines whether the pooled (9.87%) or recent (12.58%) baseline is the right one to chase.

## Target translation

A 10% lift means:

| Anchor | Target | Absolute add |
|---|---|---|
| Pooled 9.87% | 10.86% (+0.99pp) | ~+1.28M attributed signatures over a 24-mo equivalent window |
| 2026 12.58% | 13.84% (+1.26pp) | ~+241K attributed signatures across 2026's Jan–Apr send volume; ~+720K extrapolated to a full year |

## Open data questions affecting the target

1. **Year-over-year drift**: confirm whether 7.93% → 12.58% reflects real growth or tracking changes. If tracking-driven, the real baseline is closer to 12.58% throughout and the 2024–2025 numbers are under-counted.
2. **`exclude_last_send=True` in attributed_actions**: each user's most-recent send is excluded from attribution until their *next* send arrives. For a daily sender, this slightly suppresses the latest-day rate. Worth a one-off check to confirm impact.
3. **The 67% email-attribution rate**: across 24 months, 67% of *all* signatures attribute to a send within the window. That makes email the dominant signature channel and validates the project premise — but the other 33% (organic, shares, off-platform) aren't influenced by changes to email content/targeting.
