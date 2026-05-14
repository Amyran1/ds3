# PLAN: email_user_semantic_affinity

Feature family for autoloop iter 3, brainstorm `bs-iter3-01-91b117d1`.

## Why

The champion (run_01, `prior_action_recency` only) scores
`roc_auc_residualized_user_prior_x_email_popularity_pair = 0.6755` pooled but
only `roc_auc_residualized_non_streak_users_pair = 0.6442` on non-streak users.
The recency cluster carries pooled lift because it captures behavioural
auto-correlation (a user who acts often keeps acting), but it has no semantic
signal about whether *this specific email* matches what *this specific user*
has historically engaged with.

Run_02 added `email_content_nlp` (length / urgency / CTA / shout-ratio style
features at the email level). Those are user-invariant — every user sees the
same NLP score for a given email — so they cannot explain user-specific
non-streak variance either.

This family closes that gap by encoding **dense semantic similarity** between
each email and each user's prior-actioned-email history. The civic-shout email
entity already carries `dense_vector` (1536-d float32) embeddings pulled from
the production Pinecone index. We compute, per (user, email, date_sent) row,
the cosine similarity between the current email's vector and the centroid of
that user's previously actioned emails (strictly prior, sorted by `date_sent`).

Brainstorm provenance: `source.kind = planner`, `source.ref =
iter_3_champion_analysis`, `finding_id = null` (no specific
`FeatureGapFinding.id` cited — the planner derived the gap from champion
diagnostics directly).

## Cache layout

- `row_keys`: `[user_id, email_id]`
- `date_column`: `date_sent`
- `feature_cols`:
  - `semantic_affinity_to_prior_actions` (`f32`) — cosine similarity of the
    current email's dense_vector to the running sum of prior actioned email
    vectors for the user. NaN encoded as 0.0 in the cache (harness fills nulls
    with 0 anyway; explicit zero keeps the parquet schema clean).
  - `n_prior_actions_for_affinity` (`u32`) — how many prior actioned emails
    fed the centroid for this row. 0 means the affinity column is meaningless
    for this row (cold-start user or pre-first-action send).

## Cache scope (smoke-tier build)

The full v3 user_emails grain is ~129M rows. A full causal build is feasible
(~5–10 min via per-user numpy loop) but the autoloop session has a 900s wall.
For iter 3 smoke we build a *sample-tier* cache that covers exactly the rows
the smoke run will evaluate:

1. Load full v3 user_emails (~129M rows, ~14M users).
2. Filter `date_sent < max(date_sent) - 24h` (mirrors run_02 final-day cut).
3. `sample(fraction=0.001, seed=42)` → ~129k rows, ~100k users (matches
   run_02's effective scope).
4. From the full unfiltered user_emails, collect the prior history of those
   ~100k users (sorted by `(user_id, date_sent)`).
5. Run the per-user numpy loop on that ~900k-row subset.
6. Filter back to the 129k sampled (user_id, email_id, date_sent) tuples.

Run_03 mirrors steps 1–3 verbatim and joins on `(user_id, email_id, date_sent)`,
so the cache keys align by construction. A future full-tier rebuild (champion
candidate / comparison scope) replaces only the `create_cache_v1.py` sampling
to operate over all 129M rows.

## Source-to-feature steps

For each user (sorted by `date_sent` ascending), maintain:

- `running_sum` (1536-d float32): sum of dense_vectors for prior rows where
  `actioned_24h == True`.
- `n_prior` (int): count of prior actioned rows.

For each row in user history:

1. Compute `affinity = dot(current_vec, running_sum) / (||current_vec|| *
   ||running_sum||)` iff `n_prior > 0`, else `affinity = 0.0`.
2. Record `(user_id, email_id, date_sent, affinity, n_prior)`.
3. If `actioned_24h`, `running_sum += current_vec`; `n_prior += 1`.

Strict causality: the affinity for row `i` uses only rows with `date_sent <
date_sent_i`, identical to `prior_action_recency`'s `.shift(1)` discipline.

## EDA plan

Out of scope for the smoke build (autoloop budget). The numeric harness
diagnostics (`roc_auc_residualized_non_streak_users_pair`,
`roc_auc_residualized_cold_start_pair`) are the primary read.

## Performance / budget

Per [`feedback_ds_smoke_test_timing`](.../memory) discipline: cache build at
~0.001-tier is the smoke-first baseline gate. Comparable benchmarks:

- `prior_action_recency` full (129M rows, scalar cumulative ops): 47.31s.
- `email_content_nlp` full (4.6k rows, per-email string ops): 0.24s.

This family does 1536-dim vector ops per row but only across ~900k history rows
(not 129M). Projected: 30–120s. >2× comparable would bail to
`bailed_timing_blowup`; my projection is well under that.

## Similar-pattern review

- `prior_action_recency` — same `(user_id, email_id, date_sent)` row grain,
  same `.over("user_id")` + strict-causality discipline. Reuses the same cache
  schema convention.
- `email_content_nlp` — also targets the email-popularity-residual gap, but at
  user-invariant grain. Combining both with `prior_action_recency` is the
  champion-stack hypothesis for run_03.

## Leakage audit

- Embeddings (`dense_vector`) are deterministic functions of `body_plaintext`
  at send time. No outcome data in their construction.
- Running sum is over rows with `date_sent < current.date_sent` strictly.
- `actioned_24h` is the same column used by `prior_action_recency`; using it
  to define the centroid mirrors the recency lookback and is safe under the
  same causal window.

## Null policy

- `n_prior_actions_for_affinity == 0` → `semantic_affinity_to_prior_actions =
  0.0` (cold-start sentinel). Harness fills any residual nulls with 0.
