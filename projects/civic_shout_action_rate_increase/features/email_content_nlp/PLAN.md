# Feature family: email_content_nlp (v1)

## Why

Source: planner brainstorm `bs-iter2-01-47822d06` (iter 2). `source.finding_id`
is null in the brainstorm item — this proposal is reasoned from the iter 1
champion analysis, not from a specific gap-finder finding.

The iter 1 champion (`run_01_smoke`) uses zero email-side features. The diagnostic
`roc_auc_residualized_email_popularity_pair = 0.811` reveals that email-level
variance carries strong signal that no user-history feature captures. The
champion's primary `roc_auc_residualized_user_prior_x_email_popularity_pair`
is 0.6555 — leaving the email residual axis as the most promising under-fit
direction. ~67% of recipient actions are email-attributed (per project notes),
so email content should carry real signal.

This family extracts cheap, deterministic NLP features from `body_plaintext`
of `civic_shout_emails` v1: length, urgency cues, CTA cues, exclamation /
question density, uppercase-shout density, and a "first-line length" proxy
for subject line. No embedding or LLM calls — pure regex + Polars string ops,
< 1 second to build over the ~1789 unique emails in the corpus.

## Plan deviations from the brainstorm

- **No `subject` field exists** in `civic_shout_emails` v1 (only
  `body_plaintext`, `dense_vector`, `sparse_vector`). The brainstorm's
  "subject-line length" maps to `first_line_char_len` (length of the first
  newline-delimited line of `body_plaintext`).
- **No sentiment polarity** in v1. Sentiment requires a model call (LLM or
  vader) and would blow the autoloop wall budget. v1 ships count-based
  proxies (urgency keyword count, exclamation density); polarity is a
  separate family if v1 shows lift.
- **No embedding-derived features** in v1. The `dense_vector` column exists
  but plugging 1536-dim vectors into LightGBM requires either passing all
  1536 features (blows feature_fraction sampling) or projecting to ~10–32
  components (separate cache, separate plan).

## Source-to-feature steps

Per-email features (grain = `email_id`, ~1789 unique rows):

1. `body_char_len` — `pl.col("body_plaintext").str.len_chars()`
2. `body_word_count` — split on whitespace, count tokens
3. `body_sentence_count` — count of `[.!?]+` sentence terminators (capped)
4. `body_avg_word_len_chars` — `body_char_len / max(body_word_count, 1)`
5. `body_exclamation_count` — count of `!`
6. `body_question_count` — count of `?`
7. `body_uppercase_word_ratio` — ratio of all-caps tokens (len ≥ 3)
8. `body_urgency_keyword_count` — count of regex matches for
   `(?i)\b(urgent|today|now|last chance|deadline|hurry|immediate|ending|expires|don'?t miss|act fast|final)\b`
9. `body_cta_keyword_count` — count of regex matches for
   `(?i)\b(sign|click|donate|act|join|support|vote|contact|share|tell|stand)\b`
10. `first_line_char_len` — length of the first `\n`-delimited line (subject proxy)
11. `first_line_word_count` — word count of the first line

Final cache schema: `email_id` + 11 features. Joined on `email_id` in
`run_02.py`.

## Planned cache access

- Local: `data/projects/civic_shout_action_rate_increase/features/email_content_nlp/v1.parquet`
- S3: best-effort; smoke session may skip S3 upload if creds unavailable
- Required columns: `email_id` + 11 feature names

## EDA visualization plan

- Pearson correlation of each feature against `actioned_24h` (joined on `email_id`)
- Distribution histograms (log-scale where heavy-tailed)
- Email-popularity-residualized AUC per feature (computed inside harness via
  registered residualization)

## Performance / budget expectations

- Build: < 5 seconds over 1789 emails (pure Polars string ops)
- Harness join cost: O(N_rows) left-join on `email_id`, no shuffle
- No per-row inference, no LLM calls, no embeddings

## Similar-pattern review

- `prior_action_recency` v1 builds at (user, email, date_sent) grain via
  Polars rolling-window aggregates. Same Polars-native discipline applies
  here but at coarser grain (`email_id` only).
- This family does NOT shift over time (per-email features are time-invariant
  per email_id), so no leakage audit is needed beyond confirming that the
  source `body_plaintext` is the canonical send-time content.

## Null policy

- Emails missing from `civic_shout_emails` v1 → all 11 features null
- `harness` fills nulls with 0 (per project convention); rare null-content
  emails get the empty-content profile

## Leakage audit

- `body_plaintext` is the immutable send-time content; no causality risk.
- Features are deterministic functions of the email body; no future
  information leaks back.

## Acceptance gate

- `email_content_nlp.cache.get(1)` returns ≥ 1000 rows with all 11 feature
  columns present and non-null `email_id`.
- `run_02` completes with `status == "completed"` and a primary metric value
  in `results.jsonl`.
