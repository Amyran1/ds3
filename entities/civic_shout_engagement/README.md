# civic_shout_engagement

Four sub-caches of Action Network group 228691 email engagement and petition
signatures. Two are sourced from the Redshift mirror `group_228691_indexed`
(`email_activities`, `actions`); the third (`attributed_actions`) is derived
at build time from the other two via polars `join_asof`; the fourth
(`aggregate_emails`) joins Redshift email content with aggregated engagement
counters from the first two caches. Each sub-cache is a standalone versioned
entity — consumers pin the version they need.

Grain: one row per email activity event (`email_activities`), one row per
petition signature (`actions`), one row per petition signature with attribution
metadata (`attributed_actions`), or one row per sent email (`aggregate_emails`).

## Source

- **Raw inputs.** Redshift mirror at
  `can2-mirrors.cn0axdrlik4q.us-east-1.redshift.amazonaws.com`, database
  `mirror`, schema `group_228691_indexed`. Tables: `email_activities_16`
  (activity events), `signatures` + `petitions` (actions), and `emails`
  (email metadata + stats).
- **Fetch window.**
  - `email_activities`: last 1 month **+ 14-day pre-window buffer** (2× the
    7-day attribution cap) so day-1 signatures can still find their prior
    send outside the nominal month.
  - `actions`: last 1 month exactly.
  - `aggregate_emails`: **ALL sent emails** (`status = 5`, no date filter).
    Emails that predate the `email_activities` window will have `total_*=0`
    and their validation flags false; their content columns remain accurate.
- **Volume (v1 estimates).** `email_activities`: ~10⁷–10⁸ rows (~1–5 GB
  parquet). `actions`: ~10⁵–10⁶ rows (<100 MB parquet). `aggregate_emails`:
  ~1–5 K rows (one row per unique sent email in the group's full history).
- **Refresh cadence.** Ad-hoc / on-demand. Rebuild in order:
  `create_email_activities_v1.py` → `create_actions_v1.py` →
  `create_cache_v1.py` (attributed_actions) → `create_aggregate_emails_v1.py`.
  No scheduled refresh in v1.

## Sub-caches

| Cache module | Entity name | Version | Row grain | Upstream deps | Key | Local cache dir |
|---|---|---|---|---|---|---|
| `email_activities_cache.py` | `civic_shout_engagement__email_activities` | v1 | One row per email activity event | Redshift `email_activities_16` | `email_activities_v1` | `data/entities/civic_shout_engagement/email_activities/` |
| `actions_cache.py` | `civic_shout_engagement__actions` | v1 | One row per petition signature | Redshift `signatures` + `petitions` | `actions_v1` | `data/entities/civic_shout_engagement/actions/` |
| `attributed_actions/cache.py` | `civic_shout_engagement__attributed_actions` | v1 | One row per petition signature (with attribution) | `email_activities_cache` + `actions_cache` | `attributed_actions_v1` | `data/entities/civic_shout_engagement/attributed_actions/` |
| `aggregate_emails_cache.py` | `civic_shout_engagement__aggregate_emails` | v1, v2 | One row per sent email | `email_activities_cache` + `attributed_actions_cache` + Redshift `emails` | `aggregate_emails_v1` / `aggregate_emails_v2` | `data/entities/civic_shout_engagement/aggregate_emails/` |

S3 prefix root: `s3://chorus-content-assets/autonomous-data-scientist/civic_shout_news_environment/entities/civic_shout_engagement/`

## Versions

- **v1** — initial build (2026-04-20). `email_activities`: columns
  `activity_id`, `email_id`, `user_id`, `action_type` (Categorical), `created_at`
  (datetime[us, UTC]). `sent` rows retained for recipient universe. `actions`:
  columns `signature_id`, `user_id`, `petition_id`, `created_at` (datetime[us, UTC]).
  `aggregate_emails`: 19 columns — see schema subsection below.

- **v2** — adds `text_content`, `dense_vector` (OpenAI text-embedding-3-small,
  1536 dims), `tfidf_vector` (sklearn TfidfVectorizer, 20K-feature), and
  `sparse_vector` (pinecone BM25Encoder) columns to `aggregate_emails`.
  Reads v1 and extends it; rebuild v1 first if underlying data changed.
  Fit artifacts (TFIDF vocab + IDF, BM25 params) live in `v2_artifacts/`
  under the same S3 prefix.

## Schema (v1)

### email_activities

| Column | Type | Meaning |
|---|---|---|
| `activity_id` | `i64` | PK from `email_activities_16.id`. |
| `email_id` | `i64` | The email that triggered this event. |
| `user_id` | `i64` | Recipient (renamed from `recipient_id`). |
| `action_type` | `Categorical` | Event type: `sent`, `open`, `click`, `verified_open`, plus any other values present in the source. See column-name note below. |
| `created_at` | `datetime[us, UTC]` | When the event occurred. |

**Column-name note:** `verified_open` is the raw `action_type` enum value.
In the wide-join shape produced by `attributed_actions/README.md`, the
corresponding boolean aggregate column is named `verified_opened` (past
participle, boolean). Do not mix the two names.

### actions

| Column | Type | Meaning |
|---|---|---|
| `signature_id` | `i64` | PK from `signatures.id`. |
| `user_id` | `i64` | Signer. |
| `petition_id` | `i64` | Which petition was signed. |
| `created_at` | `datetime[us, UTC]` | When the signature was submitted. |

### aggregate_emails

**Coverage:** ALL sent emails for group 228691 (`status = 5`, no date filter).
Emails that predate the `email_activities` cache window (typically last
1 month + 14 days) will have `total_*=0` and their validation flags false,
because the activities cache does not cover those older sends; their content
columns (`subject`, `email_body`, etc.) remain accurate.

One row per sent email (Redshift `emails.status = 5`). Combines Redshift
content + AN's own counters ("stats_*") with deduplicated unique-user counters
from the activity caches ("total_*").

| Column | Type | Meaning |
|---|---|---|
| `email_id` | `i64` | PK — Redshift `emails.id`. |
| `date_sent` | `datetime[us, UTC]` | `emails.send_date` — when the email was dispatched. |
| `subject` | `str` | Email subject line. |
| `sender` | `str` | From address (`emails."from"`). |
| `email_body` | `str` | Full inlined HTML content (`emails.inlined_content`). |
| `pre_header` | `str` | Email pre-header text. |
| `total_sent` | `i64` | Deduplicated n_unique(user_id) for `action_type='sent'` in `email_activities`. ≤ `stats_sent` (AN may count re-sends; we deduplicate). 0 for emails outside the activities window. |
| `total_opens` | `i64` | Deduplicated n_unique(user_id) for `action_type='open'`. 0 for emails outside the activities window. |
| `total_vo` | `i64 \| null` | Deduplicated n_unique(user_id) for `action_type='verified_open'`. Null when `stats_vo` is null (pre-VO-era email — feature not tracked by AN). Also null if the email_id is absent from the activities cache (distinguish by total_sent == 0). |
| `total_clicks` | `i64` | Deduplicated n_unique(user_id) for `action_type='click'`. 0 for emails outside the activities window. |
| `total_actions` | `i64` | Count of `is_attributed=True` rows in `attributed_actions` for this email. Represents petition signatures attributed within 7 days of this send. |
| `stats_sent` | `i64` | AN's own total send count from `emails.total_sent`. May exceed `total_sent` due to re-sends to the same user. |
| `stats_opens` | `i64 \| null` | AN's open count from `emails.stats['open']`. Null if the JSON field is absent. |
| `stats_vo` | `i64 \| null` | AN's verified-open count from `emails.stats['verified_open']`. Null for pre-VO-era emails. |
| `stats_clicks` | `i64 \| null` | AN's click count from `emails.stats['click']`. Null if the JSON field is absent. |
| `stats_actions` | `i64` | AN's lifetime action count from `emails.actions_count`. Includes attributions outside the 7-day window. |
| `sent_in_range` | `bool` | True when `total_sent ≤ stats_sent` and the gap is within 25% of `stats_sent`. False if `stats_sent` is null or zero. False for emails outside the activities window (total_sent=0). |
| `opens_in_range` | `bool` | True when `total_opens ≤ stats_opens` (or both zero). False if `stats_opens` is null. |
| `clicks_in_range` | `bool` | True when `total_clicks ≤ stats_clicks` (or both zero). False if `stats_clicks` is null. |

**VO zero-to-null rule.** `total_vo` is set to null (not 0) for emails where
`stats_vo` is null, because a null `stats_vo` indicates AN never tracked
verified-opens for that email.  Where AN does report a `stats_vo` value,
`total_vo` retains its computed count (0 is a valid "tracked but zero VOs"
outcome).

**Attribution window note.** `total_actions` counts attributed signatures up
to the build date.  Emails within the most-recent 7 days may have counts
below their eventual final value — the attribution window for those sends is
still open.

### aggregate_emails — v2 additions

Four columns added on top of the 19 v1 columns. Produced by
`create_aggregate_emails_v2.py`; stored in `aggregate_emails_v2.parquet`.

| Column | Type | Meaning |
|---|---|---|
| `text_content` | `str` | `subject + "\n\n" + plain-text body` (HTML-stripped via BeautifulSoup), truncated to 32K chars. Empty body falls back to subject-only. Stored for debugging and reuse. |
| `dense_vector` | `list[f32]` | 1536-element OpenAI `text-embedding-3-small` vector, L2-normalized. |
| `tfidf_vector` | `struct{indices: list[u32], values: list[f32]}` | Sparse sklearn `TfidfVectorizer` output, L2-normalized. Config: `max_features=20_000`, `ngram_range=(1, 2)`, `min_df=3`, `max_df=0.5`, `sublinear_tf=True`, `stop_words="english"`. Vocab + IDF weights in `v2_artifacts/tfidf_vocab.json`. |
| `sparse_vector` | `struct{indices: list[u32], values: list[f32]}` | Sparse pinecone `BM25Encoder` output, L2-normalized. Config: `k1=1.2`, `b=0.75`. Encoder params in `v2_artifacts/bm25_params.json`. |

Fit artifacts (uploaded to S3 alongside the parquet):

| File | Contents |
|---|---|
| `v2_artifacts/tfidf_vocab.json` | `{"vocab": {token: idx}, "idf": [...], "config": {...}}` |
| `v2_artifacts/bm25_params.json` | `{"k1": 1.2, "b": 0.75, "n_docs": N, "n_cols": K, "avg_doc_len": ..., "doc_freq": {...}, "pinecone_text_version": "...", "nltk_data_version": "..."}` |

### Version history

- v1 (2026-04-21): one row per sent email, content + dual engagement counters.
  19 columns.
- v2: v1 + `text_content`, `dense_vector` (OpenAI 1536-dim),
  `tfidf_vector` (sklearn TfidfVectorizer, 20K-feature), `sparse_vector`
  (pinecone BM25Encoder). v2 reads v1; rebuild v1 first if underlying
  data changed. Fit artifacts (TFIDF vocab, BM25 params) live alongside
  the parquet under `v2_artifacts/`.

## Versions (v2 / v3)

### Sub-cache version table (v2 / v3 additions)

| Cache module | Version | Key | Built rows (2026-04-21) | Upstream deps | Notes |
|---|---|---|---|---|---|
| `email_activities_cache.py` | v2 | `email_activities_v2` | **208,422,830** | Redshift `email_activities_16` (24-month window) | Same schema as v1; window extended from ~1.5mo to 24 months. `BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z`. 2.0 GB parquet. |
| `actions_cache.py` | v2 | `actions_v2` | **18,935,733** | Redshift `signatures` + `petitions` (24-month window) | Same schema as v1; window extended to 24 months. 7 null-user_id rows dropped with warning (anonymous signatures). |
| `attributed_actions/cache.py` | v2 | `attributed_actions_v2` | **18,935,733** | `email_activities_cache v2` + `actions_cache v2` | 1:1 with `actions_v2`. 12,753,304 attributed (67.4%), 5,487,036 cascade (29.0%), 695,393 organic. Attribution rule changed (see below). Adds `is_bot_lag` column. 438 MB parquet. |
| `aggregate_emails_cache.py` | v3 | `aggregate_emails_v3` | **4,575** | `aggregate_emails_cache v2` (content + embeddings) + `email_activities_cache v2` + `attributed_actions_cache v2` | Same 23-column schema as v2. Only `total_*` counters and range flags recomputed. 100% dense_vector carryover. 29 MB parquet. |

### attributed_actions v2 — attribution semantics change

**v1 rule (deprecated):** 7-day absolute window cap. A user's most recent
send per session was excluded from attribution (`exclude_last_send=True`).
Signatures falling outside the 7-day cap were left unattributed.

**v2 rule:** Next-send-boundary attribution with no cap.

- Every send participates in attribution.
- `window_seconds = next_send_time - send_time` (full inter-send interval).
- A user's most recent send (no later send in the 24-month window) uses
  `BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z` as `next_send_time`. This
  is the reproducibility anchor — the build is immutable regardless of
  when it is re-run.
- Strict `lag_seconds < window_seconds` boundary preserved: a signature
  timestamped at exactly the window boundary goes to the NEXT send's window.

### attributed_actions v2 — `is_bot_lag` column

| Column | Type | Meaning |
|---|---|---|
| `is_bot_lag` | `bool` | `True` when `lag_seconds < 10`. Flags signatures that occurred within 10 seconds of the send — almost certainly a bot or pipeline artifact. `False` for unattributed rows (null lag_seconds). |

Stage-2 EDA on v1 found ~9% of attributed signatures have `lag_seconds < 10`.
Under v2's next-send-boundary attribution, the observed rate at build time
was **2.14%** (272,647 of 12,753,304 attributed rows). The lower rate is
a consequence of the changed window definition: v1's 7-day cap concentrated
attributions near send-time where bots cluster; v2's longer inter-send
windows dilute the denominator without adding bots, since bots only fire
right at send-time.

`is_bot_lag=True` rows are **retained** in v2; Stage-3 investigations decide
the filter/reweight policy. The validation gate is **informational only**
in v2 — the observed rate is logged, not gated. The three hard gates
(row-count 1:1 invariant, required-columns null check, attributed-rows
attribution-columns null check) remain blocking.

### aggregate_emails v3 — counter recomputation

v3 has the **same 23-column schema as v2**. The only difference is the source
of the `total_*` counters:

| Column | v2 source | v3 source |
|---|---|---|
| `total_sent` | `email_activities v1` (~1.5mo) | `email_activities v2` (24mo) |
| `total_opens` | `email_activities v1` (~1.5mo) | `email_activities v2` (24mo) |
| `total_vo` | `email_activities v1` (~1.5mo) | `email_activities v2` (24mo) |
| `total_clicks` | `email_activities v1` (~1.5mo) | `email_activities v2` (24mo) |
| `total_actions` | `attributed_actions v1` (7-day cap) | `attributed_actions v2` (uncapped, next-send-boundary) |
| `sent_in_range`, `opens_in_range`, `clicks_in_range` | computed from v2 totals | recomputed from v3 totals |
| Content columns (`subject`, `email_body`, etc.) | from Redshift | inherited from v2 (no re-pull) |
| Embedding columns (`dense_vector`, `tfidf_vector`, `sparse_vector`, `text_content`) | computed by `create_aggregate_emails_v2.py` | **inherited from v2 unchanged** |

`stats_*` columns pass through unchanged in both v2 and v3 — they reflect
AN's Redshift counters which are window-independent.

### Build order (v2 / v3)

```
create_email_activities_v2.py          # 24-month Redshift pull
create_actions_v2.py                   # 24-month Redshift pull
attributed_actions/create_cache_v2.py  # polars join_asof
create_aggregate_emails_v3.py          # polars aggregation, no Redshift
```

Measured wall-clock for the 2026-04-21 build:

| Script | Elapsed | Notes |
|---|---|---|
| `create_email_activities_v2.py` | 29.3 min | 24 monthly Redshift chunks; calibration projected 35 min |
| `create_actions_v2.py` | 2.0 min | single 24mo query; calibration projected 18 min |
| `attributed_actions/create_cache_v2.py` | 1.4 min | polars `join_asof` + tiebreak |
| `create_aggregate_emails_v3.py` | 5 sec | polars `group_by` over v2 upstreams |

All four scripts are anchored to `BUILD_CUTOFF_UTC = datetime(2026, 4, 21, tzinfo=UTC)`.

## Usage

```python
from entities.civic_shout_engagement.email_activities_cache import cache as acts
from entities.civic_shout_engagement.actions_cache import cache as actions
from entities.civic_shout_engagement.aggregate_emails_cache import cache as agg_emails
from entities.civic_shout_engagement.attributed_actions.cache import cache as attrib

activities_df = acts.get(1)       # pin version explicitly
actions_df = actions.get(1)
emails_df = agg_emails.get(1)     # 19 columns (v1)
emails_v2_df = agg_emails.get(2)  # 23 columns (v2, includes vector columns)

# v2 / v3 upstreams (24-month window)
activities_v2_df = acts.get(2)    # 24-month email_activities
actions_v2_df = actions.get(2)    # 24-month signatures
attrib_v2_df = attrib.get(2)      # next-send-boundary attribution + is_bot_lag
emails_v3_df = agg_emails.get(3)  # 23 columns, 24-month counters
```

Pin a specific version. Never rely on "latest" — a rebuild may produce a
different row count if the fetch window shifts.

## Consumed by

- `entities/civic_shout_engagement/attributed_actions` — derived sub-cache
  that reads `email_activities` + `actions` to build the time-based
  attribution ledger.

Append a bullet when a new project (outside this umbrella) starts consuming
any of these caches.

## Ownership

- **Author.** aaron@chorusai.co (initial build 2026-04-20).
- **Cross-project changes.** Any version bump driven by a second project's
  needs should be raised with the authoring owner before merging.
