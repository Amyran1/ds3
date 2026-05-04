# aggregate_emails v4 — `dense_vector_large` Feature/Entity Plan

## Scope note (read first)

This is **structurally an entity-cache version bump** (`aggregate_emails v3 → v4`),
not a per-(user, email, sent_at) feature family. The plan is authored in the
`/feature-plan` template per user invocation, but the unit of work is one new
column on the `aggregate_emails` entity. Downstream feature families that will
consume `dense_vector_large` (per-channel `_large` clones of
`contrastive_dense_actioned`, `max_sim_actioned_dense_k10`,
`recency_decayed_contrastive_dense_actioned`, etc.) are **out of scope** and get
their own feature plans. This plan exists so the embedding swap can be tested
in isolation before any feature is rebuilt.

## 1. Why Build This

- **Hunch or evidence**: every dense feature in the production champion runs on
  `text-embedding-3-small` (1536-d), pinned at
  `entities/civic_shout_engagement/create_aggregate_emails_v2.py:67`. Across
  `contrastive_dense_*`, `max_sim_actioned_dense_k10`, and
  `recency_decayed_contrastive_dense_actioned` — that's all of `run_12_full`'s
  post-BASE_6 r83 lift. The signal pathway is bottlenecked by one embedding
  model that has never been swapped. `text-embedding-3-large` (3072-d) is
  +3–5pp on MTEB retrieval benchmarks vs `-small`; substituting the backbone
  may lift every dense feature uniformly.
- **Decision this could change**: whether `run_12_full` (r83 +1.818pp) is a
  true ceiling or just an embedding-quality ceiling. If `-large` lifts the
  re-derived champion stack at 10pct, escalate to full re-embed and ship as
  the new champion. If flat, the embedding-quality axis is closed and the
  next move is outcome-side (label-noise / action-window audit).
- **Prior run, EDA, user request, or external data source**: discovery_01
  (`runs/run_10_predictions_for_discovery/feature_gap_backlog.json`) flagged
  leaf_27 (29% of rows, 1.91× residual lift) as dominated by
  `contrastive_*_clicked` + `contrastive_sparse_actioned`. recency_decayed
  closed part of leaf_27 at +0.005pp. The cluster-smoothing post-mortem
  (`memory/project_user_cluster_smoothing_v1_dead.md`) confirmed
  user-constant features cannot move the within-user metric — the only axes
  left are per-(user, email) signal density and outcome quality.
  Embedding-quality upgrade is the cleanest unexplored per-row axis.
- **Success signal before harness evaluation**: pairwise similarity
  distribution on a 200-email holdout sample should be measurably more
  dispersed under `-large` than `-small` (higher variance of cosine over
  random pairs, higher AUC of cosine-on-actioned-vs-cosine-on-random). If
  `-large` and `-small` produce statistically indistinguishable similarity
  distributions on the corpus, harness lift is unlikely.

## 2. Source Data

- **Source artifact or data structure**: `aggregate_emails v3.text_content`
  column — the existing email body + subject text, already HTML-stripped and
  truncated to 32K chars in `create_aggregate_emails_v2.py:_html_to_text`.
- **Current grain**: one row per `email_id` (one per sent email).
- **Expected row count / entity count**: 4,575 unique emails (verified via
  `entities.civic_shout_engagement.aggregate_emails_cache.cache.get(version=3)`).
- **Required source versions or fingerprints**: `aggregate_emails v3` is the
  read source. v4 inherits all 23 columns from v3 unchanged and appends one
  new column.
- **Data access path**:
  ```python
  from entities.civic_shout_engagement.aggregate_emails_cache import cache as aggregate_emails_cache
  v3 = aggregate_emails_cache.get(version=3)
  texts = v3.select(["email_id", "text_content"])
  ```

## 3. Source-To-Feature Steps

1. **Load**: read `aggregate_emails v3` via the entity cache, project
   `(email_id, text_content)`. ~4,575 rows.
2. **Filter/project**: drop rows where `text_content` is null or empty (same
   policy as v2 — the existing dense_vector is already null-handled at v2;
   reuse identical exclusion logic to keep v4 row count = v3 row count).
3. **Join or align**: not applicable; output is keyed 1:1 to v3 by `email_id`.
4. **Transform**: call OpenAI `text-embedding-3-large` (3072-d) on
   `text_content` in batches of 500 with concurrency 25, mirroring v2's
   `_embed_dense` (`create_aggregate_emails_v2.py:252-280`). L2-normalize the
   resulting vectors (same `sklearn.preprocessing.normalize` call as v2).
   Validate `(n_rows, 3072)` shape with `_DenseShapeError` parallel.
5. **Aggregate to harness row grain**: not applicable at this layer. The
   per-email vector is the entity-level output. Per-row alignment is the
   responsibility of downstream feature families.
6. **Validate row keys, nulls, leakage, and schema**:
   - `email_id` is unique and matches v3's row set exactly (no adds, no
     drops).
   - `dense_vector_large` is non-null for ≥99% of rows (gate 2 from v3:
     `_DENSE_VECTOR_MIN_PRESENT = 0.99`). Inherit identical gate.
   - L2-norm check: `|‖v‖₂ − 1.0| < 1e-4` for ≥99.5% of vectors.
   - Leakage: none. Embedding is content-only; no labels touched.
   - Schema: 24 columns total = 23 v3 columns + `dense_vector_large:
     list[f32]`. The existing `dense_vector` (3-small, 1536-d) stays in
     place so consumers of v3 patterns can A/B without forking.

## 4. Feature Identity And Cache Access

- **Feature family name**: `aggregate_emails_v4_dense_large` (entity-bound,
  not a project feature family).
- **Feature dictionary path**: not applicable — entity columns are documented
  in `entities/civic_shout_engagement/aggregate_emails_cache.py` docstring,
  same as v2/v3. Update that docstring to add the v4 schema block.
- **Cache module path**:
  `entities/civic_shout_engagement/aggregate_emails_cache.py` — register
  `version=4` alongside the existing `1, 2, 3` registrations. No new module.
- **Planned local path**:
  `data/entities/civic_shout_engagement/aggregate_emails/v4.parquet`
  (mirrors v3 location convention).
- **Planned remote path**:
  `s3://chorus-content-assets/autonomous-data-scientist/civic_shout_news_environment/entities/civic_shout_engagement/aggregate_emails/v4.parquet`
  (mirrors v3 S3 prefix).
- **Versioning rule**: v4 = v3 schema + `dense_vector_large` (3072-d,
  L2-normalized, OpenAI `text-embedding-3-large`). Future bumps: v5 for a
  different embedding model (e.g. domain-finetuned), v6 for re-embed of a
  re-truncated `text_content`.
- **Harness smoke load example**:
  ```python
  from entities.civic_shout_engagement.aggregate_emails_cache import cache
  df = cache.get(version=4)
  assert "dense_vector_large" in df.columns
  assert df.select(pl.col("dense_vector_large").list.len().eq(3072).all()).item()
  ```

## 5. EDA Visualizations

- **Outcome distribution**: not applicable — entity-level, no outcome.
- **Feature distributions**:
  - histogram of `‖dense_vector_large‖₂` (should be tight at 1.0 ± 1e-4).
  - histogram of cosine(`dense_vector_small_i`, `dense_vector_large_i`) for
    the same email — answers "are the two models embedding into similar
    semantic neighborhoods?" Expect mean ≥ 0.5; bimodal would be a smell.
- **Missingness**: count of null `dense_vector_large` rows; expect 0.
- **Feature-vs-outcome plots**: deferred to downstream feature plans. The
  entity layer cannot show feature-vs-outcome without rebuilding a contrastive
  feature.
- **Correlation or redundancy checks**:
  - pairwise cosine similarity distribution on a random sample of 1000 email
    pairs under `-small` vs `-large`. Plot both histograms on the same axes.
    The hypothesis predicts wider spread (less collapsed) under `-large`.
  - per-pair gap: `cos_large(a,b) − cos_small(a,b)` distribution. Mean near 0
    is expected; tail is the interesting part.
- **Segment/time/entity plots if relevant**: cosine spread by `total_sent`
  bucket — if `-large` only helps for high-volume senders, downstream feature
  plans should weight by sender activity.

## 6. Patterns And Anti-Patterns

- **Similar local patterns reviewed**:
  - `entities/civic_shout_engagement/create_aggregate_emails_v2.py:252-280`
    (`_embed_dense`) — exact pattern to clone with the model string changed.
    Reuses `OpenAIClient.embed`, batch size 500, concurrency 25,
    `cost_key="civic_shout_engagement__aggregate_emails_v4"`.
  - `entities/civic_shout_engagement/create_aggregate_emails_v3.py` — derived-only
    build pattern. v4 follows the same shape: read v3, append one column,
    write v4. No Redshift re-pull.
- **Patterns to reuse**:
  - L2-normalize via `sklearn.preprocessing.normalize` post-batch.
  - `_DenseShapeError` for shape validation.
  - `_DENSE_VECTOR_MIN_PRESENT = 0.99` gate.
  - Cost-key wiring on `OpenAIClient.embed` so the spend lands in
    `CostTracker` and shows up under `civic_shout_engagement__aggregate_emails_v4`.
  - `Container` async-context manager for client lifecycle.
- **Anti-patterns to avoid**:
  - Do **not** rebuild downstream feature families inside this plan. Each
    `_large` sibling gets its own `feature-plan` doc.
  - Do **not** drop `dense_vector` (the 1536-d -small vector). v4 keeps both
    columns so an A/B at 10pct is a single read, not two parquet loads.
  - Do **not** re-truncate `text_content`. The token boundary is part of v2
    and changing it confounds the embedding-model A/B.
  - Do **not** call OpenAI without `cost_key`; untracked spend.
- **Open design risks**:
  1. `text-embedding-3-large` may rate-limit differently than `-small`. v2's
     concurrency=25 worked for `-small`. If `-large` 429s at the same
     concurrency, drop to 10 and re-time. Mitigation: surface the error and
     log throttling rather than retry-storming.
  2. 3072-d adds memory pressure to downstream KNN-on-actioned (per-user
     state buffer doubles). Not a v4-layer concern, but flag it for the
     downstream `_large` feature plans.
  3. Stochastic embedding output: OpenAI embeddings are deterministic per
     model + text, so re-runs should be byte-identical. If they aren't, log a
     warning rather than silently overwrite.

## 7. Runtime Calibration

### Comparable features (cite `timing_performance.jsonl` rows)

There is no entity-level embedding `timing_performance.jsonl` ledger;
embedding cost was captured ad-hoc in v2's docstring (`Cost: ~$0.18 at
4,575 emails × ~2K tokens avg`). The closest measurable analogues are
the per-row contrastive features that consume the embeddings; their
throughput is irrelevant for this plan because the work here is API-bound,
not row-throughput-bound. The relevant calibration is **API tokens-per-second
under bounded concurrency**.

| Stage | Source | Scale | Cost | Throughput |
|---|---|---|---|---|
| `aggregate_emails v2` (3-small embed) | `create_aggregate_emails_v2.py:20` docstring | 4,575 emails | ~$0.18 | embed step took O(minutes) wall (no preserved row) |

### Predicted throughput for this feature

- **Tokens to embed**: 4,575 emails × ~600 tokens (p50 chars / 2.9 ≈ 600
  tokens; p90 ~830) ≈ 2.75M–3.8M tokens.
- **OpenAI cost** (`-large` @ $0.13/M tokens): ~$0.36–$0.50 per full build.
  Within an order of magnitude of v2's $0.18 (-small @ $0.02/M).
- **Wall-clock**: bounded by API latency, not local compute. With the same
  concurrency=25, batch=500 as v2, expect 2–5 minutes for 4,575 emails on
  `-large`. No local CPU bottleneck.
- **No comparable rows/s metric** — this is API-bound. Mark as
  "no comparable; timing-performance sample required" per the skill's
  pattern-review rule. Run a 200-email sample first to verify wall-clock and
  cost.

### Sample-tier gates

| Tier | Emails | Tokens | Expected wall | Cost | Bail-out trigger |
|---|---|---|---|---|---|
| 200-email sample | 200 | ~120K | 10–30s | ~$0.02 | wall > 90s OR 429 errors → drop concurrency |
| Full | 4,575 | ~2.75M | 2–5 min | ~$0.36–$0.50 | wall > 15 min OR cost > $2 → abort |

There is no 200u/2000u tier here — entities are per-email, not per-user. The
"200u" calibration step in the user's standard ladder is replaced by a
200-email API smoke.

### Cost / memory notes

- **API**: ~$0.40 one-time at full build. `cost_key` is mandatory.
- **Storage**: 4,575 × 3072 × 4B = ~56 MB extra in the v4 parquet vs v3
  (which is already ~50 MB compressed). Total v4 size estimate: <150 MB
  parquet. Trivial.
- **Peak memory**: 56 MB additional in-RAM during build. Trivial.

### Re-plan trigger

- If 200-email sample wall-clock > 90s OR rate-limit storms appear, drop
  concurrency to 10, re-sample. If still > 90s, return to plan and consider
  retry/backoff strategy or batch reduction.
- If full-build cost > $2 (vs predicted ~$0.50), abort and inspect the cost
  attribution — likely a token-counting error or repeated embedding of the
  same text.
- **Timing-performance handoff required? YES** for the 200-email API smoke.
  Reason: no preserved timing row for v2's embed step; need a fresh number
  before committing to the full-corpus call. The smoke is also the
  rate-limit / 429-storm gate.

## 8. Handoff

- **Next skill**: `entities` (the entity-cache equivalent of `feature`) for
  authoring `create_aggregate_emails_v4.py` and registering `version=4` in
  `aggregate_emails_cache.py`. **Not** `feature` — this plan does not produce
  a per-row feature family. After v4 ships, three downstream plans
  (`contrastive_dense_actioned_large`, `max_sim_actioned_dense_k10_large`,
  `recency_decayed_contrastive_dense_actioned_large`) will each invoke
  `/feature-plan` separately and consume v4.
- **Cache skill needed?** Implicitly yes — the entity cache version
  registration is part of `entities`/`cache` work. Same artifact (`aggregate_emails_cache.py`).
- **Timing-performance skill needed?** YES — 200-email API smoke before full
  build. Reason: API-bound, no preserved comparable, 429-risk on `-large`.
- **Files expected from implementation**:
  - `entities/civic_shout_engagement/create_aggregate_emails_v4.py` (build
    script — derived-only; reads v3, appends `dense_vector_large`, writes v4)
  - `entities/civic_shout_engagement/aggregate_emails_cache.py` (modified
    in-place — append v4 to the version registry; update the docstring v4
    schema block)
  - `entities/civic_shout_engagement/aggregate_emails_v4_eda.html` (cosine
    distribution + per-pair gap diagnostics on the 200-email sample, then
    repeated on full)
  - `entities/civic_shout_engagement/aggregate_emails_v4_timing_performance.jsonl`
    (1 sample row + 1 full row)

### Downstream verification (out of scope for this plan, sketched here for context)

After v4 ships, the binary go/no-go test is:

1. Clone three feature families (`contrastive_dense_actioned`,
   `max_sim_actioned_dense_k10`, `recency_decayed_contrastive_dense_actioned`)
   into `_large` siblings. Each is a thin diff: read v4 instead of v3, point
   at `dense_vector_large` instead of `dense_vector`, output a renamed
   column.
2. Run `run_NN_dense_large_10pct.py`: BASE_5 (5 sparse) + the 3 `_large`
   dense features. Compare r83 to `run_12` (10pct +1.928pp) and `run_12_full`
   (full +1.818pp).
3. **Decision rule**:
   - r83 lift ≥ +0.0010 with CI not crossing run_12 → escalate to full
     re-embed nothing additional needed (v4 already has the column for all
     4,575 emails) and a full feature rebuild + champion-promotion run.
   - Flat (within ±0.0005) → stop. Embedding-quality axis closed; pivot to
     outcome-side audit.
   - Regression → stop and document. `-large` adds noise for sparse-history
     users at this metric.

This downstream block is intentionally a sketch; each step gets its own
`feature-plan` and `run` invocation.

## Risks

1. **Embedding-quality lift may not move the within-user rank metric.**
   `spread_q4_minus_random` ranks within-user; absolute cosine improvement
   may not cross GBDT split boundaries. Pre-empt with the cosine-distribution
   diagnostic in §5 — if `-large` and `-small` produce indistinguishable
   pairwise distributions, harness lift is unlikely and downstream feature
   rebuilds become low-ROI.
2. **MTEB benchmark gain may not transfer to civic_shout email content.**
   Emails are short, US-political, and templated. Out-of-distribution
   relative to MTEB. Mitigation: cosine-on-actioned-vs-cosine-on-random AUC
   on a 200-email sample, before full corpus embed. If AUC under `-large`
   is not measurably higher than under `-small`, abort the swap.
3. **3072-d doubles per-user state in downstream KNN features.** Not a v4
   risk per se, but a known cost any `_large` feature plan inherits. Memory
   and walk-time go up roughly 2× in `knn_max_similarity_dense_actioned_v1`
   when ported to `-large`.
4. **OpenAI rate-limit asymmetry between `-small` and `-large`.** Mitigation
   = mandatory 200-email smoke before full-corpus call; drop concurrency to
   10 and retry if 429s appear.
5. **Cost attribution drift.** v4 must use a distinct `cost_key`
   (`civic_shout_engagement__aggregate_emails_v4`) so `CostTracker` can
   separate the v4 spend from any later re-runs.

## Critical Files

- **New (create)**:
  - `entities/civic_shout_engagement/create_aggregate_emails_v4.py`
  - `entities/civic_shout_engagement/aggregate_emails_v4_eda.html`
  - `entities/civic_shout_engagement/aggregate_emails_v4_timing_performance.jsonl`
- **Modify**:
  - `entities/civic_shout_engagement/aggregate_emails_cache.py` (register
    `version=4` + update schema docstring; do not break existing v1/v2/v3
    consumers).
- **Read for patterns**:
  - `entities/civic_shout_engagement/create_aggregate_emails_v2.py:67-280`
    (model constant, `_embed_dense`, batch/concurrency settings).
  - `entities/civic_shout_engagement/create_aggregate_emails_v3.py` (derived-only
    build shape).
  - `libs/clients/openai.py` (the `embed` method's signature + cost_key).
- **Out of scope (separate plans)**:
  - All `projects_v2/civic_shout_content_routing/features/*_large` feature
    family plans.
  - The harness verification run script.
