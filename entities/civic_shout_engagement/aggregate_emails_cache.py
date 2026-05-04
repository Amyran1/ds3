r"""EntityCache config for civic_shout_engagement aggregate_emails.

One row per sent email for Action Network group 228691 (ALL sent
emails — group_id=228691 AND status=5, no date filter). Emails that
predate the ``email_activities`` cache window will have ``total_*=0``
and the validation flags false; their content columns remain accurate.
Combines email content + metadata (pulled from Redshift) with two
parallel engagement counter families:

- "total_*"  aggregated from our cached event streams
               (email_activities_cache + attributed_actions_cache).
- "stats_*"  AN's own counters from the Redshift emails table.

Schema (v1, 19 columns):
    email_id:       i64               -- PK (= Redshift emails.id)
    date_sent:      datetime[us, UTC] -- emails.send_date
    subject:        str               -- emails.subject
    sender:         str               -- emails."from"
    email_body:     str               -- emails.inlined_content (HTML)
    pre_header:     str               -- emails.pre_header

    total_sent:     i64               -- n_unique(user_id) where action_type='sent'
    total_opens:    i64               -- n_unique(user_id) where action_type='open'
    total_vo:       i64 | null        -- n_unique(user_id) where
                                      --   action_type='verified_open'; null =
                                      --   pre-VO-era (stats_vo null) OR in-window
                                      --   join miss (distinguish by total_sent=0).
    total_clicks:   i64               -- n_unique(user_id) where action_type='click'
    total_actions:  i64               -- count per email of is_attributed=True sigs

    stats_sent:     i64               -- emails.total_sent
    stats_opens:    i64 | null        -- emails.stats JSON ['open']
    stats_vo:       i64 | null        -- emails.stats JSON ['verified_open']
    stats_clicks:   i64 | null        -- emails.stats JSON ['click']
    stats_actions:  i64               -- emails.actions_count (lifetime)

    sent_in_range:   bool             -- total_sent <= stats_sent within 25% tolerance
    opens_in_range:  bool             -- total_opens <= stats_opens
    clicks_in_range: bool             -- total_clicks <= stats_clicks

Schema (v2, 23 columns):
    ... [all 19 v1 columns, unchanged] ...

    text_content:    str              -- subject + "\\n\\n" + plain-text body,
                                         truncated to 32K chars. Empty body
                                         falls back to subject-only.
    dense_vector:    list[f32]        -- 1536-element OpenAI
                                         text-embedding-3-small vector,
                                         L2-normalized (matches news_stories_graph).
    tfidf_vector:    struct{           -- sparse sklearn TfidfVectorizer output,
        indices: list[u32],              L2-normalized, ngram_range=(1,2),
        values:  list[f32],              max_features=20000, sublinear_tf=True,
    }                                    stop_words="english".
    sparse_vector:   struct{           -- sparse pinecone BM25Encoder output,
        indices: list[u32],              L2-normalized, k1=1.2, b=0.75.
        values:  list[f32],
    }

    v2 reads v1 and adds the 4 new columns; it does NOT re-pull from Redshift.
    Rebuild order: email_activities v1 → attributed_actions v1 →
    aggregate_emails v1 → aggregate_emails v2.

Schema (v3, 23 columns — same schema as v2):
    ... [all 23 v2 columns] ...

    v3 inherits ALL content and embedding columns from v2 without recompute.
    Only the five counter columns (total_sent, total_opens, total_vo,
    total_clicks, total_actions) and the three range flags are recomputed —
    sourced from email_activities v2 (24-month window) and attributed_actions
    v2 (next-send-boundary, uncapped attribution). This produces 24-month
    engagement counts instead of v1/v2's ~1-month window. The stats_* columns
    pass through unchanged from v2 (still from Redshift emails table).

    Build anchor: BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z.
    Rebuild order: email_activities v2 → actions v2 → attributed_actions v2
    → aggregate_emails v3 (reads v2 for content + embeddings).

Fit artifacts (sidecars at same S3 prefix + cache_dir):
    v2_artifacts/tfidf_vocab.json     -- {"vocab": {token: idx}, "idf": [...],
                                          "config": {...}}
    v2_artifacts/bm25_params.json     -- {"k1": 1.2, "b": 0.75, "n_docs": N,
                                          "n_cols": K, "avg_doc_len": ...,
                                          "doc_freq": {...},
                                          "pinecone_text_version": "...",
                                          "nltk_data_version": "..."}

Upstream dependencies (must be built first):
    v1: civic_shout_engagement__email_activities v1
        civic_shout_engagement__attributed_actions v1
    v2: civic_shout_engagement__aggregate_emails v1
    v3: civic_shout_engagement__aggregate_emails v2 (for content + embeddings)
        civic_shout_engagement__email_activities v2 (for 24-month counters)
        civic_shout_engagement__attributed_actions v2 (for 24-month actions)

Usage::

    from entities.civic_shout_engagement.aggregate_emails_cache import cache

    df = cache.get(1)   # 19 columns
    df = cache.get(2)   # 23 columns (adds text_content, dense_vector,
                        #             tfidf_vector, sparse_vector)
    df = cache.get(3)   # 23 columns — same schema as v2, but total_* counters
                        #             sourced from 24-month upstreams (v2)
    df = cache.get(4)   # 24 columns — v3 + dense_vector_large (OpenAI
                        #             text-embedding-3-large, 3072-d, L2-normalized)

Schema (v4, 24 columns):
    ... [all 23 v3 columns, unchanged] ...

    dense_vector_large: list[f32]      -- 3072-element OpenAI
                                          text-embedding-3-large vector,
                                          L2-normalized. Same source text
                                          (text_content) as v2's dense_vector.

    v4 inherits ALL v3 columns unchanged and appends one new embedding column.
    Built by re-embedding v3.text_content with text-embedding-3-large; no
    Redshift re-pull, no counter recompute. Allows direct A/B of -small vs
    -large in a single read.

    Build order: aggregate_emails v3 → aggregate_emails v4.
"""

from __future__ import annotations

from pathlib import Path

from libs.cache.entity_cache import EntityCache, VersionMeta

_CACHE_DIR = Path("data/entities/civic_shout_engagement/aggregate_emails")
_S3_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities"
    "/civic_shout_engagement/aggregate_emails"
)

cache = EntityCache(
    entity="civic_shout_engagement__aggregate_emails",
    s3_prefix=_S3_PREFIX,
    cache_dir=_CACHE_DIR,
    versions={
        1: VersionMeta(key="aggregate_emails_v1", fmt="parquet"),
        2: VersionMeta(key="aggregate_emails_v2", fmt="parquet"),
        3: VersionMeta(key="aggregate_emails_v3", fmt="parquet"),
        4: VersionMeta(key="aggregate_emails_v4", fmt="parquet"),
    },
)
