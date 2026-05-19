# Entities

Cross-project cached datasets. Each entity is a versioned, local-first-with-S3-fallback artifact
that any DS project loads via `cache.get(version)`. Entities live at the repo root (not inside a
project) because the same cleaned dataset is usually reused across multiple projects.

S3 URIs below use **`${AWS_S3_BUCKET}`** (default for this workspace: `chorus-ds-artifacts`, set
in `.env`). Key prefixes are unchanged from the shared `chorus-content-assets` layout; only the
bucket hostname differs.

See `modules/ds/skills/entities/SKILL.md` in the overlay for the scaffold + graduation workflow,
and `modules/core/rules/artifact-filing.md` "Cross-project Entities" for the filing convention.

## Catalog

| Entity | Version | What it is | S3 prefix | Local cache dir |
|---|---|---|---|---|
| `news_sources` | v1 | Perigon-derived news source reference table (domain metadata). | `s3://${AWS_S3_BUCKET}/data-science/extractors/news_sources/` | `data/entities/news_sources/` |
| `news_stories` | v1 | Full Civic Shout news story corpus from prod Mongo `documents.news_stories` joined against `news_sources`. Keyed by `story_id`. | `s3://${AWS_S3_BUCKET}/autonomous-data-scientist/civic_shout_news_environment/entities/news_stories/` | `data/entities/news_stories/` |
| `news_stories_graph` | v1 | Three-layer similarity graph over `news_stories`: bipartite attribute edges, dense 1536-d embeddings + FAISS IVF index + cosine edges (tau=0.50), sparse TF-IDF (50K vocab) + cosine edges (tau=0.25), BM25 vectors + edges (tau=0.30). | `s3://${AWS_S3_BUCKET}/autonomous-data-scientist/civic_shout_news_environment/entities/news_stories_graph/` | `data/entities/news_stories_graph/` |
| `civic_shout_engagement` (4 sub-caches) | v1 | Action Network group 228691 engagement. Sub-caches: `email_activities` (send/open/click/verified_open events, 1 month +14-day buffer), `actions` (petition signatures, 1 month), `attributed_actions` (derived time-based attribution ledger, 7-day cap, last-send excluded), `aggregate_emails` (one row per sent email: Redshift content + dual engagement counter families). | `s3://${AWS_S3_BUCKET}/autonomous-data-scientist/civic_shout_news_environment/entities/civic_shout_engagement/` | `data/entities/civic_shout_engagement/` |
| `scout_emails` | v1, v2 | Scout email performance + AI features (25 cols) for all Chorus client orgs. v1 original extraction; v2 adds backfilled HTML. Filter by `client_organization_id` (e.g., 447=PETA, 442=FA). | `s3://${AWS_S3_BUCKET}/data-science/extractors/emails/` | `data/entities/scout_emails/` |

## Consumers

- Civic Shout news environment work, pre-migration 2026-04-20 (source project: `autonomous-data-scientist/projects/civic_shout_news_environment`).

## Notes

- `news_sources` reuses the shared Perigon parquet from `dev/backend/datascience`; it is not
  rebuilt from `entities/news_sources/create_cache_v1.py` without coordination.
- `news_stories_graph` S3 payload is ~47 GB (see `news_stories_graph/manifest_v1.json`; manifest
  still records the original `chorus-content-assets` build). Upload to the configured
  `AWS_S3_BUCKET` when rebuilding; the bucket is empty until you upload or rebuild locally.
- The `news_stories_graph` build scripts and validators reference a consumer API library at
  `projects.civic_shout_news_environment.lib.news_stories_graph_api` that has not been migrated
  into this repo. Build scripts (`create_*_v1.py`) will run; validators
  (`validate_consumer_api.py`, `validate_full_corpus_metrics.py`) will not resolve that import
  until the library is ported or replaced.
- `scout_emails` S3 keys retain legacy filenames (`emails_all.parquet`, `emails_all_v2.parquet`);
  the Python module and local cache dir use the `scout_emails` slug. The asymmetry is intentional —
  no re-upload required.
