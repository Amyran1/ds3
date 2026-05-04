# scout_emails

Scout project email performance data with AI-generated features, cached from S3.

## Overview

`scout_emails` contains one row per email sent by a Chorus client organization,
covering all Scout/Chorus client orgs in a single unified parquet. Columns include
email metadata (id, date_sent, subject_line), AI-generated topic summaries and
rubric scores, dense and sparse embedding vectors, tone analysis, and structural
features. Filter downstream by `client_organization_id` to scope to a specific org.

## Usage

```python
from entities.scout_emails import cache

# Load v2 (recommended — includes HTML backfill)
df = cache.get(2)

# Scope to a specific org
peta_emails = df.filter(df["client_organization_id"] == "447")

# Load only specific columns (projection pushdown)
df = cache.get(2, columns=["id", "date_sent", "subject_line", "client_organization_id"])

# Inspect the data dictionary
dd = cache.data_dictionary(2)
print(dd.keys())

# List registered versions
print(cache.list_versions())  # [1, 2]
```

## Versions

| Version | S3 key | Description |
|---------|--------|-------------|
| v1 | `emails_all.parquet` | Original extraction from MongoDB `analysis.emails`. Non-PETA orgs had 75–100% empty `html` fields. |
| v2 | `emails_all_v2.parquet` | v1 + HTML backfill for ~7,321 non-PETA emails (2025-03-18). **Recommended for all new work.** |

### v2 HTML coverage by org

| Org | Coverage |
|-----|----------|
| PETA | 100% |
| HIAS | 100% |
| FBotR | 99% (29% missing `<body>` tag) |
| SHH | 100% (0% body extraction — no `<body>` tag in HTML) |
| FA | 98% |
| AJWS | 97% |
| CFBNJ | 93% |
| ACS | 90% |
| FF | 16% (HTML missing from MongoDB) |

## Org ID Reference

| Org | `client_organization_id` |
|-----|--------------------------|
| PETA | `447` |
| Feeding America (FA) | `442` |
| ACS | `439` |
| AJWS | `440` |
| CFBNJ | `441` |
| FBotR | `443` |
| FF | `444` |
| HIAS | `445` |
| SHH | `448` |

## Provenance

- **S3 bucket**: `chorus-content-assets`
- **S3 prefix**: `data-science/extractors/emails/` (legacy prefix — retained verbatim so existing payloads are accessible without re-upload)
- **S3 keys**: `emails_all.parquet` (v1), `emails_all_v2.parquet` (v2)
- **Source collection**: MongoDB `analysis.emails`
- **Legacy location**: `datascience/projects/scout/extractions/emails/`

Note: the Python module and local cache dir use the `scout_emails` slug; the S3
keys retain their original `emails_all` filenames. The asymmetry is intentional —
the data is already in S3 under the legacy names and does not need to be re-uploaded.

## Rebuild

Both `create_cache_v1.py` and `create_cache_v2.py` are **documented stubs** that
raise `NotImplementedError`. S3 is the source of truth and is stable.

- **v1**: Full extraction requires the backend monorepo
  (`dev/backend/datascience/extractors/emails/create_cache.py`).
- **v2**: Produced by `projects/scout/investigations/02_relevance-labels/03_backfill_html.py`
  in the legacy `datascience` repo. Requires a live MongoDB connection (PyMongo, batch=500).

For normal usage, `cache.get(v)` downloads directly from S3 on first access and
caches locally under `data/entities/scout_emails/`. No rebuild is needed.
