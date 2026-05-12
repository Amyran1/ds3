# petition entity

One row per petition signed in the 24-month window `[2024-04-21, 2026-04-21)` for Action Network group 228691 (1,219 petitions). Includes dense and sparse vectors for semantic search and feature extraction.

## Schema (v1)

| Column | Type | Description |
|---|---|---|
| `petition_id` | `i64` | Primary key (`petitions.id`) |
| `created_at` | `datetime[us, UTC]` | Petition creation timestamp |
| `text` | `str` | `title` + HTML-stripped `description_info` (when present) |
| `dense_vector` | `list[f32]` | 1536-dim `text-embedding-3-small`, L2-normalized |
| `sparse_vector` | `struct{indices: list[u32], values: list[f32]}` | BM25 (k1=1.2, b=0.75), L2-normalized |

## Build

```bash
source .venv/bin/activate
ENVIRONMENT=PRODUCTION python -m entities.petition.create_cache_v1
```
