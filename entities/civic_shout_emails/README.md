# civic_shout_emails

One row per Civic Shout email. Combines metadata from MongoDB with precomputed
dense vectors from Pinecone and fresh sparse vectors from Pinecone Inference.

## Versions

| Version | Rows | Notes |
|---|---|---|
| v1 | ~4,634 | All `vectorized=True` emails for `organization_id="civic_shout"` |

## Schema

| Column | Type | Source |
|---|---|---|
| `email_id` | `i64` | `content_routing.emails.email_id` |
| `organization_id` | `str` | `content_routing.emails.organization_id` |
| `date_sent` | `datetime[us, UTC]` | `content_routing.emails.sent_at` |
| `body_plaintext` | `str` | `content_routing.emails.body_plaintext` |
| `dense_vector` | `list[f32]` (1536 dims) | Fetched from Pinecone index `production-content-routing-dense`, namespace `civic_shout_vv1`, keyed by `email_id` |
| `sparse_vector` | `struct{indices: list[i64], values: list[f32]}` | Computed fresh via Pinecone Inference API `pinecone-sparse-english-v0` — **NOT** fetched from any stored Pinecone index |

## Sparse vector provenance

`sparse_vector` is a BM25-style embedding computed at build time by calling
`pc.inference.embed(model="pinecone-sparse-english-v0", inputs=[...], parameters={"input_type": "passage"})`.
It is not stored in or retrieved from `production-content-routing-sparse` or any
other Pinecone index.

## Usage

```python
from entities.civic_shout_emails import cache

df = cache.get(1)
```

## Building

```bash
python -m entities.civic_shout_emails.create_cache_v1
```

Requires Mongo and Pinecone credentials in `.env`. Run a timing smoke test
before full-scale execution (see `ds_v2:timing-performance` skill).
