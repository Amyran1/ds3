# user_petition

A semantically-clean view of the petition signature stream, joining users to the petitions they signed. Derived from `civic_shout_engagement__actions v2` (24-month window anchored at 2026-04-21).

## Schema (v1)

| Column | Type | Description |
|---|---|---|
| `user_id` | `i64` | Signer |
| `petition_id` | `i64` | Which petition was signed |
| `signed_at` | `datetime[us, UTC]` | When the signature occurred (equals `actions.created_at`) |

One row per signature event. Duplicate `(user_id, petition_id)` pairs are preserved — consumers who need unique pairs should call `.unique()`.

## Build

```bash
source .venv/bin/activate
ENVIRONMENT=PRODUCTION python -m entities.user_petition.create_cache_v1
```
