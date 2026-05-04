# attributed_actions

Time-based attribution ledger linking petition signatures to the email send
that most recently preceded them. One row per signature from
`civic_shout_engagement__actions`. Every signature appears exactly once: either
attributed to an email (``is_attributed=true``) or organic/unattributed
(``email_id=null``).

Attribution is a **time-based accounting convention**, not a causal claim.
A signature attributed to email X means X was the last send before the
signature within the window; it does not prove X caused the signature.

## Source

Derived from two upstream caches. Both must be built before running
`create_cache_v1.py`:

| Upstream | Entity name | Version |
|---|---|---|
| `entities/civic_shout_engagement/email_activities_cache.py` | `civic_shout_engagement__email_activities` | v1 |
| `entities/civic_shout_engagement/actions_cache.py` | `civic_shout_engagement__actions` | v1 |

No Redshift query. No raw CSV. Pure polars `join_asof`.

- **Volume (v1 estimates).** Same row count as `actions` v1 (~10⁵–10⁶ rows).
- **Refresh cadence.** Rebuild after either upstream cache is rebuilt.

## Attribution rule (v1 defaults)

1. Per user, sort sends (`email_activities` where `action_type == 'sent'`) ascending by `created_at`.
2. Per user, sort signatures (`actions`) ascending by `created_at`, ties broken by `signature_id`.
3. **Exclude the last send per user** (default, configurable via `exclude_last_send=False`). Prevents pre/post asymmetry where a user's only remaining send is the most recent one.
4. For each signature, find the **most recent prior send** by the same user where `send_time <= sign_time` AND `sign_time - send_time < min(next_send_time - send_time, 7 days)`. The upper bound is **strict** (`<`), matching the schema contract below.
5. Within a given `(user_id, email_id)` window, the **earliest** signature (by `created_at`, ties on `signature_id`) is attributed (`is_first_in_window=True`). All subsequent signatures in the same window are organic cascade (`is_first_in_window=False`, `is_attributed=False`).
6. Signatures with no prior send in window have `email_id=null`, `is_attributed=false`.

**Window-boundary semantics:** `lag_seconds < window_seconds` (strict less-than). A signature arriving at exactly `window_seconds` is **unattributed**. This matches the R7b invariant tested in `test_attribution.py`.

## Versions

- **v1** — initial build (2026-04-20). 7-day cap, last-send-excluded default. Reads `email_activities v1` + `actions v1`.

Upstream version bumps require a corresponding `attributed_actions` version bump. Do not mix upstream versions mid-investigation.

## Schema (v1)

| Column | Type | Notes |
|---|---|---|
| `signature_id` | `i64` | PK. Every row in `actions v1` appears exactly once. |
| `user_id` | `i64` | From `actions`. |
| `petition_id` | `i64` | From `actions`. |
| `sign_time` | `datetime[us, UTC]` | From `actions.created_at`. |
| `email_id` | `i64 \| null` | Attributed email. Null = unattributed / organic / external. |
| `send_time` | `datetime[us, UTC] \| null` | Null when `email_id` is null. |
| `lag_seconds` | `i64 \| null` | `sign_time - send_time` in seconds. Null when unattributed. |
| `window_seconds` | `i64 \| null` | `min(next_send - send, 7d)` in seconds. Null when unattributed. |
| `is_first_in_window` | `bool` | True = attributed; False = organic cascade within an attributed window. |
| `is_attributed` | `bool` | `email_id is not null AND is_first_in_window`. |

**Schema contract:** `email_id`, `send_time`, `lag_seconds`, and `window_seconds`
are null or non-null **together**. A row with a non-null `email_id` but null
`lag_seconds` is a bug.

**Column-name disambiguation:** `verified_open` is an `action_type` enum value
in `email_activities`. The wide-join shape below uses `verified_opened`
(boolean aggregate). Do not mix the two.

## Usage

```python
from entities.civic_shout_engagement.attributed_actions.cache import cache

df = cache.get(1)
# Filter to attributed signatures only.
attributed = df.filter(pl.col("is_attributed"))
```

## The `(email_id × user_id)` wide table

Aaron's target shape — one row per `(email_id, user_id)` with boolean
engagement flags — is assembled at read time by joining the three in-scope
entities. It is **not** a fourth cached entity; keeping it derived means each
upstream can version independently.

`sent_at` comes from the `sent` rows in `email_activities` directly (no
`emails` entity needed). When an `emails` entity is added later, join on
`email_id` to enrich with subject and campaign metadata.

```python
import polars as pl
from entities.civic_shout_engagement.email_activities_cache import cache as acts
from entities.civic_shout_engagement.attributed_actions.cache import cache as attr

sends = acts.get(1).filter(pl.col("action_type") == "sent").select(
    "email_id",
    "user_id",
    pl.col("created_at").alias("sent_at"),
)
engagement = (
    acts.get(1)
    .filter(pl.col("action_type").is_in(["open", "click", "verified_open"]))
    .group_by(["email_id", "user_id"])
    .agg(
        pl.col("action_type").eq("open").any().alias("opened"),
        pl.col("action_type").eq("verified_open").any().alias("verified_opened"),
        pl.col("action_type").eq("click").any().alias("clicked"),
    )
)
actioned = (
    attr.get(1)
    .filter(pl.col("is_attributed"))
    .select("email_id", "user_id", pl.lit(True).alias("actioned"))
    .unique()
)

wide = (
    sends.join(engagement, on=["email_id", "user_id"], how="left")
    .join(actioned, on=["email_id", "user_id"], how="left")
    .with_columns(
        pl.col("opened").fill_null(False),
        pl.col("verified_opened").fill_null(False),
        pl.col("clicked").fill_null(False),
        pl.col("actioned").fill_null(False),
    )
)
# wide schema: email_id, user_id, sent_at, opened, verified_opened, clicked, actioned
```

Note: `verified_opened` (boolean, wide shape) vs `verified_open` (enum value,
`action_type` filter). See column-name note in the schema section above.

## Consumed by

(none yet — first consumer will be a Civic Shout investigation)

Append a bullet when a new project starts consuming this entity.

## Ownership

- **Author.** aaron@chorusai.co (initial build 2026-04-20).
- **Cross-project changes.** Raise with the authoring owner before merging any
  version bump that changes attribution semantics (cap, tiebreak, exclusion rule).
