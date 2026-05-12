# civic_shout_user_emails

**Grain:** one row per `(user_id, email_id)` pair where a `sent` event exists in `email_activities` v2.

## Columns

| Column | Type | Description |
|---|---|---|
| `user_id` | `i64` | Recipient user ID |
| `email_id` | `i64` | The email that was sent |
| `date_sent` | `datetime[us, UTC]` | Timestamp of the send event |
| `opened` | `bool` | Any `open` activity; imputed True when `actioned=True` |
| `verified_opened` | `bool` | Same as `opened` (Mongo collapses `verified_open → open`) |
| `clicked` | `bool` | Any `click` activity; imputed True when `actioned=True` |
| `actioned` | `bool` | Appears in `attributed_actions` v1 with `is_attributed=True` (7-day window cap) |
| `unsubscribed` | `bool` | Unsub event attributed to this send via `join_asof`, 7-day cap (see note) |

## Imputation rule

If `actioned=True`, `opened`, `verified_opened`, and `clicked` are all forced to `True` (logical OR). `actioned` implies the user opened and clicked to complete an action.

## Unsub attribution semantic

`unsubscribed=True` means this email was the most-recent-prior send within 7 days of an unsubscribe event for that user — it is attributed as the likely cause of the unsub. This mirrors the attribution logic in `attributed_actions` v1.

This is **not** "user is currently unsubscribed." Resubscribe → re-unsubscribe sequences attribute each unsub event independently.

## Sources

- `entities/civic_shout_engagement/email_activities_cache.py` v2 — sent/open/click events (24-month window anchored 2026-04-21)
- `entities/civic_shout_engagement/attributed_actions/cache.py` v1 — 7-day attributed signatures
- `content_routing.subscription_state_events` (Mongo prod, org=`civic_shout`) — unsub events
