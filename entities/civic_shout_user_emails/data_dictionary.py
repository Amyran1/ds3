"""Column semantics for civic_shout_user_emails (v1, v2)."""

from __future__ import annotations

V1_DICTIONARY: dict[str, str] = {
    "user_id": (
        "Recipient user ID. Primary key component. From email_activities where action_type='sent'."
    ),
    "email_id": (
        "Email ID. Primary key component. From email_activities where action_type='sent'."
    ),
    "date_sent": (
        "UTC timestamp of the send event. From email_activities.created_at "
        "where action_type='sent'."
    ),
    "opened": (
        "True iff any email_activities row with action_type='open' exists for "
        "(user_id, email_id). Imputed to True whenever actioned=True."
    ),
    "verified_opened": (
        "Same as `opened`. Mongo collapses Redshift's `verified_open` → `OPEN` "
        "(bot-filtered) so the distinction does not exist in source data. "
        "Imputed to True whenever actioned=True."
    ),
    "clicked": (
        "True iff any email_activities row with action_type='click' exists for "
        "(user_id, email_id). Imputed to True whenever actioned=True. "
        "WARN: 'clicked' is not a fidelity-preserving signal — it's imputed from 'actioned'."
    ),
    "actioned": (
        "True iff (user_id, email_id) appears in attributed_actions v1 with "
        "is_attributed=True. v1 uses a 7-day attribution window cap. "
        "KNOWN ISSUE: attributed_actions v1 only has 1 month of signatures "
        "(2026-03-20 → 2026-04-20), so v1 of this cache reports ~0% actioned "
        "for sends outside that window. Use v2 of this cache for the real baseline."
    ),
    "unsubscribed": (
        "True iff this email was attributed as the cause of an unsubscribe event — "
        "i.e., an unsub event from content_routing.subscription_state_events is "
        "attributed to this email via join_asof with a 7-day cap (mirroring "
        "attributed_actions v1). Resubscribe → re-unsubscribe sequences attribute "
        "each unsub event independently. This is NOT 'user is currently "
        "unsubscribed' — that's a different concept."
    ),
}


V2_DICTIONARY: dict[str, str] = {
    "user_id": (
        "Recipient user ID. Primary key component. From email_activities where action_type='sent'."
    ),
    "email_id": (
        "Email ID. Primary key component. From email_activities where action_type='sent'."
    ),
    "date_sent": (
        "UTC timestamp of the send event. From email_activities.created_at "
        "where action_type='sent'."
    ),
    "opened": (
        "True iff any email_activities row with action_type='open' exists for "
        "(user_id, email_id), OR clicked=True (raw), OR actioned=True. v2 imputes "
        "opened from raw clicks (a click is high-fidelity evidence the email was "
        "opened even when Apple MPP / similar suppresses the open pixel)."
    ),
    "verified_opened": (
        "Same as `opened`. Mongo collapses Redshift's `verified_open` → `OPEN` "
        "(bot-filtered) so the distinction does not exist in source data. "
        "Imputed to True whenever raw clicked OR actioned."
    ),
    "clicked": (
        "True iff any email_activities row with action_type='click' exists for "
        "(user_id, email_id). Imputed to True whenever actioned=True. "
        "WARN: 'clicked' is not a fidelity-preserving signal — it's imputed from 'actioned'."
    ),
    "actioned": (
        "True iff (user_id, email_id) appears in attributed_actions v2 with "
        "is_attributed=True. v2 covers a 24-month window (2024-04-21 → 2026-04-20) "
        "using actions v2 + email_activities v2, and is uncapped (no 7-day cap on "
        "the per-send attribution window). Real baseline action rate: ~9.76% across "
        "the full window, ranging 7.89% (2024) → 12.59% (2026)."
    ),
    "unsubscribed": (
        "True iff this email was attributed as the cause of an unsubscribe event — "
        "i.e., an unsub event from content_routing.subscription_state_events is "
        "attributed to this email via join_asof with a 7-day cap. Resubscribe → "
        "re-unsubscribe sequences attribute each unsub event independently. This "
        "is NOT 'user is currently unsubscribed'."
    ),
}


VERSIONS: dict[int, dict[str, str]] = {
    1: V1_DICTIONARY,
    2: V2_DICTIONARY,
}


DATA_DICTIONARY: dict[str, str] = V1_DICTIONARY
