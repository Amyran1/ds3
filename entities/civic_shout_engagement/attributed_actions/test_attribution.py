"""Unit tests for build_attributed_actions — 7 edge cases from plan §4 step 5."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from entities.civic_shout_engagement.attributed_actions.create_cache_v1 import (
    CAP_SECONDS,
    build_attributed_actions,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _at(h: int = 0, m: int = 0, s: int = 0) -> datetime:
    """Build a UTC datetime anchored at 2026-01-01 + h:m:s (supports h>23)."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(hours=h, minutes=m, seconds=s)


def _send(uid: int, eid: int, ts: datetime) -> dict:
    """Return a dict row representing a ``sent`` activity."""
    return {"user_id": uid, "email_id": eid, "action_type": "sent", "created_at": ts}


def _sig(sid: int, uid: int, pid: int, ts: datetime) -> dict:
    """Return a dict row representing a signature (action)."""
    return {"signature_id": sid, "user_id": uid, "petition_id": pid, "created_at": ts}


def _mk_activities(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal email_activities DataFrame from a list of dicts.

    Required keys: user_id, email_id, action_type, created_at.
    """
    return pl.DataFrame(
        {
            "user_id": [r["user_id"] for r in rows],
            "email_id": [r["email_id"] for r in rows],
            "action_type": [r["action_type"] for r in rows],
            "created_at": [r["created_at"] for r in rows],
        },
        schema={
            "user_id": pl.Int64,
            "email_id": pl.Int64,
            "action_type": pl.Utf8,
            "created_at": pl.Datetime("us", "UTC"),
        },
    )


def _mk_actions(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal actions DataFrame.

    Required keys: signature_id, user_id, petition_id, created_at.
    """
    return pl.DataFrame(
        {
            "signature_id": [r["signature_id"] for r in rows],
            "user_id": [r["user_id"] for r in rows],
            "petition_id": [r["petition_id"] for r in rows],
            "created_at": [r["created_at"] for r in rows],
        },
        schema={
            "signature_id": pl.Int64,
            "user_id": pl.Int64,
            "petition_id": pl.Int64,
            "created_at": pl.Datetime("us", "UTC"),
        },
    )


# Sentinel "far future" send (h=200 >> 7d cap) used to make a prior send non-last.
_FAR = _at(h=200)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_send_single_sig_within_window() -> None:
    """Case (a): one send at T=0, second send far in future, one sig at T+1h.

    The first send is non-last (second send exists), so the default
    exclude_last_send=True path is exercised.  The signature is in-window
    and must be attributed.
    """
    activities = _mk_activities(
        [
            _send(1, 10, _at(0)),
            _send(1, 11, _FAR),  # makes send #1 non-last
        ],
    )
    actions = _mk_actions([_sig(100, 1, 9, _at(1))])
    df = build_attributed_actions(activities, actions)
    assert len(df) == 1
    row = df.row(0, named=True)
    assert row["is_attributed"] is True
    assert row["email_id"] == 10
    assert row["lag_seconds"] == 3600
    assert row["is_first_in_window"] is True


def test_lag_equals_window_is_unattributed() -> None:
    """Case (b): strict < at boundary — sig at exactly window_seconds is unattributed.

    Send at T=0, next send at T+7d (= CAP_SECONDS away), signature at T+7d exactly.
    lag_seconds == window_seconds → must be unattributed.
    """
    send_t = _at(0)
    next_t = _at(s=CAP_SECONDS)  # T + 7d exactly
    sig_t = _at(s=CAP_SECONDS)  # same moment as next send

    activities = _mk_activities(
        [
            _send(1, 10, send_t),
            _send(1, 11, next_t),
        ],
    )
    actions = _mk_actions([_sig(100, 1, 9, sig_t)])
    df = build_attributed_actions(activities, actions)
    row = df.row(0, named=True)
    assert row["is_attributed"] is False
    assert row["email_id"] is None
    assert row["send_time"] is None
    assert row["lag_seconds"] is None
    assert row["window_seconds"] is None


def test_lag_equals_window_minus_one_is_attributed() -> None:
    """Case (c): signature at window_seconds - 1 second IS attributed (strict <)."""
    send_t = _at(0)
    next_t = _at(s=CAP_SECONDS)
    sig_t = _at(s=CAP_SECONDS - 1)  # one second inside the window

    activities = _mk_activities(
        [
            _send(1, 10, send_t),
            _send(1, 11, next_t),
        ],
    )
    actions = _mk_actions([_sig(100, 1, 9, sig_t)])
    df = build_attributed_actions(activities, actions)
    row = df.row(0, named=True)
    assert row["is_attributed"] is True
    assert row["email_id"] == 10
    assert row["is_first_in_window"] is True


def test_cascade_only_first_is_attributed() -> None:
    """Case (d): two sigs in same window — earliest attributed, second is cascade.

    Second sig has email_id set (it's in-window) but is_first_in_window=False
    and is_attributed=False.
    """
    activities = _mk_activities(
        [
            _send(1, 10, _at(0)),
            _send(1, 11, _FAR),
        ],
    )
    actions = _mk_actions(
        [
            _sig(100, 1, 9, _at(1)),
            _sig(101, 1, 9, _at(2)),
        ],
    )
    df = build_attributed_actions(activities, actions).sort("signature_id")
    assert len(df) == 2

    first = df.row(0, named=True)
    assert first["signature_id"] == 100
    assert first["is_first_in_window"] is True
    assert first["is_attributed"] is True
    assert first["email_id"] == 10

    second = df.row(1, named=True)
    assert second["signature_id"] == 101
    assert second["email_id"] == 10  # still in-window
    assert second["is_first_in_window"] is False
    assert second["is_attributed"] is False


def test_tie_on_created_at_tiebreaks_on_signature_id() -> None:
    """Case (e): two sigs at identical created_at → lower signature_id wins."""
    activities = _mk_activities(
        [
            _send(1, 10, _at(0)),
            _send(1, 11, _FAR),
        ],
    )
    tie_time = _at(1)
    actions = _mk_actions(
        [
            _sig(5, 1, 9, tie_time),
            _sig(7, 1, 9, tie_time),
        ],
    )
    df = build_attributed_actions(activities, actions).sort("signature_id")
    assert len(df) == 2

    winner = df.row(0, named=True)
    assert winner["signature_id"] == 5
    assert winner["is_first_in_window"] is True

    loser = df.row(1, named=True)
    assert loser["signature_id"] == 7
    assert loser["is_first_in_window"] is False


def test_single_send_per_user_excluded_default() -> None:
    """Case (f): user has exactly one send; exclude_last_send=True drops it.

    All signatures should be unattributed with all four attribution cols null.
    """
    activities = _mk_activities([_send(1, 10, _at(0))])
    actions = _mk_actions([_sig(100, 1, 9, _at(1))])
    df = build_attributed_actions(activities, actions)
    assert len(df) == 1
    row = df.row(0, named=True)
    assert row["is_attributed"] is False
    assert row["email_id"] is None
    assert row["send_time"] is None
    assert row["lag_seconds"] is None
    assert row["window_seconds"] is None


def test_signature_before_any_send_unattributed() -> None:
    """Case (g): signature at T=0, sends at T+1d and T+8d — sig precedes all sends.

    join_asof backward finds no prior send → all four attribution cols null.
    """
    activities = _mk_activities(
        [
            _send(1, 10, _at(h=24)),
            _send(1, 11, _at(h=24 * 8)),
        ],
    )
    actions = _mk_actions([_sig(100, 1, 9, _at(0))])
    df = build_attributed_actions(activities, actions)
    assert len(df) == 1
    row = df.row(0, named=True)
    assert row["is_attributed"] is False
    assert row["email_id"] is None
    assert row["send_time"] is None
    assert row["lag_seconds"] is None
    assert row["window_seconds"] is None
