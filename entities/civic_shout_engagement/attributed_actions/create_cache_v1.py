"""Build attributed_actions v1 from civic_shout_engagement caches.

Attribution rule (v1 defaults):
- 7-day absolute window cap (CAP_SECONDS = 604800)
- Last send per user excluded from attribution (exclude_last_send=True)
- Ties on sign_time resolved by lowest signature_id

Usage::

    python -m entities.civic_shout_engagement.attributed_actions.create_cache_v1
"""

from __future__ import annotations

import logging

import polars as pl

from entities.civic_shout_engagement.actions_cache import cache as actions_cache
from entities.civic_shout_engagement.attributed_actions.cache import (
    cache as attributed_cache,
)
from entities.civic_shout_engagement.email_activities_cache import (
    cache as email_activities_cache,
)

logger = logging.getLogger(__name__)

CAP_SECONDS: int = 7 * 24 * 3600  # 604800 seconds == 7 days


def build_attributed_actions(
    activities: pl.DataFrame,
    actions: pl.DataFrame,
    cap_seconds: int = CAP_SECONDS,
    *,
    exclude_last_send: bool = True,
) -> pl.DataFrame:
    """Build the attribution ledger from raw email activities and signatures.

    For each signature, finds the most recent prior send by the same user
    where sign_time - send_time < min(next_send_time - send_time, cap_seconds).
    Within a (user_id, email_id) window the earliest signature (by created_at,
    tiebroken by signature_id) is ``is_first_in_window=True``; subsequent ones
    are an organic cascade.

    Args:
        activities: email_activities v1 DataFrame (must include ``action_type``,
            ``user_id``, ``email_id``, ``created_at``).
        actions: actions v1 DataFrame (must include ``signature_id``, ``user_id``,
            ``petition_id``, ``created_at``).
        cap_seconds: absolute window cap in seconds.  Default 7 days.
        exclude_last_send: when True (default), each user's most recent send is
            excluded before attribution.  Users with exactly one send will have
            all their signatures appear unattributed.

    Returns:
        One row per signature with columns:
        ``signature_id, user_id, petition_id, sign_time, email_id, send_time,
        lag_seconds, window_seconds, is_first_in_window, is_attributed``.
    """
    # --- Prepare sends: filter to sent events, sort by (user_id, send_time). ---
    sends = (
        activities.filter(pl.col("action_type") == "sent")
        .select(
            pl.col("user_id"),
            pl.col("email_id"),
            pl.col("created_at").alias("send_time"),
        )
        .sort(["user_id", "send_time"])
    )

    # Attach next_send_time per user via shift(-1) within user group.
    sends = sends.with_columns(
        pl.col("send_time").shift(-1).over("user_id").alias("next_send_time"),
    )

    if exclude_last_send:
        # Drop rows where next_send_time is null => user's last send.
        sends = sends.filter(pl.col("next_send_time").is_not_null())

    # Compute effective window per send: min(next_send - send, cap).
    # When exclude_last_send=True (default), next_send_time is guaranteed
    # non-null here and the .fill_null branch is dead code — preserved so
    # the False branch continues to work correctly.
    sends = sends.with_columns(
        pl.min_horizontal(
            (pl.col("next_send_time") - pl.col("send_time")).dt.total_seconds(),
            pl.lit(cap_seconds).cast(pl.Int64),
        )
        .fill_null(cap_seconds)
        .alias("window_seconds"),
    )

    # --- Prepare signatures: sort by (user_id, created_at, signature_id). ---
    sigs = actions.sort(["user_id", "created_at", "signature_id"])

    # --- join_asof backward: each sig gets its most recent prior send. ---
    joined = sigs.join_asof(
        sends.sort(["user_id", "send_time"]),
        left_on="created_at",
        right_on="send_time",
        by="user_id",
        strategy="backward",
    )

    # Compute lag_seconds = created_at (sign_time) - send_time.
    joined = joined.with_columns(
        ((pl.col("created_at") - pl.col("send_time")).dt.total_seconds()).alias(
            "lag_seconds",
        ),
    )

    # Enforce the window with strict <.  Signatures at exactly window_seconds
    # are OUT (R7b fix).
    not_null = pl.col("send_time").is_not_null()
    in_window = pl.col("lag_seconds") < pl.col("window_seconds")
    joined = joined.with_columns(
        (not_null & in_window).alias("_in_window"),
    )

    # Null out ALL four attribution columns together when _in_window is False.
    # This is the F4 fix: prevents the schema contract from being violated
    # (send_time / lag_seconds / window_seconds retained while email_id is nulled).
    def _keep(col: str) -> pl.Expr:
        expr = pl.when(pl.col("_in_window")).then(pl.col(col)).otherwise(None)
        return expr.alias(col)

    joined = joined.with_columns(
        _keep("email_id"),
        _keep("send_time"),
        _keep("lag_seconds"),
        _keep("window_seconds"),
    ).drop("_in_window")

    # Tag first-in-window.  Earliest signature by (created_at, signature_id)
    # wins — time wins, id breaks ties.  This is the F5 fix: NOT signature_id.min()
    # alone, which would pick the wrong row when created_at values differ.
    # Use rank(method="ordinal") over the struct — pl.struct.min().over() returns null
    # in this polars version when the over-group has only one element.
    sig_struct = pl.struct(["created_at", "signature_id"])
    joined = joined.with_columns(
        pl.when(pl.col("email_id").is_null())
        .then(pl.lit(value=False))
        .otherwise(
            sig_struct.rank(method="ordinal").over(["user_id", "email_id"]) == 1,
        )
        .alias("is_first_in_window"),
    )

    joined = joined.with_columns(
        (pl.col("email_id").is_not_null() & pl.col("is_first_in_window")).alias(
            "is_attributed",
        ),
    )

    return joined.select(
        "signature_id",
        "user_id",
        "petition_id",
        pl.col("created_at").alias("sign_time"),
        "email_id",
        "send_time",
        "lag_seconds",
        "window_seconds",
        "is_first_in_window",
        "is_attributed",
    )


def main() -> None:
    """Load upstream caches, build attribution ledger, write to attributed_cache."""
    activities = email_activities_cache.get(1)
    actions = actions_cache.get(1)
    df = build_attributed_actions(activities, actions)

    total = len(df)
    n_attributed = int(df["is_attributed"].sum())
    _mean_raw = df["is_attributed"].mean()
    pct_attributed: float = 100.0 * (_mean_raw if isinstance(_mean_raw, float) else 0.0)

    # Organic cascade: rows where email_id is non-null but is_first_in_window=False.
    n_cascade = int(
        (~df["is_first_in_window"] & df["email_id"].is_not_null()).sum(),
    )

    logger.info("total signatures: %d", total)
    logger.info(
        "attributed: %d (%.1f%%)",
        n_attributed,
        pct_attributed,
    )
    logger.info("organic cascade (in-window but not first): %d", n_cascade)

    attributed_cache.put(1, df)


if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=_fmt)
    main()
