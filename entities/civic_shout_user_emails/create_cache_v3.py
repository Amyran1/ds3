"""Build civic_shout_user_emails v3.

Differences from v2:
  - Replaces ``attributed_actions``-based attribution with a strict **24-hour
    backward asof-join** against ``user_petition v1``.
  - Drops v2's ``actioned`` boolean and adds two new columns:
      ``n_actions_24h: UInt32`` — count of petition signatures attributed to
          this (user, email) pair via the 24h window.
      ``actioned_24h: Boolean`` — True iff n_actions_24h >= 1.
  - Pooled positive rate: ~9.26% across the full 24-month dataset.

Composes:
  - civic_shout_user_emails v2 base columns (user_id, email_id, date_sent,
    opened, verified_opened, clicked, unsubscribed) — NOT actioned.
  - user_petition v1 (user_id, petition_id, signed_at).

Attribution: for each petition signature, find the most-recent prior send to
the same user within 24 hours.  Group by (user_id, email_id), count distinct
signatures → n_actions_24h.

Note on imputed engagement columns: opened, verified_opened, clicked, and
unsubscribed are inherited from v2 base columns verbatim — the v2 imputation
rule (actioned=True ⇒ opened=clicked=True) is baked into the v2 source values.
The F03 drop rule still applies under v3.

Usage::

    python -m entities.civic_shout_user_emails.create_cache_v3
"""

from __future__ import annotations

import logging

import polars as pl

from entities.civic_shout_user_emails.cache import cache
from entities.civic_shout_user_emails.cache import cache as user_emails_cache
from entities.user_petition.cache import cache as user_petition_cache

logger = logging.getLogger(__name__)

USER_EMAILS_VERSION: int = 2
USER_PETITION_VERSION: int = 1
V3_BASE_COLS: list[str] = [
    "user_id",
    "email_id",
    "date_sent",
    "opened",
    "verified_opened",
    "clicked",
    "unsubscribed",
]
ATTRIBUTION_TOLERANCE: str = "24h"

EXPECTED_ROWS: int = 129_190_540
EXPECTED_ACTIONED_RATE_LO: float = 0.092
EXPECTED_ACTIONED_RATE_HI: float = 0.093


def build_v3() -> pl.DataFrame:
    user_emails_v2 = user_emails_cache.get(USER_EMAILS_VERSION).select(V3_BASE_COLS)
    user_petition = user_petition_cache.get(USER_PETITION_VERSION).select(
        ["user_id", "petition_id", "signed_at"]
    )

    logger.info("user_emails v2 base rows: %d", len(user_emails_v2))
    logger.info("user_petition v1 rows: %d", len(user_petition))

    user_emails_sorted = user_emails_v2.sort(["user_id", "date_sent"])
    user_petition_sorted = user_petition.sort(["user_id", "signed_at"])

    attributed = user_petition_sorted.join_asof(
        user_emails_sorted,
        left_on="signed_at",
        right_on="date_sent",
        by="user_id",
        strategy="backward",
        tolerance=ATTRIBUTION_TOLERANCE,
    ).filter(pl.col("email_id").is_not_null())

    logger.info("attributed signature pairs (within 24h): %d", len(attributed))

    n_actions = (
        attributed.group_by(["user_id", "email_id"])
        .agg(pl.len().alias("n_actions_24h"))
        .with_columns(pl.col("n_actions_24h").cast(pl.UInt32))
    )

    df = (
        user_emails_v2.join(n_actions, on=["user_id", "email_id"], how="left")
        .with_columns(pl.col("n_actions_24h").fill_null(pl.lit(0, dtype=pl.UInt32)))
        .with_columns((pl.col("n_actions_24h") >= 1).alias("actioned_24h"))
        .select(
            [
                "user_id",
                "email_id",
                "date_sent",
                "opened",
                "verified_opened",
                "clicked",
                "unsubscribed",
                "n_actions_24h",
                "actioned_24h",
            ]
        )
    )

    n_rows = len(df)
    n_actioned = int(df["actioned_24h"].sum())
    actioned_rate = n_actioned / n_rows

    logger.info("final rows: %d", n_rows)
    logger.info("actioned_24h: %d (%.4f%%)", n_actioned, actioned_rate * 100)

    if n_rows != EXPECTED_ROWS:
        logger.warning("row count mismatch: got %d, expected %d", n_rows, EXPECTED_ROWS)

    if not (EXPECTED_ACTIONED_RATE_LO <= actioned_rate <= EXPECTED_ACTIONED_RATE_HI):
        logger.warning(
            "actioned_24h rate %.4f outside expected [%.3f, %.3f]",
            actioned_rate,
            EXPECTED_ACTIONED_RATE_LO,
            EXPECTED_ACTIONED_RATE_HI,
        )

    return df


def main() -> None:
    df = build_v3()
    cache.put(3, df)
    logger.info("v3 cache written successfully")


if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=_fmt)
    main()
