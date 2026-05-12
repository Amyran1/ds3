"""Build civic_shout_user_emails v1.

Composes:
  - civic_shout_engagement__email_activities v2 (sent/open/click events)
  - civic_shout_engagement__attributed_actions v1 (7-day attributed signatures)
  - content_routing.subscription_state_events (unsub events via Mongo)

Attribution for unsubscribes mirrors attributed_actions v1: join_asof backward
on (user_id, date_sent), tolerance=7 days.

Usage::

    python -m entities.civic_shout_user_emails.create_cache_v1
"""

from __future__ import annotations

import asyncio
import logging

import polars as pl

from entities.civic_shout_engagement.attributed_actions.cache import cache as aa_cache
from entities.civic_shout_engagement.email_activities_cache import cache as ea_cache
from entities.civic_shout_user_emails.cache import cache
from libs.clients.mongo import FindOptions
from libs.container import Container

logger = logging.getLogger(__name__)

CIVIC_SHOUT_ORG_ID: str = "civic_shout"
EMAIL_ACTIVITIES_VERSION: int = 2
ATTRIBUTED_ACTIONS_VERSION: int = 1
UNSUB_TOLERANCE_MS: int = 7 * 24 * 3600 * 1_000  # 7 days in milliseconds


async def fetch_unsub_events(c: Container) -> pl.DataFrame:
    """Pull unsubscribe events from content_routing.subscription_state_events."""
    async with c.mongo_db("content_routing") as mongo:
        docs = await mongo.find(
            "content_routing.subscription_state_events",
            FindOptions(
                filter={
                    "organization_id": CIVIC_SHOUT_ORG_ID,
                    "status": "unsubscribed",
                },
                projection={"_id": 0, "user_id": 1, "occurred_at": 1},
                batch_size=10_000,
                cost_key="civic_shout_user_emails_v1_unsub",
            ),
        )
    if not docs:
        return pl.DataFrame(
            {
                "user_id": pl.Series([], dtype=pl.Int64),
                "occurred_at": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            }
        )
    df = pl.DataFrame(docs)
    df = df.with_columns(
        pl.col("user_id").cast(pl.Int64),
        pl.col("occurred_at").cast(pl.Datetime("us", "UTC")),
    )
    return df


async def main() -> None:
    ea = ea_cache.get(EMAIL_ACTIVITIES_VERSION)
    aa = aa_cache.get(ATTRIBUTED_ACTIONS_VERSION)
    async with Container() as c:
        unsub_events = await fetch_unsub_events(c)

    logger.info("email_activities rows: %d", len(ea))
    logger.info("attributed_actions rows: %d", len(aa))
    logger.info("unsub events: %d", len(unsub_events))

    # 1. Sends: one row per (user_id, email_id) — earliest send timestamp.
    sends = (
        ea.filter(pl.col("action_type") == "sent")
        .group_by(["user_id", "email_id"])
        .agg(pl.col("created_at").min().alias("date_sent"))
    )

    # 2. Opens: any open event → opened=True.
    opens = (
        ea.filter(pl.col("action_type") == "open")
        .select(["user_id", "email_id"])
        .unique()
        .with_columns(pl.lit(value=True).alias("opened"))
    )

    # 3. Clicks: any click event → clicked=True.
    clicks = (
        ea.filter(pl.col("action_type") == "click")
        .select(["user_id", "email_id"])
        .unique()
        .with_columns(pl.lit(value=True).alias("clicked"))
    )

    # 4. Actioned: (user_id, email_id) in attributed_actions v1 with is_attributed=True.
    actioned = (
        aa.filter(pl.col("is_attributed"))
        .select(["user_id", "email_id"])
        .unique()
        .with_columns(pl.lit(value=True).alias("actioned"))
    )

    # 5. Unsub attribution: join_asof unsub_events onto sends, backward, 7-day tolerance.
    #    For each unsub event, find the most recent prior send within 7 days.
    if len(unsub_events) > 0 and len(sends) > 0:
        sends_sorted = sends.sort(["user_id", "date_sent"])
        unsub_sorted = unsub_events.sort(["user_id", "occurred_at"])

        unsub_attributed = unsub_sorted.join_asof(
            sends_sorted,
            left_on="occurred_at",
            right_on="date_sent",
            by="user_id",
            strategy="backward",
            tolerance=UNSUB_TOLERANCE_MS * 1_000,  # polars tolerance in microseconds
        )

        unsub_attributed = (
            unsub_attributed.filter(pl.col("email_id").is_not_null())
            .select(["user_id", "email_id"])
            .unique()
            .with_columns(pl.lit(value=True).alias("unsubscribed"))
        )
    else:
        unsub_attributed = pl.DataFrame(
            {
                "user_id": pl.Series([], dtype=pl.Int64),
                "email_id": pl.Series([], dtype=pl.Int64),
                "unsubscribed": pl.Series([], dtype=pl.Boolean),
            }
        )

    # 6. Join everything onto sends, fill nulls False.
    df = (
        sends.join(opens, on=["user_id", "email_id"], how="left")
        .join(clicks, on=["user_id", "email_id"], how="left")
        .join(actioned, on=["user_id", "email_id"], how="left")
        .join(unsub_attributed, on=["user_id", "email_id"], how="left")
        .with_columns(
            pl.col("opened").fill_null(value=False),
            pl.col("clicked").fill_null(value=False),
            pl.col("actioned").fill_null(value=False),
            pl.col("unsubscribed").fill_null(value=False),
        )
    )

    # 7. Imputation: if actioned=True, force opened/verified_opened/clicked True.
    df = df.with_columns(
        (pl.col("opened") | pl.col("actioned")).alias("opened"),
        (pl.col("opened") | pl.col("actioned")).alias("verified_opened"),
        (pl.col("clicked") | pl.col("actioned")).alias("clicked"),
    )

    # 8. Final column order.
    df = df.select(
        [
            "user_id",
            "email_id",
            "date_sent",
            "opened",
            "verified_opened",
            "clicked",
            "actioned",
            "unsubscribed",
        ]
    )

    logger.info("final rows: %d", len(df))
    n_actioned = int(df["actioned"].sum())
    n_opened = int(df["opened"].sum())
    n_unsubscribed = int(df["unsubscribed"].sum())
    logger.info("actioned: %d", n_actioned)
    logger.info("opened (post-imputation): %d", n_opened)
    logger.info("unsubscribed: %d", n_unsubscribed)

    cache.put(1, df)


if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=_fmt)
    asyncio.run(main())
