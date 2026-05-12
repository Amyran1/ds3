"""Build civic_shout_user_emails v2.

Differences from v1:
  - pins attributed_actions **v2** (24-month, uncapped window) instead of v1
    (1 month, 7-day cap). v1's 1-month actions window produced a misleading
    pooled action rate of 0.544%; v2 surfaces the real ~9.76% baseline.
  - imputes ``opened`` from raw clicks (a click is high-fidelity evidence the
    email was opened — addresses Apple MPP / similar pixel suppression).

Composes:
  - civic_shout_engagement__email_activities v2 (sent/open/click events)
  - civic_shout_engagement__attributed_actions v2 (uncapped 24-month attribution)
  - content_routing.subscription_state_events (unsub events via Mongo)

Unsub attribution still uses a 7-day join_asof cap — the cap matches the
project's operational definition of "this email caused the unsub" and is not
tied to attributed_actions' cap.

Usage::

    python -m entities.civic_shout_user_emails.create_cache_v2
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
ATTRIBUTED_ACTIONS_VERSION: int = 2  # 24-month, uncapped — v2's defining change
UNSUB_TOLERANCE_MS: int = 7 * 24 * 3600 * 1_000


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
                cost_key="civic_shout_user_emails_v2_unsub",
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
    return df.with_columns(
        pl.col("user_id").cast(pl.Int64),
        pl.col("occurred_at").cast(pl.Datetime("us", "UTC")),
    )


async def main() -> None:
    ea = ea_cache.get(EMAIL_ACTIVITIES_VERSION)
    aa = aa_cache.get(ATTRIBUTED_ACTIONS_VERSION)
    async with Container() as c:
        unsub_events = await fetch_unsub_events(c)

    logger.info("email_activities rows: %d", len(ea))
    logger.info("attributed_actions v%d rows: %d", ATTRIBUTED_ACTIONS_VERSION, len(aa))
    logger.info("unsub events: %d", len(unsub_events))

    sends = (
        ea.filter(pl.col("action_type") == "sent")
        .group_by(["user_id", "email_id"])
        .agg(pl.col("created_at").min().alias("date_sent"))
    )

    opens = (
        ea.filter(pl.col("action_type") == "open")
        .select(["user_id", "email_id"])
        .unique()
        .with_columns(pl.lit(value=True).alias("opened"))
    )

    clicks = (
        ea.filter(pl.col("action_type") == "click")
        .select(["user_id", "email_id"])
        .unique()
        .with_columns(pl.lit(value=True).alias("clicked"))
    )

    actioned = (
        aa.filter(pl.col("is_attributed"))
        .select(["user_id", "email_id"])
        .unique()
        .with_columns(pl.lit(value=True).alias("actioned"))
    )

    if len(unsub_events) > 0 and len(sends) > 0:
        sends_sorted = sends.sort(["user_id", "date_sent"])
        unsub_sorted = unsub_events.sort(["user_id", "occurred_at"])

        unsub_attributed = unsub_sorted.join_asof(
            sends_sorted,
            left_on="occurred_at",
            right_on="date_sent",
            by="user_id",
            strategy="backward",
            tolerance=UNSUB_TOLERANCE_MS * 1_000,
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

    # v2 imputation: clicked OR actioned implies opened/verified_opened.
    # A raw click is high-fidelity evidence of an open (handles Apple MPP /
    # pixel-blocked clients where the open event never fires).
    df = df.with_columns(
        (pl.col("opened") | pl.col("clicked") | pl.col("actioned")).alias("opened"),
        (pl.col("opened") | pl.col("clicked") | pl.col("actioned")).alias("verified_opened"),
        (pl.col("clicked") | pl.col("actioned")).alias("clicked"),
    )

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

    n_actioned = int(df["actioned"].sum())
    n_opened = int(df["opened"].sum())
    n_clicked = int(df["clicked"].sum())
    n_unsubscribed = int(df["unsubscribed"].sum())
    logger.info("final rows: %d", len(df))
    logger.info("actioned: %d (%.4f%%)", n_actioned, n_actioned / len(df) * 100)
    logger.info("opened: %d (%.4f%%)", n_opened, n_opened / len(df) * 100)
    logger.info("clicked: %d (%.4f%%)", n_clicked, n_clicked / len(df) * 100)
    logger.info("unsubscribed: %d (%.4f%%)", n_unsubscribed, n_unsubscribed / len(df) * 100)

    cache.put(2, df)


if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=_fmt)
    asyncio.run(main())
