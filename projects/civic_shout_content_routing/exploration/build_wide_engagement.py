"""Build the (email_id, user_id) wide engagement frame for exploration.

Derived from two civic_shout_engagement entities -- not a fourth cached
entity. Mirrors the canonical recipe in
``entities/civic_shout_engagement/attributed_actions/README.md`` (the
"(email_id x user_id) wide table" section).

Output columns
--------------
email_id:         i64
user_id:          i64
sent_at:          datetime[us, UTC]   -- created_at of the 'sent' row
opened:           bool                -- any action_type == 'open'
verified_opened:  bool                -- any action_type == 'verified_open'
clicked:          bool                -- any action_type == 'click'
actioned:         bool                -- attributed signature exists for (email, user)

Defaults
--------
- email_activities version: v2 (24-month, anchored at 2026-04-21)
- attributed_actions version: v2 (next-send-boundary, uncapped)

These are the 24-month reproducible upstreams used by aggregate_emails v3+.
Override via CLI flags if you need the dynamic 1-month windows (v1/v1).

Naming gotcha
-------------
``verified_opened`` (bool, this wide shape) is NOT the same as
``verified_open`` (enum value of ``action_type`` in email_activities). The
enum value is what we filter on; the boolean is the aggregate we emit.

Usage
-----
    python projects/civic_shout_content_routing/exploration/build_wide_engagement.py
    python ... --activities-v 1 --attr-v 1                    # dynamic 1-month
    python ... --out path/to/file.parquet                     # custom output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from entities.civic_shout_engagement.attributed_actions.cache import (
    cache as attr_cache,
)
from entities.civic_shout_engagement.email_activities_cache import (
    cache as acts_cache,
)


def build_wide(activities_v: int = 2, attr_v: int = 2) -> pl.DataFrame:
    acts = acts_cache.get(activities_v)

    sends = (
        acts.filter(pl.col("action_type") == "sent")
        .select(
            "email_id",
            "user_id",
            pl.col("created_at").alias("sent_at"),
        )
    )

    engagement = (
        acts.filter(pl.col("action_type").is_in(["open", "click", "verified_open"]))
        .group_by(["email_id", "user_id"])
        .agg(
            pl.col("action_type").eq("open").any().alias("opened"),
            pl.col("action_type").eq("verified_open").any().alias("verified_opened"),
            pl.col("action_type").eq("click").any().alias("clicked"),
        )
    )

    actioned = (
        attr_cache.get(attr_v)
        .filter(pl.col("is_attributed"))
        .select("email_id", "user_id", pl.lit(True).alias("actioned"))
        .unique()
    )

    return (
        sends.join(engagement, on=["email_id", "user_id"], how="left")
        .join(actioned, on=["email_id", "user_id"], how="left")
        .with_columns(
            pl.col("opened").fill_null(False),
            pl.col("verified_opened").fill_null(False),
            pl.col("clicked").fill_null(False),
            pl.col("actioned").fill_null(False),
        )
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activities-v", type=int, default=2, choices=(1, 2))
    p.add_argument("--attr-v", type=int, default=2, choices=(1, 2))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/projects/civic_shout_content_routing/exploration/"
            "wide_engagement.parquet"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = build_wide(activities_v=args.activities_v, attr_v=args.attr_v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)

    n_rows = df.height
    n_emails = df["email_id"].n_unique()
    n_users = df["user_id"].n_unique()
    print(f"wrote {args.out}  rows={n_rows:,}  emails={n_emails:,}  users={n_users:,}")
    print(
        df.select(
            pl.col("opened").mean().alias("opened_rate"),
            pl.col("verified_opened").mean().alias("vo_rate"),
            pl.col("clicked").mean().alias("clicked_rate"),
            pl.col("actioned").mean().alias("actioned_rate"),
        )
    )


if __name__ == "__main__":
    main()
