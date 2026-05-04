"""Populate scout_emails cache v2 (HTML backfill).

v2 was produced by investigation 02 in the legacy datascience repo::

    projects/scout/investigations/02_relevance-labels/03_backfill_html.py

That script backfills ``html`` for ~7,321 non-PETA emails from
MongoDB ``analysis.emails`` (PyMongo, batch=500). v1 had 75-100% empty
``html`` fields for non-PETA orgs.

The v2 parquet is already in S3 at::

    s3://chorus-content-assets/data-science/extractors/emails/emails_all_v2.parquet

``cache.get(2)`` is the normal consumer path and will download it on first access.

Usage:
    PYTHONPATH=$(pwd) python -m libs.dsrun entities/scout_emails/create_cache_v2.py
"""

from __future__ import annotations

_MSG = (
    "v2 was created by investigation 02's 03_backfill_html.py in the legacy "
    "datascience repo. See projects/scout/investigations/02_relevance-labels/. "
    "Use cache.get(2) to load the existing S3 payload instead."
)

raise NotImplementedError(_MSG)
