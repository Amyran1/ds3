"""Populate scout_emails cache v1.

Source: MongoDB ``analysis.emails`` collection.

The full extraction logic requires a live MongoDB connection and utilities
from the backend monorepo. This stub documents the data source; the
complete extraction implementation lives at::

    dev/backend/datascience/extractors/emails/create_cache.py

The v1 parquet is already in S3 at::

    s3://chorus-content-assets/data-science/extractors/emails/emails_all.parquet

``cache.get(1)`` is the normal consumer path and will download it on first access.

Usage:
    PYTHONPATH=$(pwd) python -m libs.dsrun entities/scout_emails/create_cache_v1.py
"""

from __future__ import annotations

import logging
from typing import NoReturn

logger = logging.getLogger(__name__)


def main() -> NoReturn:
    """Raise NotImplementedError — extraction requires the backend monorepo.

    Use ``cache.get(1)`` to load the existing S3 payload instead.
    """
    logging.basicConfig(level=logging.INFO)
    msg = (
        "Requires MongoDB analysis.emails collection and utilities from the "
        "backend monorepo. See dev/backend/datascience/extractors/emails/"
        "create_cache.py for the complete implementation. "
        "Use cache.get(1) to load the existing S3 payload instead."
    )
    raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
