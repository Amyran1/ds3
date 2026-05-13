"""Project data accessors for civic_shout_action_rate_increase.

Thin facade over the project's entity caches. Each accessor exposes a uniform
interface:

    full(version=None)                  -> pl.DataFrame  (canonical pull)
    smoke_test(version=None)            -> pl.DataFrame  (first 100 rows by sort key, deterministic)
    sample_percentage(p, *, seed=42)    -> pl.DataFrame  (reproducible random sample)
    dictionary(version=None)            -> dict[str, str]
    date_range(version=None)            -> tuple[date, date]
    n_rows(version=None)                -> int           (lazy; parquet metadata only)
    by_date_range(start, end)           -> pl.DataFrame  (date-windowed slice, lazy)
    head(n)                             -> pl.DataFrame  (top-n peek, lazy)
    version (property)                  -> int

Use ``smoke_test`` for wiring checks (deterministic, tiny) and ``sample_percentage``
for distribution-faithful subsets.

Usage:
    from projects.civic_shout_action_rate_increase.data import emails, user_emails
    user_emails.smoke_test()
    emails.sample_percentage(0.05, seed=42)
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import polars as pl

from entities.civic_shout_emails.cache import cache as _emails_cache
from entities.civic_shout_user_emails.cache import cache as _user_emails_cache
from libs.cache.entity_cache import EntityCache

EMAILS_VERSION: int = 1
USER_EMAILS_VERSION: int = 3


class _DataAccessor:
    _CACHE: ClassVar[EntityCache]
    _DEFAULT_VERSION: ClassVar[int]
    _SORT_KEY: ClassVar[list[str]]
    _DATE_COL: ClassVar[str]

    @property
    def version(self) -> int:
        return self._DEFAULT_VERSION

    def _v(self, version: int | None) -> int:
        return self._DEFAULT_VERSION if version is None else version

    def full(self, version: int | None = None) -> pl.DataFrame:
        return self._CACHE.get(self._v(version))

    def smoke_test(self, version: int | None = None) -> pl.DataFrame:
        return self._CACHE.scan(self._v(version)).sort(self._SORT_KEY).limit(100).collect()

    def sample_percentage(
        self,
        fraction: float,
        *,
        seed: int = 42,
        version: int | None = None,
    ) -> pl.DataFrame:
        df = self._CACHE.get(self._v(version))
        return df.sample(fraction=fraction, seed=seed, shuffle=False)

    def dictionary(self, version: int | None = None) -> dict[str, str]:
        return self._CACHE.data_dictionary(self._v(version))

    def date_range(self, version: int | None = None) -> tuple[date, date]:
        row = (
            self._CACHE.scan(self._v(version))
            .select(
                pl.col(self._DATE_COL).min().alias("min"),
                pl.col(self._DATE_COL).max().alias("max"),
            )
            .collect()
            .row(0)
        )
        return row[0], row[1]

    def n_rows(self, version: int | None = None) -> int:
        return self._CACHE.scan(self._v(version)).select(pl.len()).collect().item()

    def by_date_range(
        self,
        start: date,
        end: date,
        *,
        version: int | None = None,
    ) -> pl.DataFrame:
        return (
            self._CACHE.scan(self._v(version))
            .filter((pl.col(self._DATE_COL) >= start) & (pl.col(self._DATE_COL) <= end))
            .collect()
        )

    def head(self, n: int, *, version: int | None = None) -> pl.DataFrame:
        return self._CACHE.scan(self._v(version)).limit(n).collect()


class _EmailsAccessor(_DataAccessor):
    _CACHE = _emails_cache
    _DEFAULT_VERSION = EMAILS_VERSION
    _SORT_KEY = ["email_id"]
    _DATE_COL = "date_sent"


class _UserEmailsAccessor(_DataAccessor):
    _CACHE = _user_emails_cache
    _DEFAULT_VERSION = USER_EMAILS_VERSION
    _SORT_KEY = ["user_id", "email_id"]
    _DATE_COL = "date_sent"


emails: _EmailsAccessor = _EmailsAccessor()
user_emails: _UserEmailsAccessor = _UserEmailsAccessor()


__all__ = [
    "EMAILS_VERSION",
    "USER_EMAILS_VERSION",
    "emails",
    "user_emails",
]
