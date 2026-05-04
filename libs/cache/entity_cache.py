"""Entity cache with local disk and S3 backing store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

import polars as pl

from libs.cache import s3

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "chorus-content-assets"

DFBackend = Literal["polars", "pandas"]
_DEFAULT_BACKEND: DFBackend = "polars"


@dataclass(frozen=True)
class VersionMeta:
    """Metadata for a single version of a cached entity."""

    key: str  # e.g. "example_entity_v1"
    fmt: str = "parquet"  # "parquet" or "csv"


def _read_pandas(
    path: Path,
    meta: VersionMeta,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Read via pandas and convert to polars (backward compat)."""
    import pandas as _pd

    pdf = (
        _pd.read_parquet(path)
        if meta.fmt == "parquet"
        else _pd.read_csv(path)
    )
    if columns is not None:
        pdf = _pd.DataFrame(pdf)[columns]
    return pl.from_pandas(_pd.DataFrame(pdf))


def _write_pandas(df: pl.DataFrame, path: Path, fmt: str) -> None:
    """Convert to pandas and write (backward compat for old extractors)."""
    pdf = df.to_pandas()
    if fmt == "parquet":
        pdf.to_parquet(path, index=False)
    else:
        pdf.to_csv(path, index=False)


class EntityCache:
    """Single-entity cache backed by local disk and S3."""

    def __init__(
        self,
        entity: str,
        s3_prefix: str,
        cache_dir: Path,
        versions: dict[int, VersionMeta],
        bucket: str = DEFAULT_BUCKET,
    ) -> None:
        self.entity = entity
        self.s3_prefix = s3_prefix
        self.cache_dir = cache_dir
        self.versions = versions
        self.bucket = bucket
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, version: int) -> VersionMeta:
        """Look up version metadata or raise ValueError."""
        if version not in self.versions:
            registered = sorted(self.versions)
            msg = (
                f"Version {version} not registered "
                f"for '{self.entity}'. "
                f"Registered: {registered}"
            )
            raise ValueError(msg)
        return self.versions[version]

    def _local_path(self, meta: VersionMeta) -> Path:
        return self.cache_dir / f"{meta.key}.{meta.fmt}"

    def _s3_key(self, meta: VersionMeta) -> str:
        return f"{self.s3_prefix}/{meta.key}.{meta.fmt}"

    def _read(
        self,
        path: Path,
        meta: VersionMeta,
        columns: list[str] | None = None,
        backend: DFBackend | None = None,
        *,
        streaming: bool = False,
    ) -> pl.DataFrame:
        backend = backend or _DEFAULT_BACKEND
        if backend == "pandas":
            return _read_pandas(path, meta, columns)
        if meta.fmt == "parquet":
            # scan_parquet enables projection pushdown (only reads
            # requested columns from disk) and predicate pushdown.
            lf = pl.scan_parquet(path)
            if columns is not None:
                lf = lf.select(columns)
            return lf.collect(engine="streaming" if streaming else "auto")
        # CSV doesn't support scan with projection pushdown
        df = pl.read_csv(path)
        if columns is not None:
            df = df.select(columns)
        return df

    def scan(
        self,
        version: int,
        *,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Return a LazyFrame for deferred execution.

        Callers chain operations on the LazyFrame and call ``.collect()``
        when ready. Only works for parquet-backed versions.
        """
        meta = self._resolve(version)
        local_path = self._ensure_local(meta)
        if meta.fmt != "parquet":
            msg = "scan() only supports parquet format"
            raise ValueError(msg)
        lf = pl.scan_parquet(local_path)
        if columns is not None:
            lf = lf.select(columns)
        return lf

    def _ensure_local(self, meta: VersionMeta) -> Path:
        """Ensure the file exists locally, pulling from S3 if needed."""
        local_path = self._local_path(meta)
        if local_path.exists():
            return local_path
        try:
            s3.download(
                self.bucket, self._s3_key(meta), local_path,
            )
        except Exception as exc:
            msg = f"Version {meta.key} not found locally or in S3"
            raise FileNotFoundError(msg) from exc
        logger.info(
            "Downloaded %s from S3", meta.key,
        )
        return local_path

    def get(
        self,
        version: int,
        *,
        columns: list[str] | None = None,
        backend: DFBackend | None = None,
        streaming: bool = False,
    ) -> pl.DataFrame:
        """Local hit -> return. Miss -> pull S3 -> store -> return.

        Args:
            version: Registered version number.
            columns: Columns to select (enables projection pushdown).
            backend: DataFrame backend — "polars" (default) or "pandas".
            streaming: Use polars streaming engine for large datasets.
        """
        meta = self._resolve(version)
        local_path = self._ensure_local(meta)
        logger.debug(
            "Cache hit: %s v%d at %s",
            self.entity, version, local_path,
        )
        return self._read(
            local_path, meta, columns, backend,
            streaming=streaming,
        )

    def put(self, version: int, df: pl.DataFrame) -> None:
        """Write local AND S3. Always both.

        Also accepts pandas DataFrames via duck-typing for backward
        compatibility with existing extractors.
        """
        meta = self._resolve(version)
        local_path = self._local_path(meta)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(df, "write_parquet"):
            # polars DataFrame
            if meta.fmt == "parquet":
                df.write_parquet(local_path)
            else:
                df.write_csv(local_path)
        else:
            # pandas DataFrame (backward compat)
            _write_pandas(df, local_path, meta.fmt)

        s3.upload(local_path, self.bucket, self._s3_key(meta))
        logger.info(
            "Saved %s v%d (%d rows) locally and to S3",
            self.entity, version, len(df),
        )

    def list_versions(self) -> list[int]:
        """Return registered version numbers, sorted."""
        return sorted(self.versions)

    def data_dictionary(self, version: int) -> dict[str, str]:
        """Return empty dict.

        Subclasses override or load from data_dictionary module.
        """
        self._resolve(version)  # validate version exists
        return {}
