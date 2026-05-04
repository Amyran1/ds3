r"""Build civic_shout_engagement__email_activities v2 from Redshift.

Anchored at BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z. Pulls all event types
(sent, open, click, verified_open, etc.) for Action Network group 228691
email activities in the half-open window [BUILD_CUTOFF - 24mo, BUILD_CUTOFF).

Data is fetched in 24 monthly chunks so that no single psql query runs for
multiple hours. Each chunk writes a staging CSV; after all 24 fetches succeed
the CSVs are concatenated, typed, validated, and written to the v2 entity
cache. Staging CSVs are deleted after the parquet write succeeds.

Re-running this script on any future date produces byte-identical output
(modulo Redshift mirror state) because no date arithmetic uses current_date
or datetime.utcnow().

Usage::

    # Calibration probe only (1-month slice, writes calibration artifact):
    ENVIRONMENT=PRODUCTION python -m libs.dsrun --timeout 600 \
        entities/civic_shout_engagement/create_email_activities_v2.py \
        --calibrate-only

    # Full 24-month build:
    ENVIRONMENT=PRODUCTION python -m libs.dsrun --timeout 18000 \
        entities/civic_shout_engagement/create_email_activities_v2.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta

from entities.civic_shout_engagement._psql import run_psql_to_csv
from entities.civic_shout_engagement.email_activities_cache import (
    cache as email_activities_cache,
)
from libs.progress_tracking.progress import ProgressReporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build anchor -- fixed UTC cutoff for full reproducibility.
# ---------------------------------------------------------------------------
UTC = timezone.utc
BUILD_CUTOFF_UTC = datetime(2026, 4, 21, tzinfo=UTC)

_N_MONTHS = 24
_TOTAL_ITEMS_EST = 300_000_000  # ~12M rows/month x 24

# Calibration constants (see ds-planning.md Sequential vs Parallel Plan Rate)
_SEQUENTIAL_PLAN_RATE = 500_000  # rows/sec -- conservative Redshift CSV stream estimate
_WALL_CLOCK_GATE_MIN = 90

_CALIBRATION_DIR = Path("entities/civic_shout_engagement/analysis")
_CALIBRATION_ARTIFACT = _CALIBRATION_DIR / "calibration-email_activities-v2.md"

# Validation bounds (soft warn / hard fail)
_SOFT_MIN_ROWS = 100_000_000
_SOFT_MAX_ROWS = 500_000_000
_HARD_MIN_ROWS = 50_000_000
_HARD_MAX_ROWS = 800_000_000

_EXPECTED_ACTION_TYPES = {"sent", "open", "click", "verified_open"}

# Null-drop tolerance: if more than this fraction of rows have null keys,
# treat it as an upstream data quality failure rather than normal noise.
_NULL_DROP_TOLERANCE = 0.01  # 1%

# Key columns that must be non-null in the final cache payload.
_KEY_COLS = ("email_id", "user_id")

# SQL template -- {start_ts} and {end_ts} are formatted datetime strings
# produced internally; no external input is ever interpolated here.
_SQL_TEMPLATE = (
    "SELECT id, email_id, recipient_id AS user_id, action_type, created_at"
    " FROM group_228691_indexed.email_activities_16"
    " WHERE created_at >= TIMESTAMP '{start_ts}'"
    "   AND created_at <  TIMESTAMP '{end_ts}'"
)


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------


def _build_chunk_sql(window_start: datetime, window_end: datetime) -> str:
    """Return the SQL for one monthly chunk using half-open [start, end) bounds."""
    start_ts = window_start.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = window_end.strftime("%Y-%m-%d %H:%M:%S")
    return _SQL_TEMPLATE.replace("{start_ts}", start_ts).replace("{end_ts}", end_ts)


# ---------------------------------------------------------------------------
# Calibration probe
# ---------------------------------------------------------------------------


def _write_calibration_artifact(
    *,
    window_start: datetime,
    window_end: datetime,
    probe_rows: int,
    probe_elapsed_s: float,
    measured_rate: float,
    est_min: float,
) -> None:
    """Write calibration Markdown to *_CALIBRATION_ARTIFACT*."""
    _CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    gate_result = "PASS" if est_min <= _WALL_CLOCK_GATE_MIN else "FAIL"
    probe_window = f"`{window_start.isoformat()}` -> `{window_end.isoformat()}`"
    lines = [
        "# email_activities v2 -- calibration probe",
        "",
        f"- **Probe window**: {probe_window}",
        f"- **Probe rows fetched**: {probe_rows:,}",
        f"- **Probe elapsed**: {probe_elapsed_s:.1f}s",
        f"- **Measured rate**: {measured_rate:,.0f} rows/sec",
        f"- **Sequential plan rate**: {_SEQUENTIAL_PLAN_RATE:,} rows/sec",
        f"- **Projected 24-month wall-clock**: {est_min:.1f} min",
        f"- **Wall-clock gate**: {_WALL_CLOCK_GATE_MIN} min",
        f"- **Gate result**: {gate_result}",
        "",
        f"_Build anchor_: `{BUILD_CUTOFF_UTC.isoformat()}`",
    ]
    _CALIBRATION_ARTIFACT.write_text("\n".join(lines) + "\n")
    logger.info("Calibration artifact written to %s", _CALIBRATION_ARTIFACT)


def run_calibration_probe() -> None:
    """Fetch 1-month probe, measure throughput, write artifact, gate check."""
    staging_dir = Path(tempfile.mkdtemp(prefix="cse_ea_v2_probe_"))
    try:
        window_end = BUILD_CUTOFF_UTC
        window_start = BUILD_CUTOFF_UTC - relativedelta(months=1)
        sql = _build_chunk_sql(window_start, window_end)
        probe_csv = staging_dir / "probe.csv"

        logger.info(
            "Calibration probe: fetching %s -> %s",
            window_start.isoformat(),
            window_end.isoformat(),
        )
        t0 = time.monotonic()
        run_psql_to_csv(sql, probe_csv)
        probe_elapsed_s = time.monotonic() - t0

        probe_df = pl.read_csv(probe_csv, try_parse_dates=True)
        probe_rows = len(probe_df)
        measured_rate = probe_rows / probe_elapsed_s if probe_elapsed_s > 0 else 0.0
        # Wall-clock projects as: (probe_elapsed / 1 month) x 24 months
        est_min = (probe_elapsed_s / 60.0) * _N_MONTHS

        logger.info(
            "Probe: %d rows in %.1fs -> %.0f rows/sec | proj. 24mo: %.1f min",
            probe_rows,
            probe_elapsed_s,
            measured_rate,
            est_min,
        )

        _write_calibration_artifact(
            window_start=window_start,
            window_end=window_end,
            probe_rows=probe_rows,
            probe_elapsed_s=probe_elapsed_s,
            measured_rate=measured_rate,
            est_min=est_min,
        )

        if est_min > _WALL_CLOCK_GATE_MIN:
            logger.error(
                "WALL-CLOCK GATE FAIL: %.0f min projected (limit %d min). "
                "See %s for details. Optimize before running full build.",
                est_min,
                _WALL_CLOCK_GATE_MIN,
                _CALIBRATION_ARTIFACT,
            )
            sys.exit(1)

        logger.info(
            "Calibration gate PASS (%.1f min < %d min limit).",
            est_min,
            _WALL_CLOCK_GATE_MIN,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Action-type breakdown helper
# ---------------------------------------------------------------------------


def _action_type_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    """Return action_type value counts sorted descending by n."""
    counts = df.group_by("action_type").agg(pl.len().alias("n"))
    return counts.sort("n", descending=True)


# ---------------------------------------------------------------------------
# Null-key filter
# ---------------------------------------------------------------------------


def _drop_null_key_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with null values in any key column; warn and gate on fraction.

    Null email_id / user_id rows can arise from upstream source table gaps.
    The production adapter skips such rows by returning None from the row
    mapper.  We match that behaviour here: drop with a warning rather than
    aborting.

    A null-drop fraction above _NULL_DROP_TOLERANCE (1%) is treated as an
    upstream data quality failure and causes sys.exit(1).

    Args:
        df: Assembled DataFrame before validation.

    Returns:
        DataFrame with null-key rows removed.

    Raises:
        SystemExit: Dropped fraction exceeds _NULL_DROP_TOLERANCE.
    """
    n_before = len(df)
    null_counts = {col: int(df[col].null_count()) for col in _KEY_COLS}
    total_null_rows = sum(null_counts.values())

    if total_null_rows > 0:
        null_pct = 100.0 * total_null_rows / n_before if n_before > 0 else 0.0
        per_col_parts: list[str] = []
        for col, count in null_counts.items():
            if count > 0:
                per_col_parts.append(f"{col}={count}")
        per_col = ", ".join(per_col_parts)
        logger.warning(
            "Dropping %d rows with null key columns (%.4f%% of %d): %s",
            total_null_rows,
            null_pct,
            n_before,
            per_col,
        )

        tolerance_pct = _NULL_DROP_TOLERANCE * 100.0
        if null_pct > tolerance_pct:
            logger.error(
                "ABORT: null-key drop fraction %.4f%% exceeds tolerance %.1f%%. "
                "Upstream data quality failure -- investigate before proceeding.",
                null_pct,
                tolerance_pct,
            )
            sys.exit(1)

        keep_mask = pl.col("email_id").is_not_null() & pl.col("user_id").is_not_null()
        df = df.filter(keep_mask)
        logger.info("After null-key filter: %d rows", len(df))

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(df: pl.DataFrame) -> None:
    """Validate the assembled DataFrame before writing to cache.

    Called after _drop_null_key_rows() so the null checks here are
    belt-and-suspenders assertions -- they should always pass 0 nulls.

    Raises:
        SystemExit: Hard bounds violated (row count or null check).
    """
    n = len(df)
    logger.info("Validation: %d rows total", n)

    if n < _HARD_MIN_ROWS or n > _HARD_MAX_ROWS:
        logger.error(
            "HARD BOUND VIOLATION: %d rows outside [%d, %d]. Aborting.",
            n,
            _HARD_MIN_ROWS,
            _HARD_MAX_ROWS,
        )
        sys.exit(1)

    if n < _SOFT_MIN_ROWS or n > _SOFT_MAX_ROWS:
        logger.warning(
            "Soft bound: %d rows outside expected [%d, %d]. Proceeding.",
            n,
            _SOFT_MIN_ROWS,
            _SOFT_MAX_ROWS,
        )

    email_id_nulls = df["email_id"].null_count()
    if email_id_nulls != 0:
        logger.error(
            "POST-FILTER ASSERTION: email_id has %d nulls -- aborting.",
            email_id_nulls,
        )
        sys.exit(1)

    user_id_nulls = df["user_id"].null_count()
    if user_id_nulls != 0:
        logger.error(
            "POST-FILTER ASSERTION: user_id has %d nulls -- aborting.",
            user_id_nulls,
        )
        sys.exit(1)

    actual_types = set(df["action_type"].cast(pl.String).unique().to_list())
    missing_types = _EXPECTED_ACTION_TYPES - actual_types
    if missing_types:
        logger.warning(
            "action_type missing expected values: %s (found: %s)",
            missing_types,
            actual_types,
        )

    min_created = df["created_at"].min()
    max_created = df["created_at"].max()
    logger.info(
        "created_at range: %s -> %s (expected ~%s -> %s)",
        min_created,
        max_created,
        (BUILD_CUTOFF_UTC - relativedelta(months=_N_MONTHS)).isoformat(),
        BUILD_CUTOFF_UTC.isoformat(),
    )


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def main(*, calibrate_only: bool = False) -> None:
    """Pull email_activities from Redshift and write to the v2 entity cache.

    Args:
        calibrate_only: When True, run only the 1-month calibration probe and
            exit.  The calibration artifact is always written before any
            sys.exit call.
    """
    if calibrate_only:
        run_calibration_probe()
        logger.info("Calibrate-only mode complete. Exiting.")
        sys.exit(0)

    staging_dir = Path(tempfile.mkdtemp(prefix="cse_ea_v2_"))
    logger.info("Staging directory: %s", staging_dir)

    reporter = ProgressReporter("cse-email-activities-v2", total=_TOTAL_ITEMS_EST)
    csv_paths: list[Path] = []

    try:
        for i in range(_N_MONTHS):
            window_end = BUILD_CUTOFF_UTC - relativedelta(months=i)
            window_start = BUILD_CUTOFF_UTC - relativedelta(months=i + 1)
            sql = _build_chunk_sql(window_start, window_end)
            csv_path = staging_dir / f"ea_month_{i:02d}.csv"

            logger.info(
                "Chunk %02d/%02d: fetching %s -> %s -> %s",
                i + 1,
                _N_MONTHS,
                window_start.isoformat(),
                window_end.isoformat(),
                csv_path.name,
            )
            t0 = time.monotonic()
            try:
                run_psql_to_csv(sql, csv_path)
            except Exception:
                logger.exception(
                    "Chunk %02d FAILED (window %s -> %s). Aborting.",
                    i + 1,
                    window_start.isoformat(),
                    window_end.isoformat(),
                )
                sys.exit(1)

            elapsed_s = time.monotonic() - t0
            chunk_df = pl.read_csv(csv_path, try_parse_dates=True)
            chunk_rows = len(chunk_df)
            del chunk_df  # free RAM; we re-read below for concat
            rows_per_sec = chunk_rows / elapsed_s if elapsed_s > 0 else 0.0
            logger.info(
                "Chunk %02d: %d rows in %.1fs (%.0f rows/sec)",
                i + 1,
                chunk_rows,
                elapsed_s,
                rows_per_sec,
            )
            reporter.update(chunk_rows)
            csv_paths.append(csv_path)

        # Assemble all monthly CSVs into a single DataFrame.
        logger.info("Concatenating %d monthly CSVs ...", len(csv_paths))
        monthly_frames = [pl.read_csv(p, try_parse_dates=True) for p in csv_paths]
        df = (
            pl.concat(monthly_frames)
            .rename({"id": "activity_id"})
            .select(
                pl.col("activity_id").cast(pl.Int64),
                pl.col("email_id").cast(pl.Int64),
                pl.col("user_id").cast(pl.Int64),
                pl.col("action_type").cast(pl.Categorical),
                pl.col("created_at").cast(pl.Datetime("us", "UTC")),
            )
        )
        del monthly_frames

        logger.info(
            "email_activities v2: %d rows | action_type breakdown:\n%s",
            len(df),
            _action_type_breakdown(df),
        )

        # Drop rows with null key columns before validation.
        # Aborts if dropped fraction exceeds 1%.
        df = _drop_null_key_rows(df)

        _validate(df)
        email_activities_cache.put(2, df)
        reporter.complete()
        logger.info("email_activities v2 written to cache.")

    finally:
        # Best-effort cleanup of staging CSVs after parquet write.
        shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info("Staging directory removed: %s", staging_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build civic_shout_engagement__email_activities v2.",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help=(
            "Run only the 1-month calibration probe, write the calibration "
            "artifact, and exit. Does not write to the entity cache."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(calibrate_only=args.calibrate_only)
