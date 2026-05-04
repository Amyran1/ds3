"""Build civic_shout_engagement__actions v2 from Redshift.

Anchored at BUILD_CUTOFF_UTC = 2026-04-21T00:00:00Z. Pulls petition
signatures for Action Network group 228691 in the half-open window
[BUILD_CUTOFF - 24mo, BUILD_CUTOFF). Only signatures for petitions
belonging to group 228691 are included (enforced by the sub-select on
``group_228691_indexed.petitions``).

Re-running this script on any future date produces byte-identical output
(modulo Redshift mirror state) because no date arithmetic uses current_date
or datetime.utcnow().

Usage::

    # Calibration probe only (1-month slice, writes calibration artifact):
    ENVIRONMENT=PRODUCTION python -m libs.dsrun --timeout 600 \
        entities/civic_shout_engagement/create_actions_v2.py \
        --calibrate-only

    # Full 24-month build:
    ENVIRONMENT=PRODUCTION python -m libs.dsrun --timeout 3600 \
        entities/civic_shout_engagement/create_actions_v2.py
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
from entities.civic_shout_engagement.actions_cache import cache as actions_cache
from libs.progress_tracking.progress import ProgressReporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build anchor -- fixed UTC cutoff for full reproducibility.
# ---------------------------------------------------------------------------
UTC = timezone.utc
BUILD_CUTOFF_UTC = datetime(2026, 4, 21, tzinfo=UTC)

_N_MONTHS = 24
_TOTAL_ITEMS_EST = 20_000_000  # ~1M rows/month x 24

# Calibration constants
_SEQUENTIAL_PLAN_RATE = 100_000  # rows/sec -- conservative estimate
_WALL_CLOCK_GATE_MIN = 30

_CALIBRATION_DIR = Path("entities/civic_shout_engagement/analysis")
_CALIBRATION_ARTIFACT = _CALIBRATION_DIR / "calibration-actions-v2.md"

# Validation bounds (hard fail)
_HARD_MIN_ROWS = 5_000_000
_HARD_MAX_ROWS = 100_000_000

# Null-drop tolerance: if more than this fraction of rows have null keys,
# treat it as an upstream data quality failure rather than normal noise.
_NULL_DROP_TOLERANCE = 0.01  # 1%

# Key columns that must be non-null in the final cache payload.
_KEY_COLS = ("signature_id", "user_id", "petition_id")

# SQL template -- {start_ts} and {end_ts} are formatted datetime strings
# produced internally; no external input is ever interpolated here.
_SQL_TEMPLATE = (
    "SELECT id AS signature_id, user_id, petition_id, created_at"
    " FROM group_228691_indexed.signatures"
    " WHERE petition_id IN ("
    "    SELECT id FROM group_228691_indexed.petitions"
    "    WHERE group_id = 228691"
    " )"
    "  AND created_at >= TIMESTAMP '{start_ts}'"
    "  AND created_at <  TIMESTAMP '{end_ts}'"
)


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------


def _build_full_sql(window_start: datetime, window_end: datetime) -> str:
    """Return the SQL for the full 24-month pull."""
    start_ts = window_start.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = window_end.strftime("%Y-%m-%d %H:%M:%S")
    return _SQL_TEMPLATE.replace("{start_ts}", start_ts).replace("{end_ts}", end_ts)


def _build_probe_sql() -> str:
    """Return the SQL for the 1-month calibration probe."""
    window_start = BUILD_CUTOFF_UTC - relativedelta(months=1)
    return _build_full_sql(window_start, BUILD_CUTOFF_UTC)


# ---------------------------------------------------------------------------
# Calibration probe
# ---------------------------------------------------------------------------


def _write_calibration_artifact(
    *,
    probe_rows: int,
    probe_elapsed_s: float,
    measured_rate: float,
    est_min: float,
) -> None:
    """Write calibration Markdown to *_CALIBRATION_ARTIFACT*."""
    _CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    probe_start = (BUILD_CUTOFF_UTC - relativedelta(months=1)).isoformat()
    gate_result = "PASS" if est_min <= _WALL_CLOCK_GATE_MIN else "FAIL"
    lines = [
        "# actions v2 -- calibration probe",
        "",
        f"- **Probe window**: `{probe_start}` -> `{BUILD_CUTOFF_UTC.isoformat()}`",
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
    staging_dir = Path(tempfile.mkdtemp(prefix="cse_act_v2_probe_"))
    try:
        sql = _build_probe_sql()
        probe_csv = staging_dir / "probe.csv"

        logger.info("Calibration probe: fetching 1-month slice for actions v2")
        t0 = time.monotonic()
        run_psql_to_csv(sql, probe_csv)
        probe_elapsed_s = time.monotonic() - t0

        probe_df = pl.read_csv(probe_csv, try_parse_dates=True)
        probe_rows = len(probe_df)
        measured_rate = probe_rows / probe_elapsed_s if probe_elapsed_s > 0 else 0.0
        est_min = (probe_elapsed_s / 60.0) * _N_MONTHS

        logger.info(
            "Probe: %d rows in %.1fs -> %.0f rows/sec | proj. 24mo: %.1f min",
            probe_rows,
            probe_elapsed_s,
            measured_rate,
            est_min,
        )

        _write_calibration_artifact(
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
# Null-key filter
# ---------------------------------------------------------------------------


def _drop_null_key_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with null values in any key column; warn and gate on fraction.

    Null user_id / signature_id / petition_id rows are anonymous petition
    signatures from the Action Network source table.  The production adapter
    (psycopg2_redshift_source.py) skips these rows by returning None from
    the row mapper.  We match that behaviour here: drop with a warning rather
    than aborting.

    A null-drop fraction above _NULL_DROP_TOLERANCE (1%) is treated as an
    upstream data quality failure and causes sys.exit(1).

    Args:
        df: Raw ingested DataFrame before validation.

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

        keep_mask = (
            pl.col("signature_id").is_not_null()
            & pl.col("user_id").is_not_null()
            & pl.col("petition_id").is_not_null()
        )
        df = df.filter(keep_mask)
        logger.info("After null-key filter: %d rows", len(df))

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(df: pl.DataFrame) -> None:
    """Validate the assembled DataFrame before writing to cache.

    Called after _drop_null_key_rows() so the null check here is a
    belt-and-suspenders assertion — it should always pass 0 nulls.

    Raises:
        SystemExit: Hard bounds violated or nulls found on key columns.
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

    for col_name in _KEY_COLS:
        null_count = df[col_name].null_count()
        if null_count != 0:
            logger.error(
                "POST-FILTER ASSERTION: %s has %d nulls -- aborting.",
                col_name,
                null_count,
            )
            sys.exit(1)

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
    """Pull actions (signatures) from Redshift and write to the v2 entity cache.

    Args:
        calibrate_only: When True, run only the 1-month calibration probe and
            exit.  The calibration artifact is always written before any
            sys.exit call.
    """
    if calibrate_only:
        run_calibration_probe()
        logger.info("Calibrate-only mode complete. Exiting.")
        sys.exit(0)

    staging_dir = Path(tempfile.mkdtemp(prefix="cse_act_v2_"))
    logger.info("Staging directory: %s", staging_dir)

    window_start = BUILD_CUTOFF_UTC - relativedelta(months=_N_MONTHS)
    window_end = BUILD_CUTOFF_UTC
    sql = _build_full_sql(window_start, window_end)
    raw_csv = staging_dir / "actions_v2.csv"

    reporter = ProgressReporter("cse-actions-v2", total=_TOTAL_ITEMS_EST)

    try:
        logger.info(
            "Phase A: pulling actions from Redshift -> %s (window %s -> %s)",
            raw_csv,
            window_start.isoformat(),
            window_end.isoformat(),
        )
        t0 = time.monotonic()
        try:
            run_psql_to_csv(sql, raw_csv)
        except Exception:
            logger.exception("Redshift pull FAILED. Aborting.")
            reporter.error("Redshift pull failed")
            sys.exit(1)

        elapsed_s = time.monotonic() - t0
        csv_size = raw_csv.stat().st_size
        logger.info(
            "Phase A: done in %.1fs. CSV size: %d bytes",
            elapsed_s,
            csv_size,
        )

        logger.info("Phase B: ingesting %s", raw_csv)
        df = (
            pl.scan_csv(raw_csv, try_parse_dates=True)
            .select(
                pl.col("signature_id").cast(pl.Int64),
                pl.col("user_id").cast(pl.Int64),
                pl.col("petition_id").cast(pl.Int64),
                pl.col("created_at").cast(pl.Datetime("us", "UTC")),
            )
            .collect(engine="streaming")
        )

        # Drop rows with null key columns (e.g. anonymous petition signatures)
        # before validation.  Aborts if dropped fraction exceeds 1%.
        df = _drop_null_key_rows(df)

        reporter.update(len(df))
        _validate(df)
        actions_cache.put(2, df)
        reporter.complete()
        logger.info("actions v2 written to cache.")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info("Staging directory removed: %s", staging_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build civic_shout_engagement__actions v2.",
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
