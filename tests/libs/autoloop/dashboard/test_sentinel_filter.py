import json
from pathlib import Path

from libs.autoloop.dashboard.render import read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_filters_ledger_sentinel(tmp_path):
    """Sentinel row with _managed_by + schema_version=ledger/v1 is filtered out."""
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {
                "_managed_by": "libs.ledgers.LedgerWriter",
                "schema_version": "ledger/v1",
                "created_at_utc": "2026-05-15T00:00:00",
            },
            {"run_id": "run_01", "primary_metric_value": 0.85},
            {"run_id": "run_02", "primary_metric_value": 0.90},
        ],
    )
    rows = read_jsonl(p)
    assert len(rows) == 2
    assert all("run_id" in r for r in rows)
    assert all("_managed_by" not in r for r in rows)


def test_filters_future_ledger_versions(tmp_path):
    """Row with _managed_by + schema_version='ledger/v2' (future) → FILTERED (forward-compatible)."""
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {"_managed_by": "libs.ledgers.LedgerWriter", "schema_version": "ledger/v2"},
            {"run_id": "run_01", "primary_metric_value": 0.85},
        ],
    )
    rows = read_jsonl(p)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_01"


def test_managed_by_without_schema_version_not_filtered(tmp_path):
    """Row with _managed_by present but no schema_version key — NOT filtered (defensive).

    Documents the conservative contract: only filter rows that explicitly
    self-identify as ledger/* schema. A malformed sentinel (or future-schema-without-version)
    is shown rather than silently dropped.
    """
    p = tmp_path / "results.jsonl"
    _write_jsonl(p, [{"_managed_by": "libs.ledgers.LedgerWriter"}])
    rows = read_jsonl(p)
    assert len(rows) == 1


def test_preserves_non_ledger_sentinels(tmp_path):
    """Row with _managed_by + schema_version='manifest/v1' (different namespace) → NOT filtered."""
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {"_managed_by": "some_other_tool", "schema_version": "manifest/v1"},
            {"run_id": "run_01"},
        ],
    )
    rows = read_jsonl(p)
    assert len(rows) == 2


def test_preserves_real_rows_with_managed_by_key(tmp_path):
    """Hypothetical real row with _managed_by as metadata (not the sentinel form) → NOT filtered."""
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {"run_id": "run_01", "_managed_by": "human_audit_tool", "primary_metric_value": 0.85},
        ],
    )
    rows = read_jsonl(p)
    assert len(rows) == 1


def test_empty_file(tmp_path):
    """Empty file → empty result."""
    p = tmp_path / "results.jsonl"
    p.touch()
    rows = read_jsonl(p)
    assert rows == []


def test_malformed_lines_skipped(tmp_path):
    """Lines that are not valid JSON are skipped gracefully."""
    p = tmp_path / "results.jsonl"
    p.write_text("not json\n" + json.dumps({"run_id": "run_01"}) + "\n")
    rows = read_jsonl(p)
    assert len(rows) == 1


def test_filters_standard_sentinel_count(tmp_path):
    """Standard ledger/v1 sentinel + 2 real rows → 2 rows returned."""
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {"_managed_by": "libs.ledgers.LedgerWriter", "schema_version": "ledger/v1"},
            {"run_id": "run_01", "primary_metric_value": 0.85},
            {"run_id": "run_02", "primary_metric_value": 0.90},
        ],
    )
    rows = read_jsonl(p)
    assert len(rows) == 2
