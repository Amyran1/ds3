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


def test_preserves_non_ledger_v1_sentinels(tmp_path):
    """Row with _managed_by but different schema_version → NOT filtered (conservative match)."""
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {"_managed_by": "libs.ledgers.LedgerWriter", "schema_version": "ledger/v2"},
            {"run_id": "run_01", "primary_metric_value": 0.85},
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


def test_parity_with_leaderboard_read_jsonl(tmp_path):
    """Cross-check: render.read_jsonl and leaderboard._read_jsonl should return same number of rows.

    NOTE: leaderboard._read_jsonl filters ANY row with _managed_by (less conservative).
    render.read_jsonl filters only rows with BOTH _managed_by AND schema_version=="ledger/v1".
    With a pure ledger/v1 sentinel, they agree on 2 data rows. With the current corpus, counts match.
    """
    from libs.leaderboard import _read_jsonl as leaderboard_read

    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            {"_managed_by": "libs.ledgers.LedgerWriter", "schema_version": "ledger/v1"},
            {"run_id": "run_01", "primary_metric_value": 0.85},
            {"run_id": "run_02", "primary_metric_value": 0.90},
        ],
    )
    dashboard_rows = read_jsonl(p)
    leaderboard_rows = leaderboard_read(p)
    assert len(dashboard_rows) == len(leaderboard_rows)
