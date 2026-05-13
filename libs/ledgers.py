from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from libs.responses import DiscoveryResultRow, EDAResultRow, RunResultRow

logger = logging.getLogger(__name__)

LEDGER_SENTINEL_KEY = "_managed_by"
LEDGER_SENTINEL_VALUE = "libs.ledgers.LedgerWriter"


class LedgerError(Exception):
    pass


class LedgerWriter:
    """Append-only writer for the three project ledgers. R2 invariant: only this class writes."""

    def __init__(self, project_root: Path, project: str) -> None:
        self._project_root = project_root
        self._project = project

    @property
    def results_path(self) -> Path:
        return self._project_root / "projects" / self._project / "runs" / "results.jsonl"

    @property
    def eda_results_path(self) -> Path:
        return self._project_root / "projects" / self._project / "runs" / "eda_results.jsonl"

    @property
    def discovery_results_path(self) -> Path:
        return self._project_root / "projects" / self._project / "runs" / "discovery_results.jsonl"

    def _sentinel_line(self) -> str:
        import json

        return json.dumps(
            {
                LEDGER_SENTINEL_KEY: LEDGER_SENTINEL_VALUE,
                "schema_version": "ledger/v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _ensure_header(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w") as f:
                f.write(self._sentinel_line() + "\n")
                f.flush()
                os.fsync(f.fileno())
            return
        self.assert_managed(path)

    def _append_line(self, path: Path, line: str) -> None:
        with path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_result(self, row: RunResultRow) -> None:
        self._ensure_header(self.results_path)
        self._append_line(self.results_path, row.model_dump_json(by_alias=True))

    def append_eda_result(self, row: EDAResultRow) -> None:
        self._ensure_header(self.eda_results_path)
        self._append_line(self.eda_results_path, row.model_dump_json(by_alias=True))

    def append_discovery_result(self, row: DiscoveryResultRow) -> None:
        self._ensure_header(self.discovery_results_path)
        self._append_line(self.discovery_results_path, row.model_dump_json(by_alias=True))

    def append_atomic(
        self,
        result_row: RunResultRow,
        eda_row: EDAResultRow | None,
        discovery_row: DiscoveryResultRow | None,
    ) -> None:
        self._ensure_header(self.results_path)
        if eda_row is not None:
            self._ensure_header(self.eda_results_path)
        if discovery_row is not None:
            self._ensure_header(self.discovery_results_path)

        result_line = result_row.model_dump_json(by_alias=True)
        eda_line = eda_row.model_dump_json(by_alias=True) if eda_row is not None else None
        discovery_line = (
            discovery_row.model_dump_json(by_alias=True) if discovery_row is not None else None
        )

        try:
            self._append_line(self.results_path, result_line)
            if eda_line is not None:
                self._append_line(self.eda_results_path, eda_line)
            if discovery_line is not None:
                self._append_line(self.discovery_results_path, discovery_line)
        except OSError as e:
            raise LedgerError(f"Atomic append failed mid-write: {e}") from e

    @classmethod
    def assert_managed(cls, path: Path) -> None:
        with path.open() as f:
            first_line = f.readline().strip()
        if not first_line:
            raise LedgerError(f"Ledger {path} is missing the managed-by sentinel header")
        import json

        try:
            header = json.loads(first_line)
        except json.JSONDecodeError as e:
            raise LedgerError(f"Ledger {path} first line is not valid JSON: {e}") from e
        if header.get(LEDGER_SENTINEL_KEY) != LEDGER_SENTINEL_VALUE:
            raise LedgerError(
                f"Ledger {path} was not created by LedgerWriter "
                f"(got {header.get(LEDGER_SENTINEL_KEY)!r})"
            )
