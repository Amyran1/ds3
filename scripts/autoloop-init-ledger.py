#!/usr/bin/env python3
"""Prepend LedgerWriter sentinel headers to unmanaged project ledgers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SENTINEL_KEY = "_managed_by"
_SENTINEL_VALUE = "libs.ledgers.LedgerWriter"

_LEDGER_NAMES = [
    "results.jsonl",
    "eda_results.jsonl",
    "discovery_results.jsonl",
]


def _sentinel_line() -> str:
    return json.dumps(
        {
            _SENTINEL_KEY: _SENTINEL_VALUE,
            "schema_version": "ledger/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


def _is_managed(path: Path) -> bool:
    try:
        with path.open() as f:
            first_line = f.readline().strip()
        if not first_line:
            return False
        header = json.loads(first_line)
        return header.get(_SENTINEL_KEY) == _SENTINEL_VALUE
    except (json.JSONDecodeError, OSError):
        return False


def init_ledger(project_root: Path, project: str) -> dict[str, str]:
    runs_dir = project_root / "projects" / project / "runs"
    results: dict[str, str] = {}

    for name in _LEDGER_NAMES:
        path = runs_dir / name

        if not path.exists():
            print(f">> skipped (missing): {path}")
            results[name] = "skipped_missing"
            continue

        size = path.stat().st_size

        if size == 0:
            with path.open("w") as f:
                f.write(_sentinel_line() + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f">> initialized empty: {path}")
            results[name] = "initialized_empty"
            continue

        if _is_managed(path):
            print(f">> already managed: {path}")
            results[name] = "already_managed"
            continue

        existing = path.read_bytes()
        sentinel_bytes = (_sentinel_line() + "\n").encode()
        with path.open("wb") as f:
            f.write(sentinel_bytes)
            f.write(existing)
            f.flush()
            os.fsync(f.fileno())
        print(f">> prepended sentinel: {path}")
        results[name] = "prepended"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepend LedgerWriter sentinel to unmanaged project ledgers."
    )
    parser.add_argument(
        "project",
        nargs="?",
        help="Project name (positional)",
    )
    parser.add_argument(
        "--project",
        dest="project_flag",
        metavar="NAME",
        help="Project name (flag form)",
    )
    args = parser.parse_args()

    project = args.project_flag or args.project
    if not project:
        parser.error("project name is required (positional or --project)")

    project_root = Path(__file__).parent.parent
    try:
        init_ledger(project_root, project)
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
