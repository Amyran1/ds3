from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Callable

import pytest

_script_path = Path(__file__).parent.parent.parent / "scripts" / "autoloop-init-ledger.py"
_spec = importlib.util.spec_from_file_location("autoloop_init_ledger", _script_path)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
init_ledger: Callable[..., dict[str, str]] = _mod.init_ledger

_SENTINEL_KEY = "_managed_by"
_SENTINEL_VALUE = "libs.ledgers.LedgerWriter"

_FAKE_ROW = json.dumps(
    {
        "run_id": "run_01",
        "project": "test_project",
        "status": "completed",
    }
)


def _make_project_tree(
    tmp_path: Path, *, with_results: bool = True, empty_results: bool = False
) -> Path:
    runs_dir = tmp_path / "projects" / "test_project" / "runs"
    runs_dir.mkdir(parents=True)
    if with_results:
        results = runs_dir / "results.jsonl"
        if empty_results:
            results.touch()
        else:
            results.write_text(_FAKE_ROW + "\n")
    return tmp_path


def _is_managed(path: Path) -> bool:
    with path.open() as f:
        first = f.readline().strip()
    if not first:
        return False
    try:
        header = json.loads(first)
    except json.JSONDecodeError:
        return False
    return header.get(_SENTINEL_KEY) == _SENTINEL_VALUE


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_init_ledger_script_is_idempotent(tmp_path: Path) -> None:
    project_root = _make_project_tree(tmp_path)
    results_path = project_root / "projects" / "test_project" / "runs" / "results.jsonl"

    first = init_ledger(project_root, "test_project")
    assert first["results.jsonl"] == "prepended"
    assert _is_managed(results_path)

    hash_after_first = _content_hash(results_path)

    second = init_ledger(project_root, "test_project")
    assert second["results.jsonl"] == "already_managed"

    assert _content_hash(results_path) == hash_after_first


def test_init_ledger_skips_missing_files(tmp_path: Path) -> None:
    project_root = _make_project_tree(tmp_path, with_results=True)

    result = init_ledger(project_root, "test_project")

    assert result["results.jsonl"] in ("prepended", "already_managed")
    assert result["eda_results.jsonl"] == "skipped_missing"
    assert result["discovery_results.jsonl"] == "skipped_missing"

    assert not (project_root / "projects" / "test_project" / "runs" / "eda_results.jsonl").exists()
    assert not (
        project_root / "projects" / "test_project" / "runs" / "discovery_results.jsonl"
    ).exists()


def test_init_ledger_handles_empty_file(tmp_path: Path) -> None:
    project_root = _make_project_tree(tmp_path, with_results=True, empty_results=True)
    results_path = project_root / "projects" / "test_project" / "runs" / "results.jsonl"

    result = init_ledger(project_root, "test_project")
    assert result["results.jsonl"] == "initialized_empty"

    content = results_path.read_text()
    lines = [l for l in content.splitlines() if l.strip()]
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header.get(_SENTINEL_KEY) == _SENTINEL_VALUE
    assert content.endswith("\n")
