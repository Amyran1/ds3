from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from libs.autoloop.prereqs import check_prereqs


def _build_complete_tree(tmp_path: Path, project: str = "proj") -> Path:
    project_root = tmp_path
    proj_dir = project_root / "projects" / project

    (proj_dir / "GOAL.md").parent.mkdir(parents=True, exist_ok=True)
    (proj_dir / "GOAL.md").write_text("Increase action rate.")

    harness = proj_dir / "harness.py"
    harness.write_text(
        "def harness(data, feature_cols, ml_model_config, outcome_variable):\n    pass\n"
    )

    eda = proj_dir / "eda.py"
    eda.write_text("def eda(data, feature_cols, outcome_variable, eda_config):\n    pass\n")

    discovery = proj_dir / "discovery.py"
    discovery.write_text(
        "def discover_gaps(data, feature_cols, outcome_variable, prediction_cols, harness_response, discovery_config):\n    pass\n"
    )

    libs_dir = project_root / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)
    (libs_dir / "run_record.py").write_text("def run_record(*args, **kwargs):\n    pass\n")
    (libs_dir / "ledgers.py").write_text("class LedgerWriter:\n    pass\n")

    outcome_dir = proj_dir / "outcomes" / "converted" / "v1"
    outcome_dir.mkdir(parents=True)
    (outcome_dir / "dictionary.yaml").write_text("outcome: converted\n")

    runs_dir = proj_dir / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "run_01.py").write_text("# baseline run\n")

    results = runs_dir / "results.jsonl"
    results.write_text(
        json.dumps({"status": "completed", "run_id": "run_01", "roc_auc": 0.75}) + "\n"
    )

    autoloop_dir = proj_dir / "autoloop"
    autoloop_dir.mkdir(parents=True)
    (autoloop_dir / "config.yaml").write_text(
        "project: proj\noutcome_variable: y\ncomparison_group: cg\nprimary_metric: roc_auc\n"
    )

    return project_root


def _mock_git_clean(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, dirty: bool = False
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "git" in cmd and "status" in cmd:
            stdout = " M projects/proj/harness.py\n" if dirty else ""
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")
        stdout = ""
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_all_prereqs_pass_when_tree_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root, dirty=False)

    report = check_prereqs(project_root, "proj")

    assert report.all_passed, [f.name + ": " + f.detail for f in report.failed]


def test_missing_goal_md_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "GOAL.md").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    failed_names = [f.name for f in report.failed]
    assert "GOAL.md present" in failed_names


def test_missing_harness_py_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "harness.py").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("harness.py" in f.name for f in report.failed)


def test_harness_py_without_def_harness_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "harness.py").write_text("# no function here\n")

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    harness_failures = [f for f in report.failed if "harness.py" in f.name]
    assert harness_failures


def test_missing_eda_py_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "eda.py").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("eda.py" in f.name for f in report.failed)


def test_missing_discovery_py_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "discovery.py").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("discovery.py" in f.name for f in report.failed)


def test_missing_run_record_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "libs" / "run_record.py").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("run_record" in f.name for f in report.failed)


def test_missing_outcomes_directory_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    outcomes_dir = project_root / "projects" / "proj" / "outcomes"
    import shutil

    shutil.rmtree(outcomes_dir)

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("outcomes" in f.name for f in report.failed)


def test_missing_run_01_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "runs" / "run_01.py").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("run_01" in f.name for f in report.failed)


def test_results_jsonl_no_completed_row_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    results = project_root / "projects" / "proj" / "runs" / "results.jsonl"
    results.write_text(json.dumps({"status": "failed", "run_id": "run_01"}) + "\n")

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("results.jsonl" in f.name for f in report.failed)


def test_missing_autoloop_config_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "autoloop" / "config.yaml").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("config.yaml" in f.name for f in report.failed)


def test_dirty_git_tree_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root, dirty=True)

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert any("git" in f.name for f in report.failed)


def test_multiple_failures_all_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "GOAL.md").unlink()
    (project_root / "projects" / "proj" / "eda.py").unlink()

    report = check_prereqs(project_root, "proj")

    assert not report.all_passed
    assert len(report.failed) >= 2


def test_format_table_no_color_is_plain_ascii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "GOAL.md").unlink()

    report = check_prereqs(project_root, "proj")
    table = report.format_table(use_color=False)

    assert "\033[" not in table
    assert "[FAIL]" in table
    assert "[OK]" in table


def test_format_table_with_color_contains_ansi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _build_complete_tree(tmp_path)
    _mock_git_clean(monkeypatch, project_root)
    (project_root / "projects" / "proj" / "GOAL.md").unlink()

    report = check_prereqs(project_root, "proj")
    table = report.format_table(use_color=True)

    assert "\033[" in table
