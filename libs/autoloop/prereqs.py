from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError


@dataclass
class PrereqResult:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str = ""
    fix_hint: str = ""

    @property
    def passed(self) -> bool:
        return self.status in ("ok", "warn")


@dataclass
class PrereqReport:
    passed: list[PrereqResult] = field(default_factory=list)
    warned: list[PrereqResult] = field(default_factory=list)
    failed: list[PrereqResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0

    def format_table(self, use_color: bool = False) -> str:
        lines: list[str] = []
        green = "\033[32m" if use_color else ""
        yellow = "\033[33m" if use_color else ""
        red = "\033[31m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        all_results = self.passed + self.warned + self.failed
        all_results.sort(key=lambda r: r.name)

        for result in all_results:
            if result.status == "ok":
                lines.append(f"{green}[OK]  {reset} {result.name}")
            elif result.status == "warn":
                lines.append(f"{yellow}[WARN]{reset} {result.name}")
                if result.detail:
                    lines.append(f"       {result.detail}")
            else:
                lines.append(f"{red}[FAIL]{reset} {result.name}")
                if result.detail:
                    lines.append(f"       {result.detail}")
                if result.fix_hint:
                    lines.append(f"       → {result.fix_hint}")

        n_passed = len(self.passed)
        n_warned = len(self.warned)
        n_failed = len(self.failed)
        total = n_passed + n_warned + n_failed
        lines.append("")
        if n_warned:
            lines.append(
                f"{total} checks: {n_passed} passed, {n_warned} warned, {n_failed} failed."
            )
        else:
            lines.append(f"{total} checks: {n_passed} passed, {n_failed} failed.")
        return "\n".join(lines)


def _read_text(path: Path) -> tuple[str | None, str]:
    try:
        return path.read_text(), ""
    except (FileNotFoundError, OSError) as exc:
        return None, str(exc)


def _check_goal_md(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "GOAL.md"
    content, err = _read_text(path)
    if content is None:
        return PrereqResult(
            name="GOAL.md present",
            status="fail",
            detail=err,
            fix_hint=f"Create projects/{project}/GOAL.md describing the project goal.",
        )
    if not content.strip():
        return PrereqResult(
            name="GOAL.md present",
            status="fail",
            detail="GOAL.md exists but is empty.",
            fix_hint=f"Create projects/{project}/GOAL.md describing the project goal.",
        )
    return PrereqResult(name="GOAL.md present", status="ok")


def _check_harness(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "harness.py"
    content, err = _read_text(path)
    if content is None:
        return PrereqResult(
            name="harness.py present",
            status="fail",
            detail=err,
            fix_hint="Implement the project harness via the `harness` skill: `/harness` slash command in Claude Code.",
        )
    if not re.search(r"^def\s+harness\s*\(", content, re.MULTILINE):
        return PrereqResult(
            name="harness.py present",
            status="fail",
            detail="harness.py exists but no `def harness(` function found.",
            fix_hint="Implement the project harness via the `harness` skill: `/harness` slash command in Claude Code.",
        )
    return PrereqResult(name="harness.py present", status="ok")


def _check_eda(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "eda.py"
    pkg_path = project_root / "projects" / project / "eda" / "__init__.py"
    content, err = _read_text(path)
    if content is None or "def eda(" not in content:
        pkg_content, pkg_err = _read_text(pkg_path)
        if pkg_content is not None and "def eda(" in pkg_content:
            return PrereqResult(name="eda.py present", status="ok")
        if content is None:
            return PrereqResult(
                name="eda.py present",
                status="fail",
                detail=err,
                fix_hint="Implement the project EDA via the `eda` skill.",
            )
        return PrereqResult(
            name="eda.py present",
            status="fail",
            detail="eda.py exists but no `def eda(` function found.",
            fix_hint="Implement the project EDA via the `eda` skill.",
        )
    return PrereqResult(name="eda.py present", status="ok")


def _check_discovery(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "discovery.py"
    content, err = _read_text(path)
    if content is None:
        return PrereqResult(
            name="discovery.py present",
            status="fail",
            detail=err,
            fix_hint="Implement project discovery via the `gap-finder` skill.",
        )
    if "def discover_gaps(" not in content:
        return PrereqResult(
            name="discovery.py present",
            status="fail",
            detail="discovery.py exists but no `def discover_gaps(` function found.",
            fix_hint="Implement project discovery via the `gap-finder` skill.",
        )
    return PrereqResult(name="discovery.py present", status="ok")


def _check_run_record(project_root: Path) -> PrereqResult:
    path = project_root / "libs" / "run_record.py"
    content, err = _read_text(path)
    if content is None:
        return PrereqResult(
            name="libs/run_record.py present",
            status="fail",
            detail=err,
            fix_hint="libs/run_record.py is required. See `modules/ds_v2/references/run-tandem.md` for the contract.",
        )
    if "def run_record(" not in content:
        return PrereqResult(
            name="libs/run_record.py present",
            status="fail",
            detail="libs/run_record.py exists but no `def run_record(` function found.",
            fix_hint="libs/run_record.py is required. See `modules/ds_v2/references/run-tandem.md` for the contract.",
        )
    return PrereqResult(name="libs/run_record.py present", status="ok")


def _check_ledgers(project_root: Path) -> PrereqResult:
    path = project_root / "libs" / "ledgers.py"
    content, err = _read_text(path)
    if content is None:
        return PrereqResult(
            name="libs/ledgers.py present",
            status="fail",
            detail=err,
            fix_hint="libs/ledgers.py with LedgerWriter is required.",
        )
    if "class LedgerWriter" not in content:
        return PrereqResult(
            name="libs/ledgers.py present",
            status="fail",
            detail="libs/ledgers.py exists but no `class LedgerWriter` found.",
            fix_hint="libs/ledgers.py with LedgerWriter is required.",
        )
    return PrereqResult(name="libs/ledgers.py present", status="ok")


def _check_outcomes(project_root: Path, project: str) -> PrereqResult:
    matches = list(project_root.glob(f"projects/{project}/outcomes/*/v1/dictionary.yaml"))
    if not matches:
        return PrereqResult(
            name="outcomes dictionary present",
            status="fail",
            detail=f"No outcome dictionaries found matching projects/{project}/outcomes/*/v1/dictionary.yaml.",
            fix_hint=f"Create at least one outcome version: projects/{project}/outcomes/{{outcome}}/v1/dictionary.yaml.",
        )
    return PrereqResult(name="outcomes dictionary present", status="ok")


def _check_run_01(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "runs" / "run_01.py"
    if not path.exists():
        return PrereqResult(
            name="runs/run_01.py present",
            status="fail",
            detail=f"projects/{project}/runs/run_01.py not found.",
            fix_hint=f"Create the baseline run script projects/{project}/runs/run_01.py.",
        )
    return PrereqResult(name="runs/run_01.py present", status="ok")


def _check_results_jsonl(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "runs" / "results.jsonl"
    content, err = _read_text(path)
    if content is None:
        return PrereqResult(
            name="results.jsonl has baseline",
            status="fail",
            detail=err,
            fix_hint="Run run_01.py at least once so results.jsonl has a baseline row.",
        )
    has_completed = False
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("status") == "completed":
                has_completed = True
                break
        except json.JSONDecodeError:
            continue
    if not has_completed:
        return PrereqResult(
            name="results.jsonl has baseline",
            status="fail",
            detail="results.jsonl exists but no row with status=completed found.",
            fix_hint="Run run_01.py at least once so results.jsonl has a baseline row.",
        )
    return PrereqResult(name="results.jsonl has baseline", status="ok")


def _check_autoloop_config(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "autoloop" / "config.yaml"
    if not path.exists():
        return PrereqResult(
            name="autoloop/config.yaml present",
            status="fail",
            detail=f"projects/{project}/autoloop/config.yaml not found.",
            fix_hint=f"Run `python -m libs.autoloop init --project {project}` first.",
        )
    return PrereqResult(name="autoloop/config.yaml present", status="ok")


def _check_config_schema(project_root: Path, project: str) -> PrereqResult:
    from libs.autoloop.state import AutoloopConfig

    path = project_root / "projects" / project / "autoloop" / "config.yaml"
    if not path.exists():
        return PrereqResult(
            name="config.yaml passes schema validation",
            status="fail",
            detail=f"projects/{project}/autoloop/config.yaml not found — cannot validate.",
        )
    try:
        AutoloopConfig.model_validate(yaml.safe_load(path.read_text()))
        return PrereqResult(name="config.yaml passes schema validation", status="ok")
    except ValidationError as e:
        errs = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        detail = "Schema violations:\n       " + "\n       ".join(f"- {e}" for e in errs)
        return PrereqResult(
            name="config.yaml passes schema validation",
            status="fail",
            detail=detail,
        )
    except yaml.YAMLError as e:
        return PrereqResult(
            name="config.yaml passes schema validation",
            status="fail",
            detail=f"YAML parse error: {e}",
        )


def _check_goal_metric(project_root: Path, project: str) -> PrereqResult:
    path = project_root / "projects" / project / "autoloop" / "config.yaml"
    cfg: dict = yaml.safe_load(path.read_text()) if path.exists() else {}
    if cfg.get("goal_metric") is None:
        return PrereqResult(
            name="goal_metric set in config.yaml",
            status="warn",
            detail=(
                "Optional. Without it, the scatter plot goal line is omitted.\n"
                "       Set `goal_metric: <target>` in autoloop/config.yaml to enable."
            ),
        )
    return PrereqResult(name="goal_metric set in config.yaml", status="ok")


def _check_git_clean(project_root: Path) -> PrereqResult:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:
        return PrereqResult(
            name="git working tree clean",
            status="fail",
            detail=str(exc),
            fix_hint="Commit or stash pending changes before starting autoloop.",
        )
    if result.returncode != 0:
        return PrereqResult(
            name="git working tree clean",
            status="fail",
            detail=result.stderr.strip() or "git status exited non-zero.",
            fix_hint="Commit or stash pending changes before starting autoloop.",
        )
    if result.stdout.strip():
        return PrereqResult(
            name="git working tree clean",
            status="fail",
            detail="Working tree has uncommitted changes.",
            fix_hint="Commit or stash pending changes before starting autoloop.",
        )
    return PrereqResult(name="git working tree clean", status="ok")


def _check_imports(project_root: Path) -> PrereqResult:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import libs.run_record, libs.ledgers"],
            cwd=project_root,
            capture_output=True,
        )
    except (FileNotFoundError, OSError) as exc:
        return PrereqResult(
            name="libs imports resolve",
            status="fail",
            detail=str(exc),
            fix_hint="Module import failed. Make sure libs/run_record.py and libs/ledgers.py are valid Python.",
        )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        return PrereqResult(
            name="libs imports resolve",
            status="fail",
            detail=stderr,
            fix_hint="Module import failed. Make sure libs/run_record.py and libs/ledgers.py are valid Python.",
        )
    return PrereqResult(name="libs imports resolve", status="ok")


def check_prereqs(project_root: Path, project: str) -> PrereqReport:
    report = PrereqReport()

    checks = [
        _check_goal_md(project_root, project),
        _check_harness(project_root, project),
        _check_eda(project_root, project),
        _check_discovery(project_root, project),
        _check_run_record(project_root),
        _check_ledgers(project_root),
        _check_outcomes(project_root, project),
        _check_run_01(project_root, project),
        _check_results_jsonl(project_root, project),
        _check_autoloop_config(project_root, project),
        _check_config_schema(project_root, project),
        _check_goal_metric(project_root, project),
        _check_git_clean(project_root),
        _check_imports(project_root),
    ]

    for result in checks:
        if result.status == "ok":
            report.passed.append(result)
        elif result.status == "warn":
            report.warned.append(result)
        else:
            report.failed.append(result)

    return report
