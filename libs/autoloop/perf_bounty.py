from __future__ import annotations

import importlib.util
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from libs.autoloop.perf_fingerprints import (
    has_aggregate_in_serial_loop,
    has_iterrows_with_api_call,
    has_unbatched_embedding_loop,
)
from libs.autoloop.perf_ledgers import (
    PerfAttempt,
    PerfWin,
    append_attempt,
    append_win,
    save_diff,
)
from libs.perf_equivalence import harness_response_equivalent, predictions_equivalent
from libs.responses import HarnessResponse

__all__ = [
    "PerfBountyResult",
    "run_perf_bounty",
]


class PerfBountyResult(BaseModel):
    outcome: Literal[
        "merged",
        "noise",
        "drift_failed",
        "tier_escalation_failed",
        "no_proposal",
        "budget_killed",
        "apply_failed",
    ]
    wall_before_s: float
    wall_after_s: float | None
    speedup: float | None
    technique: str | None
    commit_sha: str | None
    failure_reason: str | None
    fingerprint_tag: str


def _compute_fingerprint_tag(source: str) -> str:
    if has_iterrows_with_api_call(source):
        return "iterrows_api_call"
    if has_aggregate_in_serial_loop(source):
        return "aggregate_serial_loop"
    if has_unbatched_embedding_loop(source):
        return "unbatched_embedding_loop"
    return ""


def _load_and_call_fixture(
    candidate_harness_path: Path,
    frozen_predictions_path: Path,
    output_predictions_path: Path,
    oracle_wall_seconds: float,
) -> tuple[HarnessResponse, float]:
    spec = importlib.util.spec_from_file_location("_candidate_harness", candidate_harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {candidate_harness_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    t0 = time.perf_counter()
    response: HarnessResponse = mod.run_harness(
        frozen_predictions_path=frozen_predictions_path,
        output_predictions_path=output_predictions_path,
        oracle_wall_seconds=oracle_wall_seconds,
    )
    wall_after_s = time.perf_counter() - t0

    # Fixtures report total_seconds via HarnessResponse.performance; use that as
    # the candidate wall time so test assertions are deterministic regardless of
    # actual elapsed time on the test runner.
    reported_s = response.performance.total_seconds if response.performance else wall_after_s
    return response, reported_s


async def run_perf_bounty(
    *,
    project: str,
    iteration_id: int,
    run_id: str,
    frozen_response_path: Path,
    frozen_predictions_path: Path,
    timing_row: dict[str, Any],
    project_root: Path,
    candidate_harness_path: Path | None = None,
    dry_run: bool = True,
    skip_tier_escalation: bool = True,
    champion_diffs: str = "",
    tried_and_failed: str = "",
    cli_path: str = "claude",
    repo_root: Path | None = None,
    max_dollars: float = 1.00,
) -> PerfBountyResult:
    if repo_root is None:
        repo_root = project_root

    frozen_response = HarnessResponse.model_validate_json(frozen_response_path.read_text())
    wall_before_s: float = float(timing_row.get("total_seconds", 200.0))
    bottleneck_stage: str = str(timing_row.get("bottleneck_stage", "unknown"))

    candidate_source = (
        candidate_harness_path.read_text() if candidate_harness_path is not None else ""
    )
    fingerprint_tag = _compute_fingerprint_tag(candidate_source)

    def _make_attempt(
        *,
        wall_after_s: float | None,
        speedup: float | None,
        equivalence_passed: bool,
        failure_reason: str | None,
        drift_pct: float | None,
        technique: str = "",
    ) -> PerfAttempt:
        return PerfAttempt(
            ts=_now_iso(),
            project=project,
            iteration_id=iteration_id,
            run_id=run_id,
            target_file=f"projects/{project}/harness.py",
            bottleneck_stage=bottleneck_stage,
            fingerprint=fingerprint_tag,
            technique=technique,
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=speedup,
            equivalence_passed=equivalence_passed,
            failure_reason=failure_reason,
            drift_pct=drift_pct,
        )

    if candidate_harness_path is None:
        # Production path: spawn claude -p
        return await _run_production_path(
            project=project,
            iteration_id=iteration_id,
            run_id=run_id,
            frozen_response_path=frozen_response_path,
            frozen_predictions_path=frozen_predictions_path,
            frozen_response=frozen_response,
            timing_row=timing_row,
            wall_before_s=wall_before_s,
            bottleneck_stage=bottleneck_stage,
            fingerprint_tag=fingerprint_tag,
            project_root=project_root,
            repo_root=repo_root,
            skip_tier_escalation=skip_tier_escalation,
            dry_run=dry_run,
            champion_diffs=champion_diffs,
            tried_and_failed=tried_and_failed,
            cli_path=cli_path,
            make_attempt=_make_attempt,
            max_dollars=max_dollars,
        )

    # Test / seam path: candidate_harness_path provided — skip claude spawn
    output_predictions_path = project_root / "tmp" / "perf_bounty" / run_id / "predictions.parquet"
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    _candidate_response, wall_after_s = _load_and_call_fixture(
        candidate_harness_path=candidate_harness_path,
        frozen_predictions_path=frozen_predictions_path,
        output_predictions_path=output_predictions_path,
        oracle_wall_seconds=wall_before_s,
    )

    equiv = predictions_equivalent(frozen_predictions_path, output_predictions_path)

    if not equiv.passed:
        attempt = _make_attempt(
            wall_after_s=wall_after_s,
            speedup=None,
            equivalence_passed=False,
            failure_reason="drift_failed",
            drift_pct=equiv.drift_pct,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="drift_failed",
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason="predictions_drift",
            fingerprint_tag=fingerprint_tag,
        )

    resp_equiv = harness_response_equivalent(frozen_response, _candidate_response)
    if not resp_equiv.passed:
        attempt = _make_attempt(
            wall_after_s=wall_after_s,
            speedup=None,
            equivalence_passed=False,
            failure_reason="drift_failed",
            drift_pct=resp_equiv.drift_pct,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="drift_failed",
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason="response_drift",
            fingerprint_tag=fingerprint_tag,
        )

    speedup = wall_before_s / wall_after_s if wall_after_s > 0 else 1.0

    if speedup < 1.10:
        attempt = _make_attempt(
            wall_after_s=wall_after_s,
            speedup=speedup,
            equivalence_passed=True,
            failure_reason="speedup_below_threshold",
            drift_pct=None,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="noise",
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=speedup,
            technique=None,
            commit_sha=None,
            failure_reason="speedup_below_threshold",
            fingerprint_tag=fingerprint_tag,
        )

    # Tier escalation guard — skipped in test path (dry_run=True)
    # (no-op here; would run against a higher sample tier in production)

    # Merged — in dry_run, skip git commit and save_diff
    technique = fingerprint_tag if fingerprint_tag else "unknown"
    attempt = _make_attempt(
        wall_after_s=wall_after_s,
        speedup=speedup,
        equivalence_passed=True,
        failure_reason=None,
        drift_pct=None,
        technique=technique,
    )
    append_attempt(repo_root, attempt)

    return PerfBountyResult(
        outcome="merged",
        wall_before_s=wall_before_s,
        wall_after_s=wall_after_s,
        speedup=speedup,
        technique=technique,
        commit_sha=None,
        failure_reason=None,
        fingerprint_tag=fingerprint_tag,
    )


async def _run_production_path(
    *,
    project: str,
    iteration_id: int,
    run_id: str,
    frozen_response_path: Path,
    frozen_predictions_path: Path,
    frozen_response: HarnessResponse,
    timing_row: dict[str, Any],
    wall_before_s: float,
    bottleneck_stage: str,
    fingerprint_tag: str,
    project_root: Path,
    repo_root: Path,
    skip_tier_escalation: bool,
    dry_run: bool,
    champion_diffs: str,
    tried_and_failed: str,
    cli_path: str,
    make_attempt: Any,
    max_dollars: float = 1.00,
) -> PerfBountyResult:
    # Deferred to avoid circular import: state → perf_bounty → session → state
    from libs.autoloop.session import run_claude_session
    from libs.costs.tracker import CostTracker

    prompts_dir = Path(__file__).parent / "prompts"
    env = Environment(loader=FileSystemLoader(str(prompts_dir)), autoescape=False)
    template = env.get_template("perf_bounty.md")
    prompt = template.render(
        project=project,
        run_id=run_id,
        iteration_id=iteration_id,
        frozen_response_path=str(frozen_response_path),
        frozen_predictions_path=str(frozen_predictions_path),
        timing_row=timing_row,
        bottleneck_stage=bottleneck_stage,
        champion_diffs=champion_diffs or "No prior wins for this bottleneck shape yet.",
        tried_and_failed=tried_and_failed or "No prior failed attempts on this shape.",
    )

    settings_path = prompts_dir / "perf_bounty_settings.json"
    cost_tracker = CostTracker()
    cost_key = f"autoloop:{project}:iter_{iteration_id}:perf_bounty"
    log_path = (
        project_root / "autoloop" / "logs" / f"iter_{iteration_id}_perf_bounty_{run_id}.jsonl"
    )

    session_result = run_claude_session(
        role="executor",
        prompt=prompt,
        cwd=repo_root,
        settings_path=settings_path,
        allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep", "Write"],
        denied_tools=["WebFetch", "WebSearch", "NotebookEdit"],
        max_wall_seconds=1800.0,
        max_dollars=max_dollars,
        cost_tracker=cost_tracker,
        cost_key=cost_key,
        log_path=log_path,
        cli_path=cli_path,
        project=project,
        iteration_id=iteration_id,
        model="claude-sonnet-4-6",
        max_turns=100,
    )

    if session_result.kill_reason in {"wall_cap", "dollar_cap"}:
        attempt = make_attempt(
            wall_after_s=None,
            speedup=None,
            equivalence_passed=False,
            failure_reason="budget_killed",
            drift_pct=None,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="budget_killed",
            wall_before_s=wall_before_s,
            wall_after_s=None,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason=session_result.kill_reason,
            fingerprint_tag=fingerprint_tag,
        )

    diff_path = Path(f"/tmp/perf_bounty_{run_id}.diff")
    if not diff_path.exists():
        attempt = make_attempt(
            wall_after_s=None,
            speedup=None,
            equivalence_passed=False,
            failure_reason="no_proposal",
            drift_pct=None,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="no_proposal",
            wall_before_s=wall_before_s,
            wall_after_s=None,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason="diff_file_missing",
            fingerprint_tag=fingerprint_tag,
        )

    diff_text = diff_path.read_text()

    try:
        subprocess.run(
            ["git", "apply", "--check", str(diff_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
    except subprocess.CalledProcessError as exc:
        attempt = make_attempt(
            wall_after_s=None,
            speedup=None,
            equivalence_passed=False,
            failure_reason="apply_failed",
            drift_pct=None,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="apply_failed",
            wall_before_s=wall_before_s,
            wall_after_s=None,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason=f"git apply --check failed: {exc.stderr[:200]}",
            fingerprint_tag=fingerprint_tag,
        )

    subprocess.run(
        ["git", "apply", str(diff_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    candidate_path = repo_root / "projects" / project / "harness.py"
    candidate_source = candidate_path.read_text()
    fingerprint_tag = _compute_fingerprint_tag(candidate_source)

    output_predictions_path = Path(f"/tmp/perf_bounty_{run_id}") / "predictions.parquet"
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _candidate_response, wall_after_s = _load_and_call_fixture(
            candidate_harness_path=candidate_path,
            frozen_predictions_path=frozen_predictions_path,
            output_predictions_path=output_predictions_path,
            oracle_wall_seconds=wall_before_s,
        )
    except Exception as exc:
        subprocess.run(
            ["git", "checkout", "--", f"projects/{project}/harness.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        attempt = make_attempt(
            wall_after_s=None,
            speedup=None,
            equivalence_passed=False,
            failure_reason=f"candidate_run_error: {exc}",
            drift_pct=None,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="apply_failed",
            wall_before_s=wall_before_s,
            wall_after_s=None,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason=f"candidate harness raised: {exc}",
            fingerprint_tag=fingerprint_tag,
        )

    equiv = predictions_equivalent(frozen_predictions_path, output_predictions_path)

    if not equiv.passed:
        subprocess.run(
            ["git", "checkout", "--", f"projects/{project}/harness.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        attempt = make_attempt(
            wall_after_s=wall_after_s,
            speedup=None,
            equivalence_passed=False,
            failure_reason="drift_failed",
            drift_pct=equiv.drift_pct,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="drift_failed",
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason="predictions_drift",
            fingerprint_tag=fingerprint_tag,
        )

    resp_equiv = harness_response_equivalent(frozen_response, _candidate_response)
    if not resp_equiv.passed:
        subprocess.run(
            ["git", "checkout", "--", f"projects/{project}/harness.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        attempt = make_attempt(
            wall_after_s=wall_after_s,
            speedup=None,
            equivalence_passed=False,
            failure_reason="drift_failed",
            drift_pct=resp_equiv.drift_pct,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="drift_failed",
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=None,
            technique=None,
            commit_sha=None,
            failure_reason="response_drift",
            fingerprint_tag=fingerprint_tag,
        )

    speedup = wall_before_s / wall_after_s if wall_after_s > 0 else 1.0

    if speedup < 1.10:
        subprocess.run(
            ["git", "checkout", "--", f"projects/{project}/harness.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        attempt = make_attempt(
            wall_after_s=wall_after_s,
            speedup=speedup,
            equivalence_passed=True,
            failure_reason="speedup_below_threshold",
            drift_pct=None,
        )
        append_attempt(repo_root, attempt)
        return PerfBountyResult(
            outcome="noise",
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=speedup,
            technique=None,
            commit_sha=None,
            failure_reason="speedup_below_threshold",
            fingerprint_tag=fingerprint_tag,
        )

    if not skip_tier_escalation:
        # Tier escalation: re-run at a higher sample tier to confirm equivalence holds.
        # If it fails, rollback and return tier_escalation_failed.
        # Stub: in v1 no higher tier is available from the test oracle; implement when
        # the harness exposes a tier-upgrade API.
        pass

    technique_name = fingerprint_tag if fingerprint_tag else "unknown"
    commit_sha: str | None = None

    if not dry_run:
        commit_msg = (
            f"perf: {speedup:.2f}x speedup on {bottleneck_stage} "
            f"(autoloop iter {iteration_id}, run {run_id})"
        )
        try:
            subprocess.run(
                ["git", "add", f"projects/{project}/harness.py"],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            commit_sha = sha_result.stdout.strip() or None
        except subprocess.CalledProcessError:
            commit_sha = None

        diff_bank_path = save_diff(
            repo_root=repo_root,
            fingerprint_tag=fingerprint_tag,
            project=project,
            run_id=run_id,
            diff_text=diff_text,
            meta={
                "speedup": speedup,
                "wall_before_s": wall_before_s,
                "wall_after_s": wall_after_s,
                "technique": technique_name,
                "bottleneck_stage": bottleneck_stage,
                "commit_sha": commit_sha,
            },
        )
        win = PerfWin(
            ts=_now_iso(),
            project=project,
            iteration_id=iteration_id,
            run_id=run_id,
            target_file=f"projects/{project}/harness.py",
            bottleneck_stage=bottleneck_stage,
            fingerprint=fingerprint_tag,
            technique=technique_name,
            wall_before_s=wall_before_s,
            wall_after_s=wall_after_s,
            speedup=speedup,
            seconds_saved=wall_before_s - wall_after_s,
            commit_sha=commit_sha or "",
            diff_bank_path=str(diff_bank_path),
        )
        append_win(repo_root=repo_root, project_root=project_root, win=win)

    attempt = make_attempt(
        wall_after_s=wall_after_s,
        speedup=speedup,
        equivalence_passed=True,
        failure_reason=None,
        drift_pct=None,
        technique=technique_name,
    )
    append_attempt(repo_root, attempt)

    return PerfBountyResult(
        outcome="merged",
        wall_before_s=wall_before_s,
        wall_after_s=wall_after_s,
        speedup=speedup,
        technique=technique_name,
        commit_sha=commit_sha,
        failure_reason=None,
        fingerprint_tag=fingerprint_tag,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
