from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from libs.autoloop.dedup import content_hash
from libs.autoloop.state import (
    AutoloopConfig,
    BrainstormItem,
    BrainstormSource,
    BudgetCfg,
    CheckpointCfg,
    CurriculumCfg,
    GapFinderCfg,
    LoopDetectionCfg,
    PlateauCfg,
    SessionCfg,
    append_jsonl,
    atomic_write_text,
)


def _prompt(msg: str, default: str = "") -> str:
    if default:
        display = f"{msg} [{default}]: "
    else:
        display = f"{msg}: "
    try:
        val = input(display).strip()
    except EOFError:
        return default
    return val or default


def _prompt_required(msg: str) -> str:
    while True:
        try:
            val = input(f"{msg}: ").strip()
        except EOFError:
            print("\nAborted.")
            sys.exit(130)
        if val:
            return val
        print("  (required — cannot be empty)")


def init_project(
    project_root: Path,
    project: str,
    *,
    non_interactive: bool = False,
    defaults_only: bool = False,
) -> None:
    """Interactive setup: validate project exists, prompt for config values, write config.yaml,
    prompt for 3-5 seed brainstorm items, write brainstorm.jsonl, create logs/ and dedup_index/ dirs.

    Args:
        project_root: root of ds3 checkout
        project: project name (e.g., "civic_shout_action_rate_increase")
        non_interactive: if True, use defaults and skip prompts (for tests / scripted setup)
        defaults_only: if True, write defaults without prompting but still warn if seeds missing
    """
    try:
        _run_init(
            project_root, project, non_interactive=non_interactive, defaults_only=defaults_only
        )
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)


def _run_init(
    project_root: Path,
    project: str,
    *,
    non_interactive: bool,
    defaults_only: bool,
) -> None:
    goal_path = project_root / "projects" / project / "GOAL.md"
    if not goal_path.exists():
        raise FileNotFoundError(
            f"Project '{project}' not found. Expected GOAL.md at: {goal_path}\n"
            "Create the project directory and GOAL.md before running autoloop init."
        )

    lines = goal_path.read_text().splitlines()
    print("\n--- GOAL.md (first 30 lines) ---")
    for line in lines[:30]:
        print(line)
    print("---\n")

    state_dir = project_root / "projects" / project / "autoloop"
    config_path = state_dir / "config.yaml"

    if config_path.exists() and not non_interactive and not defaults_only:
        answer = _prompt("Existing config found. Overwrite? [y/N]", default="N")
        if answer.lower() != "y":
            print("Aborted — existing config preserved.")
            return
    elif config_path.exists() and non_interactive:
        return

    skip_prompts = non_interactive or defaults_only

    if skip_prompts:
        outcome_variable = ""
        comparison_group = f"{project}_v1"
        primary_metric = "roc_auc"
        metric_direction = "higher_is_better"
        sample_frac = 0.10
        iterations_max = 70
        dollars_max = 80.0
        gap_finder_cadence = "every_n_iters"
        gap_finder_n = 5
    else:
        outcome_variable = _prompt_required("outcome_variable (required, e.g. 'acted_within_24h')")
        comparison_group = _prompt("comparison_group", default=f"{project}_v1")
        primary_metric = _prompt("primary_metric", default="roc_auc")

        raw_direction = _prompt(
            "metric_direction [higher_is_better/lower_is_better]",
            default="higher_is_better",
        )
        metric_direction = (
            raw_direction
            if raw_direction in ("higher_is_better", "lower_is_better")
            else "higher_is_better"
        )

        raw_frac = _prompt("sample_frac", default="0.10")
        try:
            sample_frac = float(raw_frac)
        except ValueError:
            sample_frac = 0.10

        raw_iters = _prompt("iterations_max", default="70")
        try:
            iterations_max = int(raw_iters)
        except ValueError:
            iterations_max = 70

        raw_dollars = _prompt("dollars_max", default="80.0")
        try:
            dollars_max = float(raw_dollars)
        except ValueError:
            dollars_max = 80.0

        raw_cadence = _prompt(
            "gap_finder_cadence [every_n_iters/never/every_iter/on_demand]",
            default="every_n_iters",
        )
        gap_finder_cadence = (
            raw_cadence
            if raw_cadence in ("every_n_iters", "never", "every_iter", "on_demand")
            else "every_n_iters"
        )

        if gap_finder_cadence == "every_n_iters":
            raw_n = _prompt("gap_finder_n", default="5")
            try:
                gap_finder_n = int(raw_n)
            except ValueError:
                gap_finder_n = 5
        else:
            gap_finder_n = 5

    cfg = AutoloopConfig(
        project=project,
        outcome_variable=outcome_variable,
        comparison_group=comparison_group,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        sample_frac=sample_frac,
        gap_finder=GapFinderCfg(cadence=gap_finder_cadence, n=gap_finder_n),
        budget=BudgetCfg(iterations_max=iterations_max, dollars_max=dollars_max),
        plateau=PlateauCfg(),
        loop_detection=LoopDetectionCfg(),
        curriculum=CurriculumCfg(),
        checkpoint=CheckpointCfg(),
        session=SessionCfg(),
    )

    state_dir.mkdir(parents=True, exist_ok=True)
    yaml_str = yaml.dump(cfg.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(config_path, yaml_str)
    print(f"Config written to {config_path}")

    seeds_written = 0

    if not non_interactive:
        print(
            "\nSeed the brainstorm with 3–5 feature ideas (type 'done' for title after 3 to stop).\n"
        )
        brainstorm_path = state_dir / "brainstorm.jsonl"

        for i in range(5):
            if i >= 3:
                raw_title = _prompt("Seed title (or 'done' to stop)")
                if raw_title.lower() == "done" or not raw_title:
                    break
                title = raw_title
            else:
                title = _prompt_required(f"Seed {i + 1} title")

            hypothesis = _prompt_required(f"Seed {i + 1} hypothesis (1-2 sentences)")
            row_grain = _prompt(f"Seed {i + 1} row_grain", default="(user_id)")

            item_id = (
                f"BS-{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}"
                f"-{hashlib.sha256(title.encode()).hexdigest()[:6]}"
            )
            item = BrainstormItem(
                id=item_id,
                ts=datetime.now(tz=timezone.utc).isoformat(),
                title=title,
                hypothesis=hypothesis,
                source=BrainstormSource(kind="seed"),
                feature_family_name_hint=title.lower().replace(" ", "_"),
                row_grain=row_grain,
                content_hash=content_hash(title, row_grain, hypothesis),
                tier="T1",
                tier_score=0.5,
                tier_rationale="user-seed; not yet evaluated",
                status="queued",
            )
            append_jsonl(brainstorm_path, item)
            seeds_written += 1
            print(f"  Added seed {seeds_written}: {title}")

        if seeds_written == 0:
            print(
                "Warning: no seed items written. Add seeds manually to brainstorm.jsonl before running."
            )
    elif defaults_only:
        print("Warning: running with defaults_only — no seed items written. Add seeds manually.")

    (state_dir / "logs").mkdir(exist_ok=True)
    (state_dir / "dedup_index").mkdir(exist_ok=True)

    print(f"""
Autoloop initialized for project: {project}
Config: {config_path}
Brainstorm seeded with {seeds_written} items.

Next:
  python -m libs.autoloop check-prereqs --project {project}
  python -m libs.autoloop run --project {project} --budget 3 --dollars 12   # small dogfood run
""")
