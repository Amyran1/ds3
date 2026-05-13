# Autoloop — autonomous feature-engineering wrapper

The autoloop runs an open-ended loop of planner + executor `claude -p` sessions
to generate, rank, and implement feature ideas against a project harness. Each
iteration produces at most one new feature family, one new `run_NN.py`, and one
ledger row. The supervisor (pure Python, no LLM calls) owns the loop logic:
budget accounting, termination, git commits, and Slack notifications. It does
not touch harness code, outcome definitions, or entity caches, and it never
promotes a champion — champion selection remains a deliberate human step.
Everything project-specific lives in `projects/{name}/autoloop/`; the
`libs/autoloop/` package is fully project-agnostic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Supervisor (Python, no LLM)                          │
│                                                                 │
│  libs/autoloop/supervisor.py                                    │
│  ├── check termination conditions (termination.py)              │
│  ├── spawn planner session  ──────────────────────────────────┐  │
│  │   wait for exit + read sentinel from brainstorm.jsonl      │  │
│  ├── pop top-scored queued idea from brainstorm.jsonl         │  │
│  ├── spawn executor session ──────────────────────────────────┘  │
│  │   wait for exit + read sentinel from brainstorm.jsonl         │
│  ├── prune stale predictions (retention.py)                     │
│  ├── git commit (scope-limited allowlist)                       │
│  └── append IterationRecord to iterations.jsonl                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Sessions (LLM, ephemeral, fresh context per iter)    │
│                                                                 │
│  Planner  ← claude -p prompts/planner.md   (max 300 s)         │
│  │  reads gap-finder / EDA findings, existing brainstorm        │
│  │  writes new BrainstormItem rows + planner_summary sentinel   │
│  │  into projects/{name}/autoloop/brainstorm.jsonl              │
│  │                                                              │
│  Executor ← claude -p prompts/executor.md  (per_session limit) │
│     picks idea handed to it by the supervisor                   │
│     implements feature family + run_NN.py                       │
│     invokes run_record(...) → harness + eda + ledger rows       │
│     writes executor_summary sentinel to brainstorm.jsonl        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — State (JSONL + JSON, append-only or atomic-rename)   │
│                                                                 │
│  projects/{name}/autoloop/                                      │
│  ├── config.yaml          (atomic-write, supervisor only)       │
│  ├── brainstorm.jsonl     (append-only, event-sourced replay)   │
│  ├── idea_registry.md     (regenerated each iter by registry.py)│
│  ├── iterations.jsonl     (append-only, one row per iter)       │
│  ├── failure_modes.jsonl  (append-only, LLM-extracted summaries)│
│  ├── budget.json          (atomic-write after each iter)        │
│  ├── STOP                 (sentinel file, touch to halt)        │
│  ├── logs/                (per-iter planner + executor JSONL)   │
│  └── dedup_index/         (FAISS index shards for semantic dedup│
│                            falls back to content-hash only)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker; no public symbols. |
| `__main__.py` | Entry point for `python -m libs.autoloop`; delegates to `cli.py`. |
| `budget.py` | `AutoloopBudget` — tracks iterations, dollars (via CostTracker), and wall-clock elapsed; writes `budget.json` after each iter. |
| `cli.py` | Argument parser for `init`, `check-prereqs`, `run`, `status`, `stop` subcommands. |
| `curriculum.py` | `decide(iter_id, max_iters, explore_fraction)` — returns EXPLORE or EXPLOIT phase for the iteration. |
| `dedup.py` | Content-hash and FAISS-based semantic deduplication of incoming brainstorm ideas. Falls back to hash-only when `faiss` is absent. |
| `failure_modes.py` | LLM-assisted failure fingerprinting: `extract_failure_fingerprint` summarises a crashed session log; `format_for_prompt` injects recent failures into planner/executor prompts. |
| `init.py` | Interactive (or `--defaults-only`) project initialisation: creates `autoloop/config.yaml` with user-supplied or default values. |
| `prereqs.py` | `check_prereqs(project_root, project)` — 14-check preflight gate. Blocks `run_supervisor` if any check fails. |
| `progress.py` | `event(msg)` timestamped console logger; `Heartbeat` background ticker; `fmt_dur` for human-readable durations. |
| `registry.py` | `rebuild_registry(state)` — rewrites `idea_registry.md` from the current brainstorm snapshot after each planner phase. |
| `retention.py` | `prune_old_predictions` — deletes `predictions.parquet` files for runs outside the top-K to contain disk use. |
| `session.py` | `run_claude_session(...)` — subprocess wrapper around `claude -p`; streams JSONL output to a log file; enforces wall-clock and dollar caps; returns `SessionResult`. |
| `slack.py` | `notify_start`, `notify_termination`, `notify_milestone`, `notify_hard_failure` — Slack notifications via the overlay's `claude-ping` script. No-ops if the script is absent. |
| `state.py` | All Pydantic models (`AutoloopConfig`, `BrainstormItem`, `IterationRecord`, etc.) and the `State` helper that wraps `brainstorm.jsonl` + `iterations.jsonl`. |
| `supervisor.py` | `run_supervisor(SupervisorArgs)` — the outer loop; owns the planner → executor → commit cycle. |
| `termination.py` | `check_all(...)` — evaluates all termination conditions and returns a `TerminationVerdict`. |
| `hooks/autoloop-write-guard.sh` | PreToolUse hook injected into every planner/executor session via `--settings`. Denies writes to protected paths and git mutations. |
| `prompts/denied_writes.txt` | Regex patterns that the write guard matches file paths against. |
| `prompts/planner.md` | Jinja-style template rendered into the planner session's initial prompt. |
| `prompts/executor.md` | Jinja-style template rendered into the executor session's initial prompt. |
| `prompts/session_settings_template.json` | Claude Code `--settings` template; the supervisor substitutes `{{hook_path}}` before writing a per-iter copy. |

---

## The stage enum

The supervisor does not use an explicit stage enum at runtime — stage is inferred
from what the current iteration is doing. The dashboard and log events use these
five labels:

| Stage | When it applies |
|-------|----------------|
| `idle` | Supervisor is between iterations: termination check passed, next iter not yet started. |
| `planning` | Planner session is running (`claude -p prompts/planner.md`). Heartbeat ticker is active. |
| `building` | Executor session is running (`claude -p prompts/executor.md`). Heartbeat ticker is active. |
| `running` | Executor has invoked `run_record(...)` and is waiting for harness + EDA to complete inside the same session. (Visible in executor session logs, not a separate supervisor state.) |
| `committing` | Supervisor is executing `_commit_iteration`: git status audit, scope-limited `git add`, `git commit`. |

The dashboard's session card derives the displayed stage from the most recent
`event(...)` line in the supervisor's stdout stream.

---

## State files (per-project)

All files live under `projects/{name}/autoloop/`.

| File | Schema / type | Writer | Purpose |
|------|---------------|--------|---------|
| `config.yaml` | `AutoloopConfig` (Pydantic) | Human / `init.py` only | Project identity, budgets, plateau/loop-detection thresholds, gap-finder cadence, curriculum fractions, Slack settings, model names. |
| `brainstorm.jsonl` | Append-only JSONL; event-sourced (last row per `id` wins) | Planner (new `BrainstormItem` rows), supervisor (status-update rows via `mark_consumed`/`mark_failed`), executor (sentinel rows with schema `planner_summary/v1` and `executor_summary/v1`) | Idea queue and completion ledger. `State.load_brainstorm()` replays the full log. |
| `idea_registry.md` | Markdown table | `registry.rebuild_registry` (called by supervisor after each planner phase) | Human-readable snapshot of all brainstorm ideas with status, tier, and run_id. Overwritten each iter. |
| `iterations.jsonl` | Append-only JSONL; `IterationRecord` rows (schema `iteration/v1`) | Supervisor (`state.append_iteration`) | Per-iteration timing, planner result, executor result, budget snapshot, termination checks, fingerprint. One row per completed iteration. |
| `failure_modes.jsonl` | Append-only JSONL; `FailureMode` rows | `failure_modes.append_failure_mode` (supervisor, after a planner or executor crash) | LLM-extracted crash summaries injected into subsequent planner/executor prompts to discourage repeated mistakes. |
| `budget.json` | `BudgetSnapshot` (Pydantic, atomic-write) | `AutoloopBudget.snapshot(...)` (supervisor, after each iter) | Latest budget state: iterations/dollars/wall used and max, top-3 run IDs and metrics. Used by `autoloop status` and the dashboard. |
| `STOP` | Empty sentinel file | Human (`touch`), `autoloop stop` CLI, `make autoloop-stop` | Causes the supervisor to terminate cleanly at the next iteration boundary with exit code 30. |
| `logs/` | Directory of per-iter JSONL files | `session.run_claude_session` | `iter_NNN_planner.jsonl`, `iter_NNN_planner_settings.json`, `iter_NNN_executor.jsonl`, `iter_NNN_executor_settings.json`. Full Claude Code stream output for post-mortem inspection. |
| `dedup_index/` | FAISS index shards + metadata | `dedup.py` | Embedding-based semantic deduplication index. Falls back to content-hash when `faiss` is not installed. |

---

## Configuration reference

All fields are in `projects/{name}/autoloop/config.yaml`.

### Top-level required fields

| Field | Type | Description |
|-------|------|-------------|
| `project` | `str` | Project directory name under `projects/`. Must match exactly. |
| `outcome_variable` | `str` | Column name of the target variable (e.g. `median_house_value`). Passed to executor prompts. |
| `comparison_group` | `str` | `comparison_group` value passed to `run_record(...)`. Identifies the run family in ledgers. |
| `primary_metric` | `str` | Metric name (e.g. `r2`, `roc_auc`) used for leaderboard ranking and plateau detection. |

### Top-level optional fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metric_direction` | `"higher_is_better"` \| `"lower_is_better"` | `"higher_is_better"` | Whether a larger primary_metric value is better. Controls plateau top-K comparison direction. |
| `sample_frac` | `float` | `0.10` | Fraction of training data the executor uses when building features (passed into executor prompt). |
| `goal_metric` | `float` | absent | Optional target primary_metric value. When set, the dashboard scatter plot draws a horizontal goal line. Without it the line is omitted. `check_prereqs` issues a warning (not a failure) when this field is missing. |

### `gap_finder` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cadence` | `"never"` \| `"every_iter"` \| `"every_n_iters"` \| `"on_demand"` | `"every_n_iters"` | When the executor runs the gap-finder (`discover_gaps`). `every_n_iters` uses the `n` field. `on_demand` means the planner decides per-iteration. |
| `n` | `int` | `5` | Number of iterations between gap-finder runs when `cadence: every_n_iters`. |
| `champion_only` | `bool` | `true` | If true, gap-finder runs only against the current champion's predictions. |

### `budget` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iterations_max` | `int` | `70` | Hard iteration cap. Supervisor exits with code 10 when hit. |
| `dollars_max` | `float` | `80.0` | Hard dollar cap across all API calls attributed to this project. Exit code 11. |
| `wall_seconds_max` | `int` | `86400` | Hard wall-clock cap in seconds (24 h). Exit code 12. |
| `per_session_dollars_max` | `float` | `5.0` | Per-session dollar limit passed to `session.run_claude_session`. Session is killed when exceeded. |
| `per_session_wall_seconds_max` | `int` | `4200` | Per-session wall-clock limit in seconds (70 min default). Planner is capped at 300 s regardless. |

### `plateau` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | `int` | `3` | Number of top run IDs tracked for plateau comparison. |
| `window_iters` | `int` | `6` | If the top-K run IDs are unchanged across the last `window_iters` iterations, the loop is considered plateaued. |

### `loop_detection` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dedup_reject_streak` | `int` | `4` | Consecutive iterations where the planner adds zero new or re-ranked ideas before the supervisor declares brainstorm exhaustion. |
| `identical_args_streak` | `int` | `3` | Stored in config for reference; the active streak check uses `loop_fingerprint` logic in `termination.py`. |

### `curriculum` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `explore_fraction` | `float` | `0.4` | Fraction of the total iteration budget allocated to the EXPLORE phase (earlier iterations). The remainder is EXPLOIT. |
| `exploit_top_k` | `int` | `5` | Number of top runs whose predictions are retained by `retention.prune_old_predictions`. |

### `checkpoint` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `human_every_n_iters` | `int` | `0` | Reserved for future human-in-the-loop checkpoints. Currently unused (`0` means disabled). |
| `slack_milestone_iters` | `list[int]` | `[1, 10, 25, 50, 70]` | Iteration numbers at which `slack.notify_milestone` sends a progress message. |

### `session` block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `planner_model` | `str` | `"claude-opus-4-7"` | Claude model for planner sessions. |
| `executor_model` | `str` | `"claude-opus-4-7"` | Claude model for executor sessions. |
| `cli_path` | `str` | `""` | Absolute path to the `claude` binary. Empty string resolves via `shutil.which("claude")`. |

---

## Termination conditions

Evaluated by `termination.check_all(...)` at the top of every iteration in
priority order. Once a condition fires, the supervisor writes `FINAL_REPORT.md`
and exits.

| # | Condition | Default threshold | Exit code |
|---|-----------|-------------------|-----------|
| 1 | `STOP` sentinel file present | any | 30 (MANUAL_STOP) |
| 2 | Iteration cap exceeded | `budget.iterations_max` (default 70) | 10 (ITER_BUDGET_HIT) |
| 3 | Dollar cap exceeded | `budget.dollars_max` (default $80) | 11 (DOLLAR_BUDGET_HIT) |
| 4 | Wall-clock cap exceeded | `budget.wall_seconds_max` (default 86400 s) | 12 (WALL_BUDGET_HIT) |
| 5 | Plateau — top-K run IDs unchanged across window | `plateau.top_k=3`, `plateau.window_iters=6` | 13 (PLATEAU) |
| 6 | Loop fingerprint — last 3 iterations identical and non-empty | hardcoded 3 | 14 (LOOP_REPEATING) |
| 7 | Dedup-rejected streak — planner adds zero ideas N times in a row | `loop_detection.dedup_reject_streak=4` | 15 (LOOP_NO_NEW_IDEAS) |
| 8 | Brainstorm exhausted — no queued items remain after planner | checked after planner phase | 16 (BRAINSTORM_EXHAUSTED) |
| 9 | Planner crash streak — consecutive non-zero exit codes | hardcoded 2 | 20 (PLANNER_FAILED) |
| 10 | Executor crash streak — consecutive non-zero exit codes | hardcoded 3 | 21 (EXECUTOR_FAILED) |

Conditions 1–4 are evaluated before any history is needed and take precedence
unconditionally. Conditions 5–10 require at least some iteration history and are
evaluated in the order shown. `brainstorm_exhausted` (condition 8) is checked by
the supervisor inline after the planner phase, not inside `check_all`.

---

## Running the autoloop

### Via Makefile (recommended)

```bash
# Smoke-test: single iteration + live dashboard
make autoloop-once PROJECT=california_housing_demo

# Full loop (runs until a termination condition fires)
make autoloop PROJECT=california_housing_demo

# Override iteration and dollar budgets
make autoloop PROJECT=california_housing_demo ITERS=5 BUDGET=10

# Check prerequisites before starting
make autoloop-prereqs PROJECT=california_housing_demo

# Show current status (budget consumed, last 3 iterations)
make autoloop-status PROJECT=california_housing_demo

# Halt a running loop gracefully at the next iteration boundary
make autoloop-stop PROJECT=california_housing_demo

# Launch the live dashboard (port defaults to 8765)
make autoloop-dashboard PROJECT=california_housing_demo PORT=8765
```

`make autoloop-once` and `make autoloop` both call `scripts/autoloop.sh`, which
activates the venv, starts the dashboard server in a companion process, and
launches the supervisor. The live dashboard is accessible at
`http://localhost:8765/` while the loop is running.

### Raw CLI

```bash
source .venv/bin/activate

# Initialise a project (first time only)
python -m libs.autoloop init --project california_housing_demo

# Run one iteration
python -m libs.autoloop run --project california_housing_demo --once

# Run the full loop with budget overrides
python -m libs.autoloop run \
    --project california_housing_demo \
    --budget 20 \
    --dollars 15 \
    --wall-hours 4

# Check prerequisites
python -m libs.autoloop check-prereqs --project california_housing_demo

# Write a STOP sentinel
python -m libs.autoloop stop --project california_housing_demo
```

The supervisor exits with a structured exit code (see `SupervisorExitCode` in
`supervisor.py`). Use `--dry-run` to step through the logic without spawning
real sessions.

---

## Defense in depth

The autoloop uses three layers to prevent a rogue session from mutating
infrastructure code, outcome definitions, or control-state files.

### Layer 1 — Per-session `--settings` deny patterns

`libs/autoloop/prompts/denied_writes.txt` contains regex patterns. Before
every `Edit`, `Write`, or `MultiEdit` tool call inside a planner or executor
session, the `autoloop-write-guard.sh` PreToolUse hook matches the target file
path against these patterns and denies the call with exit code 2 if it matches.
Protected categories: `harness.py`, `eda.py`, `discovery.py`, `libs/`,
`outcomes/`, `entities/`, `autoloop/config.yaml`, `autoloop/iterations.jsonl`,
`autoloop/budget.json`, `autoloop/failure_modes.jsonl`, `autoloop/idea_registry.md`.

The hook also denies `git commit`, `git push`, `git reset`, `git checkout`,
`git clean`, and `git stash` in any Bash command — git mutations are reserved
for the supervisor.

### Layer 2 — PreToolUse hook enforcement

The hook is injected into every session via `--settings` (written by
`_write_settings` before each `claude -p` invocation). `bypassPermissions` mode
is set at the CLI level so the session does not prompt for confirmation; the
hook is the actual safety boundary. Because the hook is path-based regex
matching, it catches file paths regardless of how the session constructs them.

### Layer 3 — Git status audit and scope-limited commit

After a successful executor session, `_commit_iteration` in `supervisor.py`
runs `git status --porcelain` and compares each dirty path against a hardcoded
`allowlist_globs` list. Only paths matching the allowlist are staged with
`git add --`. The allowlist covers:

- `projects/{project}/features/*/...`
- `projects/{project}/runs/run_*.py` and `run_*/`
- `projects/{project}/runs/results.jsonl`, `eda_results.jsonl`, `discovery_results.jsonl`
- `projects/{project}/runs/CHAMPION.md`
- `projects/{project}/autoloop/brainstorm.jsonl`, `iterations.jsonl`, `budget.json`, `failure_modes.jsonl`, `idea_registry.md`, `logs/iter_*.jsonl`

Anything outside this list is never staged, even if a session wrote it. This
means a bug in a session that creates an arbitrary file outside the allowlist
will leave the file untracked but not committed.

---

## Debugging

### Brainstorm parse failures

Symptom: supervisor exits with `BRAINSTORM_EXHAUSTED` immediately after the
planner, or the planner result shows `brainstorm_added=0` every iteration.

Check `projects/{name}/autoloop/logs/iter_NNN_planner.jsonl` for `ValidationError`
lines. The most common cause is a `BrainstormSource.kind` value not in the
`Literal` set (`"discovery"`, `"eda"`, `"user"`, `"seed"`, `"reflection"`,
`"planner"`). The planner prompt instructs the session to use `"planner"` for
self-generated ideas; sessions that use `"model"` or `"harness"` will fail
validation silently.

### Supervisor exits with no iteration activity

If the supervisor exits immediately, inspect the exit code. Exit code 40 means
`check_prereqs` failed — run `make autoloop-prereqs` to see which check failed.
Exit code 0 with no `iterations.jsonl` rows means `--once` was passed but
something short-circuited. If the process exits with no output, `tee` in
`autoloop.sh` may have masked the exit code. Inspect `iterations.jsonl` directly
or re-run with the raw CLI to see stderr.

### Champion predictions missing

If gap-finder fails with "predictions.parquet not found", the champion run's
predictions file was pruned or never written. Re-run the champion script
directly:

```bash
source .venv/bin/activate
python -m projects.california_housing_demo.runs.run_01
```

The executor prompt references `CHAMPION.md` to determine which run to treat as
baseline; if `CHAMPION.md` is stale, update it to a run that has a valid
`predictions.parquet` under `runs/{run_id}/artifacts/`.

### Dirty git tree blocking prereqs

`check_prereqs` requires a clean working tree (check name: `git working tree clean`).
Stash the relevant paths before starting:

```bash
git stash push -m "wip" -- projects/california_housing_demo/
```

### `cadence: every_n` rejected in config

The `GapFinderCfg.cadence` field is a `Literal`. The accepted values are exactly:

- `never`
- `every_iter`
- `every_n_iters`
- `on_demand`

`every_n` (without `_iters`) will fail Pydantic validation. `check_prereqs` runs
`AutoloopConfig.model_validate(...)` and will report the ValidationError with the
offending field path before the loop starts.

---

## Where the code came from

The autoloop was designed in
`/Users/aaronmyran/.claude/plans/please-review-my-harness-refactored-elephant.md`,
a plan that went through consensus review from Gemini and GPT before
implementation began. That plan is the canonical record of the architectural
decisions: why the supervisor is pure Python, why sessions are ephemeral with
fresh context per iteration, why brainstorm.jsonl is event-sourced, and why
champion selection is deliberately kept outside the loop. The implementation
follows the plan closely; where it diverges, the code is authoritative.
