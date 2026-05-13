# ds3 — Data Science v3 project workspace

ds3 is a workspace for harness-evaluated supervised-learning projects with an
autonomous feature-engineering loop on top. Each project defines a harness
(evaluation), an EDA module, a gap-finder, and a set of runs. The autoloop
generates feature ideas, implements them via ephemeral LLM sessions, and
appends results to append-only ledgers — all without touching harness code or
promoting a champion. Production work begins with the `civic_shout_action_rate_increase`
project; `california_housing_demo` is the end-to-end smoke-test target.

---

## Quick start

```bash
make dev-setup            # install deps into .venv via uv
make autoloop-once        # single iteration + live dashboard (smoke test)
make                      # show all available targets
```

Common follow-ups:

- Run the full autonomous loop: `make autoloop PROJECT=california_housing_demo`
- Switch to a different project: `make autoloop PROJECT=civic_shout_action_rate_increase`
- Check prerequisites before starting: `make autoloop-prereqs PROJECT=california_housing_demo`
- View loop status: `make autoloop-status PROJECT=california_housing_demo`
- Halt a running loop: `make autoloop-stop PROJECT=california_housing_demo`

---

## Repo layout

```
libs/         Shared library code
              autoloop/     Autonomous feature-engineering supervisor + session runner
              costs/        API cost tracking and ledger
              ledgers.py    LedgerWriter — typed, append-only run result rows
              run_record.py run_record(...) wrapper enforcing the three-ledger contract
              responses.py  HarnessResponse, EDAResponse, DiscoveryResponse shapes
              container.py  Runtime services container (OpenAI, Anthropic, Mongo, S3)

projects/     DS project workspaces — one per dataset / problem statement
              Each project owns harness.py, eda.py, discovery.py, runs/, features/,
              outcomes/, autoloop/, and GOAL.md

entities/     Cross-project cached datasets shared across multiple projects

tests/        pytest suite

scripts/      Orchestration shell scripts
              autoloop.sh   Invoked by Makefile autoloop targets

tmp/          Ephemeral artifacts (gitignored): agent findings, visualizations, scout output
```

---

## What's inside

**Autoloop** — The autonomous feature-engineering loop lives in `libs/autoloop/`.
See `libs/autoloop/README.md` for architecture, configuration reference,
termination conditions, and debugging guidance.

**Harness / EDA / Discovery** — Each project implements `def harness(...)`,
`def eda(...)`, and `def discover_gaps(...)`. The project-structure conventions,
ownership boundaries, and the run-tandem invariant are documented in `CLAUDE.md`
under "DS v2 Project Structure".

**Run tandem invariant** — `libs/run_record.py` and `libs/ledgers.py` enforce
the rule that every run produces exactly three ledger rows (harness, EDA,
discovery) or none. The three append-only ledgers (`results.jsonl`,
`eda_results.jsonl`, `discovery_results.jsonl`) are populated only via typed
projection methods on the response objects. Hand-built dicts in any ledger are
forbidden.

**Cost tracking** — `libs/costs/` maintains a per-key spend ledger. The
autoloop's `AutoloopBudget` reads from this ledger to enforce dollar caps and
report `api_spend` in the dashboard and Slack notifications.

---

## Adding a new project

1. Create `projects/{name}/`.
2. Write `GOAL.md`, `harness.py` (with `def harness(...)`), `eda.py` (with
   `def eda(...)`), and `discovery.py` (with `def discover_gaps(...)`).
3. Define `outcomes/{outcome_name}/v1/dictionary.yaml` with the outcome variable
   and population filter.
4. Implement `runs/run_01.py` as the baseline run using `run_record(...)`. Run
   it once to produce a `predictions.parquet` file and a row in `results.jsonl`
   with `status=completed`.
5. Set `runs/CHAMPION.md` to document the baseline run (e.g. `run_01`).
6. Create `autoloop/config.yaml`. The fastest path is to copy from
   `projects/california_housing_demo/autoloop/config.yaml` and edit `project`,
   `outcome_variable`, `comparison_group`, `primary_metric`, `goal_metric`, and
   `metric_direction` to match your problem.
7. Run `make autoloop-prereqs PROJECT={name}` to verify all 14 preflight checks
   pass before starting the loop.

---

## Conventions

`CLAUDE.md` at the repo root is the canonical reference for AI agents and human
contributors. It documents the project-structure layout, ownership boundaries,
the run-tandem invariant, the `Container` runtime-services contract, and agent
routing. Any AI agent working in this repo follows the rules there.

---

## Status

Active development. `california_housing_demo` validates the autoloop end-to-end;
production targets begin with `civic_shout_action_rate_increase`.
