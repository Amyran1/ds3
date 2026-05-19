# ds3 Makefile
#
# Ergonomic entrypoints for the autoloop, the live dashboard, and dev setup.
# Override defaults with env vars on the command line, e.g.:
#   make autoloop-once PROJECT=civic_shout_action_rate_increase
#   make autoloop PORT=9000

# agent-tooling-overlay: worktree targets (auto-managed; do not edit)
-include .claude/make/worktree.mk
# end agent-tooling-overlay worktree

PROJECT ?= california_housing_demo
PORT    ?= 8765

# Runtime cap overrides (empty = use config.yaml defaults)
ITERS   ?=
BUDGET  ?=

# Lowercase aliases — `make autoloop iters=10 budget=50` reads naturally
iters   ?=
budget  ?=
ifneq ($(iters),)
ITERS := $(iters)
endif
ifneq ($(budget),)
BUDGET := $(budget)
endif

# Treat the venv activator as the canonical "is the venv ready" check.
VENV_ACTIVATE := .venv/bin/activate

.DEFAULT_GOAL := help
.PHONY: help \
        autoloop autoloop-once autoloop-dashboard autoloop-dashboard-restart autoloop-up autoloop-render \
        autoloop-prereqs autoloop-status autoloop-stop autoloop-init-ledger \
        dev-setup install-deps \
        leaderboard-civic-shout leaderboard-render-civic-shout

# ─── Help ──────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Autoloop · one-command kickoff"
	@echo "─────────────────────────────────────────────────────────────────"
	@echo "  make autoloop-once              Single iter + live dashboard"
	@echo "  make autoloop                   Full run to cap + live dashboard"
	@echo "  make autoloop-up                Background dashboard + open browser"
	@echo "  make autoloop-dashboard         Live dashboard (foreground; terminal-pin)"
	@echo "  make autoloop-dashboard-restart Restart dashboard server in background"
	@echo "  make autoloop-render            Render dashboard HTML once"
	@echo "  make autoloop-prereqs           Check prereqs for current project"
	@echo "  make autoloop-status            Print current state from JSON files"
	@echo "  make autoloop-stop              Graceful stop (after current iter)"
	@echo "  make autoloop-init-ledger       Prepend sentinel to unmanaged ledgers"
	@echo ""
	@echo "Options (override on the command line)"
	@echo "─────────────────────────────────────────────────────────────────"
	@echo "  PROJECT=name                    Which project to run"
	@echo "                                  (default: $(PROJECT))"
	@echo "  PORT=8765                       Dashboard HTTP port"
	@echo "  iters=N                         Override iter cap (default: from config.yaml)"
	@echo "  budget=X                        Override dollar cap (default: from config.yaml)"
	@echo ""
	@echo "Examples"
	@echo "─────────────────────────────────────────────────────────────────"
	@echo "  make autoloop-once"
	@echo "  make autoloop-once PROJECT=civic_shout_action_rate_increase"
	@echo "  make autoloop PROJECT=civic_shout_action_rate_increase iters=70 budget=50"
	@echo "  make autoloop iters=20                              # 20 iters, default \$\$ cap"
	@echo "  make autoloop budget=10                             # default iters, \$10 cap"
	@echo "  make autoloop-stop PROJECT=civic_shout_action_rate_increase"
	@echo ""
	@echo "Dev setup"
	@echo "─────────────────────────────────────────────────────────────────"
	@echo "  make dev-setup                  Install all deps via uv"
	@echo "  make install-deps               Install from lockfile"
	@echo ""

# ─── Autoloop ──────────────────────────────────────────────────────────────

autoloop-once:
	@PROJECT=$(PROJECT) PORT=$(PORT) MODE=--once ITERS=$(ITERS) BUDGET=$(BUDGET) ./scripts/autoloop.sh

autoloop:
	@PROJECT=$(PROJECT) PORT=$(PORT) MODE= ITERS=$(ITERS) BUDGET=$(BUDGET) ./scripts/autoloop.sh

autoloop-dashboard:
	@. $(VENV_ACTIVATE) && \
	  python -m libs.autoloop.dashboard.serve --port $(PORT) --project $(PROJECT)

autoloop-dashboard-restart:
	@./scripts/restart-dashboard.sh $(PROJECT)

autoloop-up:
	@./scripts/restart-dashboard.sh $(PROJECT)
	@sleep 1
	@command -v open >/dev/null 2>&1 && open http://localhost:$(PORT)/ || true
	@echo ""
	@echo "Dashboard at http://localhost:$(PORT)/ (logs: tail -f /tmp/dashboard.log)"
	@echo "Now run autoloop in a separate terminal: make autoloop PROJECT=$(PROJECT)"

autoloop-render:
	@. $(VENV_ACTIVATE) && python -m libs.autoloop.dashboard.render --project $(PROJECT)

autoloop-prereqs:
	@. $(VENV_ACTIVATE) && python -m libs.autoloop check-prereqs --project $(PROJECT)

autoloop-status:
	@. $(VENV_ACTIVATE) && python -m libs.autoloop status --project $(PROJECT)

autoloop-stop:
	@mkdir -p projects/$(PROJECT)/autoloop
	@touch projects/$(PROJECT)/autoloop/STOP
	@echo ">> STOP sentinel placed at projects/$(PROJECT)/autoloop/STOP"
	@echo ">> The autoloop will exit after the current iteration completes."
	@echo ">> Remove the sentinel with: rm projects/$(PROJECT)/autoloop/STOP"

autoloop-init-ledger:
	@. $(VENV_ACTIVATE) && python scripts/autoloop-init-ledger.py --project $(PROJECT)

# ─── Dev setup ─────────────────────────────────────────────────────────────

dev-setup:
	@. $(VENV_ACTIVATE) && uv pip install -e ".[dev]"

install-deps:
	@. $(VENV_ACTIVATE) && uv pip install -r uv.lock

# ─── Leaderboard ───────────────────────────────────────────────────────────

leaderboard-civic-shout:
	@. $(VENV_ACTIVATE) && python -m libs.leaderboard civic_shout_action_rate_increase --watch --port $(PORT)

leaderboard-render-civic-shout:
	@. $(VENV_ACTIVATE) && python tmp/visualize/civic_shout_action_rate_increase/render_leaderboard.py
