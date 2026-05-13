# ds3 Makefile
#
# Ergonomic entrypoints for the autoloop, the live dashboard, and dev setup.
# Override defaults with env vars on the command line, e.g.:
#   make autoloop-once PROJECT=civic_shout_action_rate_increase
#   make autoloop PORT=9000

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
        autoloop autoloop-once autoloop-dashboard autoloop-render \
        autoloop-prereqs autoloop-status autoloop-stop \
        dev-setup install-deps \
        leaderboard-civic-shout leaderboard-render-civic-shout

# ─── Help ──────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Autoloop · one-command kickoff"
	@echo "─────────────────────────────────────────────────────────────────"
	@echo "  make autoloop-once              Single iter + live dashboard"
	@echo "  make autoloop                   Full run to cap + live dashboard"
	@echo "  make autoloop-dashboard         Live dashboard only (no autoloop)"
	@echo "  make autoloop-render            Render dashboard HTML once"
	@echo "  make autoloop-prereqs           Check prereqs for current project"
	@echo "  make autoloop-status            Print current state from JSON files"
	@echo "  make autoloop-stop              Graceful stop (after current iter)"
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
	  if [ -f tmp/visualize/autoloop/serve_dashboard.py ]; then \
	    python tmp/visualize/autoloop/serve_dashboard.py --port $(PORT); \
	  else \
	    echo "Live server not built yet — falling back to --watch render"; \
	    python tmp/visualize/autoloop/render_dashboard.py --watch; \
	  fi

autoloop-render:
	@. $(VENV_ACTIVATE) && python tmp/visualize/autoloop/render_dashboard.py

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
