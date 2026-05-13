#!/usr/bin/env bash
# scripts/autoloop.sh
#
# One-command autoloop kickoff. Starts the live dashboard server in the
# background, opens it in the browser, then runs the autoloop in the
# foreground so events stream to the terminal. Ctrl-C stops both.
#
# Usage (typically via Makefile):
#   PROJECT=name PORT=8765 MODE=--once ./scripts/autoloop.sh
#
# Env vars:
#   PROJECT  — DS project name        (default: california_housing_demo)
#   PORT     — dashboard HTTP port    (default: 8765)
#   MODE     — supervisor mode flag   (default: empty = full run; --once = single iter)

set -euo pipefail

PROJECT="${PROJECT:-california_housing_demo}"
PORT="${PORT:-8765}"
MODE="${MODE:-}"
ITERS="${ITERS:-}"
BUDGET="${BUDGET:-}"

# Build runtime cap override flags
EXTRA_ARGS=""
[[ -n "$ITERS"  ]] && EXTRA_ARGS="$EXTRA_ARGS --budget $ITERS"
[[ -n "$BUDGET" ]] && EXTRA_ARGS="$EXTRA_ARGS --dollars $BUDGET"

cd "$(dirname "$0")/.."
ROOT="$PWD"

# ─── 1. Activate venv ──────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -f .venv/bin/activate ]]; then
    echo ">> .venv not found at $ROOT/.venv — run 'make dev-setup' first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ─── 2. Prereq check ───────────────────────────────────────────────────────
echo ">> Checking prereqs for project '$PROJECT'..."
if ! python -m libs.autoloop check-prereqs --project "$PROJECT"; then
  echo ""
  echo ">> Prereqs failed. Common cause: dirty working tree from unrelated WIP."
  echo ""
  echo "   To stash and proceed:"
  echo "     git stash push -m 'autoloop wip' -- <paths-to-stash>"
  echo "     make autoloop${MODE:+-once} PROJECT=$PROJECT"
  echo "     git stash pop   # after autoloop finishes"
  echo ""
  exit 1
fi

# ─── 3. Start dashboard in background ──────────────────────────────────────
DASHBOARD_LOG="/tmp/autoloop-dashboard-${PROJECT}.log"
SERVE_SCRIPT="tmp/visualize/autoloop/serve_dashboard.py"
WATCH_SCRIPT="tmp/visualize/autoloop/render_dashboard.py"

echo ""
if [[ -f "$SERVE_SCRIPT" ]]; then
  echo ">> Starting live dashboard server at http://localhost:$PORT/"
  python "$SERVE_SCRIPT" --port "$PORT" > "$DASHBOARD_LOG" 2>&1 &
  DASHBOARD_PID=$!
  DASHBOARD_URL="http://localhost:$PORT/"
elif [[ -f "$WATCH_SCRIPT" ]]; then
  echo ">> Live server not built yet; using file:// watcher fallback."
  python "$WATCH_SCRIPT" --watch > "$DASHBOARD_LOG" 2>&1 &
  DASHBOARD_PID=$!
  DASHBOARD_URL="file://$ROOT/tmp/visualize/autoloop/run-dashboard.html"
else
  echo ">> No dashboard renderer found. Running autoloop without dashboard." >&2
  DASHBOARD_PID=""
  DASHBOARD_URL=""
fi

cleanup() {
  if [[ -n "${DASHBOARD_PID:-}" ]] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
    kill "$DASHBOARD_PID" 2>/dev/null || true
    echo ""
    echo ">> Dashboard stopped (pid $DASHBOARD_PID). Logs at $DASHBOARD_LOG"
  fi
}
trap cleanup EXIT INT TERM

# Give the server a moment to bind / first render
sleep 0.6

# ─── 4. Open browser (macOS) ───────────────────────────────────────────────
if [[ -n "$DASHBOARD_URL" ]] && command -v open >/dev/null 2>&1; then
  open "$DASHBOARD_URL" 2>/dev/null || true
fi

# ─── 5. Run the autoloop in foreground ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Autoloop · project=$PROJECT · mode=${MODE:-full-run}"
[[ -n "$DASHBOARD_URL" ]] && echo " Dashboard: $DASHBOARD_URL"
echo " Ctrl-C to stop. The dashboard server will shut down with it."
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Note: MODE may contain --once (one arg) or be empty.
# shellcheck disable=SC2086
python -m libs.autoloop run --project "$PROJECT" ${MODE} ${EXTRA_ARGS} --verbose

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Autoloop finished. Dashboard still running at $DASHBOARD_URL"
echo " Press Ctrl-C to stop the dashboard and exit."
echo "════════════════════════════════════════════════════════════════════════"

# Keep the script alive so the dashboard stays up for post-run review.
# Ctrl-C triggers the cleanup trap which kills the dashboard PID.
if [[ -n "${DASHBOARD_PID:-}" ]]; then
  wait "$DASHBOARD_PID" 2>/dev/null || true
fi
