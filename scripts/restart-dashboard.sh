#!/usr/bin/env bash
# Kill whatever is bound to $PORT (default 8765), then start a fresh dashboard
# for $PROJECT (default california_housing_demo). Uses .venv/bin/python directly
# so no venv-activate is needed. Logs go to /tmp/dashboard.log.
#
# Usage:
#   ./scripts/restart-dashboard.sh civic_shout_action_rate_increase
#   PORT=8765 ./scripts/restart-dashboard.sh civic_shout_action_rate_increase

set -u
PORT="${PORT:-8765}"
PROJECT="${1:-${PROJECT:-california_housing_demo}}"

cd "$(dirname "$0")/.."

# Kill anyone currently bound to the port. SIGKILL to bypass anything caught.
for PID in $(lsof -ti :"$PORT" 2>/dev/null); do
  echo ">> killing PID $PID on port $PORT"
  kill -9 "$PID" 2>/dev/null || true
done
# Also clean up any orphan curl connections to that port
for PID in $(lsof -tiTCP:"$PORT" -sTCP:ESTABLISHED 2>/dev/null); do
  kill -9 "$PID" 2>/dev/null || true
done
sleep 1

# Launch the new server using the venv's python (no activate needed)
nohup ./.venv/bin/python tmp/visualize/autoloop/serve_dashboard.py \
  --port "$PORT" --project "$PROJECT" >/tmp/dashboard.log 2>&1 &
disown
sleep 1.5

NEWPID=$(lsof -ti :"$PORT" 2>/dev/null | head -1)
if [[ -n "$NEWPID" ]]; then
  echo ">> new dashboard PID $NEWPID at http://localhost:$PORT/"
  echo ">> logs: tail -f /tmp/dashboard.log"
else
  echo ">> server failed to start; last 20 lines of /tmp/dashboard.log:"
  tail -20 /tmp/dashboard.log
fi
