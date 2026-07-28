#!/usr/bin/env bash
# Run the API and the Vite dev server together. Ctrl-C stops both.
#
#   ./apps/dev.sh          # API on 8000, Web on 5173
#   API_PORT=9000 ./apps/dev.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8000}"

# Sanity: ensure web deps are installed
if [ ! -d "$REPO_ROOT/apps/web/node_modules" ]; then
  echo "→ Installing web dependencies (first run)…"
  (cd "$REPO_ROOT/apps/web" && npm install)
fi

# Prefer the project venv if present (has the training/inference deps such
# as sb3_contrib that the trained-model adapters need).
if [ -x "$REPO_ROOT/.venv/bin/uvicorn" ]; then
  UVICORN="$REPO_ROOT/.venv/bin/uvicorn"
else
  UVICORN="uvicorn"
fi

# Start API in background; track PID so we can clean up on exit
echo "→ Starting API on http://127.0.0.1:$API_PORT (using $UVICORN)"
(cd "$REPO_ROOT" && "$UVICORN" apps.api.main:app --reload --port "$API_PORT" --host 127.0.0.1) &
API_PID=$!

cleanup() {
  echo
  echo "→ Stopping API (pid $API_PID)…"
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Vite in foreground; Ctrl-C stops both via the trap
echo "→ Starting Vite on http://127.0.0.1:5173"
(cd "$REPO_ROOT/apps/web" && npm run dev)
