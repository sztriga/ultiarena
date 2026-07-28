#!/usr/bin/env bash
# Run the Ulti API + Vite dev server together. Ctrl-C stops both.
# Open http://127.0.0.1:5173 — it boots straight into the Ulti splash (Új játék + Villámtalon).
#
#   ./apps/dev.sh                 # API on 8000, Web on 5173
#   API_PORT=9000 ./apps/dev.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8000}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # ulti/, ultisolver/, apps/ importable

PY="python"; [ -x "$REPO_ROOT/.venv/bin/python" ] && PY="$REPO_ROOT/.venv/bin/python"
UVICORN="uvicorn"; [ -x "$REPO_ROOT/.venv/bin/uvicorn" ] && UVICORN="$REPO_ROOT/.venv/bin/uvicorn"

# Build the Cython endgame solver once (ultisolver._solver_core).
if ! ls "$REPO_ROOT"/ultisolver/_solver_core*.so >/dev/null 2>&1; then
  echo "→ Building Cython extensions (first run)…"
  ( cd "$REPO_ROOT" && "$PY" setup_cython.py build_ext --inplace )
fi

# Install web deps once.
if [ ! -d "$REPO_ROOT/apps/web/node_modules" ]; then
  echo "→ Installing web dependencies (first run)…"
  ( cd "$REPO_ROOT/apps/web" && npm install )
fi

echo "→ Starting API on http://127.0.0.1:$API_PORT"
( cd "$REPO_ROOT" && "$UVICORN" apps.api.main:app --reload --port "$API_PORT" --host 127.0.0.1 ) &
API_PID=$!
cleanup() { kill "$API_PID" 2>/dev/null || true; wait "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "→ Starting Vite on http://127.0.0.1:5173  (open this — boots to the Ulti splash)"
( cd "$REPO_ROOT/apps/web" && npm run dev )
