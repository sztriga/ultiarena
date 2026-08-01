#!/bin/zsh
# Production serving for ultiarena.hu (single process: API + built SPA on :8000).
# Caddy terminates TLS on 80/443 and reverse-proxies here — see infra/Caddyfile.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# uvicorn's AI-pool workers are spawn processes that INHERIT the listening socket —
# a kill -9 on uvicorn leaves them holding :8000, so the next start silently serves
# stale code (cost us a confusing debug session 2026-08-02). Clear them first.
pkill -9 -f "uvicorn apps.api.main" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
sleep 1
lsof -ti tcp:8000 2>/dev/null | xargs kill -9 2>/dev/null || true

echo "[serve] building web app…"
(cd apps/web && npm run build --silent)
echo "[serve] starting uvicorn on 0.0.0.0:8000 (AI_WORKERS=${AI_WORKERS:-default})"
exec python3 -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --no-access-log
