#!/bin/zsh
# Production serving for ultiarena.hu (single process: API + built SPA on :8000).
# Caddy terminates TLS on 80/443 and reverse-proxies here — see infra/Caddyfile.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[serve] building web app…"
(cd apps/web && npm run build --silent)
echo "[serve] starting uvicorn on 0.0.0.0:8000 (AI_WORKERS=${AI_WORKERS:-default})"
exec python3 -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --no-access-log
