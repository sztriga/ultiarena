#!/bin/zsh
# Production serving for ultiarena.hu (single process: API + built SPA).
# Chain: internet → Cloudflare edge (TLS) → cloudflared tunnel → Caddy (127.0.0.1:8080)
#        → THIS (127.0.0.1:8000).
#
# Binds 127.0.0.1 ON PURPOSE: the app trusts the CF-Connecting-IP header for its
# per-IP abuse limits, and that trust only holds if the tunnel is the sole way in.
# A 0.0.0.0 bind would let anything on the LAN spoof the header and bypass every cap.
#
# SKIP_BUILD=1 skips the SPA build (launchd crash-restarts must not npm-build in a loop).
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO_ROOT/data"

"$REPO_ROOT/apps/stop.sh"          # targeted stop of any previous instance (see stop.sh)

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "[serve] building web app…"
    (cd apps/web && npm run build --silent)
fi

echo "[serve] starting uvicorn on 127.0.0.1:8000 (AI_WORKERS=${AI_WORKERS:-default})"
echo $$ > "$REPO_ROOT/data/serve.pid"      # exec keeps this pid → stop.sh can find us
exec python3 -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
