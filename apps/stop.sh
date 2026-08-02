#!/bin/zsh
# Stop the UltiArena app server (uvicorn + its AI worker processes) — and NOTHING else.
#
# Two targeted mechanisms, no broad pkill (research tournaments on this machine run
# the same kind of multiprocessing.spawn workers and must never be touched):
#   1. data/serve.pid → TERM the recorded process GROUP (uvicorn + spawn workers share
#      it), but only after verifying the pid still belongs to our uvicorn.
#   2. Backstop: kill whatever still holds tcp:8000 — spawn workers inherit the
#      listening socket, and a stale holder means the next start silently serves old
#      code (cost us a debug session 2026-08-02). lsof targets ONLY the socket holders.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$REPO_ROOT/data/serve.pid"

if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if ps -o command= -p "$OLD_PID" 2>/dev/null | grep -q "uvicorn apps.api.main"; then
        PGID=$(ps -o pgid= -p "$OLD_PID" 2>/dev/null | tr -d ' ')
        if [[ -n "$PGID" ]]; then
            echo "[stop] TERM process group $PGID (uvicorn pid $OLD_PID + workers)"
            kill -TERM -"$PGID" 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f "$PIDFILE"
fi

STALE=$(lsof -ti tcp:8000 2>/dev/null || true)
if [[ -n "$STALE" ]]; then
    echo "[stop] killing stale :8000 holders: $STALE"
    echo "$STALE" | xargs kill -9 2>/dev/null || true
fi
echo "[stop] done"
