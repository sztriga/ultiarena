#!/bin/zsh
# UltiArena service manager — installs the serving chain as launchd user agents so it
# survives reboots and crashes (launchd restarts a dead job within ~10s).
#
#   ./infra/services.sh install     render templates → ~/Library/LaunchAgents + start all
#   ./infra/services.sh uninstall   stop + remove the agents (manual serve.sh still works)
#   ./infra/services.sh start|stop  start/stop all three jobs (installed agents stay)
#   ./infra/services.sh restart     kickstart all three (picks up config changes)
#   ./infra/services.sh deploy      git-safe redeploy: build the SPA, then restart the app
#   ./infra/services.sh status      one line per job + a live health probe
#   ./infra/services.sh logs        tail all three logs
#
# Jobs (all logs under data/logs/):
#   hu.ultiarena.app     apps/serve.sh  → uvicorn on 127.0.0.1:8000 (SKIP_BUILD=1)
#   hu.ultiarena.caddy   caddy          → 127.0.0.1:8080, proxies to :8000
#   hu.ultiarena.tunnel  cloudflared    → ultiarena.hu edge → :8080
#
# NOTE these are LaunchAgents: they run while this user is logged in. For an
# unattended box enable automatic login, and keep the Mac awake on AC power:
#   sudo pmset -c sleep 0        (never sleep on charger)
#   sudo pmset -c disablesleep 1 (optional: ignore the closed lid too)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
TEMPLATES="$REPO_ROOT/infra/launchd"
JOBS=(hu.ultiarena.app hu.ultiarena.caddy hu.ultiarena.tunnel)
UID_N=$(id -u)

_render() {
    local caddy cloudflared pydir
    caddy=$(command -v caddy) || { echo "caddy not found (brew install caddy)"; exit 1; }
    cloudflared=$(command -v cloudflared) || { echo "cloudflared not found (brew install cloudflared)"; exit 1; }
    pydir=$(dirname "$(command -v python3)")
    mkdir -p "$AGENTS_DIR" "$REPO_ROOT/data/logs"
    for job in $JOBS; do
        sed -e "s|__REPO__|$REPO_ROOT|g" \
            -e "s|__CADDY__|$caddy|g" \
            -e "s|__CLOUDFLARED__|$cloudflared|g" \
            -e "s|__PYDIR__|$pydir|g" \
            "$TEMPLATES/$job.plist.template" > "$AGENTS_DIR/$job.plist"
    done
}

_bootstrap() {
    for job in $JOBS; do
        launchctl bootstrap "gui/$UID_N" "$AGENTS_DIR/$job.plist" 2>/dev/null \
            || echo "[services] $job already loaded"
    done
}

_bootout() {
    for job in $JOBS; do
        launchctl bootout "gui/$UID_N/$job" 2>/dev/null || true
    done
}

case "${1:-}" in
    install)
        [[ -f "$REPO_ROOT/infra/Caddyfile" ]] \
            || { echo "infra/Caddyfile missing — copy infra/Caddyfile.example first"; exit 1; }
        [[ -d "$REPO_ROOT/apps/web/dist" ]] \
            || { echo "apps/web/dist missing — run: cd apps/web && npm run build"; exit 1; }
        _render
        _bootstrap
        echo "[services] installed + started. check: ./infra/services.sh status"
        ;;
    uninstall)
        _bootout
        for job in $JOBS; do rm -f "$AGENTS_DIR/$job.plist"; done
        "$REPO_ROOT/apps/stop.sh"
        echo "[services] uninstalled"
        ;;
    start)
        _bootstrap
        ;;
    stop)
        _bootout
        "$REPO_ROOT/apps/stop.sh"
        ;;
    restart)
        for job in $JOBS; do
            launchctl kickstart -k "gui/$UID_N/$job" 2>/dev/null \
                || echo "[services] $job not loaded — run install first"
        done
        ;;
    deploy)
        echo "[services] building web app…"
        (cd "$REPO_ROOT/apps/web" && npm run build --silent)
        launchctl kickstart -k "gui/$UID_N/hu.ultiarena.app" 2>/dev/null \
            || echo "[services] app job not loaded — run install first"
        echo "[services] deployed"
        ;;
    status)
        for job in $JOBS; do
            if launchctl print "gui/$UID_N/$job" >/dev/null 2>&1; then
                pid=$(launchctl print "gui/$UID_N/$job" 2>/dev/null | awk '/^\tpid = /{print $3}')
                echo "$job: loaded${pid:+ (pid $pid)}"
            else
                echo "$job: NOT loaded"
            fi
        done
        printf "app health: "
        curl -s -m 3 http://127.0.0.1:8000/api/health || echo "unreachable"
        echo
        printf "edge health: "
        curl -s -m 8 https://ultiarena.hu/api/health || echo "unreachable"
        echo
        ;;
    logs)
        tail -n 30 -f "$REPO_ROOT"/data/logs/*.log
        ;;
    *)
        echo "usage: $0 install|uninstall|start|stop|restart|deploy|status|logs"
        exit 1
        ;;
esac
