# UltiArena deployment (ultiarena.hu)

```
internet ── Cloudflare edge (TLS, DNS) ── cloudflared tunnel ── Caddy 127.0.0.1:8080 ── uvicorn 127.0.0.1:8000
                                                                (gzip, headers,          (FastAPI: API + built SPA,
                                                                 body-size cap)           AI worker pool, limits)
```

No inbound port is ever open on this machine (the ISP is DS-Lite — no public IPv4
anyway). `cloudflared` dials OUT to Cloudflare; visitors reach the app only through
the tunnel.

## Security model

- **Loopback binds everywhere.** uvicorn and Caddy listen on `127.0.0.1` only. This
  is load-bearing, not cosmetic: the app's per-IP abuse limits (`apps/api/limits.py`)
  trust the `CF-Connecting-IP` header, which is only unspoofable while the tunnel is
  the sole way in. Never rebind to `0.0.0.0`.
- **No password (by choice).** Abuse is handled in the app: per-IP request rate,
  per-IP concurrent AI work, session caps with oldest-eviction. To re-gate the site,
  uncomment `basic_auth` in `infra/Caddyfile` and restart Caddy.
- **Secrets.** The tunnel credential is `~/.cloudflared/*.json` (never in the repo).
  `infra/Caddyfile` is gitignored; `infra/Caddyfile.example` is the committed shape.

## Running it

**Manual (a terminal per process):**

```sh
./apps/serve.sh                          # build SPA + uvicorn :8000  (stop: ./apps/stop.sh)
caddy run --config infra/Caddyfile       # :8080
cloudflared tunnel run ultiarena         # edge
```

**As services (survives reboots and crashes):**

```sh
./infra/services.sh install    # render infra/launchd templates → ~/Library/LaunchAgents, start all
./infra/services.sh status     # job states + health probes (app + edge)
./infra/services.sh deploy     # after a git pull: rebuild SPA, restart the app
./infra/services.sh logs       # tail data/logs/{app,caddy,tunnel}.log
./infra/services.sh uninstall  # back to manual mode
```

launchd restarts any crashed job within ~10s. The app job runs `serve.sh` with
`SKIP_BUILD=1` so a crash loop never spins npm; `deploy` is the explicit build.

**These are LaunchAgents** — they run while this user is logged in. For an unattended
box: System Settings → enable automatic login, and keep the Mac awake on AC power:

```sh
sudo pmset -c sleep 0          # never system-sleep on charger
sudo pmset -c disablesleep 1   # optional: keep serving with the lid closed
```

## Never do this

- `pkill -f multiprocessing.spawn` — research tournaments on this machine run the
  same kind of spawn workers; `apps/stop.sh` exists precisely so restarts are
  targeted (pidfile process group + `:8000` socket holders only).
- Binding the app or Caddy to `0.0.0.0` (see security model).
- Editing `~/Library/LaunchAgents/hu.ultiarena.*.plist` by hand — they are rendered
  from `infra/launchd/*.template`; edit the template and re-run `install`.

## Operational notes

- `data/games.db` (SQLite, WAL) records every finished game — safe to read while the
  server writes. `data/serve.pid` is the app's pidfile.
- Health: `https://ultiarena.hu/api/health` returns `{"status":"ok", ...}` plus live
  session counts — exempt from rate limits, safe for uptime monitors.
- All knobs (`RATE_LIMIT_RPM`, `MAX_SESSIONS_PER_IP`, `AI_WORKERS`, …) live in
  `ulti/config.py`; `RATE_LIMIT_RPM=0` disables the whole limits layer for local work.
