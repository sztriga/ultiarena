# UltiArena

An **Ulti** AI (base-event bidder + full web app), deployed at **ultiarena.hu**, with an
isolated **Snapszer** engine kept alongside.

The Ulti app is the front door: run it and you land straight in the Ulti starting window —
play against the AI, play live at a 3-seat table, browse your games on the profile, or use
the public research API.

## Layout

```
apps/            Ulti web app — the product
  api/           FastAPI: ONE game engine (engine/auction_flow/kontra_flow/ai_play/snapshots)
                 + routes (play, live tables, me/profile, puzzle, pis, public /api/v1),
                 accounts + api keys, abuse limits, game recording, AI worker pool
  web/           Vite/React SPA — boots straight into PlayVsAI
  dev.sh serve.sh stop.sh      local dev / production serve / targeted stop
ulti/            Ulti AI/engine library
  card.py config.py            base card model + the ONE env-knob table
  solvers/ scoring/ eval/      PIMC + determinization / scoring oracle / matchup + dealers
  bidding/                     the base-event bidder (ladder, bidder, auction, provider, …)
  betli/defense.py             learned betli-defense net
  vnet/ pipeline/              featurizers + the end-to-end frontier-head trainer
ultisolver/      decoupled Cython Ulti endgame solver (_solver_core, games/ulti model)
snapszer/        ISOLATED Snapszer engine (its own `trickster` package + api + web) — run standalone
models/ulti/     deployed bidder heads (LFS) + isotonic calibrations + betli models
assets/cards/    Hungarian Tell-deck card images (served at /cards)
infra/           launchd services + Caddy + Cloudflare tunnel (see infra/README.md)
docs/            MULTIPLAYER.md, PUBLIC_API.md — architecture + decisions
migrations/      one-shot applied epoch scripts (kept as the record of e.g. the suit re-encode)
research/        experiments (frozen research; not on the runtime path)
tests/           api/ + ulti/ unit tests, golden/ transcript harness, snapszer/ engine tests
```

`ulti/` + `apps/` have **zero** `trickster` imports — that package name belongs only to the
isolated snapszer package, so the two games can never collide.

## Run the Ulti app

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,fast]"                  # deps + pytest + cython
python setup_cython.py build_ext --inplace    # builds ultisolver._solver_core + snapszer _fast_minimax
./apps/dev.sh                                 # API :8000 + Vite :5173 → open http://127.0.0.1:5173
```

Production (ultiarena.hu): `./infra/services.sh install` — see `infra/README.md`.

## Run Snapszer (standalone)

```bash
PYTHONPATH=snapszer uvicorn api.main:app --port 8010     # + snapszer/web via vite
```

## Tests

```bash
pytest                                        # tests/ (api + ulti + snapszer units)
python tests/golden/capture.py                # golden transcript of the Ulti AI (behavior net)
cd apps/web && npx vitest run                 # frontend DOM-snapshot suite
```

Provenance (formerly `trickster`): consolidated from the `oldtawer` fork + the original repo's Snapszer engine; the pre-consolidation
state of both is tagged `pre-consolidation`.
