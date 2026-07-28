# trickster

An **Ulti** AI (base-event bidder + full web app), with an isolated **Snapszer** engine kept alongside.

The Ulti app is the front door: run it and you land straight in the Ulti starting window (no tabs).

## Layout

```
apps/            Ulti web app — the product
  api/           FastAPI: main (play/puzzle/pis routers), play, puzzle, pis, serialize
  web/           Vite/React SPA — boots straight into PlayVsAI (Új játék + Villámtalon)
  dev.sh         run API + web together
ulti/            Ulti AI/engine library
  card.py game.py            base card + game model
  solvers/ scoring/ eval/     PIMC / oracle / matchup + dealers
  bidding/                    the base-event bidder (ladder, bidder, auction, provider, kontra, …)
  betli/defense.py            learned betli-defense net
  vnet/ agents/ training/     value nets, agents, training pipeline
ultisolver/      decoupled Cython Ulti endgame solver (_solver_core, games/ulti model)
snapszer/        ISOLATED Snapszer engine (its own `trickster` package + api + web) — run standalone
models/ulti/     deployed bidder heads + betli models
assets/cards/    Hungarian Tell-deck card images (served at /cards)
research/        all 37 experiments (frozen research; not on the runtime path)
tests/           ulti golden transcripts + snapszer engine tests
```

`ulti/` + `apps/` have **zero** `trickster` imports — the name `trickster` belongs only to the
isolated snapszer package, so the two games can never collide.

## Run the Ulti app

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[fast]"                      # deps + cython
python setup_cython.py build_ext --inplace    # builds ultisolver._solver_core + snapszer _fast_minimax
./apps/dev.sh                                  # API :8000 + Vite :5173 → open http://127.0.0.1:5173
```

## Run Snapszer (standalone)

```bash
PYTHONPATH=snapszer uvicorn api.main:app --port 8010     # + snapszer/web via vite
```

## Tests

```bash
PYTHONPATH=snapszer pytest tests/snapszer     # snapszer engine
python tests/golden/capture.py                # golden transcript of the Ulti AI (behavior net)
```

Provenance: consolidated from the `oldtawer` fork + this repo's Snapszer engine; the pre-consolidation
state of both is tagged `pre-consolidation`.
