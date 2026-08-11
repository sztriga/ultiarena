# UltiArena public API — design

milan's intent (2026-08-11): expose the machinery — the game database, the god
solver, the full game representation (dealing, legality, scoring), and the frontier
AI — so outside researchers can bring their own agents; potentially fully open
source, a community effort.

## What we actually have to offer (ranked by uniqueness)

1. **The game database** — real recorded games (deal + full auction + every play +
   kontra + marriages + scores), human and AI, growing daily. Nobody else has this;
   it cannot be regenerated locally. This is the community asset.
2. **A cheat-clean game environment** — the multiplayer engine's per-viewer
   snapshot means a seat's view PROVABLY contains no hidden information. An
   external agent occupying a seat cannot cheat even if it wants to — information
   hygiene enforced at the API boundary, not by trust. This is what makes results
   produced against our environment publishable.
3. **The frontier AI as a reference opponent** — a strong, fixed baseline to
   measure against (the "Stockfish role").
4. **The god solver** — exact double-dummy evaluation of any position, any
   contract, including the multi-payoff weights. Expensive (CPU-seconds).
5. **The rules kernel** — dealing, legal moves, the scoring oracle. Cheap, pure.

## The one big insight

**A research match IS a live table.** The stage-2 engine already runs any mix of
human/AI seats with per-viewer snapshots and turn-validated actions over HTTP. An
external agent is just "a player whose browser is a python script": it occupies a
seat, polls its own view, acts on its turn, and the empty chairs are the frontier.
The public match API is therefore a thin, versioned veneer over `Session` — not a
second engine (the same rule that has held all along).

## Design decisions

**D1 — Auth: API keys tied to accounts.** An account (the existing users table)
can mint named API keys (`ua_` + 40 hex, stored HASHED — unlike browser tokens,
keys are long-lived so a DB leak must not yield them). Key ≠ browser token:
separate lifecycle, separate rate class, revocable individually, usage attributed
per key. Research agents therefore have an identity — their games can carry
`players[].user_id` + an agent name, which is what makes a leaderboard possible.
RECOMMENDED over anonymous access: costs one login, buys attribution, quotas,
and revocation. (Anonymous read-only access to game EXPORTS can still be allowed.)

**D2 — Namespace and stability: `/api/v1/*`, separate from the app's private
API.** `/api/play/*`, `/api/live/*` etc. remain internal and unstable; the public
surface is curated, versioned, and documented (FastAPI gives OpenAPI/Swagger for
free — `/api/v1/docs`). Breaking changes → `/api/v2`, v1 kept until sunset.

**D3 — Cost classes.** This runs on one Mac. Every endpoint belongs to a class,
each with its own per-key budget:
  - `free`   (rules kernel, reads): generous rpm.
  - `metered`(bot decisions, match steps): moderate rpm — a PIMC move is ~1-2s.
  - `quota`  (god solver): N solver-jobs/day per key, submitted as ASYNC JOBS
    (submit → job id → poll), executed on the existing ai_pool with a global
    "research lane" concurrency cap so the live site always wins contention.

**D4 — External agents' games are recorded** into the same games.db with
`kind: "bot"`, the agent's name, and the owner's user_id. Default ON (it grows
the dataset — the whole point), documented clearly.

**D5 — Standardized evaluation.** A frozen, versioned seed set (e.g.
`eval-2026-08`, 1000 deals × 3 rotations) playable via the match API; finishing
the set yields an official GP/seat-deal vs the frontier → a public leaderboard.
This converts "we have an API" into "we have a benchmark" — the strongest
community magnet we can build with what exists.

**D6 — Open source is orthogonal to the API and is milan's call.** The hosted API
works whether or not the repo opens. Notable split: the ENGINE (rules, solver,
env) can open source cleanly; the frontier WEIGHTS can stay ours (the API exposes
the frontier's decisions, not its parameters) — Stockfish-model vs Leela-model,
decidable later.

## Surface sketch (v1)

```
AUTH
  POST /api/v1/keys                    (browser-token authed) → create named key
  GET  /api/v1/keys / DELETE …/{id}    manage; shows per-key usage

DATASET (class: free)
  GET  /api/v1/games?contract=&kind=&since=&limit=&cursor=   paginated metadata
  GET  /api/v1/games/{id}                                    full transcript
  GET  /api/v1/export/games.jsonl.gz                         nightly bulk dump

RULES KERNEL (class: free)
  POST /api/v1/deal      {seed?}                          → hands + talon
  POST /api/v1/legal     {position}                       → legal card ids
  POST /api/v1/score     {final_position, bid, kontras}   → full PayoffVector
  (wire format: card ids 0-31, the same shapes /pis/explore already uses)

FRONTIER AS OPPONENT (class: metered)
  POST /api/v1/bot/bid   {my_hand, auction_state}         → the frontier's action
  POST /api/v1/bot/move  {my_view_of_position}            → the frontier's card

MATCHES — the environment (class: metered)
  POST /api/v1/matches         {seats: ["me","frontier","frontier"], seed?}
  GET  /api/v1/matches/{id}    → YOUR seat's snapshot (rev-cursor, same shape the web client eats)
  POST /api/v1/matches/{id}/act {action: bid|pass|pickup|trump|kontra|move, ...}
  → thin veneer over Session(humans={api seat}); cheat-clean by construction

GOD SOLVER (class: quota, async)
  POST /api/v1/solve           {position, contract, weights?} → {job_id}
  GET  /api/v1/solve/{job_id}  → pending | {best_move, value, pv}

EVAL (D5)
  POST /api/v1/eval/runs       {set: "eval-2026-08", agent: "name"} → seeded match sequence
  GET  /api/v1/leaderboard
```

## Rollout stages

- **Stage A**: key minting + `/api/v1` skeleton + rules kernel + dataset reads.
  Small, safe, immediately useful; sets the auth/cost-class machinery.
- **Stage B**: matches (the env veneer) + bot endpoints — "bring your agent".
- **Stage C**: async solver jobs + quotas.
- **Stage D**: eval sets + leaderboard; a tiny `ultiarena` python client package
  (`pip install ultiarena` → `env = ultiarena.Match(key, opponents="frontier")`).

## Decisions (confirmed by milan, 2026-08-11)

1. **D1 ✓ keys per account** — everything on /api/v1 needs a minted key.
2. **D4 ✓ record by default** — agents' finished games enter the shared dataset.
3. **D6 ✓ engine, not weights** — open-source the rules kernel / solver / env
   code; the frontier's trained parameters stay private, exposed only as an
   opponent through the API.
4. **Rollout: A+B first** — built same day (see below).

## Status

- **BUILT 2026-08-11** (stages A+B): `apps/api/apikeys.py` (mint/list/revoke,
  SHA-256-hashed keys, per-key cost-class budgets: free 120/min, metered 30/min),
  `apps/api/public.py` (the `/api/v1` sub-app, own OpenAPI docs at `/api/v1/docs`):
  rules kernel (deal/legal/score), dataset reads (games list + full transcript),
  and MATCHES — the environment veneer over Session (exactly one "me" seat per
  key; empty chairs = frontier; recorded as kind=bot + agent name + owner).
  Tests: `tests/api/test_public_api.py` (9, incl. a @slow full agent-vs-frontier
  deal through /api/v1 alone).
- **Pending**: nightly bulk export file; async solver jobs + quotas (stage C);
  eval seed sets + leaderboard + `pip install ultiarena` client (stage D); a web
  profile page for key management (today: mint via curl with a browser token).
