# Bidding loop — protocol

Goal: iterate training strategies / architectures for the bidding model toward
human/superhuman play with no per-contract bleeding. This folder is the **testing
framework**; the bidding *script* lives in `../23_bidding_integration`.

## What's fixed (the harness)
- **Deals**: uniform random, fixed seed bank (`seed_base`, default 500M) → every
  candidate is scored on the SAME deals. Reproducible.
- **Auction**: `auction.run_auction(seed, bid_fn)` — architecture-agnostic, full
  22+11-rung ladder to 48, talon-passing, exp-20 debias.
- **Scorer (inner loop)**: `scorers.god_outcome` — god double-dummy defenders.
  Points games → exact multi-solve + replay + oracle (scores **silent games**);
  trick contracts → `god_says` makeability + rates; combos → component-wise (rare).
- **Metric**: `harness.evaluate(make_bid_fn) → {metric, nonfloor, seat_gp, stats}`.
  - `metric` = mean soloist GP/game (floor-dominated; structural pre-kontra).
  - **`nonfloor`** = mean soloist GP on bids ABOVE piros parti — the discriminating
    signal between bidders.
  - `stats` = per-contract bids / winrate / GP/game → the **bleed view** (you judge).

## How to add a strategy (one file)
Write a module-level factory `make_<name>_bid_fn() -> bid_fn` (signature
`bid_fn(cards12, current_rung) -> (ev, rung, trump, discard, hand10) | None`).
The current baseline is `net_bidder.make_net_bid_fn` (7 calibrated heads →
composer → debias). Then:

```python
from harness import evaluate, print_report
res = evaluate(make_<name>_bid_fn, n=3000)
print_report(res, "<name>")
```

Levers to iterate (cheapest → deepest): debias pctl · per-head calibration ·
rare-head confidence floors (betli/rebetli/20-100 bleed) · net architecture/data
(god vs PIMC labels) · joint composer (drop the independence approx) · a direct
policy/value net (replace the factorization) · self-play.

## Phases
1. **Now** — tighten the current bidder against the god metric; kill the leakers.
2. **Periodic check** (to wire): PIMC defenders + head-to-head vs the deployed
   exp17/20 bidder (`scorers.pimc_outcome` is the stub).
3. **Kontra** — adds the real pass/raise economics; the floor tax becomes a
   decision. (Oracle ×2^level hook exists; ladder/economics next.)

## Baseline (net+composer, calibrated, debias 0.80, god defenders)
Run `python harness.py` (N env). Reference numbers + the bleeders are in STATUS.
