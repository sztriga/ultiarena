# Exp 22 — Contract suite: spec → validate → tweak → integrate with the bidding loop

Goal (milan, 2026-06-15): go contract by contract. For each, milan explains how
he wants it to work; I validate whether the current oracle/solver/engine can do
it, tweak code where needed, and finally bring them all together with the
**bidding loop** we built (exp 20 / `auction_h2h.py`, the canon DEBIAS_BID
bidder).

## The foundation we're standing on (exp 21)
- **Oracle** `scoring/oracle.py` — payoff vector, milan-exact silent-100 rule,
  scores bid+silent for parti/ulti/durchmars/40-100/20-100. Single source of truth.
- **Solver** `trickster/_solver_core.pyx` — `EV_MULTI` weighted minimax
  (`set_multi_weights`, `set_multi_cull`); dedicated `EV_PARTI/ULTI/BETLI/DURCHMARS`
  for bid contracts. Cull proven sound for silent ulti.
- **Engine** `engine.py` — `solver_weights(parti, silent_ulti, silent_100,
  silent_dm, bid, has_40, has_20)` → weight vector; `oracle_bid(...)` → BidSet.
  `recipe.py` — marriage detection + silent-100 weights.
- **Validated levers (exp21):** silent ulti +1.25 (sol) / defender denial −0.40;
  silent 100 +0.12; silent duri redundant for sol but defender breaks half the
  soft sweeps; bid 40-100 doubles make-rate (32→65%, +2.8 GP).

## Per-contract workflow (repeat for each)
1. **milan's spec** — how the contract should behave (scoring, thresholds,
   bid vs silent, who can hold what).
2. **Validate** — `harness.run_ab(...)` A/B: does pricing it produce milan's
   behavior? (soloist-side: capture more; defender-side: deny it.) A few lines.
3. **Tweak** — fix the oracle (rates/rules), engine recipe, or solver
   (`set_multi_*` toggles, new component) if the behavior isn't reachable.
4. **Record** — mark the contract DONE in `contracts.md` with the result.

## The harness (`harness.py`)
`run_ab(dealer, filt, oracle_bid, configs, show, n_cand, seed_base)` runs paired
god-vs-god A/B over filtered deals → metric table + B−A deltas (t-stats).
Each config = {sol_contract, sol_weights, def_contract, def_weights}. Both sides
can carry independent multi objectives (per-ply weight set). Dealers: ulti /
parti / duri. Filters: marriage / has_40_only / has_20_only / def_holds_7 / …
Smoke (`_smoke.py`) reproduces silent-ulti +1.25.

## Integration target (the finale)
Wire the engine's per-contract valuation into the **auction**: at bid time, for
each candidate contract, value it via the oracle+solver (multi-payoff EV) and
pick/raise accordingly — replacing/augmenting the deployed value-net bidder in
`experiments/17_clean_pickup_net/auction_h2h.py` (canon `DEBIAS_BID=1`). The
exp20 CFR infra + the seat-rotated tournament are the evaluation harness.

## Status
See `contracts.md`. Ready to start — milan drives the contract order.
</content>
