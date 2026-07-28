# Exp 19 — Split colorless contracts into structural models

**Period:** 2026-06-13
**Status:** Done. Built CompositePickup (v18a trump + structural betli/duri
nets); ran god_check + bleeders vs v18a. **Negative result with a clear
diagnosis.**

## TL;DR

- Built dedicated structural pickup nets for betli & durchmars (20
  suit-invariant hole/void/length features in the **10-low colorless
  order**, tiny MLP), routed via `CompositePickup` (parti/ulti still
  v18a). Drop-in for the whole harness.
- **A mid-course correction:** structural models trained on the *biased*
  god data calibrate perfectly within-distribution but are badly
  overconfident on α=0 (duri predicts 0.87 where actual is 0.06). The
  biased dealer correlates the soloist hand with favourable opponent
  splits, so `P(win | structure)` is genuinely higher under biased
  dealing — **structure is not a sufficient statistic across split
  distributions.** Retrained on the α=0 deployment data → calibrated.
- **The split did NOT reduce the betli bleed.** Composite betli: 4.3%
  freq, 9.2% won, −4.08 GP/def, 11.5% god-win on committed hands —
  vs v18a's 4.7% / 12.1% / −3.79 / 14.9%. Seat totals unchanged
  (P0 −0.300 vs −0.302).
- **Root cause: argmax-over-66-discards inflation (optimizer's curse),
  not model calibration.** Even a per-hand-calibrated model, maxed over
  66 discards, surfaces an over-valued betli; the low bid floor (betli
  clears −2 at p≥0.3) lets it through. And god itself says the best
  findable betli is only ~11–15% winnable at α=0 vs god — betli is
  structurally a losing bid here. The fix is a **bidding-policy
  threshold**, not a better value net.

## Results (N=3000, seeds 100000–102999, def=god, sol=PIMC32)

### god-win% of committed hands (decision-region overconfidence)

| contract | composite n / god-win% | v18a n / god-win% |
|---|---:|---:|
| ulti/hearts | 931 / 78.0 | 929 / 77.5 |
| parti/hearts | 797 / 21.7 | 790 / 21.5 |
| ulti/bells | 385 / 79.2 | 385 / 80.0 |
| ulti/leaves | 383 / 73.6 | 378 / 73.3 |
| ulti/acorns | 359 / 77.4 | 362 / 76.8 |
| **betli** | **130 / 11.5** | 141 / 14.9 |
| durchmars | 15 / 46.7 | 15 / 40.0 |

Trump contracts unchanged (same v18a — confirms the refactor is clean).
**Betli not improved** (slightly worse). Durchmars n=15 = noise.

### Playout GP

| | composite (freq/won/GP-def) | v18a |
|---|---|---|
| betli | 4.3% / 9.2% / −4.08 | 4.7% / 12.1% / −3.79 |
| durchmars | 0.5% / 46.7% / −0.40 | 0.5% / 40.0% / −1.20 |

Seat totals: P0 −0.300 (v18a −0.302), P1 +0.221 (+0.205), P2 +0.079
(+0.098). No headline movement.

## Why a calibrated model didn't help

The auction commits betli via `argmax` over the 66 possible discards.
Maxing 66 calibrated-but-noisy predictions systematically over-estimates
the chosen option (optimizer's / winner's curse). The committed betlis
therefore have inflated predicted p even though the model is calibrated
per-hand. Low capacity doesn't fix this — the inflation is in the `max`,
not the model. And α=0 betli vs god is genuinely ~11–15% winnable for
the best findable discard, so the only winning move is to bid it less.

## What's worth keeping

- Clean separated architecture, correct 10-low colorless rank order,
  calibrated structural nets, no trump-side regression. Good infra.
- Durchmars structural net is mildly better (GP −1.20→−0.40), but duri
  volume is negligible.

## Next step (the actual fix)

Bidding-policy, not value-model:
1. **Betli pass-threshold** — require predicted p ≥ ~0.6 (margin for the
   argmax inflation) before betli can be bid; sweep the threshold and
   measure GP. Likely cuts most of the −1,070 betli bleed.
2. Or effectively **kill betli** (near-never bid it) and measure.
3. The argmax-over-discards inflation is general (affects all contracts
   mildly) — a held-out / shrinkage estimate of the chosen discard's p
   would debias it everywhere, but betli is where it bites.

## Files

```
experiments/19_colorless_split/
  PLAN.md  results.md
  train_colorless.py     α=0 structural training (betli + duri)
  eval_composite.py      god_check + bleeders for CompositePickup
  colorless_betli.pt  colorless_durchmars.pt
vnet/pickup/
  colorless.py   features + ColorlessStructNet + ColorlessPickup
  composite.py   CompositePickup router
```
