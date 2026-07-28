# Exp 18 — Canonical features + betli/durchmars data fix

**Started:** 2026-06-12

Two independent fixes to the exp 17 net, ablated separately:

1. **Canonical (suit-invariant) features.** P(make) is invariant under
   relabeling suits as long as the trump label moves too (piros ×2 is
   applied outside the net). Exp 17 only exploited this partially:
   augmentation with trump pinned (6× for parti/ulti, 24× for
   betli/duri), and the invariance was learned, not exact. Exp 18 maps
   every hand to a canonical representative instead: trump suit row
   first, remaining suit rows sorted by descending rank-pattern key
   (trumpless: all 4 rows sorted). Input shrinks 36 → 32 dims, the
   trump one-hot disappears, augmentation becomes a no-op.
   Code: `vnet/pickup/canonical.py`, wrapper `vnet/pickup/v18.py`.

2. **Betli/durchmars positive starvation.** 1M α=0 deals give only
   ~2,800 betli and ~400 durchmars positives — the source of the
   Tier 1 underconfidence (betli mid-bin Δ +0.215, duri +0.157). Fix:
   mix in the α-biased god-labeled 250k sets already on disk from
   exp 15 (betli α=1.5 → 48.5% pos, duri α=3.0 → 34.7% pos). Labels
   are god-truth, so this is pure covariate shift; whether bin-level
   calibration on α=0 survives the shift is exactly what Tier 1 gates.
   Parti/ulti data untouched.

## Variants

| variant | features | betli/duri data |
|---|---|---|
| v18a | canonical 32-dim | α=0 1M only |
| v18b | exp 17 (36-dim + aug) | α=0 1M + biased 250k |
| v18c | canonical 32-dim | α=0 1M + biased 250k |
| v17  | (baseline, weights from exp 17) | |

Val split is carved from the α=0 portion only, so best-epoch selection
and val tables stay on the deployment distribution and comparable
across variants.

## Eval (locked, same as exp 17)

- **Tier 1:** 50k fresh α=0 deals/contract, seeds 900M+ (identical to
  exp 17's Tier 1 → directly comparable), god-labeled once and cached
  to npz. All four nets scored on the same set.
  Pass: no bin (n≥30) with |pred − actual| > 0.05; parti/ulti must not
  regress. **The story numbers: betli bins [0.10,0.25) +0.103 and
  [0.25,0.50) +0.215, duri [0.10,0.25) +0.157 → inside ±0.05.**
- **Tier 3:** symmetric auction, N=3000 seeds 100000–102999, def=god,
  sol=PIMC32, PASS_PENALTY=−2. Story number: betli avg GP/def per bid
  (exp 17: −3.65 at 2.5% freq; exp 16: −4.00) moving toward 0.
- **Head-to-head tournament:** winner variant vs exp 17, 6 seat
  configs × 500 seeds (exp 17 vs 16 gave Δ +0.257 GP/seat-deal).

Tier 1 picks the variant; only the winner goes to Tier 3 + tournament.

## Files

```
train.py                 variant-aware trainer (a / b / c)
tier1_eval.py            gen+cache 50k/contract eval set; score all nets
baseline_tier3.py        symmetric auction runner (reuses exp 17 auction_h2h)
tournament.py            h2h vs exp 17
results.md               written at end
```
