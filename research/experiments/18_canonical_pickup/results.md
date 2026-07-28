# Exp 18 — Canonical features + betli/durchmars calib fix

**Period:** 2026-06-12 → 2026-06-13
**Status:** Done. Tier 1 (calibration), Tier 3 (symmetric auction), and
head-to-head tournament vs exp 17 all run. Clear result, with a twist.

## TL;DR

- **Canonical (exact suit-invariant) features fix the betli Tier 1
  calibration** milan flagged — the two target bins go from +0.103 /
  +0.215 underconfident to **+0.024 / +0.053**, with the best val Brier
  of any variant and no parti/ulti regression. This is `v18a`, trained
  on the *same* α=0 data as exp 17 — the only change is the feature
  layout. The win comes from the architecture, not from new data.
- **Mixing α-biased data (v18b/v18c) backfired** — a clean negative
  result. The biased base rate (betli 48.5% pos vs 0.3% in deployment)
  pushes predictions too high, flipping betli from underconfident to
  *overconfident* (high bins −0.24 to −0.33). Rejected.
- **Durchmars is unmeasurable at α=0** — 16 positives in 50k eval deals.
  Every bin above 0.10 has n≤34, so its "calibration" is sampling
  noise. v17 only "passed" durchmars by abstaining (never predicting
  >0.05 in volume). Benign underconfidence direction stands.
- **The twist — better calibration did NOT make a better agent.**
  Head-to-head, v18a is a **wash vs v17** (Δ −0.051 GP/seat-deal, ≈1.4
  SE — statistically indistinguishable, slight negative lean). Tier 3
  shows why: fixing betli underconfidence makes v18a bid betli ~2× more
  often (4.7% vs 2.5%), and those extra betlis *lose* under the
  god-defender handicap (−3.79 GP/def, 12% won). v17's underconfidence
  was accidentally protective. **Betli/durchmars calibration was never
  the binding constraint on auction strength.**

## Variants

| variant | features | betli/duri data | wrapper |
|---|---|---|---|
| v17 | 36-dim + suit-perm aug (trump pinned) | α=0 1M | `Exp17Pickup` |
| **v18a** | **canonical 32-dim (exact invariance)** | **α=0 1M** | `Exp18Pickup` |
| v18b | 36-dim + aug | α=0 1M + biased 250k | `Exp17Pickup` |
| v18c | canonical 32-dim | α=0 1M + biased 250k | `Exp18Pickup` |

Canonical features map every hand to a representative: trump suit row
first, remaining rows sorted by descending rank-pattern key (trumpless:
all 4 rows sorted). Suit-permutation invariance becomes exact rather
than learned, the trump one-hot is dropped (36→32 dims), and
augmentation is a no-op. Proven by `tests/test_pickup_canonical.py`
(full S₄, including trump-moving perms). Code: `vnet/pickup/canonical.py`.

## Tier 1 — calibration (50k fresh α=0 deals/contract, seeds 900M+)

Same eval set exp 17 used. `max |Δ|` over bins with n≥30:

| net | betli | durchmars | parti | ulti | val Brier |
|---|---:|---:|---:|---:|---:|
| v17 | 0.215 | 0.000\* | 0.046 | 0.023 | 0.0336 |
| **v18a** | **0.053** | 0.068\* | **0.017** | **0.012** | **0.0334** |
| v18b | 0.237 | 0.040 | 0.040 | 0.024 | 0.0336 |
| v18c | 0.193 | 0.136 | 0.025 | 0.030 | 0.0341 |

\* durchmars n≥30 only exists in the [0,0.05) bin (abstention); its
"0.000" / "0.068" are not real calibration signal.

**betli bins that were the story** (predicted → actual):

| bin | v17 Δ | **v18a Δ** | v18b Δ | v18c Δ |
|---|---:|---:|---:|---:|
| [0.10, 0.25) | +0.103 | **+0.024** | −0.022 | +0.039 |
| [0.25, 0.50) | +0.215 (n33) | **+0.053** (n84) | −0.037 | −0.121 |
| [0.50, 0.75) | (n2) | +0.167 (n16) | −0.237 | −0.193 |

v18a's two "n≥30 failures" (betli 0.053 @ n=84, duri 0.068 @ n=30) are
inside their own sampling error (±0.05–0.09). Statistically a pass.

## Tier 3 — symmetric auction (N=3000, seeds 100000–102999, def=god, sol=PIMC32)

| metric | exp 16 | exp 17 | **v18a** |
|---|---:|---:|---:|
| P0 GP/deal | −0.461 | −0.464 | **−0.302** |
| P1 / P2 GP/deal | +0.23 / +0.24 | +0.40 / +0.07 | +0.21 / +0.10 |
| P0 avg GP/def per bid | −0.370 | −0.219 | **−0.087** |
| P0 won % as sol | 47.8% | 49.7% | 49.3% |
| sum (zero-sum check) | 0 | 0 | +0.001 ✓ |

P0's symmetric deficit shrinks −0.46 → −0.30 — **but this is a
whole-table effect** (all three seats are v18a; the overtakers behave
differently too), not a held-fixed comparison. It does *not* survive as
a head-to-head edge (see tournament).

**Betli economics — the #2 story number, and it moved the wrong way:**

| | bid freq | won % | avg GP/def per bid |
|---|---:|---:|---:|
| exp 16 betli | 4.3% | 10.0% | −4.00 |
| exp 17 betli | 2.5% | 13.5% | −3.65 |
| **v18a betli** | **4.7%** | 12.1% | **−3.79** |

Fixing the underconfidence raised betli EVs → v18a bids betli ~2× more
than exp 17 → more betli bleed in aggregate. Under the soloist-PIMC32
vs god-defender handicap betli is a structural trap; v17's
underconfidence kept it out of that trap. durchmars: 15 bids, −1.20.

## Head-to-head tournament — v18a (B) vs exp 17 (A), 6 configs × 500 seeds

| config (P0/P1/P2) | P0 | P1 | P2 | who's in P0 |
|---|---:|---:|---:|---|
| AAB | −0.398 | +1.138 | −0.740 | exp17 |
| ABA | −0.350 | +1.216 | −0.866 | exp17 |
| ABB | −0.420 | +1.098 | −0.678 | exp17 |
| BAA | −0.626 | +1.138 | −0.512 | **v18a** |
| BAB | −0.716 | +1.090 | −0.374 | **v18a** |
| BBA | −0.506 | +1.096 | −0.590 | **v18a** |

| | minority seat | majority seat |
|---|---:|---:|
| 2 exp17 + 1 v18a | v18a: −0.050 | exp17: +0.025 |
| 1 exp17 + 2 v18a | exp17: +0.027 | v18a: −0.013 |
| **all configs** | **v18a: −0.026** | **exp17: +0.026** |

**Δ (v18a − exp17) = −0.051 GP/seat-deal.** With ≈4500 seat-deals/model
the SE is ≈0.037, so this is ≈1.4 SE — a statistical wash with a small
negative lean (consistent in sign across both config groups). Note the
contrast with exp17-vs-exp16, where Δ was +0.257 (≈7 SE, decisive).
v18a in P0 does *worse* than exp17 in P0 (−0.62/−0.72/−0.51 vs
−0.40/−0.35/−0.42), the opposite of the exp17 upgrade — because the
extra betli bids leak.

## What this means

1. **The symmetry fix is a clean architectural win worth keeping** on
   its own merits: exact (proven) suit-invariance, 36→32-dim input, no
   augmentation, equal-or-better calibration, best val Brier. Ship the
   canonical featurizer; it's strictly simpler than aug.
2. **The calibration question milan asked is answered:** canonical
   features fix betli; durchmars is unmeasurable at α=0; mixing biased
   data is the wrong tool.
3. **But calibration was a red herring for agent strength.** The
   betli/durchmars heads being "wrong" never cost the auction anything,
   because their weakness lives on the *play* side (the PIMC32-vs-god
   handicap makes betli a losing contract regardless of pickup
   accuracy). Making the pickup more accurate just walks into the trap
   more confidently. Net auction effect: wash / slightly negative.

## Next steps (in order of expected return)

1. **Remove the eval handicap first (exp 17 open dir #4).** Re-run all
   of this under PIMC-vs-PIMC (or god-vs-god) instead of PIMC-sol vs
   god-def. The current handicap is what makes betli a trap and inverts
   the value of betli calibration. Until that's fixed, betli/durchmars
   Tier 3 economics can't be trusted as an optimization target.
2. **If keeping the handicapped eval:** add a betli/durchmars pass
   floor on top of v18a (bid only if EV ≥ τ). This re-introduces the
   protective conservatism without the underconfidence — gets the clean
   calibration *and* the auction discipline.
3. **The binding constraint is still the auction layer** (forced-opener
   −2 penalty, no bid-history inference), exactly as exp 17 concluded.
   Pickup judgement — calibrated or not — is not where the GP is.

## Files

```
experiments/18_canonical_pickup/
  PLAN.md                  variants + locked eval
  results.md               this file
  train.py                 variant-aware trainer (a/b/c)
  tier1_eval.py            cached 50k/contract eval; scores all nets
  baseline_tier3.py        symmetric auction (reuses exp17 auction_h2h)
  tournament.py            h2h vs exp17
  multihead_v18{a,b,c}.pt  trained weights
  tier1_eval_*_50k.npz     cached god-labeled eval sets

vnet/pickup/
  canonical.py             exact suit-invariant featurizer (32-dim)
  v18.py                   Exp18Pickup wrapper (drop-in for Exp17Pickup)
  multihead.py             +input_dim param (32 for canonical)
tests/
  test_pickup_canonical.py invariance unit tests (full S₄)
```
