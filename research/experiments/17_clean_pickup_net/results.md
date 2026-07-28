# Exp 17 — Clean pickup net (no calibration)

**Period:** 2026-06-08 → 2026-06-09
**Status:** Datagen, training, Tier 1, Tier 3, and head-to-head tournament
done. Tier 2 (single-soloist) and Tier 4 (confusion matrix) skipped — the
tournament result is the cleanest "is it better" signal.

## TL;DR

- **Replaced the v2-net + isotonic-calibrator stack with one multi-head net
  trained end-to-end on α=0 god-labeled deals.**
- **Tier 1 (calibration):** parti & ulti pass cleanly (every bin within
  ±0.05). Betli & durchmars *underconfident* in mid bins on small n —
  failure mode is "bids too rarely," not "bids overconfidently."
- **Tier 3 (symmetric auction vs exp 16):** P0 GP/deal essentially
  unchanged (−0.46 → −0.46), but **per-bid quality improved across the
  board** (non-piros ultis flipped negative → positive).
- **Head-to-head tournament (the real test):** **exp 17 beats exp 16 by
  +0.26 GP per seat-deal.** Biggest gain when exp 17 sits in P0 — the
  forced-opener deficit shrinks from ~−0.55 to ~−0.19 (a +0.35 swing).
  exp 17 also gains more when it's in the minority (+0.20) than majority
  (+0.09), suggesting it specifically exploits exp 16's mistakes.
- **The auction-layer P0 deficit is the binding constraint, not pickup
  quality.** Moving the headline further needs auction-level changes
  (opponent bid-history inference, dynamic pass-penalty), not better
  pickup judgement.

## Step 1 — α=0 datagen

`gen_alpha0.py` — 1M random-α deals per contract, one random discard per
deal, binary god-solver label. Wall time on laptop (with one sleep
interruption mid-run): ~8h total, dominated by parti (~6h).

Positive rates (the actual deployment distribution):

| contract | rate | positives in 1M |
|---|---:|---:|
| parti | 22.9% | 228,944 |
| ulti | 25.7% | ~257,000 |
| betli | 0.28% | ~2,800 |
| **durchmars** | **0.04%** | **~400** |

Durchmars is rarer than the prior calibration estimate (0.1%) suggested.
Genuinely a 1-in-2500 contract on random deals — the game-theoretic floor
is that high.

## Step 2 — Training

`train.py` — multi-head net (`MultiHeadPickupNet`, 21k params: 36→128→64
body, 4×(64→32→1) heads). 30 epochs, batch 256/contract = 1024 total,
suit-permutation augmentation, BCE loss. ~9 min wall.

Best val Brier (mean): 0.045. Train-val calibration on parti/ulti is
tight (max bin Δ +0.046). Betli/durchmars val bins are too sparse to be
informative.

## Step 3 — Tier 1: held-out calibration check

`tier1_calibration.py` — fresh 50k α=0 deals/contract, god-labeled.

| contract | result | tightest miss |
|---|---|---|
| parti | ✅ all bins within ±0.05 | mid bin +0.046 |
| ulti  | ✅ all bins within ±0.02 | trivial |
| betli | ⚠️ underconfident bin [0.10, 0.25): raw 0.153 → actual 0.256 (+0.103, n=223); [0.25, 0.50): raw 0.330 → actual 0.545 (+0.215, n=33) | +0.215 |
| durchmars | ⚠️ underconfident bin [0.10, 0.25): raw 0.156 → actual 0.312 (+0.157, n=28) | +0.157 |

**The miss direction is "too cautious" not "too eager"** — the inverse of
v2's failure mode. Net will bid betli/durchmars less than it should
rather than overbid them.

## Step 4 — Tier 3: symmetric auction (replicates exp 16)

`baseline_tier3.py` — N=3000 seeds 100000-102999, same setup as exp 16.

| metric | exp 16 calib v2 | **exp 17** |
|---|---:|---:|
| P0 GP/deal | −0.461 | −0.464 |
| P1 GP/deal | +0.226 | +0.397 |
| P2 GP/deal | +0.236 | +0.067 |
| sum | 0 | 0 ✓ |
| P0 avg/bid as sol | −0.370 | **−0.219** |
| P0 won % as sol | 47.8% | 49.7% |

Per-contract performance (all winners):

| contract | exp 16 GP/def | exp 17 GP/def | exp 17 bid freq |
|---|---:|---:|---:|
| ulti/hearts (piros) | +0.86 | +0.86 | 31.8% |
| parti/hearts (piros) | −1.34 | −1.28 | 25.8% |
| ulti/leaves | −0.27 | **+0.16** | 13.4% |
| ulti/acorns | −0.14 | **+0.15** | 13.0% |
| ulti/bells | +0.27 | **+0.83** | 13.1% |
| betli | −4.00 | −3.65 | **2.5%** (was 4.3%) |
| durchmars | −6.00 | 0.00 (n=10) | 0.3% |

**Non-piros ultis flipped negative → positive.** Betli got more selective
(2.5% vs 4.3%) and the hit rate rose (13.5% won vs 10.0%) — consistent
with the Tier 1 "underconfident, bids less often" pattern.

**But P0 total didn't move:** the per-bid gains were absorbed because
exp 17 also bids more often (2096 vs 2066 P0 bids), and because all 3
players now make better overtake decisions (P1/P2 punish P0 harder when
they do overtake). Net: P0 was already taking most deals, and the
binding constraint on the headline is the auction structure (forced
opener + −2 penalty + 26% piros-parti dog rate), not pickup judgement.

## Step 5 — Head-to-head tournament (the real eval)

`tournament.py` — runs all 6 mixed seat assignments × 500 seeds = 3000
deal-runs. Each model gets 4500 seat-deals total.

### Per-seat-deal aggregate

| group | minority | majority |
|---|---:|---:|
| 2 exp16 + 1 exp17 | **exp17: +0.20** | exp16: −0.10 |
| 1 exp16 + 2 exp17 | exp16: −0.18 | **exp17: +0.09** |
| **all mixed configs** | **exp17: +0.129** | **exp16: −0.129** |

**Δ = +0.257 GP per seat-deal in exp 17's favor.** Zero-sum holds exactly
in both groups (sanity passes).

### Per-config breakdown

| config (P0/P1/P2) | P0 GP | P1 GP | P2 GP | notes |
|---|---:|---:|---:|---|
| AAB | −0.560 | +0.736 | −0.176 | exp17 in P2 |
| ABA | −0.502 | +0.968 | −0.466 | exp17 in P1 |
| ABB | −0.536 | +0.862 | −0.326 | exp17 in P1+P2 |
| **BAA** | **−0.190** | +0.806 | −0.616 | **exp17 in P0** |
| **BAB** | **−0.196** | +0.788 | −0.592 | **exp17 in P0+P2** |
| **BBA** | **−0.346** | +1.154 | −0.808 | **exp17 in P0+P1** |

**The P0 deficit drops from ~−0.55 to ~−0.19 when exp 17 sits there.**
+0.35 GP/deal on the worst seat. exp 17's opener judgement is materially
better — but only visible when an opponent isn't exp 17 itself.

### Why exp 17 wins more when it's alone

The +0.20 (minority) vs +0.09 (majority) split says exp 17 specifically
exploits exp 16's mistakes. Alone, it pockets the GP that two exp 16s
leak. As a pair, two exp 17s have to split the same total exploitation
budget. This is consistent with exp 17 being a strict pickup-judgement
upgrade (not a tactical innovation that other agents could miss
together).

## Calibration: what changed vs exp 16

Exp 16's calibration was a band-aid: train on biased α to oversample
positives, then bend the raw output with an isotonic regression fit on
α=0. Two stitched-together hacks.

Exp 17 trains directly on the deployment distribution. The calibration
diagnostic is now a *check*, not a corrector — and on parti/ulti the net
emerges already-calibrated. Betli/durchmars are the residual problem
(too few positives in 1M α=0 deals), but their failure mode is benign
(underconfident = under-bids, not over-bids).

## Files

```
experiments/17_clean_pickup_net/
  PLAN.md                      eval plan (frozen seeds, 4-tier criteria)
  results.md                   this file
  gen_alpha0.py                α=0 god-label datagen
  train.py                     multi-head training
  multihead_v17.pt             21k-param trained weights
  *_god_alpha0_1M.npz          training data (4 contracts, ~30 MB total)
  tier1_calibration.py         held-out calibration diagnostic
  tier3_auction.py             auction copy with exp 17 picker
  baseline_tier3.py            tier 3 runner with per-contract table
  auction_h2h.py               per-seat-picker auction harness
  tournament.py                6-config head-to-head runner
  run_all.sh                   datagen → train → tier1 → tier3 orchestrator

vnet/pickup/
  v17.py                       Exp17Pickup wrapper (CalibratedPickupNet-compatible)
```

## Open directions

1. **Fix betli/durchmars tails.** Mix α-biased data for these two contracts
   to give the heads more positive examples (current α=0 gives 2,800 betli
   and 400 durchmars positives — not enough). Don't rebuild parti/ulti
   data; they're fine.
2. **Auction-level changes** to move P0's −0.46 further: opponent
   bid-history inference (currently each decision ignores prior bids),
   dynamic pass-penalty, or fold the auction into a learned policy net
   that's trained against itself.
3. **Distill the 231-talon pre-pickup oracle** using exp 17 as the new
   teacher — same plan as before, just with a cleaner v-net underneath.
4. **PIMC-vs-PIMC re-eval** of all numbers. The current "soloist PIMC32
   vs defenders god" handicap drives the negative-GP-for-soloists pattern
   throughout. Symmetric play would show absolute numbers closer to zero;
   relative deltas between models should be stable.

## Stopping point

Exp 17 is a clean +0.26 GP/seat-deal upgrade over exp 16 in head-to-head
play, and removes the two-step "train biased + calibrate post-hoc" hack
in favor of one model trained on the deployment distribution. The
pickup-judgement question is now answered to first order; further
headline gains live in the auction layer, not the value layer.
