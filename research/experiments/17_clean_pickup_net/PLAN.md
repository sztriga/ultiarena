# Exp 17 — Clean pickup net (no calibration)

**Goal.** Replace the exp 15 v2 + isotonic-calibrator stack with one net
trained end-to-end on the deployment distribution (random α=0 deals).
Output should be drop-in for `auction_v2.py` and beat the calibrated v2
on the same 3000 seeds.

## Locked eval setup

- **Held-out seeds:** 100000–102999 (same as exp 15 baseline and exp 16
  auction). Training MUST use a disjoint seed range
  (`SEED_BASE = 800_000_000` and up).
- **Defenders:** god. **Soloist:** PIMC32. Matches all prior baselines
  so the new numbers compare cleanly.
- **PASS_PENALTY:** −2 (game rule, not tunable here).

## Eval tiers — run in order, stop on fail

### Tier 1 — calibration (cheap, gate)

Fresh 50k α=0 deals per contract. New net's raw output vs god labels.
Same diagnostic as `experiments/16_pre_pickup/fit_calibration.py`:

- MAE, Brier per contract
- 6-bin (predicted, actual) table per contract

**Pass:** no bin has |predicted − actual| > 0.05.
**Fail:** if a bin diverges by more than that, the net didn't learn the
deployment distribution — go back, don't waste hours on Tier 3.

### Tier 2 — single-soloist GP/deal (replicates exp 15)

Seeds 100000–102999. P0 forced soloist per contract. Compare to:

| pickup decider | GP/deal | source |
|---|---:|---|
| PIMC256 (oracle ceiling) | TBD | run this fresh |
| calibrated v2, threshold=0 | +0.41 | exp 15 |
| calibrated v2, aggressive thresholds | +0.80 | exp 15, current SOTA |
| **exp 17 net, decision rule: bid if EV > 0** | ? | target |

**Pass:** match or beat +0.80 without per-contract thresholds.
**Soft fail (+0.41 only):** EV head didn't learn the tail; calibration
was doing the real work. Diagnose before Tier 3.

### Tier 3 — auction drop-in (replicates exp 16)

Swap exp 17 net into `auction_v2.py` (replace `CalibratedPickupNet`).
Same 3000 seeds. PASS_PENALTY=−2.

| metric | exp 16 calibrated v2 | exp 17 |
|---|---:|---:|
| P0 GP/deal | −0.46 | ? |
| P1 GP/deal | +0.23 | ? |
| P2 GP/deal | +0.24 | ? |
| sum (zero-sum check) | 0 | must be 0 |
| piros parti freq / won% | 25.4% / 16.5% | ? |
| betli won% | 10.0% | ? |
| durchmars won% | 0.0% | ? |

**Pass:** zero-sum holds; P0 ≥ −0.46 (no regression). Bonus: piros parti
or betli/durchmars bleeders shrink.

### Tier 4 — confusion vs exp 16 (qualitative)

On those 3000 seeds, cross-tab exp 17 pickup decisions vs calibrated v2's.
Where they disagree, who's right vs god. This is the "what actually
changed" read.

## Design choices (decide before datagen)

### Q1 — labels: binary win or full EV?

Current v2 predicts P(make). EV is then `gp_value × (2p − 1)` for fixed-stake
contracts (parti/ulti/betli/durchmars). For these contracts, **binary label
is sufficient and the EV head is mathematically redundant** — same model,
different output transformation.

For piros-modified contracts (piros parti, piros ulti) the GP is just 2×
the base, still linear in p_win.

**Proposal: keep binary labels, derive EV at inference.** Avoids EV-label
noise from game-flow variance. If we later add silent-bonus modeling, we'd
revisit.

### Q2 — datagen: α=0 only, or mixed?

Per-contract α=0 positive rates (measured during exp 16 calibration):
- parti: 23.2% → 1M deals → 232k positives. Fine.
- ulti: 25.7% → 1M → 257k. Fine.
- betli: **0.3%** → 1M → 3000 positives. **Sparse.**
- durchmars: **0.1%** → 1M → 1000 positives. **Very sparse.**

The whole point of exp 17 is to learn the right answer in deployment, and
in deployment betli/durchmars are *supposed* to be rare positives. So the
net mostly needs to predict ~0 and only occasionally fire — which 3000
positives might cover, especially with a multi-head shared body that
borrows representation from parti/ulti.

**Proposal: α=0 only, 1M per contract.** If Tier 1 calibration fails on
betli/durchmars specifically, *then* mix in a smaller α-biased tail for
those two contracts and reweight.

### Q3 — architecture

Reuse `vnet/pickup/multihead.py` (`MultiHeadPickupNet`: 36→128→64 body,
4×(64→32→1) heads, ~21k params). Already exists, already integrated. Train
with binary cross-entropy on combined contract data, batched per-contract
inside each step.

### Q4 — featurization

Reuse `vnet/pickup/features.py` (`featurize(hand, trump, has_trump)`,
36-dim). No change. Decoupling the data work from the feature work — if
we want richer features later, that's a separate exp.

### Q5 — discard strategy in datagen

Current gen_god_labels uses ONE random discard per deal. For exp 17 the
net needs to learn to pick the *best* discard, not just score a random
one. Two options:

- **(a) Same as before:** one random discard per deal, 1M deals. Net
  learns the average over discards. Decision rule at inference: enumerate
  all `C(12,2)=66` discards, pick argmax.
- **(b) Best-discard label:** enumerate all 66 discards per deal, take
  the max p_win as the label. More expensive (66× god calls per deal —
  way too slow for 1M deals).

**Proposal: (a).** The current pipeline already does this and it works.
Argmax over 66 discards at inference is cheap (single forward pass with
batch=66).

## File structure

```
experiments/17_clean_pickup_net/
  PLAN.md                       this file
  results.md                    (written at end)
  gen_alpha0.py                 datagen, α=0, multi-contract
  train.py                      multi-head training on α=0 god labels
  tier1_calibration.py          fresh 50k/contract diagnostic
  tier2_single_soloist.py       reuse exp 15 harness with new net
  tier3_auction.py              auction_v2 with new net
  tier4_confusion.py            cross-tab vs exp 16
  net_v17.pt                    trained weights
```

## Open question for milan before datagen kicks off

Are you OK with α=0-only and seeing how betli/durchmars Tier 1 looks
before deciding to mix? Or do you want to hedge and start with a
mixed dataset (e.g. 700k α=0 + 300k α=biased per contract)?

The honest answer is α=0-only is the cleaner experiment — it directly
tests "is calibration really necessary." If it fails on betli/durchmars,
that's a meaningful negative result that tells us mixing or upweighting
is mandatory.
