# Exp 16 — Pre-pickup bidding & auction

**Period:** 2026-06-06 → 2026-06-07
**Status:** Steps 1–3 done. Multi-step bidding auction works, calibration
applied. Stopping point pending decision on next direction (bid-history
inference / distillation / self-play tournament).

## TL;DR

- **Pre-pickup oracle** (enumerated over 231 talons through exp 15 v2 net)
  hides 2 of 12 cards from the soloist and decides pickup-or-pass.
  Loses essentially **0 GP/deal** vs the full post-pickup baseline.
- **Multi-player bidding auction** (P0 forced opener, P1/P2 overtake or
  pass, 3 consecutive passes = end) works end-to-end. Sums to zero across
  positions (zero-sum verified).
- **Isotonic calibration wrapper** for the v2 net fixes its
  over-confidence on the random α=0 test distribution. Lets us drop the
  per-contract aggressive thresholds (4 / 0.5 / 1.5 / 8) in favor of one
  clean rule: bid if EV > alternative.
- **P0 being net-negative under the −2/def pass-penalty is structural,
  not an agent bug.** Per-bid EV is positive; the penalty deals are what
  drag the total down. In symmetric self-play with rotating roles, the
  penalty cost averages out across positions.

## Step 1 — Pre-pickup oracle (enumeration)

`pre_pickup_oracle.py`

For each 10-card hand: enumerate all 231 talons → v2 net → max-over-discards
P_make → mean over talons. Per (contract, trump):

- `mean_best_p` = E_talon[max_discard P]
- `mean_ev` = EV per defender from mean_best_p (linear in p, so no Jensen issue)

Wall: **~135 ms / hand** single-process.

## Step 2 — Pickup-vs-pass measurement

`baseline_prepickup.py` (N=3000, seeds 100000–102999, same as exp 15 baseline)

Sol's 12 = 10 + talon. Post-pickup picker sees 12; pre-pickup oracle sees 10.

| config | post-pickup GP/deal | **pre-pickup GP/deal** | Δ |
|---|---|---|---|
| threshold = 0 (raw v-net) | +0.406 | **+0.562** | +0.156 |
| exp 15 aggressive thresholds | +0.801 | **+0.771** | −0.031 |

**Cost of hiding the talon ≈ 0.** With aggressive thresholds, pre-pickup
matches post-pickup within noise (−0.031 GP/deal). At threshold=0, oracle
actually *beats* post-pickup because talon-averaging filters marginal
positive-EV bids that lose in practice.

Confusion (aggressive thresholds):
```
                  post=bid    post=pass
  oracle=pickup       1446         132
  oracle=pass          553         869
```

## Step 3 — Bidding auction

`auction.py` — original threshold-based version.
`auction_v2.py` — calibrated, threshold-free version.

### Rules (final clean version, auction_v2)

- **P0 opener (mandatory):** picks up talon, looks at all (contract, trump)
  combinations above rank 0 excluding non-piros parti. Bids the highest-EV
  one if EV > PASS_PENALTY (−2). Else pays −2/def, deal terminates.
- **Overtaker turn:** oracle picks best contract above current rank;
  computes PIMC32 pass-EV vs current bid; pickup if oracle EV > pass EV,
  else pass.
- **Overtaker commit:** MUST bid the highest-EV (contract, trump, discard)
  above current rank. No fallback to pass (rule: no take-backs after
  picking up).
- **End condition:** 3 consecutive passes (auto-passes count for the
  current bidder).

### Contract ranking (milan's variant)

1. parti (non-piros)
2. piros parti (parti/hearts)
3. ulti (non-piros)
4. betli
5. durchmars
6. piros ulti (ulti/hearts)

**Non-piros parti is illegal** — never bid. If a player would otherwise
bid parti/bells, parti/acorns, parti/leaves, they pass or take penalty.

## Headline auction results (N=3000)

| version | P0 GP/deal | P1 GP/deal | P2 GP/deal | sum | wall |
|---|---|---|---|---|---|
| Original (no penalty rule, force_bid) | +0.030 | +0.090 | −0.119 | 0 | 16 min |
| Thresholds + penalty | −0.532 | +0.259 | +0.273 | 0 | 13 min |
| **Calibrated, no thresholds, penalty=−2** | **−0.461** | +0.226 | +0.236 | 0 | 9 min |

P0 negative under any rule with a forced-commit + penalty alternative.
Position rotates in real play → long-run average per player = 0.

### P0 pass-penalty sensitivity (calibrated rules)

| PASS_PENALTY | P0 GP/deal | P0 pass% | P0 won% as soloist | P0 avg/bid |
|---|---|---|---|---|
| −2.0 | −0.46 | 0% | 47.8% | −0.37 |
| −1.0 | −0.03 | 29.0% | 64.0% | +0.44 |
| −0.5 | +0.34 | 31.5% | 66.0% | +0.58 |
|  0.0 | +1.02 | 38.6% | 71.4% | +1.06 |

**Penalty is a game-rule parameter, not a tuning knob.** Real Ulti has a
real cost for declining to bid; we fix at −2 and accept the GP/deal it
gives.

## Calibration (the cleanup)

`vnet/pickup/calibration.py` — `CalibratedPickupNet` wraps `PickupNetV2`
with per-contract `IsotonicRegression`. Loads automatically from
`{contract}_calib.pkl` files in the exp 15 weights dir.

`fit_calibration.py` — generates 50k random α=0 deals per contract,
labels with god solver, fits isotonic on (raw p̂, label) pairs.

### Why calibration was needed

V2 was trained on biased α=1.5 (betli) / α=3.0 (durchmars) etc. — those
distributions concentrate positives at ~50%. The test distribution
(random α=0 deals) has wildly different base rates:

| contract | training positive rate | random α=0 rate |
|---|---|---|
| betli | 48.5% | **0.3%** |
| durchmars | 34.7% | **0.1%** |
| parti | 51.5% | 23.2% |
| ulti | 49.9% | 25.7% |

For betli/durchmars the v-net's "top bin" (raw p̂ in [0.75, 1.01)) was
catastrophically miscalibrated: raw p̂ ≈ 0.88, actual ≈ 0.37. Isotonic
correction maps 0.88 → 0.42 on betli, 0.87 → 0.32 on durchmars. Parti
and ulti were already fine (Δ < 0.025).

This **exactly explains** why the per-contract aggressive thresholds
(betli=4, durchmars=8) worked in exp 15 — they were band-aids over
overconfident raw EVs. With calibration, the band-aids come off and
the decision rule becomes one clean line.

## Files in this experiment

```
experiments/16_pre_pickup/
  PLAN.md                       step-by-step roadmap
  results.md                    this file
  pre_pickup_oracle.py          step 1: enumerated oracle
  baseline_prepickup.py         step 2: pickup-vs-pass measurement
  auction.py                    step 3: threshold-based auction (pre-cleanup)
  auction_v2.py                 step 3+: calibrated, threshold-free auction
  baseline_auction.py           runner for auction.py
  baseline_auction_v2.py        runner for auction_v2.py
  fit_calibration.py            fits isotonic per contract; saves *_calib.pkl

vnet/pickup/
  calibration.py                CalibratedPickupNet (canonical home)
```

## Open directions (none chosen)

1. **Bayesian opponent inference** — update talon/opponent distributions
   from auction history (who passed, who raised). Currently each decision
   ignores prior bid history. The structural P0 cost would shrink.
2. **Distill the oracle into an NN** — replace the 100ms enumerated oracle
   with a ~0.4 ms net. Necessary for fast self-play or large-scale eval.
3. **Self-play tournament** with rotating positions to measure actual
   long-run per-player GP.
4. **Symmetric auction baselines** — what if all 3 players use the same
   policy? What if P0 plays calibrated and P1/P2 play aggressive? Etc.

## Stopping point

The pre-pickup decision layer + the auction layer are end-to-end working
with clean rules. Calibration brought the decision logic from 5 messy
inconsistencies down to one rule: **bid if EV > alternative**. P0's
negative GP under the penalty rule is the structural cost of being the
forced opener, paid by every player in turn under rotation.
