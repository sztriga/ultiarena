# Exp 20 — CFR bidding vs the composite bidder

**Period:** 2026-06-13
**Status:** Done. Built a standard-auction CFR bidder (god-exact leaves) + a
fair tournament harness. CFR beats the **faithful deployed bidder by +0.58
GP/seat-deal (t=9.2)**; vs a clean greedy bidder the edge is small (+0.03 table
/ +0.17 opener) — so most of the win is removing the argmax-over-66 inflation,
not the equilibrium reasoning.

## TL;DR

- Built a full CFR pipeline for the **bidding/auction** phase (not the card
  play): standard 10/10/10+2 auction, **god-exact leaf payoffs**, deployable
  hand buckets, external-sampling MCCFR, and a seat-rotated tournament that
  holds the card-play fixed (PIMC32 soloist vs god defenders) so only bidding
  quality is compared. Both agents see only (own 10 cards, public bid history).
- **CFR conditions overtaking on the auction** — the belief-updating asked for.
  With a player's own hand held fixed, P1's overtake-vs-defend decision *flips*
  depending on whether P0 opened parti vs ulti. The composite is history-blind.
- **vs the FAITHFUL deployed bidder** (talon-averaged argmax-over-66, the real
  logic): **CFR +0.581 GP/seat-deal, t=9.2**, significant on every seat. The
  inflation makes the deployed bidder overbid every contract; CFR solos less
  but far stronger (ulti +3.62 vs +0.59, piros-ulti +7.48 vs +1.08), and never
  bleeds betli/duri (deployed: −6.0/−7.6).
- **vs a CLEAN greedy bidder** (raw-10 eval, no inflation): CFR Δ ≈ +0.03 table
  / +0.17 opener. **Lesson: ~95% of CFR's win over the deployed bidder is the
  inflation removal** — recoverable by a cheap decision-rule change. CFR proves
  the ceiling and adds a robust opener edge.
- **god-leaf training is the corrective.** CFR refuses betli/duri except on the
  rare `hi` bins where they're *genuinely* +EV (betli-hi = 57% god-makeable,
  +0.67/def). The exp19 betli bleed was the argmax-over-66 inflation, not the
  net's calibration — which is fine in aggregate.

## Why a separate value model wasn't the answer (and CFR is the right frame)

The clean composite evaluates each contract on the **raw 10 cards** (one
prediction), so it never does the argmax-over-66-discards that caused the
optimizer's curse — i.e. the betli inflation is a property of the old auction
*logic*, fixed by any clean decision rule. What's left, and what CFR adds, is
**belief-aware bidding**: the non-uniform talon/opponent prior. Both agents use
the *same* value net; we compare bidding *logic*, which is exactly the question
("an alternative bidding logic … CFR").

## Value-model bins are well-calibrated to god outcomes

god-makeable rate (and mean god-EV/def) stratified by the value-model bucket
bin, over 1.8M player-deals:

| action | bin lo | bin md | bin hi |
|---|---|---|---|
| parti | 19.2% / −1.23 | 73.8% / +0.95 | 90.5% / +1.62 |
| ulti | 25.3% / −4.96 | 73.6% / +0.83 | 92.9% / +3.15 |
| betli | 1.4% / −4.86 | 36.7% / −1.33 | **56.7% / +0.67** |
| duri | 0.7% / −5.92 | **59.8% / +1.18** | — |
| ulti_piros | 19.7% / **−11.27** | 74.1% / +1.79 | 92.8% / +6.26 |

The bins genuinely track makeability, so the composite is a fair baseline and
CFR's edge is not from a miscalibrated net.

## Belief-updating (the headline mechanism)

Holding P1's hand-bucket fixed and varying P0's open:

```
P1 [medium piros-ulti hand]:
   P0 opened parti → P1 pass 0.96      (defend the weak opener)
   P0 opened ulti  → P1 ulti_piros 1.00 (overtake the strong opener)

P1 [betli-md, ulti_piros-md hand]:
   P0 opened parti → P1 ulti_piros 0.96 (take over)
   P0 opened ulti  → P1 pass 0.82       (can't beat it profitably → defend)
```

The overtake/defend choice is a function of the opponent's bid — reach-weighted
over the *posterior* opponent hands, never a uniform talon prior.

## Tournament (N=2000, seeds 100000+, def=god, sol=PIMC32)

Paired "lone hero vs 2 composites", seat-rotated (composite-hero ≡ 0 by
zero-sum self-play; CFR-hero is its edge):

| | composite-hero | CFR-hero | Δ |
|---|---:|---:|---:|
| overall GP/seat-deal | +0.000 | **+0.074** | **+0.074** |
| seat 0 (forced opener) | −1.006 | −0.833 | **+0.173** |
| seat 1 | +0.377 | +0.310 | −0.067 |
| seat 2 | +0.629 | +0.746 | +0.117 |

Hero-seat outcome mix (freq%, mean GP):

| contract | composite | CFR |
|---|---|---|
| defend/passout | 66.7% / −0.71 | 62.3% / −0.78 |
| parti | 19.3% / −2.09 | 18.3% / −2.17 |
| ulti | 10.2% / +5.17 | 13.5% / +3.89 |
| ulti_piros | 3.8% / +9.11 | 5.8% / +7.36 |
| betli/duri | 0 | ~0 |

CFR is **more aggressive** (more solos, lower mean quality each), net +EV.

## Robustness: composite make-prob prior `Q`

The clean composite's overtake conservatism depends on `Q = P(holder makes |
bid)`. As the baseline is made fairer (higher `Q` = it overtakes more), CFR's
*overall* edge erodes but the *opener* edge is invariant:

| `Q` | overall Δ | t | seat 0 | seat 1 | seat 2 |
|---|---:|---:|---:|---:|---:|
| 0.50 | +0.086 | 1.7 | **+0.181** | −0.039 | +0.117 |
| 0.70 | +0.051 | 1.1 | **+0.171** | −0.077 | +0.058 |
| 0.85 | +0.026 | 0.6 | **+0.175** | −0.063 | −0.033 |

`Q` only touches overtaking, so the stable +0.17 opener edge is CFR's robust,
baseline-independent win; the overtaking advantage is real but only beats a
*passive* composite (a well-tuned one overtakes about as well).

## Faithful baseline: CFR vs the DEPLOYED bidder (the real comparison)

`OracleComposite` replicates the deployed logic — **talon-averaged argmax-over-
66-discards** evaluation (the optimizer's-curse inflation source), `Q`=0.85,
−2 open floor. N=3000:

| | oracle-hero | CFR-hero | Δ | t |
|---|---:|---:|---:|---:|
| **overall** | +0.000 | **+0.581** | **+0.581 ± 0.124** | **9.2** |
| seat 0 (opener) | 0 | +0.281 | +0.281 | 2.7 |
| seat 1 | 0 | +0.761 | +0.761 | 6.7 |
| seat 2 | 0 | +0.701 | +0.701 | 6.2 |

Mechanism — the inflation makes the deployed bidder commit low-quality versions
of every contract; CFR solos less but far stronger:

| contract | oracle | CFR |
|---|---|---|
| defend | 66.7% | 74.8% |
| ulti | 15.1% / +0.59 | 9.4% / +3.62 |
| ulti_piros | 9.8% / +1.08 | 6.1% / +7.48 |
| betli | 0.1% / −6.0 | ~0 |
| duri | 0.1% / −7.6 | ~0 |

## The decomposition (the real lesson) — now MEASURED directly

Same metric (lone hero vs an **oracle field**, Q=0.85, N=3000), two heroes:

| hero vs oracle-field | overall Δ | t | seat 0 | seat 1 | seat 2 |
|---|---:|---:|---:|---:|---:|
| **CFR** | +0.581 ± 0.124 | 9.2 | +0.281 | +0.761 | +0.701 |
| **clean greedy** | +0.574 ± 0.138 | 8.1 | +0.182 | +0.792 | +0.747 |

**CFR and the clean greedy bidder beat the deployed bidder by the *same* amount**
(+0.581 vs +0.574 — difference +0.007 is noise). So:

- The deployed bidder's **argmax-over-66 inflation costs +0.57 GP/seat-deal**
  (t=8.1) — the dominant lever, and it's recoverable by a **trivial decision-rule
  change** (evaluate on the raw hand instead of max-over-66 discards).
- **CFR adds ≈0 over the clean greedy rule at the table level.** Its only
  unique edge is a modest opener gain (seat 0: +0.281 vs clean's +0.182, so
  ~+0.10), offset by being slightly *worse* at overtaking (seats 1–2). Net wash.

**Bottom line: the big money in bidding is killing the optimizer's-curse
inflation, not the equilibrium reasoning.** CFR is the principled gold standard
that proves the ceiling and confirms belief-updating is learnable, but for
production GP a one-line decision-rule fix captures essentially all of it.

Caveat on scope: this is all measured *vs the deployed/oracle field*. CFR's
distinctive property is unexploitability vs an *adaptive adversary*, which this
fixed-opponent metric does not test.

## CANONICAL BIDDER — vnet greedy + debias (now the default)

Decision (2026-06-14): the canonical bidding model is the **value net
(`CompositePickup`, exp19) + greedy argmax-EV + the debias patch** — not CFR.
CFR ties it on GP vs fixed opponents and is far more complex; it stays as a
benchmark / the answer for adaptive adversaries (unexploitability).

Patched `experiments/17_clean_pickup_net/auction_h2h.py`, **ON BY DEFAULT**
(`DEBIAS_BID=1 DEBIAS_PCTL=0.80`; set `DEBIAS_BID=0` to reproduce the old
inflated exp15–19 bidder): a contract's decision-p is the `DEBIAS_PCTL` quantile
of the 66 discard scores instead of the `max` (the optimizer's-curse source);
the discard actually PLAYED is still the argmax. A per-picker `debias_pctl`
attribute lets patched and deployed seats share a table (head-to-head).

Note: in the real harness P0 holds the talon (12 cards), unlike exp20's
symmetric game, so the right transfer is the **quantile** (uses all 12 cards),
not raw-10 (which ignores the talon and over-conserves — drops good ultis).

god-check (N=1000) — debias sharply raises commit quality and kills the bleed:

| config | ulti commits / god-win | betli commits |
|---|---|---|
| deployed (max) | 671 / 68–81% | 45 (4.5%) |
| PCTL 0.75 | 499 / 85–91% | 5 (0.5%) |
| PCTL 0.90 | 596 / 77–87% | 17 (1.7%) |

Real-harness head-to-head (patched vs deployed, N=2000, PCTL=0.80):

| | Δ GP/seat-deal | t |
|---|---:|---:|
| **overall** | **+0.380 ± 0.144** | **5.2** |
| seat 0 (opener) | +0.854 | 5.3 |
| seat 2 | +0.248 | 2.3 |

Winning-bid shift (deployed → patched): betli 84→22, durchmars 11→0, ulti/hearts
622→424 (drops inflated marginals), parti 531→642. **+0.38 is the
captured-in-production figure (~⅔ of exp20's +0.57 ceiling)** — less because of
the asymmetric deal and the conservative percentile; PCTL is tunable for more.

## Residual betli bleed — diagnosed, but NO clean legal fix (accept it)

The canon still commits ~0.8% betli at ~8% won (−4.18/def, below the −2 pass
floor). Long investigation; the honest end state:

- **Not the net.** The betli net is *underconfident* on random held-out hands
  (pred≥0.3 → 58% true; `betli_calib_diag.py`). Recalibrating it would bid MORE
  betli — backwards. The P0 betli decision is well-calibrated too (dec_p 0.3 →
  72%; `calib_colorless.py`).
- **It's a forced realization.** The overtaker commits to taking over on an
  optimistic 231-talon-averaged estimate (`_oracle_evaluate`), then — required
  to bid *something* above the current rank on its actual hand — is occasionally
  stuck realizing a hopeless betli as the least-bad option. The betli is a
  *symptom of an over-eager overtake*, not a betli decision (so colorless
  floors/shrinks miss it entirely — verified, no effect).
- **An overtaker-realization floor (`ev_floor=pass_ev`) works but CHEATS.** It
  lets the overtaker check the bid against the REAL talon after a hypothetical
  pickup and *retreat to a pass* (betli 0.90%→0.08%, survivors 100% makeable).
  Illegal in Ulti — once you take the talon you must declare. **Tried, rejected,
  removed from the code.**
- **No legal lever removes it cleanly.** An overtake margin (require the overtake
  to beat defending by a margin) reduces betli (33→17→11 at margin 0/1/2) but
  only by killing good overtakes: margin 1 sheds 16 junk betlis at the cost of
  ~150 genuine ultis (87% makeable, ~+3 GP) falling back to losing parti (~−1.3
  GP) — a clearly bad trade. Colorless floor/shrink: no effect. All removed.

**Verdict: accept the residual.** It's the honest price of committing blind to
an overtake; ~0.8% of deals, a few hundredths of a GP — a rounding error vs the
+0.38 the inflation debias bought. The bidding code carries **only** the
`DEBIAS_BID` knob; all betli-investigation flags were stripped. The only real
removal would be a smarter overtake model that predicts realization quality (a
project, not a knob).
NB: the overtake decision also already cheats via `pass_ev` (PIMC on the
holder's real hand) — pre-existing, not introduced here.

## What's worth keeping / next

- Clean, fair auction-CFR infra (god leaves are cheap: ~3 min / 200k deals).
- Coverage gaps on rare contested infosets show as uniform/noisy strategy
  (the `0.25/0.25/...` rows); more iters or finer history would sharpen them.
- Next: (1) faithful "oracle composite" baseline (talon-averaged argmax-over-66,
  with the inflation) to measure the betli-fix component too; (2) richer
  buckets / talon-aware evaluation; (3) deeper auction with a non-forced opener.

## Files

```
experiments/20_cfr_bidding/
  PLAN.md  results.md
  common.py game.py buckets.py        game + featurization primitives
  leaves.py        god-exact leaf precompute → leaves_200000.npz
  cfr.py           external-sampling MCCFR → strategy.pkl
  bid_agents.py    CompositeAgent + CFRAgent
  tournament.py    CFR vs composite, seat-rotated, fixed playout
  inspect_strategy.py  belief-updating evidence
  bucket_makerate.py   bin→god-makeability calibration
  benchmark.py test_game.py
```
