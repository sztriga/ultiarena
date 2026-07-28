# exp 28 — pickup/discard net headroom (started 2026-07-21)

## Goal (milan)
The pickup net is suspected undertrained; can refining it beat the current frontier?
The talon put-down (discard) is chosen by the NET (argmax over 4 trumps × 66 discards,
scoring each 10-hand with the 7 base-event heads) — NOT PIMC. One net, evaluates the
10-card kept hand; never sees the 2 buried cards. Confirmed to milan.

## Research findings (agent, read-only audit — see below; drives the plan)
- Nets are NOT undertrained by DATA VOLUME: 7 heads, 1M god labels each, val AUC 0.95–0.99.
  Weakness is CALIBRATION of the RARE heads: colorless_duri (~400 pos/1M, 0.04%), betli
  (~2800, 0.28%), reach100_20 (0.87% at α=0, trained on inflated α=3, pos_weight 13.6) —
  OVERCONFIDENT in the exact high-conf bins the argmax selects (betli conf .93→acc .60).
  Propped up by isotonic calibration fit on tiny positive samples. parti/ulti/betli val-Brier
  PEAK at epoch 4–9 of 25 (class-weighted loss overfits majority; best-Brier snapshot saved).
- The DISCARD choice was NEVER oracle-audited. The debias (DEBIAS_PCTL=0.80) fixes the
  contract-CONFIDENCE (percentile over 66) but the PLAYED discard is still the raw net argmax,
  unverified. exp20 measured the argmax-over-66 INFLATION at +0.57 GP/seat-deal (t=8.1); shipped
  debias recovers +0.38. exp19 diagnosed the residual as "optimizer's curse in the max, not the
  model → fix is bid-it-less, not a better net."
- CEILING (exp25): full bidding-net RETRAIN capped ~+1.3–1.5 GP/deal (82% of the god edge is
  IRREDUCIBLE info gap; N≥500 firm). Improving pickup CALIBRATION alone is a demonstrated WASH
  (exp18 canonical: better Brier, agent Δ −0.05). Real lever = in-play opponent modeling.
- Feature repr = 32-dim hand multi-hot (+4 trump). Canonical (exp18) & structural (exp19)
  features tried, NOT adopted (strength wash). No suit-perm augmentation in the deployed heads.

## Plan (measure headroom BEFORE investing in a retrain)
1. DISCARD ceiling: for opener 12-hands, take the net's chosen (contract, trump), then god-solve
   ALL 66 discards for that contract → does the net's discard preserve makeability when a better
   discard exists? Per-contract (headroom concentrates in ulti/100/duri, not parti).
2. CONTRACT/trump ceiling: does the net bid the god-best contract? (separate lever from discard.)
3. If the discard leaks: prototype a cheat-clean PIMC-scored discard (sample opponents, roll out
   each of 66, pick best) on top of the net's contract — measure the ACHIEVABLE gain.
4. Tournament-validate any candidate vs the current frontier bidder (self-play auction + PIMC
   play + oracle, N≥500, held-out) — same gate as exp27. Nothing promoted without a win.

## Guardrails
Sandbox only. Never touch the deployed engine/checkpoints. Cheat-clean for any deployable
candidate. N≥500 for headlines. Frequent flushed logging. Honest expectation: retrain ceiling
~+1.3; the discard-oracle lever is the un-tested one and the most promising thing to measure.

## RESULTS — discard ceiling (harness28 build, N=1500 opener bids, god perfect-info)
Per contract: net-discard make% / god-best-discard make% / discard-regret (net LOSES but a
better discard WINS, contract held fixed):
  parti   n=1281  20% / 24% / 4%    (binary point battle; discard barely matters realistically)
  ulti    n= 158  91% / 97% / 6%    (the one trick contract where the discard genuinely matters)
  40_100  n=  26  100%/100%/ 0%     (already optimal — 44/66 discards win)
  ulti+40_100 n=17 100%/100%/0% ;  20_100/ulti+20_100 100%/100%/0
  durchmars n=9 44/56/11% ; betli n=4 50/75/25%  (thin)
OVERALL: net make 30% vs god-best 35% → **discard-fixable losses 4.3%** of opener bids.
This is the GOD CEILING (knowing opponents' cards). Concentrated in parti(4pp)+ulti(6pp);
100-games already optimal. Rough GP: ~+0.3 GP/deal ceiling, mostly unrealizable cheat-clean.

## RUNNING — achievable (cheat-clean PIMC discard vs net argmax) on ulti/duri hands
harness28 `pimc` (K=6 sampled worlds, N=2600 seeds → ~290 ulti/duri hands). Compares god-make
of the net's discard vs a cheat-clean PIMC-chosen discard vs god-best, on the actual deal.
Answers: is the ulti discard headroom capturable WITHOUT seeing opponents' cards? (Expect small.)

## RESULTS — achievable (cheat-clean PIMC discard, K=6, N=335 ulti/duri)
ulti: net 90% / PIMC-discard 82% / god-best 96%. **PIMC discard is −6.9pp WORSE than the net**
(optimizer's curse on 6 samples; the net's 1M-hand prior generalises better). Higher K might help
but is deployment-infeasible (bid discard uses ZERO rollouts today). → **discard is a STRENGTH,
not a lever.** 40-100/20-100 discards already 100% optimal.

## VERDICT (SUMMARY.md) — LOOP COMPLETE
Refining the pickup net will NOT beat the frontier meaningfully. Discard is near-optimal (naive
PIMC replacement is worse). Only pickup lever = contract-choice retrain on realistic labels,
ceiling ~+1.3 GP/deal (exp25), calibration a wash (exp18). Real frontier = in-play opponent
modeling. Nothing changed in the deployed engine. No background jobs running.

## Progress
- 2026-07-21: DONE. research + discard ceiling + achievable PIMC-discard. Verdict written.
  If milan wants the +1.3 retrain, that's a separate bounded project (datagen+train+tournament).
