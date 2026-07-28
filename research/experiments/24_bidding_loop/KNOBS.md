# Improvement surface — what the loop can tweak

The agent pipeline: **deal → auction (bid) → play → kontra decisions → score → GP.**
The loop's job: change one or more knobs below, eval the candidate **head-to-head
vs the reigning champion** (PIMC realistic + god fast), keep it iff it wins. Fixed
seed bank → reproducible. Every knob lists: what · where · what it targets.

---

## 0. FIXED — the game itself (NEVER change; the goal is to play these well)
- Ladder ordering & values — `23/ladder.py` (milan's rules).
- Scoring incl. kontra/silent/terített — `scoring/oracle.py` (the rulebook).
- Auction timing, talon-passing, kontra timing.
→ These define "correct." A candidate that changes them is disqualified, not better.

---

## 1. RETRAIN — the learned models (biggest levers, most effort)
1. **Base-event value nets** (the 7 heads: parti, ulti, reach100_40/20, duri_colored,
   betli, colorless_duri). Tweak: architecture (width/depth, shared vs separate,
   transformer), features (36-dim hand+trump vs richer/structural), labels (god vs
   PIMC), dealer α, data volume. `23/train_base_head.py`, `vnet/pickup`.
   → *Makeability accuracy — feeds every downstream decision.*
2. **Calibration** — isotonic per head; recalibrate to the deployment/kontra
   distribution; temperature scaling. `23/calibrate.py`, `*_isotonic.npz`.
   → *The rare-head over-firing (20-100 / terített bleed).*
3. **Kontra-decision net** (the distillation, later): `(defender hand, contract) →
   P(soloist makes)`. → *Kontra decision speed + quality at scale.*
4. **Learned play net** (bigger bet) — replace PIMC play with a policy/value net.
   → *Play strength + speed.*

## 2. TUNE — closed-form logic (fast, no training → cheap loop iterations)
5. **Debias percentile** `DEBIAS_PCTL` — argmax-over-discards debias. `23/auction.py`.
   → *Over-bidding / winner's curse.*
6. **Per-head confidence floors** — don't declare a rare contract unless calibrated
   prob clears a bar. `23/bidder.py`. → *The bleeders, directly.*
7. **EV composition** — independence sum → correlation/joint-aware; risk attitude.
   `23/bidder.py::rung_ev`. → *Combo valuation.*
8. **Kontra-risk curve** — `P(kontra), E[level] | contract, p_sol`, and how it folds
   into the bid EV. `23/kontra.py`. → *Kontra-awareness of the bidder.*
9. **Overcall / pass economics** — defender-EV threshold, pass floor. `23/auction.py`.
   → *Who wins the auction, the P0 deficit.*
10. **Kontra/rekontra decision rule** — the backward-induction EV thresholds.
    `23/kontra.py`. → *When to double.*

## 3. INFERENCE — search knobs (speed ⇄ quality)
11. **PIMC N** (determinizations) — play + kontra-decision quality vs speed.
12. **Discard/trump search breadth** — argmax over 66×4; prune or widen.
13. **Combo play-objective weights** — the multi-solver weights. `24/scorers.py`.

## 4. REDESIGN — architecture swaps (biggest bets, most risk)
14. **Factorized base-events → a direct per-rung value net, or a policy net**
    (hand → bid) — drop the composer. → *Ceiling on bid quality.*
15. **Separate colored/colorless → unified** (or vice versa).
16. **Learned belief/opponent model** for the PIMC determinization sampling.

---

## Known problems → which knob
- 20-100 family / terített combos **bleed** → 6 (floors), 2 (calibration), 1 (better
  reach100_20 net).
- Full ladder **≈ ties** the 4-contract bidder (extra contracts don't pay yet) →
  6 + 7 + 8 (stop over-firing, price kontra) so the combos net positive.
- **P0 deficit** structural → 9 (pass economics) + kontra (already shifts it).
- **god-pessimism** (terített dominates, betli looks bad) → eval on PIMC (metric),
  not god alone.

## Loop protocol
`champion` = current best. `candidate` = champion + a knob change (or retrain).
Eval `h2h(candidate, champion)` (24/h2h.py) → keep candidate as champion iff edge
> 0 and significant. Log the knob + result each round.
