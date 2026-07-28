# exp40 — Realistic (imperfect-defense) recalibration of the bidder heads

**Status: SHELVED (negative result). Documented for later revisit.**
Dates: 2026-07-25 → 2026-07-28. Owner: milan + Claude.

---

## TL;DR

We retrained the bidder's base-event make-probability heads on **realistic** labels (play the
contract out vs a PIMC opponent, "did it actually make") instead of the deployed **god / double-dummy**
labels ("can it be forced vs perfect defense"). The motivation was milan's intuition that a strong AI
should *risk* games like duri the way a human does, plus a real 40-100-duri negative-games leak.

**Three independent head-to-head tournaments all say the recalibration LOSES to the current god bidder:**

| run | heads swapped | eval opponent | GP/game vs FRONTIER | verdict |
|---|---|---|---|---|
| whole-slate | all 6 | PIMC-16 | **−1.62** (±0.30) | significant − |
| duri-only | duri_colored, colorless_duri | PIMC-16 | **−0.01** (±0.29) | tie |
| matched 3-head | ulti, duri_colored, colorless_duri | PIMC-32 | **−1.71** (±0.35) | significant − |

The 40-100-duri leak was **not** fixed (duri-only and matched made terített-40-100-duri *worse*).

**Root cause (important, not a bug):** a well-calibrated make-*probability* is not a good *bid signal*
for asymmetric-penalty games. The realistic odds boost **ulti's** apparent value the most (ulti had the
biggest god-vs-real gap), so the bidder starts **preferring ulti over its other options** — trading
away better contracts and safe passes for merely-OK ultis. It's a **relative-preference / displacement**
problem, so training+evaluating on the *same* opponent (milan's proposed fix) does **not** cure it — we
verified this directly at PIMC-32.

**The recalibration is doing what it was designed for** — it makes the AI braver, which is *correct vs a
human* (who misdefends marginal ultis) but *wrong vs a near-perfect AI*. Self-play is the wrong judge,
and we have no way to score the human benefit. → Keep the god bidder. Revisit only with a human/weak
opponent model.

---

## 1. Motivation

- Every deployed base-event head (`provider.py`) is trained + isotonic-calibrated on **double-dummy /
  god** labels at α=0: *"can the soloist FORCE this event vs optimal minimax defense?"* — see
  `experiments/17_clean_pickup_net/gen_alpha0.py:56`, `experiments/23_bidding_integration/gen_base_events.py:113`.
- That is systematically **pessimistic** for real bidding: a god defender is never poisoned / never
  slips. e.g. plain duri is ~unmakeable double-dummy (6.5%) but realistically makes ~24%.
- milan's framing: *"sometimes I play duri to risk it"* — a strong AI should be able to too. Also a real
  leak: the 40-100-duri family was −EV in the exp39 self-play table.
- exp37 already did this for **betli** (realistic head → the AI can bid dd-lost-but-winnable plain betli).
  exp40 = generalize exp37 to the rest of the slate.

## 2. Approach & architecture

For each head: deal a boundary-biased hand (`eval/dojo.py` biased dealers) → make ONE 2-card discard
(mirrors pickup) → **play the contract out move-by-move, PIMC soloist + PIMC defenders** → label =
*actually made* (oracle / `defenders_won`). Store the god label alongside (one perfect-info solve) to
measure the gap. Isotonic-calibrate on a held-out slice. The head is a drop-in for its god twin
(same `train_base_head.Head`, same featurize dim), swapped into the provider by overriding
`prov.heads[h]` / `prov.calib[h]`.

**Critical correctness point:** the label must play the contract the way PRODUCTION plays it. play.py /
`experiments/24_bidding_loop/scorers.py::pimc_outcome` route ulti / parti / colored-duri / 100-games
through the **`multi` solver + silent-game weights** (build as `parti`), NOT their dedicated solvers;
colorless-duri + betli keep dedicated solvers. So the label's defense == the live defense. (The god
comparison label still uses the dedicated per-head solver — a perfect-info solve, no world sampling.)

### Gotchas found & handled
- **The dedicated `ulti` PIMC solver crashes** (~12%: `determinize.sample_world` "pop from empty list").
  The production `multi` path both fixes it and is more faithful. Dedicated solvers → god label only.
- `deal_biased` (parti/ulti/duri_colored) **resamples** `alpha=U(0,alpha)` internally; `deal_betli` /
  `deal_durchmars_colorless` use `alpha` **directly**. The worker branches on `direct_alpha` so the
  make-rate sweeps the decision boundary cleanly.
- **durchmars is monotone** → early-stop the instant a defender wins a trick (same label, big speedup).
- **reach100_40** uses a local `ContractSpec` with `mandatory_trump_ranks=(king,upper)` to guarantee the
  trump 40; **reach100_20** filters `deal_parti` for a non-trump marriage (~61% skip — rare).
- Worker returns `"ERR"`/None on any exception → **skip-on-error** so one bad deal can't abort a
  multi-hour run. (0 errors observed across all runs.)

## 3. Files

| file | what |
|---|---|
| `datagen.py` | one script, `HEAD` env-selected; realistic + god labels for any head |
| `train.py` | train head + isotonic + reliability table + god-vs-real comparison report |
| `tournament.py` | GP gate: RECAL (`_RECAL_HEADS` swapped in via `_load_recal`) vs FRONTIER (god); self-play + h2h |
| `run_all.sh` | PIMC-8 full-slate datagen+train (the first pass) |
| `batch_one.sh <head> <N> [PIMC_N]` | recalibrate ONE head as a batch (used for the PIMC-32 retrain) |
| `run_tournament.sh` | the 3 gate matchups + report (PIMC_N env, default 32) |
| `run_pimc32_all.sh` | end-to-end PIMC-32: retrain ulti+duri_colored → clear → gate |
| `<head>_real_baseline.pt`, `<head>_real_isotonic.npz` | trained heads (the .pt is PIMC-32 for the 3 retrained; PIMC-8 for parti/reach) |
| `<head>_real.npz` / `<head>_real_p32.npz` | datagen sets (PIMC-8 / PIMC-32) |
| `TOURNAMENT.md` | the last (PIMC-32 matched) gate result |

## 4. Head results — the god-vs-real gap is real and head-specific

The god head systematically **under-predicts** the trick-denial games (duri, ulti) and is ~right on the
point games (parti, 100s). All realistic heads are well-calibrated (ECE_cal 2–9%).

### PIMC-8 datagen (first pass)
| head | realistic make | AUC / ECE_cal | god vs realistic outcome |
|---|---|---|---|
| colorless_duri | 0.224 | 0.92 / 0.021 | god 6.5% → **UNDER by 0.173** |
| duri_colored | 0.135 | 0.90 / 0.088 | **UNDER by 0.083** |
| ulti | 0.484 | 0.94 / 0.035 | **UNDER by 0.077** |
| reach100_40 | 0.428 | 0.91 / 0.041 | over by 0.018 (≈right) |
| reach100_20 | 0.091 | 0.88 / 0.049 | over by 0.017 (≈right) |
| parti | 0.645 | 0.86 / 0.070 | over by 0.025 (≈right) |

### PIMC-32 datagen (matched retrain of the 3 that bled)
| head | realistic make | AUC / ECE_cal | god vs realistic outcome |
|---|---|---|---|
| colorless_duri | 0.206 | 0.93 / 0.031 | **UNDER by 0.138** |
| ulti | 0.503 | 0.93 / 0.030 | **UNDER by 0.108** |
| duri_colored | 0.137 | 0.93 / 0.073 | **UNDER by 0.086** |

Note: the gap does **not** shrink much at the stronger opponent (ulti even grew 0.077→0.108, because a
PIMC-32 *soloist* also plays better). The heads are genuinely more accurate than god — that was never
the problem.

## 5. The three gate tournaments

All: RECAL (realistic heads) vs FRONTIER (god heads), **identical** engine (terített reveal, exp36 betli
net, PIMC soloist, kontra-aware), both with betli_real + rebetli ON. Only the swapped heads differ.

1. **PIMC-8 whole-slate (all 6): −1.62 GP/game (±0.30).** ulti flooded: piros ulti 35%→52%, GP
   +9.99→+1.35, passz 9.8%→0.2%. Duri looked fine only because ulti crowded it out (tiny n).
2. **PIMC-16 duri-only: −0.01 GP/game (±0.29) — tie.** With god ulti restored, the mix barely moved
   (duri family 6.5%→6.2%). But duri *itself* was over-bid: teritett duri +10.9→−3.7, teritett
   40-100-duri −4.8→−11.1, plain duri −6→−9.3. The leak did **not** close — it moved.
3. **PIMC-32 matched (ulti + both duri), train==eval: −1.71 GP/game (±0.35).** ulti flooded again:
   piros ulti 35%→49%, GP +10.4→+3.8, passz 9.8%→2.7%. teritett-40-100-duri −4.0→−12.9.

## 6. Root cause — why matched-opponent did NOT save it

milan's hypothesis: *"calibrate vs opponent X, evaluate vs X → a well-trained net should be positive."*
That holds for a **policy** net (predict the best action). Our net is **not** a policy — it's a
make-*probability* feeding a hand-built EV formula + a FLOOR=0.80 gate. Two things break the hypothesis:

1. **Each ulti RECAL bids is still slightly +EV (+3.8 GP/bid).** The −1.71 is **not** from bad ultis —
   it's from what ulti *displaces*. The realistic upgrade raises **ulti's** EV the most (biggest gap),
   while parti/40-100/betli keep lower (god or ~unchanged) EVs, so the bidder **over-selects ulti** over
   genuinely-better contracts and profitable passes. A relative-preference shift, not an accuracy error.
2. **Matched opponent can't fix a relative-preference problem.** Making the odds "truer on average" only
   makes the bidder bid ulti *more*. For a game whose value is dominated by its asymmetric downside
   (bukott doubles; kontra amplifies), "more accurate average make-odds" is the wrong lever — the god
   head's *pessimism* was accidentally the right caution.

**Confirmed empirically:** train==eval at PIMC-32 still lost −1.71. So the earlier "train≠test (PIMC-8
vs PIMC-16) optimism" story was only part of it; the deeper issue is displacement, which no opponent
match removes.

## 7. Why it might still be right — vs humans

Being braver about ulti/duri is **correct against a fallible (human) defender**: those marginal ultis
make more often when the opponent misdefends, so the trade pays. It is **wrong against a near-perfect
AI**, which is exactly what self-play tests. So the recalibration is behaving as designed; the judge is
the problem. We have **no human/weak-opponent model** to measure the intended benefit — that is the gap.

## 8. Conclusion & recommendation

- **Shelve the realistic recalibration.** Three runs agree: the god bidder wins fair AI-vs-AI play.
- **Keep** the god bidder + everything shipped this week: rebetli (exp39), betli-defense net (exp36),
  terített reveal (all deployed in `apps/api/play.py`, default-ON env flags).
- The **40-100-duri leak** is real but is best fixed by a **targeted valuation clamp** (dial down
  `DURI_TERIT_MULT`, or a terített-40-100-duri penalty in `bidder.py`) — a one-liner, independent of all
  this machinery. (Not yet applied.)

## 9. Open threads to resume (what "come back to this later" means)

1. **Calibrate/evaluate against a WEAK opponent model.** The realistic recalibration's whole premise is
   beating fallible opponents. Build an ε-god or a learned human-error defender, retrain the heads and
   run the gate against *it*. That is the only test that can show the intended win. (Risk: an ε is a
   guess; we'd want a validated human-play model.)
2. **Bid-signal, not make-prob.** For asymmetric-penalty games, consider training a head that predicts
   **bid EV / should-bid** directly (policy-style), or add an ulti-specific caution term, rather than a
   raw make-probability that the FLOOR gate over-trusts.
3. **The 40-100-duri leak** — one-line clamp (above), or the never-built **Phase 2 terített-duri reveal
   head** (play terit-duri out with the soloist hand revealed → true open-hand make-rate, replacing the
   `DURI_TERIT_MULT` fudge). Note: given §6, a realistic terit-duri head may *also* just over-bid; the
   clamp is the safer fix.
4. **milan's discard-poisoning idea** (bid a cheap duri to bait a re-raise that leaves a favorable talon
   for a terített-duri escalation) — a multi-step, opponent-modeling *discard/auction-search policy*.
   Realistic labels were meant to be step one; since they don't pay vs strong AI, this needs the
   weak-opponent model (#1) first, then the exp28-style discard-ceiling diagnostic on realistic labels.

## 10. Reproduction

```bash
cd experiments/40_realistic_recal
# one head at a chosen opponent strength (datagen + train, overwrites <head>_real_baseline.pt):
bash batch_one.sh ulti 20000 32
# the gate (edit _RECAL_HEADS in tournament.py to choose which heads to swap):
PIMC_N=32 bash run_tournament.sh      # → TOURNAMENT.md
```
Heads/artifacts are preserved on disk. `_RECAL_HEADS` in `tournament.py` selects which heads the RECAL
config swaps in; `_load_recal` overrides `prov.heads`/`prov.calib` from the `_real_baseline.pt` files.

## 11. Lessons

- **A well-calibrated probability ≠ a good decision signal** when a hand-built EV + threshold sits
  downstream, especially for asymmetric-payoff games. Gate on GP, never on AUC/ECE (the ulti head looked
  *perfect* on paper and lost games every time).
- **Self-play is the wrong judge for a "beat weak opponents" feature.** Pick the evaluation opponent to
  match the deployment opponent (here: humans), or you measure the opposite of what you want.
- **Small-N headlines mislead** (the whole-slate run "closed the leak" only because ulti starved duri of
  samples). Confirmed at N≥several-thousand with per-contract mean/SE.
