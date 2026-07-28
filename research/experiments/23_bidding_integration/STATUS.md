# Exp 23 — status (autonomous session 2026-06-17)

## TL;DR
Architecture decided + foundations built & verified. The **ladder controller is
done and tested** against your confirmed 22-rung table. The **god-label pipeline
is built and validated correct** (labels match the trusted oracle), and a **1M-per-
head datagen is running in the background** for the 3 new base events. A **baseline
trainer is ready** and fires when data lands. Net architecture + the auction
wiring's modeling choices are left for you to confirm (flagged below).

## Done & verified ✅
- **`ladder.py` + `test_ladder.py`** — generates the full 22-rung ladder from the
  combo basis {ulti, durchmars, points∈(–,40,20), piros} + betli, ordered by
  (value, Σsq, piros-first). Reproduces your table EXACTLY: both equivalent-pairs
  (ulti-duri≡40-100-duri @10, piros twins @20), rebetli double-rung, piros-before-
  nonpiros @8, colorless-duri no-piros-twin, plain-parti-not-biddable. `overcalls()`,
  `rung_for()`, pass-economics included. **ALL CHECKS PASS.**
- **`gen_base_events.py` + `validate_labels.py`** — god-label datagen for the 3
  events exp 17 didn't have. Labels **cross-checked against the oracle: 100% match**
  (reach100_40/20). duri_colored uses exp-17's trusted method; its only "mismatches"
  are a benign PV-replay artifact (durchmars early-terminates → truncated PV), proven.
- **`bidder.py` + `test_bidder.py`** — the COMPOSER (integration centerpiece):
  base-event probs → every rung's EV (per-component, mirroring the oracle's
  independent components) → bid off the ladder. Handles marriage gates, the
  hearts-for-piros gate, equivalent-pair selection ("pick either"), pass
  economics. ALL tests pass. This is the concrete proof of the "predict base
  events, compose — don't train per-rung" thesis.
- **Memory fix**: obs is `envs/obs.py` (223-dim), not `game/obs.py` (264). Bidding
  nets use the 36-dim pickup `featurize`, not the play obs.

## AUCTION-ONLY PROTOTYPE — built & eval'd ✅ (2026-06-18)
`provider.py` (7 nets → BaseProbs) · `auction.py` (full-ladder 3-player talon-pass,
trump+discard search, exp-20 debias) · `eval_auction.py` (god-checked GP) ·
`calibrate.py` (per-head isotonic). **The full 22-contract auction runs end-to-end.**

The eval surfaced — and fixed — two known issues:
1. **Calibration is essential.** Raw heads are over-confident → soundness 0.115.
   Per-head isotonic (ECE 0.08→0.0002) → 0.263, and kills most bad combo bids.
2. **Argmax-over-discards inflation** (your exp-20 winner's-curse). DEBIAS_PCTL
   sweep (calibrated, N=600, god defenders):

| DEBIAS_PCTL | piros parti | piros ulti (made) | combos | P0 / P1 / P2 GP |
|---|---|---|---|---|
| 1.0 (max) | 64% | 12% (0.34) | many, made≈0 | −1.28 / +1.09 / +0.19 |
| 0.90 | 79% | 14% (0.56) | ~gone | −1.14 / +0.80 / +0.35 |
| **0.80** (canon) | 84% | 14% (0.60) | gone | −1.34 / +0.98 / +0.37 |
| 0.70 | 90% | 8% (0.74) | gone | −1.37 / +0.87 / +0.50 |

**Read:** after calibration + debias the bidder is sane — mostly the **piros parti
floor**, escalating to **piros ulti** (made 0.60–0.74) and **40-100** (made ~1.0)
only on strong hands; the value-20/28 combo over-bidding is gone. The aggregate
"soundness" is anchored low by the floor — that's BY DESIGN (never-pass economics:
a weak piros parti still beats the −2 pass), not a defect. P0 −1.3 = the structural
never-pass opener tax you predicted kontra will fix.

**Honest limitations:** GP is component-wise god double-dummy — exact for simple
contracts (the bulk), an over-estimate for combos (joint play-out deferred). God
defenders are a hard bar (real/weaker defenders → higher soloist GP). P1>P2 is a
first-overcaller position effect.

## OPEN QUESTION surfaced by the composer (needs your ruling)
**Piros-parti floor vs trump choice.** I gate *colored* piros rungs (piros ulti /
40-100 / duri) on hearts being the soloist's trump — clearly right, you can't make
a hearts-ulti unless hearts is strong. But **piros parti is the auction floor**,
and with that gate a soloist whose best suit ISN'T hearts and who only has a weak
parti now PASSES. Your exp-20 note ("bid piros parti whenever you have a chance")
implicitly assumed hearts was available. Question: can a soloist open **piros
parti by *choosing* hearts** (committing to a hearts game) even off a non-hearts
hand, or do they pass? If the former, the hearts gate should NOT apply to piros
parti (only to the trick/100 piros rungs). Easy one-line change once you rule.

## Baseline net results ✅ (1M god labels/head, standalone MLP 36→128→64→32→1)
| head | base rate | val AUC | Brier | ECE | read |
|---|---|---|---|---|---|
| reach100_40 | 41.2% | **0.960** | 0.081 | 0.023 | well-discriminated, decent calibration |
| duri_colored | 10.1% | **0.988** | 0.044 | 0.061 | sweeps very predictable from structure |
| reach100_20 | 6.8% | **0.976** | 0.056 | 0.073 | great ranking; over-confident raw (class-wt) → needs isotonic |

**The base events are highly learnable from the hand alone** — this is the
empirical validation of the factorization thesis. Discrimination (AUC) is what the
"predict base events → compose" architecture needs, and it's excellent. The
calibration spread (reach100_20 over-confident from `pos_weight=13.6`) is exactly
what the flagged isotonic step fixes; AUC is unaffected. Weights saved as
`<head>_baseline.pt`. (reused exp-17's 1M parti/ulti/betli/colorless-durchmars
datasets for the other base events — not retrained here.)

## ~~Running~~ Done 🔄→✅
`run_gen_all.sh` (background, caffeinated, log `run_gen.log`) — 1M god labels each:
| head | drives | dealer α | keep | pos-rate | label |
|---|---|---|---|---|---|
| reach100_40 | 40-100 rungs | ulti 1.0 | 15% | 41% | ≥100 holding the 40 |
| duri_colored | colored-duri combos | duri 4.0 | 100% | 10% | sweep all 10 w/ trump |
| reach100_20 | 20-100 rungs | ulti 3.0 | 16% | 6.6% | ≥100 holding only a 20 |

(reuses exp 17's existing 1M parti/ulti/betli/colorless-durchmars datasets for the
other base events). ETA ~2h. When done I train baselines (`train_base_head.py`,
reports AUC/Brier/ECE per head).

## Design forks — YOUR call (sensible defaults chosen, all reversible)
1. **Dealer α / calibration.** Generated biased-for-balance (exp-15 precedent);
   deployment needs isotonic calibration to α=0 (exp-16 precedent). I did NOT build
   the calibration step — it depends on whether you want these folded into the
   exp-17 multihead or kept standalone.
2. **Net architecture.** I'm training STANDALONE baseline heads (one MLP each) to
   measure learnability without committing. The real choice — fold {parti, ulti,
   reach100_40, reach100_20, duri_colored} into one **colored multihead** (+ keep
   the exp-19 colorless net for betli/colorless-duri) — is yours.
3. **Composer / auction wiring.** The thesis: predict base events → compose every
   rung's EV via oracle rates × ladder arithmetic. The combination rule (independence
   approx vs joint), bukott handling, and bluff dynamics (rebetli) are modeling
   choices I left for you rather than bake in. `ladder.py` is the net-agnostic spine
   it plugs into.
4. **`silent_ulti` cull fix** — the one solver perf bottleneck (9ms vs 3ms). NOT
   done: it's shared-Cython, risky, and NOT on the critical path (rung decisions use
   the cheap dedicated solvers). Greenlight it and I'll do it against the existing
   value-equivalence test.

## Files
`PLAN.md` · `ladder.py` · `test_ladder.py` · `gen_base_events.py` ·
`validate_labels.py` · `recipe_local.py` · `train_base_head.py` · `run_gen_all.sh`
