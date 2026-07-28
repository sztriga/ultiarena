# exp 27 — full-ladder kontra/rekontra revamp (started 2026-07-21)

## Goal (milan)
A COMPLETE revamp of the kontra logic so the AI kontras/rekontras with JUSTIFIED
CONFIDENCE across the WHOLE ladder — every contract (incl. combined/100/terített that
today have NO kontra), both roles (defender kontra + soloist rekontra). Scope =
**full ladder**. Deliverable = **validated proposal only** (report + proposed patch;
do NOT touch apps/api/play.py). Builds on exp26 [[project_exp26_defender_kontra]].

## Core method (generalizes exp26's ulti finding)
The deployed kontra uses own-hand GOD-makeability, which measures "beatable by PERFECT
defense" — systematically wrong vs real (PIMC) defense (ulti: god ~30%, real 83%). Fix:
per-UNIT, per-ROLE **calibration** of the make-signal to the REAL make rate, then an
EV-optimal per-unit kontra/rekontra decision by backward induction on calibrated beliefs.

## Foundation (read from ulti.scoring/oracle.py — LOCKED)
- **6 kontra units**: parti, ulti, betli, durchmars, 40_100, 20_100 (`_unit_of`, oracle:33).
  Silent riders (silent_40_100/20_100/durchmars + def_silent_*) ride the PARTI unit;
  silent_ulti rides NO unit. A combined bid = a SET of these units, each kontra'd indep.
- **Per-unit made**: `PayoffVector.components[unit]` = per-unit soloist GP; `.made(unit)` =
  components[unit] > 0 (kontra scales magnitude, never flips sign → made is well-defined).
- **Scoring**: `score(final_pos=, bid=, kontras={unit: level | (d1,d2)})`. Colored units
  SHARED (one level); colorless (betli / no-trump duri) SEPARATE per defender (d1,d2)
  via `def_split`. GPTable rates in oracle.py; bukott-ulti special (2/3/5×) — confirm.
- **Per-unit god objectives** (from gen_base_events.py, reuse for signals):
  parti→solver "parti"; ulti→"ulti"; betli→"betli"; durchmars→"durchmars";
  40_100/20_100→solver "multi" weights {"score_geq_100":1.0} restrict "40"/"20".
- **α-biased dealers** for coverage: `eval.dojo.deal_ulti_biased(seed,alpha)` (fat trump),
  `deal_durchmars_colored(seed,alpha)`. Need coverage of ALL units incl. rare betli/duri/combos.

## Plan / phases
- P0 rules foundation: units_of(bid), unit_made(final_pos,bid,unit), per-unit god
  makeability, per-unit GP under kontra vector. Cross-check colored/colorless + silent.
- P1 eval set: α-biased + champion-auction deals across ALL contracts; play once; cache
  per-unit made + component GP + hands/features. (heavy, background, resumable)
- P2 signals+calibration: per (unit,role) god-makeability + structural features →
  isotonic/binned calibration to REAL per-unit make rate (train split).
- P3 decision+validation: per-unit backward-induction kontra/rekontra on calibrated
  beliefs; out-of-sample per unit AND per contract vs deployed + oracle ceiling.
- P4 deliverable: per-contract confidence report (SUMMARY.md) + proposed patch to
  _kontra_primary/_ai_defender_kontras/_ai_soloist_rekontras/_kontra_dict.

## SECOND deliverable — teaching material (milan 2026-07-21)
Same data, two audiences. Alongside the AI proposal, produce **teaching tables** for
people learning Ulti: per-contract P(soloist makes) under observable conditions
(defender trump count, high-card/marriage holdings, etc.), as memorable rules-of-thumb.
DISCIPLINE — always report BOTH regimes, clearly labeled, never conflated:
  * GOD / perfect-play prob = theoretical (aspirational tip: "a 4-trump defender beats
    the ulti X% with perfect play").
  * PIMC / realistic prob = what happens vs strong-but-imperfect play (what to expect).
NOTE: exp26's "4 trumps → 37% make" was REALISTIC (PIMC), not god — under perfect defense
it's even lower. Label rigorously; capture sample sizes. The GOD-vs-REAL GAP is itself a
headline teaching insight (why over-eager kontra loses) AND the AI-bug root cause.
Capture rich structural features per contract so tips are concrete. → TEACHING.md / artifact.

## Guardrails
Sandbox only. NEVER modify apps/api/play.py, champion config, or checkpoints. Cheat-clean
(own-hand + public only). N≥500 per unit for headline calibration/validation. Frequent
flushed logging. Nothing integrated — milan reviews.

## Harness design (harness27.py — VALIDATED)
- Units scale independently → play each deal ONCE, cache per live unit U:
  `made_U` (pvec.made) + `iso_U[0,1,2]` = U's ISOLATED soloist per-def GP at kontra
  level 0/1/2 (captures piros + bukott-ulti 2/3/5× exactly; verified iso[-16,-24,-40]).
  Total GP additive: colored per_def = Σ_U iso_U[lvl_U]; colorless = iso[L_d0]+iso[L_d1].
- Live units derived from scored components via `_unit_of`. Rich per-hand FEATURES
  (trump count, high trumps, trump-ace, cardpts, aces/tens, voids) for calib + teaching.
- Commands: `build` (champion auction, uniform), `pools` (per-unit GOD makeability from
  viewers 0/1/2, per-unit objective from gen_base_events), later calibrate/decide/teach.

## Progress log
- 2026-07-21: reframed exp26→full-ladder revamp; teaching-material track added. Foundation
  locked from oracle.py (agent-confirmed: 6 units, made=sign, {unit:lvl|(d1,d2)}, colored
  shared/colorless separate, bukott-ulti 2/3/5×, combined games currently get NO kontra).
- harness27 build + pools BUILT + validated (per-unit iso GP exact; per-unit god makeability
  works). Champion build N=8000 RUNNING (~45m, uniform deals → realistic distribution).
- build N=8000 DONE (coverage: parti 7238, ulti 1851, 40_100 342, durchmars 211, betli 183,
  20_100 99). pools + godactual DONE. analyze + teaching RUN.

## KEY FINDING — corrects exp26 mechanism (2026-07-21)
Four numbers per unit (n, DEFENDER-BLIND god / TRUE perfect-play god_actual / realistic PIMC):
  * ulti      : blind 6-11%  | perfect 80% | realistic 83%
  * parti     : blind 6-12%  | perfect 29% | realistic 36%
  * durchmars : blind 0%     | perfect 36% | realistic 39%
  * betli     : blind 0-1%   | perfect 66% | realistic 86%   ← ONLY real god-vs-real gap (20pp)
TRUE perfect-play ≈ realistic for ulti/parti/durchmars → these contracts are genuinely
as strong as they play. The AI over-kontras because its DEFENDER-BLIND makeability estimate
(samples random soloist hands, ignores that the soloist BID the contract) is ~6-11% vs the
true ~80%. NOT a god-vs-PIMC gap (exp26 mechanism was wrong; the trump-gate fix still stands).
Betli is the exception: perfect defense (66% make) genuinely beats it more than typical
play (86%) — betli rewards defensive SKILL. Correctly-labeled teaching gem.

## AI decision results (analyze, OOS, results_units.md) — soloist per-def GP, gain vs deployed
  * ulti      : trumps>=4 gate → **+14.5** (deployed +17.6 = disaster; kontra only w/ 4 trumps)
  * durchmars : trumps>=3 gate → **+5.4** (durchmars very beatable w/ 3+ trumps: make 0-5%)
  * parti     : god<0.02      → **+0.9** (tighten the makeability threshold)
  * 40_100 / 20_100 / betli : NO cheat-clean signal → best is DON'T kontra (abstain). The
    combined/100 kontra extension correctly concludes "don't" (naive kontra would be −EV).

## Teaching tables (TEACHING.md) — trump-count is the key actionable feature, BOTH regimes:
  ulti by def max-trumps: 1→99/100%, 2→89/88%, 3→76/72%, 4→32/27% (realistic/perfect)
  durchmars: 0→50/44%, 3→2/5%, 4→0/0%   |  20_100: 2→89/96%, 4→23/15%

## WIN-PROB vs STRUCTURE — is the estimated win-prob useful beyond trump count? (milan Q)
OOS test-AUC for predicting bukott (structural feats / +god win-prob / god alone):
  * ulti      : 0.766 / 0.769 / 0.657   → win-prob adds ~0 beyond trumps; god ALONE worse than trumps
  * durchmars : 0.662 / 0.635 / 0.500   → win-prob USELESS (0.500). Trump count is the whole signal.
  * 40_100    : 0.677 / 0.684 / 0.569   → win-prob negligible
  * parti     : 0.804 / 0.869 / 0.771   → win-prob adds REAL +0.065; strongest single component
  * 20_100    : 0.853 / 0.853 / 0.500   → useless (thin n=99)
PRINCIPLE: win-probs help where the outcome is a distributed multi-card computation no simple
feature captures (parti's ≥50-pt battle); dead weight where ONE structural fact dominates
(trump count → win/steal a trick). milan's duri intuition confirmed exactly.
DESIGN: per-unit — use god-makeability ONLY for parti; pure structural rules (trump count,
+high-trump/voids) for ulti/duri/40_100. Simpler, faster (no god-solves at kontra time for
trick units), AND more robust (drops a noisy signal that hurt ulti: god-alone 0.657 < trumps 0.766).

## TOURNAMENT (TOURNAMENT.md, held-out N=4000) — candidate vs current frontier
- A self-play: all-deployed soloist +6.69 vs all-candidate soloist −1.04 → candidate
  DEFENDERS concede **+7.73 GP/deal LESS**.
- B head-to-head (per table): **candidate wins +7.74 GP/deal** (decisions differ 51%).
- C defender-only (rekontra fixed): candidate defenders **+7.68 GP/deal**.
- C2 rekontra isolation (candidate defenders fixed): deployed −0.99 / candidate −1.04 /
  never −1.19 soloist GP → **rekontra is a MINOR lever (±0.2)**. Once defenders stop
  over-kontra-ing, rekontra choice barely matters → LEAVE rekontra untouched (lower risk).
- By contract: piros ulti defender gain **+30.5** (deployed over-kontra + rekontra
  amplification disaster); piros parti +1.16. ~97% of the win is simple ulti+parti.
- CANDIDATE = change ONLY the defender kontra rule (per-unit gates): ulti own-trumps≥4,
  colored durchmars own-trumps≥3, parti blind-makeability≈0 (more selective than deployed),
  betli/40-100/20-100/colorless-duri ABSTAIN. Rekontra unchanged. All cheat-clean.

## PROMOTION — DONE (milan authorized, gated on tournament — PASSED)
Stage 1 IMPLEMENTED in apps/api/play.py `_ai_defender_kontras`: ulti trumps≥4, colored duri
trumps≥3, parti makeability<0.10 (_KONTRA_PARTI_MAKE), betli/40-100/20-100/colorless-duri
abstain. Constants added. Rekontra + everything else UNCHANGED. Live-validated (fire 77%→24%,
e2e green, 88 games no errors; --reload hot-loaded it). Stage 2 (combined-game per-unit kontra,
thin data, multi-unit state machine) = PROPOSAL in SUMMARY.md, not applied.

## DELIVERABLES (all written)
- SUMMARY.md — AI revamp: bug, per-unit policy, tournament, code change, Stage 2 proposal.
- TEACHING.md — student study guide: per-contract make tables (realistic vs perfect) by
  defender trump count + rules-of-thumb + betli defensive-skill section. (Could → web artifact.)
- TOURNAMENT.md — head-to-head numbers. results_units.md — per-unit decision analysis.
- Memory: project_exp27_kontra_revamp, reference_ulti_makeprobs, MEMORY.md index updated;
  exp26 mechanism corrected.

## LOOP COMPLETE. No background jobs running; no scheduled wakeups.
Open follow-ups for milan: (1) Stage 2 combined-game kontra if wanted; (2) turn TEACHING.md
into a shareable web artifact for students; (3) betli defense (only real skill gap) could be
its own study/AI focus.
