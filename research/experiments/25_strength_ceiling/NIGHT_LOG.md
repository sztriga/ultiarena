# Overnight run — 2026-07-05 → 07-06

GOAL: strongest REALISTIC agent (own-hand bidding + PIMC play + hand-based kontra)
that could beat humans. Camera-from-POV target ⇒ NO cheating: own hand + public
info only. Every candidate h2h vs the reigning champion; promote only on a real win.

Rules I hold myself to: (1) audit for info-leakage regularly; (2) realistic =
NEVER the god-provider in the agent (god is ceiling-only); (3) keep all logs;
(4) never drop a contract.

## Journal
- [start] Anti-cheating audit first (foundational). Then: fix bleeders → realistic
  champion → capture +1.5 → biggest lever → report.

## Bidder sweep (god metric, N=2000) — bleeder fix
- pctl0.80 FLOOR=0: METRIC -2.31, NON-FLOOR +9.7 — baseline, bleeders present
  (20-100 -19, rebetli -12, ter-20-100-duri -40).
- pctl0.80 FLOOR=0.5: -2.83 — WORSE; floor redirected bids into terített rebetli
  (133×, -10) at mediocre confidence. Floor too low for the over-confident betli head.
- pctl0.80 FLOOR=0.7: **-1.41 (best so far)** — bleeders gone, terített rebetli
  flips -10→+21 (only bid when p_betli>0.7). Clear win.
- Confidence FLOOR works; ~0.7 is the sweet spot at pctl 0.80. Confirm 0.90 half +
  the realistic (PIMC) h2h before promoting. Note: god metric has terített-dominance
  bias — realistic check matters.

## Sweep conclusion
FLOOR=0.7 wins: god METRIC -2.31->-1.41, bleeders gone, ter-rebetli -10->+21. pctl 0.80~=0.90. PROMOTE FLOOR=0.7 (pending realistic confirm).

## Play-side anti-cheating audit — CLEAN
traced pimc_decision->build_info_set->sample_world: info set = own hand + set of
unknown cards (deducible) + observed voids + public must-holds; hidden cards pooled
and RESHUFFLED (assignment never leaked). Hand sizes public. Realistic PIMC play is
cheat-free and maps to camera-POV (pool = deck minus observed).

## FLOOR=0.7 confirmed REALISTIC (PIMC, N=150)
FLOOR=0 METRIC +0.61 vs FLOOR=0.7 METRIC +3.23 (P0 -1.61). Bleeder fix holds under
realistic play. KEY: under imperfect (PIMC) defenders the agent is NET POSITIVE
(+3.23 GP/game as soloist) — it BEATS imperfect opponents. teritett rebetli +80
(made 1.00) — open-hand betli exploits imperfect defenders (the play-ceiling insight).
=> Champion #1 = net + FLOOR=0.7 + PIMC play. P0 still -1.6 (no kontra to pass yet).
Next: wire kontra into full agent (simple contracts dominate → big P0 win expected).

## Full-ladder kontra (SCORER=kontra, god play, N=3000, FLOOR=0.7) — CONFIRMED WIN
KONTRA=0: METRIC -2.43, P0 -2.91, pass 0, piros parti 71% @ -5.31 (kontra'd, bleeds).
KONTRA=1: METRIC -1.89, P0 -1.89 (+1.0!), pass 69%, piros parti 16% @ -0.57 (wr .22->.62).
Kontra-aware bidding extends the simple-game win to the full ladder: weak hands PASS
instead of getting kontra'd double. Champion = net + FLOOR=0.7 + KONTRA=1 + PIMC play.
Caveat: SCORER=kontra uses god play (combined-kontra-in-PIMC-play not wired) — the
kontra P0-win + the realistic PIMC +3.23 are measured separately; a unified
realistic+kontra number is the next build.

## PIMC_N play probe (FLOOR=0.7, N=150) — play quality is a BIG lever
PIMC_N 4→8→16→32 = METRIC +0.75 → +1.92 → +3.23 → +2.80. Sharp rise to ~16, then
plateau (32 dip is N=150 noise). REFINES the play-ceiling finding: realistic play
QUALITY helps a lot (bigger lever than bidding nets' +1.5), plateauing ~N=16. BUT
perfect-INFO god play HURTS (gives up on dd-lost contracts). So: invest in a stronger
REALISTIC play policy + opponent-modeling; never train play toward double-dummy.

## Unified realistic+kontra scorer BUILT (kontra_pimc_outcome)
The deferred "plumbing piece": PIMC play + HAND-BASED kontra decision (each side's
P(make) from its OWN root info set — cheat-clean, same construction as play PIMC),
scored with kontra via the oracle. Smoke (12 deals) sane: hopeless partis (p_def=0)
get kontra'd → big losses (KONTRA=1 bidder passes these); makeable contracts the
defenders wrongly kontra → soloist REKONTRA → +40 (asymmetric-info exploitation fires
as designed). Colored simple only (parti/ulti, ~90% of bids); betli/duri/combos fall
back to pimc_outcome. Running KONTRA=0 vs 1 under it = the definitive realistic all-rules champion.

## DEFINITIVE realistic+kontra champion (SCORER=kpimc, FLOOR=0.7, PIMC_N=16, N=200)
KONTRA=0: METRIC +6.28, P0 +0.40, pass 0%, piros parti 134× @ -3.97 (weak hands bleed,
  now kontra'd double under realistic play), piros ulti 38× @ +32.84.
KONTRA=1: METRIC +5.74, P0 **+3.34**, pass 65%, piros parti 30× @ **+6.27** (only strong
  hands bid it), piros ulti 26× @ +34.92.
=> Unified number CONFIRMS the two separate measurements: kontra-aware bidding lifts the
FORCED OPENER P0 +0.40 → +3.34 by passing the weak hands that otherwise get kontra'd
double (-3.97), while keeping the strong colored contracts. As soloist under realistic
(imperfect) defenders the agent is strongly net-POSITIVE; kontra multipliers amplify
made-contract winnings (made colored contract under rekontra pays big). Robust signal =
P0 + pass-rate (METRIC is inflated by those kontra multipliers, so I lead with P0).
CHAMPION #1 FINAL = net bidder + FLOOR=0.7 + KONTRA=1 + PIMC play. All 33 rungs, cheat-clean.

## N=600 perception-SPLIT CONFIRM — ratio EXACT, magnitudes tightened
N=600: total net->god +7.27, TRAINABLE net->PIMC +1.28 (18%), irreducible PIMC->god +5.99 (82%).
=> 82/18 ratio reproduces EXACTLY vs N=150; magnitudes came down slightly (N=150 was high:
total 8.53->7.27, trainable 1.50->1.28). Third lever-map pillar now firmed at N=600. The
trainable slice is ~+1.3 (even smaller than the +1.5 I'd cited) — bidding-net retraining is
firmly the MINOR lever. All 3 pillars now rest on N>=500. No qualitative change.

## N=500 PIMC_N curve CONFIRM — play lever is MODEST, not huge (N=150 was noise)
PIMC_N 4/8/16: METRIC +0.71/+1.29/+1.14; P0 (stabler) -2.54/-2.13/-1.58.
=> FIRMED: more PIMC search helps realistic play MONOTONICALLY on the opener (P0 +0.96
across 4→16) but plateaus ~16 — a MODEST lever (~+1 GP), COMPARABLE to bidding-net
accuracy (~+1.5), NOT the +2.5 the noisy N=150 ("+0.75→+3.23") suggested. Correction:
play-search depth and bidding nets are both modest, comparable knobs. The QUALITATIVE
finding stands (imperfect-INFO play beats perfect-INFO — exploits defender mistakes); the
real headroom is opponent-MODELING (FRONTIER.md), not cranking either existing knob.

## N=600 CONFIRMATION — firms the headline (N=200 was optimistic noise)
KONTRA=0: METRIC +4.57, P0 **-1.47**, pass 0%, piros parti 420× @ -4.65, piros ulti 116× @ +33.2.
KONTRA=1: METRIC +2.53, P0 **+0.91**, pass 69%, piros parti 89× @ +3.82, piros ulti 68× @ +30.5.
=> FIRMED headline: kontra-aware bidding lifts the FORCED OPENER P0 **-1.47 → +0.91**
(crosses from losing to winning), by passing 69% of weak hands that bleed as kontra'd piros
parti (-4.65); the piros parti it DOES bid flips -4.65 → +3.82 (only strong hands). METRIC as
soloist stays strongly positive (+2.5..+4.6) — beats imperfect defenders when holding a
contract (piros ulti +30). NOTE: the earlier N=200 "+0.40→+3.34" was noise; use ±2.4 swing at
N=600. Champion holds; number corrected in SUMMARY.

## Anti-cheating audit #3 — kontra decision (code trace) — CLEAN
Traced _hand_makeability: handed the full deal (eval harness) but the true off-viewer
hands feed ONLY build_info_set(root, viewer); sampling draws from the pooled/reshuffled
info set and god_says_soloist_wins runs on the SAMPLED world, never the true position ⇒
p_def/p_sol depend only on the viewer's own hand + public info. Bonus check: talon_known
is None for a defender ⇒ talon (soloist's discards) hidden to defenders, known to soloist
(clone_with_hands) — exactly the real-game info split. Kontra decision is camera-POV safe.
Launched fresh N=600 confirmation of the champion (run_kpimc_confirm.log) to firm the headline.

## Frontier design note written (FRONTIER.md)
Turned "go at opponent-modeling" into a concrete cheat-clean build order: weighted
determinization (belief update from observed plays) → defender-mistake exploitation
(realistic imperfect defender at PIMC leaves, not double-dummy) → belief-conditioned
kontra. Each step measurable via kpimc METRIC, each falls back to today's uniform path,
guardrail = re-run audit_cheating after each. This is the primary track; bidding-net +1.5 secondary.
