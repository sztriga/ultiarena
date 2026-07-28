# Overnight autonomous session — 2026-07-21 → 22 (milan asleep, 8h mandate)

Context: found + fixed a CRITICAL bidder bug ([[reference_bidder_generator_bug]]) — `_search`
exhausted the discards generator after the first trump, so the frontier only ever bid PIROS
(hearts) contracts, never makk/zöld/tök. All prior benchmarks ran on this crippled bidder.

## Goals (in priority order)
1. **Fixed-frontier table** (exp29 re-run with the fix) + buggy-vs-fixed comparison. [RUNNING]
2. **Post-fix config re-tune** (FLOOR × DEBIAS_PCTL) — these were tuned on the CRIPPLED bidder,
   so the optimum may have moved now that non-piros contracts compete. Highest-value new experiment.
3. **exp27 kontra re-confirm** with the fixed bidder — does the promoted per-unit kontra
   (ulti trumps≥4 etc.) still win now that non-piros ultis are in the mix? (suit-agnostic → should).
4. **Duri leak** — the one bleed exp29 found (over-bidding terített duri, miscalibrated duri heads).
   Test a per-contract FLOOR bump / threshold to reduce it.

## Method / metric
- Bidder strength = harness.evaluate (exp24) METRIC = mean soloist GP/game vs a FIXED strong
  (PIMC) defender — NOT zero-sum (defender is config-independent), so it's config-comparable.
  Also the NON-FLOOR GP (escalations above piros parti = the discriminating signal).
- FLOOR is a bidder.py import-time global → each config runs in its OWN subprocess (env FLOOR/
  DEBIAS_PCTL). pimc scorer (realistic play) — god scorer would wrongly favour over-bidding.
- N≥1000/config for the sweep; N≥500 for any headline. Nothing promoted without a head-to-head win.

## Guardrails
- Sandbox only (experiments/30_*, 29_*). The auction.py generator FIX is applied+validated (a clear
  bug). NO other deployed-engine changes without a tournament gate. Cheat-clean throughout.
- Frequent flushed logging; resumable; everything documented in SUMMARY.md so milan can review.

## Files
- 29_frontier_table/ : selfplay_fixed.jsonl (re-run), selfplay_buggy.jsonl (preserved), analyze.py
- 30_frontier_tuning/ : run_one.py (one config eval), sweep.py (driver), sweep_results.tsv, SUMMARY.md

## Log
- 2026-07-21: bug fixed. exp29 re-run launched (N=6000, fixed bidder). Config-sweep harness prepped.
- 2026-07-21: exp29 FIXED table DONE — dramatic transformation vs buggy:
  * passz 33%→10%; played 67%→90%; non-piros colored 0%→36% of deals; auction contested 4%→44%.
  * piros parti 32.5%→3.6% (opener now bids non-piros ultis not the weak floor); ulti (non-piros)
    0→27.5% (2nd most common!); piros ulti 26→35%.
  * P0 forehand tax GONE: −1.03 → +0.045 (now bids instead of paying pass penalty). Per-seat:
    P1 +0.60 (best), P0 +0.045, P2 −0.64 (rear now worst).
  * **DURI LEAK EXPLODED**: was 88 deals/1.5%/−0.11 GP/deal → now **599 deals/10%/−0.705 GP/deal,
    made only 25%**. The bug MASKED it. terített-duri contracts made 13–26% but bid constantly:
    teritett 40-100-duri 166 deals −11.35 (made 14%), teritett 20-100-duri 60 −11.87 (13%),
    teritett ulti-40-100-duri 72 −6.11 (14%), etc. → the #1 leak now, biggest single lever.
  * HYPOTHESIS: composer uses the CLOSED-hand p_duri for TERÍTETT (open-hand) duri — open hand is
    much harder (defenders see your cards) so real make is 13-26% while the net says high → over-bid.
- Config sweep DONE (FLOOR×DEBIAS, pimc, N=1200): **FLOOR=0.80 DEBIAS=0.85 BEST metric +3.478**
  vs current FLOOR=0.70/0.80 +2.635 (8th of 9) → **raising FLOOR to 0.80 is worth ~+0.84 GP/game**.
  All 3 FLOOR=0.80 configs top the ranking; all 3 FLOOR=0.70 bottom. (The fix opened non-piros
  contracts; a higher floor gates the overconfident escalations incl. some terített-duri.)
- Duri fix: env-gated DURI_TERIT_MULT added to bidder.py (default 1.0=OFF). duri_sweep LAUNCHED at
  the retuned config (FLOOR=0.80 DEBIAS=0.85), sweeping mult ∈ {1.0,0.7,0.5,0.3,0.15,0.0}.
- exp27 kontra RE-CONFIRMED on the fixed distribution (from exp29 data, no re-run needed):
  ulti kontra fires only 4% (trump-gate; old blind rule ~94%), 70% profitable when it fires;
  works suit-agnostically (piros ulti 4%, non-piros ulti 3%). parti kontra 81%, 55% profitable.
  The promoted per-unit kontra HOLDS with non-piros ultis in the mix.
- Exploit play (exp31) DEFERRED — the god-leaf replacement is expensive; needs careful tractable
  design (PLAN.md ready). Not rushing it; it's the next dedicated session's big bet.
- Duri sweep DONE: monotonic, mult=0.3 recommended (~+0.26 GP/game; terített-duri EV always neg at
  mult≤0.5; pass rate unchanged → redirects to a better contract).
- VALIDATION DONE (N=2500 same deals): current 0.70/0.80/1.0 = +2.954 → RECOMMENDED 0.80/0.85/0.3
  = +3.643 = **Δ +0.689 GP/game** (on top of the bug fix).
- CAPSTONE DONE (self-play N=6000 recommended cfg): duri-family −0.705 → **+0.257 GP/deal**,
  terített-duri bids 599→73 — the leak is GONE in the full auction.
- FINAL STATE: e2e green. Deployed changes = auction.py generator fix ONLY (clear bug). bidder.py
  DURI_TERIT_MULT hook added but DEFAULT-OFF (1.0). FLOOR/DEBIAS unchanged (0.70/0.80) — the
  recommended 0.80/0.85/0.3 is a PROPOSAL for milan, NOT auto-applied. No background jobs running.

## SESSION COMPLETE. Morning deliverables: SUMMARY.md (this dir), the 3 self-play tables
(29_frontier_table/selfplay_{buggy,fixdefault,}.jsonl), sweep/duri/validate results (*.tsv/*.log),
exp31/PLAN.md (exploit play, next big lever). Recommended config awaiting milan's sign-off.
