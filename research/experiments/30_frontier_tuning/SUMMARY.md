# Overnight session summary — 2026-07-21 → 22 (frontier improvements)

## Headline: a real deployed bug fixed + the frontier re-tuned on top of it

### 1. THE BUG (the big one) — the bidder was blind to non-piros contracts
`auction.py::_search` iterated a one-shot `itertools.combinations` discards generator *inside*
the `for trump in TRUMPS` loop, so it was exhausted after the first trump (hearts). **Only piros
(hearts) contracts were ever scored** — the bidder could NEVER bid a makk/zöld/tök (acorns/leaves/
bells) parti/ulti/etc. This crippled the DEPLOYED play-tab AI and every prior benchmark.
**Fix:** `discards = list(discards)` (one line). Verified live end-to-end (non-piros ultis now bid,
play, and score correctly; e2e green). See [[reference_bidder_generator_bug]].

### 2. What the fix did (fixed self-play table, N=6000, exp29)
| | buggy (piros-only) | FIXED (all suits) |
|---|---|---|
| passz | 33% | **10%** |
| played | 67% | **90%** |
| non-piros colored | 0% | **36% of deals** |
| auction contested | 4% | **44%** |
| P0 forehand GP/deal | −1.03 | **+0.045** (tax gone) |
| duri-family bleed | 1.5% of deals, −0.11 | **10% of deals, −0.705** (exposed) |

The opener now bids makk/zöld/tök ultis instead of folding a third of hands; the forehand tax
vanished. But the fix EXPOSED a big terített-duri over-bidding leak the bug had masked.

### 3. Config re-tune — the deployed constants were set on the CRIPPLED bidder
Sweep FLOOR × DEBIAS_PCTL (pimc-scored soloist GP vs PIMC defenders, same deals):
- **FLOOR=0.80, DEBIAS=0.85 is best (+3.478)**; current FLOOR=0.70/0.80 ranked **8th of 9 (+2.635)**.
- **Raising FLOOR 0.70 → 0.80 is worth ~+0.84 GP/game.** All 3 FLOOR=0.80 configs top the ranking.
- Mechanism: the fix opened non-piros contracts; a higher floor gates the overconfident escalations.

### 4. The duri leak — root cause + fix
The composer (`bidder.py:122-125`) scores terített (open-hand) durchmars with the **closed-hand**
`p_duri` and amplifies the value ×2/×4. The duri heads are the worst-calibrated in the model
(overconfident — exp28), so `(2p−1)×24` turns a small calibration error into a huge EV error →
massive terített-duri over-bidding (made only 13–26%). Fix: env-gated **`DURI_TERIT_MULT`**
(default 1.0 = off) discounts the terített-duri make-prob. Sweep (FLOOR=0.80, DEBIAS=0.85):
| DURI_TERIT_MULT | 1.0 (off) | 0.7 | 0.5 | 0.3 | 0.15 | 0.0 |
|---|---|---|---|---|---|---|
| metric | +3.165 | +3.229 | +3.251 | +3.389 | +3.397 | +3.429 |
Monotonic — **~+0.26 GP/game** from suppressing terített-duri. At mult ≤ 0.5 its EV is always
negative (the net can't identify makeable ones), and the pass rate is unchanged → suppressing it
**redirects to a better contract, not a fold**. Recommend **0.3** (robust; keeps the door cracked).

### 5. exp27 kontra — re-confirmed on the fixed distribution
The promoted per-unit kontra HOLDS with non-piros ultis in the mix: ulti kontra fires only **4%**
(trump-gate; old blind rule ~94%), **70% profitable** when it fires; suit-agnostic (piros ulti 4%,
non-piros ulti 3%). No re-run needed.

## RECOMMENDED CONFIG — validated (TWO tests)
`FLOOR=0.80, DEBIAS_PCTL=0.85, DURI_TERIT_MULT=0.3`.
1. **Proxy** (harness.evaluate, soloist GP vs a fixed PIMC defender, N=2500 same deals):
   CURRENT +2.954 → RECOMMENDED +3.643 = Δ +0.689 GP/game.
2. **TRUE head-to-head** (exp32, the two configs COMPETE at the same table — after making
   FLOOR/DEBIAS/DURI per-bidder params; N=2000 deals × 3 seatings = 6000 games, position-neutralized):
   **RECOMMENDED beats CURRENT by +0.399 GP/game.** It wins fewer bids (26% — more selective)
   but nets more by avoiding the leaks and punishing the current config's over-bids as a defender.
   Edge concentrates at the forehand (+1.13/seat0).
The true head-to-head (+0.40) is the rigorous number — smaller than the proxy (+0.69) but solidly
positive. Both stack ON TOP of the bug fix (the larger, separate gain).

## Capstone — the duri leak is GONE in the full auction (self-play N=6000)
| config | duri-family freq | duri GP/deal | made% | terített-duri bids |
|---|---|---|---|---|
| buggy (piros-only) | 1.5% | −0.11 | 35% | 88 |
| fixed (default 0.70/0.80/1.0) | 10.0% | **−0.705** | 25% | 599 |
| fixed + RECOMMENDED (0.80/0.85/0.3) | 4.1% | **+0.257** | 34% | **73** |
The recommended config cuts terített-duri bids 599→73 and flips the duri-family from −0.705 to
**+0.257 GP/deal** — the frontier now only bids duri when genuinely makeable. Per-seat with the
recommended config: P0 forehand **+0.67** (now the best seat), P1 +0.02, P2 −0.69 (rotates out).

## Deployed-engine changes this session
1. `auction.py` `_search` generator fix — a clear correctness bug (applied, validated live).
2. `bidder.py` `DURI_TERIT_MULT` hook — **default 1.0 = OFF, deployed behaviour byte-identical**
   until the env var is set. The recommended 0.3 is a proposal to flip, pending milan's sign-off.
Nothing else changed. FLOOR/DEBIAS are env config (not code) — recommend updating play.py's
`os.environ.setdefault` defaults to 0.80/0.85 after milan reviews.

## Deferred (scoped, ready for a dedicated session)
- **Exploit play / opponent modeling** ([[project_exp25_strength_ceiling]] FRONTIER — the biggest
  lever): exp31/PLAN.md has the full Experiment #1 design (replace the god double-dummy leaf with
  a rollout against a modeled imperfect defender; test X>P>G on betli). Needs a careful tractable
  leaf design (per-candidate rollout is expensive) — a dedicated build, not a rushed prototype.
- Realistic-label bidding-net retrain (~+1.3 ceiling, exp25); duri-head recalibration (proper fix
  vs the DURI_TERIT_MULT band-aid).

## Files
`experiments/30_frontier_tuning/` : NIGHT_PLAN.md, sweep.py/run_one.py (config), duri_sweep.py,
validate.py, *_results.tsv, *.log. `experiments/29_frontier_table/` : selfplay.jsonl (fixed) vs
selfplay_buggy.jsonl, analyze.py. `experiments/31_exploit_play/PLAN.md`.
