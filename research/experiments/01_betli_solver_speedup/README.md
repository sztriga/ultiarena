# Experiment 1 — Betli α-β solver speedup

## Goal
Make the exact betli solver fast enough to use as ground truth in PIMC outcome
benches at realistic α (0.5–0.7), without losing soundness.

## Changes
Two additions in `trickster/_solver_core.pyx`:

1. **Early-WIN predicate (no-holes).** At any node before a trick is led, if in
   every suit the soloist holds cards their max card is strictly outranked by
   every live opponent card in that suit → soloist can never take a trick →
   return WIN. Sound because in betli the soloist leads exactly trick 1; from
   trick 2 on, defenders keep the lead until the end, so sol's high cards are
   forced discards.
2. **Defender equivalence-group cull.** For MIN nodes, defender cards in the
   same suit are interchangeable iff no live opponent card has strictly-between
   strength. Collapses redundant branches.

The existing soloist suit-dominance cull is kept.

## Soundness check
10 random α=0.7 seeds, slow vs fast solver: **10/10 verdicts agree**
(2 slow-solver runs timed out >10 min and were excluded; the 8 that finished all
matched).

## Speed
20 fresh random α=0.7 seeds, fast solver:

| metric  | value     |
|---------|-----------|
| total   | ~0.45 s   |
| mean    | 0.022 s   |
| median  | 0.01 s    |
| max     | 0.25 s    |
| split   | 11 def / 9 sol |

Prior slow-solver baseline (same α, 5 seeds): mean ≈ 200 s, max 577 s. Roughly
**~2000–3000× speedup**, no >1 s outlier in 20 deals.

## Status
Fast solver is the default going forward. Both culls live in
`_solver_core.pyx`; the slow path is gone.

See experiment 2 (`../02_betli_alpha_sweep/`) for an α-sweep characterization
of win rate and timing across the whole α range.
