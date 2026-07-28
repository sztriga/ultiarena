# Experiment 9 — Durchmars pipeline

End-to-end research pipeline for the **durchmars** (duri) contract,
matching the workflow we established for betli (exp 1+2) and ulti (exp 8):

1. **Solver bench** — characterise god-solver speed with the úr-vagyok
   predicate ON vs OFF (`solver_bench.py`).
2. **α sweep** — god-solver win rate vs hand-bias strength, both colored
   and colorless (`alpha_sweep.py`).
3. **PIMC vs god α sweep** — measure how many winnable seats PIMC takes
   and how many unwinnable ones it stops, vs god as opponent
   (`pimc_alpha_sweep.py`).

Colored duri uses an ordinary trump suit (10 second-highest in trick
order, A,10,K,Q,J,9,8,7 descending). Colorless duri has no trump and
the natural numerical rank order (A,K,Q,J,10,9,8,7 — i.e. 10 sits
between 9 and J, like betli demotion). The Cython core picks the right
rank function via the existing `betli` flag; the dojo dealer pair lives
in `eval/dojo.py` (`deal_durchmars_colored`, `deal_durchmars_colorless`).

## Solver bench results (200 deals each, M-series Mac)

| mode | α | win% | mean | p99 | max |
|---|---|---|---|---|---|
| colored (úr ON) | 0.6 | 10.0% | **13.6 ms** | 35 ms | 199 ms |
| colored (úr OFF) | 0.6 | 10.0% | 17.2 ms | 228 ms | 252 ms |
| colorless (úr ON) | 1.5 | 64.5% | **12.1 ms** | 13.4 ms | 13.8 ms |
| colorless (úr OFF) | 1.5 | 64.5% | 12.0 ms | 13.0 ms | 13.1 ms |

Predicate helps a lot on colored tails (p99 6×); colorless is already
flat so the predicate is roughly free.

## Reproducing

```bash
PYTHONPATH=. python3 experiments/09_duri_pipeline/solver_bench.py
PYTHONPATH=. python3 experiments/09_duri_pipeline/alpha_sweep.py
PYTHONPATH=. python3 experiments/09_duri_pipeline/pimc_alpha_sweep.py
```

Each script writes its own `*.json` next to itself. PIMC sweep
parallelises across 8 workers (configurable in the file) and takes
~40 minutes for 5×5 α grid × 50 deals × 2 matchups.

## Related work

- `eval/pimc_matchup.py` — shared PIMC ↔ god primitives used by every
  contract's PIMC α sweep (centralises the defender-side argmin flip).
- `experiments/02_betli_alpha_sweep` — betli god-solver sweep.
- `experiments/08_ulti_alpha_sweep` — ulti god-solver sweep.
- `experiments/10_betli_pimc_audit` — corrected vanilla-PIMC defender
  baseline for betli (post defender-MIN bug fix).
