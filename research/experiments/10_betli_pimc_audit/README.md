# Experiment 10 — Betli PIMC defender audit

## TL;DR

The V-net research in [`vnet/betli/`](../../vnet/betli/)
compared MCTS+V against a PIMC defender. The PIMC defender it
compared against was anti-optimal — the matchup script used the raw
`pimc_decision(...)` return (soloist-perspective argmax) without
flipping for defender seats. Corrected vanilla PIMC matches or beats
the best published NN at every α value.

## The bug

`solvers.pimc.pimc_decision` returns `(argmax, averaged)` where
`argmax` is the soloist-perspective best move. The two affected scripts
fed `chosen` straight to both seats:

```python
chosen, _ = _pimc.pimc_decision(true_pos=pos, contract=_CONTRACT, ...)
return chosen   # ← defender plays the move best for the SOLOIST
```

Fix: defenders take the argmin of `averaged` instead. Centralised in
[`eval/pimc_matchup.py`](../../eval/pimc_matchup.py) so it can't recur:

```python
viewer = pis_bridge.current_player(pos)
if viewer != 0 and averaged:
    chosen = min(averaged, key=lambda c: averaged[c])
```

## A/B confirming the bug, N=80, α=0.7

| variant | def_stop | n |
|---|---|---|
| unflipped (bug) | **5.1% ± 3.5%** | 2/39 |
| flipped (patch) | **41.0% ± 7.9%** | 16/39 |
| **Δ** | **+35.9 pp** | |

## Corrected α sweep (this experiment), N=200, pimc_n=16

| α | n_def | **vanilla PIMC** | MCTS+V ref (CONCLUSIONS.md) | Δ |
|---|---|---|---|---|
| 0.30 | 188 | **0.771 ± 0.031** | 0.755 | +1.6 pp |
| 0.50 | 147 | **0.537 ± 0.041** | 0.503 | +3.4 pp |
| 0.70 |  94 | **0.457 ± 0.051** | 0.444 | +1.3 pp |
| 0.80 |  72 | **0.361 ± 0.057** | 0.320 | +4.1 pp |
| 1.00 |  36 | **0.417 ± 0.082** | 0.295 | **+12.2 pp** |

MCTS+V ref = "1M K=1, 308k params" row from
`vnet/betli/` (CONCLUSIONS retired). Vanilla PIMC
matches the NN within 1σ for α ≤ 0.8 and pulls clearly ahead at α=1.0
— the hardest distribution, where the NN was claimed to approach the
"strategy-fusion ceiling".

## Implication

The whole NN architectural ladder
(V-head → PIMC+V → MCTS+V → policy head → K=8 PIMC-soft labels) was
real *relative to its starting baseline* (PIMC+V ~0.30) but **never
crossed back over vanilla PIMC**. Reporting that suggests "MCTS+V
approaches the strategy-fusion ceiling that PIMC sets" needs revising;
vanilla PIMC already sits at or above that ceiling on all evaluated α.

## Reproducing

```bash
PYTHONPATH=. python3 experiments/10_betli_pimc_audit/alpha_sweep.py
```

Writes `sweep.json` next to the script. Parallel run with 8 workers,
~45 minutes on an M-series Mac (limited by perf+efficiency core split
plus L3 contention from 8 × 32 MB transposition tables).

## Related work

- [`eval/pimc_matchup.py`](../../eval/pimc_matchup.py) — shared,
  defender-aware PIMC primitives.
- [`vnet/betli/`](../../vnet/betli/)
  — original V-net training scaffolding. The eval artifacts (matchup,
  α-sweep, ROC, CONCLUSIONS) were retired with the bug discovery; the
  data/model pipeline (`data_gen.py`, `features.py`, `net.py`,
  `net_nnue.py`, `train.py`) is kept for a future retrain.
- [`experiments/09_duri_pipeline/pimc_alpha_sweep.py`](../09_duri_pipeline/pimc_alpha_sweep.py)
  — the same pipeline applied to durchmars.
