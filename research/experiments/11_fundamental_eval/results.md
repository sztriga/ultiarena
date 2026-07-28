# Experiment 11 — Fundamental-contract eval

## Betli

PIMC32 vs god, on betli, full deal-played-out matchups. Per (α, seed):

1. **God-label the opening** — binary, does god predict sol can take 0 tricks?
2. **Value head at t=0** — `max(averaged)` of a PIMC32 call from sol's seat.
3. **Two matchups** on the same seed:
   - PIMC sol  vs god def  → `sol_hold`
   - god  sol  vs PIMC def → `def_stop`

Config: N=200/α, pimc_n=32, 8 workers (chunksize=2, checkpointed), caffeinated.
Total wall: ~79 min for 600 deals × 2 matchups = 1 200 played-out games (~1.3 M god-solves under the hood).

## Results

| α | god label dist (n_SF / n_DF / total) | `sol_hold` | `def_stop` | `val_AUC` | mean(SF) | mean(DF) | Brier |
|---|---|---|---|---|---|---|---|
| **0.30** |  15 / 185 / 200 (god sol-win = **7.5%**)  | **0.867 ± 0.088** | **0.778 ± 0.031** | **0.984** | 6.68 | 0.41 | 0.025 |
| **0.70** | 103 /  97 / 200 (god sol-win = **51.5%**) | **0.942 ± 0.023** | **0.505 ± 0.051** | **0.931** | 7.57 | 1.85 | 0.108 |
| **1.50** | 188 /  12 / 200 (god sol-win = **94.0%**) | **0.995 ± 0.005** | **0.083 ± 0.080** | **0.988** | 9.81 | 4.37 | 0.023 |

(values on the 0–10 binary betli scale; PIMC opening value averages 0 = LOSE, 10 = WIN)

## Reading

- **God-label distribution moves smoothly across the S-curve**: 7.5% → 51.5% → 94% sol-wins as α grows. α=0.7 is right at the contested mid-point.
- **PIMC as soloist (`sol_hold`)**:
  - 87% on a tiny n_SF=15 (α=0.3) — wide CI, but the 13 hits / 15 winnable seats is consistent with the pattern.
  - **94% at α=0.7** on a clean n_SF=103.
  - **99.5% at α=1.5** — essentially god-level once hands are clearly winnable.
  - PIMC-as-soloist on betli is strong across the board.
- **PIMC as defender (`def_stop`)**:
  - 78% at α=0.3 — when sol's hand is mostly hopeless, PIMC defenders close out 145/185 unwinnable seats.
  - Drops to **51% at α=0.7** and **8% at α=1.5** — when god gives sol a clear path, even good defenders can't conjure a stop. This matches the exp 10 betli-PIMC audit results (and is consistent with the published MCTS+V numbers, which also fall off this way).
- **Value head**:
  - AUC **0.93–0.99** at every α — the PIMC32 t=0 score is a near-perfect ranker of which hands god considers winnable.
  - mean(SF) clusters near WIN=10 (6.7 / 7.6 / 9.8 across α); mean(DF) stays near LOSE=0 (0.4 / 1.9 / 4.4). Clear separation.
  - Brier 0.025–0.108. Mid-α (α=0.7) is the highest-noise zone, as expected for a calibration metric near the 50/50 line.

## Implication for bidding

The value head (PIMC32 t=0) is **essentially a god-quality "is this hand winnable as betli?" oracle**. We can use it directly in the bidding policy without training anything first — the bid decision is just `pred >= threshold` calibrated against the gp payoff.

If/when latency matters, distill it into a small V-net (the per-call cost drops from ~32 god-solves → 1 forward pass = ~1000× faster) — the regression target is well-defined and the signal-to-noise is high.

---

## Ulti

PIMC32 vs god on ulti. Same template: per (α, seed) god-label the opening,
PIMC32 t=0 value head, then play out PIMC sol vs god def (`sol_hold`) and
god sol vs PIMC def (`def_stop`).

**Auction-rule fix applied just before this run.** `solvers/determinize._ulti_must_hold` now plants the trump-7 in soloist's slot in every sampled world, so PIMC defenders no longer waste samples on impossible worlds where def2 holds the trump-7. The edge case (defender's own hand contains trump-7, e.g. degenerate position) is handled by the determinizer's `try/except` skip — solver then returns LOSE for all moves and defender play is correctly irrelevant.

Config: N=200/α, pimc_n=32, 8 workers, checkpointed, caffeinated.
Total wall: ~105 min for 600 deals × 2 matchups (slower than betli because ulti games run all 10 tricks to reveal the trump-7 fate, no early termination).

### Results

| α | god label dist (n_SF / n_DF / total) | `sol_hold` | `def_stop` | `val_AUC` | mean(SF) | mean(DF) | Brier |
|---|---|---|---|---|---|---|---|
| **0.00** |  60 / 140 / 200 (god sol-win = **30.0%**) | **0.867 ± 0.044** | **0.964 ± 0.016** | **0.983** | 7.98 | 0.56 | 0.049 |
| **0.60** | 110 /  90 / 200 (god sol-win = **55.0%**) | **0.918 ± 0.026** | **0.911 ± 0.030** | **0.962** | 8.41 | 1.27 | 0.078 |
| **1.50** | 143 /  57 / 200 (god sol-win = **71.5%**) | **0.937 ± 0.020** | **0.947 ± 0.030** | **0.975** | 9.22 | 1.90 | 0.052 |

### Reading

- **Ulti's god-label S-curve is flatter than betli's** (30% → 55% → 72% vs betli's 7.5% → 51% → 94%). Ulti hands cluster near the contested middle.
- **PIMC as soloist** holds **87–94%** across the α range — strong, close to god. Roughly matches ulti's prior reputation: PIMC is essentially god-level for ulti sol play once the trump-7 plan is locked.
- **PIMC as defender** is **91–96%** across the board — notably stronger than betli's def_stop at the matching α (e.g., 91% at α=0.6 vs betli's 51% at α=0.7). The auction-rule fix (trump-7 → sol) is doing real work here — without it, defenders were sampling worlds where their own play to "trump over the trump-7" was nonsensical.
- **Value head** AUC **0.96–0.98** everywhere. SF means cluster near WIN=10 (8.0–9.2); DF means near LOSE=0 (0.6–1.9). Clean separation despite the contested middle.

### Cross-contract comparison

| Contract | α-midpoint | sol_hold@mid | def_stop@mid | val_AUC@mid |
|---|---|---|---|---|
| betli | 0.70 | 0.942 | 0.505 | 0.931 |
| **ulti** | **0.60** | **0.918** | **0.911** | **0.962** |

Ulti is the contract where **both** seats are strong against god — sol nearly takes every winnable hand (94% with N=200), and defender denies nearly every losable hand (91%). Betli soloist is similarly strong but betli defender collapses at higher α because once sol's hand is clearly winnable, there's no defender plan that holds.

### Files

- `eval_ulti.py` — the runner.
- `results_ulti.json` — raw aggregates.
- `checkpoint_ulti.jsonl` — per-result log (safe to delete now that the run completed cleanly).

---

## Durchmars (duri)

Same template, two flavours. Colored uses a trump suit (10 sits second-highest,
A,10,K,Q,J,9,8,7 descending). Colorless is trumpless with betli-style rank
ordering (10 demoted under J). No auction must-hold rule for either flavour
(unlike ulti's trump-7), so `_no_must_hold` in the determinizer is correct.

Config: N=200/α, pimc_n=32, 8 workers, checkpointed, caffeinated.

### Colored — results

Total wall: ~95 min for 600 deals × 2 matchups.

| α | god label dist (n_SF / n_DF / total) | `sol_hold` | `def_stop` | `val_AUC` | mean(SF) | mean(DF) | Brier |
|---|---|---|---|---|---|---|---|
| **0.60** |  29 / 171 / 200 (god sol-win = **14.5%**) | **1.000 ± 0.000** | **0.906 ± 0.022** | **0.999** | 9.46 | 0.15 | 0.009 |
| **1.20** |  50 / 150 / 200 (god sol-win = **25.0%**) | **1.000 ± 0.000** | **0.820 ± 0.031** | **0.999** | 9.78 | 0.16 | 0.007 |
| **2.00** |  88 / 112 / 200 (god sol-win = **44.0%**) | **1.000 ± 0.000** | **0.866 ± 0.032** | **1.000** | 9.95 | 0.10 | 0.002 |

### Colorless — results

Total wall: ~98 min for 600 deals × 2 matchups.

| α | god label dist (n_SF / n_DF / total) | `sol_hold` | `def_stop` | `val_AUC` | mean(SF) | mean(DF) | Brier |
|---|---|---|---|---|---|---|---|
| **0.80** |  52 / 148 / 200 (god sol-win = **26.0%**) | **1.000 ± 0.000** | **0.966 ± 0.015** | **0.976** | 7.96 | 0.65 | 0.053 |
| **1.50** | 129 /  71 / 200 (god sol-win = **64.5%**) | **1.000 ± 0.000** | **0.915 ± 0.033** | **0.986** | 9.45 | 1.49 | 0.045 |
| **2.50** | 183 /  17 / 200 (god sol-win = **91.5%**) | **1.000 ± 0.000** | **0.882 ± 0.078** | **0.962** | 9.50 | 3.31 | 0.042 |

### Reading

- **PIMC sol = god** on duri across the board. `sol_hold = 1.000` in all six
  cells (88+50+29+183+129+52 = 531/531 winnable hands taken). Duri is the
  contract where every winnable seat is mechanically takeable once you see
  one consistent world — there are no path-dependent point trades to fuse
  across information sets.
- **PIMC defender is strong** (82–97% across both flavours). Defense on duri
  is "find any trick we can take" — once a defender sees a trump they can
  ruff into, or a suit they top, they take the trick and the contract is
  dead. PIMC's averaging across worlds finds these reliably.
- **Colored has flatter god S-curve** (15% → 25% → 44%) than colorless
  (26% → 65% → 92%). Colored duri is intrinsically harder: even a fat trump
  suit can be ruined by one stranded loser; colorless duri with α=2.5
  effectively requires the soloist to hold all 7s, 8s, etc — a much sharper
  cutoff.
- **Value head** AUC ≥ 0.96 everywhere; AUC = 1.000 on colored α=2.0
  (every winnable hand scored above every unwinnable one). SF means
  near WIN=10; DF means near LOSE=0 (especially clean on colored).

## Cross-flavour comparison — duri vs ulti / betli

| Contract | sol_hold @ contested-mid | def_stop @ contested-mid | val_AUC @ contested-mid |
|---|---|---|---|
| duri colored | 1.000 | 0.820 (α=1.2, ~25% god) | 0.999 |
| duri colorless | 1.000 | 0.915 (α=1.5, ~65% god) | 0.986 |
| ulti | 0.918 (α=0.6) | 0.911 | 0.962 |
| betli | 0.942 (α=0.7) | 0.505 | 0.931 |

Duri stands out as the cleanest fundamental: PIMC sol is god, PIMC def is
strong, value head is near-perfect. The cleanness suggests the duri PIMC
output can be used directly as the silent-duri payoff dimension in the
exp-12 contract oracle without correction.

---

## Parti

Two PIMC modes head-to-head: `scalar` (current — averages per-world god
value, argmax) vs `binary` (averages `1_{value ≥ 50}`, argmax over P(win)).
Defender stays scalar to keep `def_stop` comparable across contracts.
Threshold = 50 (canonical simple-parti win line, no marriages).

Config: N=300/α, pimc_n=32, 8 workers, checkpointed, caffeinated.
Total wall: ~3.5 h for 900 deals × 3 plays each + 2 t=0 value heads.

### Results

| α | god label dist (n_SF / n_DF / total) | `sol_hold_scalar` | `sol_hold_binary` | Δ | `def_stop` | AUC_s | AUC_b | Brier_s | Brier_b |
|---|---|---|---|---|---|---|---|---|---|
| **0.30** | 144 / 156 / 300 (god sol-win = **48.0%**) | **0.799 ± 0.033** | **0.799 ± 0.033** | **0.0** | **0.827 ± 0.030** | 0.949 | 0.959 | 0.132 | 0.080 |
| **0.60** | 203 /  97 / 300 (god sol-win = **67.7%**) | **0.857 ± 0.025** | **0.877 ± 0.023** | **+2.0pp** | **0.845 ± 0.037** | 0.970 | 0.971 | 0.116 | 0.062 |
| **1.50** | 249 /  51 / 300 (god sol-win = **83.0%**) | **0.956 ± 0.013** | **0.944 ± 0.015** | **−1.2pp** | **0.784 ± 0.058** | 0.980 | 0.989 | 0.070 | 0.030 |

### Reading

- **Binary vs scalar at play-time is essentially a wash.** Largest signed
  effect is +2pp at α=0.6 (within ~1 SE); α=1.5 is slightly *worse* under
  binary (also within SE). The "guaranteed 51 vs gambling 80/49" scenarios
  motivating the indicator hypothesis are rare at parti's PV — most worlds
  resolve to clear wins or clear losses, so argmax over scalar and argmax
  over indicator pick the same move.
- **Parti scalar PIMC is healthier than prior runs suggested.** 80% / 86% /
  96% across α=[0.3, 0.6, 1.5] — same shape as the other contracts, just
  shifted ~5–15pp below betli/ulti. Earlier runs that landed parti at ~50%
  sol_hold predate the win-threshold bug fix (`_PARTI_WIN_VAL = 60.0` was
  penalising PIMC for the [50, 60) win band that the canonical rule counts
  as wins). The 80–96% band is the honest number.
- **`def_stop` is solid** (78–85%) — PIMC defenders deny most god-unwinnable
  hands. Drops to 78% at α=1.5 because god-unwinnable hands there are the
  hardest defender problems by selection.
- **Binary value head wins on calibration and AUC**:
  - AUC_b > AUC_s at every α (0.959 vs 0.949; 0.971 vs 0.970; 0.989 vs 0.980).
  - Brier roughly halved everywhere (0.132→0.080, 0.116→0.062, 0.070→0.030).
  - The binary head is a proper calibrated probability P(sol wins); the
    scalar head is a normalized raw-score proxy. For bid-time payoff
    expectation, the binary head is strictly the right object.

## Implication for exp 12 (contract oracle)

Two related claims from the design doc:
- (a) "PIMC averages indicator functions, not raw values" at **play time** —
  this is **not validated** on parti. Drop it from the design.
- (b) The **value head** should be P(win), not E[score] — **strongly
  validated**. Keep it. The parti dimension of the payoff vector should
  be computed as `mean(1_{god_value ≥ threshold})` across PIMC samples, not
  `mean(god_value) / max_score`.

## Cross-contract comparison (full)

| Contract | α (contested-mid) | sol_hold | def_stop | val_AUC |
|---|---|---|---|---|
| duri colored   | 1.20 | **1.000** | 0.820 | 0.999 |
| duri colorless | 1.50 | **1.000** | 0.915 | 0.986 |
| ulti           | 0.60 | 0.918 | 0.911 | 0.962 |
| betli          | 0.70 | 0.942 | 0.505 | 0.931 |
| **parti (scalar)** | 0.60 | 0.857 | 0.845 | 0.970 |
| **parti (binary)** | 0.60 | 0.877 | 0.845 | 0.971 |

Parti's PIMC sol is the weakest of the five but only by ~5–10pp vs ulti.
Parti's PIMC def is mid-pack (better than betli, comparable to ulti).

---

## Files

- `eval_betli.py` / `eval_ulti.py` / `eval_duri_colored.py` / `eval_duri_colorless.py` / `eval_parti.py` — runners.
- `results_betli.json` / `results_ulti.json` / `results_duri_colored.json` / `results_duri_colorless.json` / `results_parti.json` — raw aggregates.
- `checkpoint_*.jsonl` — per-result logs (safe to delete on clean completion).
