# parti V-net — pipeline status

Per-contract research log. See `docs/RESEARCH_PIPELINE.md` for the canonical
4-phase recipe; this file embeds the parti-specific artifacts as they land.

---

## Phase 1 — Register ✅

- Dealer: `eval.dojo.deal_parti(seed, alpha, suit_sigma)` — uses `PARTI_SPEC`
  (`bid_action_offset=0`, trump-count weights `{2: 0.10, 3: 0.40, 4: 0.30,
  5: 0.15, 6: 0.05}`, no mandatory trump ranks).
- Solver: `ultisolver._solver_core` with `_term_parti` + `_cull_parti_blocks`
  (point-boundary-respecting block-equivalence cull).
- Audit: 30 trick-1 deals + 30 mid-game positions across α∈{0.0, 0.5, 1.0},
  exact-match vs no-cull baseline; 36× speedup at trick-1, 7.8× mid-game,
  0 cross-suit violations.
- Registered in `vnet.contracts.REGISTRY["parti"]`: `feature_dim=152`
  (132 base + 20 trump tail), `target_kind="regression"`.

## Phase 2 — α-calibration ✅

100 deals/α, seed_base=42, god solver only.

| α | mean | median | p25 | p75 | min | max | sol win-rate (≥51) | mean wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 29.9 | 30.0 | 20.0 | 47.5 | 0 | 80 | 0.110 | 0.99 |
| 0.30 | 39.9 | 40.0 | 20.0 | 60.0 | 0 | 90 | 0.260 | 0.58 |
| 0.50 | 47.5 | 50.0 | 30.0 | 70.0 | 0 | 90 | 0.390 | 0.77 |
| 0.70 | 52.3 | 55.0 | 40.0 | 70.0 | 0 | 90 | 0.500 | 0.69 |
| 1.00 | 60.7 | 65.0 | 40.0 | 80.0 | 10 | 90 | 0.630 | 0.36 |

**Reading**: monotonic, well-spread. The 51-pt win threshold gets crossed
around **α ≈ 0.7** (sol_wr 50%). α∈[0.0, 1.0] uniform mix should give a
balanced phase-4 dataset (no saturation at extremes).

**Wall-time implication**: ~0.7s/deal average, culled solver. Estimated
datagen time for 200k deals at 8 workers ≈ 5 hours.

**Update (post-move-order experiment)**: see `move_order_experiment.md`.
Switched parti's default ordering to `pts_first` (3× speedup vs the legacy
generic ordering, bit-identical values). Phase 4a datagen estimate now
~1.7h for 200k deals at 8 workers.

## Phase 3 — PIMC eval ✅ (soloist, t=0)

100 deals/α × pimc_n=32, seed_base=42, 4 workers, ~7h wall. Defender-side
and t>0 eval still pending. See `../../vnet/parti/results/phase3_pimc_eval_soloist_t0.md`.

| α | god mean | PIMC mean | bias (P−G) | MAE | RMSE | Pearson r |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 29.90 | 29.85 | −0.05 | 5.32 | 7.25 | 0.922 |
| 0.30 | 39.90 | 41.03 | +1.13 | 5.47 | 7.38 | 0.934 |
| 0.50 | 47.50 | 47.04 | −0.46 | 5.20 | 7.28 | 0.944 |
| 0.70 | 52.30 | 52.91 | +0.61 | 5.59 | 8.01 | 0.936 |
| 1.00 | 60.70 | 60.32 | −0.38 | 4.21 | 7.15 | 0.949 |

**Reading**: PIMC is essentially unbiased (|bias| ≤ 1.13 pts) but has a
flat ~5-pt MAE / ~7-pt RMSE floor across all α, r∈[0.92, 0.95]. The
critical α=0.7 bucket sits right on the 51-pt win threshold, where the
~5.6-pt MAE straddles win/loss for many borderline deals. That's the gap
a V-net can target — closing it on the bias side is essentially nothing,
but trimming MAE/RMSE around the threshold should improve bid-time EV.

## Phase 4a — Train (scaling sweep) ⏳

Pending phase 3 + chain script.

## Phase 4b — MCTS+V eval ⏳

Pending phase 4a + contract-parameterised `scripts/mcts_v_eval.py`.
