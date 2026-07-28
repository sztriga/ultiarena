# betli V-net — pipeline status

Per-contract research log. See `docs/RESEARCH_PIPELINE.md` for the canonical
4-phase recipe; this file embeds the betli-specific artifacts.

---

## Phase 1 — Register ✅ (legacy)

- Dealer: `eval.dojo.deal_betli(seed, alpha, suit_sigma)` — trumpless,
  weights cards by inverse betli-strength.
- Solver: `trickster._solver_core` with `_term_betli` + `_cull_betli_dominant`
  (soloist suit-dominance + defender equivalence-group cull).
- Registered in `vnet.contracts.REGISTRY["betli"]` (`feature_dim=132`,
  `target_kind="binary"`).

## Phase 2 — α-calibration ✅ (legacy)

Original 2026-05 sweep predates the contract-parameterised
`scripts/alpha_calibrate.py`; not re-run because the existing data_gen
chains (200k / 1M / 10M) already span α∈[0.3, 1.0] uniform and the deal
distribution is well-characterised by the existing chain logs.

If you want a clean re-run in the new format:
```bash
PYTHONPATH=. python3 scripts/alpha_calibrate.py \
    --contract betli --alphas 0.3,0.5,0.7,0.8,1.0 --n 200 --seed-base 42
```
→ `vnet/betli/results/phase2_alpha_calib.{json,md,png}`

## Phase 3 — PIMC eval ✅ (legacy, via exp 03)

Done in `experiments/03_pimc_information_limits/` (the old framework). Not
re-cast in the new pipeline format because the legacy results are already
published in that experiment's CONCLUSIONS.

## Phase 4a — Train scaling sweep ✅

12-cell grid: 4 archs × 3 dataset sizes (200k / 1M / 4M-subsample of 10M).
See `../../vnet/betli/results/phase4_scaling.md` for the table.

Headline: **best val_auc = 0.8475** (NNUE small @ 4M). Cell-to-cell spread
within 4M = 0.0007 AUC. Capacity buys ~0; data buys ~+0.008 going 200k→1M
and ~+0.003 going 1M→4M.

## Phase 4b — MCTS+V eval ✅

500 deals × 5 α × 12 models = 30k full-game playouts. See
`../../vnet/betli/results/phase4_mcts_v_eval.md` for the table.

Per-α column spread across all 12 cells sits at or just above ±1 SE — no
cell systematically beats another.

## Headline finding — strategy-fusion ceiling

The MCTS+V defender's `def_stop` is **architecturally bounded** by the
PIMC-based inference scheme, not by training-data quality, training-set
size, V-head capacity, or feature engineering. The 12-cell grid shows:

- 20× more data (200k → 4M) buys ~+0.01 AUC and **does not shift `def_stop`
  outside noise**.
- 35× more capacity (MLP 67k → NNUE 3.3M) buys **0 measurable AUC** and 0
  measurable `def_stop` shift.
- Per-α `def_stop` is pinned in a tight band: e.g. α=0.3 ≈ 0.74±0.01 across
  all 12 cells.

**Closing the strategy-fusion gap** requires a fundamentally different
inference scheme — ISMCTS, CFR, or a value-aware PIMC variant that
selectively biases determinization. Not more V-net work.

## Recommendation for further betli work

- **Stop training betli V-nets.** Pipeline is saturated.
- Keep the current `v_nnue_s_4M.pt` and `v_mlp_l_1M.pt` as the production
  checkpoints (the latter is what the UI wires).
- Future inference-scheme work (ISMCTS / CFR) gets its own experiment; the
  V-nets above can be plugged in unchanged as the value head.

## Artifacts

```
vnet/betli/
  data/    train_mix_a03_10_t0_8_{200k,1M,10M}.npz (+ mmap caches)
  models/  v_{mlp_s,mlp_l,nnue_s,nnue_l}_{200k,1M,10M}.pt + .json
  results/
    phase4_scaling.{json,md}
    phase4_mcts_v_eval.{json,md}
```
