# Exp 19 — Split the colorless contracts off into structural models

**Started:** 2026-06-13

## Motivation (from exp 18 god-check + audit)

- god_check showed betli/durchmars bleed because the v18a shared net is
  **overconfident in its decision region** — it commits to hands god
  rates ~15% winnable. Not the play handicap.
- Audit (`experiments/18.../audit_colorless.py`): durchmars/betli win-prob
  is largely determined by **hole/void structure**. A 6-feature logistic
  beats the shared net's Brier ~2× (duri 0.13 vs 0.32, betli 0.12 vs 0.20).
- Colorless contracts use the **10-low rank order** (Ten demoted under the
  Jack), which the shared body — dominated by 230k+ trump examples in the
  10-high order — represents with the wrong strength.

## Design

Three pickup models routed by contract (`vnet/pickup/composite.py`):

```
CompositePickup.predict(X, contract):
  parti, ulti  → Exp18Pickup (v18a trump multi-head)   KEEP AS-IS
  betli        → ColorlessPickup (structural net)       NEW
  durchmars    → ColorlessPickup (structural net)       NEW
```

betli and durchmars get **separate** structural nets (opposite objective:
run-every-trick vs duck-every-trick; both have ample positives in the
biased data, so no need to share a body).

### Colorless structural model (`vnet/pickup/colorless.py`)

- Input: 20 suit-invariant structural features in the **10-low** order —
  per suit (sorted) length / top-run / losers, plus aggregates (voids,
  singletons, longest, n_ace/ten/king, total losers/top-run).
- Tiny MLP 20→32→16→1.
- **Trained on the biased god data** (exp 15 `*_god_250k.npz`, 35–48%
  positive). Structure is distribution-invariant — P(win | structure) is
  the same however the hand was dealt — so a biased-trained structural
  model should calibrate on the α=0 deployment distribution. That is the
  hypothesis Tier 1 checks; if it fails, mix in α=0.

## Eval (reuse exp 18 harness, locked seeds 100000–102999)

1. **Tier 1 calibration:** structural nets vs god on biased held-out AND
   the cached α=0 50k sets (distribution-robustness test).
2. **god_check (decisive):** god-win% of the betli/duri hands the
   *composite* auction commits to. Target: climb from v18a's ~15% — i.e.
   stop bidding un-winnable hands.
3. **bleeders / Tier 3:** does betli's −3.79 GP/def and the −1,070 total
   bleed shrink? parti/ulti must be unchanged (same v18a).

## Files

```
train_colorless.py     trains betli + duri structural nets on biased god data
eval_composite.py      god_check + bleeders for CompositePickup vs v18a
colorless_betli.pt
colorless_durchmars.pt
results.md             (written at end)

vnet/pickup/colorless.py    features + ColorlessStructNet + ColorlessPickup
vnet/pickup/composite.py    CompositePickup router
```
