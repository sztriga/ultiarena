# Exp 21 — Payoff oracle, the final attack: finishing the 40/20

Goal milan stated: finish the **40/20 handling** in the multi-payoff machinery
so the soloist, while bidding/playing parti, also **prices the silent 40-100 and
silent 20-100** (his mental model: "for each card, also price silent ulti,
silent duri, silent 40-100, silent 20-100").

## Where the work actually was (start-of-exp audit)

Two layers, both partly built before this exp:

1. **Python scoring oracle** `scoring/oracle.py` — DONE for all components,
   including silent_40 / silent_40_100 / silent_20 / silent_20_100
   (`oracle.py:201-225`). Leans on real (non-stub) GameState helpers
   (`soloist_has_40/20`, `last_trick_ulti_check`). No defender-marriage GP
   component exists in the oracle.

2. **Weighted-minimax solver** `trickster/_solver_core.pyx` (`EV_MULTI`,
   `contract='multi'`, `set_multi_weights(...)`). 9 designed components; of those
   **5 live**: `parti_pts` (binary ±1 sol>def), `silent_ulti_signed`,
   `silent_durchmars`, `score_geq_100`, `sol_tricks_zero`. **4 stubbed → return
   0**: `silent_40`, `silent_20`, `def_40`, `def_20`. Stub reason in code: *"no
   marriage state in CState"* (`pyx:1686`). This is the "started on 40/20" milan
   remembered — and where it stopped.

## The reframe (why we are NOT editing the compiled solver)

Studying the marriage mechanics changed the plan:

- `declare_all_marriages` (`game.py:264`) runs at the start of play and adds
  marriage **points directly into `scores`**; defenders **always** declare,
  soloist declares all held pairs by default. `_to_cs` copies `gs.scores` into
  `CState` (`pyx:2213`). So **marriage points are already in the parti channel**,
  for both sides.
- Marriage **holdings are static at solve-root**: declared pre-play, and the
  soloist can't discard into/out of a pair mid-solve (discard happened before
  the solve). A constant `+w` added to every leaf does **not** change the
  argmax. So the literal `silent_40`/`silent_20` holder flags are **value-only,
  policy-irrelevant** — unstubbing them in C buys nothing for play.
- `def_40`/`def_20` are **fully static AND already in `scores`** (defenders
  always declare). A separate GP indicator would double-count; the oracle
  rightly has none. → Resolves design open-question #2: **fold into parti, do
  not add a defender-marriage indicator.**

What *is* policy-relevant is the **100 threshold**, and the solver already has
`score_geq_100` live. Because the holdings are known at root, the **weight
layer** can gate it:

```
score_geq_100 weight = (SILENT_40_100 - SILENT_40)·has_40      # 4-2 = 2
                     + (SILENT_20_100 - SILENT_20)·has_20      # 2-1 = 1
```

i.e. the *marginal* GP of crossing 100 given the marriage you hold. The flat
holder GP (`SILENT_40·has_40 + SILENT_20·has_20`) is a constant added to EV
outside the search. That reproduces the oracle exactly:

```
hold 40, <100 → +2   (silent_40)
hold 40, ≥100 → +4   (silent_40_100)   = 2 (const) + 2·1{≥100}  ✓
```

**Conclusion:** the solver already has every primitive needed. "Finishing the
40/20" = the **weight recipe** (`bid_to_weights`, which the design doc always
placed in the oracle layer), NOT new Cython. This also keeps the shared compiled
`_solver_core` untouched (no recompile, no risk to exps 11-20 that depend on it).

## Build

- `recipe.py` — `sol_marriages(hand, trump) → (has_40, has_20)`,
  `parti_multi_weights(...)` (the gated weight vector), `base_holder_gp(...)`
  (the additive constant). Marginals read off `GPTable` defaults.
- `eval_4020.py` — A/B/C ablation on marriage-holding, parti-feasible deals
  (defenders = god parti, sol = PIMC32, scored by the oracle):
  - **A** vanilla parti PIMC (deployed baseline)
  - **B** multi[`parti_pts=1, silent_ulti=2`] (current multi capability)
  - **C** multi[B + gated `score_geq_100`] (the new 40/20 pricing)
  - **C − B** isolates the 40/20 contribution; **B − A** is the silent-ulti
    sanity check.
  Hypothesis: C reaches 100 more often when holding a marriage → banks the
  silent-40-100/20-100 bonus the oracle already credits → higher GP/def.

## Status / next
- [ ] Run eval_4020 small (smoke) → scale.
- [ ] If C beats B, upstream `bid_to_weights` into `scoring/oracle.py` (design's
      intended home) and wire into the bidding evaluator.
- Open: defer to milan whether silent-40-100 should be a *bid* path too (it can
  be priced the same way with the bid rates 8/4 instead of the silent 4/2).
</content>
