# Exp 21 results — one engine + corrected silent-100

## Oracle audit (milan's rule, 2026-06-14) — FIXED + tested

`scoring/oracle.py` silent-100 had two bugs: values swapped (had 40-100=4 /
20-100=2) and a TOTAL-points threshold. Corrected to: silent 100 = ONE marriage
+ CARD points (marriages excluded) ≥ 100; **40-100** needs card≥60 worth **2**,
**20-100** needs card≥80 worth **4** (piros doubles); naked marriage = **0**;
both qualify → score the better. `test_silent_100.py`: **13/13 pass**, incl.
`40+20+20+20 = 100-in-marriages → NOT a silent 100`.

## One-engine eval (`eval_engine.py`, N=226 marriage deals, PIMC32 vs god-parti)

A contract = a weight config on the SAME multi solver + SAME oracle.

| row | parti WR | reached 100 | 40-100 | 20-100 | GP/def |
|---|---|---|---|---|---|
| A0 `contract='parti'` (dedicated) | 0.580 | 0.208 | 0.115 | 0.075 | +0.912 |
| A1 `multi[parti_pts=1]`           | 0.571 | 0.226 | 0.111 | 0.093 | +0.987 |
| B `multi[parti+ulti]`             | 0.571 | 0.283 | 0.168 | 0.111 | +2.137 |
| C `multi[parti+ulti+100]`         | 0.562 | 0.350 | 0.186 | 0.137 | +2.235 |

**Effects (corrected oracle):**
- **C − B (silent-100): +0.097, t=1.25** — mechanism intact (reached-100 +6.7pp,
  40-100 +1.8pp, 20-100 +2.6pp) but GP small and now *not* significant. Smaller
  than the first (wrong-oracle) run because the corrected 40-100 is only worth 2
  (was 4) and the per-marriage threshold qualifies fewer deals.
- **B − A1 (silent-ulti): +1.150, t=9.33** — the big, robust multi win (exp 12).

## Invariant: `multi[parti]` vs dedicated `parti` — NOT byte-exact

**23/226 deals diverge, mean GP +0.075** (A1 ≈ A0, within noise). Cause: both
optimise the SAME binary parti objective, but the dedicated `EV_PARTI` has
parti-specific culls + ordering (`_cull_parti_blocks`, `_order_parti_pts_first`)
that the generic `EV_MULTI` lacks. With a binary objective many lines tie (parti
already won), so the two break ties to different — equally parti-optimal — lines,
which incidentally bank different silent bonuses (note A1's slightly higher
reached-100 / 20-100). It is NOT a skill gap; the parti win/loss is the same on
those deals. **To make it byte-identical, port the parti culls/ordering into the
multi evaluator** (Cython) — that would also confirm the cause (should drop to
0/N).

## Flipped execution: GOD soloist vs PIMC32 defenders (`eval_godsol.py`, N=226)

Perfect-play ceiling of the silent-capture lever (no PIMC noise on the soloist).

| row | parti WR | silent-ulti | reached 100 | 40-100 | 20-100 | GP/def |
|---|---|---|---|---|---|---|
| A god[parti]     | 0.650 | 0.035 | 0.195 | 0.084 | 0.093 | +1.066 |
| B god+ulti       | 0.668 | 0.500 | 0.279 | 0.142 | 0.115 | +2.221 |
| C god+ulti+100   | 0.668 | 0.504 | 0.319 | 0.155 | 0.128 | +2.310 |

- **B − A (silent-ulti): +1.155, t=11.4.** Bare-parti perfect play captures
  silent ulti only **3.5%**; with the objective, **50%**. The points-optimal line
  and the silent-ulti line genuinely diverge — the objective bridges them. Same
  size as the PIMC-soloist run → structural, not an execution artifact.
- **C − B (silent-100): +0.088, t=2.26 (significant).** No soloist PIMC noise
  tightens it (was t=1.25 with the PIMC soloist; same size). reached-100 +4pp.
- **Zero parti cost** — god WR ticked UP (0.650→0.668): a perfect player only
  spends the don't-care freedom, unlike the PIMC soloist that over-chased 100.

## Defender side of silent ulti (`eval_defender_ulti.py`, N=1457)

Deals where a DEFENDER holds the trump-7; fixed god-parti soloist; defenders flip
objective (god-vs-god → deterministic, A/B diff is purely the defender objective).

| | sol parti WR | DEF silent-ulti | DEF bukott | 7 in trick 10 | sol GP/def |
|---|---|---|---|---|---|
| A def god[parti]             | 0.686 | 0.002 | 0.001 | 0.003 | +0.880 |
| B def god[parti+silent_ulti] | 0.686 | 0.100 | 0.000 | 0.100 | +0.681 |

**B − A = −0.199 GP/def, t=−12.6 (defenders gained).** Both behaviours confirmed:
- **Go for it:** aware defenders win trick 10 with the 7 on 10% of deals vs 0.2%.
- **Dump the 7 early to avoid losing one:** in B, silent-ulti% == 7-in-trick10%
  and bukott = 0 → the defender keeps the 7 to the last trick ONLY when it will
  win it, and discards it early otherwise (never falls).
- Zero parti-defence cost (WR 0.686 unchanged).

**Symmetric picture:** silent ulti is a lever for BOTH seats — soloist captures
50% (ulti-biased, holds 7 + trump strength), defender 10% (parti deal, lone 7
rarely beats the soloist's trumps). Pure-parti play leaves both gaps open.

## Cull soundness for silent ulti — AIRTIGHT (milan's worry, resolved)

EV_MULTI uses `_cull_parti_blocks` (NOT a noop — the design doc was stale). It
guards the trump-7 with a `-1` points sentinel whenever `silent_ulti` weight is
non-zero, isolating it for *whoever* holds it (sol OR def); only
provably-interchangeable cards (same suit, adjacent strength, no opponent
between, same points) ever merge. Added a runtime toggle `set_multi_cull(0|1)`
(default 1, no behaviour change) + recompiled.

Two independent verifications:
1. **Value-equivalence** (`test_cull_silent_ulti.py`): solve_root values cull-on
   vs cull-off, silent_ulti active, across **463** endgame 7-holding positions
   (**266 sol-mover + 197 def-mover**) → **0 mismatches, max |diff| 0.0**.
2. **Whole-game deterministic** (`eval_defender_ulti.py` CULL=1 vs CULL=0,
   N=575): every output byte-identical (parti WR 0.677; DEF silent-ulti
   0.002→0.103; sol GP A +0.892 / B +0.690) — only wall time differs (4s vs 12s).

**Conclusion:** the cull is a pure ~3× speed optimization with ZERO effect on the
multi value — for soloist AND defender silent ulti. Doc fixed in
`experiments/12_contract_oracle/design_ev_multi.md`. NB found+fixed a test bug
along the way: deal cards are `ulti.card.Card` (rank `'7'` string), not trickster
`Rank.SEVEN` — the parti filter must use the string.

## Silent durchmars (colored duri) — REDUNDANT for the soloist

Added `silent_dm` to `engine.solver_weights` (weight = `durchmars_silent` = 3,
piros doubles; milan-confirmed 2026-06-15) and a row D to `eval_godsol.py`. Also
switched the god-soloist eval to **god-parti defenders** (DEF=god, default) —
deterministic, ~10× faster than PIMC32 (2000 deals in ~17s).

god soloist vs god-parti defenders, N=775 marriage deals:
| row | silent-ulti | reached100 | DURI | GP/def |
|---|---|---|---|---|
| A god[parti]   | 0.031 | 0.209 | 0.066 | +1.169 |
| B +ulti        | 0.581 | 0.305 | 0.056 | +2.421 |
| C +100         | 0.569 | 0.351 | 0.054 | +2.541 |
| D +duri        | 0.569 | 0.351 | 0.054 | +2.541 |

B−A (silent-ulti) **+1.252 t=23.6**; C−B (silent-100) **+0.121 t=4.9**; **D−C
(silent-duri) +0.000, t=0.00 — exactly identical to C.** Weight plumbing verified
(get_multi_weights shows silent_durchmars=3.0), so it's genuine redundancy: a
sweep is the supremum of every points-based objective (all 10 tricks ⟹ all 90
points ⟹ parti + 100; win trick 10 with the 7 ⟹ silent ulti), so a
parti+ulti+100 soloist already plays it whenever achievable. Even **bare parti
(A) captures sweeps at 0.066** (> C's 0.054; the gap is tie-break noise). The
sweep rate is hand-dominance, not objective. **Caveat:** vs GOD defenders a
sweep is forced-or-impossible; vs PIMC defenders a duri weight *might* help by
exploiting a defender mistake into a full sweep (untested).

## Defender side of durchmars — NOT redundant (breaks soft sweeps)

`eval_defender_duri.py`: durchmars-colored deals, fixed god-parti soloist (EV_PARTI,
so it doesn't share the multi global), defenders toggle silent_durchmars OFF (A,
god-parti) vs ON (B, god-multi[parti+silent_durchmars], minimising → grab a trick
to deny the sweep). god-vs-god, N=1500:

| | sweeps allowed | sol GP/def |
|---|---|---|
| A parti defenders (let-slide) | 339 (22.6%) | +2.743 |
| B duri-aware defenders        | 169 (11.3%) | +2.337 |

- Forced sweeps (unbreakable, both): **169**. A-allowed-but-B-broke: **170**
  (= exactly HALF of A's sweeps). **B−A = −0.407 GP/def, t=−13.2** (defenders gain).
- Example seeds (B stops a duri A allows): 320000004, 320000008, 320000010, …

**Asymmetry with the soloist side:** soloist duri is redundant (a sweep maximizes
parti, so it's pursued for free), but DEFENDER duri is NOT — breaking a sweep
usually means taking a WORTHLESS (0-point) trick that doesn't help the defender's
parti, so a parti-only defender is indifferent ("lets it slide") and only the
duri objective motivates the grab. Confirms milan's intuition.

## Bid 40-100 / 20-100 — gap CLOSED (`eval_bid_100.py`)

Wired into the engine: `solver_weights(bid='40_100'|'20_100')` sets score_geq_100
to the made↔bukott SWING (40-100: +4/-4 → 8; 20-100: +8/-8 → 16; the -bid floor
is a sunk constant; policy scale-invariant under piros). `oracle_bid(bid=...)`
declares it so the oracle scores ±4/±8. Validated on single-declared-marriage
ulti-biased deals (so the solver's total>=100 threshold is EXACT), god-multi sol
(A parti-only vs B price-the-bid) vs god-parti defenders (which resist the 100
for free by grabbing points):

| bid | A made | B made | B−A GP/def | parti WR |
|---|---|---|---|---|
| 40-100 | 0.322 | **0.651** | +2.825 (t=15.0) | 0.862 (both) |
| 20-100 | 0.112 | **0.156** | +0.691 (t=7.2)  | 0.531 (both) |

Pricing the declared bid roughly DOUBLES the 40-100 make-rate (32%→65%) and turns
GP −0.31 (mostly bukott) into +2.51 — zero parti cost (crossing 100 ⟹ parti won).
20-100 is intrinsically hard (needs 80 of 90 card points) so it mostly bukotts on
random has-20 hands (GP stays negative); the recipe still helps (+0.69, t=7.2) —
the contract is just one you bid selectively.

**Remaining edge (not the common case):** a bid 40-100 while ALSO holding a 20
(multi-marriage) makes the solver's fixed total>=100 threshold fire too early
(card_pts>=40 instead of 60). Filtered out here. Fixable with a configurable
score threshold (small Cython, like set_multi_cull) = 100 + other-declared
marriages — unbuilt.

## Verdict
- Oracle = correct + tested; it's the single source of truth.
- One engine works: parti/+ulti/+silent-100 are weight configs on one solver.
- Configurability invariant holds in *objective* (optimal parti) but not
  byte-exact GP (±0.075 tie-break noise) until the parti culls are ported.
- Silent-100 is a small, free correctness win (mechanism proven). The bigger
  lever is the **bid** 40-100/20-100 (corrected: +4/−4 and +8/−8 → score_geq_100
  weight 8 / 16, 4× the silent incentive) — exp-22.
</content>
