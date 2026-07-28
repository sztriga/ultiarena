# EV_MULTI — generic weighted-payoff solver

Design doc for the Cython solver extension that backs every multi-payoff
PIMC built on top of the contract oracle.

## Why this exists

The per-contract solvers (`EV_PARTI`, `EV_ULTI`, `EV_BETLI`,
`EV_DURCHMARS`) each maximise a single scalar tied to one contract's
win condition. They can't trade off across components — they don't know
about silent ultis, silent durchmars, declared 40/20 marriages, or
combined bids like "40-100 ulti with silent duri" alive in the same
hand.

Per-combination evaluators (`EV_PARTI_W_SILENT_ULTI`,
`EV_PARTI_W_DURI`, `EV_40_100_W_ULTI`, ...) explode combinatorially as
we add silent / combined variants. We refuse to write them.

**Instead: one generic evaluator parameterised by a weight vector.**
The terminal value is `Σ_c w_c · indicator_c(state)`. The solver does
weighted minimax. Every contract becomes a *recipe* — a `PayoffWeights`
config — not a new C evaluator.

## Components (v1 scope)

Atomic indicators, each a pure function `(GameState) → float` with
static bounds. Fixed in C; combinations only at the weight layer.

| # | name | terminal value | range |
|---|---|---|---|
| 1 | `parti_pts` | `state.scores[sol]` (sol's trick + marriage points) | `[0, 90]` |
| 2 | `silent_ulti_signed` | +1 sol won 7, −2 sol bukott, −1 def won 7, +2 def bukott, 0 else | `[−2, 2]` |
| 3 | `silent_durchmars` | `1{sol took 10 tricks}` | `[0, 1]` |
| 4 | `silent_40` | `1{sol declared trump K+Q}` | `[0, 1]` |
| 5 | `silent_20` | `1{sol declared any non-trump K+Q}` | `[0, 1]` |
| 6 | `score_geq_100` | `1{state.scores[sol] ≥ 100}` | `[0, 1]` |
| 7 | `sol_tricks_zero` | `1{soloist_tricks(state) == 0}` (betli) | `[0, 1]` |
| 8 | `def_40` | `−1{def declared trump K+Q}` | `[−1, 0]` |
| 9 | `def_20` | `−1{def declared any non-trump K+Q}` | `[−1, 0]` |

Component 2 encodes the asymmetric bukott rule in a single signed
indicator: multiplying by the silent-ulti GP rate (2) yields exactly
`{+2, −4, −2, +4, 0}` per defender — the rule milan specified.

## Weights and contract recipes

A `PayoffWeights` C struct holds one `float` per component (default 0).
The oracle layer translates a `BidSet` to a `PayoffWeights`. Examples:

| Contract bid | non-zero weights |
|---|---|
| simple parti | parti_pts=1, silent_ulti=2, silent_durchmars=3, silent_40=2, silent_20=1, def_40=2, def_20=1 |
| bid ulti | as above plus silent_ulti replaced by bid rate=4 (bukott still ×2 via component 2) |
| bid durchmars | silent_durchmars replaced by bid rate=6; parti_pts dropped (binary contract) |
| bid 40-100 | parti_pts=1, silent_ulti=2, has_40 needed → uses score_geq_100 × bid rate gating |
| bid betli | only sol_tricks_zero (with sign flipped if needed) |
| 40-100 ulti combined | union of bid 40-100 and bid ulti weights |

Bid contracts and silent variants share the same indicators — they
differ only in weight magnitudes. The oracle owns the recipe table; the
solver only sees the resulting weight vector.

## Build mechanics

- **PayoffWeights**: C struct, one `float` per component.
- **Setter API**: `set_multi_weights(parti_pts=…, silent_ulti=…, …)`
  sets module-level globals. Matches existing pattern (`set_dm_proven_safe`,
  `_g_ulti_order_id`, etc.). One worker process = one weight set at a
  time; threadsafe per-process.
- **`_term_multi(state)`**: iterates components, multiplies by weight,
  sums. Zero-weight components short-circuit. Inlined per component for
  speed.
- **`_bounds_multi(state)`**: precompute
  `hi = Σ w_c · (hi_c if w_c ≥ 0 else lo_c)`
  `lo = Σ w_c · (lo_c if w_c ≥ 0 else hi_c)`
  once per solve. Re-used at every node.
- **`_early_multi`**: conservative v1 — only at trick 10 (all components
  resolved). Optimisations later (e.g., if only `sol_tricks_zero` is
  active and sol has taken a trick, early LOSE; if only
  `silent_durchmars` and sol has lost a trick, early LOSE).
- **Cull (as built, NOT the v1 plan below)**: `EV_MULTI` does **not** use a
  noop cull — `_get_eval` wires it to **`_cull_parti_blocks`** (the same
  dominance cull as parti/ulti/duri). That cull has an explicit guard: when the
  `silent_ulti` weight is non-zero it tags the **trump-7 with a `-1` points
  sentinel** so the block-splitter always isolates it, for **whoever** holds it
  (soloist OR a defender — both can silent-ulti / bukott). Point cards keep
  their real `_pts`, so reaching-100 moves are never merged either. Net: the
  cull only ever merges provably-interchangeable cards (same suit, adjacent
  strength with no opponent between, same points) and isolates the one card that
  carries silent value, so it is **sound for silent ulti and silent 100**.
  Verified airtight: `set_multi_cull(0)` swaps in `_cull_noop`, and
  `experiments/21_payoff_oracle/test_cull_silent_ulti.py` confirms `solve_root`
  values are identical cull-on vs cull-off across hundreds of 7-holding endgame
  positions (sol- and def-mover), max diff 0. (Original v1 plan was "`_cull_multi`
  noop"; superseded.)
- **`_order_multi`**: default ordering in v1.
- **TT key**: include a 64-bit hash of `PayoffWeights` so cached values
  from different weight configs don't collide. Hash computed once per
  solve, mixed into every probe.

## Python interface

```
from ulti.solvers import pis
weights = PayoffWeights(parti_pts=1.0, silent_ulti=2.0)
values = pis.solve_all(pos, contract='multi', weights=weights)
# values[card] = w_parti_pts * sol_pts(card_PV) + w_silent_ulti * silent_ulti_signed(card_PV)
```

`PayoffWeights` is mirrored as a Python dataclass; conversion to the C
struct happens at the wrapper boundary. The oracle exposes a helper
`bid_to_weights(bid: BidSet, gp: GPTable) -> PayoffWeights` that
implements the recipe table above.

## Scope discipline

- `EV_PARTI`, `EV_ULTI`, `EV_BETLI`, `EV_DURCHMARS` stay as fast paths
  for the fundamental eval. `EV_MULTI` is **additive**. Once it's within
  ~2× their wall-clock per solve, we revisit deprecation.
- v1 ships 9 components, all 9 always in code, but most have weight 0
  for any given experiment. Adding the next experiment = setting another
  weight, not editing C.
- Solver-side optimisations (early term, culling, ordering, predicates
  like úr-vagyok) are intentionally deferred. The first goal is *correct*
  and *general*. Speed is a second pass.

## Unit-testability

Each component is a pure function over the final position. Tests are
crafted positions plus expected scalar — no solver involved. The 9
components × ~5 positions each ≈ 45 micro-tests, all instant.

The terminal sum is tested at one or two combined recipes (e.g., parti
with silent ulti) against hand-computed expected values.

The solver wrapper is tested for: (a) recovers `EV_PARTI` results when
only `parti_pts=1` is active; (b) recovers `EV_BETLI` results when only
`sol_tricks_zero` is active (with sign flip); (c) novel multi-component
case matches a hand-computed multi-payoff oracle on a tiny end-of-game
position.

## Open design questions

1. **Continuous vs threshold for parti**: `parti_pts` is continuous
   (0..90) so alpha-beta has smooth bounds. The oracle threshold at 50
   is applied *after* the solver, at the bidding-time aggregation. This
   matches the exp 11 parti finding: binary vs scalar at play time was
   a wash, but binary at bid time was strictly better. Document this
   split clearly.
2. **`def_40` / `def_20` semantics**: defender marriages reduce the
   soloist's effective parti win margin (they count for def's `scores[]`).
   Whether to give them an explicit indicator (current proposal) or fold
   them into `parti_pts` via the existing `state.scores[]` accounting.
   Likely the latter is simpler — defender marriages already affect
   `state.scores[def]` which already affects who wins `soloist_won_simple`.
   **Decision pending milan's read.**
3. **Bid-vs-silent gating for `silent_ulti_signed`**: when ulti is bid,
   the bid rate replaces the silent rate. But bukott pays double in both
   modes (so the +1/−2/−1/+2 indicator works for both, only the weight
   magnitude changes). Confirmed.

## Estimated work

| piece | LOC | hours |
|---|---|---|
| Cython: struct, setters, 9 components, terminal, bounds, factory | ~200 | 4 |
| Python wrapper + `BidSet → PayoffWeights` translation | ~60 | 1 |
| Unit tests (component-level + integration) | ~150 | 2 |
| Recompile + first end-to-end smoke (parti + silent ulti) | – | 1 |

About a day. The parti × silent ulti A/B experiment runs the next
overnight after that.

## Related

- `scoring_oracle.py` — the Python oracle. Defines `BidSet`, `GPTable`,
  `PayoffVector`. The recipe table (`bid_to_weights`) lives here.
- `README.md` — high-level contract oracle motivation and dependency
  order (oracle → cheap multi-payoff PIMC → correct multi-payoff PIMC
  via this solver).
- `experiments/11_fundamental_eval/results.md` — parti scalar vs binary
  finding that justifies the "continuous in solver, threshold in
  bidding" split.
