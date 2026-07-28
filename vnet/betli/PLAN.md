# Experiment 4 — Roadmap from current V-head to a pure MCTS defender

This file lays out the concrete step-by-step path from the current state
(a supervised V-head trained on god-solver labels) to the end goal: a
defender agent that uses MCTS with a learned value (and eventually
policy) head and is evaluated head-to-head against the god-soloist on
the stoppable-deal set.

Each step is gated on a clear eval; if a step fails, the next step's
expected return changes and we may pivot.

---

## Where we are now

- **V-head**: `ValueNet(132→256→128→1, tanh)`, 67k params, trained on
  200k deals with α~U(0.3,1.0), t∈[0,8], K=1 god-solver labels.
- **Opening-hand ROC eval**: V_avg beats Defender PIMC(N=32) by
  +0.05 AUC at α=0.5 and +0.07 AUC at α=0.7 (see `roc_mix_*_at_a05.png`,
  `roc_mix_*_at_a07.png`).
- **Substrate that already exists in the repo:**
  - `solvers/pis.py` — god-solver on partial states
  - `solvers/determinize.py` — `build_info_set` + `sample_world`
    (void-aware, contract-aware, supports α-biased posterior)
  - `solvers/pimc.py` — perfect-info-MC defender baseline (god-solver
    as evaluator)
  - `trickster/src/trickster/mcts.py` (mirrored at
    `oldtawer/trickster/mcts.py`) — AlphaZero-style MCTS with PUCT,
    determinization, lazy expansion, dual-head NN interface, batched
    eval, Dirichlet root noise. Game-agnostic via a `GameInterface`.

So no MCTS code needs writing — just glue.

---

## Step 1 — PIMC + V (1-ply value-only lookahead)

### What
Write `solvers/pimc_v.py` mirroring `solvers/pimc.py`, but swapping
god-solver evaluation for the trained `ValueNet`. For each legal action
`a` at the defender's turn:

  1. Sample K worlds consistent with the info-set
     (`sample_world(info_set, rng)`).
  2. In each world, apply `a` to get next state `s'`.
  3. Score `a` ← mean over K worlds of `V(features(s'))`.
  4. Return `argmax_a score(a)`.

No tree, no recursion, no policy head. Just K determinizations × per-
action V evaluations.

### Why this first
Cheapest possible test of "does V carry usable action-selection
signal?". Reuses our existing V verbatim, reuses the determinizer
verbatim, doesn't require a `GameInterface` adapter or MCTS at all.

### Eval
Head-to-head play, 500 deals at α ∈ {0.5, 0.7}, against god-soloist:
- Random defender (lower bound)
- PIMC defender — `solvers/pimc.py` as-is
- **PIMC+V defender — the new agent**

Metric: defender win-rate **on the stoppable-deal set** (deals where god-
defenders would win). 1.0 = perfect; the gap to 1.0 is the missed-stop
rate, our true end-goal failure mode.

### Decision gate
| Outcome | Read |
|---|---|
| PIMC+V > PIMC | V's generalization across info-sets helps action selection. Strong green light for MCTS. |
| PIMC+V ≈ PIMC | V matches Monte Carlo per state — about as expected. Yellow light; MCTS still worth building for the lookahead advantage. |
| PIMC+V < PIMC | V is *worse* than per-state Monte Carlo at action selection. Red light — pivot back to improving V (state distribution, training data) before any MCTS investment. |

Estimated cost: ~half a day to wire, ~30 min to eval.

---

## Step 2 — Betli `GameInterface` for the existing MCTS

### What
Write `solvers/betli_game.py` (or similar): an adapter implementing the
`GameInterface` protocol expected by `trickster.mcts`. Methods needed:

- `legal_actions(state) -> List[Card]`
- `apply_action(state, action) -> state` (mutating or returning new)
- `is_terminal(state) -> bool`
- `winner(state) -> int` or team-relative value
- `determinize(state, player, rng) -> state` — wraps
  `solvers.determinize.sample_world` so MCTS can sample worlds
  consistent with the current player's info-set
- `encode_state(state, player) -> np.ndarray` — wraps our
  `features.extract_features` for the V-head input
- `current_player(state)`, `same_team(p1, p2)` etc.

Each method is a thin wrapper over things in `solvers/pis.py` and
`solvers/determinize.py`. ~150 lines total.

### Why
Unblocks reuse of the existing MCTS verbatim. No MCTS rewrite.

### Eval
Sanity tests:
- Game plays through to terminal under MCTS-driven choices.
- Determinized states are consistent with the info-set (no card visible
  to two players, hand sizes correct, voids respected).
- Encoded features match `features.extract_features` (regression test
  against a few hand-crafted positions).

No win-rate eval at this step; this is just the adapter.

Estimated cost: ~1 day.

---

## Step 3 — MCTS + V (no policy head)

### What
Plug the V-head into `trickster.mcts.alpha_mcts_choose` via a value-only
adapter:
- The MCTS expects `net.predict_both(features, mask) -> (value, policy)`.
- Value = our V's output.
- Policy = uniform-over-legal (mask-then-normalize). MCTS will explore
  with PUCT using this uniform prior, which is identical to standard UCB1
  exploration when `c_puct` is set appropriately.

### Why
Tests the lookahead advantage in isolation. Same V as Step 1, but now
with multi-ply tree search instead of 1-ply.

### Eval
Same head-to-head harness as Step 1. Add MCTS+V as a fourth agent:

- Random / PIMC / PIMC+V / **MCTS+V**

Metric: defender win-rate on stoppable deals.

### Decision gate
| Outcome | Read |
|---|---|
| MCTS+V > PIMC+V | Lookahead is contributing real value. Continue to Step 4. |
| MCTS+V ≈ PIMC+V | Tree search isn't helping much — either V's leaves are dominating, or the search budget is too small. Tune budget; otherwise yellow-light Step 4. |
| MCTS+V < PIMC+V | Something's wrong with the MCTS wiring. Debug before Step 4. |

Estimated cost: ~1 day (mostly hooking V into the existing MCTS).

---

## Step 4 — Self-play data collection + V retraining

### What
Run MCTS+V games and collect a new training dataset from the states
visited:

```
for each self-play game:
    play with MCTS+V on defender side vs god-soloist
    record every defender info-set state encountered
    label each state with v* = god_solver(actual_world_state)
        (god-solver is cheap, so use it instead of game outcome)
```

Retrain V on the union of old data + new self-play data (or, simpler,
just on new data, treating the old V as a warm start).

### Why
The current V was trained on states from *random play*. MCTS visits a
different state distribution. Retraining on the MCTS-visited
distribution refines V where it actually matters.

### Eval
- Opening-hand ROC AUC at α=0.5/0.7: should hold or improve (sanity).
- Head-to-head defender win-rate on stoppable deals: should improve
  meaningfully. This is the metric that justifies the loop.

### Decision gate
| Outcome | Read |
|---|---|
| Win-rate improves measurably | The loop is working. Iterate Step 4 (multiple rounds). |
| Win-rate plateaus after 1-2 rounds | V has saturated on the realistic state distribution. Step 5 (policy head). |
| Win-rate drops | Distribution shift damaged V. Investigate (mix old + new data, lower learning rate). |

Estimated cost: ~1 day per iteration (mostly self-play wall time).

---

## Step 5 — Add a policy head

### What
Extend V into a dual-head net: same body, two output heads
- value (existing, tanh)
- policy (softmax over the 32-card action space, masked to legal moves)

Train policy head on **MCTS visit-count distributions** collected during
self-play. (Visit counts already get recorded by the existing MCTS;
this is the AlphaZero recipe.)

### Why
A learned policy lets MCTS spend its budget on promising actions
instead of exploring uniformly. The expected gain is bigger search
efficiency at the same number of rollouts — i.e. either better play at
the same compute, or same play at lower compute.

Add this only if Step 3/4 showed MCTS *was* helping (otherwise the
policy head doesn't matter — we're not search-bound).

### Eval
Same head-to-head harness:
- MCTS+V (uniform prior) vs **MCTS+V+π (learned prior)**
- At the same per-move budget (e.g. 256 simulations)

Look for either:
- Win-rate improvement at fixed budget, OR
- Same win-rate at reduced budget (search efficiency win)

Estimated cost: ~1 day for the network change + retraining, plus eval.

---

## Step 6 — Iterated self-play loop

### What
Once Step 4 (and optionally Step 5) is working, automate the loop:

```
V_0, π_0 = current best net
for iteration k = 1, 2, ...:
    play N games with MCTS using (V_k, π_k); collect (state, v*, visits)
    retrain net on accumulated data → V_{k+1}, π_{k+1}
    eval head-to-head; stop when win-rate plateaus
```

### Eval
Track defender win-rate per iteration. Standard convergence diagnostics.

Estimated cost: open-ended; depends on iteration count needed.

---

## Summary table

| Step | Builds | Reuses | Eval | Decides |
|---|---|---|---|---|
| 1 — PIMC+V | 1-ply value-only agent | det, V | Win-rate vs PIMC | Whether V is action-useful |
| 2 — GameInterface | adapter | mcts.py | Sanity tests | (unblock 3) |
| 3 — MCTS+V | MCTS wired to V | mcts.py, step 1's harness | Win-rate vs PIMC+V | Whether lookahead helps |
| 4 — Self-play + retrain | data collection loop | step 3's harness | Win-rate progression | Whether loop converges |
| 5 — Policy head | dual-head net | step 4's data | Win-rate at fixed budget | Whether π improves efficiency |
| 6 — Iterated loop | automation | all above | Win-rate per iter | When to stop |

Each step's win-rate eval reuses the same head-to-head harness; that
harness is the single artifact that defines "are we done?" at every
stage.
