# Experiment 4 — Defender NN feasibility

> **Status (2026-05-30)**: eval artifacts retired. The PIMC-defender
> baseline used in the original comparison was anti-optimal (missing
> argmin flip for defender seats); corrected vanilla PIMC matches or
> beats the best NN at every α — see
> [`experiments/10_betli_pimc_audit/`](../10_betli_pimc_audit/).
>
> The **training scaffolding is kept** for a future retrain against the
> corrected baseline: `data_gen.py`, `features.py`, `net.py`,
> `net_nnue.py`, `train.py`. The broken eval scripts
> (`play_matchup.py`, `alpha_sweep.py`, `eval_roc.py`) and their ROC
> snapshots + `CONCLUSIONS.md` have been removed.
>
> `load_net_from_ckpt` (used at runtime by `apps/api/betli_hu.py`) now
> lives in `net.py`.



## Question
Can we replace the defender's PIMC averaging step with a learned function
to break the AUC ≈ 0.64 ceiling we hit in exp 3? Specifically: train an
MCTS-style NN agent for the *defender* side, with the **god solver as a
fixed soloist opponent**, and see whether the network can learn the
strategy-fusion-adjusted scoring that PIMC structurally cannot.

This file is just the feasibility discussion. No code, no data yet.

## Why it's appealing

- **Bottleneck identified.** Exp 3 showed defender AUC plateaus at ≈0.64
  at α=0.7 regardless of N. The problem isn't sample budget, it's that
  uniform averaging across determinizations doesn't model strategy fusion.
  A learned scoring function is exactly the right kind of replacement.
- **Clean teacher.** God-soloist removes opponent drift — there's no
  policy oscillation to manage during training.
- **Clean labels.** Game outcome z ∈ {±1} from the team's POV. No reward
  shaping.
- **Most of the substrate exists.** We already have the deal generator,
  god solver, determinizer, info-set builder. The Morris MCTS+NN work
  earlier this month proved we can stand up the pipeline.

## Why it's harder than Morris (be honest about scope)

### 1. It's IS-MCTS, not MCTS
Morris is perfect-information — single state, one game tree, standard
PUCT. Betli from the defender's POV is an *information set* of possible
worlds (unknown hands, possibly hidden talon).

Two standard substitutes:
- **IS-MCTS** (Cowling, Powley, Whitehouse 2012): tree over information
  sets; sample a determinization at each rollout. Visit counts at the
  root are still a valid AlphaZero-style policy target.
- **Determinized-MCTS / PIMC++**: what we have now, but with NN scoring
  at leaves instead of uniform sampling.

IS-MCTS is the canonical AlphaZero-style answer. It's roughly 1.5× the
size of Morris MCTS plus the determinizer integration.

### 2. Two cooperative defenders with private info
Both defenders play independently with their own info-set, on the same
team, with no explicit communication. They must:
- **signal through play** (e.g., leading a void suit, ducking specific
  cards), and
- **model each other's policy**.

Standard moves:
- One shared defender net; each defender runs it independently with its
  own features. Outcome signal is the team result. This is the
  Hanabi pattern — it works but learning conventions through self-play
  is slow.
- Or condition on `is_def1 / is_def2` via a one-hot in features so the
  same net can specialize a bit.

The cooperative-asymmetric-info aspect is the genuinely hard research
piece. AlphaZero-style training won't automatically produce smart
signaling. Hanabi-level agents took years of dedicated work on
coordination-aware architectures. For betli we might get lucky because
the action space is small and suit-following forces a lot of moves —
but no guarantees.

### 3. The training distribution problem
At α=0.7, only ~38% of deals are god-defender-winnable. On the other
~62%, god-soloist will win regardless of defender play — every defender
action gets z = -1. The net learns "all moves lose" on those deals,
which is technically correct but uninformative for learning defensive
priorities.

Useful counter:
- Value head still learns correctly on hopeless deals ("this position
  is lost").
- Policy head visit counts still encode "least-bad-move", which is a
  meaningful gradient.

Net effect: effective training-signal-per-game is smaller than in Morris.
Going to lower α (where defender wins are more common) would help, but
shifts the distribution we evaluate against.

### 4. Feature encoding is real work
Relevant features for betli:
- Defender's own hand (32 one-hots)
- Cards played per player per suit (history)
- Voids revealed (per player per suit, 12 bits)
- Current trick state (cards on table, who led, suit constraint)
- Plies into the game / cards remaining

If `game/obs.py` (the canonical 264-dim obs) already covers most of
this, we should reuse it. Otherwise add a betli-specific view. Either
way it's about a session of work to nail down + test.

### 5. Compute is non-trivial
Each MCTS-NN game in Morris cost ~1-2 s. A betli game with IS-MCTS
(determinization at every sim) and the solver in the inner loop will be
5-10× slower per game. Manageable, but a 20-iter / 32-game self-play
run that took 1 minute in Morris will take 5-10 minutes for betli.
Scaling sweeps cost real time.

## Recommended incremental plan

The full IS-MCTS-NN is the right destination, but it's 4-6 focused
sessions of work and any one bug delays everything. Smaller first step:

**Phase 1 — Value net only (2 sessions).**
Replace PIMC's "average over uniformly-sampled determinizations" with a
single neural function:
```
defender info-set features  →  P(soloist wins)
```
Train supervised on `(features, god verdict)` pairs sampled across deals
and play states. No IS-MCTS, no policy head, no self-play loop.

If this lifts AUC from 0.64 to even 0.75, it's a strong win and tells us
strategy fusion can be substantially fixed with the right scoring
function. If it can't even reach 0.75, IS-MCTS on top won't save us —
phase 2 isn't worth starting.

**Phase 2 — Add policy head + IS-MCTS (3-4 sessions).**
Now we have a tested value head. Add a policy head trained against
IS-MCTS visit counts. This is the AlphaZero loop proper. Includes:
self-play game generation against the god soloist, IS-MCTS over
defender info-sets, training loop, gating eval.

**Phase 3 — Co-adaptation.**
The recipe sketched in an earlier session — soloist threshold + defender
α — but now with a real defender policy. Soloist starts bidding
selectively based on its expected return against this defender. This is
where it becomes a research result rather than a system-building
exercise.

## Bottom line

- **Feasibility: yes.**
- **Phase 1 should happen first.** Single supervised value net, ~2
  sessions, gives a clean go/no-go signal for whether the bigger
  investment is worth it.
- **Phase 2 is the AlphaZero analog.** Real work, real reward if phase 1
  shows headroom.
- **Don't expect AlphaZero magic for the coordination problem.** Betli's
  cooperative-with-private-info structure is hard; the small action
  space and forced suit-following may save us, but no guarantees.

## Open design questions to resolve before phase 1

1. **Which α(s) for training data?** α=0.7 matches exp 3 baseline, but
   only 38% of deals are defender-winnable. Mixing α ∈ {0.3, 0.5, 0.7}
   might give a more balanced label distribution.
2. **What play states to sample?** Just openings (matches exp 3 part A),
   or interior states from god-played trajectories (matches exp 3
   part B)? The latter would test the *whole game* not just the
   bidding-equivalent.
3. **Feature design starting point.** Reuse `obs.py` or define a
   betli-specific view? Reuse is faster but may carry irrelevant fields.
4. **Symmetry.** Betli has no obvious board symmetry like Morris's 16,
   but the two defender positions are symmetric — train one net, share
   it, condition on `which_defender`.

## Files
- `README.md` — this writeup.
- (no code yet; populated when phase 1 starts)
