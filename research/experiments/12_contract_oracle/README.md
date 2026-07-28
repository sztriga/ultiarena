# Experiment 12 — Contract oracle (design)

Design doc for the bidding-phase machinery. No code yet — this captures
the agreed mental model so subsequent files (`scoring_oracle.py`,
`holding_probs.py`, multi-payoff PIMC) are built against a fixed spec.

## The idea (verbatim, milan)

> I want to start building the bidding phase contract by contract. But
> we're not going to do *only* fundamental contracts — we'll add the
> silent games and combined games too. For 40-100 / 20-100 we do a
> mathematical estimation of the chances that the opponent has 20s or
> 40s — that should be taken into account when bidding. Also: if I want
> to play a parti, I don't want to only use the PIMC parti path. For
> each card I want to also consider the silent contracts. So for each
> card I'd price ulti and duri as well (if applicable) and compute an
> expected payoff based on those. Same goes for 100 — if a path gets me
> to a 100 (with a 20 or a 40), it should be added to the expected
> payoff as a silent contract.

## Mental model

Every play has a **vector** of possible payoffs, not just the bid one.
The bid decision is the expectation, under the dealer's prior, of the
soloist's best line *through that vector*.

Three building blocks, in increasing order of dependency:

### 1. Scoring oracle  (gateway — build first)

Pure function:

```
score(final_position, trump, bid_set) -> payoff_vector
```

Returns *every* component the Ulti rules would award for the played-out
deal, regardless of what was bid:

- parti trick points (sol vs def, including talon-credit to def)
- silent ulti bonus (sol took trick 10 with trump-7, not announced)
- silent durchmars bonus (sol took all 10 tricks, not announced)
- 40-100 thresholds (sol declared 40 in trump + sol scored ≥ 100)
- 20-100 thresholds (sol declared a 20 + sol scored ≥ 100)
- silent 40 / silent 20 (declared marriages, no 100 bid)
- kontra / rekontra multipliers
- piros multipliers

No solver changes. Pure post-hoc accounting from the final `GameState`.
This is the spec everything else assumes.

### 2. Multi-payoff PIMC

Two flavours, in increasing order of complexity:

- **Cheap** — god-solver optimises a single scalar (the bid contract's
  value). After each sample we *simulate forward with the PV* (god's
  chosen line) and apply the scoring oracle to the final state. Silent
  bonuses appear for free when the parti-optimal line happens to hit
  them. Limitation: the soloist never *deviates* from parti-optimal to
  grab a silent ulti.
- **Correct** — solver evaluates the weighted sum directly:
  `α·parti + β·P(silent_ulti) + γ·P(silent_duri) + …`. Requires
  extending the Cython evaluator to carry a payoff vector through the
  search. Picks up the "give up 5 parti points to lock in a silent
  durchmars" trades.

Start with **cheap**. Quantify the gap from **correct** before paying
the Cython cost.

### 3. Holding probabilities (40/20s)

Pure combinatorics over the 22 unseen cards given the soloist's hand:

- `P(K+upper of trump suit in same opponent hand)` → 40 holding prior
- `P(K+upper of non-trump suit S in same opponent hand)` per side suit
  → 20 holding prior

These plug into the bidding-time expected-payoff calculation:
the soloist's *own* 40/20 contributions are known exactly; the
*opponent's* contributions are weighted by these priors.

Cheap to compute (closed-form hypergeometric). No PIMC needed.

## Bidding policy

For each legal bid `b`:

1. Run multi-payoff PIMC from `t=0` with `bid_set = {b}` plus all
   silent contracts that survive `b` (e.g., bidding parti keeps silent
   ulti / silent duri / silent 40-100 alive).
2. Sum across the payoff vector, weighted by GP per component.
3. Subtract the opponent-40/20 expected loss from holding probabilities.

Pick `argmax_b` total expected payoff.

Silent contracts and combined bids fall out as *additional dimensions
of the payoff vector* — same machinery handles them.

## Why this matters for parti specifically

From exp 11 (cross-contract eval): PIMC-as-sol on parti tops out at
~80% of god, while on duri / ulti / betli it's ≥90%. The gap is
strategy fusion — parti is the only contract where path-dependent,
multi-trick choreography matters. Adding silent-contract awareness to
the play-time search (multi-payoff PIMC) gives the soloist new reasons
to take specific lines, which should narrow the gap *and* recover GP
the current PIMC-parti throws away whenever the optimal scalar line
walks past a silent ulti.

## First concrete step

**Build the scoring oracle.** Unit-test it against hand-crafted
deals from the rulebook. Once it's locked, every subsequent piece is
either a wrapper around it (cheap multi-payoff PIMC) or a substrate
beneath it (holding probabilities). Everything composes against this
one interface.

## Out of scope (for now)

- Kontra strategy (when to kontra, when to accept) — handled at a
  separate decision layer; for v1 assume defenders never kontra.
- The actual bidding *auction* (passing order, must-overbid rules) —
  this experiment is about the **valuation** of each legal bid. The
  auction layer composes the per-bid values into a sequential decision.
- Learned value heads — exp 11 already showed PIMC32 t=0 is good
  enough as a substitute oracle. Distillation is purely a latency play
  and should come later.

## Related

- `experiments/11_fundamental_eval` — per-contract sol_hold / def_stop /
  AUC numbers that motivate the parti gap analysis.
- `eval/pimc_matchup.py` — the substrate `play_one` will call into for
  multi-payoff PIMC playouts.
- `trickster.games.ulti.game.soloist_won_simple` — current canonical
  parti scoring (50-pt threshold, talon credit to def). The scoring
  oracle generalises this to a vector.
