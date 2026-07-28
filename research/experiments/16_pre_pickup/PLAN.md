# Exp 16 — Pre-pickup bidding oracle

**Started:** 2026-06-06

The pre-pickup question: "Given my 10-card hand and nothing else, should
I bid contract X?" Picks must commit before seeing the talon.

This experiment builds the inference layer above exp 15 (which evaluated
post-pickup 10-card hands assuming a known talon).

## Mental model

From the bidder's POV pre-pickup:
- Talon = 2 unseen cards drawn uniformly from the 22 unseen → 231 combos.
- For each (talon, contract, trump): exp 15's v2 net gives
  `max_discard P(make X)` in ~0.4 ms.
- Pre-pickup oracle = aggregate over the 231 talons.

## Step-by-step plan

Each step is a small isolated artifact we eval before moving on.

### Step 1 — Pre-pickup oracle (enumerated)

Build `pre_pickup_oracle.py`:
- Input: 10-card hand
- Loop over 231 talon combos
- For each talon, call exp 15 v2 net → `max_discard P(make X)`
- Aggregate to: `E_talon[max_discard P(make X)]` per contract
- Wall budget: ~100 ms per (hand, contract) — enumerate live.

**Sanity check:** on a held-out set of (hand, "true" talon) pairs,
verify oracle's pre-pickup P estimate matches the post-pickup
distribution averaged the same way.

### Step 2 — Pickup-vs-pass measurement (N=3000)

The clean isolation of oracle quality. Setup mirrors exp 14 minigame:
deal 12-10-10, defenders are god. The soloist nominee has two
"identities":

| picker         | sees       | uses                          |
|---             |---         |---                            |
| post-pickup    | 12 cards   | exp 15 v2 + thresholds        |
| **pre-pickup** | 10 cards   | oracle, threshold from exp 15 |

Procedure per deal:
1. Run post-pickup picker on 12 cards → records GP_post (baseline).
2. Hide the talon. Run pre-pickup oracle on 10 cards → pickup yes/no.
   - If yes: reveal talon, run exp 15 to choose contract, play out
     → GP_pre.
   - If no: GP_pre = 0.

Compare GP_post vs GP_pre at N=3000. The interesting deals are where
the two pickers disagree — those isolate pre-pickup oracle errors.

### Step 3 — Add overtake decision (PIMC-defender threshold)

Once step 2 lands, add a second player B (10 cards) who must decide
whether to overtake A's bid.

B's decision rule:
```
E[GP_defender | B passes, A plays X] = -PIMC(B's hand, A's contract)
E[GP_soloist  | B overtakes, B plays X'] = oracle(B's hand, X')
overtake iff E_solo > E_def
```

This makes the one-step auction measurable. Eval on N=3000:
who plays, what they bid, total GP per player.

### Step 4 — Distill oracle into pre-pickup net

If steps 2-3 show the oracle is useful, train a small NN:
- Input: 32-dim 10-card hand (+ optional trump for trump-aware)
- Label: oracle's E[P(make X)] per contract
- Architecture: small MLP, mirror exp 15 v2 scale
- Speed: ~0.4 ms vs oracle's ~100 ms → 250× faster

### Step 5 — Opponent inference (future)

Update talon distribution by opponent bid/pass signals. Requires
opponent bidding model.

### Step 6 — CFR / equilibrium (much later)

CFR over the bidding tree for Nash equilibrium bidding. Deferred
until heuristic + threshold tuning is solid.

## Working principle

**Pause after each step** to look at numbers and decide whether the
next step is still worth doing. Don't pre-commit to the full chain.

## What we are NOT doing yet

- Multi-player bidding auction logic (whose turn, raise/pass mechanics)
- Kontra / rekontra
- Silent contracts beyond what exp 15 already scores
- Opponent modeling

Once step 2 lands a working oracle pickup decision, we can decide
whether to invest in steps 3-5.
