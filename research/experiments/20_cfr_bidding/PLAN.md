# Exp 20 — CFR bidding policy with belief-aware overtaking

## Motivation

The current bidder (`auction_h2h.simulate` + `CompositePickup`) decides each
bid by **argmax expected value over a uniform talon/opponent prior** — it
averages over all 231 talons and treats opponents' hands as unconditioned.
It throws away the information in the auction: when the player to my right
bids ulti, the posterior over his hand (and the talon, and the remaining
defender) shifts hard, and a good bidder should condition on that.

Two known weaknesses of the argmax bidder:
1. **Optimizer's curse** — argmax over 66 discards inflates the chosen p,
   which (combined with betli's low bid floor) lets in losing betlis.
2. **History-blind uniform prior** — no belief updating from the bids.

CFR fixes both *for free*: at equilibrium, bids carry information, beliefs are
reach-consistent, and a contract whose god-true leaf is negative is simply not
bid. This experiment builds a CFR bidding policy and compares it head-to-head
with the composite model in a tournament where the **card-play is held fixed**
(PIMC32 soloist vs god defenders), so we measure bidding quality in isolation.

## Game (standard Ulti auction)

- **Deal:** 10/10/10 + a fixed hidden 2-card talon (symmetric across seats —
  cleaner than exp17's "P0 gets 12, talon passes"). The auction winner picks
  up the talon → 12 → discards 2 → plays 10.
- **Opener:** P0 forced. P0 may pass-out (eats the −2 floor: payoff
  [−4, +2, +2]) or open with any available contract. Only P0 opens.
- **Bidding actions** (escalating rank, mirrors `contract_rank`):
  `parti`=2 (piros parti), `ulti`=3 (best non-piros suit), `betli`=4,
  `duri`=5 (durchmars colorless), `ulti_piros`=6. Plus `pass`.
- **Availability (deployable, from own 10 cards):** parti/betli/duri always;
  `ulti` iff a non-hearts trump-7 is in hand; `ulti_piros` iff hearts-7 in
  hand. (The agent keeps its 7, so an "available" ulti is always playable.)
- **Turn order:** strict round-robin starting at P0. The current holder
  auto-passes its own turn. 3 consecutive passes after a bid → the holder is
  the soloist.
- **Payoff (zero-sum GP, matches the harness):** soloist gets `2·EV/def`,
  each defender `−EV/def`.

## Leaf payoff = god-exact (the honest training signal)

For each deal and each (player, action), the leaf = **god-exact EV/def** of
the value-model's best realization (best suit∈allowed, best discard from the
real 12) under perfect double-dummy play. Using god (not the value model) at
the leaf is what makes the experiment meaningful:
- it removes the optimizer's-curse inflation from the *reward*, so CFR won't
  re-learn to bid inflated betlis;
- it uses the **actual opponent hands**, so the reach-weighted counterfactual
  values encode the hand correlations that belief-updating exploits.

Consistency note: the composite's value model is itself god-label-trained, so
both agents reason in the same "god-makeability" currency; the tournament's
PIMC32 soloist play is the common reality check.

Cost (measured): ~27 ms/deal, ~15 god solves/deal → 50k deals ≈ 3 min on 8
cores. Cheap; precompute once, reuse across all CFR iterations.

## Abstraction

- **Infoset = (own hand bucket, public bid history).** History is the full
  round-robin action sequence (bids + passes) — fully public.
- **Bucket (deployable):** discretized value-model P(make) for the 5 actions
  computed on the raw 10-card hand. Same function used in training (to index
  the strategy) and at deployment (the agent has only its cards).

## Trainer

External-sampling MCCFR over the bidding tree, leaves + buckets cached per
deal. Belief-updating is implicit: P1's regrets at "(bucket, P0-bid-ulti)" are
reach-weighted over the deals where P0's bucket actually bids ulti, i.e. over
the *posterior* opponent hands — never a uniform prior.

## Tournament

`tournament.py` runs both a `CompositeAgent` (argmax-EV, the current bidder)
and a `CFRAgent` (strategy lookup) through the **same** standard auction +
PIMC32-sol-vs-god-def playout, rotating seats so the forced-opener penalty is
shared. Reports per-seat / per-contract GP and the head-to-head delta.

## Files

```
common.py      deal, action set, availability, value-model realizations
game.py        pure bidding-tree logic (states, legal actions, payoffs, infoset)
buckets.py     hand → discrete bucket (deployable)
leaves.py      god-exact leaf precompute over a deal pool → npz
cfr.py         external-sampling MCCFR trainer → strategy table
agents.py      BiddingAgent: CompositeAgent + CFRAgent
tournament.py  CFR vs composite, fixed playout, seat-rotated
benchmark.py   (done) god-leaf cost de-risk
```
