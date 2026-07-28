# exp36 — Betli defense: a learned policy beats PIMC (research report, 2026-07-23)

**Question (milan):** betli is where the PIMC defenders are weakest — "strategy is underutilized."
Benchmark the current defense vs god, then see if a learned policy/architecture can help.

## The distribution caveat (important)
The frontier bidder **essentially never bids *plain* betli** — it bids *terített* (open-hand) betli
or other contracts. So the "weak betli defense" is about **plain, hidden-info betli a *human* opponent
would bid**. All numbers below use realistic bid-worthy plain-betli hands (`deal_betli(α=1.0)`, strong
soloist → 21% double-dummy-lost).

## Phase 1 — Benchmark: how much are we leaking?
On **605 defender-holdable** betlis (perfect defense beats them):

| soloist \ defender | PIMC def | god def |
|---|---|---|
| PIMC soloist | **60.3% steal** | 0.0% |
| god soloist  | 66.1% steal | 0.0% |

**The frontier PIMC defense lets the soloist STEAL ~60% of the betlis it should hold** — it takes only
~40% of what perfect defense takes. **~60 points of pure defensive headroom.** (A random legal defensive
move is safe 79–83% of the time — PIMC is barely above random *per move*, so gains compound.)

## Phase 2 — A learned defense policy
Cheat-clean net (139-dim: own hand + public/played + unknown + current trick + lead/position/viewer),
trained supervised to predict **safe moves** (a move is safe iff the soloist *still* double-dummy-loses
after it). Data: dd-lost deals, god soloist + exploring defenders, safe-labelled at recoverable nodes.

### Architecture sweep (steal rate on held-out dd-lost betlis; PIMC baseline 62%)
| arch | steal | Δ vs PIMC |
|---|---|---|
| **mlp/256** | **51.9%** | **−10.3pp** |
| mlp/384 | 53.2% | −9.1 |
| mlp/512 | 54.3% | −8.0 |
| wide/256 | 54.7% | −7.6 |
| deep/256 | 70.1% | **+7.8 (worse than PIMC)** |

The **simplest MLP wins**; wider/bigger overfit, and *deeper actively hurts*.

### Data scaling (mlp/256 — the big lever)
| training positions | steal | Δ vs PIMC | val top1-safe |
|---|---|---|---|
| 43k | 51.9% | −9.2 | 0.934 |
| 108k | 44.0% | −18.6 | 0.949 |
| **303k** | **41.4%** | **−21.3** | 0.954 |

More data drove the biggest gains (trend now flattening). **Final: 41.4% steal (−21.3pp) — recovers
~34% of the headroom.**

### The search hybrid is *worse* than the net alone
Net-guided PIMC (net proposes top-K, PIMC picks among them): **51% vs the pure net's 44%.** PIMC's noisy
world-sampling *second-guesses the net's good moves* → **PIMC is the weak link**; the pure learned policy
wins, and it needs no search (one forward pass — faster than PIMC).

## Bottom line
**A learned betli-defense policy clearly beats PIMC (−21.3pp steal, ~34% of the headroom recovered) — the
first play-policy win over search in this project.** It validates milan's instinct that betli defense is
underutilized strategy, and it's the contract where a net *should* win: the search weakness is largest and
the hidden-info gap smallest.

**Deployable:** `infer.py` (feature-encoder verified identical to training over 2268 positions), model at
`models/betli/betli_defense.pt`. Wiring: when the AI is a defender in a *plain* betli (a human bid it), call
`betli_defense_pick(pos, viewer)` instead of PIMC. Low-risk (plain betli is human-initiated & rare) and a
clear improvement.

**Next levers:** even more data (flattening but not flat); eval vs a god/exploit soloist (stronger attacker);
a richer target (win-prob under imperfect play, not binary-safe); extend the approach to other contracts.
