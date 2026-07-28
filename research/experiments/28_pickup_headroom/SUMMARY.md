# exp 28 — Can refining the pickup/discard net beat the frontier? (2026-07-21)

## Verdict
**No — not meaningfully, and the discard specifically is a STRENGTH, not a weakness.**
The pickup net is not undertrained for the discard; the naive "use PIMC for the put-down"
idea actually makes it WORSE. The only real pickup lever is a contract-choice retrain, capped
at ~+1.3 GP/deal (exp25, known), and calibration alone is a proven wash (exp18). The real
frontier is in-play opponent modeling.

## How the discard works (confirmed)
The talon put-down is chosen by the NET, not PIMC: the pickup enumerates 4 trumps × 66
two-card discards, scores each kept 10-hand with the 7 base-event heads, and plays the argmax.
One net, evaluates the 10 kept cards; it never sees the 2 buried cards. PIMC (N=16) is play-only.

## Measurement 1 — discard CEILING (god perfect-info, N=1500 opener bids)
Given the net's chosen contract, would a *better* discard flip a loss to a win?

| contract | n | net make% | god-best make% | fixable-loss |
|---|---|---|---|---|
| parti | 1281 | 20% | 24% | 4% |
| ulti | 158 | 91% | 97% | 6% |
| 40-100 | 26 | 100% | 100% | **0%** |
| 20-100 | 3 | 100% | 100% | 0% |
| durchmars | 9 | 44% | 56% | (thin) |

**Overall 4.3% of opener bids** have a discard-fixable loss, concentrated in parti (4pp,
binary point battle — barely matters realistically) and ulti (6pp). The 40-100/20-100 discards
are already perfectly optimal. This is the CEILING (requires seeing opponents' cards) ≈ +0.3 GP/deal.

## Measurement 2 — ACHIEVABLE (cheat-clean PIMC discard, K=6, N=335 ulti/duri hands)
Replace the net's argmax discard with a cheat-clean PIMC-chosen discard (sample opponents,
roll out each of 66, pick best). Does it capture the ceiling?

| contract | n | net make% | PIMC-discard make% | god-best% |
|---|---|---|---|---|
| ulti | 294 | **90%** | **82%** | 96% |
| ulti+40-100 | 24 | 100% | 100% | 100% |
| durchmars | 13 | 62% | 69% | 69% |

**Overall: cheat-clean PIMC discard is −6.9pp WORSE than the net** (82% vs 90% on ulti;
ceiling +5.1pp). The K=6 rollout overfits its 6 sampled worlds (the same argmax-over-66
optimizer's-curse that exp19/20 diagnosed) and generalises worse than the net, which carries a
smooth 1M-hand prior. Higher K might close the gap but is deployment-infeasible (the bid discard
currently uses ZERO rollouts; K×66 god-solves per bid is far too slow for interactive play).

**So the net's O(1) discard at 90% (ulti, ceiling 96%) is excellent and beats the obvious
alternative. The discard is not a lever.**

## The only remaining pickup lever — contract choice
From exp25 (firmed N≥500): a realistic-label RETRAIN of the base-event heads is worth ~+1.3–1.5
GP/deal — 18% of the god edge; the other 82% is irreducible information gap (the bidder can't see
opponents' hands). Improving pickup CALIBRATION alone is a demonstrated strength-wash (exp18
canonical: better Brier, agent Δ −0.05). The heads are not undertrained by data (1M god labels,
AUC 0.95–0.99); the rare heads (betli, colorless_duri, reach100_20) are overconfident but propped
up by isotonic calibration, and fixing that didn't help the agent.

## Recommendation
- **Do NOT touch the discard** — it's near-optimal and the naive PIMC replacement is worse.
- A pickup-net retrain on realistic (PIMC) labels is the only pickup lever, ceiling ~+1.3 GP/deal;
  a real but bounded project (datagen + train + tournament gate) if milan wants it.
- The bigger frontier remains **in-play opponent/ exploit modeling** (exp25 FRONTIER), not the pickup net.

## Reproduce
`experiments/28_pickup_headroom/harness28.py` — `build` (discard ceiling), `analyze`,
`pimc` (cheat-clean achievable), `pimc_analyze`. Data: `discard.jsonl`, `pimc_discard.jsonl`.
