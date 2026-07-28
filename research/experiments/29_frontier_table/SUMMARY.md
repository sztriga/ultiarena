# exp 29 — Frontier self-play table & bleed audit (2026-07-21)

3 frontier models (KONTRA-aware bidder that passes weak hands, full auction where any seat may
open, PIMC play, promoted per-unit kontra, oracle+silents) played **6,000 deals**. Seat 0 =
forehand/opener (fixed frame → positional analytics).

## Contract table — frequency & avg soloist GP
| contract | freq | avg soloist GP | made% | kontra% |
|---|---|---|---|---|
| passz | 32.6% | opener pays −4 | — | — |
| piros parti | 32.5% | +2.30 | 52% | 80% |
| piros ulti | 26.2% | +10.38 | 88% | 4% |
| piros 40-100 | 3.3% | +10.65 | 82% | 0% |
| teritett rebetli | 2.2% | +59.4 | 87% | 0% |
| piros ulti-40-100 | 1.2% | +26.6 | 89% | 0% |
| **teritett színtelen duri** | 0.7% | **−4.80** | 45% | 0% |
| piros 20-100 | 0.4% | +14.6 | 72% | 0% |
| **piros terített 40-100-duri** | 0.4% | **−14.0** | 25% | 0% |
| **piros terített 20-100-duri** | 0.2% | **−18.3** | 21% | 0% |
| (5 more combined, <0.1%) | | | | |

Played 67% of deals, soloist makes 69%, auctions rarely contested (avg 1.04 bids, 4% overcalled).

## Per-seat (position)
| position | mean GP/deal | wins bid | GP as soloist | GP as defender |
|---|---|---|---|---|
| P0 forehand | **−1.03** | 31% | +6.36 | −4.74 |
| P1 middle | **+1.04** | 19% | +10.87 | −3.46 |
| P2 rear | −0.01 | 17% | +7.96 | −4.03 |
Zero-sum; rotates out over a match. Forehand is the worst seat (pays −4 on the 33% all-pass
deals — the forced-opener tax); middle is best. The FULL auction is harsher on P0 than exp25's
run_auction (+0.91) because P0 also loses as a defender when it passes into a P1/P2 bid.

## Bleed audit — TWO checks

### 1. The DURI family — a real, fixable over-bid leak
Every negative-GP contract involves durchmars. The 88 duri-family deals (1.5%) are made only
**35%** and average **−7.64 GP** → total **−0.13 GP/deal**. The frontier OVER-BIDS terített duri.
**Mechanism (ties to exp28):** the `colorless_duri` / `duri_colored` net heads are the worst-
calibrated in the model (~400 positives in 1M, overconfident in the high-confidence bins the
argmax selects). So the composer over-values duri combos → over-bids → the soloist loses.
**Fix:** raise the FLOOR / a per-contract threshold for the terített-duri contracts, or
recalibrate the duri heads. Small aggregate but genuine and mechanistically explained.

### 2. Over-passing? NO — the bidder's conservatism is correct
Hypothesis: KONTRA=1 passes on a threshold that assumes OPTIMAL god defender kontra, but the
deployed defenders now kontra leniently (post-exp27) → maybe the forehand should bid the floor
instead of paying −4 on 33% of deals. **Tested** (force piros parti on all 1957 passz hands,
play out vs the lenient defenders): realized **−7.89 GP** vs the −4 passz — bidding is **−3.89
WORSE**, and beats passing on only 24% of them (those hands make piros parti just 20%). So
**passing is correct**; the forehand tax is purely structural (rotates out), not a fixable leak.

## Bottom line
The frontier is healthy on the bulk: piros parti +2.30, piros ulti +10.38, 40-100 +10.65,
rebetli +59. The only bleed is **over-bidding terített durchmars** (−0.13 GP/deal), traced to the
miscalibrated duri heads — the one concrete, fixable leak. The forehand positional deficit is
structural (the all-pass rule) and correct (bidding more loses). Bidding is otherwise well-tuned.

## Reproduce
`experiments/29_frontier_table/` — `frontier_selfplay.py` (build), `analyze.py` (TABLE.md),
`force_parti.py` (over-pass test). Data: `selfplay.jsonl` (6000 deals), `force_parti.jsonl`.
