# Exp 14 — Bidding minigame (fundamental contracts only)

## Goal

First end-to-end bidding-phase experiment built on top of the existing
fundamental-contract solvers. Per deal:

1. Sol receives 12 cards (10-card hand + 2-card talon they'll pick).
2. Sol evaluates **every** (talon discard × contract × trump) combo via
   PIMC32 and picks the maximum-EV option.
3. If the best EV is negative, sol **passes** (0 GP).
4. Otherwise sol plays the chosen bid out (PIMC32 on both sides) and
   the terminal is scored under the simplified GP table.

No combined contracts. No bidding loop yet. This is the scaffolding
that the production bidding system will sit on top of.

## Simplified GP table (per defender)

| Contract              | Made | Bukott / lost |
|-----------------------|------|---------------|
| parti                 | +1   | −1            |
| parti (piros = ♥)     | +2   | −2            |
| ulti                  | +2   | **−4**  (bukott duplán fizet) |
| ulti (piros = ♥)      | +8   | **−16**       |
| durchmars (colorless) | +6   | −6            |
| betli (colorless)     | +5   | −5            |
| pass                  | 0    | —             |

Sol's total = 2 × per-def. Only ulti has the "bukott pays double" rule.
Piros doubles parti and quadruples ulti (per milan's rule).

## Architecture

- `_lib.py` — `deal_12_10_10`, `eval_one_deal`, `best_record`. Exhaustive
  loop over 66 discards × 4 trumps × {parti, ulti} + {betli, duri}.
  Skips ulti when sol doesn't hold the trump-7 in the post-discard
  10-card hand.
- `perf_one_deal.py` — single-deal timing harness.
- `perf_thread_sweep.py` — multiprocessing sweep over worker counts.
- `run_minigame.py` — end-to-end: eval → pick best → play out → score.

## Calls per deal

- Upper bound: 66 × (4·parti + 4·ulti + betli + duri) = **660**
- Actual: depends on how many trump-7s sol holds. With ~1.5 expected,
  typically **400–460** PIMC32 calls per deal.

## Performance (PIMC32, uniform shuffle)

**Single-deal perf** (450 actual calls, 33 ms/call average):
- 1 thread: **~15 s/deal**

**Across-deal multiprocessing** (N=8 deals):

| Workers | Wall  | s/deal | Speedup |
|---------|-------|--------|---------|
| 1       | 126 s | 15.8   | 1.00×   |
| 2       |  67 s |  8.3   | 1.89×   |
| 4       |  52 s |  6.5   | **2.44×** |
| 8       |  45 s |  5.7   | 2.79×   |

Diminishing past 4 workers. Sweet spot ≈ 4 workers, ~6.5 s/deal.

Within-PIMC threading exists (the `alloc_context` machinery), but at
33 ms/call the thread-spawn overhead eats the gain. Across-deal Pool
parallelism is the right granularity here.

## N=20 results

169 s wall, 4 workers, uniform shuffle, no biased dealing.

| | |
|---|---|
| Pass rate                    | 6/20 (30%) |
| Bid rate                     | 14/20 (70%) |
| Mean predicted EV (bids)     | +1.96 |
| Mean actual GP (bids)        | +2.29 |
| Calibration delta            | +0.33 (slight pessimism) |
| **Mean GP/deal (incl pass)** | **+1.60 per def** (+3.20 to sol) |

### Bid distribution

| Contract            | n | Pred EV | Actual | Won % |
|---------------------|---|---------|--------|-------|
| ulti / hearts (piros) | 3 | +4.75 | **+8.00** | 100% |
| ulti / leaves         | 2 | +2.00 | +2.00 | 100% |
| ulti / bells          | 2 | +1.91 | **−1.00** | 50% |
| parti / leaves        | 3 | +0.54 | +1.00 | 100% |
| parti / bells         | 2 | +0.38 | 0.00 | 50% |
| parti / acorns        | 1 | +1.00 | +1.00 | 100% |
| parti / hearts        | 1 | +2.00 | +2.00 | 100% |

**Betli and duri never bid.** Likely structural at uniform deals: a
biddable betli or duri needs a very specific hand shape that rarely
appears in random 12-card subsets, and the parti/ulti EV usually
dominates if anything else is biddable. Worth investigating with
biased dealing if we care.

**Piros ulti is the biggest source of GP** — predicted +4.75, actual
+8.00 over 3 deals. PIMC's P(make) estimates tend to be conservative
when sol has a clearly winning hand (god solver would say 100%, PIMC
might say 84%).

## Forensic: seed 100004 (the bust)

Sol bid `ulti/bells` with PIMC `p_make = 0.97`, predicted EV +1.81 →
actual outcome = bukott (−4). Investigation:

1. **The bid was correct.** Running the god solver on the true
   position confirms sol has a **forced 100% ulti win** from this
   hand (best openings: ace-bells, 10-bells, ace-leaves, 10-leaves).
   PIMC's 97% estimate was accurate.

2. **Sol's PIMC during play picked a series of slightly-suboptimal
   moves** and lost the forced win. Trace (T = trump = bells):

   ```
   T1  SOL ace♠ → DEF1 8♠, DEF2 7♣        (sol wins)
   T2  SOL ace♦leaves → DEF1 upper, DEF2 8 (sol wins)
   T3  SOL 10♦leaves → DEF1 king, DEF2 9   (sol wins)
   T4  SOL lower♦leaves → DEF1 9♠TRUMP, DEF2 9♥  (def1 wins, void-trumps)
   T5  DEF1 upper♣acorns → DEF2 king, SOL 8     (def2 wins)
   T6  DEF2 9♣acorns → SOL upper♠TRUMP, DEF1 ace  (sol wins, burns upper)
   T7  SOL upper♥ → DEF1 king, DEF2 ace      (def2 wins)
   T8  DEF2 lower♥ → SOL 10♠TRUMP, DEF1 7    (sol wins, burns 10)
   T9  SOL 7♦leaves → DEF1 lower♠TRUMP, DEF2 lower  (def1 wins)
   T10 DEF1 king♠TRUMP → DEF2 10♣, SOL 7♠TRUMP   (def1 wins → SOL BUKOTT)
   ```

   By T9 sol had spent every trump except the 7 trying to fight back
   into the lead. T10 was a forced trump-follow that lost.

3. **This is the strategy-fusion limitation of PIMC.** PIMC averages
   over sampled worlds and commits to a *single* policy across them,
   while the true optimal play is sometimes a narrow line specific to
   the actual world. God-vs-god finds the line; PIMC32-vs-PIMC32
   doesn't. More samples (`PIMC_N`) smooths noise but doesn't fix the
   structural problem — that's what CFR or info-set-aware
   algorithms address.

**Takeaway for the minigame:**

- The **bid eval is sound**; it correctly identifies the +EV play.
- The **play layer leaks EV** because PIMC isn't god. The actual GP
  earned will systematically under-predict optimal-policy GP. The
  `ulti/bells` row (50% won vs 96% predicted) is the visible signature.
- Fixing this means stronger play: deeper PIMC (smooths noise),
  V-net-distilled god (cheap god approximation), or CFR-style
  info-set lookahead (proper fix for strategy fusion).

## What "depth" means here

There is no "depth" parameter in the usual minimax sense — within each
sampled world the god solver runs **to terminal**, full alpha-beta.
The lever is `PIMC_N` (currently 32 = number of sampled worlds per
decision). Increasing it smooths the per-move estimate; it does not
make PIMC any closer to god, just a less noisy estimate of the
PIMC-equilibrium value.

## Reproduce

```
PYTHONPATH=. python3 experiments/14_minigame_bid_eval/perf_one_deal.py        # 1-deal perf
PYTHONPATH=. python3 experiments/14_minigame_bid_eval/perf_thread_sweep.py    # worker sweep
PYTHONPATH=. python3 experiments/14_minigame_bid_eval/run_minigame.py         # full eval
```
