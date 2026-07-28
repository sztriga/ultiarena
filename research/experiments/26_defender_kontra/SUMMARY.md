# exp 26 — Defender-kontra fix (overnight, 2026-07-20 → 21)

## TL;DR
The deployed AI defender **kontras almost everything** (98% of colored-simple games).
That is a **−3.1 GP/deal drag on the defenders** — and it is almost entirely a
**piros-ulti** problem. A trivial, cheat-clean, contract-specific gate recovers
**+4.37 GP/deal for the defenders vs the deployed rule**, out-of-sample (N=5374,
held-out test). No retraining. ~4 lines in `apps/api/play.py`.

## What the deployed defender does today
`_ai_defender_kontras` (play.py:415): kontra iff own-hand **god-makeability**
`_sol_ev(p) < 0`. Because piros parti/ulti pay asymmetrically, that fires whenever
p < ~0.5 — i.e. **almost always** (98%).

## Why it's wrong (the smoking gun)
God-makeability measures "beatable by **perfect** defense." The real game has
**imperfect (PIMC) defense**, so the soloist makes far more than god predicts —
catastrophically so for ulti:

| contract | god-makeability signal | ACTUAL make rate |
|---|---|---|
| ulti | < 0.2 | **77–83 %** |
| ulti | 0.2–0.4 | **90 %** |
| parti | < 0.2 | ~22 % (well-calibrated) |

So the defender kontras a piros ulti it estimates at ~30%, the soloist makes it 83%
of the time, and — because kontra is a payoff multiplier and the soloist rekontras a
strong hand — the defenders pay **double (often ×2 again)**. Measured cost:

| contract | deployed defender gain vs never-kontra (held-out) |
|---|---|
| parti | **+0.56** (kontra-ing parti is fine — 73% is bukott) |
| **ulti** | **−16.4** (catastrophic) |

## What we tried, and what actually works
| signal (defender's own hand) | best ulti gain vs never, out-of-sample |
|---|---|
| god-makeability threshold (N=6→40) | +0.11 (basically "never kontra ulti") |
| **auction-conditioned** makeability (reject soloist worlds the champion wouldn't bid) | +0.0–0.4 — helps ulti a little, no-op for parti (piros parti is the floor) |
| **realistic-defense** makeability (PIMC playout per world, not god) | +0.42 — still miscalibrated (soloist hand ignores that they BID ulti) |
| **#trumps the defender holds** (structural) | **+0.44** — the winner |

Trump count is the only own-hand signal that separates ulti:

| defender's max #trumps | soloist ulti make rate |
|---|---|
| 1 | 100 % |
| 2 | 92 % |
| 3 | 71 % |
| **4** | **37 %** ← kontra is +EV here |

No 2-feature rule beats it: the 3-trump group stays 64–78% made regardless of
high/low composition. **The +3.3 ulti oracle ceiling is mostly UNREACHABLE
cheat-clean** — beatability lives in the *joint* defender holdings, which one
defender can't see.

## The recommended policy (held-out test, N=5374)
- **Ulti**: kontra iff the defender holds **≥4 trumps** (fires ~5%).
- **Parti**: tighten the makeability threshold (kontra iff p ≲ 0.06 instead of <0.5).

| policy | soloist GP/deal (lower = better defense) | vs deployed |
|---|---|---|
| deployed | +3.01 | — |
| **recommended** | **−1.36** | **+4.37** |
| oracle ceiling | −2.38 | +5.38 |

Split: parti **+0.56**, ulti **+16.85** vs deployed. The recommendation captures ~81%
of the reachable (oracle) improvement over the deployed rule.

## Proposed change (NOT applied — milan's call)
`apps/api/play.py`, `_ai_defender_kontras`:
```python
def _ai_defender_kontras(sess, pidx):
    # exp26: ulti is ~83% made vs real defense but god-makeability says ~30%, so the
    # makeability rule kontras makeable ulti and loses double. Gate ulti on the one
    # cheat-clean own-hand signal that separates it: hold >=4 trumps (-> 37% made).
    if sess.k_primary == "ulti":
        own = sess.play_hands0[pidx]
        return sum(1 for c in own if c.suit == sess.trump) >= 4
    p = _makeability(sess, pidx, 100 + pidx)
    return _sol_ev(p, sess.bid, 0) < 0    # parti/betli/durchmars unchanged
```
(Parti threshold tuning is a smaller, separate optional win; τ must be re-tuned on the
deployed N=6 signal, not the study's N=40.)

## Caveats / not covered
- Study covers **colored-simple** contracts (parti/ulti ≈ 90% of bids). betli /
  colorless-durchmars kontra is rare and unstudied.
- The soloist **rekontra** rule was held fixed across policies (isolates the defender
  lever). Fixing the defender kontra also removes the rekontra amplification.
- Numbers are self-play (all seats = champion) vs the exp24 harness; consistent with
  the live-engine measurement that motivated this (defender kontra fires 77%, right 28%).

## Reproduce
`experiments/26_defender_kontra/harness26.py` — subcommands `build`, `pools`,
`policies`, `realpools`, `analyze_real`, `analyze_combined`. Data cached in
`played.jsonl` (5374 deals) + `pools.jsonl`. Tables in `results.md`,
`results_combined.md`, `results_real.md`.
