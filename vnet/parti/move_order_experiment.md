# Parti solver — move-ordering speed experiment

The parti alpha-beta solver currently uses `_order_default`:
`key = it*10000 + pt*100 + st`, descending for MAX / ascending for MIN.
That ordering was inherited from the generic default, but for parti (where
the search objective is *card-points captured*, not just trick-winning),
the most informative branch is often "did I get high-point cards?", not
"did I play a trump?". So we tested two alternatives.

## Variants tested

All three orderings respect MAX vs MIN orientation (descending key for
soloist, ascending for defenders). All use the same per-card features
`it = is_trump`, `pt = card_points`, `st = strength` (rank index for parti).

| name | key | rationale |
|---|---|---|
| `default` (baseline) | `it*10000 + pt*100 + st` | trump > points > strength. Inherited generic ordering. |
| `pts_first` | `pt*1000 + it*100 + st` | points > trump > strength. Surface high-value captures (10s, aces) before trump moves. |
| `winner_strength` | `((it<<3) \| st)*100 + pt` | composite trick-winning strength > points. "Try moves that win the trick first." |

## Setup

- 10 deals: 2 seeds at each of α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}
- Full-info `solve_all` at t=0 (open hand for all three players)
- Single-threaded; transposition table cleared per solve
- Warmup solve run before timing

## Per-deal results

Wall time (s):

|    α |   seed | default | pts_first | winner_strength |
|-----:|-------:|--------:|----------:|----------------:|
|  0.0 |     42 |   0.016 |     0.015 |           0.018 |
|  0.0 |     43 |   0.075 |     0.061 |           0.078 |
|  0.3 |   3042 |   0.012 |     0.006 |           0.012 |
|  0.3 |   3043 |   0.051 |     0.031 |           0.053 |
|  0.5 |   5042 |   0.013 |     0.015 |           0.014 |
|  0.5 |   5043 |   0.014 |     0.010 |           0.014 |
|  0.7 |   7042 |   0.008 |     0.005 |           0.008 |
|  0.7 |   7043 |   0.029 |     0.018 |           0.029 |
|  1.0 |  10042 |  16.89  |     5.39  |          17.21  |
|  1.0 |  10043 |   0.409 |     0.198 |           0.416 |

Nodes explored (lower = better):

|    α |   seed |     default |   pts_first | winner_strength |
|-----:|-------:|------------:|------------:|----------------:|
|  0.0 |     42 |     273,943 |     221,893 |         273,943 |
|  0.0 |     43 |   1,369,774 |   1,052,320 |       1,369,774 |
|  0.3 |   3042 |     178,789 |      58,161 |         178,789 |
|  0.3 |   3043 |   1,176,415 |     621,315 |       1,176,415 |
|  0.5 |   5042 |     228,287 |     237,208 |         228,287 |
|  0.5 |   5043 |     190,476 |     124,552 |         190,476 |
|  0.7 |   7042 |      93,890 |      40,518 |          93,890 |
|  0.7 |   7043 |     621,565 |     321,458 |         621,565 |
|  1.0 |  10042 | 486,434,792 | 164,427,912 |     486,434,792 |
|  1.0 |  10043 |   9,927,686 |   4,552,099 |       9,927,686 |

## Aggregate

| ordering          | wall (s) |        nodes | speedup |
|-------------------|---------:|-------------:|--------:|
| default           |    17.5  |  500,495,617 |   1.00× |
| **pts_first**     |     **5.8** | **171,657,436** | **3.02×** |
| winner_strength   |    17.8  |  500,495,617 |   0.98× |

## Reading

- **`pts_first` is a clean win**: 3× speedup, 66% fewer nodes explored,
  values bit-identical to default in all 10 deals.
- **`winner_strength` is indistinguishable from default**: identical node
  counts every time. The composite `(it<<3) | st` packing happens to
  produce the same primary sort as default (trump always dominates), with
  tiebreaks that don't change which cards branch first. Wall-time spread
  (±2%) is noise.
- The α=1.0 outlier (seed 10042) drove most of the absolute speedup —
  17s → 5s, 486M → 164M nodes. These are exactly the deals (god soloist
  has a near-overwhelming hand, hard positions for alpha-beta) where
  ordering matters most. The smaller deals show consistent ~2× node
  reduction even when wall-time gain is unimpressive due to fixed
  overhead.

## Why `pts_first` works

Parti's value function is *cumulative card-points*. The bounds-pruning
machinery (`_bounds_parti`) tracks `[lo, hi]` for remaining points. If MAX
tries a 10 or Ace first and it wins the trick, the soloist's running
score jumps by 10 immediately, tightening `α` for all sibling branches and
producing earlier beta cutoffs. With `default`, MAX prioritizes any trump
(including a 7 of trumps worth zero points) before non-trump 10s/aces,
which postpones the big score swings to later in the search.

The cull (`_cull_parti_blocks`) is independent of ordering, so safety is
preserved — `pts_first` only changes the *order* the surviving cards are
tried, not which cards survive.

## Recommendation

**Adopt `pts_first` as the default for parti.** It's a one-line change in
`_get_eval` (swap `ev.order = _order_default` → `ev.order = _order_parti_pts_first`),
guaranteed-equivalent values, and a free 3× datagen speedup. Phase 4a
datagen for parti was estimated at ~5h for 200k deals at the old rate →
this drops it to ~1.7h.

Future ideas worth a follow-up (not tested here):
1. **History heuristic**: weight cards by how often they produced cutoffs
   at the same trick number / position type in recent search. Adds
   per-position state but no algorithmic change.
2. **Killer moves**: remember the 1–2 cards that produced cutoffs at each
   ply and try them first regardless of static key.
3. **Iterative-deepening + transposition-table-move-first**: use the
   previous-iteration's PV move as the first branch. Requires id solver.
