# Parti — win-rule audit and threshold fix

While wiring up the parti contract into the `betli.hu` playable UI we
realised the win threshold used across the parti pipeline (`>= 51`
soloist card-points) was a **bug**, not the actual game rule. This note
records what the canonical rule is, where the wrong threshold lives, and
what numbers it invalidates.

## Canonical rule

The trickster game state authority is
`trickster/games/ulti/game.py:soloist_won_simple`:

```python
def soloist_won_simple(state):
    """Did the soloist win Simple? (more total points than defenders combined).
    Strict inequality: ties go to the defenders."""
    return soloist_points(state) > defender_points(state)
```

with:

- `soloist_points(state)` = soloist's accumulated trick scores.
- `defender_points(state)` = combined defender trick scores **+ the
  card-point value of the 2 talon discards**. The talon goes to the
  defenders by rule (`game.py:412-420`).

Card points come from:

- Aces and Tens: 10 each (`cards.py:Card.points`, 4 + 4 = 8 cards × 10 = 80)
- Last-trick bonus: +10 to whoever takes trick 10 (`LAST_TRICK_BONUS`,
  applied in `game.py:375`).

Total in play = **90 points**, always in 10-point increments. Therefore:

- `sol > def` ⟺ `sol > 90 - sol` ⟺ `sol > 45` ⟺ **`sol >= 50`** (smallest
  integer multiple of 10 satisfying it). Ties (sol = def = 45) can't
  occur because increments are 10, but the rule would award them to the
  defenders anyway.

## What was wrong

Every parti-side decision was using `>= 51` as the threshold. Off by 5
points: the boundary hand (`sol = 50`, `def = 40`) was being scored as a
**defender win** when the soloist actually won it.

Locations of the wrong threshold (pre-fix):

| file | symbol |
|---|---|
| `solvers/parti_game.py:79` | `sol_wins = sol_score >= 51` (MCTS leaf outcome) |
| `scripts/play_matchup.py` | `_god_def_wins` and `_final_def_wins` (51-pt cutoff) |
| `apps/api/betli_hu.py` (added in this session) | `_PARTI_WIN_PTS = 51.0` (now fixed) |

## What was fixed

The playable UI path is now correct:

- `apps/api/betli_hu.py`: snapshot terminal check calls
  `soloist_won_simple` directly; the winnability probe (`_sol_can_win`)
  uses `>= 50`.
- `apps/api/pis.py`: branch-exploration verdict uses `>= 50`.

The MCTS+V experiment paths (`solvers/parti_game.py`,
`scripts/play_matchup.py`) **still use `>= 51`**. Production parti AI is
PIMC on both seats (per the phase-4 sweep result, MCTS+V couldn't beat
the strategy-fusion ceiling), so the bug is latent — but the historical
eval tables under `vnet/parti/results/` were computed against the wrong
threshold and are off in a known direction.

## Estimated impact on existing eval tables

Affects `vnet/parti/results/phase4b_full_matchup.md` (n=50/α, pimc_n=8) and
the budget sweep printouts in `/tmp/parti_budget_sweep.log`.

The error mode is: any deal where the soloist ended on **exactly 50
points** was credited to the defenders. Effect on the headline metrics:

- `god vs god` rows: `n_def / n_sol` counts may shift slightly (deals
  on the boundary mis-bucketed).
- `pimc vs god`, `mcts_v vs god` rows: `sol_hold` ratio is the fraction
  of god-soloist-wins the AI actually held; a soloist with a true score
  of 50 was being recorded as "lost the hold". So `sol_hold` is biased
  **downward** by however many runs landed at exactly 50.
- Mean soloist score is unaffected (the score itself was reported
  correctly; only the win/loss flag was wrong).

The biased magnitude isn't dramatic — 50-pt outcomes are a tail event,
not the median — but the directional caveat applies to every parti
matchup number we've published so far.

## Talon-points-to-defenders — UX implications

Because the talon's 2 cards count for the defenders, a soloist who
shuffles aces/tens into the talon is literally handing the defenders
points. This is invisible to a defender in the UI (only the soloist
sees the talon), but it affects the on-screen "≥ 50 wins" threshold
identity from both sides: trick + talon always sum to 90, so the
threshold the soloist is racing toward is still `sol_trick_pts >= 50`,
regardless of what's in the talon.

Potential follow-up: surface "Talon → defenders" near the trump badge
so the rule is legible to defenders who never see the talon contents.

## Status

- Live playable UI: fixed.
- MCTS+V code path: bug remains (`solvers/parti_game.py:79`,
  `scripts/play_matchup.py`). Acceptable because parti production is
  PIMC; flagged here so anyone re-running MCTS+V is aware.
- Historical eval tables: still reflect the wrong threshold; not
  re-run. Re-run only if a parti experiment depends on those numbers
  being exact.
