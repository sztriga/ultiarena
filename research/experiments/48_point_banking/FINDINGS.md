# exp48 — point banking (kenés): the AI is indifferent where humans are not

**Status: DEAD — negative result. Do not build the tie-break.**

The chain "8.6 points/deal fed -> 84.4% of those are ties -> a free GP win" is wrong at the
last step. Replaying 400 of those exact positions both ways with the deployed stack:
**defenders gain +0.000 +- 0.223 (unchanged in 95% of positions), soloist -0.082 +- 0.540
(unchanged in 78%).** Shedding the ten does not save it — you still hold it, you play it a
few tricks later, and the same side captures it anyway. Banking changes WHEN the points
move, not WHERE they end up. See `counterfactual.py` and the section at the bottom.
Raised by milan 2026-08-03 from two of his own games; measured on the exp47 natural corpus.

## What he noticed

Two games, same shape.

**`e9c16097793d` — 40-100, trick 4.** milan led ♦F, def1 won with ♦10, and def2 played **♥8**
while holding the **♥10**. A human plays the ten there — *kenés* — because the partner has
already won the trick, so the points are free. Instead def2 kept it and milan **ruffed it
away on trick 5**. The post-game analysis flagged nothing.

**`6dabc533b0e2` — ulti, trick 10.** The defenders held ♦A and ♦10 to the last trick and
milan took **both (20 points)** with the trump 7. Again nothing flagged.

## Why the analysis says nothing — and is right

At the 40-100 moment, **all seven legal moves score +1.00**, ♥10 included. At the ulti
moment, ♦K and ♦A both score **+9.00**. The evaluator is not missing the ten; it rates it
exactly as good as the eight.

And for those two deals it is correct. milan finished the 40-100 on **90** — ten short —
and that 90 *includes* the ♥10 he ruffed; banking it would have made it 80. Still short.
The ulti finished on **80**; banking would have made it 70, still far past the parti
threshold. Both outcomes were settled.

The structural reason is the objective:

    bid 40-100:  1.0 x parti(+-1)  +  8.0 x reaches_100(0/1)
    bid ulti:    1.0 x parti(+-1)  +  8.0 x ulti_made(0/1)

**It is a weighted sum of BINARY indicators.** Card points enter only through the parti
win/lose flag, which is usually settled long before the endgame. So nearly every
point-card decision is a numerical tie — and the AI breaks ties **arbitrarily**.

Two consequences, and the second is the expensive one:

1. Double-dummy cannot value **insurance**. Kenés banks points so they cannot be lost
   later; god already knows which tricks it will win, so it never needs to hedge. The real
   defenders were not playing double-dummy — they lost the cards to a ruff. This is the
   same blind spot as [exp46](../46_rekontra_calibration/FINDINGS.md): per-world
   double-dummy is over-confident about what it can still arrange later.
2. The AI plays the same arbitrary way when the parti is **close**, where it costs real GP.
   It is only saved by the margin usually being wide.

## How often it costs money

800 parti-live deals from the exp47 natural corpus (deployed frontier, cheat-free bidder).
The unambiguous error is a defender playing a ten or ace into a trick the SOLOIST won while
a zero-point card was legal:

| | |
|---|---|
| deals containing at least one such feed | **48%** |
| points handed over, average | **5.8 / deal** |
| soloist parti wins that flip if only those feeds are fixed | **78 / 692 = 11.3%** |
| deals decided by a single 10-point card | 27.2% |
| median parti margin | 15 points |

**Roughly one deal in nine, the defenders hand the soloist the parti with points they did
not have to give.** Parti is kontra'd 81% of the time and doubles again on piros, so that
single card is often worth 2-4 GP rather than 1.

### Numbers NOT to quote from this analysis

Two intermediate figures are inflated and are recorded here so they are not reused:

* *"18.8 points/deal of missed kenés"* counts **opportunities**, not errors. Playing low
  while holding a ten is frequently correct — the ten may be needed to win a later trick.
* *"42% of partis would flip"* assumed every feed AND every missed kenés could be fixed
  **simultaneously**, which is not achievable (banking a ten spends it).

**11.3% is the defensible number**, and even that is a mild over-estimate: some feeds are
forced by what the defender must keep for later tricks.

## Proposed fix

Among moves the solver rates **equally**:

* opponent currently winning the trick → play the **lowest**-point card
* your own side winning it → play the **highest**

Both halves are god-value-neutral by construction, so this cannot regress — the same
guarantee that makes the anti-tell equivalence mixer free
(`tests/ulti/test_block_equivalence.py`). It replaces a coin flip with the human-correct
move, and it pays exactly in the close deals where indifference is expensive.

Roughly 20 lines in the play path (`apps/api/ai_play._ai_play_pick`, alongside
`_mix_equivalent`), plus one arm in `research/experiments/47_overnight/ablate.py`.

Expect **less** than the naive `48% x 5.8 points`: PIMC's averaging probably already takes
points by accident in many of these spots. The gate is the arbiter, and it is built.

## Secondary value

This also makes the AI's defence *look* right to a human. milan spotted both cases
immediately from the table; an engine people actually play against is judged on whether
its moves read as sane, not only on its GP.


---

# Measurement (2026-08-03, `measure.py`, exp47 natural corpus, 6,936 deals)

## Both sides do it, and it is mostly free

| | |
|---|---|
| positions/deal where the mover held BOTH a 10-pt and a 0-pt legal card | **13.5** |
| points/deal fed to the OPPONENT when a 0-pt card was legal | **8.6** |
| — by the defenders | 6.0 |
| — by the soloist | 2.6 |

Then 500 of those positions were solved move-by-move, asking whether a zero-point card was
**also god-optimal**:

| | n | a 0-pt card was also optimal |
|---|---|---|
| all | 500 | **84.4%** |
| soloist | 169 | 83.4% |
| defenders | 331 | 84.9% |

Spread across tricks 1-9 (rate 77-90% at every trick, so this is not an endgame artefact).
**~7.3 points per deal are given away for nothing.**

## Root cause: the objective cannot see points

`n_legal` averages 4.6 and `n_optimal` averages 4.1 — **81% of these positions have EVERY
legal move tied.** The solver maximises a weighted sum of BINARY indicators
(`parti(+-1)`, `ulti(0/1)`, `reaches_100(0/1)`); card points enter only through the parti
flag, which is usually settled early. So the solver genuinely has nothing to say, the
choice falls through to arbitrary tie-breaking, and points leak.

Verified the solver is not simply returning a constant: a trick-1 `solve_all` takes 12 ms
and separates 10 moves into 2 distinct values. It discriminates — there is just nothing to
discriminate 81% of the time.

## Recommendation

Add a **schmier tie-break** over the god-optimal set only:

    among moves the solver rates equally:
        opponent currently winning the trick -> play the LOWEST-point card
        your own side winning it             -> play the HIGHEST-point card

* **Free by construction.** Restricted to the optimal set, so the god value is unchanged —
  the same guarantee that makes the anti-tell mixer safe
  (`tests/ulti/test_block_equivalence.py`). It cannot regress.
* **Serves both roles**, since soloist and defenders leak at the same rate (83% vs 85%).
* Sits next to `_mix_equivalent` in `apps/api/ai_play._ai_play_pick`; the mixer must run
  AFTER it (mixing inside an equivalence block is still free, but it must not undo the
  schmier choice — so exclude point-differing cards from the mix, or apply the tie-break
  last).

### Expected size, honestly

Points convert to GP only through the parti flag. 27.2% of deals are decided by a single
10-point card, and a seat that stops leaking saves roughly 3 points/deal against opponents
that still leak. That is order **0.1-0.2 GP/seat-deal** — real, but below the ~0.6 GP
resolution of a 400-deal gate. **Gate it at n>=1500**, or the result will be another null
that means nothing.

### Why it is worth doing anyway

It is the only change found in two days that is *provably* free. Every other candidate
(pre-pickup model, kontra thresholds, FLOOR/DEBIAS) traded something. And it makes the
engine's play read as sane to a human, which milan spotted unaided in two consecutive
games — an engine people actually play against is judged on whether its moves look right,
not only on its GP.


---

# Step 2 — the counterfactual kills it (2026-08-03, `counterfactual.py`)

milan's objection: "sometimes you want to keep the highest card for later — banking is a
real decision, not a rule." Correct, and the truth is stronger than the objection.

**Method.** For each position where the AI fed a ten AND a zero-point card was tied-optimal,
replay the REAL deal from that ply twice — once with the card played, once with the tied
zero — continuing both with the deployed stack under the same seed, and score both with the
oracle. Only that one card differs.

| | n | change in soloist GP | unchanged |
|---|---|---|---|
| defender banks | 278 | **+0.000 ± 0.223** | **95%** |
| soloist banks | 122 | −0.082 ± 0.540 | 78% |

**Zero. In 95% of defender positions the deal came out bit-identical.**

The mechanism: the ten does not disappear when you decline to feed it. You keep it, you
play it a few tricks later, and it is usually captured by the same side regardless. The
82%-of-positions-are-tied result from step 1 was not a signal that value was being left on
the table — it was the solver correctly reporting that **these choices do not matter**.

## The methodological lesson

Every step of the original chain was measured correctly:

* 8.6 points/deal fed to the opponent when a zero was legal — true
* 84.4% of those had a tied zero-point alternative — true
* therefore a free GP win — **false**

Both intermediates are PROXIES. The outcome — GP after the hand is played on — was never
measured until milan pushed back, and it is flat. Recorded here because the same mistake
appeared three times on 2026-08-03: a truncated sweep table, a per-seat gate figure without
a matched control, and this.

## What survives

The exploit-rollout idea (milan's, and the right instrument) is NOT refuted — it is simply
not applicable here, because there is no value hiding among tied moves to find. It keeps
its original purpose: defender-side exploitation of a FALLIBLE opponent, where the value
comes from the opponent's mistakes rather than from the card choice. That is exp49.
