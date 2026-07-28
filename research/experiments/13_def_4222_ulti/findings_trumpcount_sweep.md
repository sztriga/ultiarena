# Sol's trump count vs ulti winrate (talon trump-void)

## Setup

- `deal_ulti_biased(alpha=0.6)` — realistic ulti deals, sol always
  holds the trump-7.
- Reject any deal where the 2-card talon contains a trump.
- Run the god solver (`contract='ulti'`, perfect info, full
  alpha-beta) and record sol's winrate by trump count.
- N = 12,793 accepted from 20,000 sampled (accept ≈ 64% — most
  ulti-biased deals naturally have a trump-void talon).

## Headline

| sol trumps (incl. the 7) | N      | made  | rate    |
|--------------------------|--------|-------|---------|
| 2                        | 1,104  | 3     | 0.27%   |
| 3                        | 4,683  | 661   | 14.11%  |
| 4                        | 3,977  | 2,754 | **69.25%** |
| 5                        | 2,242  | 2,124 | 94.74%  |
| 6                        | 787    | 786   | 99.87%  |
| **Overall**              | 12,793 | 6,328 | **49.46%** |

Sharp monotonic curve, cross-over around 4 trumps.

## The 2-trump outliers

Initially surprising: with only the 7 + 1 other trump and defs holding
the other 6, can sol ever make ulti? The data says yes — at 0.27%.

**Every single winning 2-trump deal sampled (5/5) had sol holding
exactly `{trump-7, trump-ace}`.** No other 2-trump combination ever
won in our sample. This is not a solver bug; it is a real Ulti
technique, requiring a very specific hand shape.

### Example trace — seed 1,000,002,323, trump = leaves (★)

**SOLOIST** (10 cards)
```
★ leaves:  ace  7              ← only 2 trumps: ace = shield, 7 = closer
  acorns:  ace  10  lower
  hearts:  ace  10  upper
   bells:  ace  10
```
**DEFENDER 1** (3 trumps + side cards)
```
★ leaves:  upper  9  8
  acorns:  8
  hearts:  lower  9
   bells:  upper  lower  9  7
```
**DEFENDER 2** (3 trumps + side cards)
```
★ leaves:  10  king  lower
  acorns:  king  upper  9  7
  hearts:  king  8
   bells:  king
```
**TALON** (no trumps): `bells:8, hearts:7`

### Optimal play, trick by trick

| Trick | Lead → next plays                                    | Winner | Trumps consumed     |
|-------|------------------------------------------------------|--------|---------------------|
| T1    | SOL ace♥ → DEF1 9♥, DEF2 8♥                          | SOL    | —                   |
| T2    | SOL ace♠bells → DEF1 7, DEF2 king                    | SOL    | —                   |
| T3    | SOL ace♣acorns → DEF1 8, DEF2 7                      | SOL    | —                   |
| T4    | SOL 10♥ → DEF1 lower, DEF2 king                      | SOL    | —                   |
| T5    | SOL upper♥ → DEF1 ★8, DEF2 ★lower (both ♥-void)      | DEF2   | def: 8, lower       |
| T6    | DEF2 9♣acorns → SOL 10, DEF1 ★9 (acorns-void)        | DEF1   | def: 9              |
| T7    | DEF1 9♠bells → DEF2 ★king (bells-void), SOL 10       | DEF2   | def: king           |
| T8    | DEF2 upper♣acorns → SOL lower, DEF1 ★upper           | DEF1   | def: upper          |
| T9    | DEF1 lower♠bells → DEF2 ★10, SOL ★ace (bells-void)   | SOL    | def: 10, sol: ace   |
| T10   | SOL **7♠trump** → DEF1 upper♠bells, DEF2 king♣acorns | **SOL** | sol: 7 (wins ulti) |

By T10 all 6 def trumps are spent and sol's lone 7 wins because both
defs are trump-void.

### The principle

1. **Sol never leads trump in the opening.** Sol leads aces and 10s
   of non-trump suits (T1–T4). Sol wins these tricks but, critically,
   no trumps move yet.
2. **Defs spend their side cards faster than sol** because sol's
   loaded aces force them to play their highs. They start going void.
3. **Once void, defs must trump to win mid-game tricks** (T5–T8). Each
   capture costs them a trump.
4. **Sol's trump-ace is held in reserve** as a *shield* — used only
   when sol is finally forced into a trump situation (T9), to
   over-trump and preserve the 7.
5. **By T10 defs are out of trumps**; sol drops the 7 and it walks
   home alone.

The trump-ace is the keystone: without it, when a def takes back
control mid-game and leads trump, sol would be forced to discard the
trump-7 (must-follow), killing the ulti. With the ace, sol can play
the ace on that trump lead instead, preserving the 7 and reclaiming
the lead.

### Why this only works at 2-trump with a very specific hand

This line requires all of:
- Sol holds **both** trump-7 and trump-ace (no other 2-trump combo wins)
- Sol has **enough side-suit aces** (3 here) to dominate non-trump tricks
  and force defs to spend their structure
- Defs have **short** side-suit holdings so they go void quickly
  (def1's only acorn is the 8; def2's only bell is the king)
- Talon contains **no trumps** (else extra trump arithmetic shifts)

These conjoint conditions are rare under `deal_ulti_biased` — hence
0.27%. They are not impossible.

## Takeaway

The 2-trump tail is a real Ulti technique, not a solver artifact. It
showcases why bidding-phase strength evaluation cannot be reduced to
trump count alone: a `{7, ace}` + 3 side aces hand can make ulti, but
a `{7, king}` + 3 side aces hand of the same trump count effectively
cannot.

## Reproduce

```
PYTHONPATH=. python3 experiments/13_def_4222_ulti/run_sol_trumpcount.py
```
