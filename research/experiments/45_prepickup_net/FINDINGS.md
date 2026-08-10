# exp45 — a pre-pickup model

**Status: SHELVED, not promoted, cause NOT established.** The model is near-oracle on its
own target (R²=0.972) and cuts the blind rule's 33.9% miss rate to 1.8%, and it still loses
GP at a real table (−1.31 ± 0.56). The leading explanation — a mispriced opening threshold
at the non-forehand seats — was tested directly and came back +0.12 ± 0.32, i.e. not
confirmed. The remaining suspect is the training target itself (below). Do not promote.

## Why

Ulti's auction contains two decisions with two information sets:

| decision | sees | asks |
|---|---|---|
| PICKUP | your own 10 cards | is this worth committing to? |
| ANNOUNCE | the real 12 | which game, and what do I bury? |

Until 2026-08-02 the bidder answered both from the same stack of heads — and worse, it
answered the first while *looking at the talon*, which is the cheat this whole day started
with. Closing the cheat (`ulti/bidding/auction.py`, guard in
`tests/ulti/test_information_boundaries.py`) exposed the second problem: the announce-stage
heads are the wrong instrument for the pickup question, because they price a 10-card hand
as if it were final. **Picking up is worth +3.08 GP on average** — you keep the best 10 of
12 — and none of that uplift is in the blind number.

## Target

For a hand H and a standing rung R, `y(H,R) = E_talon[ EV of the game you announce after
picking up ]`, computed by running the real announce-stage search over 16 sampled talons,
including the commitment fallback (having picked up, you must announce something even if
the confidence floor would rather you had not). Across-talon sd ≈ 1.4–3.4 GP, so 16 talons
pin y to roughly ±0.4. 6,000 examples, no solver — pure net arithmetic, ~0.7 s each.

Features are the blind head outputs themselves: seven probabilities per candidate trump,
the blind best EV, marriage gates, hand shape. The model can therefore only *recalibrate*
information the blind hand already contains — it cannot invent any. `assert_blind()` proves
the featuriser is a pure function of the 10 cards and the public rung.

## Fit

| candidate | MAE | R² |
|---|---|---|
| raw (blind EV, what ships today) | 3.09 | 0.524 |
| offset (blind EV + one constant) | 1.93 | 0.811 |
| **model** | **0.65** | **0.972** |

## Decision quality — the number that matters

Out-of-fold, against the threshold a perfect predictor would use.

**Opening (threshold −2), n=3010**

| candidate | picks up | accuracy | missed | false | regret GP |
|---|---|---|---|---|---|
| raw | 32.7% | 66.1% | **33.9%** | 0.0% | 0.789 |
| offset | 100.0% | 66.5% | 0.0% | 33.5% | 0.357 |
| **model** | 68.8% | **94.2%** | 1.8% | 4.1% | **0.028** |
| oracle | 66.5% | 100% | — | — | 0.000 |

**Overcall, n=2990**

| candidate | picks up | accuracy | missed | false | regret GP |
|---|---|---|---|---|---|
| raw | 9.9% | 85.9% | **14.1%** | 0.0% | 0.375 |
| offset | 19.4% | 87.1% | 8.7% | 4.1% | 0.250 |
| **model** | 23.7% | **95.3%** | 2.5% | 2.2% | **0.045** |
| oracle | 24.0% | 100% | — | — | 0.000 |

Two things fall out. The blind rule's error is **entirely one-sided** — 0.0% false alarms,
33.9% misses. It never bids a hand it shouldn't; it passes a third of the hands it should
play. That is the over-passing, quantified, and it explains the quiet auction (11%
contested against exp29's 44%): overcalling requires clearing a standing bid from ten
cards, which the uncorrected number almost never manages.

And a single constant is not enough. The offset fixes the *mean* and promptly picks up
100% of openings — the uplift is not constant, it depends on the hand.

## The gate says no

Rotation: candidate in each of the three seats, incumbent in the other two. Ulti is
zero-sum, so the candidate's mean GP per seat-deal is the signal and identical configs
cancel to a literal 0.000 (verified).

| arm | n (deals) | GP/seat-deal | t |
|---|---|---|---|
| model on BOTH decisions | 328 | **−1.31 ± 0.56** | −2.36 |
| model on OPENING only | 600 | −0.33 ± 0.36 | −0.91 |
| control (incumbent vs itself) | 600 | **+0.000** | — |

Control per-seat baseline (n=600): seat 0 **+0.933**, seat 1 −0.532, seat 2 −0.402.

Driving both decisions loses outright. Restricting to openings brings the aggregate to a
wash, which is not a rescue — it is the loss diluted by doing nothing most of the time.

### Reading per-seat requires the control, and I got this wrong first

A gate arm's per-seat mean is NOT a delta: it contains the positional baseline, which is
large and noisy at these sample sizes (this seed set at n=600: +0.933/−0.532/−0.402;
nothing like exp44's 6000-deal −0.783/+0.564/+0.220, and it moved by a full GP between
n=316 and n=600). Subtracting a control run on the SAME seeds is mandatory. Sanity check
that the subtraction is right: in the threshold arm seat 0 is configured identically to the
incumbent, and its delta comes out **exactly +0.000**.

Deltas vs control:

| arm | seat 0 | seat 1 | seat 2 |
|---|---|---|---|
| model BOTH | +1.01 | **−2.75** | −2.19 |
| model OPENING only | +1.15 | **−1.37** | −0.76 |

The model is mildly positive at the forehand and clearly negative behind it. (An earlier
reading of "+2 at the forehand" was the positional baseline, not a delta.)

## The threshold hypothesis — tested, not confirmed

The seat asymmetry suggested the auction's economics. Every seat opens against −2/def, but
over exp44's 6,000 deals only the forehand actually forfeits it:

```
passz pays:      seat 0  −4.00    seat 1  +2.00    seat 2  +2.00
defending costs: seat 0  −4.60    seat 1  −5.08    seat 2  −5.09
```

For seats 1 and 2 the alternative to opening is +2 (deal dies, forehand pays) or about −5
(someone else opens) — never −2. Tested directly, no model on either side, varying ONLY the
non-forehand opening threshold from −2 to 0:

| | overall | seat 0 | seat 1 | seat 2 |
|---|---|---|---|---|
| threshold −2 → 0 | **+0.12 ± 0.32** | +0.000 | +0.14 | +0.21 |

Positive in the predicted direction at exactly the two seats it touches, and comfortably
inside the noise. **Not confirmed.** It does not account for the model's loss, and on this
evidence it is not worth shipping on its own either. A wider sweep (+2, and the −5 side)
would settle it; this run only tested one value.

## The catch

Model quality on its own target does not imply GP. `y` is the **announced contract's model
EV**, not realised GP, so a bidder that picks up whenever `y > threshold` inherits whatever
optimism the announce-stage EV already carries. The blind rule's pessimism may have been
silently offsetting it — exactly the displacement failure exp40 documented, where fixing a
calibration in isolation lost because it had been compensating for something else.

With the threshold arm coming back inside noise, this is now the leading explanation. The
model is fit to `E[announce-stage EV]`, and the gate scores realised GP; a bidder that picks
up whenever the predicted EV clears the bar inherits every bit of optimism that EV already
carries. The blind rule's pessimism was plausibly offsetting it — the exp40 displacement
pattern, where correcting one calibration loses because it had been compensating for
another.

**The next experiment is therefore the expensive one:** retarget on
`E[realised GP after pickup]`, which needs a play-out per sample rather than net arithmetic
— roughly 100× this datagen budget (6,000 examples took ~50 min of net evals; the same
count with play-outs is days). Before spending that, it is worth confirming the direction
cheaply on a few hundred hands: sample talons, play the announced contract out with the
deployed stack, and check whether `E[realised GP] − E[announce EV]` is systematically
negative and hand-dependent. If it is flat, the target is not the problem and the search
moves elsewhere.
