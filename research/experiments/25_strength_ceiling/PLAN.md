# Exp 25 — what drives agent strength (ceiling-first attribution)

Going all out on the question: **what makes the agent strong, and can retraining
the perception nets help?** The trap is to retrain nets, see a wash, and not know
if the net was bad or if perception just doesn't matter. So we measure the CEILING
of each component *before* spending training compute.

## The idea
Give the agent a **perfect** version of one component and measure the strength
jump (h2h vs the current agent). That jump is the UPPER BOUND on what improving
that component can buy — cheap, no training. Then invest only where the ceiling
shows headroom.

Two perception ceilings bracket the net question:
- **god-provider** (perfect INFO — sees all hands) → absolute ceiling of bidding.
- **PIMC-provider** (perfect MARGINAL — best estimate from own hand, N→∞) → the
  *trainable* ceiling (a net can only ever approximate this marginal).
- `net → PIMC` = **trainable headroom** (what better nets can buy).
  `PIMC → god` = **irreducible info gap** (can't be trained away).

## Phases
**Phase 1 — component ceilings (no training).**
- [running] **god-provider perception ceiling** — `god_provider.py`, `ceiling.py`
  (h2h god-agent vs net-agent, same composer/auction/play). → the perception gap.
- PIMC-provider ceiling (splits trainable vs info gap).
- Perfect calibration / perfect kontra decision / perfect play (god vs PIMC) — one
  swap each, same h2h.

**Phase 2 — attribution.** Decompose the full gap (current → all-oracle) across
components (Shapley-style over swap order): "X% perception, Y% calibration, Z%
kontra, W% play." The map of where strength lives.

**Phase 3 — invest where the headroom is.** Only the flagged components get the
multi-hour compute. If perception wins: architecture (factorized heads vs joint vs
structural vs deeper) × labels (god vs PIMC vs self-play) × data — each scored
h2h vs the reigning champion, per-contract attribution.

## Metric
h2h vs champion, **god (fast inner) + PIMC (realistic) + per-contract**. Fixed seed
bank. See `../24_bidding_loop/KNOBS.md` for the full tweak surface each phase feeds.
