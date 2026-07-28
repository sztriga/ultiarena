# exp37 — Imperfect / bluff betli bidding (Tier 2)

**Goal (milan):** the bidder only ever bids *terített* betli, never plain — because it scores betli
by the **double-dummy** make-prob `p_betli`, under which spreading the hand is free (defenders already
assumed omniscient) so terített (4×) always dominates, and a dd-*lost* betli is scored a certain loss
and never bid. But exp36 measured dd-lost betlis are **made ~60% vs realistic (PIMC) defenders**. So a
huge class of *imperfect / bluff* betli that humans love is invisible to the bidder.

**Fix:** feed a **realistic-defense make-probability** into the betli bid EV instead of the god prob.
- plain betli → `p_betli_real` (defenders blind → the ~60% headroom)
- terített betli → `p_betli` (god) or a terített-realistic head (defenders see the hand ≈ dd)

This automatically (a) unlocks plain/marginal betli, (b) makes correct EV *hedge* ("bid the cheap 5p
betli, skip the risky terített 20"), (c) gives exploitative bluffing for free.

## Design (mirror the exp23 base-head pipeline, swap god label → realistic label)
- **Features:** identical to the deployed `betli` head — `vnet.pickup.featurize(hand10, None, False)`
  (32-dim colorless) of the **post-discard 10-card** betli hand the auction featurizes. [confirm via trace]
- **Label:** realistic betli outcome — play the post-discard betli position out with the DEPLOYED engine
  (soloist = PIMC/exploit, defenders = PIMC) → made(1)/stolen(0). Binary (net regresses to make-prob),
  large N (matches how the god head trains on binary god labels).
- **Dealer:** `deal_betli(alpha)` (betli-plausible hands, sample near the decision boundary). Mirror the
  god betli head's dealer/discard if it differs. [confirm]
- **Head:** reuse `train_base_head.Head(32)` + BCE (via `DATA=` override) → `betli_real_baseline.pt`,
  + isotonic calibration on a held-out realistic set.

## Integration (gated, default OFF — deployed engine unchanged)
- `provider.py`: load `betli_real_baseline.pt` if present → add `p_betli_real` to `BaseProbs`.
- `bidder.py`: `betli_real_prob` param (default None) — when set, plain-betli EV uses `p_betli_real`,
  terített stays on `p_betli`. Mirrors how `DURI_TERIT_MULT`/`DEBIAS` were added as per-bidder params.

## Evaluation
1. **Head quality / opportunity:** god `p_betli` vs `p_betli_real` on held-out betli hands — calibration,
   AUC, and the count of hands with `p_god<0.5` but `p_real>0.5` (the missed bluff/imperfect betlis).
2. **Full-game GP tournament (headline):** paired per-deal, baseline bidder vs exp37 bidder, realistic
   play (deployed engine, all seats same bidder). GP/deal delta; betli bid frequency + realized make-rate.
3. **Robustness sweep:** exp37's realized betli GP vs defender strength (PIMC / exp36-net / god) — does
   the bluff edge survive a stronger defender (exploitative vs robust)?

## Files
- `datagen.py`   realistic betli make-prob labels (deal → post-discard hand → play out → made/stolen)
- `train.py`     (or reuse train_base_head with DATA=) → betli_real_baseline.pt + isotonic
- `headcheck.py` Eval 1 (god vs realistic head, opportunity count)
- `tournament.py` Eval 2/3 (paired GP, robustness sweep)
- `RESULTS.md`
