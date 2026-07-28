# exp 29 — frontier self-play table (2026-07-21)

## Goal (milan)
3 frontier models sit down, bid + play over a long run. Table of contract frequency +
avg soloist GP (for the soloist who played it), passes included, per-seat analytics,
"where are we bleeding?".

## Faithfulness (matches the DEPLOYED engine)
- KONTRA=1 kontra-aware bidder (opener PASSES weak hands) — critical: harness27/exp27 data
  used KONTRA=0 (bidder default) so it NEVER passed → not faithful. exp29 sets KONTRA=1.
- FULL auction (any seat may open after a forehand pass) — reimplemented faithfully from
  play.py::_advance_auction; run_auction (exp24) only models forehand-opens, so its per-seat
  distribution is wrong (seat 0 almost always soloist). Full auction lets P1/P2 win bids.
- PIMC play both sides (harness27._play_terminal), promoted per-unit kontra (ulti trumps>=4,
  duri trumps>=3, parti makeability<0.10, else abstain; rekontra unchanged), oracle + silents.
- Seat 0 = forehand/opener (fixed frame → positional analytics). Files: frontier_selfplay.py
  (build), analyze.py (TABLE.md), selfplay.jsonl.

## Early signal (N=216 partial — FIRM numbers pending N=6000)
- passz ~38% (opener passes weak hands); piros parti ~31%, piros ulti ~21%, rest rare.
- **piros parti avg soloist GP −1.35** (floor tax; made 43%, kontra'd 87% by defenders) —
  negative but RATIONAL vs the −4 passz penalty (bidding the floor beats paying).
- piros ulti +11.8, 40-100 +11.4, terített/combined big ± (rare).
- **P0 forehand mean −1.5 GP/deal** (structural: pays −4 on the 38% passz + bids marginal
  floor + loses as defender to P1/P2 strong bids). Note: exp25's KONTRA→P0 +0.91 was on
  run_auction (forehand-opens); the FULL auction is harsher on P0 (loses as defender when it
  passes into a P1/P2 bid). Over a ROTATING match each seat is forehand 1/3 → nets out.

## Hypothesis to check (potential leak)
The KONTRA=1 pass threshold uses kontra_adjusted_ev (assumes OPTIMAL god defender kontra), but
the deployed defenders now kontra LENIENTLY (post-exp27). So the bidder may OVER-PASS marginal
piros-parti hands that would realize better than −4 vs the lenient defenders. Follow-up: force
a piros parti bid on the passz hands, play out, compare realized GP vs −4.

## RESULTS (N=6000, TABLE.md)
- Distribution: passz 32.6%, piros parti 32.5% (+2.30 soloist GP, made 52%, kontra'd 80%),
  piros ulti 26.2% (+10.38, made 88%), piros 40-100 3.3% (+10.65), terített rebetli 2.2%
  (+59.4!), rest <1.5%. Played 67%, soloist made 69% overall, avg 1.04 bids (only 4% contested).
- PER-SEAT (position): P0 forehand **−1.03**, P1 middle **+1.04**, P2 rear −0.01 (zero-sum;
  rotates out over a match). P0 tax = pays −4 on the 33% passz (forced-opener structural tax).
  P1 (middle) is the best seat. NB the FULL auction is harsher on P0 than exp25's run_auction
  (+0.91) because P0 loses as a defender when it passes into a P1/P2 bid.
- **BLEEDING = the DURI family**: duri-family contracts (88 deals, 1.5%) made only 35%, avg
  soloist **−7.64**, total **−0.13 GP/deal**. teritett colorless duri (40 deals, −4.80),
  piros teritett 40-100-duri (24, −14.0), 20-100-duri (14, −18.3). The frontier OVER-BIDS
  terített duri → soloist loses. MECHANISM: the colorless_duri/duri heads are the WORST-
  calibrated (exp28: ~400 positives/1M, overconfident) → over-bid. FIXABLE (raise FLOOR/
  threshold for duri contracts, or recalibrate the duri heads). Small aggregate but a genuine leak.

## Over-pass test RUNNING (force_parti.py on the 1957 passz hands)
Does forcing piros parti beat the −4 passz? piros parti averages +2.30 over ALL bids, but the
passz hands are the WEAKEST — testing whether they still beat −4 vs the lenient defenders.

## OVER-PASS TEST — RESULT: no leak, passing is correct
Forcing piros parti on the 1957 passz hands realized −7.89 GP vs the −4 passz (−3.89 WORSE;
beats passing on only 24%; those hands make parti just 20%). So the KONTRA=1 conservatism is
CORRECT; the forehand tax is purely structural (rotates out), not fixable by bidding more.

## LOOP COMPLETE (SUMMARY.md)
Frontier healthy on the bulk (piros parti +2.30, piros ulti +10.38, rebetli +59). ONE fixable
leak = over-bidding terített DURI (−0.13 GP/deal, made 35%), traced to the miscalibrated
colorless_duri/duri heads (exp28). Forehand deficit structural + correct. Nothing changed in the
deployed engine. No background jobs running.

## Progress
- 2026-07-21: DONE. Table + per-seat + bleed audit + over-pass test. SUMMARY written.
