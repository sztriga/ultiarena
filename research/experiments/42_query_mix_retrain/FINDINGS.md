# exp42 — query-mixture retrain of the exp41-convicted heads

**PROMOTED 2026-08-01: duri_colored.** reach100_20 SHELVED (see below).

## The disease (exp41 head audit)
Every head is calibrated on its uniform query distribution and wrong at the argmax —
the auction's argmax hunts each head's over-optimistic tail. duri_colored was the
catastrophe: trained only on 5–8-trump dealer hands, queried on everything;
argmax-selected configs claimed 0.59, god said 0.016 (milan's −80 kontra'd
piros 20-100-duri came from exactly this).

## The fix
Mixture datagen: ~70% query-distribution samples (random deal × random keep-10 ×
random legal trump — what the sweep actually asks) + ~30% strong-dealer (α=1.0)
with SMART discards as positive oversampling (random discards give 0.5–5% positives;
smart give 18–24%). God labels (exp40 verdict: realistic labels lose vs near-perfect
opponents). Class-weighted BCE → the net is the RANKER; isotonic fitted on query
holdout + fresh biased holdout (fixed-edge bins) restores calibration — the
query-only isotonic had 2 positives in 25k and MUTED the head (0.2% ceiling).

## Gates
* two-head candidate (duri+r20): +0.116 ± 0.325 — wash. r20's honest ceiling (0.74)
  under global FLOOR=0.80 mutes plain 20-100, forfeiting bids worth +19.2 solGP.
  LESSON: the audit diagnoses the head; only the tournament judges the system —
  a lying head inside compensating machinery (DEBIAS/FLOOR) can beat an honest head
  the machinery silences.
* duri-only: +0.199 ± 0.330 (n=1800). Exact-zero controls on all non-duri contracts
  (rotation design); whole duri family positive (terített duri +10.7/game — self-play
  soloist −48 → +16 solGP; 20-100-duri +8.4; ulti-duri +8.0). Promoted on the
  exp39/exp27 pattern: targeted mechanism proven + aggregate non-negative.

## Post-promotion
Golden hash UNCHANGED (656879bb) — none of the 24 golden games contains a
duri-relevant decision. 51 tests pass.

## Shelved / next
* reach100_20 candidate is trained + calibrated (candidates/) — needs PER-HEAD FLOOR
  machinery before it can serve (honest 0.74 ceiling vs global 0.80 floor).
* parti/ulti/reach100_40 gaps (~+0.15) appear compensated by DEBIAS — left alone.
* betli god head +0.33 gap matters only for rebetli/terített gates — revisit if
  those leak.
