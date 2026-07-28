# exp27 TOURNAMENT — current frontier (deployed kontra) vs candidate (per-unit)

Held-out test N=4000 deals. Candidate parti τ*=0.02 (tuned on train).
Soloist GP is per-deal to the soloist; defender GP is the pair's total (=-soloist). Kontra doesn't change play, so this isolates the kontra/rekontra decisions.

## A. Self-play (all 3 seats one brain) — soloist GP/deal (lower = better defense)
- all-deployed  soloist +6.689  → defenders -6.689
- all-candidate soloist -1.039  → defenders +1.039
- candidate defenders concede +7.728 GP/deal LESS than deployed defenders

## B. Head-to-head (candidate vs current frontier, per table)
- **candidate wins +7.737 GP/deal** head-to-head (decisions differ on 51% of deals)

## C. Defender kontra only (rekontra = deployed for both) — the isolated defender fix
- candidate defenders gain +7.676 GP/deal vs deployed defenders (same soloist)

## C2. Rekontra rule (candidate defenders fixed) — soloist GP/deal (HIGHER = better for soloist)
- deployed-rekontra: soloist -0.987
- candidate-rekontra: soloist -1.039
- never-rekontra: soloist -1.185

## D. By contract — soloist GP/deal, deployed vs candidate self-play
| contract | n | deployed sol GP | candidate sol GP | defender gain |
|---|---|---|---|---|
| piros parti | 2740 | -4.46 | -5.62 | +1.16 |
| piros ulti | 873 | +37.58 | +7.07 | +30.52 |
| piros 40-100 | 126 | +9.24 | +9.24 | +0.00 |
| teritett rebetli | 94 | +56.17 | +56.17 | +0.00 |
| teritett colorless duri | 61 | -5.51 | -5.51 | +0.00 |
| piros ulti-40-100 | 34 | +18.94 | +18.94 | +0.00 |
| piros 20-100 | 20 | +7.00 | +7.00 | +0.00 |
| piros teritett 20-100-duri | 15 | -48.00 | -96.00 | +48.00 |
| piros teritett 40-100-duri ≡ piros teritett ulti-duri | 14 | -21.71 | -45.71 | +24.00 |
| ulti | 9 | +10.89 | +2.22 | +8.67 |
| piros ulti-20-100 | 6 | +0.00 | +0.00 | +0.00 |
| piros teritett ulti-40-100-duri | 6 | +26.67 | +18.67 | +8.00 |
