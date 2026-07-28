# exp40 — realistic recalibration GP gate: does swapping god→realistic heads win games?

FRONTIER (current champion: deployed god heads + betli_real + rebetli) vs RECAL (same, with these heads swapped for exp40 realistic heads: **duri_colored, colorless_duri, ulti**). Identical corrected engine; soloist PIMC. Any delta = the recalibration alone.

## (1) GP GATE — RECAL vs FRONTIER head-to-head
- **RECAL (lone) vs 2×FRONTIER: -1.7117 GP/game (±0.348, n=8383; 617 redealt) → SIGNIFICANT −** — 0 = tie, + = recalibration helps.

## (2) Contract mix — FRONTIER vs RECAL self-play (does it risk duri/ulti more?)
| contract | FRONTIER %·GP | RECAL %·GP |
|---|---|---|
| piros ulti | 35.1% · +10.43 | 49.3% · +3.82 |
| ulti | 27.4% · +6.68 | 24.4% · +1.39 |
| rebetli | 8.5% · +6.39 | 7.7% · +7.90 |
| piros parti | 5.5% · +2.89 | 1.5% · +3.23 |
| piros ulti-40-100 | 3.0% · +29.20 | 2.4% · +26.38 |
| betli | 3.2% · +1.86 | 1.9% · +2.13 |
| piros 40-100 | 3.3% · +11.19 | 1.8% · +12.46 |
| ulti-40-100 | 2.3% · +13.19 | 1.6% · +13.56 |
| teritett betli | 2.2% · +15.26 | 1.5% · +17.46 |
| piros 20-100 | 1.3% · +18.73 | 0.6% · +15.85 |
| teritett duri | 1.0% · +10.91 | 0.6% · -1.55 |
| teritett ulti-duri | 0.6% · -5.43 | 0.7% · -4.11 |
| ulti-40-100-duri | 0.7% · +8.90 | 0.6% · +10.90 |
| 20-100-duri | 0.8% · +5.37 | 0.5% · +7.20 |
| piros 20-100-duri | 0.6% · +16.59 | 0.6% · +6.19 |
| piros ulti-duri | 0.4% · +20.00 | 0.7% · +5.26 |
| 40-100 | 0.7% · +7.35 | 0.3% · +7.25 |
| teritett 40-100-duri | 0.4% · -4.00 | 0.6% · -12.89 |
| piros 40-100-duri | 0.2% · +2.67 | 0.7% · +0.23 |
| ulti-duri | 0.4% · +4.94 | 0.4% · +6.11 |
| piros ulti-40-100-duri | 0.3% · +6.77 | 0.3% · +11.43 |
| 20-100 | 0.4% · +2.78 | 0.1% · +5.33 |
| piros ulti-20-100 | 0.3% · +3.71 | 0.1% · +14.67 |
| ulti-20-100 | 0.3% · +19.00 | 0.1% · +28.80 |
| ulti-20-100-duri | 0.2% · +29.45 | 0.1% · +31.20 |
| piros teritett 40-100-duri | 0.1% · +6.40 | 0.2% · -21.60 |
| 40-100-duri | 0.2% · -5.78 | 0.1% · -4.00 |
| teritett ulti-40-100-duri | 0.2% · +2.67 | 0.1% · +4.00 |
| duri | 0.1% · -6.00 | 0.1% · -12.00 |
| piros teritett ulti-duri | 0.1% · +32.00 | 0.1% · -4.57 |
| teritett 20-100-duri | 0.2% · -10.29 | 0.0% · -24.00 |
| piros ulti-20-100-duri | 0.1% · +37.33 | 0.1% · +24.00 |
| teritett ulti-20-100-duri | 0.0% · +0.00 | 0.1% · +8.00 |

**family shares (any rung containing the word):**
- duri: FRONTIER 6.5% → RECAL 6.8%
- ulti: FRONTIER 71.3% → RECAL 81.1%
- betli: FRONTIER 13.9% → RECAL 11.1%
- parti: FRONTIER 5.5% → RECAL 1.5%
- 40-100: FRONTIER 11.3% → RECAL 8.7%
- passz: FRONTIER 9.8% → RECAL 2.7%

## (3) Negative contracts — FRONTIER vs RECAL (did the 40-100-duri leak close? new leaks?)

### FRONTIER
| contract | n | GP/bid | 95% CI | |mean|/SE | verdict |
|---|---|---|---|---|---|
| teritett 20-100-duri | 7 | -10.29 | ±19.9 | 1.0 | leaning − |
| duri | 4 | -6.00 | ±11.8 | 1.0 | leaning − |
| 40-100-duri | 9 | -5.78 | ±3.5 | 3.2 | **real −EV** |
| teritett ulti-duri | 28 | -5.43 | ±9.7 | 1.1 | leaning − |
| teritett 40-100-duri | 19 | -4.00 | ±10.1 | 0.8 | noise |
- **FRONTIER: 1 significantly −EV: 40-100-duri**

### RECAL
| contract | n | GP/bid | 95% CI | |mean|/SE | verdict |
|---|---|---|---|---|---|
| piros teritett 40-100-duri | 10 | -21.60 | ±29.2 | 1.4 | leaning − |
| teritett 40-100-duri | 27 | -12.89 | ±6.4 | 3.9 | **real −EV** |
| duri | 7 | -12.00 | ±0.0 | 0.0 | noise |
| piros teritett ulti-duri | 7 | -4.57 | ±34.7 | 0.3 | noise |
| teritett ulti-duri | 35 | -4.11 | ±8.8 | 0.9 | noise |
| 40-100-duri | 6 | -4.00 | ±0.0 | 0.0 | noise |
| teritett duri | 31 | -1.55 | ±17.2 | 0.2 | noise |
- **RECAL: 1 significantly −EV: teritett 40-100-duri**
