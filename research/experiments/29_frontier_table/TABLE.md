# Frontier self-play — 6000 deals (3 frontier models bid + play)

KONTRA-aware bidder (opener passes weak hands), full auction (any seat may open), PIMC play, promoted per-unit kontra, oracle scoring incl. silents. Seat 0 = forehand/opener.

## Contracts — frequency & soloist GP (sorted by frequency)

| contract | count | freq | avg soloist GP | made% | kontra% | avg /def | avg bids |
|---|---|---|---|---|---|---|---|
| piros ulti | 2096 | 34.9% | +8.63 | 85% | 4% | +4.31 | 1.66 |
| ulti | 1648 | 27.5% | +5.85 | 90% | 3% | +2.93 | 1.18 |
| piros 40-100 | 220 | 3.7% | +9.87 | 80% | 0% | +4.94 | 1.50 |
| piros parti | 217 | 3.6% | +2.40 | 53% | 81% | +1.20 | 1.00 |
| teritett rebetli | 175 | 2.9% | +58.06 | 86% | 0% | +29.03 | 2.42 |
| teritett 40-100-duri ≡ teritett ulti-duri | 166 | 2.8% | -11.35 | 14% | 0% | -5.67 | 2.47 |
| teritett colorless duri | 139 | 2.3% | -1.04 | 49% | 0% | -0.52 | 2.75 |
| piros ulti-40-100 | 134 | 2.2% | +28.66 | 90% | 0% | +14.33 | 1.75 |
| ulti-40-100 | 133 | 2.2% | +13.14 | 86% | 0% | +6.57 | 1.65 |
| teritett ulti-40-100-duri | 72 | 1.2% | -6.11 | 14% | 0% | -3.06 | 2.71 |
| piros 20-100 | 63 | 1.1% | +3.75 | 56% | 0% | +1.87 | 2.17 |
| piros teritett 40-100-duri ≡ piros teritett ulti-duri | 61 | 1.0% | -8.92 | 26% | 0% | -4.46 | 2.33 |
| teritett 20-100-duri | 60 | 1.0% | -11.87 | 13% | 0% | -5.93 | 2.77 |
| 40-100 | 44 | 0.7% | +3.73 | 73% | 0% | +1.86 | 1.09 |
| piros teritett 20-100-duri | 38 | 0.6% | -14.53 | 26% | 0% | -7.26 | 2.42 |
| piros teritett ulti-40-100-duri | 25 | 0.4% | -0.64 | 20% | 0% | -0.32 | 2.68 |
| ulti-20-100 | 24 | 0.4% | +10.17 | 58% | 0% | +5.08 | 2.71 |
| piros teritett ulti-20-100-duri | 23 | 0.4% | -3.48 | 22% | 0% | -1.74 | 3.17 |
| piros ulti-20-100 | 23 | 0.4% | -0.87 | 30% | 0% | -0.43 | 2.83 |
| 20-100 | 16 | 0.3% | +0.12 | 50% | 0% | +0.06 | 1.81 |
| teritett ulti-20-100-duri | 15 | 0.2% | +9.60 | 27% | 0% | +4.80 | 2.73 |
| **passz** | 608 | 10.1% | — | — | — | — | — |

## Overall
- deals: 6000 | played: 5392 (90%) | passz: 608 (10%)
- soloist made 78% of played contracts
- mean soloist GP across played contracts: +7.89
- auction: avg 1.62 bids/played-deal; 44% were contested (overcalled)

## Per-seat (position) — seat 0 = forehand/opener

| seat | mean GP/deal | won bid (soloist) | GP as soloist | GP as defender |
|---|---|---|---|---|
| P0 forehand | +0.045 | 2425 (40%) | +6.59 | -4.48 |
| P1 middle | +0.595 | 1572 (26%) | +10.02 | -3.51 |
| P2 rear | -0.640 | 1395 (23%) | +7.75 | -3.97 |

- zero-sum check: seat means sum to +0.000 (should be ~0)
- passz: seat 0 is the payer on all 608 passzes (10% of deals), −4 GP each → a structural forehand tax

## Bleeding check

Contracts where the soloist LOSES GP on average (negative avg soloist GP):
- **piros teritett 20-100-duri**: -14.53 GP/deal over 38 deals (FREQUENT — a real leak)
- **teritett 20-100-duri**: -11.87 GP/deal over 60 deals (FREQUENT — a real leak)
- **teritett 40-100-duri ≡ teritett ulti-duri**: -11.35 GP/deal over 166 deals (FREQUENT — a real leak)
- **piros teritett 40-100-duri ≡ piros teritett ulti-duri**: -8.92 GP/deal over 61 deals (FREQUENT — a real leak)
- **teritett ulti-40-100-duri**: -6.11 GP/deal over 72 deals (FREQUENT — a real leak)
- **piros teritett ulti-20-100-duri**: -3.48 GP/deal over 23 deals (rare)
- **teritett colorless duri**: -1.04 GP/deal over 139 deals (FREQUENT — a real leak)
- **piros ulti-20-100**: -0.87 GP/deal over 23 deals (rare)
- **piros teritett ulti-40-100-duri**: -0.64 GP/deal over 25 deals (rare)

- worst positional seat: P2 rear at -0.640 GP/deal
- biggest GP contributors (contract × count × avg): 
    piros ulti: 2096 deals × +8.63 = +18084 total soloist GP
    teritett rebetli: 175 deals × +58.06 = +10160 total soloist GP
    ulti: 1648 deals × +5.85 = +9646 total soloist GP
    piros ulti-40-100: 134 deals × +28.66 = +3840 total soloist GP
    piros 40-100: 220 deals × +9.87 = +2172 total soloist GP
