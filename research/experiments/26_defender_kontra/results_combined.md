# exp26 — FINAL combined policy (full colored-simple mix, held-out test)

parti threshold τ*=0.06 (tuned on train); ulti gate = defender holds >=4 trumps.
Soloist GP/deal, LOWER = better defense. Δ = defender gain vs deployed.

| policy | soloist GP/deal | fire% | vs never | vs deployed |
|---|---|---|---|---|
| never | -0.096 | 0% | +0.000 | +3.104 |
| deployed | +3.008 | 98% | -3.104 | +0.000 |
| recommended | -1.363 | 56% | +1.267 | +4.371 |
| oracle (ceiling) | -2.375 | 60% | +2.280 | +5.384 |

- parti (test n=2059): never -1.473  deployed -2.434  recommended -2.993 (fire 72%)  → def gain vs deployed +0.559

- ulti (test n=629): never +4.412  deployed +20.824  recommended +3.973 (fire 5%)  → def gain vs deployed +16.851
