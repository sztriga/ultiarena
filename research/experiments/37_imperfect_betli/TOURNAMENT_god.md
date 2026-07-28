# exp37 — imperfect/bluff betli bidding tournament  (DEF=god)

1500 deals × 3 seats = 4500 seat-games. R = betli_real ON, C = deployed baseline (same frontier config otherwise). diff = GP(seat=R, opp=C) − GP(seat=C, all-C table).

## Headline
- **mean diff = -0.1580 GP/seat-deal   (t=-2.4, n=4500)**  → the GP gained by adopting betli_real vs the deployed baseline
- auction diverged (R bid a betli C wouldn't, or plain-vs-terített) on 197/4500 seat-deals (4.4%)
- GP actually changed on 191 seat-deals; mean diff there = -3.723

## The newly-unlocked betlis (R bid plain betli, C didn't): n=107
- realised make-rate 21% (22/107 made)
- mean diff on those = -6.430 GP/seat-deal
- what C bid instead: ulti=49  pass=18  piros ulti=17  piros parti=7  20-100=3  piros 40-100=2
