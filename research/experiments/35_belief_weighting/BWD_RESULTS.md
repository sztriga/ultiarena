# exp35 — BELIEF-WEIGHTED DETERMINIZATION (defender) vs uniform PIMC

N=334 played · BELIEF_EPS=0.15 · WINDOW=3 · NW=16. Metric = soloist GP (LOWER = better defense). diff = BWD − uniform → NEGATIVE = BWD defense held the soloist to fewer points.

- soloist GP/deal:  uniform-def +11.754   BWD-def +12.222
- **diff (BWD − uniform) = +0.467 GP/deal  (t=+0.7)**  [negative = BWD BETTER]
- decisions differ on 58 deals (17%); when they differ mean +2.690
    · dd-makeable (soloist favored — infer to break) n=302  diff +0.033 (t=+0.0)
    · dd-LOST (defenders favored)                    n=32   diff +4.562 (t=+1.6)
  by contract: piros ulti=+0.06(n134)  ulti=-0.05(n112)  piros parti=-0.75(n16)  teritett rebetli=+11.43(n14)  piros ulti-40-100=-1.33(n12)  piros 40-100=+3.20(n10)  ulti-40-100=-4.22(n9)
- BWD thinking time: 1732 ms/move
