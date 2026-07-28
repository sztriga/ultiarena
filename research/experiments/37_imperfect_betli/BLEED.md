# exp37 — why the higher betli rungs bleed

## PART A — head calibration vs god  (n=25000 uniform deals, argmax-discard)
- overall: net_p(argmax) mean 0.028 · pctl_p(0.85) mean 0.004 · true god-make 0.013
- **argmax inflation** net_p−god = +0.015 ; DEBIAS'd pctl_p−god = -0.010

### calibration curve (net_p bin → actual god-make rate):
| net_p bin | n | mean net_p | god-make | gap | terit EV/bid = 40·(god−net_p) |
|---|---|---|---|---|---|
| [0.00,0.30) | 24498 | 0.019 | 0.007 | +0.012 | -0.5 |
| [0.30,0.50) | 293 | 0.396 | 0.208 | +0.188 | -7.5 |
| [0.50,0.70) | 177 | 0.607 | 0.429 | +0.178 | -7.1 |
| [0.70,0.85) | 32 | 0.809 | 0.781 | +0.027 | -1.1 |
- where DEBIAS decision pctl_p≥0.5 (terített-bid region): n=4 (0.0%), true god-make 1.000 → terített EV/bid ≈ +15.4, plain-betli EV ≈ +3.9

## PART B — realised bleed in faithful self-play  (n=25000 deals, 456 betli-family bids)
| rung | n | net_p | god-make | realised GP/bid |
|---|---|---|---|---|
| teritett betli | 456 | 0.833 | 0.706 | **+29.65** |
- ALL betli-family bids: mean realised GP/bid **+29.65** (net_p 0.833 vs god-make 0.706)
