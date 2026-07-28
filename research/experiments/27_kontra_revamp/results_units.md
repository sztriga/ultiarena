# exp27 — per-unit kontra decision (held-out test). Soloist per-def GP, LOWER = better defense. Rekontra fixed (soloist god>0.5).


## parti  (n=7238, test 3629, PIMC made 36%)
- never -0.986 | oracle -2.955 (ceiling gain +1.969)
- deployed -1.062 (fire 98%, gain vs never +0.076)
- calibrated rules (test):
    god<0.02                     -1.992  fire 54%  gain +1.006
  → best: god<0.02 (gain vs deployed +0.930)

## ulti  (n=1851, test 936, PIMC made 83%)
- never +3.641 | oracle -0.115 (ceiling gain +3.756)
- deployed +17.637 (fire 94%, gain vs never -13.996)
- calibrated rules (test):
    god<0.00                     +3.641  fire 0%  gain +0.000
    trumps>=4                    +3.145  fire 4%  gain +0.496
    god<0.00 or trumps>=4        +3.145  fire 4%  gain +0.496
  → best: trumps>=4 (gain vs deployed +14.491)

## 40_100  (n=342, test 174, PIMC made 79%)
- never +4.322 | oracle -0.092 (ceiling gain +4.414)
- deployed +4.322 (fire 0%, gain vs never +0.000)
- calibrated rules (test):
    god<0.00                     +4.322  fire 0%  gain +0.000
    trumps>=5                    +4.322  fire 0%  gain +0.000
    god<0.00 or trumps>=5        +4.322  fire 0%  gain +0.000
  → best: god<0.00 (gain vs deployed +0.000)

## 20_100  (n=99, test 43, PIMC made 62%)
- never +0.372 | oracle -14.140 (ceiling gain +14.512)
- deployed +0.372 (fire 0%, gain vs never +0.000)
- calibrated rules (test):
    god<0.00                     +0.372  fire 0%  gain +0.000
    trumps>=5                    +0.000  fire 2%  gain +0.372
    god<0.00 or trumps>=5        +0.000  fire 2%  gain +0.372
  → best: trumps>=5 (gain vs deployed +0.372)

## durchmars  (n=211, test 98, PIMC made 39%)
- never -6.857 | oracle -26.694 (ceiling gain +19.837)
- deployed -6.857 (fire 0%, gain vs never +0.000)
- calibrated rules (test):
    god<0.00                     -6.857  fire 0%  gain +0.000
    trumps>=3                    -12.245  fire 24%  gain +5.388
    god<0.00 or trumps>=3        -12.245  fire 24%  gain +5.388
  → best: trumps>=3 (gain vs deployed +5.388)

## betli  (n=183, test 94, PIMC made 86%)
- never +28.085 | oracle +17.872 (ceiling gain +10.213)
- deployed +28.085 (fire 0%, gain vs never +0.000)
- calibrated rules (test):
    god<0.00                     +28.085  fire 0%  gain +0.000
  → best: god<0.00 (gain vs deployed +0.000)
