
# realistic-makeability on ULTI (test n=198)
- never  : +3.889
- oracle : -0.455   (gain +4.343)

- god  τ*=0.02: test GP +3.505  gain +0.384  fire 13%
- real τ*=0.02: test GP +3.465  gain +0.424  fire 7%

real-signal calibration (does it track actual make?):
| bin | n | mean real-sig | actual make% |
|---|---|---|---|
| [0.0,0.2) | 130 | 0.09 | 67% |
| [0.2,0.4) | 212 | 0.28 | 87% |
| [0.4,0.6) | 63 | 0.42 | 92% |
