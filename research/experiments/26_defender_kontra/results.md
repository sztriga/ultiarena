# exp26 defender-kontra — N=5374 colored-simple deals (train 2686 / test 2688)

Soloist GP/deal, reported on the HELD-OUT test half (LOWER = better defense). `never`=no kontra. `oracle`=kontra iff the hand was actually bukott (ceiling). `current`=deployed rule (_sol_ev(sig)<0). `τ*`=kontra iff signal<τ*, τ* tuned on TRAIN, gain measured on TEST.


## all  (test n=2688, soloist made 40%)
- never  : -0.096
- oracle : -2.375   (defender ceiling, gain +2.280)

| signal | current GP | current fire% | curr gain | τ* (train) | τ* test GP | τ* gain | τ* fire% |
|---|---|---|---|---|---|---|---|
| god_n6 | +3.008 | 98% | -3.104 | 0.00 | -0.096 | +0.000 | 0% |
| god_n20 | +3.049 | 99% | -3.145 | 0.02 | -1.034 | +0.938 | 49% |
| god_n40 | +3.068 | 99% | -3.164 | 0.02 | -1.171 | +1.075 | 38% |
| cond_n20 | +2.873 | 98% | -2.968 | 0.06 | -1.137 | +1.041 | 62% |
| cond_n40 | +2.922 | 99% | -3.017 | 0.08 | -1.218 | +1.122 | 63% |

## parti  (test n=2059, soloist made 26%)
- never  : -1.473
- oracle : -3.429   (defender ceiling, gain +1.956)

| signal | current GP | current fire% | curr gain | τ* (train) | τ* test GP | τ* gain | τ* fire% |
|---|---|---|---|---|---|---|---|
| god_n6 | -2.434 | 98% | +0.962 | 0.02 | -2.848 | +1.375 | 82% |
| god_n20 | -2.381 | 99% | +0.908 | 0.12 | -2.852 | +1.379 | 86% |
| god_n40 | -2.356 | 99% | +0.883 | 0.06 | -2.993 | +1.520 | 72% |
| cond_n20 | -2.381 | 99% | +0.908 | 0.12 | -2.852 | +1.379 | 86% |
| cond_n40 | -2.356 | 99% | +0.883 | 0.06 | -2.993 | +1.520 | 72% |

## ulti  (test n=629, soloist made 83%)
- never  : +4.412
- oracle : +1.073   (defender ceiling, gain +3.339)

| signal | current GP | current fire% | curr gain | τ* (train) | τ* test GP | τ* gain | τ* fire% |
|---|---|---|---|---|---|---|---|
| god_n6 | +20.824 | 100% | -16.412 | 0.00 | +4.412 | +0.000 | 0% |
| god_n20 | +20.824 | 100% | -16.412 | 0.00 | +4.412 | +0.000 | 0% |
| god_n40 | +20.824 | 100% | -16.412 | 0.02 | +4.297 | +0.114 | 8% |
| cond_n20 | +20.070 | 96% | -15.658 | 0.00 | +4.412 | +0.000 | 0% |
| cond_n40 | +20.197 | 97% | -15.785 | 0.00 | +4.412 | +0.000 | 0% |

## calibration: god-makeability (god_n40 signal) vs ACTUAL make rate
If actual make >> signal, god underestimates the soloist → over-kontra.

| contract | signal bin | n | mean signal | actual make% |
|---|---|---|---|---|
| parti | [0.0,0.2) | 3721 | 0.03 | 22% |
| parti | [0.2,0.4) | 277 | 0.26 | 64% |
| parti | [0.4,0.6) | 62 | 0.47 | 77% |
| parti | [0.6,0.8) | 23 | 0.68 | 91% |
| parti | [0.8,1.0) | 2 | 0.80 | 100% |
| ulti | [0.0,0.2) | 836 | 0.09 | 78% |
| ulti | [0.2,0.4) | 449 | 0.25 | 92% |
| ulti | [0.4,0.6) | 6 | 0.40 | 100% |
