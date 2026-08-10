# exp47 — overnight research report

Budget 6.0h · elapsed 3.64h · 8 workers · 400 deals/ablation

## Coverage

- completed: 16
- skipped: none
- failed: none


## Track B — what each moving part is worth

Rotation gate vs the deployed frontier. Candidate rotates through all three seats, so **0.000 = parity** and a positive delta means the CHANGE is better than what ships. Aggregate only — per-seat numbers need a matched control (see gate_lib).

| change | delta GP/seat-deal | se | t | verdict |
|---|---|---|---|---|
| `exploit_off` | -0.303 | 0.296 | -1.02 | no detectable difference |
| `floor_070` | +0.101 | 0.318 | +0.32 | no detectable difference |
| `floor_090` | -0.278 | 0.275 | -1.01 | no detectable difference |
| `floor_050` | +0.328 | 0.342 | +0.96 | no detectable difference |
| `open_thr_0` | -0.412 | 0.285 | -1.44 | no detectable difference |
| `betli_def_off` | +0.008 | 0.299 | +0.03 | no detectable difference |
| `duri_terit_10` | -0.020 | 0.317 | -0.06 | no detectable difference |
| `betli_real_off` | -0.030 | 0.299 | -0.10 | no detectable difference |
| `rebetli_real_off` | -0.073 | 0.299 | -0.25 | no detectable difference |
| `pimc_32` | +0.105 | 0.299 | +0.35 | no detectable difference |
| `pimc_8` | -0.017 | 0.299 | -0.06 | no detectable difference |
| `mix_equiv_off` | +0.048 | 0.297 | +0.16 | no detectable difference |

Interpretation: a strongly NEGATIVE delta means turning the component off costs GP, i.e. the component is earning its keep. A delta near zero on a knob means the knob is not binding and could be simplified away.

## Track A — kontra

Full per-unit signal tables are in `night.log` (search for `KONTRA SIGNALS`); the parti deep-dive is under `PARTI DEEP-DIVE`.

- `natural`: done — {'deals': 12000, 'kept': 7190}
- `forced`: done — {'betli': 858, 'durchmars': 611, 'teritett_betli': 817, '20_100': 764, '40_100': 700, 'ulti': 845}
- `parti`: done — {'rows': 13746}
- `signals`: done — {'natural': {'parti': 11994, 'ulti': 9192, '40_100': 1684, '20_100': 240, 'durchmars': 126, 'betli': 486}, 'forced': {'parti': 1752, 'ulti': 1690, '40_100': 1400, '20_100': 1528, 'durchmars': 1222, 'betli': 3350}}

