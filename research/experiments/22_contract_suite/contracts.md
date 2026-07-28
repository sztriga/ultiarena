# Contract checklist (exp 22)

Status per contract: **spec** (milan explained it) · **validate** (A/B confirms
the behavior) · **tweak** (code change needed/done) · **integrate** (wired into
the bidding loop). Fill the result column as we go.

| # | Contract | Bid | Silent | spec | validate | tweak | integrate | result / notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Parti — SCORING BASE only | NOT biddable on its own | — | ✓ | | | | scoring/play base (`parti_pts`); bare non-piros parti is a no-go — ladder bottoms at PASS → piros parti (`_is_legal_bid`) |
| 1b | Piros parti | ✓ (cheapest declarable, rank 2) | rides silent ulti/100/duri | ✓ | ✓ | ✓ | | **scoring spec DONE** (`test_parti_scoring.py` 13/13): silent 100/duri REPLACE parti & stack; silent ulti STACKS (bukott ×2); def-side silent 100/duri added; talon in parti-win fixed. See [[reference_parti_scoring_model]] |
| 2 | Ulti | ✓ EV_ULTI (7-hold, bukott) | ✓ silent_ulti_signed | ✓ | ✓ | ✓ | | **scoring DONE** (`test_ulti.py` 8/8): bid ulti = ulti(4, piros 8) + ACCOMPANYING parti(1, piros 2), same color → "4+1"; bukott −8 (piros −16) w/ parti scored alongside; silent duri/40-100/20-100 REPLACE the parti part. Oracle already correct. exp21: sol +1.25, def denial −0.40 |
| 3 | 40-100 | ✓ (4, piros 8) | ✓ score_geq_100 gate | ✓ | ✓ | ✓ | | **scoring DONE**: bid leaves parti OUT (the 100 is the whole stake); made/bukott ±8 piros; silent ulti/duri still hold. `test_parti_scoring` +8/−8. card≥60. |
| 4 | 20-100 | ✓ (8, piros 16) | ✓ | ✓ | ✓ | ✓ | | **scoring DONE**: same, ±16 piros, card≥80. Ladder: 20-100 (8, non-piros) ranks above piros 40-100 (8) — simplest wins on ties. |
| 5 | Durchmars (colored) | NEVER standalone — only **combos** (ulti-duri/40-100-duri @10) or **silent** (+3) | ✓ silent_durchmars | ✓ | ✓ | ✓ | | **scoring DONE** (`test_betli_duri` ±6/±12, `test_universal_riders` 6/6): combos carry silent riders (silent ulti/100 now UNIVERSAL — ulti-duri+sweep+40 = +12). exp21: sol-side redundant; def breaks ½ soft sweeps |
| 6 | Durchmars (colorless) | ✓ STANDALONE (the "duri 6" rung, piros 12) | — | ✓ | ✓ | | | **scoring DONE** ±6/±12. **10-low rank A-K-O-U-10-9-8-7** (solver concern, not oracle). No trump/marriages → no silent riders (clean). |
| 7 | Betli | ✓ **5** (piros betli = REBETLI 10) | — (not a real silent) | ✓ | ✓ | ✓ | | **scoring DONE** (`test_betli_duri.py` ±5/±10): binary, no parti. **REBETLI = double-rung: betli sits in the ladder TWICE (5 and 10); rebetli(10) ranks above piros ulti(10)** — see [[reference_bidding_ladder]] |
| 8 | Marriage 40/20 (naked) | — | points only, 0 GP | | | | | folded into parti |
| — | Modifier: **piros** (×2 all components) | ✓ | | ✓ | ✓ | | | done |
| — | Modifier: **terített** (open cards; ×2 duri/betli colored, ×4 colorless) | ✓ | ✓ | ✓ | ✓ | | | **DONE** `BidSet.teritett` (`test_teritett.py` 8/8): betli 20, colorless duri 24, combo duri ×2. Open-hand play = solver concern. |
| — | Modifier: kontra/rekontra | oracle mult (×2^level) | | | | | | untested |
| — | **Combinations** (pairs: ulti/duri × 100/parti; triples: ulti+100+duri) | ✓ | ✓ | ✓ | ✓ | | | **DONE** — oracle scores any combo (independent components); validated `test_universal_riders` (incl. triples ±14/±18) |

## Known gaps / edges to revisit
- **Multi-marriage 100 threshold** — a bid/silent 100 while holding a 2nd marriage
  makes the solver's fixed `total>=100` fire early; fix = configurable score
  threshold (small Cython like `set_multi_cull`). Unbuilt.
- **Bid ulti/duri/betli are NOT multi components** — they live in dedicated
  evaluators (constraints + bukott). Fine for play; means the unified engine
  prices only their silent forms.

(Per-contract validation scripts land here as `c<NN>_<name>.py`, logs as
`run_c<NN>_*.log`.)
</content>
