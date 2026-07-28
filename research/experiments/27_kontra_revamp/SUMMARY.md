# exp 27 — Full-ladder kontra/rekontra revamp (2026-07-21)

## Outcome
The kontra/rekontra decision logic was rebuilt per-unit and calibrated. In a held-out
tournament vs the previous frontier, the new logic **wins +7.7 GP/deal** (cheat-clean,
N=4000 test). It is **now the frontier** — `apps/api/play.py` `_ai_defender_kontras`
was replaced (validated core, "Stage 1"). Rekontra was left unchanged (shown to be a
minor lever). A second, thinner extension (per-unit kontra for combined games) remains a
proposal ("Stage 2").

## The bug in the old logic
The deployed defender kontra'd iff its own-hand **blind god-makeability** `_sol_ev(p) < 0`
(≈ p < 0.5). That estimate samples RANDOM soloist hands and ignores that the soloist
*chose to bid* the contract — so it "sees" ~6–11% makeability when the true make is ~80%.
Result: it kontra'd almost everything (98% of games; live-measured 77%), and the soloist's
rekontra amplified the loss. It was worst on **ulti**: kontra'd a contract made 83% of the
time and paid double/quadruple.

The mechanism is NOT "god ≠ realistic" (my exp26 framing was wrong). True perfect-play make
≈ realistic make for ulti/parti/duri; the estimate was simply **blind to the auction**.

## What the new logic does (per unit, own hand only — cheat-clean)
| unit | kontra rule | why |
|---|---|---|
| **ulti** | a defender holds **≥4 trumps** | make drops 76%→32% at 4 trumps; trump count is the whole signal (win-prob adds ~0) |
| **durchmars** (colored) | a defender holds **≥3 trumps** | make 50%→~3% at 3+ trumps |
| **parti** | blind makeability ≈ 0 | parti is made only 36% → kontra is robustly +EV; the threshold trims the rest |
| betli, 40-100, 20-100, colorless duri | **abstain** | no own-hand signal beats not-kontra-ing (validated) |

**Key analysis result (answers "are the win-probs useful?"):** for the *trick* contracts
the estimated win-probability is redundant with — even weaker than — trump count (duri:
win-prob AUC 0.500, useless; ulti: +0.003 over trumps). It only earns its keep on **parti**
(a distributed point battle, +0.065 AUC). So the new logic uses **structure (trump count)
for trick contracts and makeability only for parti** — simpler, faster, and more robust.

Rekontra: isolated test showed ±0.2 GP/deal once defenders stop over-kontra-ing, so the
existing post-trick-1 rekontra rule was kept untouched.

## Tournament (held-out N=4000, PIMC vs PIMC, cheat-clean)
- Self-play: all-old soloist +6.69 vs all-new soloist −1.04 → new defenders concede **+7.73 GP/deal less**.
- Head-to-head: **new wins +7.74 GP/deal** (decisions differ on 51% of deals).
- Defender-only (rekontra fixed): **+7.68 GP/deal**.
- By contract: piros ulti defender gain **+30.5** (the disaster fixed); piros parti +1.16.
  ~97% of the win is the simple ulti + parti games.

## Live validation (the deployed engine after the change)
Defender kontra fire rate 77% → **24%**; ulti no longer reflexively kontra'd; e2e regression
green; no errors over 88 auto-played games. (Live "profitability%" is not a clean metric —
the human seat plays badly in the driver — the tournament is the clean measure.)

## What changed in code
`apps/api/play.py` → `_ai_defender_kontras` (per-unit gates + 3 module constants
`_KONTRA_ULTI_TRUMPS=4`, `_KONTRA_DURI_TRUMPS=3`, `_KONTRA_PARTI_MAKE=0.10`). Nothing else
touched — same auction, play, scoring, kontra state machine, rekontra.

## Stage 2 (proposal, NOT applied)
Extend kontra to **combined/100/terített** games per-unit (they currently get none). The
harness shows small further gains (e.g. kontra-ing a beatable duri unit inside a combined
bid) but on thin data (n=14–15/contract) and it needs a multi-unit kontra state machine.
Deferred until it has firm data.

## Reproduce
`experiments/27_kontra_revamp/harness27.py` — `build` / `pools` / `godactual` / `analyze`
/ `teaching` / `tournament`. Data: `played.jsonl` (8000 deals, all 6 units), `pools.jsonl`,
`godactual.jsonl`. Reports: `results_units.md`, `TOURNAMENT.md`, `TEACHING.md`.
