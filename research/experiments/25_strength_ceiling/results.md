# Exp 25 results

## Phase 1a — god-provider perception ceiling (2026-07-05)
h2h god-agent vs net-agent, N=150, god scorer, same composer/auction/play:

| | GP/deal |
|---|---|
| god-agent (perfect INFO) | **+5.68** |
| net-agent (current nets) | **−2.84** |
| **perception headroom (god edge)** | **+8.53** |

**Perception is a massive lever** — perfect perception buys ~+8.5 GP/deal.
→ retraining/improving the nets can help a lot; question answered YES.

Headroom concentrates on **rare high-value contracts** the net mis-perceives:
god-agent wins on teritett rebetli (18×, value 40), terített colorless duri,
piros ulti combos. The net misses/over-bids these.

**Caveats:** absolute (perfect-info) ceiling — trainable slice = gap to the PIMC
marginal ceiling (Phase 1b), rest is irreducible info gap. God scorer
(pessimistic). N=150 → wide CI from the rare ±40 contracts; direction certain.

## Phase 1b — perception headroom SPLIT (2026-07-05, N=150, N_DET=8, god scorer)
| | GP/deal | of total |
|---|---|---|
| total (net → god, perfect INFO) | +8.53 | 100% |
| **TRAINABLE (net → PIMC marginal)** | **+1.50** | **18%** |
| irreducible (PIMC → god, info gap) | +7.03 | 82% |

**Retraining the bidding nets is a MODEST lever (~+1.5/deal, a lower bound at
N_DET=8), not +8.5.** 82% of the god headroom is INFORMATION (seeing opponents'
hands) — unattainable for any bidder. The ceiling-first method paid for itself here:
it stopped us chasing an irreducible gap.

Reframe: (1) bidding-net retraining ≈ +1.5, secondary. (2) The whole measurement
used GOD PLAY for both agents → the **play-quality lever (PIMC→god) is unmeasured**
and likely bigger. (3) The +7.0 info gap is partly recoverable in PLAY via
opponent-hand inference (belief modeling), not by bidding nets.

## Phase 1c — PLAY ceiling (2026-07-05, N=120, PIMC_N=16) — the big surprise
| policy | GP/def |
|---|---|
| (pimc-sol, pimc-def) realistic | +0.90 |
| (god-sol, pimc-def) perfect play | +0.38 |
| (god-sol, god-def) both perfect | −1.05 |

**Soloist-play headroom (pimc→god) = −0.52 (NEGATIVE).** Double-dummy soloist play
is WORSE than PIMC vs imperfect defenders — god gives up on dd-unmakeable contracts,
PIMC keeps fighting and EXPLOITS defender errors. → the play lever is EXPLOITATIVE
play / opponent-modeling, NOT perfection. (Defender-strength pimc→god = +1.43.)

## Concrete wins (2026-07-05 night)
- **Anti-cheating** verified: bidder (behavioural, 0/25 leak) + PIMC play (code trace).
  god control leaks 11/25 → test is sensitive. `audit_cheating.py`.
- **FLOOR=0.7** (per-component confidence floor) kills bleeders: god METRIC −2.31→−1.41;
  realistic PIMC +0.61→+3.23 (N=150) / champion +1.89 (N=400).
- **Kontra** (full ladder, god play): P0 −2.91→−1.89, pass 0→69%, piros-parti wr .22→.62.
- **Champion #1**: net + FLOOR=0.7 + kontra-aware + PIMC play → realistic soloist
  **+1.89 GP/game** vs imperfect defenders (beats them). Residual: terített colorless
  duri −34 (open-hand harder than the closed net thinks).

## Next
- **Frontier = opponent-modeling / exploitative play** (recovers the +7 info gap in-play).
  Bidding-net retraining capped at ~+1.5 → secondary.
- Wire the unified realistic+kontra eval (kontra decisions during PIMC play).
- Fix terített colorless duri (open-hand makeability / hard floor).
