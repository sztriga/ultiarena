# exp47 — overnight research: kontra signals + full model audit

Ran 3.64h of compute, 16/16 phases, nothing skipped, nothing failed.
Corpora: 12,000 natural deals (deployed frontier, cheat-free bidder) + 5,400 forced deals.

---

## Track A — the kontra research

### Headline: nothing the bidder bids is kontra-able.

Kontra is +EV for the defenders iff **P(soloist makes) < 0.5**. On the NATURAL
distribution — the contracts the frontier actually chooses — every unit is far above it:

| unit | deals | make % (natural) | make % (forced) |
|---|---|---|---|
| ulti | 4596 | **93.8%** | 66.7% |
| 40-100 | 842 | **91.3%** | 49.3% |
| 20-100 | 120 | **78.3%** | 19.2% |
| betli | 243 | **73.7%** | 60.2% |
| durchmars | 63 | **68.3%** | 30.0% |
| parti | 5997 | **64.8%** | 40.3% |

Always-kontra loses on every single unit, catastrophically under god-rekontra (ulti
−15.21, 40-100 −15.01, 20-100 −27.20, duri −16.48). **The deployed rule — kontra ulti at
≥4 trumps, colored duri at ≥3, abstain on everything else — is essentially correct, and
"don't kontra" is the right answer nearly everywhere.**

The mechanism is selection: a bidder that only declares what it is confident of leaves the
defenders nothing to attack. exp44's 39% passz rate is the same fact seen from the other
side.

### This CORRECTS an earlier claim of mine

On 2026-08-02 I reported "betli makes 38%, durchmars 33% — both genuinely kontra-able".
That came from the exp43 corpus built with the **leaky, uncalibrated** bidder, which bid
far worse hands. On the fixed frontier betli makes **73.7%** and durchmars **68.3%**.
The claim was an artifact of a broken corpus. Disregard it.

### The signal is dormant, not absent

On the FORCED corpus (difficulty deliberately spread across the breakeven, containing hands
the bidder would never bid), real signal appears and survives adverse selection:

| unit | best rule | Δ GP under god-rekontra | t |
|---|---|---|---|
| 20-100 | GBM p<0.12 | **+2.51** | +7.0 |
| durchmars | GBM p<0.15 | +0.56 | +3.1 |
| 40-100 | GBM p<0.10 | +0.21 | +2.8 |
| parti | GBM p<0.12 | +0.05 | +1.1 (n.s.) |
| ulti, betli | — | nothing beats abstain | — |

GBM AUCs 0.62–0.73, so the defender's information set *does* carry signal. It simply never
gets exercised, because the bidder does not offer hands where it matters. **If the bidder
ever becomes more aggressive, kontra work becomes valuable — this is a shelf-ready result,
not a dead end.**

### The one change worth making: drop the parti kontra rule

13,746 defender-positions, 6,843 deals. Soloist GP per position, LOWER is better for the
defenders:

| policy | soloist GP | fires |
|---|---|---|
| never kontra | **+0.497** | 0% |
| **DEPLOYED `mk < 0.10`** | **+0.522** | 58% |
| always kontra | +0.993 | 100% |
| swept optimum | +0.497 | **0%** |

**The optimum is never.** The deployed rule fires on 58% of positions and loses 0.025
GP/position doing it.

Why no threshold works: the blind makeability estimate is biased **−0.490** (says 0.127,
truth 0.617), and its reliability curve is so compressed that even its lowest bin
`[0.00,0.05)` still makes **49.4%** — right on the breakeven. The signal cannot separate
below-50% from above-50% anywhere, so no cut-point exists. Structural alternatives
(own card points, aces, tens, trump count, marriages) all do worse than abstaining.

Removing it also deletes **a PIMC solve per defender per deal** — a speed win on top.

Where parti's GP actually comes from, per deal: parti proper +0.267, silent 40-100 +0.150,
silent durchmars +0.083, silent 20-100 +0.029, defender silents −0.033. **Roughly half the
"parti" unit is silent riders**, which is why a 64.8%-make contract still pays.

---

## Track B — the full model audit

12 rotation gates, 400 deals × 3 rotations each, vs the deployed frontier. **Every single
arm is null** (|t| < 2):

| change | Δ GP/seat-deal | t |
|---|---|---|
| `open_thr_0` | −0.412 | −1.44 |
| `floor_050` | +0.328 | +0.96 |
| `exploit_off` | −0.303 | −1.02 |
| `floor_090` | −0.278 | −1.01 |
| `pimc_32` | +0.105 | +0.35 |
| `floor_070` | +0.101 | +0.32 |
| `rebetli_real_off` | −0.073 | −0.25 |
| `mix_equiv_off` | +0.048 | +0.16 |
| `betli_real_off` | −0.030 | −0.10 |
| `duri_terit_10` | −0.020 | −0.06 |
| `pimc_8` | −0.017 | −0.06 |
| `betli_def_off` | +0.008 | +0.03 |

**se ≈ 0.30, so this run can only detect effects above ~0.6 GP/seat-deal.** The honest
statement is "no component is worth more than 0.6 GP against an equal opponent", not "no
component matters".

Three things do follow:

1. **`mix_equiv_off` = +0.048 (t=0.16) is the harness check and it PASSES.** The mixer is
   value-neutral by construction, so a non-zero result would have invalidated the other
   eleven rows. It didn't.
2. **`pimc_8` = −0.017 (t=−0.06): halving the play search costs nothing measurable.**
   The engine could be ~2× cheaper, which doubles the sample size of every future
   experiment. `pimc_32` = +0.105 says doubling doesn't help either — **search depth is
   saturated at 16.** This is the highest-leverage result in Track B and deserves
   confirmation at n≥1500.
3. **The knobs are not where the strength is.** FLOOR moves the auction (bite rate 12/40 at
   0.90) yet changes GP by −0.278 ± 0.275. exp30 tuned these against the cheating bidder;
   on the honest one they are not binding.

`exploit_off` at −0.303 is directionally "exploit helps" but is unpriceable here by
construction — self-play cannot value a component designed to punish weak opponents
(verified: turning it off changes the cards played on 16 of 17 deals).

---

## What to do next

1. **Drop the parti kontra rule** — `apps/api/kontra_flow._ai_defender_kontras_unit`,
   the `parti` branch. Small GP gain, real speed gain, and it deletes the one kontra rule
   that isn't structural.
2. **Confirm `pimc_8` at n≥1500.** If it holds, 2× cheaper engine.
3. **Shelve the kontra signal work** with the forced-corpus tables intact. It is correct
   and unusable *because the bidder is selective*. Revisit if the bidder gets braver.
4. **[exp48](../48_point_banking/FINDINGS.md) is now the best open lever** — point banking
   / kenés. 48% of deals contain an avoidable feed, and 11.3% of soloist parti wins flip
   without them. That is larger than anything measured tonight.
