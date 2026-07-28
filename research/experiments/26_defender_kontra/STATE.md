# exp 26 — defender-kontra fix (overnight loop, started 2026-07-20 night)

## The problem (measured, live engine, N=210 auto-played deals)
The deployed AI **defender kontras 77% of played games**, but the soloist still
**makes it 72% of those** → the kontra is profitable only **28%** of the time. It
keeps kontra-ing piros parti (29×) and piros ulti (28×) and paying DOUBLE on
makeable games. Net: the defender kontra is **−EV**.

## Root-cause hypothesis
The defender's makeability estimate (`_hand_makeability`, own hand only, N=6
god-solves) samples the soloist's hand **uniformly** given the defender's hand. It
**ignores the auction**: the soloist CHOSE to bid this contract, which is strong
evidence of strength. So the soloist looks beatable far more often than it is →
over-kontra. The `_sol_ev(p,bid,0) < 0` threshold is also a low bar; N=6 is noisy.

## The fix under test
**Auction-conditioned makeability**: when a defender samples the soloist's hand,
reject worlds where the champion bidder would NOT have declared ≥ the observed
contract with that hand. Plus ablate N (6/20/40) and a confidence margin.

## Design (efficient — play once, re-score many policies)
Kontra is a payoff multiplier (play is UNCHANGED), so per deal we play the hand
ONCE (PIMC both sides, cheat-clean) and cache the oracle GP at kontra level 0/1/2.
Any policy then just picks a level from its makeability estimate and reads the
cached GP. Backbone reused: `exp24/harness.run_auction` (champion auction) +
`exp24/scorers` play/score + `exp23/{provider,auction,bidder,kontra}`.
Champion config: FLOOR=0.7, DEBIAS_PCTL=0.80, NetProvider(calibrate=True).

## Metric
Per policy over the eval set: **mean soloist total_per_def GP** (LOWER = better
defense), kontra fire-rate, kontra profitability (soloist bukott | kontra), and
**defender GP gained vs never-kontra**. Baseline (`deployed`) is expected NEGATIVE
(kontras hurt defenders); the win is a policy that pushes it ≥ 0 without throwing
away the genuinely +EV kontras.

## Guardrails
Sandbox only (this dir). NEVER touch apps/api/play.py, champion config, or
deployed checkpoints. Every AI decision cheat-clean. Headline numbers at N≥500.
Nothing auto-integrated — milan decides in the morning.

## Files
- `harness26.py` — driver. Subcommands: `smoke`, `build`, `policies`.
- `played.jsonl` — cached eval set (one played hand per line, GP at level 0/1/2).
- `results.md` — policy comparison table (rewritten each `policies` run).
- `build.log` / `policies.log` — flushed heartbeat progress.

## Progress log (append newest last)
- 2026-07-20 night: harness built + smoke-validated. Smoke (N=15) already showed the
  story: kontra-ing PARTI is fine (+1.3 defender gain), kontra-ing ULTI is the disaster
  (−23 GP/deal) — ulti made ~100% vs real defense but god-makeability says ~30%.
  Refined analysis to per-contract threshold sweep + train/test split + god→real
  calibration diagnostic. Conditioning is a NO-OP for parti (piros parti is the floor,
  every hand "would bid ≥ parti") but does filter for ulti.
- build DONE (N=2000 seeds → 1785 kept: 1380 parti, 405 ulti; ~10 min @ 200/min).
- pools RUNNING (POOL=40 worlds × 3 viewers/deal, ~260 deals/min, ETA ~6 min).
- pools DONE. PHASE 1 RESULTS (N=1785, train/test split, in results.md):
  * PARTI (test 698, made 27%): deployed kontra already HELPS defenders +0.8;
    tuned τ +1.3 (oracle ceiling +1.86). Kontra-ing parti is correct. Conditioning=no-op.
  * ULTI (test 198, made 80%): deployed kontra COSTS −15.4 GP/deal (catastrophic).
    Oracle ceiling +4.3 (value in catching the 20% bukott ulti), but best god/cond
    threshold recovers only +0.38 out-of-sample → effectively "don't kontra ulti".
  * CALIBRATION smoking gun: for ulti, god-makeability <0.2 → soloist ACTUALLY makes
    77%; 0.2-0.4 → 90%. God measures "beatable by PERFECT defense", not real defense.
  * CHEAP DEPLOYABLE FIX already found: contract-specific τ (keep parti kontra, ~never
    kontra ulti) swings the mix from −2.7 to +1.1 GP/deal (+3.9 defender swing).
- PHASE 2 RUNNING: realistic-defense makeability (PIMC playout per sampled world) on
  the 405 ulti deals (POOL_R=20, viewers 1,2). Q: can p_real recover more of the +4.3
  ulti ceiling than god's +0.38? → results_real.md.
- PHASE 2 DONE: realistic-defense makeability does NOT help (real τ* gain +0.42 vs
  god +0.38; both « +4.3 ceiling). Even p_real is miscalibrated on ulti (sig 0.09 →
  make 67%) because sampled soloist hands ignore that the soloist BID ulti (strong).
- PHASE 3 (structural): max-defender-#trumps predicts ulti make cleanly — 4 trumps→37%
  made, 3→71%, 2→92%, 1→100%. Out-of-sample a "kontra iff ≥4 trumps" ulti gate gains
  +0.5 vs never (beats makeability's +0.38). ≥3 is net-negative (71% still made).
- FINAL COMBINED POLICY (results_combined.md, held-out test, N=1785):
  * deployed kontra: soloist GP +2.48 (kontra flood HURTS defenders).
  * RECOMMENDED (parti: makeability τ*=0.08; ulti: kontra iff a defender holds ≥4
    trumps): soloist GP −1.37 → **+3.85 GP/deal defender gain vs deployed**, ~75% of
    the +5.11 oracle-vs-deployed gap. Split: parti +0.43, ULTI +15.9 (the whole story).
  * The +4.3 ulti oracle ceiling is MOSTLY UNREACHABLE cheat-clean: beatability lives
    in the JOINT defender holdings, not one hand. Best own-hand signal ≈ never-kontra.
- CONCLUSION: the deployed ulti kontra is a −15.9 GP/deal disaster; the fix is a
  trivial, cheat-clean, contract-specific gate. NOT a retraining problem.
- FIRM (N=5374 kept: 4083 parti, 1291 ulti; held-out test 2688):
  * deployed kontra soloist GP +3.01; RECOMMENDED −1.36 → **+4.37 GP/deal defender
    gain vs deployed** (+1.27 vs never); oracle ceiling −2.38 (+5.38). Captures ~81%
    of the reachable improvement. Split: parti +0.56, ULTI +16.85.
  * ulti trump-gate (≥4) beats never (+0.44), best-makeability-τ (+0.11), and realistic
    makeability (+0.42). Confirms trump count is THE ulti signal.
- LOOP COMPLETE. Deliverable = SUMMARY.md (morning briefing + proposed 4-line patch to
  _ai_defender_kontras). Nothing applied to the deployed engine (milan's call).
  No background jobs left running; no scheduled wakeups.
