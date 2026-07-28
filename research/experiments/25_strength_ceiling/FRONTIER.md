# The frontier: exploitative / opponent-modeling play

Tonight's map says the biggest realistic lever is **play quality against imperfect
opponents**, not bidding nets (+1.5 cap) and not perfect play (which HURTS: −0.5).
This note turns "go at opponent-modeling" into a concrete, cheat-clean build order.

## Why there is headroom
The +7.0 GP "irreducible" info gap is irreducible **at bid time**. In-play it is
partly recoverable: every card an opponent plays (or declines to play) leaks their
holding. Our PIMC today samples consistent worlds **uniformly** (hard voids only).
A human expert instead:
  1. weights worlds by how likely each defender's *actual plays* were under them
     (Bayesian belief), and
  2. assumes defenders play a **realistic imperfect policy**, then plays the line
     that maximizes EV against THAT policy — i.e. sets traps a perfect defender
     would sidestep but a real one walks into.
Both are pure inference from public actions ⇒ **cheat-clean** (camera-POV safe).

## Build order (each step measurable, each falls back cleanly)
1. **Weighted determinization (belief update).** In `solvers/determinize.py`, keep the
   pool+reshuffle, but score each sampled world by the product of per-trick play
   likelihoods under a cheap defender model (e.g. "followed suit low when safe",
   "didn't ruff ⇒ prob(void trump) up"). Importance-weight the PIMC vote by that
   score. Uniform weights = today's behavior, so it's a strict generalization.
   MEASURE: realistic METRIC vs today at matched PIMC_N (kpimc scorer, N≥600).
2. **Defender-mistake exploitation.** At the PIMC leaves we currently (roughly) assume
   optimal defense. Replace the defender rollout with the *realistic imperfect policy*
   (the PIMC-N-low policy we already have) so the soloist's search prefers lines that
   beat the real defender, not the double-dummy defender. This is exactly the effect
   that made imperfect-INFO play BEAT god play tonight — make it explicit.
   MEASURE: soloist GP on contracts that are double-dummy LOST but realistically
   makeable (the terített-rebetli / open-hand class already shows +67).
3. **Belief-conditioned kontra.** Feed the same per-world weights into the hand-based
   kontra estimate (`_hand_makeability`) so the kontra/rekontra decision uses the
   sharpened belief, not the uniform one. Still own-hand + public only.

## Guardrails (do not regress on these)
- Re-run `audit_cheating.py` after each step: the soloist's decision must be invariant
  to opponents' hidden cards *given the same public actions*. A weighted model that
  peeks at the true assignment would FAIL this — that's the tripwire.
- Never optimize play toward the god/double-dummy target (tonight: −0.5). The metric to
  climb is the **realistic** kpimc METRIC / P0, against imperfect defenders.
- Keep the uniform-weight and optimal-defender paths as the fallback (env-gated), so a
  bad model can't silently make the champion worse.

## Expected payoff
Play quality moved GP from +0.75 (N=4) to +3.23 (N=16) just by adding determinizations.
Belief-weighting + mistake-exploitation is the qualitative version of that lever and
should recover a meaningful slice of the +7 in-play — the single most promising line
toward "beat humans." Bidding-net retraining (+1.5) is the secondary track.
