"""Track B — what is each moving part actually worth?

Every knob below was tuned, or promoted, against a bidder that CHEATED (it read the talon
before deciding whether to enter the auction; fixed 2026-08-02). FLOOR and DEBIAS_PCTL in
particular exist partly to suppress over-bidding that no longer happens the same way, so
their tuned values are suspect on their face. Nothing here has been re-measured since.

Each entry is one rotation gate: incumbent = the deployed frontier, candidate = the
frontier with exactly ONE thing changed. Ulti is zero-sum and the candidate rotates
through all three seats, so the aggregate mean IS the delta and 0.000 means parity — no
control run needed for the headline number (only for per-seat breakdowns, which is a trap
documented in gate_lib).

Ordering is by expected value × current doubt, because the overnight driver works down the
list until its budget runs out. Anything not reached is simply not measured — the driver
says so rather than leaving a gap.

READ NULLS CAREFULLY. This is SELF-PLAY: both arms are the same frontier AI, so an
ablation measures a component's value AGAINST AN EQUAL OPPONENT. Anything designed to
punish a WEAK opponent is structurally invisible here. `exploit_off` is the clearest case
— verified 2026-08-02, turning it off changes the played cards on 16 of 17 deals yet moved
GP by exactly 0.000 over 45 seat-deals, because both sides defend well enough that the
different line reaches the same outcome. exp31 promoted it on +0.6..0.75 GP/deal *versus a
fallible defender* and explicitly measured 0 regression versus a perfect one. So a null
here means "worth nothing against ourselves", NOT "worth nothing" — and for exploit that
is the expected result, not a discovery. The same caution applies to the anti-tell mixer.
"""
from __future__ import annotations

# MEASURED BITE RATES (2026-08-02, 40 deals, auction outcome changed vs the frontier).
# Taken before committing budget, because an ablation that never changes a decision spends
# 15 minutes to report 0.000 and reads like evidence of "no effect":
#
#     floor 0.95        14/40      floor 0.90        12/40
#     floor 0.50         7/40      open_thr 0 (nf)    3/40
#     floor 0.70         1/40      rebetli off        1/40
#     DEBIAS 0.30        1/40      DEBIAS 1.00 (off)  0/40
#     blind_pctl 0.75    0/40      blind_pctl 0.50    0/40
#
# The debias knobs are DEAD and are not run. Two reasons, both worth knowing:
#   * DEBIAS_PCTL corrects an argmax over 66 discards. Since the talon fix, the pass/bid
#     decision is made BEFORE any discard exists, so the correction can only reorder
#     contracts after a seat has already committed - and the top rung wins by a wide margin.
#   * A debias on the blind stage (blind_pctl, added and measured) is inert for the same
#     structural reason FLOOR dominates: on most hands the blind stage has exactly ONE
#     candidate rung, `piros parti`, because FLOOR=0.80 gates every other rung out.
#     `piros parti` is hearts-only, so there is a single trump estimate and no
#     max-over-trumps curse to correct.
# That last point is the real headline: for a majority of hands the current pickup decision
# reduces to "is my piros parti EV above -2?". FLOOR decides whether any other contract is
# even a candidate, which is why the FLOOR arms lead the list.

# (name, candidate spec, why)
ABLATIONS = [
    ("exploit_off",
     {"bid": {}, "play": {"exploit": False}},
     "exp31 safe-exploit soloist. Expected to read ~0.000 in self-play by construction "
     "(see the module note) — the point is to confirm it costs NOTHING against a strong "
     "opponent while being the most expensive component we run (~1.3-1.6 s/move). A "
     "clearly NEGATIVE result would be the surprise: it would mean exploit is now hurting."),

    ("floor_070",
     {"bid": {"floor": 0.70}, "play": None},
     "FLOOR was raised 0.70 -> 0.80 by exp30 to suppress overconfident escalations by the "
     "CHEATING bidder. The honest bidder already passes 39% of deals; the floor may now be "
     "double-counting caution."),

    ("floor_090",
     {"bid": {"floor": 0.90}, "play": None},
     "The other direction — if the ladder is clean because the bidder only plays certainties, "
     "tightening further should cost GP. A null here says FLOOR is not the binding constraint."),

    ("floor_050",
     {"bid": {"floor": 0.50}, "play": None},
     "Widens the FLOOR bracket downward. With the honest bidder passing 39% of deals, the "
     "question is whether the ladder is clean because FLOOR is right or because it is "
     "strangling participation; 0.50/0.70/0.80/0.90 brackets that."),

    ("open_thr_0",
     {"bid": {}, "play": None, "opens": 0.0},
     "The non-forehand opening threshold. Every seat opens against -2/def, but only the "
     "FOREHAND forfeits it: on a passz the forehand pays -4 while the other two COLLECT +2 "
     "(exp44/6000). A first probe came back +0.12 +- 0.32 (n=600, exp45) - inside noise, so "
     "this re-runs it inside a coherent sweep rather than alone."),

    ("betli_def_off",
     {"bid": {}, "play": {"betli_def": False}},
     "exp36 learned betli-defense net (-21pp soloist steal vs PIMC). Promoted 2026-07-24 and "
     "never ablated end-to-end in GP; the benchmark measured steal rate, not table GP."),

    ("duri_terit_10",
     {"bid": {"duri_mult": 1.0}, "play": None},
     "DURI_TERIT_MULT=0.3 was exp30's clamp on a terited-duri over-bid leak that the honest "
     "table no longer shows at all (0 deals bleeding). If the leak is gone the clamp may now "
     "just be suppressing profitable bids."),

    ("betli_real_off",
     {"bid": {"betli_real": False}, "play": None},
     "exp37 realistic-defense betli make-prob for PLAIN betli (+0.16 GP/game when promoted). "
     "Betli make has since collapsed 86% -> ~38% as exp36 defense and the terit reveal shipped, "
     "so the head is pricing a world that moved."),

    ("rebetli_real_off",
     {"bid": {"rebetli_real": False}, "play": None},
     "exp39 betli->rebetli escalation. Promoted on +8.14 GP/bid but a h2h that was only "
     "+0.16 (n.s.) — the weakest evidence of any promoted feature."),

    ("pimc_32",
     {"bid": {}, "play": {"pimc_n": 32}},
     "Double the play search. Prices depth directly: exp25 put play-depth at ~+1 GP, measured "
     "before the exploit soloist and the betli net existed."),

    ("pimc_8",
     {"bid": {}, "play": {"pimc_n": 8}},
     "Half the search. If this is free, the whole engine gets 2x cheaper and every future "
     "experiment doubles in sample size — the highest-leverage null result available."),

    ("mix_equiv_off",
     {"bid": {}, "play": {"mix_equiv": False}},
     "The anti-tell mixer. Proved value-neutral by construction (max delta 0.0 over 480 real "
     "picks), so this is a HARNESS CHECK: a non-zero result means the gate is measuring noise "
     "as signal, and everything above it is suspect."),
]

INCUMBENT = {"bid": {}, "play": None}
