"""Outcome scorers: given the auction result, what GP does the soloist realize?

god_outcome — double-dummy (god) defenders, the fast deterministic inner-loop
metric. Plays the soloist's optimal line for the declared contract and scores the
terminal with the trusted oracle, so SILENT riders (silent ulti/100/duri) and
terített are scored exactly:
  • pure points game (parti / 40-100 / 20-100)  → multi solver + PV-replay + oracle
  • single trick contract (ulti)                → EV_ULTI + PV-replay + oracle
  • betli / standalone colorless duri (binary)  → dedicated solve, rate (early-term,
        no replay; colourless ⇒ no riders anyway)
  • multi-trick COMBO (ulti-duri, …)            → component-wise approx (rare; flagged)

pimc_outcome — placeholder for the periodic realistic (closed-play) check.
"""
from __future__ import annotations

import os
import sys

_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        pass

from bidding.ladder import GPTable                                   # noqa: E402
from bidding.recipe import sol_marriages                       # noqa: E402

GP = GPTable()


def resolve_bidset(rung, sol10, trump):
    if len(rung.bids) == 1:
        return rung.bids[0]
    has40, has20 = sol_marriages(sol10, trump)
    for b in rung.bids:
        if b.forty_hundred and has40:
            return b
        if b.twenty_hundred and has20:
            return b
    for b in rung.bids:
        if not (b.forty_hundred or b.twenty_hundred):
            return b
    return rung.bids[0]


def _multi_weights(bid, sol10, trump):
    has40, has20 = sol_marriages(sol10, trump)
    if bid.forty_hundred:
        sg = 8.0
    elif bid.twenty_hundred:
        sg = 16.0
    else:
        sg = (2.0 if has40 else 0.0) + (1.0 if has20 else 0.0)   # silent-100 margin
    return dict(parti_pts=(0.0 if (bid.forty_hundred or bid.twenty_hundred) else 1.0),
                silent_ulti=2.0, silent_durchmars=3.0, score_geq_100=sg)


def _play_and_score(bid, contract, weights, sol, d1, d2, trump, talon, restrict):
    from solvers import pis
    from scoring.oracle import score as score_oracle
    from trickster._solver_core import set_multi_weights
    if weights is not None:
        set_multi_weights(**weights)
    bkw = dict(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
               contract=("parti" if contract == "multi" else contract),
               trump=trump, talon=list(talon),
               declare_marriages=(trump is not None), marriage_restrict=restrict)
    pos = pis.build_position(**bkw)
    pv = pis.principal_variation(pos, contract=contract)
    sim = pis.build_position(**bkw)
    for _pid, card in pv:
        if pis.is_terminal(sim):
            break
        pis.apply_move(sim, card)
    pvec = score_oracle(final_pos=sim, bid=bid)
    return pvec


def _primary_made(bid, pvec):
    """Did the declared (bid) components all land positive?"""
    keys = []
    if bid.ulti:
        keys.append("ulti")
    if bid.durchmars:
        keys.append("durchmars")
    if bid.betli:
        keys.append("betli")
    if bid.forty_hundred:
        keys.append("40_100")
    if bid.twenty_hundred:
        keys.append("20_100")
    if not keys:
        keys = ["parti"]
    comps = pvec.components
    return all(comps.get(k, 0) > 0 for k in keys)


def god_outcome(rung, trump, sol, d1, d2, talon, seed=0):
    """(realized GP/def, primary_made) under god double-dummy defenders.

    Pure POINTS games use the exact multi-solve + PV-replay + oracle (captures
    silent riders). Anything with a TRICK contract (ulti/duri/betli) uses
    god_says makeability + GPTable rates — because EV_ULTI/EV_DURCHMARS/EV_BETLI
    EARLY-TERMINATE, so their PV is truncated and can't be replayed/oracle-scored.
    Combos are component-wise (independence approx; rare; flagged)."""
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if n_trick == 0:
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        pvec = _play_and_score(bid, "multi", _multi_weights(bid, sol, trump),
                               sol, d1, d2, trump, talon, restrict)
        return pvec.total_per_def, _primary_made(bid, pvec)
    return _component_combo(bid, rung, trump, sol, d1, d2, talon)


def _component_combo(bid, rung, trump, sol, d1, d2, talon):
    """Trick contracts + riders via god_says makeability (no replay)."""
    from solvers import pis
    from eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights

    def g(contract, t, restrict=None, multi=False):
        if multi:
            set_multi_weights(score_geq_100=1.0)
        pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                                 leader=0, contract=("parti" if multi else contract),
                                 trump=t, talon=list(talon),
                                 declare_marriages=(multi or t is not None),
                                 marriage_restrict=restrict)
        return bool(god_says_soloist_wins(pos, contract=("multi" if multi else contract)))

    gp = 0.0
    makes = []
    if bid.ulti:
        m = g("ulti", trump); makes.append(m)
        gp += GP.ulti_bid if m else -GP.ulti_bid_bukott
    if bid.durchmars:
        dt = None if rung.colorless else trump
        terit = (2 if not rung.colorless else 4) if bid.teritett else 1
        m = g("durchmars", dt); makes.append(m)
        gp += (GP.durchmars_bid if m else -GP.durchmars_bid) * terit
    if bid.betli:
        terit = 4 if bid.teritett else 1
        m = g("betli", None); makes.append(m)
        gp += (GP.betli if m else -GP.betli) * terit
    if bid.forty_hundred:
        m = g(None, trump, restrict="40", multi=True); makes.append(m)
        gp += GP.forty_hundred_bid if m else -GP.forty_hundred_bid
    if bid.twenty_hundred:
        m = g(None, trump, restrict="20", multi=True); makes.append(m)
        gp += GP.twenty_hundred_bid if m else -GP.twenty_hundred_bid
    # accompanying parti rider on a points-based trick game (bid ulti = "4+1")
    points_based = not (bid.durchmars or bid.betli)
    if points_based and not (bid.forty_hundred or bid.twenty_hundred):
        gp += GP.parti if g("parti", trump) else -GP.parti
    if bid.piros:
        gp *= 2
    return gp, (all(makes) if makes else True)


def kontra_god_outcome(rung, trump, sol, d1, d2, talon, seed=0):
    """Simple-game god outcome WITH kontra: god makeability → post-trick-1 kontra
    decision (under god, p∈{0,1}, single belief ⇒ defenders kontra failures, no
    rekontra) → analytical kontra-scaled GP (primary + scaled parti rider for
    bid ulti). Simple contracts only (milan 2026-06-26)."""
    from solvers import pis
    from eval.pimc_matchup import god_says_soloist_wins
    from bidding.kontra import contract_payoffs, kontra_level_for

    def gsw(contract, t):
        pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                                 leader=0, contract=contract, trump=t,
                                 talon=list(talon), declare_marriages=(t is not None))
        return bool(god_says_soloist_wins(pos, contract=contract))

    bid = resolve_bidset(rung, sol, trump)
    if bid.ulti:
        primary = gsw("ulti", trump)
    elif bid.betli:
        primary = gsw("betli", None)
    elif bid.durchmars:
        primary = gsw("durchmars", None if rung.colorless else trump)
    else:
        primary = gsw("parti", trump)

    p = 1.0 if primary else 0.0
    level = kontra_level_for(p, bid)
    made_gp, buk_gp = contract_payoffs(bid, level)
    gp = made_gp if primary else buk_gp
    if bid.ulti:                                     # accompanying parti rider, scaled
        pw = gsw("parti", trump)
        gp += (1 if pw else -1) * (2 ** level) * (2 if bid.piros else 1)
    return gp, primary


def kontra_full_outcome(rung, trump, sol, d1, d2, talon, seed=0):
    """Full-ladder kontra: simple contracts (parti/ulti/betli/duri) get the kontra
    decision + scoring; combos/100s fall back to god (kontra there is rare; the
    combined-kontra-in-play piece isn't wired). Most bids are simple, so this
    captures the dominant kontra value — weak hands passing instead of bleeding."""
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if (not (bid.forty_hundred or bid.twenty_hundred or bid.teritett)
            and n_trick <= 1):
        return kontra_god_outcome(rung, trump, sol, d1, d2, talon, seed)
    return god_outcome(rung, trump, sol, d1, d2, talon, seed)


def _play_weights(bid, sol, trump):
    has40, has20 = sol_marriages(sol, trump)
    w = dict(parti_pts=1.0, silent_ulti=2.0, silent_durchmars=3.0)
    if bid.ulti:
        w["silent_ulti"] = 8.0                     # play toward the bid ulti
    if bid.durchmars:
        w["silent_durchmars"] = 8.0
    if bid.forty_hundred:
        w["score_geq_100"] = 8.0
    elif bid.twenty_hundred:
        w["score_geq_100"] = 16.0
    else:
        w["score_geq_100"] = (2.0 if has40 else 0.0) + (1.0 if has20 else 0.0)
    return w


def pimc_outcome(rung, trump, sol, d1, d2, talon, seed=0, pimc_n=None):
    """Realistic (imperfect-info, closed-play) outcome: play the contract out
    MOVE-BY-MOVE with PIMC for all three seats, score the real terminal with the
    oracle. Move-by-move ⇒ no early-term PV issue ⇒ exact scoring for EVERY
    contract incl. silent games. Slower than god → the periodic check, smaller N."""
    import random
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import pimc_pick
    from scoring.oracle import score as score_oracle
    from trickster._solver_core import set_multi_weights

    if pimc_n is None:
        pimc_n = int(os.environ.get("PIMC_N", "16"))
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)

    if bid.betli:
        contract, t, restrict = "betli", None, None
    elif bid.durchmars and rung.colorless and n_trick == 1:
        contract, t, restrict = "durchmars", None, None
    else:
        contract, t = "multi", trump
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        set_multi_weights(**_play_weights(bid, sol, trump))

    bkw = dict(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
               contract=("parti" if contract == "multi" else contract),
               trump=t, talon=list(talon),
               declare_marriages=(t is not None), marriage_restrict=restrict)
    pos = pis.build_position(**bkw)
    voids = _det.Voids()
    vd = voids.as_dict()
    rng = random.Random(seed)
    mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        ch = pimc_pick(pos=pos, contract=contract, n_samples=pimc_n,
                       seed=seed * 31337 + mi, voids_dict=vd)
        if ch is None:
            ch = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch)
        vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, ch)
        mi += 1
    pvec = score_oracle(final_pos=pos, bid=bid)
    return pvec.total_per_def, _primary_made(bid, pvec)


def _hand_makeability(sol, d1, d2, trump, talon, primary, viewer, n, seed):
    """P(soloist makes `primary` | `viewer`'s own hand), by sampling opponents from
    the viewer's ROOT info set (own hand only — cheat-proof, no trick-1 cards) and
    god-solving each. milan-approved hand-based kontra signal."""
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import god_says_soloist_wins
    root = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                              leader=0, contract=primary, trump=trump,
                              talon=list(talon), declare_marriages=(trump is not None))
    iset = _det.build_info_set(root, viewer, primary, voids=None)
    import random
    rng = random.Random(seed)
    w = 0
    for _ in range(n):
        hands, tal = _det.sample_world(iset, rng)
        spos = (pis.clone_with_hands_and_talon(root, hands, tal)
                if iset.talon_known is None else pis.clone_with_hands(root, hands))
        w += 1 if god_says_soloist_wins(spos, contract=primary) else 0
    return w / float(n)


def kontra_pimc_outcome(rung, trump, sol, d1, d2, talon, seed=0):
    """REALISTIC + kontra for colored simple contracts (parti / ulti — ~90% of
    bids). Play is PIMC; the kontra level is decided HAND-BASED (each side estimates
    P(soloist makes) from its OWN hand — cheat-proof) and applied to the whole
    contract; scored with kontra. Kontra is a payoff multiplier (play unchanged), so
    we play once and apply the level at scoring. betli/duri/combos → pimc_outcome."""
    import os as _os
    from dataclasses import replace
    from bidding.kontra import kontra_level_for            # exp23 (on sys.path)
    from scoring.oracle import score as score_oracle

    bid = resolve_bidset(rung, sol, trump)
    colored_simple = (trump is not None and not bid.betli and not bid.durchmars
                      and not (bid.forty_hundred or bid.twenty_hundred or bid.teritett))
    if not colored_simple:
        return pimc_outcome(rung, trump, sol, d1, d2, talon, seed=seed)

    primary = "ulti" if bid.ulti else "parti"
    n_dec = int(_os.environ.get("KONTRA_NDET", "6"))
    p_def = _hand_makeability(sol, d1, d2, trump, talon, primary, 1, n_dec, seed + 11)
    p_sol = _hand_makeability(sol, d1, d2, trump, talon, primary, 0, n_dec, seed + 23)
    level = kontra_level_for(p_def, bid, p_sol=p_sol)

    # play with PIMC, then score the REAL terminal with kontra applied (kontra is a
    # payoff multiplier → play is unchanged). kontra_level shorthand scales every
    # unit incl. silent substitutes — the oracle handles it.
    import random
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import pimc_pick
    from trickster._solver_core import set_multi_weights
    pimc_n = int(_os.environ.get("PIMC_N", "16"))
    set_multi_weights(**_play_weights(bid, sol, trump))
    bkw = dict(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
               contract="parti", trump=trump, talon=list(talon),
               declare_marriages=True, marriage_restrict=None)
    pos = pis.build_position(**bkw)
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        ch = pimc_pick(pos=pos, contract="multi", n_samples=pimc_n,
                       seed=seed * 31337 + mi, voids_dict=vd)
        if ch is None:
            ch = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, ch); mi += 1
    pvec = score_oracle(final_pos=pos, bid=replace(bid, kontra_level=level))
    return pvec.total_per_def, _primary_made(bid, pvec)
