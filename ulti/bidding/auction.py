"""Ladder-aware 3-player talon-passing auction (mirrors auction_h2h's mechanics,
but bids the FULL 22-rung ladder via the composer, and chooses trump + discard).

NOT editing the deployed auction_h2h.py — this is the exp-23 full-ladder driver.

Flow (milan's rules): P0 opens (has the talon). To bid, a seat picks up the talon
(12 cards), searches every (trump × 2-card discard), and the composer picks the
best rung above the current high bid. P0 opens iff its best EV beats the pass
penalty (−2/def). A later seat overcalls iff its best rung above the current bid
out-values staying a DEFENDER of the current contract (= −holder's announced EV).
Auction ends after a full round of passes; last bidder wins.

v1 approximations (flagged): the defender value uses the HOLDER's own announced EV
as the estimate of its per-def strength (an overcaller can't see the holder's hand;
this treats the announced contract as revealing its EV). Greedy, no bluffing.
"""
from __future__ import annotations

import itertools
from ulti.config import env_float
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402

from ulti.bidding.ladder import GPTable, overcalls  # noqa: E402
from ulti.bidding.bidder import rung_ev  # noqa: E402
from ulti.bidding.deal import deal_12_10_10  # noqa: E402

TRUMPS = ("hearts", "acorns", "leaves", "bells")
PASS_PENALTY = 2.0   # opener forfeits −2/def by passing
# exp-20 canon: the DECISION value for a contract is a percentile (not the max)
# over the 66 discards — debiases the argmax-over-discards winner's-curse inflation
# (DEBIAS_BID). The discard actually PLAYED is still the argmax.
DEBIAS_PCTL = env_float("DEBIAS_PCTL", 0.80)


def _search(cards, probs_fn, current, gp, discards, pctl=DEBIAS_PCTL, allowed=None,
            floor=None, duri_mult=None, betli_real=None, rebetli_real=None):
    """Best (decision_ev, rung, trump, discard, hand10) over trump × discard.

    `probs_fn(hand10, trump, talon) → BaseProbs` supplies the perception (net,
    god, PIMC, …). For each (trump, rung), the decision EV = `pctl`-quantile of
    that contract's EV across the discards (debias); the played discard is the
    argmax. None if no biddable rung above `current`. pctl=None ⇒ plain max."""
    cand = overcalls(current)
    if allowed is not None:
        cand = [r for r in cand if r.index in allowed]   # restrict contract set (h2h ablation)
    n = len(cards)
    # BUGFIX (2026-07-21): `discards` arrives as a one-shot itertools generator; iterating
    # it inside the `for trump` loop exhausted it after the FIRST trump (hearts), so only
    # hearts (piros) contracts were ever evaluated — the bidder could NEVER bid a non-piros
    # colored contract (makk/zöld/tök parti/ulti/…). Materialize it so every trump is scored.
    discards = list(discards)
    # per (trump, rung.index): lists of (ev, discard_idx) + keep argmax bundle
    acc = {}
    for trump in TRUMPS:
        for disc in discards:
            ds = set(disc)
            hand10 = [cards[i] for i in range(n) if i not in ds]
            talon = [cards[i] for i in disc]
            pr = probs_fn(hand10, trump, talon)
            for rung in cand:
                ev = rung_ev(rung, pr, gp, floor=floor, duri_mult=duri_mult,
                             betli_real=betli_real, rebetli_real=rebetli_real)
                if ev is None:
                    continue
                key = (trump, rung.index)
                bundle = (ev, talon, hand10, rung)
                if key not in acc:
                    acc[key] = {"evs": [ev], "argmax": bundle}
                else:
                    acc[key]["evs"].append(ev)
                    if ev > acc[key]["argmax"][0]:
                        acc[key]["argmax"] = bundle
    best = None
    for (trump, _ridx), d in acc.items():
        evs = d["evs"]
        dec = max(evs) if pctl is None else float(np.quantile(evs, pctl))
        _, disc, hand10, rung = d["argmax"]
        if best is None or dec > best[0]:
            best = (dec, rung, None if rung.colorless else trump, disc, hand10)
    return best


def _best_pickup(hand_plus_talon, probs_fn, current, gp, allowed=None, pctl=DEBIAS_PCTL,
                 floor=None, duri_mult=None, betli_real=None, rebetli_real=None):
    return _search(hand_plus_talon, probs_fn, current, gp,
                   itertools.combinations(range(len(hand_plus_talon)), 2),
                   pctl=pctl, allowed=allowed, floor=floor, duri_mult=duri_mult,
                   betli_real=betli_real, rebetli_real=rebetli_real)


def _blind_best(hand10, probs_fn, current, gp, allowed=None, floor=None, duri_mult=None,
                betli_real=None, rebetli_real=None, blind_pctl=None):
    """Best (ev, rung, trump) from the 10 cards you were DEALT — the PICKUP decision.

    The talon is deliberately not a parameter: you cannot look at cards you have not
    picked up, and picking up commits you to announcing a game. Keeping it out of the
    signature is what makes the cheat unrepresentable rather than merely discouraged
    (it was representable before, and it came back — see the module note).

    DEBIAS_PCTL does not apply — that corrects an argmax over 66 DISCARDS, and there are
    no discards to choose between yet. But this stage has a winner's curse of its own: it
    maximises over 4 trumps × every legal rung, so the winner is the estimate that happens
    to be most optimistic. `blind_pctl` optionally replaces the max-over-TRUMPS for each
    rung with that quantile, which is the same correction applied where it now belongs.
    None = plain max (the behaviour as shipped).

    Worth knowing why this exists (measured 2026-08-02): once the pickup decision moved
    here, DEBIAS_PCTL went inert — even pctl=1.0, i.e. no debias at all, changed 0 of 40
    auction outcomes, because it can now only reorder contracts AFTER a seat has committed
    and the top rung wins by a wide margin anyway.
    """
    cand = overcalls(current)
    if allowed is not None:
        cand = [r for r in cand if r.index in allowed]
    per_rung = {}
    for trump in TRUMPS:
        pr = probs_fn(hand10, trump, None)
        for rung in cand:
            ev = rung_ev(rung, pr, gp, floor=floor, duri_mult=duri_mult,
                         betli_real=betli_real, rebetli_real=rebetli_real)
            if ev is None:
                continue
            per_rung.setdefault(rung.index, {"rung": rung, "evs": [], "best": None})
            e = per_rung[rung.index]
            e["evs"].append(ev)
            if e["best"] is None or ev > e["best"][0]:
                e["best"] = (ev, None if rung.colorless else trump)
    best = None
    for e in per_rung.values():
        score = (max(e["evs"]) if blind_pctl is None
                 else float(np.quantile(e["evs"], blind_pctl)))
        if best is None or score > best[0]:
            best = (score, e["rung"], e["best"][1])
    return best


def net_bid_fn(provider, gp=None, allowed=None, pctl=DEBIAS_PCTL, floor=None, duri_mult=None,
               betli_real=None, rebetli_real=None, pickup_model=None,
               pickup_floor=None, blind_pctl=None):
    """Build a bid_fn(hand10, talon, current, threshold, others) → bundle | None.

    TWO-STAGE, because Ulti is two-stage (milan 2026-08-02; originally fixed 2026-02-20
    in the since-deleted `apps/api/ulti.py`, then reintroduced when this bidder was
    written from scratch):

      1. PICKUP — decide from your own 10 cards alone whether the hand is worth taking
         the talon up for. `threshold` is what you must beat: the −2 pass penalty when
         opening, or the value of defending the standing contract when overcalling.
         Below it we return None WITHOUT EVER TOUCHING `talon`.
      2. ANNOUNCE — only now do the 12 cards exist. Search every (trump × discard) and
         name the game. You are committed, so if the confidence FLOOR leaves nothing
         biddable we re-search with the floor dropped rather than pass: having picked
         up, you must announce something.

    `others` is accepted for API uniformity but IGNORED — the net perceives only its own
    hand. `allowed` restricts the contract set (h2h ablations); `pctl`/`floor`/
    `duri_mult`/`betli_real`/`rebetli_real` set this bidder's CONFIG per call, so two
    configs can run head-to-head in one process. Perfect-perception bidders (god/PIMC)
    supply a probs_fn that uses `others`; see research/experiments/25_strength_ceiling.
    """
    gp = gp or GPTable()

    def _pf(hand10, trump, talon):
        return provider.base_probs(hand10, trump)

    def _bid(hand10, talon, current, threshold, others=None):
        if pickup_model is not None:
            # exp45: predict the value of the hand AFTER a pickup. The blind EV alone
            # understates it by ~+3 GP, because picking up lets you keep the best 10 of
            # 12 — evaluating the dealt 10 as if it were final makes the bidder passive.
            value = pickup_model(list(hand10), current)
        else:
            # The confidence FLOOR is an announce-stage guard, so `pickup_floor` (default:
            # the announce floor, i.e. legacy behaviour) governs this stage separately.
            blind = _blind_best(hand10, _pf, current, gp, allowed,
                                floor=(floor if pickup_floor is None else pickup_floor),
                                duri_mult=duri_mult, betli_real=betli_real,
                                rebetli_real=rebetli_real, blind_pctl=blind_pctl)
            value = None if blind is None else blind[0]
        if value is None or value <= threshold:
            return None                       # pass — the talon was never inspected
        twelve = list(hand10) + list(talon)
        picked = _best_pickup(twelve, _pf, current, gp, allowed, pctl=pctl, floor=floor,
                              duri_mult=duri_mult, betli_real=betli_real,
                              rebetli_real=rebetli_real)
        if picked is None:
            # Committed by the pickup: announce the best thing available with the
            # confidence floor lifted, instead of the impossible "put it back".
            picked = _best_pickup(twelve, _pf, current, gp, allowed, pctl=pctl, floor=0.0,
                                  duri_mult=duri_mult, betli_real=betli_real,
                                  rebetli_real=rebetli_real)
        return picked

    return _bid


def run_auction(seed, bid_fn, gp=None):
    """Architecture-agnostic 3-player talon-passing auction. `bid_fn` is either a
    single callable (all seats) or a list of 3 (per-seat, for head-to-head):

        bid_fn(hand10, talon, current_rung, threshold, others)
            → (decision_ev, rung, trump, discard, hand10) | None

    The hand and the talon are passed SEPARATELY and the pass/bid threshold is passed
    IN, so the bidder itself decides whether the hand is worth a pickup before it may
    look at the talon. A bidder that returns None has not seen those two cards."""
    if gp is None:
        gp = GPTable()
    fns = list(bid_fn) if isinstance(bid_fn, (list, tuple)) else [bid_fn] * 3
    sol12, d1, d2 = deal_12_10_10(seed)
    hands = [list(sol12[:10]), list(d1), list(d2)]
    talon = list(sol12[10:])

    op = fns[0](hands[0], talon, None, -PASS_PENALTY, (hands[1], hands[2]))
    if op is None:
        return {"seed": seed, "winner": None, "contract": "PASS",
                "trump": None, "ev": -PASS_PENALTY, "bid_seq": [],
                "n_bids": 0}
    ev, rung, trump, discard, hand10 = op
    hands[0] = hand10
    talon = discard
    current = {"pid": 0, "rung": rung, "trump": trump, "ev": ev}
    bid_seq = [(0, rung.name, trump, round(ev, 2))]

    passes = 0
    nxt = 1
    while passes < 3:
        if nxt == current["pid"]:
            passes += 1
        else:
            defender_value = -current["ev"]
            others = tuple(hands[s] for s in range(3) if s != nxt)
            pick = fns[nxt](hands[nxt], talon, current["rung"], defender_value, others)
            if pick is None:
                passes += 1
            else:
                ev2, rung2, trump2, disc2, hand10b = pick
                hands[nxt] = hand10b
                talon = disc2
                current = {"pid": nxt, "rung": rung2, "trump": trump2, "ev": ev2}
                bid_seq.append((nxt, rung2.name, trump2, round(ev2, 2)))
                passes = 0
        nxt = (nxt + 1) % 3

    w = current["pid"]
    return {"seed": seed, "winner": w, "rung": current["rung"],
            "contract": current["rung"].name, "trump": current["trump"],
            "ev": current["ev"], "bid_seq": bid_seq, "n_bids": len(bid_seq),
            "sol": hands[w], "def1": hands[(w + 1) % 3], "def2": hands[(w + 2) % 3],
            "talon": talon}


if __name__ == "__main__":
    from ulti.bidding.provider import NetProvider
    prov = NetProvider()
    for seed in range(5):
        r = run_auction(seed, net_bid_fn(prov))
        print(f"seed {seed}: winner P{r['winner']} → {r['contract']}"
              f"/{r['trump']}  bids={r['bid_seq']}")
