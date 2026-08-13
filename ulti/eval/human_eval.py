"""Counterfactual evaluation of recorded games: replay every human decision through the
engine and price the disagreement in GP.

This is DIVAT (Billings & Kan, computer poker) applied to Ulti: assess each DECISION
against a baseline policy rather than assessing the outcome, because the outcome is
mostly luck. A piros ulti kontra swings +-96 GP, so raw results need hundreds of games to
say anything; a per-decision measure says something after a dozen.

WHAT THE NUMBER MEANS, precisely. It is "agreement with our engine, priced in GP" — NOT
"skill". The literature is explicit that DIVAT-style measures are baseline-dependent, and
ours has known blind spots (exp48: the solver's objective is a sum of binary indicators
and cannot see card points; exp46: per-world double-dummy is optimistic about what it can
arrange later). A player who is right where the engine is wrong scores as wrong. That is
a limitation to state on the label, not a reason to avoid the measurement — a human who
deviates and consistently WINS is exactly how we would find an engine error.

FIVE DECISION BUCKETS, because GP is lost in different places for different reasons:

    open      enter the auction at all, from your dealt 10   (exact)
    contract  given you entered, which rung                  (needs the 12 you held)
    discard   which two to bury                              (needs the 12 you held)
    kontra    double, per live unit                          (approximate — see below)
    play      every card                                     (exact, GP regret)

Per bucket we report the agreement rate AND the GP split by agreement. The second is the
one that matters: "you deviate 40% of the time" says nothing without "and it costs 6 GP a
game".

RECONSTRUCTION. The database stores the final hands, not the dealt ones, so a bidder's
pre-pickup hand looks unrecoverable. It is not: the `seed` is recorded and
`deal_12_10_10` is deterministic, so the whole deal replays exactly. Every bucket below
is built on that rather than on the stored hands.

Two known gaps, reported as coverage rather than hidden:
  * OVERCALL decisions need the threshold the seat actually faced, which is the standing
    bidder's announced EV — reproducible in principle, not yet done. Counted and skipped.
  * KONTRA on a COLORED unit is shared between the defenders (együtt sírunk), and the
    transcript stores per-unit levels without attribution, so "who doubled" is ambiguous.
    Colourless units keep separate counters and are exact.

Run:
    python3 -m ulti.eval.human_eval                 # all recorded games
    python3 -m ulti.eval.human_eval --vs-ai         # only games with AI opponents
    python3 -m ulti.eval.human_eval --user <id>
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

from ulti.bidding.auction import _best_pickup, _blind_best, DEBIAS_PCTL   # noqa: E402
from ulti.bidding.deal import deal_12_10_10                              # noqa: E402
from ulti.bidding.frontier import (BETLI_REAL_BID, REBETLI_REAL_BID,     # noqa: E402
                                   frontier_provider)
from ulti.bidding.ladder import LADDER, GPTable, overcalls               # noqa: E402
from ulti.bidding.scorers import resolve_bidset, _play_weights           # noqa: E402
from ulti.card import card_from_id                                       # noqa: E402
from ulti.scoring.units import kontra_units                              # noqa: E402

PASS_PENALTY = 2.0
_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "games.db")

# exp27 deployed defender gates, mirrored from apps/api/kontra_flow (parti removed exp47)
_KONTRA_ULTI_TRUMPS = 4
_KONTRA_DURI_TRUMPS = 3


# ── one decision ────────────────────────────────────────────────────────────────

def _dec(bucket, seat, is_human, agreed, gp, **extra) -> dict:
    d = {"bucket": bucket, "seat": seat, "human": is_human, "agreed": agreed, "gp": gp}
    d.update(extra)
    return d


def _weakest_two(cards12, trump=None):
    """What the forehand buries when it declines to open (apps/api/auction_flow)."""
    def junk(c):
        return (1 if (trump is not None and c.suit == trump) else 0,
                c.points, 1 if c.rank == "7" else 0, c.rank_index)
    o = sorted(cards12, key=junk)
    return o[:2], o[2:]


class Evaluator:
    def __init__(self):
        self.prov = frontier_provider()
        self.gp = GPTable()
        self.pf = lambda h, t, tal: self.prov.base_probs(h, t)
        self.skipped: Dict[str, int] = defaultdict(int)

    # ── bidding ──
    def _blind(self, hand10, current=None):
        return _blind_best(hand10, self.pf, current, self.gp,
                           betli_real=BETLI_REAL_BID, rebetli_real=REBETLI_REAL_BID)

    def _announce(self, twelve, current=None):
        return _best_pickup(twelve, self.pf, current, self.gp, pctl=DEBIAS_PCTL,
                            betli_real=BETLI_REAL_BID, rebetli_real=REBETLI_REAL_BID)

    def auction(self, rec, dealt, humans) -> List[dict]:
        """Walk the auction in order. Openings are exact; overcalls are counted and
        skipped (the threshold a later seat faced is the standing bidder's announced EV,
        which this version does not reconstruct)."""
        out: List[dict] = []
        # Cards, not ids, from here down — the bidder works on Card objects while the
        # analysis job below wants ids, so the two are converted at their own boundary.
        talon = [card_from_id(i) for i in dealt["talon"]]
        standing = False
        my_gp = rec["seat_gp"]
        for a in rec["auction"]:
            seat = a["pid"]
            is_human = seat in humans
            if standing:
                self.skipped["overcall"] += 1
                if a["kind"] == "bid":
                    break                       # later talon state becomes unrecoverable
                continue
            hand10 = [card_from_id(i) for i in dealt["hands"][seat]]
            b = self._blind(hand10)
            ev = b[0] if b else -99.0
            eng_bid = ev > -PASS_PENALTY
            did_bid = a["kind"] == "bid"
            out.append(_dec("open", seat, is_human, eng_bid == did_bid, my_gp[seat],
                            did="bid" if did_bid else "pass",
                            engine="bid" if eng_bid else "pass",
                            engine_ev=round(ev, 2),
                            engine_rung=(b[1].name if b else None)))
            if did_bid:
                # This seat picked the talon up: contract choice + discard, on the real 12.
                twelve = hand10 + talon
                pick = self._announce(twelve)
                if pick is None:
                    self.skipped["contract"] += 1
                else:
                    _ev, rung, _tr, disc, _h10 = pick
                    out.append(_dec("contract", seat, is_human,
                                    rung.index == a["rung_index"], my_gp[seat],
                                    did=a["contract"], engine=rung.name))
                    # Discard is only checkable for the FINAL bidder, whose buried pair
                    # is what the transcript stores.
                    if not any(x["kind"] == "bid" for x in rec["auction"]
                               if rec["auction"].index(x) > rec["auction"].index(a)):
                        theirs = set(dealt["final_talon"])
                        out.append(_dec("discard", seat, is_human,
                                        theirs == {c.id for c in disc}, my_gp[seat],
                                        did=sorted(theirs),
                                        engine=sorted(c.id for c in disc)))
                    else:
                        self.skipped["discard"] += 1
                standing = True
            else:
                if not any(x["kind"] == "bid" for x in rec["auction"][:rec["auction"].index(a)]):
                    _d, keep = _weakest_two(hand10 + talon)
                    if len(hand10) == 10 and rec["auction"].index(a) == 0:
                        talon, _ = _weakest_two(hand10 + talon)
        return out

    # ── kontra ──
    def kontra(self, rec, dealt, humans, bid, trump, sol_seat) -> List[dict]:
        out = []
        units = kontra_units(bid)
        if not units:
            return out
        levels = rec["kontra"] or {}
        for seat in (0, 1, 2):
            if seat == sol_seat:
                continue
            own = [card_from_id(i) for i in dealt["hands"][seat]]
            ntr = sum(1 for c in own if trump and c.suit == trump)
            for U in units:
                if U == "ulti":
                    eng = ntr >= _KONTRA_ULTI_TRUMPS
                elif U == "durchmars" and trump is not None:
                    eng = ntr >= _KONTRA_DURI_TRUMPS
                elif U == "40_100" and trump is not None:
                    eng = any(c.suit == trump and c.rank in ("king", "upper") for c in own)
                else:
                    eng = False
                lv = levels.get(U, 0)
                did = bool(lv[seat - 1] if isinstance(lv, list) else lv)
                if trump is not None and not isinstance(lv, list):
                    self.skipped["kontra_attrib"] += 1   # shared: cannot attribute
                out.append(_dec("kontra", seat, seat in humans, eng == did,
                                rec["seat_gp"][seat], unit=U,
                                did="kontra" if did else "pass",
                                engine="kontra" if eng else "pass"))
        return out

    # ── play ──
    def play(self, rec, dealt, humans, bid, sol_seat, worlds: int) -> List[dict]:
        from apps.api import ai_worker
        if not rec["plays"]:
            return []
        trump = rec["trump"]
        sol10 = dealt["play_hands"][0]
        n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
        if bid.betli:
            bc, sc, t, rs, w = "betli", "betli", None, None, None
        elif bid.durchmars and trump is None and n_trick == 1:
            bc, sc, t, rs, w = "durchmars", "durchmars", None, None, None
        else:
            bc, sc, t = "parti", "multi", trump
            rs = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
            w = _play_weights(bid, [card_from_id(i) for i in sol10], trump)
        rows = ai_worker.op_analysis({
            "hands0": dealt["play_hands"], "talon": dealt["final_talon"],
            "build_c": bc, "solve_c": sc, "trump": t, "restrict": rs,
            "has_ulti": bool(bid.ulti), "weights": w,
            "history": [(p, c) for p, c, *_ in rec["plays"]],
            "bid": bid, "seed": rec["seed"], "analysis_worlds": worlds,
            "knowable_always": True})
        out = []
        for r in rows:
            if "gp_loss" not in r:
                self.skipped["play"] += 1
                continue
            seat = (r["player_id"] + sol_seat) % 3          # play index → real seat
            # Info-set regret is PRIMARY when available — double-dummy hindsight is far
            # too forgiving (over 186 real moves it flagged exactly one). Hindsight is
            # kept alongside as "what it actually cost".
            kn = r.get("gp_loss_knowable")
            primary = kn if kn is not None else r["gp_loss"]
            out.append(_dec("play", seat, seat in humans, primary <= 0.001,
                            -primary, ply=r["ply_index"],
                            loss=primary, hindsight=r["gp_loss"],
                            severity=r.get("severity")))
        return out


# ── driver ──────────────────────────────────────────────────────────────────────

def load(db_path: str, vs_ai: Optional[bool] = None, user: Optional[str] = None):
    con = sqlite3.connect(db_path)
    out = []
    for row in con.execute("select id,seed,contract,trump,soloist_seat,human_seat,"
                           "kontra_level,winner,made,seat_gp,players,transcript "
                           "from games order by created_at"):
        players = json.loads(row[10])
        humans = {p["seat"] for p in players if p["kind"] in ("human", "bot")}
        if vs_ai is True and len(humans) != 1:
            continue
        if vs_ai is False and len(humans) < 2:
            continue
        if user and not any((p.get("user_id") == user) for p in players):
            continue
        t = json.loads(row[11])
        out.append({"id": row[0], "seed": row[1], "contract": row[2], "trump": row[3],
                    "soloist_seat": row[4], "kontra_level": row[6], "made": row[8],
                    "seat_gp": json.loads(row[9]), "humans": humans,
                    "auction": t["auction"], "plays": t["plays"], "kontra": t["kontra"],
                    "stored_hands": t["deal"]["hands"], "final_talon": t["deal"]["talon"]})
    return out


def evaluate(games, worlds: int = 0) -> tuple:
    ev = Evaluator()
    decisions: List[dict] = []
    for g in games:
        sol12, d1, d2 = deal_12_10_10(g["seed"])
        passed = g["contract"] == "passz"
        sol_seat = g["soloist_seat"]
        dealt = {
            "hands": [[c.id for c in sol12[:10]], [c.id for c in d1], [c.id for c in d2]],
            "talon": [c.id for c in sol12[10:]],
            "final_talon": g["final_talon"],
            "play_hands": g["stored_hands"],
        }
        decisions += ev.auction(g, dealt, g["humans"])
        if not passed:
            rung = next(r for r in LADDER
                        if r.name == next(a["contract"] for a in g["auction"]
                                          if a["kind"] == "bid"))
            bid = resolve_bidset(rung, [card_from_id(i) for i in dealt["play_hands"][0]],
                                 g["trump"])
            decisions += ev.kontra(g, dealt, g["humans"], bid, g["trump"], sol_seat)
            decisions += ev.play(g, dealt, g["humans"], bid, sol_seat, worlds)
    return decisions, ev.skipped


def report(decisions, skipped, n_games=0, log=print) -> None:
    """Two tables, because the buckets measure different things.

    For the BIDDING buckets a decision is followed by one game outcome, so the useful
    split is "what did you score when you agreed with the engine vs when you didn't".
    For PLAY there are ~30 decisions per game sharing one outcome, so that split is
    meaningless — there the quantity is the GP each move cost, summed per game.
    """
    by = defaultdict(lambda: defaultdict(list))
    for d in decisions:
        by[d["bucket"]]["human" if d["human"] else "ai"].append(d)

    log(f"\n=== BIDDING · outcome by agreement with the engine ===")
    log(f"{'bucket':<10s} {'who':<6s} {'n':>5s} {'agree':>7s} "
        f"{'GP|agree':>9s} {'GP|differ':>10s}")
    log("-" * 52)
    for bucket in ("open", "contract", "discard", "kontra"):
        for who in ("human", "ai"):
            ds = by[bucket][who]
            if not ds:
                continue
            n = len(ds)
            ag = [d["gp"] for d in ds if d["agreed"]]
            di = [d["gp"] for d in ds if not d["agreed"]]
            f_ag = f"{sum(ag)/len(ag):9.2f}" if ag else f"{'—':>9s}"
            f_di = f"{sum(di)/len(di):10.2f}" if di else f"{'—':>10s}"
            log(f"{bucket:<10s} {who:<6s} {n:5d} {100*len(ag)/n:6.0f}% {f_ag} {f_di}")

    log(f"\n=== PLAY · GP given away, info-set regret ===")
    log(f"{'who':<6s} {'moves':>6s} {'clean':>7s} {'GP/move':>8s} {'GP/seat-game':>13s} "
        f"{'worst':>7s}")
    log("-" * 52)
    per_move = {}
    for who in ("human", "ai"):
        ds = by["play"][who]
        if not ds:
            continue
        losses = [d["loss"] for d in ds]
        pm = sum(losses) / len(ds)
        per_move[who] = pm
        # AI seats outnumber the human 2:1, so a raw per-GAME figure flatters the human.
        # Normalise by SEAT-games instead.
        seats = 2 if who == "ai" else 1
        log(f"{who:<6s} {len(ds):6d} "
            f"{100*sum(1 for x in losses if x <= 0.001)/len(ds):6.0f}% {pm:8.3f} "
            f"{sum(losses)/(n_games*seats) if n_games else float('nan'):13.2f} "
            f"{max(losses):7.1f}")
    if "human" in per_move and "ai" in per_move:
        # The AI seats are the CONTROL. The engine is within ~0.52 GP/deal of perfect
        # play (the god-ceiling measurement), so whatever regret this harness assigns
        # THEM is very largely its own sampling noise. The human's EXCESS over that is
        # the part worth believing.
        log(f"\n  noise floor (AI seats): {per_move['ai']:.3f} GP/move")
        log(f"  human excess over it  : {per_move['human'] - per_move['ai']:+.3f} GP/move")

    sev = defaultdict(lambda: defaultdict(int))
    for d in by["play"]["human"] + by["play"]["ai"]:
        if d.get("severity") and d["severity"] != "ok":
            sev["human" if d["human"] else "ai"][d["severity"]] += 1
    if sev:
        log("\n  flagged moves:")
        for who, c in sev.items():
            log(f"    {who:<6s} " + "  ".join(f"{k} {v}" for k, v in sorted(c.items())))

    if skipped:
        log("\nNOT MEASURED (coverage gaps, not zeros):")
        for k, v in sorted(skipped.items()):
            log(f"  {k:<16s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_DB)
    ap.add_argument("--vs-ai", action="store_true", help="only games with 1 human")
    ap.add_argument("--vs-human", action="store_true", help="only games with 2+ humans")
    ap.add_argument("--user", default=None)
    ap.add_argument("--worlds", type=int, default=0,
                    help="sampled worlds for the 'knowable' split (0 = skip, faster)")
    a = ap.parse_args()
    vs = True if a.vs_ai else (False if a.vs_human else None)
    games = load(a.db, vs_ai=vs, user=a.user)
    print(f"{len(games)} games")
    decisions, skipped = evaluate(games, worlds=a.worlds)
    report(decisions, skipped, n_games=len(games))


if __name__ == "__main__":
    main()
