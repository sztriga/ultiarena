"""Auction state machine + auction->play setup (any seat may open; AI turns resolve synchronously)."""


from typing import Dict, List, Optional

from ulti.bidding.ladder import overcalls
from ulti.bidding.bidder import rung_ev
from ulti.bidding.auction import PASS_PENALTY
from ulti.bidding.scorers import resolve_bidset, _play_weights
from ulti.solvers import pis as pis_bridge
from ulti.solvers import determinize as _det
from ulti.scoring.units import kontra_units as _kontra_units
from fastapi import HTTPException

from .engine import Session, _GP, _SUIT_HU, _bid_fn, _bid_label, _provider
from .ai_play import _advance_play


# ── Auction (any seat may open) ─────────────────────────────────────────────────

def _bid_ai(sess: Session, pid: int, current_rung, threshold: float) -> Optional[tuple]:
    """One AI seat's auction turn. The hand and the talon go in SEPARATELY, with the
    pass/bid threshold, so the bidder makes the pickup decision blind — it may only look
    at the talon once it has committed to announcing a game (ulti.bidding.auction)."""
    return _bid_fn()(list(sess.a_hands[pid]), list(sess.a_talon), current_rung,
                     threshold, None)


def _weakest_two(cards12: list, trump: Optional[str]):
    """The 2 cards least worth keeping — and least useful to whoever picks up the
    talon next: non-trump, low card-points, low rank first. Returns (discard2,
    keep10). Used when a player PASSES (buries the 2 junk cards as the new talon).

    We deliberately KEEP 7s (shed 8s/9s ahead of them): a buried 7 — the trump 7
    is the ulti card — could make the next player's ulti. (milan 2026-07)"""
    def junk_key(c):
        is_trump = 1 if (trump is not None and c.suit == trump) else 0
        is_seven = 1 if c.rank == "7" else 0
        return (is_trump, c.points, is_seven, c.rank_index)
    ordered = sorted(cards12, key=junk_key)
    return ordered[:2], ordered[2:]


def _apply_bid(sess: Session, pid: int, bundle: tuple, bid_override=None) -> None:
    ev, rung, trump, discard, hand10 = bundle
    sess.a_hands[pid] = list(hand10)
    sess.a_talon = list(discard)
    # Pin the SPECIFIC game on this rung. The human names it explicitly (bid_index);
    # the AI resolves it from its own hand. This is what breaks the two
    # interchangeable "≡" contracts (40-100-duri vs ulti-duri) apart.
    bid = bid_override if bid_override is not None else resolve_bidset(rung, hand10, trump)
    sess.a_current = {"pid": pid, "rung": rung, "trump": trump, "ev": ev, "bid": bid}
    sess.a_history.append({
        "pid": pid, "kind": "bid",
        "contract": _bid_label(bid), "trump": trump, "rung_index": rung.index,
    })


def _human_bundle(sess: Session, rung, trump: Optional[str], discard_ids: List[int],
                  seat: Optional[int] = None) -> tuple:
    seat = sess.seat if seat is None else seat
    cards12 = list(sess.a_hands[seat]) + list(sess.a_talon)
    discard = [c for c in cards12 if c.id in discard_ids]
    if len(discard) != 2:
        raise HTTPException(status_code=400, detail="must discard exactly 2 of your 12 cards")
    hand10 = [c for c in cards12 if c.id not in discard_ids]
    # EV for the AI's overcall threshold. A colored contract whose trump is
    # DEFERRED (you pick the suit after the auction) is scored at the BEST over the
    # candidate suits — your true strength, without revealing which suit you'll play.
    # (betli/színtelen-duri use a placeholder trump; their EV is colorless-only.)
    if trump is not None:                       # piros (hearts) or an explicit suit
        ev = rung_ev(rung, _provider().base_probs(hand10, trump), _GP)
    elif rung.colorless:
        ev = rung_ev(rung, _provider().base_probs(hand10, "hearts"), _GP)
    else:                                        # non-piros colored → best over 3 suits
        cand = [rung_ev(rung, _provider().base_probs(hand10, t), _GP)
                for t in ("acorns", "leaves", "bells")]
        cand = [e for e in cand if e is not None]
        ev = max(cand) if cand else None
    ev = float(ev) if ev is not None else float(rung.value)
    return (ev, rung, (None if rung.colorless else trump), discard, hand10)


def _advance_auction(sess: Session) -> None:
    """Run AI bid turns until it's the user's turn or the auction resolves. Any
    seat may open; a full round of passes (3 consecutive) ends it."""
    while True:
        if sess.a_passes >= 3:
            # Everyone passed with no bid → the forehand (the 12-holder, seat 0) gets
            # ONE last look: pick the talon back up and play, or passz for good and
            # pay the penalty. Only offered interactively when the forehand is a
            # HUMAN seat; an AI forehand already declined, so it just pays.
            if sess.a_current is None and not sess.a_reclaim_offered and 0 in sess.humans:
                sess.a_reclaim_offered = True
                sess.a_turn = 0
                sess.a_awaiting_bid = False    # auction step: Felveszem (play) or Passz (pay)
                return
            _resolve_auction(sess)
            return
        turn = sess.a_turn
        if sess.a_current is not None and turn == sess.a_current["pid"]:
            # Everyone else has passed and it is back to the holder. The holder may
            # take the talon back up and RAISE their own bid — the bluff: win cheap,
            # then climb to your real contract. A human decides (pause here); an
            # AI holder declines (it always bids its true best, nothing to raise to).
            if turn in sess.humans:
                return
            sess.a_passes += 1
            sess.a_turn = (turn + 1) % 3
            continue
        if turn in sess.humans:
            return  # a human decides (open or overcall)
        # AI seat
        if sess.a_current is None:
            pick = _bid_ai(sess, turn, None, -PASS_PENALTY)
        else:
            pick = _bid_ai(sess, turn, sess.a_current["rung"], -sess.a_current["ev"])
        if pick is not None:
            _apply_bid(sess, turn, pick)
            sess.a_passes = 0
        else:
            # Only the FOREHAND puts the talon back (buries 2) when passing the
            # opening — it's the one holding the 12. Sheds the 2 least-useful cards
            # so it hands the next picker-up junk. Later passers never held the
            # talon, so they decline without touching it.
            if sess.a_current is None and not sess.a_history:
                disc, hand10 = _weakest_two(list(sess.a_hands[turn]) + list(sess.a_talon), None)
                sess.a_hands[turn] = hand10
                sess.a_talon = disc
            sess.a_history.append({"pid": turn, "kind": "pass"})
            sess.a_passes += 1
        sess.a_turn = (turn + 1) % 3


def _resolve_auction(sess: Session) -> None:
    sess.a_done = True
    if sess.a_current is None:
        sess.a_winner = None
        _finish_passed(sess)        # dead deal → SCORED like any round (see _finish_passed)
        return
    sess.a_winner = sess.a_current["pid"]
    # The human won a plain colored game with the trump deferred → declare it now,
    # before play (Ulti: the color is hidden until the game begins). AI winners and
    # piros/colorless games already carry their trump.
    rung = sess.a_current["rung"]
    if sess.a_winner in sess.humans and sess.a_current["trump"] is None and not rung.colorless:
        sess.phase = "trump_select"
        return
    _setup_play(sess)
    _advance_play(sess)


def _finish_passed(sess: Session) -> None:
    """Nobody bid (and the forehand declined the reclaim) → the deal is scored like
    any other round: passz is on the ladder, we just never PLAY it. The forehand
    (real seat 0, the 12-holder) forfeits PASS_PENALTY per defender, the result box
    appears exactly as after a played hand (phase "passed" → the UI's scoring window
    + round tally; Elemzés is meaningless, the client greys it), and Következő
    ROTATES the 12-holder like after any round (milan 2026-08-02)."""
    pen = float(PASS_PENALTY)
    seat_gp = [-2.0 * pen, pen, pen]          # real-seat space; forehand = real seat 0
    human_gp = seat_gp[sess.seat]
    sess.phase = "passed"
    sess.result = {
        "winner": "defenders",                # the two non-openers collect
        "made": False,
        "sol_gp_per_def": -pen,
        "human_gp": float(human_gp),
        "user_won": human_gp > 0,
        "contract": "passz",
        "kontra_level": 0,
        "seat_gp": seat_gp,
        "soloist_seat": 0,                    # the payer — the forehand
        "silents": [],
    }


# ── Auction -> Play setup ───────────────────────────────────────────────────────

# Hungarian suit names for the adu (trump) announcement — the declarer names the
# color out loud before the first card ("Színe tök!"). The specific suit stays
# hidden throughout the auction and is only revealed here.
def _setup_play(sess: Session) -> None:
    """Re-index the winner to play-index 0 and build the pis position (no play yet)."""
    w = sess.a_winner
    sol = list(sess.a_hands[w])
    d1 = list(sess.a_hands[(w + 1) % 3])
    d2 = list(sess.a_hands[(w + 2) % 3])
    talon = list(sess.a_talon)
    rung = sess.a_current["rung"]
    trump = sess.a_current["trump"]

    sess.rung = rung
    sess.trump = trump
    sess.human_play_index = (sess.seat - w) % 3
    sess.human_pis = {(s - w) % 3 for s in sess.humans}   # human PLAY indices (== {hpi} solo)

    bid = sess.a_current["bid"]
    sess.bid = bid
    sess.bid_name = _bid_label(bid)

    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        solve_c, build_c, t, restrict, weights = "betli", "betli", None, None, None
    elif bid.durchmars and rung.colorless and n_trick == 1:
        solve_c, build_c, t, restrict, weights = "durchmars", "durchmars", None, None, None
    else:
        solve_c, build_c, t = "multi", "parti", trump
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        weights = _play_weights(bid, sol, trump)

    sess.p_solve_contract = solve_c
    sess.p_build_contract = build_c
    sess.p_restrict = restrict
    sess.p_weights = weights
    sess.play_hands0 = [sol, d1, d2]
    sess.play_talon = talon

    # Building the position never solves — no weights, no lock. The weight vector
    # travels with every worker job instead (sess.p_weights via _recipe).
    sess.p_pos = pis_bridge.build_position(
        hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
        contract=build_c, trump=t, talon=list(talon),
        declare_marriages=(t is not None), marriage_restrict=restrict,
        has_ulti=bool(bid.ulti),   # 7esre tartás: hold the trump 7 when the game has an ulti
    )
    sess.voids = _det.Voids()
    # Live kontra-able units: a simple game has one (its primary); a combined game
    # exposes each committed unit separately. Colorless (betli / no-trump duri) keep
    # separate per-defender counters; colored units are shared (együtt sírunk).
    sess.k_units = _kontra_units(bid)
    sess.k_colorless = (trump is None)
    sess.k_def = {U: {1: False, 2: False} for U in sess.k_units}
    sess.k_rekontra = {U: False for U in sess.k_units}
    sess.phase = "play"

    # Marriage announcements (bemondás) → "Van 40-em!" / "Van 20-am!".
    # NOT in a bid 40-100/20-100: the declared 100 announces the marriage by itself,
    # and no other marriage scores anything there (the 100 replaces the párti, so
    # even the silent riders are off) — the bubbles would be pure noise (milan).
    marr: Dict[int, str] = {}
    announce_marriages = not (bid.forty_hundred or bid.twenty_hundred)
    for player in (range(3) if announce_marriages else ()):
        parts = [f"Van {pts}-{'em' if pts == 40 else 'am'}!"
                 for (p, _suit, pts) in getattr(sess.p_pos, "marriages", []) if p == player]
        if parts:
            marr[player] = " ".join(parts)
    # Defenders call their marriage as they play their first card (play-index P at ply P).
    for player in (1, 2):
        if player in marr:
            sess.bubbles.append({"player": player, "text": marr[player], "ply": player})
    # The soloist declares the adu color AND their own marriage together, up front
    # (ply=-1 → the instant play starts) — one bubble so BOTH are visible instead of
    # the trump announcement clobbering the marriage: "Színe piros! Van 40-em!".
    if sess.trump is not None:
        txt = f"Színe {_SUIT_HU.get(sess.trump, sess.trump)}!"
        if 0 in marr:
            txt += " " + marr[0]
        sess.bubbles.append({"player": 0, "text": txt, "ply": -1})
    elif 0 in marr:                              # trumpless guard (colorless games have no marriage)
        sess.bubbles.append({"player": 0, "text": marr[0], "ply": 0})



def _legal_bids(sess: Session) -> List[dict]:
    current = sess.a_current["rung"] if sess.a_current else None
    out: List[dict] = []
    # passz only on the forehand's VERY FIRST turn (the opening). Once you've picked
    # the talon up (an overcall, or the all-pass reclaim), you're committed to a bid,
    # and start-play / accept is handled by the auction-step Elfogadom button.
    if sess.a_current is None and not sess.a_history:
        out.append({"kind": "pass", "rung_index": -1, "bid_index": -1,
                    "label": "passz", "value": -2,
                    "piros": False, "colorless": False, "trump_options": []})
    for r in overcalls(current):
        if r.colorless:
            trumps = []
        elif r.piros:
            trumps = ["hearts"]
        else:
            trumps = ["bells", "leaves", "acorns"]   # tök · zöld · makk (milan's order);
            # display-only for deferred-trump bids: play_bid recomputes the trump
        # One option per interchangeable contract on the rung, so the user picks
        # the specific game (40-100-duri vs ulti-duri) rather than us guessing.
        for bi, b in enumerate(r.bids):
            out.append({
                "kind": "bid", "rung_index": r.index, "bid_index": bi,
                "label": _bid_label(b), "value": r.value,
                "piros": r.piros, "colorless": r.colorless, "trump_options": trumps,
            })
    return out


