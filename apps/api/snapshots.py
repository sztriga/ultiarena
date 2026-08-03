"""JSON snapshots of a Session for the web UI (auction / kontra / play / result views)."""


from typing import List, Optional

from ulti.bidding.ladder import overcalls
from ulti.solvers import pis as pis_bridge
from ulti.card import sort_hand

from .serialize import card_to_dict
from .engine import Session, _bid_label
from .auction_flow import _legal_bids
from .kontra_flow import _UNIT_HU
from .ai_play import _terit_revealed


def _auction_snapshot(sess: Session) -> dict:
    is_turn = (not sess.a_done) and (sess.a_turn == sess.seat)
    # You are the holder deciding whether to raise iff it's your turn and you own
    # the standing bid (everyone else has passed back to you).
    is_holder = is_turn and sess.a_current is not None and sess.a_current["pid"] == sess.seat
    # forehand's post-all-pass last look: pick the talon back up (bid) or passz → pay
    reclaim = is_turn and sess.a_reclaim_offered and sess.a_current is None
    awaiting_bid = is_turn and sess.a_awaiting_bid        # bid step: you hold 12
    # auction step (holding 10): you may pick the talon up iff a legal bid exists
    cur_rung = sess.a_current["rung"] if sess.a_current else None
    can_pickup = is_turn and (not sess.a_awaiting_bid) and bool(overcalls(cur_rung))
    own = sort_hand(sess.a_hands[sess.seat])
    bid_hand = None
    talon_ids: List[int] = []
    if awaiting_bid:                # bid step → reveal your 10 + the talon (12)
        talon_ids = [c.id for c in sess.a_talon]
        twelve = sort_hand(list(sess.a_hands[sess.seat]) + list(sess.a_talon))
        bid_hand = [card_to_dict(c) for c in twelve]
    cur = None
    if sess.a_current is not None:
        cur = {
            "pid": sess.a_current["pid"],
            "contract": _bid_label(sess.a_current["bid"]),
            # Ulti keeps the adu (trump) color hidden during the auction — the
            # contract name alone is public (piros/colorless are named in it). The
            # specific suit is announced only when play begins (see _setup_play).
            "trump": None,
            "rung_index": sess.a_current["rung"].index,
        }
    return {
        "own_hand": [card_to_dict(c) for c in own],
        "bid_hand": bid_hand,
        "talon_ids": talon_ids,
        "auction": {
            "turn": None if sess.a_done else sess.a_turn,
            "current": cur,
            "passes": sess.a_passes,
            # Strip the trump from bid history too — hidden until announced.
            "history": [dict(h, trump=None) for h in sess.a_history],
            "done": sess.a_done,
            "winner": sess.a_winner,
            "is_human_turn": is_turn,
            "is_holder": is_holder,
            "reclaim": reclaim,
            "awaiting_bid": awaiting_bid,
            "can_pickup": can_pickup,
            "opening": sess.a_current is None,
            "picked_up": awaiting_bid and sess.a_picked_up,
            "legal_bids": _legal_bids(sess) if awaiting_bid else None,
        },
    }


def _play_hands_dict(sess: Session, reveal_all: bool, reveal_sol: bool = False) -> List[List[Optional[dict]]]:
    hands = pis_bridge.hands_by_player(sess.p_pos)
    colorless = sess.trump is None                      # betli / színtelen duri
    out: List[List[Optional[dict]]] = []
    for pid in range(3):
        h = sort_hand(hands[pid], colorless)
        # terített reveal: the soloist (play-index 0) is shown to everyone once open
        if reveal_all or pid == sess.human_play_index or (reveal_sol and pid == 0):
            out.append([card_to_dict(c) for c in h])
        else:
            out.append([None for _ in h])
    return out


def _play_snapshot(sess: Session) -> dict:
    terminal = pis_bridge.is_terminal(sess.p_pos)
    in_play = sess.phase == "play" and not terminal
    current = pis_bridge.current_player(sess.p_pos) if in_play else None
    legal_ids: Optional[List[int]] = None
    if in_play and current == sess.human_play_index:
        legal_ids = [c.id for c in pis_bridge.legal_actions(sess.p_pos)]
    trick = [
        {"player_id": pid, "card": card_to_dict(pis_bridge._to_o(card))}
        for pid, card in sess.p_pos.trick_cards
    ]
    kontra = None
    if sess.phase == "kontra" and sess.k_next is not None:
        avail = sess.k_next.get("units", [])
        kontra = {
            "pending": {"role": sess.k_next["role"], "play_index": sess.k_next["play_index"]},
            "is_human_turn": True,
            "role": sess.k_next["role"],
            "units": [{"key": U, "label": _UNIT_HU.get(U, U)} for U in avail],
            # primary kept for back-compat display (first available unit)
            "primary": avail[0] if avail else (sess.k_units[0] if sess.k_units else None),
        }
    caps = sess.p_pos.captured
    scores = sess.p_pos.scores
    # Card-point score incl. declared marriages (already in pos.scores). The talon's
    # points count for the DEFENDERS but only the SOLOIST knows the talon during play,
    # so reveal it to the soloist (and everyone at the end), hide it from defenders.
    talon_pts = sum(int(getattr(c, "points", 0)) for c in sess.play_talon)
    reveal_talon = (sess.human_play_index == 0) or terminal
    # Split the marriage DECLARATION bonus (40/20) out of the raw score so the UI can withhold
    # it until each holder "declares" it (with the bemondás bubble): the soloist up front, a
    # defender at their first card. marr[play-index] = that player's declared marriage points.
    marr_pts = [0, 0, 0]
    for (mp, _suit, mpts) in getattr(sess.p_pos, "marriages", []):
        marr_pts[mp] += int(mpts)
    def_talon = talon_pts if reveal_talon else 0
    score = {
        "sol_points": int(scores[0]),                          # full (back-compat)
        "def_points": int(scores[1]) + int(scores[2]) + def_talon,
        "talon_points": talon_pts if reveal_talon else None,   # null = hidden from you
        "sol_card": int(scores[0]) - marr_pts[0],              # card points only (no marriage bonus)
        "def_card": int(scores[1]) + int(scores[2]) - marr_pts[1] - marr_pts[2] + def_talon,
        "marr": marr_pts,                                      # [soloist, def1, def2] bonus
        # Trick counts — the meaningful running tally for the COLORLESS games
        # (betli: soloist must take none; duri: all ten). Card points mean nothing
        # there, so the UI shows these instead (milan 2026-08-01).
        "sol_tricks": len(caps[0]) // 3,
        "def_tricks": (len(caps[1]) + len(caps[2])) // 3,
        # What the running tally MEANS in this contract (milan 2026-08-02):
        #   durchmars (any colour) + betli → the objective is TRICKS; card points are
        #     noise, so the UI shows the trick race instead.
        #   a bid 40-100/20-100 → the declared 100 replaces the párti, so a DEFENDER's
        #     marriage scores nothing at all. Adding it to their displayed total (as we
        #     did) invented points that cannot affect the result.
        "mode": ("tricks" if (getattr(sess.bid, "durchmars", False)
                              or getattr(sess.bid, "betli", False)) else "points"),
        "def_marriage_counts": not (getattr(sess.bid, "forty_hundred", False)
                                    or getattr(sess.bid, "twenty_hundred", False)),
    }
    # The human's own captured cards (won tricks), for the bottom-of-screen pile.
    captured = [card_to_dict(pis_bridge._to_o(c)) for c in caps[sess.human_play_index]]
    return {
        "soloist": 0,
        "human_play_index": sess.human_play_index,
        "contract": sess.bid_name,
        "contract_value": int(sess.rung.value),
        "trump": sess.trump,
        "kontra_level": sess.k_level,
        "kontra": kontra,
        "score": score,
        "captured": captured,
        # The 2 talon cards: count is always known (rendered face-down beside the
        # table); the actual cards are revealed only at game end.
        "talon_count": len(sess.play_talon),
        "talon": ([card_to_dict(c) for c in sess.play_talon] if terminal else None),
        "reveal_soloist": _terit_revealed(sess),               # terített: soloist hand face-up
        "hands": _play_hands_dict(sess, reveal_all=terminal, reveal_sol=_terit_revealed(sess)),
        "hand_sizes": [len(h) for h in pis_bridge.hands_by_player(sess.p_pos)],
        "current_trick": trick,
        "current_player": current,
        "legal_card_ids": legal_ids,
        "history": list(sess.p_history),
        "terminal": terminal,
        "result": sess.result,
    }


def _trump_snapshot(sess: Session) -> dict:
    """You won a plain colored game — declare the trump before play begins."""
    hand = sort_hand(sess.a_hands[sess.seat])
    return {
        "contract": _bid_label(sess.a_current["bid"]),
        "trump_options": ["bells", "leaves", "acorns"],   # tök · zöld · makk (milan's order)
        "own_hand": [card_to_dict(c) for c in hand],
    }


def _snapshot(sess: Session) -> dict:
    base = {"game_id": sess.id, "seat": sess.seat, "seed": sess.seed, "phase": sess.phase}
    # Drain speech-bubble events (send once, then clear).
    base["bubbles"] = list(sess.bubbles)
    sess.bubbles = []
    if sess.phase in ("bid", "passed"):
        base.update(_auction_snapshot(sess))
        if sess.phase == "passed":
            base["result"] = sess.result      # all-pass penalty outcome
    elif sess.phase == "trump_select":
        base.update(_trump_snapshot(sess))
    else:
        base.update(_play_snapshot(sess))
    return base


