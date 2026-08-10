"""exp43 — featurisation: one row per (deal, kontra unit, defender viewer).

Turns the raw corpus from datagen.py into a modelling table. Play is expensive and
already done; features are cheap and get re-cut constantly, so nothing here is allowed
to need the solver.

THE INFORMATION SET IS THE WHOLE POINT (milan 2026-08-02). A defender decides kontra
right after playing their own first card, so:

    def1 (plays 2nd)  →  auction + soloist's LEAD          + own 10 cards
    def2 (plays 3rd)  →  auction + soloist's lead + D1's card + own 10 cards

Every def1 row therefore has the d1-card block MASKED (has_d1 = 0). Leaking it would
manufacture alpha that the deployed rule could never realise. `assert_no_leak()` checks
this, and the model script refuses to train without it.

DELIBERATELY EXCLUDED, though technically visible to def2: whether def1 KONTRA'd. It is
a function of the very policy this experiment is trying to replace, so training on it
would bake today's rule into tomorrow's. Revisit as an ablation once a policy is fixed.

Feature families (prefix → hypothesis):
    t_    trump structure    — the incumbent signal, plus the quality it can't see
    s_    suit shape         — voids / length, the ruffing and hold-up geometry
    p_    points & honours   — parti and 100-game currency; the trump-marriage 40-kill
    c_    colourless shape   — betli / no-trump duri, where 10 sits UNDER the jack
    a_    auction (public)   — contract level, contestedness, and PARTNER strength
    k_    trick 1            — the soloist's lead; for def2 also the partner's card
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from ulti.card import RANK_POINTS, RANKS, SUITS, card_from_id

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAYED = os.path.join(_HERE, "played.jsonl")

UNITS = ("parti", "ulti", "40_100", "20_100", "durchmars", "betli")

# Colourless power order — betli and trumpless durchmars demote the Ten UNDER the jack,
# so a "10" is the 4th-lowest card, not the 2nd-highest (milan; ulti.card.COLORLESS_RANK).
COLORLESS_RANK = {"7": 0, "8": 1, "9": 2, "10": 3, "lower": 4, "upper": 5, "king": 6, "ace": 7}

TOP_TRUMP = frozenset({"ace", "10", "king", "upper"})
LOW_TRUMP = frozenset({"7", "8", "9"})

# Rung name → ladder index, for auction features.
_RUNG_INDEX: Dict[str, int] = {}


def _rung_index(name: str) -> int:
    if not _RUNG_INDEX:
        from ulti.bidding.ladder import LADDER
        for r in LADDER:
            _RUNG_INDEX[r.name] = r.index
    return _RUNG_INDEX.get(name, -1)


# ── own hand ────────────────────────────────────────────────────────────────────

def _hand_feats(cards, trump) -> Dict[str, float]:
    f: Dict[str, float] = {}
    by_suit: Dict[str, list] = {s: [] for s in SUITS}
    for c in cards:
        by_suit[c.suit].append(c)

    # ── t_: trump structure ──
    tr = [c for c in cards if trump is not None and c.suit == trump]
    ranks = sorted((c.rank_index for c in tr), reverse=True)
    f["t_n"] = len(tr)
    f["t_n_top"] = sum(1 for c in tr if c.rank in TOP_TRUMP)
    # The ULTI hypothesis: the soloist must win trick 10 with the trump 7, the LOWEST
    # trump. Any trump a defender still holds at trick 10 beats it — and cheap trumps
    # are the ones that survive, because high ones get drawn or spent winning earlier.
    # So low trumps may be worth more than a raw count, which weighs them equally.
    f["t_n_low"] = sum(1 for c in tr if c.rank in LOW_TRUMP)
    f["t_has_ace"] = float(any(c.rank == "ace" for c in tr))
    f["t_has_ten"] = float(any(c.rank == "10" for c in tr))
    f["t_has_7"] = float(any(c.rank == "7" for c in tr))
    f["t_max"] = float(ranks[0]) if ranks else -1.0
    f["t_min"] = float(ranks[-1]) if ranks else -1.0
    # consecutive run down from the ace — how much of the top of the suit I own outright
    run = 0
    held = {c.rank_index for c in tr}
    for ri in range(len(RANKS) - 1, -1, -1):
        if ri in held:
            run += 1
        else:
            break
    f["t_top_run"] = float(run)

    # ── s_: suit shape ──
    lens = sorted((len(v) for v in by_suit.values()), reverse=True)
    for i in range(4):
        f[f"s_len{i}"] = float(lens[i])
    f["s_voids"] = float(sum(1 for s, v in by_suit.items() if s != trump and not v))
    f["s_singletons"] = float(sum(1 for s, v in by_suit.items() if s != trump and len(v) == 1))
    f["s_nsuits"] = float(sum(1 for v in by_suit.values() if v))

    # ── p_: points & honours ──
    f["p_cardpts"] = float(sum(RANK_POINTS[c.rank] for c in cards))
    f["p_n_ace"] = float(sum(1 for c in cards if c.rank == "ace"))
    f["p_n_ten"] = float(sum(1 for c in cards if c.rank == "10"))
    f["p_n_king"] = float(sum(1 for c in cards if c.rank == "king"))
    f["p_n_upper"] = float(sum(1 for c in cards if c.rank == "upper"))
    f["p_n_marriage"] = float(sum(
        1 for s, v in by_suit.items()
        if any(c.rank == "king" for c in v) and any(c.rank == "upper" for c in v)))
    # The deployed 40-100 rule: a 40-100 declares the TRUMP marriage, so holding either
    # of its two cards proves the soloist cannot have it.
    f["p_trump_marr_card"] = float(
        trump is not None and any(c.suit == trump and c.rank in ("king", "upper") for c in cards))

    # ── c_: colourless shape (betli / trumpless duri) ──
    cl = sorted(COLORLESS_RANK[c.rank] for c in cards)
    f["c_n_low"] = float(sum(1 for r in cl if r <= 2))          # 7/8/9
    f["c_n_lowish"] = float(sum(1 for r in cl if r <= 3))       # + the demoted Ten
    f["c_min"] = float(cl[0]) if cl else -1.0
    f["c_mean"] = float(np.mean(cl)) if cl else -1.0
    # Hold-up geometry: a suit where I hold both a low and a high card can duck and then
    # win — the structure that actually beats a betli. A count of low cards can't see it.
    f["c_span_suits"] = float(sum(
        1 for v in by_suit.values()
        if len(v) >= 2 and min(COLORLESS_RANK[c.rank] for c in v) <= 2
        and max(COLORLESS_RANK[c.rank] for c in v) >= 5))
    f["c_longest_low"] = float(max(
        (sum(1 for c in v if COLORLESS_RANK[c.rank] <= 3) for v in by_suit.values()),
        default=0))
    return f


# ── auction (public knowledge only) ─────────────────────────────────────────────

def _auction_feats(rec, viewer_seat: int, partner_seat: int) -> Dict[str, float]:
    """`bid_seq` is [(seat, rung name, trump)] — what everyone at the table heard. The
    announced EV that run_auction also carries is NOT public and never enters here."""
    seq = rec["bid_seq"]
    w = rec["winner"]
    f: Dict[str, float] = {}
    f["a_rung"] = float(rec["rung_index"])
    f["a_nbids"] = float(len(seq))
    f["a_contested"] = float(len(seq) >= 3)

    def _max_by(seat):
        idx = [_rung_index(b[1]) for b in seq if b[0] == seat]
        return max(idx) if idx else -1

    f["a_my_max"] = float(_max_by(viewer_seat))
    f["a_i_bid"] = float(f["a_my_max"] >= 0)
    # Partner information — free, public, and completely unused by the deployed rule.
    # A partner who fought to a high rung and lost the auction is holding real cards.
    f["a_partner_max"] = float(_max_by(partner_seat))
    f["a_partner_bid"] = float(f["a_partner_max"] >= 0)
    sol_idx = [_rung_index(b[1]) for b in seq if b[0] == w]
    f["a_sol_first"] = float(sol_idx[0]) if sol_idx else -1.0
    f["a_sol_climb"] = float(max(sol_idx) - sol_idx[0]) if sol_idx else 0.0
    f["a_sol_opened"] = float(bool(seq) and seq[0][0] == w)
    f["a_ntrumps_named"] = float(len({b[2] for b in seq if b[2] is not None}))
    return f


# ── trick 1, masked per viewer ──────────────────────────────────────────────────

def _trick1_feats(rec, viewer: int, own_cards, trump) -> Dict[str, float]:
    """viewer 1 = def1 (sees the lead only); viewer 2 = def2 (also sees def1's card)."""
    f: Dict[str, float] = {}
    hist = rec["hist"]
    lead = card_from_id(hist[0][1]) if hist else None
    own_by_suit: Dict[str, int] = {}
    for c in own_cards:
        own_by_suit[c.suit] = own_by_suit.get(c.suit, 0) + 1

    f["k_lead_trump"] = float(lead is not None and trump is not None and lead.suit == trump)
    f["k_lead_rank"] = float(lead.rank_index) if lead else -1.0
    f["k_lead_pts"] = float(RANK_POINTS[lead.rank]) if lead else -1.0
    f["k_lead_clrank"] = float(COLORLESS_RANK[lead.rank]) if lead else -1.0
    f["k_lead_my_len"] = float(own_by_suit.get(lead.suit, 0)) if lead else -1.0
    f["k_lead_i_void"] = float(lead is not None and own_by_suit.get(lead.suit, 0) == 0)

    # def1's card — visible to def2 ONLY. Masked (zeros, has_d1=0) for def1 rows.
    d1c = card_from_id(hist[1][1]) if (viewer == 2 and len(hist) > 1) else None
    f["k_has_d1"] = float(d1c is not None)
    f["k_d1_trump"] = float(d1c is not None and trump is not None and d1c.suit == trump)
    f["k_d1_rank"] = float(d1c.rank_index) if d1c else 0.0
    f["k_d1_pts"] = float(RANK_POINTS[d1c.rank]) if d1c else 0.0
    f["k_d1_clrank"] = float(COLORLESS_RANK[d1c.rank]) if d1c else 0.0
    f["k_d1_followed"] = float(d1c is not None and lead is not None and d1c.suit == lead.suit)
    # Is the partner trying to WIN this trick, or ducking under it? A partner who
    # overtakes the soloist's lead is telling you something about their hand.
    if d1c is not None and lead is not None:
        if trump is not None and d1c.suit == trump and lead.suit != trump:
            beats = True
        elif d1c.suit == lead.suit:
            beats = d1c.rank_index > lead.rank_index
        else:
            beats = False
        f["k_d1_beats_lead"] = float(beats)
    else:
        f["k_d1_beats_lead"] = 0.0
    return f


# ── row assembly ────────────────────────────────────────────────────────────────

def _row(rec, unit: str, viewer: int) -> Tuple[Dict[str, float], int, List[int]]:
    trump = rec["trump"]
    hands = {1: rec["d1"], 2: rec["d2"]}
    own = [card_from_id(i) for i in hands[viewer]]
    w = rec["winner"]
    viewer_seat = (w + viewer) % 3
    partner_seat = (w + (3 - viewer)) % 3      # the other defender's ORIGINAL seat
    f = _hand_feats(own, trump)
    f.update(_auction_feats(rec, viewer_seat, partner_seat))
    f.update(_trick1_feats(rec, viewer, own, trump))
    f["v_is_def2"] = float(viewer == 2)
    # Public: the contract announces whether there is a trump at all. Needed because a
    # colourless game and a colored game where I simply hold no trumps both give t_n=0.
    f["v_colorless"] = float(trump is None)
    u = rec["units"][unit]
    return f, int(bool(u["made"])), list(u["iso"])


def build_table(path: str = PLAYED, units=UNITS) -> Dict[str, dict]:
    """{unit: {X, y, iso, viewer, seed, names}} — one table per kontra unit."""
    acc: Dict[str, dict] = {u: {"X": [], "y": [], "iso": [], "viewer": [], "seed": [],
                                "contract": []}
                            for u in units}
    names: List[str] = []
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("kept"):
                continue
            for unit in rec["units"]:
                if unit not in acc:
                    continue
                for viewer in (1, 2):
                    f, y, iso = _row(rec, unit, viewer)
                    if not names:
                        names = sorted(f)
                    acc[unit]["X"].append([f[k] for k in names])
                    acc[unit]["y"].append(y)
                    acc[unit]["iso"].append(iso)
                    acc[unit]["viewer"].append(viewer)
                    acc[unit]["seed"].append(rec["seed"])
                    acc[unit]["contract"].append(rec["contract"])
    out = {}
    for u, d in acc.items():
        if not d["X"]:
            continue
        out[u] = {
            "X": np.asarray(d["X"], dtype=np.float32),
            "y": np.asarray(d["y"], dtype=np.int8),
            "iso": np.asarray(d["iso"], dtype=np.float32),
            "viewer": np.asarray(d["viewer"], dtype=np.int8),
            "seed": np.asarray(d["seed"], dtype=np.int64),
            "contract": np.asarray(d["contract"], dtype=object),
            "names": names,
        }
    return out


def assert_no_leak(tables: Dict[str, dict]) -> None:
    """def1 rows must carry ZERO information about def1's played card. If this ever
    trips, every downstream number is fiction — the model would be reading a card the
    real defender had not seen when they decided."""
    for u, d in tables.items():
        names = d["names"]
        m = d["viewer"] == 1
        if not m.any():
            continue
        for i, n in enumerate(names):
            if n.startswith("k_d1_") or n == "k_has_d1":
                col = d["X"][m, i]
                if np.any(col != 0.0):
                    raise AssertionError(
                        f"LEAK: unit {u} has non-zero {n} on {int((col != 0).sum())} def1 rows")


def main():
    tables = build_table()
    assert_no_leak(tables)
    print(f"{'unit':10s} {'rows':>7s} {'deals':>7s} {'made%':>7s}  (def1/def2 split)")
    for u in UNITS:
        if u not in tables:
            continue
        d = tables[u]
        n = len(d["y"])
        print(f"{u:10s} {n:7d} {len(set(d['seed'].tolist())):7d} "
              f"{100*d['y'].mean():6.1f}%  {int((d['viewer']==1).sum())}/"
              f"{int((d['viewer']==2).sum())}")
    print(f"\n{len(tables[UNITS[0]]['names']) if tables else 0} features, no def1 leak.")


if __name__ == "__main__":
    main()
