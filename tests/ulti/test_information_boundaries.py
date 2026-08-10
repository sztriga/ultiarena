"""Every AI decision must depend ONLY on what that decider is allowed to see.

Why this file exists (2026-08-02). We already had an anti-cheat audit —
`research/experiments/25_strength_ceiling/audit_cheating.py` — and it passed while the
bidder was cheating. Its test was:

    sol12, _d1, _d2 = deal_12_10_10(seed)
    cards = list(sol12)                                    # hand + talon as ONE blob
    decisions = {key(bid_fn(cards, None, others))          # vary only the OTHER 20 cards
                 for others in opp_variations(cards, 5, seed)}

It held the twelve cards fixed and varied the opponents' hands, so it proved the decision
does not leak across the boundary it ASSUMED. But the talon sat on the wrong side of that
boundary by assumption, so varying it was never tested — and the bid/pass decision was
reading two cards the seat had not picked up.

The lesson, and the rule for this file: a leak test must vary **everything the decider is
not entitled to see**, and the entitlement must be stated per decision point rather than
inherited from whatever the function signature happens to accept. Each test below names
its information set explicitly, perturbs the complement, and asserts invariance. Positive
controls (deliberate cheaters) prove each test is actually sensitive — a green suite of
insensitive tests is exactly how we got here.

Information sets, per milan's rules:
    pickup (bid vs pass)   own 10 cards + public auction history
                           NOT the talon: picking it up commits you to announcing a game
    announce (post-pickup) the real 12 — you are holding them
    card play              own hand + played cards + public voids (+ a terített reveal)
    defender kontra        own hand + public
    soloist rekontra       own hand + played + the talon it buried itself
"""
from __future__ import annotations

import random

import pytest

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

from ulti.bidding.auction import net_bid_fn                     # noqa: E402
from ulti.bidding.deal import deal_12_10_10                     # noqa: E402
from ulti.bidding.provider import NetProvider                   # noqa: E402
from ulti.card import DECK                                      # noqa: E402
from ulti.solvers import determinize as det, pis                # noqa: E402

N_DEALS = 25
N_VARIATIONS = 5


@pytest.fixture(scope="module")
def bid_fn():
    return net_bid_fn(NetProvider())


def _decision_key(bundle):
    """What the seat actually did, ignoring float noise in the announced EV."""
    if bundle is None:
        return "PASS"
    _ev, rung, trump, discard, _hand10 = bundle
    return (rung.name, trump, tuple(sorted(c.id for c in discard)))


def _unseen(*held):
    ids = {c.id for group in held for c in group}
    return [c for c in DECK if c.id not in ids]


# ── 1. the pickup decision may not depend on the talon ──────────────────────────

def test_pickup_decision_ignores_the_talon(bid_fn):
    """THE REGRESSION TEST. Fix a seat's 10 cards, vary the talon: whether it bids at
    all must not move. Picking the talon up commits you, so a seat that passes has
    never seen those two cards.

    Only the PASS/BID decision is asserted — once a seat commits, it legitimately holds
    twelve cards, so which game it announces and what it buries SHOULD depend on them.
    """
    rng = random.Random(20260802)
    for s in range(N_DEALS):
        sol12, _d1, _d2 = deal_12_10_10(9_000 + s)
        hand10 = list(sol12[:10])
        pool = _unseen(hand10)
        bids = set()
        for _ in range(N_VARIATIONS):
            rng.shuffle(pool)
            bids.add(bid_fn(hand10, pool[:2], None, -2.0, None) is not None)
        assert len(bids) == 1, (
            f"deal {s}: the bid/pass decision changed with the talon — the seat is "
            f"reading cards it has not picked up")


def test_pickup_test_is_sensitive(bid_fn):
    """Positive control. A bidder that peeks — decides from all twelve, the exact bug we
    shipped — MUST fail the test above, otherwise the test proves nothing."""
    rng = random.Random(20260802)
    flips = 0
    for s in range(N_DEALS):
        sol12, _d1, _d2 = deal_12_10_10(9_000 + s)
        hand10 = list(sol12[:10])
        pool = _unseen(hand10)
        bids = set()
        for _ in range(N_VARIATIONS):
            rng.shuffle(pool)
            talon = pool[:2]
            # the cheat: evaluate the 12 and let THAT decide whether to enter
            peek = bid_fn(hand10 + talon, [], None, -2.0, None)
            bids.add(peek is not None)
        flips += len(bids) > 1
    assert flips > 0, "the talon-invariance test cannot detect a peeking bidder"


# ── 2. no auction decision may depend on the opponents' hands ───────────────────

def test_bidding_ignores_opponent_hands(bid_fn):
    """The boundary the old audit did check — kept, because it is still a real one."""
    rng = random.Random(4242)
    for s in range(N_DEALS):
        sol12, _d1, _d2 = deal_12_10_10(3_000 + s)
        hand10, talon = list(sol12[:10]), list(sol12[10:])
        pool = _unseen(hand10, talon)
        keys = set()
        for _ in range(N_VARIATIONS):
            rng.shuffle(pool)
            others = (pool[:10], pool[10:20])
            keys.add(_decision_key(bid_fn(hand10, talon, None, -2.0, others)))
        assert len(keys) == 1, f"deal {s}: the bid moved with the opponents' hidden hands"


# ── 3. a defender's play search may not know where the talon went ───────────────

def _info_set_for(viewer, hands, talon, trump="hearts", contract="multi"):
    pos = pis.build_position(hands=[list(h) for h in hands], soloist=0, leader=0,
                             contract="parti", trump=trump, talon=list(talon),
                             declare_marriages=True)
    return pos, det.build_info_set(pos=pos, viewer=viewer, contract=contract)


def test_defender_determinization_hides_the_talon():
    """A defender's sampled worlds must scatter the real talon cards, not pin them.
    Soloist is the control: it buried the talon itself, so it legitimately knows it."""
    sol12, d1, d2 = deal_12_10_10(555)
    hands = [list(sol12[:10]), list(d1), list(d2)]
    talon = list(sol12[10:])
    true_ids = {c.id for c in talon}

    _pos, iset_sol = _info_set_for(0, hands, talon)
    assert iset_sol.talon_known is not None, "the soloist buried the talon; it must know it"

    _pos, iset_def = _info_set_for(2, hands, talon)
    assert iset_def.talon_known is None, "a defender must not be handed the talon"
    assert iset_def.talon_size == len(talon)

    rng = random.Random(1)
    exact = 0
    for _ in range(200):
        _h, sampled = det.sample_world(iset_def, rng)
        exact += {c.id for c in sampled} == true_ids
    # 2 of the 22 unseen cards land in the talon slot by chance ~<2% of the time.
    assert exact < 40, (
        f"defender worlds reproduced the true talon {exact}/200 times — it is not hidden")


def test_defender_determinization_never_reveals_own_secrets():
    """Sampled worlds must always return the viewer's OWN hand untouched, and must never
    place a card the viewer holds into someone else's hand."""
    sol12, d1, d2 = deal_12_10_10(777)
    hands = [list(sol12[:10]), list(d1), list(d2)]
    talon = list(sol12[10:])
    rng = random.Random(2)
    for viewer in (1, 2):
        _pos, iset = _info_set_for(viewer, hands, talon)
        own_ids = {c.id for c in hands[viewer]}
        for _ in range(50):
            sampled, tal = det.sample_world(iset, rng)
            assert {c.id for c in sampled[viewer]} == own_ids
            for other in range(3):
                if other == viewer:
                    continue
                assert not (own_ids & {c.id for c in sampled[other]})
            assert not (own_ids & {c.id for c in tal})


# ── 4. the betli-defense net sees own hand + played cards, nothing else ─────────

def test_betli_defense_encoder_is_blind_to_hidden_hands():
    """The exp36 net is handed the raw position, which contains all three hands. Its
    encoding must not change when the OTHER hands are permuted."""
    defense = pytest.importorskip("ulti.betli.defense")
    sol12, d1, d2 = deal_12_10_10(888)
    hands = [list(sol12[:10]), list(d1), list(d2)]
    talon = list(sol12[10:])

    def encode_with(other_hands):
        h = [hands[0], other_hands[0], other_hands[1]]
        pos = pis.build_position(hands=[list(x) for x in h], soloist=0, leader=0,
                                 contract="betli", trump=None, talon=list(talon))
        return defense.encode(pos, 1).tobytes()

    base = encode_with([hands[1], hands[2]])
    rng = random.Random(3)
    pool = [c for c in hands[2]]
    for _ in range(5):
        rng.shuffle(pool)
        # viewer 1's own hand is fixed; only the unseen seat is permuted
        assert encode_with([hands[1], list(pool)]) == base, (
            "the betli-defense encoding moved with a hand the defender cannot see")
