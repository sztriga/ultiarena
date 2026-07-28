"""Audit: is durchmars/betli win-prob determined by hole structure?

Uses the biased god-labeled data already on disk (exp 15, ~35%/48% pos)
so it spans the win/loss boundary — no new labeling.

PART A: tabulate god-win by a structural 'loser estimate' (the user's
        "by hole count" idea); fit a tiny logistic on hand-structure
        features and report how well structure ALONE predicts god-win.
PART B: archetype Monte-Carlo (the user's exact idea) — construct a hand
        with a chosen hole, sample defender splits, god-label, report
        the fail rate.

Usage: python audit_colorless.py
"""
from __future__ import annotations

import random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.card import SUITS, RANKS, Card, DECK
from solvers import pis
from eval.pimc_matchup import god_says_soloist_wins
from vnet.pickup.v18 import Exp18Pickup

EXP15 = Path(__file__).parent.parent / "15_vnet_pickup"
EXP18 = Path(__file__).parent
# Colorless (betli/duri) trick strength, strongest→weakest *rank_index*.
# The 10 is demoted under the J — see solvers/pis.py is_colorless_duri /
# trickster BETLI_STRENGTH: A > K > Q(upper) > J(lower) > 10 > 9 > 8 > 7.
STR_ORDER = [7, 5, 4, 3, 6, 2, 1, 0]


# ── hand-structure features from a 4×8 presence matrix ───────────────
def suit_losers(row):
    """Cover-walk loser estimate for one suit, in colorless strength order.

    Walk strongest→weakest; your high cards 'cover' outstanding cards
    above the rest of your holding. Uncovered outstanding cards within
    your range are losers. 0 losers ⇒ the suit runs clean from the top.
    """
    held_pos = [k for k, ri in enumerate(STR_ORDER) if row[ri]]
    if not held_pos:
        return 0
    last = max(held_pos)
    cover = 0
    losers = 0
    for k in range(last + 1):
        if row[STR_ORDER[k]]:
            cover += 1
        elif cover > 0:
            cover -= 1
        else:
            losers += 1
    return losers


def _top_run(row):
    n = 0
    for ri in STR_ORDER:
        if row[ri]:
            n += 1
        else:
            break
    return n


def features(M):
    """M: (4,8) presence. Returns structural feature dict."""
    lengths = M.sum(axis=1).astype(int)
    losers = sum(suit_losers(M[s]) for s in range(4))
    top_run = sum(_top_run(M[s]) for s in range(4))
    return {
        'losers': int(losers),
        'top_run': int(top_run),
        'n_void': int((lengths == 0).sum()),
        'n_single': int((lengths == 1).sum()),
        'longest': int(lengths.max()),
        'n_ace': int(M[:, 7].sum()),
        'lengths_sorted': tuple(sorted(lengths, reverse=True)),
    }


def feat_vector(M):
    f = features(M)
    return [f['losers'], f['top_run'], f['n_void'], f['n_single'],
            f['longest'], f['n_ace']]


def hand_str(M):
    out = []
    for s in range(4):
        rs = [RANKS[r] for r in STR_ORDER if M[s, r]]
        if rs:
            out.append(f"{SUITS[s][0]}:{'·'.join(r[:2] for r in rs)}")
    return "  ".join(out)


# ── PART A ────────────────────────────────────────────────────────────
def part_a(contract):
    d = np.load(EXP15 / f"{contract}_god_250k.npz")
    X, y = d['X'], d['y']
    M = X.reshape(-1, 4, 8)
    losers = np.array([sum(suit_losers(m[s]) for s in range(4)) for m in M])

    print(f"\n===== PART A — {contract} (N={len(y)}, pos={y.mean():.3f}) =====")
    print("god-win rate by structural loser-estimate:")
    print(f"  {'losers':>6}  {'n':>7}  {'god-win%':>9}")
    for L in range(0, losers.max() + 1):
        m = losers == L
        if m.sum() < 50:
            continue
        print(f"  {L:>6}  {m.sum():>7}  {100*y[m].mean():>8.1f}")

    # structure-only logistic
    from sklearn.linear_model import LogisticRegression
    Xf = np.array([feat_vector(m) for m in M], dtype=np.float32)
    n = len(y); ntr = int(n * 0.8)
    idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[:ntr], idx[ntr:]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xf[tr], y[tr])
    p = clf.predict_proba(Xf[va])[:, 1]
    brier_struct = float(((p - y[va]) ** 2).mean())

    # multi-head net (v18a) on the same val hands, for contrast
    net = Exp18Pickup.load(EXP18 / "multihead_v18a.pt")
    pn = net.predict(X[va], contract)
    brier_net = float(((pn - y[va]) ** 2).mean())
    base = float(((y[va].mean() - y[va]) ** 2).mean())

    fnames = ['losers', 'top_run', 'n_void', 'n_single', 'longest', 'n_ace']
    print(f"\n  Brier (lower=better)  base-rate={base:.4f}  "
          f"structure-logistic={brier_struct:.4f}  v18a-net={brier_net:.4f}")
    print(f"  structure features used: {fnames}")
    print("  logistic coef: " +
          "  ".join(f"{nm}={c:+.2f}" for nm, c in zip(fnames, clf.coef_[0])))

    # structure-logistic calibration table
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    print("  structure-logistic calibration:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = (p >= lo) & (p < hi)
        if mm.sum() < 30:
            continue
        print(f"    [{lo:.2f},{hi:.2f})  n={mm.sum():>6}  "
              f"pred={p[mm].mean():.3f}  actual={y[va][mm].mean():.3f}")


# ── PART B — archetype Monte-Carlo (the user's exact idea) ───────────
def _card(suit, rank):
    return Card(suit, rank)


def mc_archetype(name, sol10, n_samples=400, seed=0):
    """god-win rate of a fixed 10-card soloist hand over random defender
    splits (durchmars). talon = 2 random of the remaining 22."""
    rng = random.Random(seed)
    rest = [c for c in DECK if c not in sol10]
    wins = 0
    for i in range(n_samples):
        pool = list(rest)
        rng.shuffle(pool)
        talon = pool[:2]
        d1, d2 = pool[2:12], pool[12:22]
        pos = pis.build_position(
            hands=[list(sol10), d1, d2], soloist=0, leader=0,
            contract='durchmars', trump=None, talon=talon,
        )
        if god_says_soloist_wins(pos, contract='durchmars'):
            wins += 1
    return wins / n_samples


def part_b():
    print("\n===== PART B — durchmars archetype Monte-Carlo =====")
    # Build 10-card hands with controlled hole structure.
    # Top 3 in colorless are A, K, Q (Ten is weak here).
    A = lambda s: _card(s, 'ace'); K = lambda s: _card(s, 'king')
    Q = lambda s: _card(s, 'upper'); J = lambda s: _card(s, 'lower')
    archetypes = {
        "solid A-K-Q x3 +A": [A('acorns'),K('acorns'),Q('acorns'),
                               A('leaves'),K('leaves'),Q('leaves'),
                               A('hearts'),K('hearts'),Q('hearts'),
                               A('bells')],
        "one hole (A-K-J, Q out)": [A('acorns'),K('acorns'),J('acorns'),
                               A('leaves'),K('leaves'),Q('leaves'),
                               A('hearts'),K('hearts'),Q('hearts'),
                               A('bells')],
        "two holes (A-K + A-K)": [A('acorns'),K('acorns'),
                               A('leaves'),K('leaves'),
                               A('hearts'),K('hearts'),Q('hearts'),
                               A('bells'),K('bells'),Q('bells')],
    }
    for name, hand in archetypes.items():
        assert len(hand) == 10, (name, len(hand))
        M = np.zeros((4, 8))
        for c in hand:
            M[c.suit_index, c.rank_index] = 1
        wr = mc_archetype(name, hand, n_samples=400, seed=1)
        print(f"  {name:>28}  losers={features(M)['losers']}  "
              f"god-win={wr*100:.1f}%   [{hand_str(M)}]")


if __name__ == "__main__":
    part_a('durchmars')
    part_a('betli')
    part_b()
