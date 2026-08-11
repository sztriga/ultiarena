"""THE suit re-encoding (milan 2026-08-11): card ids move to display order.

Old wire order: acorns, leaves, hearts, bells   (alphabetical English, historical)
New wire order: hearts, bells, leaves, acorns   (piros, tök, zöld, makk — and the
                order ultisolver's own Suit enum ALWAYS had)

After this, wire == display == solver: one order everywhere, forever. The cost,
accepted deliberately: every seed now deals a different hand, so pre-migration
experiment seed references are history (their FINDINGS stand; docs unchanged).

The model weights are not retrained — they are PERMUTED. Every featurizer's
input permutation is derived EMPIRICALLY: we capture feature matrices for fixed
physical states (cards by name) under the old encoding, recompute under the new,
and match columns. No hand-mapped offsets → no offset mistakes; a non-permutation
change fails loudly. The betli-defense net's 32-way output is permuted by the
card-id bijection.

Usage (in order):
  1. OLD encoding checked out:  python migrations/suit_reencode_2026_08_11.py probe_old <dir>
  2. Edit ulti/card.py + apps/web/src/ui/cards.ts to the new SUITS order.
  3. NEW encoding:              python migrations/suit_reencode_2026_08_11.py apply <dir>
     — derives permutations, rewrites models/ulti/**/*.pt in place, migrates
       data/games.db transcripts.
  4. Verify: python tests/golden/physical_ref.py verify <dir>/physref_old.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OLD_SUITS = ['acorns', 'leaves', 'hearts', 'bells']
N_PROBES = 800
RANKS = ['7', '8', '9', 'lower', 'upper', 'king', '10', 'ace']
_ALL_NAMES = [f"{s}:{r}" for s in OLD_SUITS for r in RANKS]


def _card(name):
    from ulti.card import Card
    s, r = name.split(":")
    return Card(suit=s, rank=r)


def _rng():
    return random.Random(20260811)


# ── probe-state generation (names only — encoding-independent) ──────────────────

def _pickup_states(rng):
    out = []
    for _ in range(N_PROBES):
        hand = rng.sample(_ALL_NAMES, 10)
        out.append({"hand": hand, "trump": rng.choice(OLD_SUITS)})
    return out


def _real_states(rng):
    out = []
    for _ in range(N_PROBES):
        deck = _ALL_NAMES[:]
        rng.shuffle(deck)
        n_played = 3 * rng.randint(0, 6)
        played, rest = deck[:n_played], deck[n_played:]
        tlen = rng.randint(0, 2)
        leader = rng.randint(0, 2)
        trick = [((leader + i) % 3, rest[i]) for i in range(tlen)]
        rest = rest[tlen:]
        k = (30 - n_played - tlen) // 3
        hands = [rest[0:k], rest[k:2 * k], rest[2 * k:3 * k]]
        out.append({
            "hands": hands, "played": played, "trick": trick, "leader": leader,
            "trick_no": n_played // 3, "viewer": rng.choice([1, 2]),
            "voids": {str(p): sorted(rng.sample(OLD_SUITS, rng.randint(0, 2)))
                      for p in range(3)},
        })
    return out


def _defense_states(rng):
    out = []
    for _ in range(N_PROBES // 4):                 # needs real positions — dearer
        deck = _ALL_NAMES[:]
        rng.shuffle(deck)
        out.append({"hands": [deck[0:10], deck[10:20], deck[20:30]],
                    "talon": deck[30:32], "plies": rng.randint(1, 7)})
    return out


# ── featurize under the LIVE encoding ───────────────────────────────────────────

def _pickup_X(states):
    from ulti.vnet.pickup.features import featurize
    Xc = np.stack([featurize([_card(n) for n in s["hand"]], s["trump"], True)
                   for s in states])
    Xn = np.stack([featurize([_card(n) for n in s["hand"]], None, False)
                   for s in states])
    return Xc, Xn


def _real_X(states):
    from ulti.vnet.betli.features import extract_features
    rows = []
    for s in states:
        rows.append(extract_features(
            hands=[[_card(n) for n in h] for h in s["hands"]],
            played_cards=[_card(n) for n in s["played"]],
            current_trick=[(p, _card(n)) for p, n in s["trick"]],
            leader=s["leader"], trick_no=s["trick_no"],
            voids={int(p): set(v) for p, v in s["voids"].items()},
            soloist=0, viewer=s["viewer"]))
    return np.stack(rows)


def _defense_X(states):
    from ulti.betli.defense import encode
    from ulti.solvers import pis as pis_bridge
    rows = []
    for s in states:
        pos = pis_bridge.build_position(
            hands=[[_card(n) for n in h] for h in s["hands"]],
            soloist=0, leader=0, contract="betli", trump=None,
            talon=[_card(n) for n in s["talon"]], declare_marriages=False)
        for _ in range(s["plies"]):
            legal = sorted(pis_bridge.legal_actions(pos), key=lambda c: (c.suit, c.rank))
            if pis_bridge.is_terminal(pos):
                break
            pis_bridge.apply_move(pos, legal[0])
        viewer = pis_bridge.current_player(pos)
        if viewer == 0:
            viewer = 1
        rows.append(encode(pos, viewer))
    return np.stack(rows)


# ── permutation derivation ──────────────────────────────────────────────────────

def _match_columns(X_old: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    """perm[old_dim] = new_dim such that X_new[:, perm[i]] == X_old[:, i].
    Columns must match uniquely — duplicates across N_PROBES random states would
    mean the featurizer has truly identical dims (never observed; asserted)."""
    assert X_old.shape == X_new.shape
    d = X_old.shape[1]
    # hashable column keys
    key_new = {}
    for j in range(d):
        key_new.setdefault(X_new[:, j].tobytes(), []).append(j)
    perm = np.full(d, -1, dtype=np.int64)
    used = set()
    for i in range(d):
        cands = [j for j in key_new.get(X_old[:, i].tobytes(), []) if j not in used]
        assert cands, f"old dim {i}: no matching new column — not a pure permutation!"
        perm[i] = cands[0]
        used.add(cands[0])
    assert len(used) == d
    return perm


def _id_bijection():
    """old card id -> new card id, by name."""
    from ulti.card import SUITS as NEW_SUITS
    old_of = {f"{s}:{r}": OLD_SUITS.index(s) * 8 + RANKS.index(r)
              for s in OLD_SUITS for r in RANKS}
    new_of = {f"{s}:{r}": NEW_SUITS.index(s) * 8 + RANKS.index(r)
              for s in NEW_SUITS for r in RANKS}
    m = np.zeros(32, dtype=np.int64)
    for name, oid in old_of.items():
        m[oid] = new_of[name]
    return m


# ── weight surgery ──────────────────────────────────────────────────────────────

def _permute_first_linear(sd: dict, perm: np.ndarray) -> None:
    """Reorder input columns of the FIRST linear layer: new_W[:, perm[i]] = old_W[:, i]."""
    import torch
    first = min((k for k in sd if k.endswith(".weight") and sd[k].dim() == 2),
                key=lambda k: (len(k), k))
    W = sd[first]
    assert W.shape[1] == len(perm), f"{first}: in={W.shape[1]} vs perm {len(perm)}"
    newW = torch.empty_like(W)
    newW[:, torch.as_tensor(perm)] = W
    sd[first] = newW


def _permute_last_rows(sd: dict, idmap: np.ndarray) -> None:
    """Reorder output rows (a 32-way per-card policy head) by the id bijection."""
    import torch
    last = [k for k in sd if k.endswith(".weight") and sd[k].dim() == 2][-1]
    lastb = last[:-len("weight")] + "bias"
    assert sd[last].shape[0] == 32
    t = torch.as_tensor(idmap)
    for key in (last, lastb):
        if key in sd:
            new = torch.empty_like(sd[key])
            new[t] = sd[key]
            sd[key] = new


def _load_sd(path: Path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        return ck, ck["model"]
    if isinstance(ck, dict) and all(hasattr(v, "shape") for v in ck.values()):
        return None, ck
    for k in ("state", "state_dict", "net"):
        if isinstance(ck, dict) and k in ck:
            return ck, ck[k]
    raise SystemExit(f"unknown checkpoint format: {path}")


# ── modes ───────────────────────────────────────────────────────────────────────

def probe_old(outdir: str) -> None:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rng = _rng()
    states = {"pickup": _pickup_states(rng), "real": _real_states(rng),
              "defense": _defense_states(rng)}
    json.dump(states, open(out / "states.json", "w"))
    Xc, Xn = _pickup_X(states["pickup"])
    np.savez(out / "X_old.npz", pickup_c=Xc, pickup_n=Xn,
             real=_real_X(states["real"]), defense=_defense_X(states["defense"]))
    print(f"probed old encoding -> {out}")


def apply(outdir: str) -> None:
    out = Path(outdir)
    states = json.load(open(out / "states.json"))
    old = np.load(out / "X_old.npz")
    Xc, Xn = _pickup_X(states["pickup"])
    perms = {
        "pickup_c": _match_columns(old["pickup_c"], Xc),
        "pickup_n": _match_columns(old["pickup_n"], Xn),
        "real": _match_columns(old["real"], _real_X(states["real"])),
        "defense": _match_columns(old["defense"], _defense_X(states["defense"])),
    }
    idmap = _id_bijection()
    print({k: "identity" if (v == np.arange(len(v))).all() else "permuted"
           for k, v in perms.items()})

    import torch
    jobs = [
        ("models/ulti/bidding/parti_baseline.pt", "pickup_c", False),
        ("models/ulti/bidding/ulti_baseline.pt", "pickup_c", False),
        ("models/ulti/bidding/reach100_40_baseline.pt", "pickup_c", False),
        ("models/ulti/bidding/reach100_20_baseline.pt", "pickup_c", False),
        ("models/ulti/bidding/duri_colored_baseline.pt", "pickup_c", False),
        ("models/ulti/bidding/betli_baseline.pt", "pickup_n", False),
        ("models/ulti/bidding/colorless_duri_baseline.pt", "pickup_n", False),
        ("models/ulti/betli/betli_real_baseline.pt", "pickup_n", False),  # 32-dim pickup-style head
        ("models/ulti/betli/betli_defense.pt", "defense", True),
    ]
    for rel, pkey, out_rows in jobs:
        path = _REPO / rel
        ck, sd = _load_sd(path)
        _permute_first_linear(sd, perms[pkey])
        if out_rows:
            _permute_last_rows(sd, idmap)
        torch.save(ck if ck is not None else sd, path)
        print(f"permuted {rel}")

    _migrate_games_db(idmap)
    print("apply done — now run physical_ref verify")


def _migrate_games_db(idmap: np.ndarray) -> None:
    import sqlite3
    db = _REPO / "data" / "games.db"
    if not db.exists():
        print("no games.db — skipped")
        return
    con = sqlite3.connect(db)
    rows = con.execute("SELECT id, transcript FROM games").fetchall()
    n = 0
    for gid, tj in rows:
        t = json.loads(tj)
        t["deal"]["hands"] = [[int(idmap[c]) for c in h] for h in t["deal"]["hands"]]
        t["deal"]["talon"] = [int(idmap[c]) for c in t["deal"]["talon"]]
        t["plays"] = [[p, int(idmap[c]), ti] for p, c, ti in t["plays"]]
        con.execute("UPDATE games SET transcript = ? WHERE id = ?",
                    (json.dumps(t, separators=(",", ":")), gid))
        n += 1
    con.commit(); con.close()
    print(f"migrated {n} game transcripts")


if __name__ == "__main__":
    {"probe_old": probe_old, "apply": apply}[sys.argv[1]](sys.argv[2])
