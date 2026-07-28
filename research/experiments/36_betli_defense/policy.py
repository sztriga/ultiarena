"""exp36 Phase 2 — a LEARNED betli-DEFENSE policy, to beat PIMC's 60% steal rate.

The hope: PIMC defense suffers strategy fusion (Frank & Basin) + small-sample noise; a single
learned policy trained on god-labelled SAFE moves ("which cards keep the soloist losing") might
defend more robustly. Cheat-clean: the net sees only the defender's own hand + public cards.

Pipeline (subcommands):
  datagen : dd-lost betli deals → play out (god soloist, exploring defenders); at each defender
            node with a real choice, record (features, safe_mask) where safe = the soloist STILL
            double-dummy-loses after that move. → data.npz
  train   : supervised — predict the safe moves (BCE over legal). → model.pt
  eval    : net-defense vs PIMC-defense on held-out dd-lost betlis, steal rate. → EVAL.md
Env: N (deals), WORKERS, EPOCHS, HIDDEN, LR, ARCH.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/31_exploit_play", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ulti.solvers import pis, determinize as _det                         # noqa: E402
from ulti.eval.pimc_matchup import defenders_won, god_says_soloist_wins, pimc_pick  # noqa: E402
from ulti.eval.dojo import deal_betli                                     # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "8"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
DATA = os.path.join(_HERE, "data.npz")
MODEL = os.path.join(_HERE, "model.pt")
FEAT_DIM = 139


# ── feature encoder (cheat-clean defender view) ──────────────────────────────
def _ids(cards):
    return [c.id for c in cards]


def encode(pos, viewer):
    """139-dim cheat-clean features for `viewer`'s decision at `pos`."""
    hands = pis.hands_by_player(pos)
    own = hands[viewer]
    trick = [pis._to_o(c) for _p, c in pos.trick_cards]
    played = []
    for cap in pos.captured:
        played += [pis._to_o(c) for c in cap]
    played += trick
    x = np.zeros(FEAT_DIM, dtype=np.float32)
    for c in own:
        x[c.id] = 1.0                                  # 0..31  own hand
    seen = set()
    for c in played:
        x[32 + c.id] = 1.0                             # 32..63 all public/played cards
        seen.add(c.id)
    own_ids = {c.id for c in own}
    for i in range(32):
        if i not in own_ids and i not in seen:
            x[64 + i] = 1.0                            # 64..95 unknown (opp hands + talon)
    for c in trick:
        x[96 + c.id] = 1.0                             # 96..127 current-trick cards
    # 128..132 lead suit one-hot (none=132); 133..135 position; 136..137 viewer; 138 hand frac
    suit_ix = {"acorns": 0, "leaves": 1, "hearts": 2, "bells": 3}
    if trick:
        x[128 + suit_ix[trick[0].suit]] = 1.0
    else:
        x[132] = 1.0
    x[133 + min(2, len(trick))] = 1.0
    x[136 + (viewer - 1)] = 1.0
    x[138] = len(own) / 10.0
    return x


# ── datagen ──────────────────────────────────────────────────────────────────
def _safe_mask(pos):
    """32-dim mask of moves that keep the soloist LOSING (defender viewer to move), + legal mask.
    Returns (safe, legal) or None if the position is already lost (no safe move)."""
    legal = pis.legal_actions(pos)
    safe = np.zeros(32, dtype=np.float32)
    lmask = np.zeros(32, dtype=np.float32)
    any_safe = False
    for c in legal:
        lmask[c.id] = 1.0
        child = pos.clone(); pis.apply_move(child, c)
        if not god_says_soloist_wins(child, contract="betli"):     # soloist still loses → safe
            safe[c.id] = 1.0; any_safe = True
    return (safe, lmask) if any_safe else None


def _datagen_worker(seed):
    deal = deal_betli(seed=seed, alpha=1.0)
    pos0 = pis.build_position(hands=[list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)],
                              soloist=0, leader=0, contract="betli", trump=None, talon=list(deal.talon))
    if god_says_soloist_wins(pos0, contract="betli"):              # keep only defender-holdable
        return []
    rng = random.Random(seed * 7 + 1)
    rows = []
    pos = pos0.clone(); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        legal = pis.legal_actions(pos)
        if p == 0:                                                  # soloist plays god (strong attacker)
            mv = pis.solve_best(pos, contract="betli")[0]
        else:                                                       # defender: record + explore
            if len(legal) > 1:
                sm = _safe_mask(pos)
                if sm is not None:
                    rows.append((encode(pos, p), sm[0], sm[1]))
            # explore: 55% god, else random (to reach off-path recoverable positions)
            mv = (pis.solve_best(pos, contract="betli")[0] if rng.random() < 0.55
                  else rng.choice(legal))
        if mv is None:
            mv = rng.choice(legal)
        pis.apply_move(pos, mv); mi += 1
    return rows


def datagen(n):
    seeds = [370_000_000 + i for i in range(n)]
    print(f"exp36 datagen: {n} deals (alpha=1.0, keep dd-lost, god soloist + exploring defenders)", flush=True)
    X, Y, L = [], [], []
    t0 = time.perf_counter(); done = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        for rows in pool.imap_unordered(_datagen_worker, seeds, chunksize=8):
            for (x, y, l) in rows:
                X.append(x); Y.append(y); L.append(l)
            done += 1
            if done % 500 == 0:
                el = time.perf_counter() - t0
                print(f"[datagen] {done}/{n} {el:.0f}s eta {(n-done)/(done/el)/60:.1f}m rows {len(X)}", flush=True)
    X = np.array(X, dtype=np.float32); Y = np.array(Y, dtype=np.float32); L = np.array(L, dtype=np.float32)
    np.savez_compressed(DATA, X=X, Y=Y, L=L)
    print(f"saved {len(X)} positions → {DATA}", flush=True)


# ── model + train ─────────────────────────────────────────────────────────────
def _make_net(hidden, arch="mlp", dropout=0.0):
    import torch.nn as nn
    if arch == "wide":
        dims = [FEAT_DIM, hidden, hidden, hidden, 32]
    elif arch == "deep":
        dims = [FEAT_DIM, hidden, hidden, hidden, hidden, 32]
    else:  # mlp
        dims = [FEAT_DIM, hidden, hidden, 32]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)


def train():
    import torch
    import torch.nn as nn
    d = np.load(DATA)
    X, Y, L = d["X"], d["Y"], d["L"]
    n = len(X); nv = max(1, int(n * 0.1)); tr = n - nv
    idx = np.random.RandomState(0).permutation(n)
    X, Y, L = X[idx], Y[idx], L[idx]
    Xt, Yt, Lt = (torch.from_numpy(a[:tr]) for a in (X, Y, L))
    Xv, Yv, Lv = (torch.from_numpy(a[tr:]) for a in (X, Y, L))
    hidden = int(os.environ.get("HIDDEN", "256")); arch = os.environ.get("ARCH", "mlp")
    epochs = int(os.environ.get("EPOCHS", "60")); lr = float(os.environ.get("LR", "1e-3"))
    wd = float(os.environ.get("WD", "1e-4")); dropout = float(os.environ.get("DROPOUT", "0.2"))
    save_to = os.environ.get("MODEL_OUT", MODEL)
    import copy
    net = _make_net(hidden, arch, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    print(f"train: {tr} pos, arch={arch} hidden={hidden} ep={epochs} lr={lr} wd={wd} drop={dropout}", flush=True)
    bs = 512

    def masked_loss(logits, y, l):
        return (bce(logits, y) * l).sum() / l.sum().clamp(min=1)

    def top1_safe(Xe, Ye, Le):                                     # frac where the argmax-legal move is SAFE
        with torch.no_grad():
            lg = net(Xe); lg = lg.masked_fill(Le < 0.5, -1e9)
            pick = lg.argmax(1)
            return Ye[torch.arange(len(Ye)), pick].mean().item()
    best_val = -1.0; best_state = None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(tr)
        for i in range(0, tr, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = masked_loss(net(Xt[b]), Yt[b], Lt[b])
            loss.backward(); opt.step()
        net.eval()
        v = top1_safe(Xv, Yv, Lv)
        if v > best_val:
            best_val = v; best_state = copy.deepcopy(net.state_dict())
        if (ep + 1) % 15 == 0 or ep == epochs - 1:
            print(f"  ep{ep+1}: val top1-safe {v:.3f} (best {best_val:.3f})", flush=True)
    import torch as _t
    _t.save({"state": best_state, "hidden": hidden, "arch": arch, "dropout": dropout}, save_to)
    rand_safe = (Yv.sum(1) / Lv.sum(1)).mean().item()
    print(f"saved {save_to}. best val top1-safe {best_val:.3f} (random-legal safe {rand_safe:.3f})", flush=True)


# ── eval: net-defense vs PIMC-defense on held-out dd-lost betlis ──────────────
_NET = None


def _load_net():
    global _NET
    import torch
    ckpt = torch.load(os.environ.get("MODEL_IN", MODEL), weights_only=False)
    net = _make_net(ckpt["hidden"], ckpt["arch"], ckpt.get("dropout", 0.0))
    net.load_state_dict(ckpt["state"]); net.eval()
    _NET = net


def _net_pick(pos, vd, seed):
    import torch
    viewer = pis.current_player(pos)
    with torch.no_grad():
        lg = _NET(torch.from_numpy(encode(pos, viewer)).unsqueeze(0))[0].numpy()
    return max(pis.legal_actions(pos), key=lambda c: lg[c.id])


def _pimc_pick(pos, vd, seed):
    return pimc_pick(pos=pos, contract="betli", n_samples=PIMC_N, seed=seed, voids_dict=vd)


_HYB_K = int(os.environ.get("HYB_K", "3"))
_HYB_NW = int(os.environ.get("HYB_NW", "16"))


def _hybrid_pick(pos, vd, seed):
    """Net-guided PIMC: the net proposes its top-K legal moves, then a focused PIMC (over the
    defender's sampled worlds, restricted to those K) picks the one that keeps the soloist losing
    in the most worlds. Combines the net's policy prior with search verification."""
    import torch
    viewer = pis.current_player(pos)
    legal = pis.legal_actions(pos)
    if len(legal) == 1:
        return legal[0]
    with torch.no_grad():
        lg = _NET(torch.from_numpy(encode(pos, viewer)).unsqueeze(0))[0].numpy()
    cand = sorted(legal, key=lambda c: lg[c.id], reverse=True)[:_HYB_K]
    if len(cand) == 1:
        return cand[0]
    rng = random.Random(seed)
    iset = _det.build_info_set(pos, viewer, "betli", voids=vd)
    safe = {c.id: 0 for c in cand}; nv = 0
    for _ in range(_HYB_NW):
        try:
            hands, tal = _det.sample_world(iset, rng)
            world = (pis.clone_with_hands_and_talon(pos, hands, tal)
                     if iset.talon_known is None else pis.clone_with_hands(pos, hands))
        except Exception:
            continue
        nv += 1
        for c in cand:
            child = world.clone(); pis.apply_move(child, c)
            if not god_says_soloist_wins(child, contract="betli"):
                safe[c.id] += 1
    if nv == 0:
        return cand[0]
    return max(cand, key=lambda c: (safe[c.id], lg[c.id]))


def _god_sol(pos, vd, seed):
    return pis.solve_best(pos, contract="betli")[0]


def _playout(deal, sol, dfn, seed):
    pos = pis.build_position(hands=[list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)],
                             soloist=0, leader=0, contract="betli", trump=None, talon=list(deal.talon))
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        mv = (sol if p == 0 else dfn)(pos, vd, seed * 131 + mi)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, mv); mi += 1
    return 0 if defenders_won(pos, "betli") else 1


def _eval_worker(seed):
    deal = deal_betli(seed=seed, alpha=1.0)
    pos0 = pis.build_position(hands=[list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)],
                              soloist=0, leader=0, contract="betli", trump=None, talon=list(deal.talon))
    if god_says_soloist_wins(pos0, contract="betli"):
        return None
    out = {"pimc": _playout(deal, _pimc_pick, _pimc_pick, seed),    # pimc sol, pimc def
           "net":  _playout(deal, _pimc_pick, _net_pick, seed)}     # pimc sol, NET def
    if os.environ.get("HYBRID") == "1":
        out["hyb"] = _playout(deal, _pimc_pick, _hybrid_pick, seed)  # pimc sol, net-guided PIMC def
    return out


def evaluate(n):
    seeds = [385_000_000 + i for i in range(n)]                    # held-out (train used 370M+)
    print(f"eval: net-defense vs pimc-defense on {n} deals' dd-lost betlis (pimc soloist)", flush=True)
    res = []
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_load_net) as pool:
        for r in pool.imap_unordered(_eval_worker, seeds, chunksize=8):
            if r is not None:
                res.append(r)
    nh = len(res)
    pimc = 100 * sum(r["pimc"] for r in res) / nh
    net = 100 * sum(r["net"] for r in res) / nh
    lines = [f"# exp36 — betli-defense EVAL, {nh} held-out dd-lost betlis (pimc soloist)\n",
             "Soloist STEAL rate (made a defender-holdable betli), LOWER = better defense; god = 0%:",
             f"- PIMC defense:   **{pimc:.1f}%**",
             f"- NET  defense:   **{net:.1f}%**  (Δ {net-pimc:+.1f}pp)"]
    if "hyb" in res[0]:
        hyb = 100 * sum(r["hyb"] for r in res) / nh
        lines.append(f"- HYBRID defense: **{hyb:.1f}%**  (Δ {hyb-pimc:+.1f}pp, net-guided PIMC)")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(_HERE, "EVAL.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "datagen"
    if cmd == "datagen":
        datagen(int(os.environ.get("N", "20000")))
    elif cmd == "train":
        train()
    elif cmd == "eval":
        evaluate(int(os.environ.get("N", "8000")))
    else:
        print(f"unknown {cmd}", flush=True)
