"""PIMC AUC vs cards-remaining, N=32, α=0.7, 50 deals.

For each deal:
  * play out with god-optimal moves on both sides;
  * at each trick start where each player has K cards left (K=10..3),
    record the god label at that state and the N=32 PIMC probability
    from each of the three viewpoints (voids accumulated from playout).

Aggregate AUC over the 50 deals at each K.

Usage:
    PYTHONPATH=. python3 scripts/pimc_auc_vs_cards.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from ulti.eval.dojo import deal_betli
from ulti.solvers import determinize as _det
from ulti.solvers import pis as pis_bridge

_BETLI_WIN_VAL = 10.0
_CONTRACT = "betli"
N_SAMPLES = 32
N_DEALS = 50
ALPHA = 0.7
K_GRID = [10, 9, 8, 7, 6, 5, 4, 3]


def _god_label(pos) -> int:
    vals = pis_bridge.solve_all(pos, contract=_CONTRACT)
    return 1 if max(vals.values()) >= _BETLI_WIN_VAL - 1e-6 else 0


def _god_best_move(pos):
    vals = pis_bridge.solve_all(pos, contract=_CONTRACT)
    # current player's POV: soloist maximises, defenders minimise
    cur = pis_bridge.current_player(pos)
    soloist = pos.soloist
    if cur == soloist:
        return max(vals, key=lambda c: vals[c])
    return min(vals, key=lambda c: vals[c])


def _viewpoint_prob(pos, viewer: int, n: int, seed: int, voids_dict) -> float:
    rng = random.Random(seed)
    iset = _det.build_info_set(pos, viewer, _CONTRACT, voids=voids_dict)
    wins = 0
    for _ in range(n):
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            sample_pos = pis_bridge.clone_with_hands_and_talon(pos, hands, talon)
        else:
            sample_pos = pis_bridge.clone_with_hands(pos, hands)
        vals = pis_bridge.solve_all(sample_pos, contract=_CONTRACT)
        if max(vals.values()) >= _BETLI_WIN_VAL - 1e-6:
            wins += 1
    return wins / n


def _auc(labels, scores) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    paired = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return paired / (len(pos) * len(neg))


def _hand_size(pos, player: int) -> int:
    return len(pis_bridge.hands_by_player(pos)[player])


def _trick_start(pos) -> bool:
    return len(getattr(pos, "trick_cards", []) or []) == 0


def main() -> None:
    print(f"PIMC AUC vs cards remaining: N={N_SAMPLES}, α={ALPHA}, "
          f"{N_DEALS} deals, K∈{K_GRID}\n")

    # checkpoint[K] = list of (god_label, sol_p, def_p) over the 50 deals
    checkpoints: dict[int, list[tuple[int, float, float]]] = {k: [] for k in K_GRID}

    t0 = time.perf_counter()
    for s in range(N_DEALS):
        deal = deal_betli(seed=s, alpha=ALPHA)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis_bridge.build_position(
            hands=hands, soloist=0, leader=0, contract=_CONTRACT,
            talon=list(deal.talon),
        )
        voids = _det.Voids()

        next_k_idx = 0  # walk through K_GRID as the game shrinks
        while not pis_bridge.is_terminal(pos) and next_k_idx < len(K_GRID):
            if _trick_start(pos):
                cur = pis_bridge.current_player(pos)
                k_here = _hand_size(pos, cur)
                target_k = K_GRID[next_k_idx]
                if k_here == target_k:
                    label = _god_label(pos)
                    voids_dict = voids.as_dict()
                    sol_p = _viewpoint_prob(pos, 0, N_SAMPLES, s * 7919 + 1 + next_k_idx,    voids_dict)
                    d1_p  = _viewpoint_prob(pos, 1, N_SAMPLES, s * 7919 + 2 + 17 * next_k_idx, voids_dict)
                    d2_p  = _viewpoint_prob(pos, 2, N_SAMPLES, s * 7919 + 3 + 23 * next_k_idx, voids_dict)
                    def_p = (d1_p + d2_p) / 2
                    checkpoints[target_k].append((label, sol_p, def_p))
                    next_k_idx += 1
            # advance one ply
            player = pis_bridge.current_player(pos)
            mv = _god_best_move(pos)
            voids.observe(pos, player, mv)
            pis_bridge.apply_move(pos, mv)

        print(f"  seed={s:>3}  checkpoints collected={next_k_idx}  "
              f"wall={time.perf_counter() - t0:>6.1f}s", flush=True)

    print()
    print(f"{'K':>3}  {'n':>3}  {'pos_rate':>8}  {'AUC_sol':>8}  {'AUC_def':>8}  "
          f"{'sol_p̄':>7}  {'def_p̄':>7}")
    rows = []
    for k in K_GRID:
        data = checkpoints[k]
        if not data:
            continue
        labels = [x[0] for x in data]
        sol_p  = [x[1] for x in data]
        def_p  = [x[2] for x in data]
        pos_rate = sum(labels) / len(labels)
        auc_sol = _auc(labels, sol_p)
        auc_def = _auc(labels, def_p)
        row = {
            "k": k, "n": len(data), "pos_rate": pos_rate,
            "auc_sol": auc_sol, "auc_def": auc_def,
            "sol_p_mean": sum(sol_p) / len(sol_p),
            "def_p_mean": sum(def_p) / len(def_p),
        }
        rows.append(row)
        print(f"{k:>3}  {len(data):>3}  {pos_rate:>8.2%}  "
              f"{auc_sol:>8.3f}  {auc_def:>8.3f}  "
              f"{row['sol_p_mean']:>7.3f}  {row['def_p_mean']:>7.3f}")

    out = {"alpha": ALPHA, "n_samples": N_SAMPLES, "n_deals": N_DEALS,
           "k_grid": K_GRID, "rows": rows,
           "wall_s": time.perf_counter() - t0}
    Path("/tmp/auc_vs_cards.json").write_text(json.dumps(out, indent=2))
    print(f"\nTotal wall: {out['wall_s']:.1f}s")
    print("saved → /tmp/auc_vs_cards.json")


if __name__ == "__main__":
    main()
