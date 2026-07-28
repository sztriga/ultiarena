"""PIMC plays the whole hand. AUC vs cards-remaining + final outcome.

50 deals α=0.7. All three players use PIMC(N) to choose moves. At each
trick start where the on-move player has K cards, we record:
  * fresh god label at that state
  * per-viewpoint PIMC prob from N fresh samples (same N as the policy)
  * voids feed: tracked from the actual playout
At terminal, compare final outcome vs god opening verdict.

Usage:
    PYTHONPATH=. python3 scripts/pimc_play_auc_vs_cards.py --n 1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from eval.dojo import deal_betli
from solvers import determinize as _det
from solvers import pis as pis_bridge
from solvers import pimc as pimc_player

_BETLI_WIN_VAL = 10.0
_CONTRACT = "betli"
N_DEALS = 50
ALPHA = 0.7
K_GRID = [10, 9, 8, 7, 6, 5, 4, 3]


def _god_label(pos) -> int:
    vals = pis_bridge.solve_all(pos, contract=_CONTRACT)
    return 1 if max(vals.values()) >= _BETLI_WIN_VAL - 1e-6 else 0


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


def _final_outcome(pos) -> int:
    return 1 if len(pos.captured[pos.soloist]) == 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    n = args.n

    print(f"PIMC plays out: N={n}, α={ALPHA}, {N_DEALS} deals\n")

    checkpoints: dict[int, list[tuple[int, float, float]]] = {k: [] for k in K_GRID}
    final_rows = []   # (seed, god_open, pimc_final)

    t0 = time.perf_counter()
    for s in range(N_DEALS):
        deal = deal_betli(seed=s, alpha=ALPHA)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis_bridge.build_position(
            hands=hands, soloist=0, leader=0, contract=_CONTRACT,
            talon=list(deal.talon),
        )
        god_open = _god_label(pos)
        voids = _det.Voids()
        seed_ctr = s
        next_k_idx = 0

        while not pis_bridge.is_terminal(pos):
            if _trick_start(pos) and next_k_idx < len(K_GRID):
                cur = pis_bridge.current_player(pos)
                k_here = _hand_size(pos, cur)
                target_k = K_GRID[next_k_idx]
                if k_here == target_k:
                    label = _god_label(pos)
                    vd = voids.as_dict()
                    sol_p = _viewpoint_prob(pos, 0, n, s * 7919 + 1 + 31 * next_k_idx, vd)
                    d1_p  = _viewpoint_prob(pos, 1, n, s * 7919 + 2 + 41 * next_k_idx, vd)
                    d2_p  = _viewpoint_prob(pos, 2, n, s * 7919 + 3 + 53 * next_k_idx, vd)
                    def_p = (d1_p + d2_p) / 2
                    checkpoints[target_k].append((label, sol_p, def_p))
                    next_k_idx += 1
            # PIMC chooses the move
            seed_ctr += 1
            player = pis_bridge.current_player(pos)
            chosen, averaged = pimc_player.pimc_decision(
                true_pos=pos, contract=_CONTRACT, n_samples=n, seed=seed_ctr,
                voids=voids.as_dict(),
            )
            # pimc_decision returns soloist-perspective argmax. Defenders
            # need argmin (see experiments/10_betli_pimc_audit).
            if player != 0 and averaged:
                chosen = min(averaged, key=lambda c: averaged[c])
            if chosen is None:
                break
            voids.observe(pos, player, chosen)
            pis_bridge.apply_move(pos, chosen)

        pimc_final = _final_outcome(pos)
        final_rows.append((s, god_open, pimc_final))
        print(f"  seed={s:>3}  god={god_open}  pimc={pimc_final}  "
              f"chk={next_k_idx}  wall={time.perf_counter() - t0:>6.1f}s",
              flush=True)

    # AUC table
    print()
    print(f"{'K':>3}  {'n':>3}  {'pos_rate':>8}  {'AUC_sol':>8}  {'AUC_def':>8}  "
          f"{'sol_p̄':>7}  {'def_p̄':>7}")
    auc_rows = []
    for k in K_GRID:
        data = checkpoints[k]
        if not data:
            print(f"{k:>3}    0    (no data)")
            continue
        labels = [x[0] for x in data]
        sol_p  = [x[1] for x in data]
        def_p  = [x[2] for x in data]
        pos_rate = sum(labels) / len(labels)
        auc_sol = _auc(labels, sol_p)
        auc_def = _auc(labels, def_p)
        auc_rows.append({
            "k": k, "n": len(data), "pos_rate": pos_rate,
            "auc_sol": auc_sol, "auc_def": auc_def,
            "sol_p_mean": sum(sol_p) / len(sol_p),
            "def_p_mean": sum(def_p) / len(def_p),
        })
        print(f"{k:>3}  {len(data):>3}  {pos_rate:>8.2%}  "
              f"{auc_sol:>8.3f}  {auc_def:>8.3f}  "
              f"{sum(sol_p)/len(sol_p):>7.3f}  {sum(def_p)/len(def_p):>7.3f}")

    # Final outcome confusion matrix
    print()
    print("Final-outcome confusion (god opening vs PIMC played-out):")
    print("                     pimc=sol   pimc=def")
    for g_label, g_name in [(1, "god=soloist  "), (0, "god=defenders")]:
        ps = sum(1 for _, g, p in final_rows if g == g_label and p == 1)
        pd = sum(1 for _, g, p in final_rows if g == g_label and p == 0)
        print(f"  {g_name}        {ps:>3}        {pd:>3}")

    out = {"alpha": ALPHA, "n_samples": n, "n_deals": N_DEALS,
           "k_grid": K_GRID, "auc_rows": auc_rows,
           "final_rows": final_rows,
           "wall_s": time.perf_counter() - t0}
    Path("/tmp/pimc_play_auc.json").write_text(json.dumps(out, indent=2))
    print(f"\nTotal wall: {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
