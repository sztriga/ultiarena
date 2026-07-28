"""Generate (defender info-set features, god-solver label) training pairs.

Contract-parameterised: pass ``--contract betli|parti|...``, the rest of the
flow (dealer, solver, feature extraction, label conversion) is dispatched
through ``vnet.contracts.REGISTRY[contract]``.

For each deal:
  1. Sample a contract-biased deal at the given α.
  2. Play forward to a random trick boundary t ∈ [t_min, t_max] using a
     random-legal policy on every move. (Random play, not god-play — this
     gives state-coverage breadth.)
  3. At that trick boundary, compute v* on the *actual* world via the
     contract's god solver. Reduce to a scalar label via ``cfg.label_fn``.
  4. If ``--k-labels K > 1`` and we're at the opening (t==0): sample K-1
     additional worlds from the α-biased posterior consistent with the
     defender's info-set, solve each, and average the K labels into a soft
     label.
  5. Emit two examples — one per defender — each with viewer-specific
     features and viewer-specific soft label.

Output: .npz with ``X`` (N × cfg.feature_dim float32), ``y`` (N float32),
``meta`` (dict including ``contract`` and ``target_kind`` so ``train.py``
can self-configure).

Parallelism: ``--workers N`` runs deals across N processes via
``multiprocessing`` (spawn context). Workers are seeded deterministically
from ``seed_base + deal_index``, so output is reproducible regardless of
worker count.

Usage:
    PYTHONPATH=. python3 -m vnet.data_gen \\
        --contract parti \\
        --alpha 0.0 --alpha-max 1.0 --n-deals 200000 --seed-base 60000000 \\
        --t-min 0 --t-max 8 --k-labels 1 --workers 8 \\
        --out vnet/parti/data/train_a00_10_t0_8_200k.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np


def _god_label(pos, cfg) -> float:
    """Run the contract's god solver and reduce to a scalar label."""
    from solvers import pis as pis_bridge
    vals = pis_bridge.solve_all(pos, contract=cfg.solver_name)
    return cfg.label_fn(max(vals.values()))


def _trick_start(pos) -> bool:
    return len(getattr(pos, "trick_cards", []) or []) == 0


def _play_random_to_trick(pos, target_trick, rng, voids, played, tricks_completed):
    from solvers import pis as pis_bridge
    cur_trick = []
    while not pis_bridge.is_terminal(pos):
        if _trick_start(pos):
            if cur_trick:
                tricks_completed.append(cur_trick)
                played.extend(c for _, c in cur_trick)
                cur_trick = []
            if pos.trick_no >= target_trick:
                return
        player = pis_bridge.current_player(pos)
        legal = pis_bridge.legal_actions(pos)
        if not legal:
            return
        chosen = rng.choice(legal)
        voids.observe(pos, player, chosen)
        cur_trick.append((player, chosen))
        pis_bridge.apply_move(pos, chosen)
    if cur_trick:
        played.extend(c for _, c in cur_trick)


def _process_deal(args):
    """Worker: process a single deal. Returns list of (features, label) rows."""
    seed, alpha_lo, alpha_hi, t_min, t_max, k_labels, seed_base, contract_name = args
    # Lazy imports inside worker so spawn-mode startup doesn't re-import in main.
    from solvers import determinize as _det
    from solvers import pis as pis_bridge
    from .contracts import get as _get_contract
    from .features import extract_features

    cfg = _get_contract(contract_name)

    rng = random.Random(seed_base * 1_000_003 ^ seed)
    alpha = alpha_lo if alpha_hi == alpha_lo else rng.uniform(alpha_lo, alpha_hi)
    deal = cfg.deal_fn(seed=seed, alpha=alpha)
    hands_init = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands_init, soloist=0, leader=0, contract=cfg.solver_name,
        trump=deal.trump, talon=list(deal.talon),
    )
    target = rng.randint(t_min, t_max)

    voids = _det.Voids()
    played:           List = []
    tricks_completed: List = []
    _play_random_to_trick(pos, target, rng, voids, played, tricks_completed)

    if pis_bridge.is_terminal(pos):
        return []

    label_actual = _god_label(pos, cfg)
    voids_dict = voids.as_dict()
    is_opening = (pos.trick_no == 0 and not played)

    rows = []
    for viewer in (0, 1, 2):
        feats = extract_features(
            cfg=cfg,
            hands=pis_bridge.hands_by_player(pos),
            played_cards=played,
            current_trick=[],
            leader=pos.leader,
            trick_no=pos.trick_no,
            voids=voids_dict,
            soloist=0,
            viewer=viewer,
            trump=deal.trump if cfg.has_trump else None,
        )

        if k_labels > 1:
            iset = _det.build_info_set(
                pos, viewer=viewer, contract=cfg.solver_name, voids=voids_dict,
            )
            label_sum = label_actual
            n_solved  = 1
            for _ in range(k_labels - 1):
                alt_hands, alt_talon = _det.sample_world(
                    iset, rng, sol_prior_alpha=alpha,
                )
                if is_opening:
                    alt_pos = pis_bridge.build_position(
                        hands=alt_hands, soloist=0, leader=0,
                        contract=cfg.solver_name, trump=deal.trump,
                        talon=alt_talon,
                    )
                else:
                    alt_pos = pis_bridge.clone_with_hands_and_talon(
                        pos, alt_hands, alt_talon,
                    )
                label_sum += _god_label(alt_pos, cfg)
                n_solved  += 1
            label = label_sum / n_solved
        else:
            label = label_actual

        rows.append((feats, label))
    return rows


def generate(contract_name, alpha_lo, alpha_hi, n_deals, seed_base,
             t_min=0, t_max=8, k_labels=1, workers=1, verbose=False):
    from .contracts import get as _get_contract
    cfg = _get_contract(contract_name)

    X = np.zeros((3 * n_deals, cfg.feature_dim), dtype=np.float32)
    y = np.zeros((3 * n_deals,),                  dtype=np.float32)
    n_kept = 0
    t0 = time.perf_counter()

    seeds = [seed_base + d for d in range(n_deals)]
    args_iter = ((s, alpha_lo, alpha_hi, t_min, t_max, k_labels,
                  seed_base, contract_name) for s in seeds)

    if workers <= 1:
        results_iter = map(_process_deal, args_iter)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        results_iter = pool.imap_unordered(_process_deal, args_iter, chunksize=8)

    try:
        for d, rows in enumerate(results_iter):
            for feats, label in rows:
                X[n_kept] = feats
                y[n_kept] = label
                n_kept += 1
            if verbose and (d + 1) % 200 == 0:
                if cfg.target_kind == "binary":
                    wr = (y[:n_kept] > 0).mean() if n_kept else 0.0
                    extra = f"def_winrate={wr:.3f}"
                else:
                    extra = f"mean_y={y[:n_kept].mean():.2f}"
                print(f"  {d+1}/{n_deals}  n_examples={n_kept}  {extra}  "
                      f"wall={time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    return X[:n_kept], y[:n_kept], cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=str, required=True,
                    help="Contract name (e.g. 'betli', 'parti').")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="If set, draw α ~ Uniform(--alpha, --alpha-max) per deal.")
    ap.add_argument("--n-deals", type=int, default=5000)
    ap.add_argument("--seed-base", type=int, default=20260517)
    ap.add_argument("--t-min", type=int, default=0)
    ap.add_argument("--t-max", type=int, default=8)
    ap.add_argument("--k-labels", type=int, default=1,
                    help="Worlds solved per (deal, viewer); labels averaged. "
                         "K>1 requires t-max=0.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    alpha_lo = args.alpha
    alpha_hi = args.alpha_max if args.alpha_max is not None else args.alpha
    print(f"generating: contract={args.contract}  "
          f"alpha∈[{alpha_lo},{alpha_hi}]  n_deals={args.n_deals}  "
          f"t∈[{args.t_min},{args.t_max}]  k_labels={args.k_labels}  "
          f"workers={args.workers}  seed_base={args.seed_base}")
    X, y, cfg = generate(args.contract, alpha_lo, alpha_hi, args.n_deals,
                         args.seed_base, t_min=args.t_min, t_max=args.t_max,
                         k_labels=args.k_labels, workers=args.workers,
                         verbose=not args.quiet)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(
        contract     = cfg.name,
        target_kind  = cfg.target_kind,
        feature_dim  = cfg.feature_dim,
        alpha_lo     = alpha_lo,
        alpha_hi     = alpha_hi,
        n_deals      = args.n_deals,
        seed_base    = args.seed_base,
        t_min        = args.t_min,
        t_max        = args.t_max,
        k_labels     = args.k_labels,
    )
    np.savez_compressed(out, X=X, y=y, meta=meta)
    print(f"\nsaved: {out}")
    print(f"  shape: X={X.shape}  y={y.shape}")
    if cfg.target_kind == "binary":
        print(f"  defender-favoring fraction (y>0): {(y > 0).mean():.3f}")
        print(f"  mean y: {y.mean():.3f}  std y: {y.std():.3f}")
        print(f"  fraction of soft (|y|<1) labels: {(np.abs(y) < 0.999).mean():.3f}")
    else:
        print(f"  y range: [{y.min():.2f}, {y.max():.2f}]  "
              f"mean={y.mean():.2f}  std={y.std():.2f}")


if __name__ == "__main__":
    main()
