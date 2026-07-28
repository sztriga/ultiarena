"""God-labeled datagen for pickup-net v2.

For each deal: pick ONE random discard pair and label the resulting
position with the god solver (binary: 1 if sol wins, 0 otherwise).

Usage: python gen_god_labels.py <contract> [n_deals]
  default n_deals = 250_000
  output: <contract>_god_{N}.npz with keys X (features), y (binary label)
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from _vlib import CONTRACT_CONFIGS, EXP_DIR, featurize, input_dim

SEED_BASE  = 800_000_000
N_WORKERS  = 4
N_DEFAULT  = 250_000

# Per-contract alphas tuned for ~50% sol-wins (see alpha sweep 2026-06-06).
# durchmars caps ~37% even at high alpha — handle imbalance via pos_weight.
ALPHAS = {
    'betli':     1.5,
    'ulti':      1.4,
    'parti':     0.5,
    'durchmars': 3.0,
}

_WORKER_CONTRACT = None


def _init_worker(contract_name: str):
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = contract_name


def worker(seed: int):
    cfg = CONTRACT_CONFIGS[_WORKER_CONTRACT]
    deal = cfg.dealer(seed=seed, alpha=ALPHAS[_WORKER_CONTRACT])
    sol12 = list(deal.sol_hand) + list(deal.talon)
    if len(sol12) != 12:
        raise RuntimeError(f"expected 12 cards, got {len(sol12)}")
    d1 = list(deal.def1_hand); d2 = list(deal.def2_hand)
    trump = deal.trump if cfg.has_trump else None

    rng = random.Random(seed ^ 0xA5A5A5A5)
    idx = rng.sample(range(12), 2)
    discard = [sol12[i] for i in idx]
    remaining = [sol12[i] for i in range(12) if i not in idx]

    pos = pis.build_position(
        hands=[remaining, d1, d2], soloist=0, leader=0,
        contract=cfg.solver, trump=trump, talon=discard,
    )
    label = 1 if god_says_soloist_wins(pos, contract=cfg.solver) else 0
    return featurize(remaining, trump, cfg.has_trump), label


def god_data_path(contract: str, n: int) -> Path:
    if n % 1000 == 0:
        n_tag = f"{n//1000}k" if n < 1_000_000 else f"{n//1000_000}M"
    else:
        n_tag = str(n)
    return EXP_DIR / f"{contract}_god_{n_tag}.npz"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CONTRACT_CONFIGS:
        print(f"Usage: gen_god_labels.py <contract> [n_deals]  "
              f"(contracts: {list(CONTRACT_CONFIGS)})")
        sys.exit(1)
    contract = sys.argv[1]
    n_deals = int(sys.argv[2]) if len(sys.argv) >= 3 else N_DEFAULT
    cfg = CONTRACT_CONFIGS[contract]
    out_path = god_data_path(contract, n_deals)

    print(f"=== God-label datagen: contract={contract} ===")
    print(f"  dealer:     {cfg.dealer.__name__}")
    print(f"  has_trump:  {cfg.has_trump}  input_dim: {input_dim(cfg)}")
    print(f"  N_deals:    {n_deals} (1 record / deal, random discard)")
    print(f"  workers:    {N_WORKERS}  alpha: {ALPHAS[contract]}")
    print(f"  out:        {out_path.name}", flush=True)

    seeds = [SEED_BASE + i for i in range(n_deals)]
    t0 = time.perf_counter()
    Xs = []; ys = []
    report_every = max(1000, n_deals // 50)
    done = 0
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(contract,)) as pool:
        for hv, label in pool.imap_unordered(worker, seeds, chunksize=128):
            Xs.append(hv); ys.append(label)
            done += 1
            if done % report_every == 0:
                wall = time.perf_counter() - t0
                rate = done / wall
                eta = (n_deals - done) / rate
                print(f"  {done}/{n_deals}  wall={wall:.0f}s  "
                      f"rate={rate:.0f}/s  eta={eta:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    X = np.stack(Xs)
    y = np.array(ys, dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=y)
    print()
    print(f"Saved {X.shape[0]} records → {out_path}")
    print(f"Wall: {wall:.0f}s  ({1000*wall/X.shape[0]:.2f} ms/record)")
    print(f"Label balance: sol_wins = {y.mean():.4f}  "
          f"({int(y.sum())}/{len(y)})")


if __name__ == "__main__":
    main()
