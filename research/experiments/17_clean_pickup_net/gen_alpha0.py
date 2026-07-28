"""α=0 god-labeled datagen for exp 17 multi-head pickup net.

Per contract: N deals at α=0 (random Ulti deals), one random discard per
deal, binary god-solver label. Saves <contract>_god_alpha0_<tag>.npz.

Usage: python gen_alpha0.py [n_deals_per_contract]   (default 1_000_000)
Runs all 4 contracts sequentially. Contract is per-process via initializer.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from solvers import pis
from eval.pimc_matchup import god_says_soloist_wins
from vnet.pickup import CONTRACT_CONFIGS, featurize, input_dim

SEED_BASE = 800_000_000   # disjoint from eval seeds 100000-102999
N_WORKERS = 6
N_DEFAULT = 1_000_000

EXP_DIR = Path(__file__).parent

_WORKER_CONTRACT = None


def _init_worker(contract_name: str):
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = contract_name


def worker(seed: int):
    cfg = CONTRACT_CONFIGS[_WORKER_CONTRACT]
    # alpha=0 → uniform random Ulti deal (the deployment distribution).
    deal = cfg.dealer(seed=seed, alpha=0.0)
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


def data_path(contract: str, n: int) -> Path:
    if n % 1000 == 0:
        n_tag = f"{n//1000}k" if n < 1_000_000 else f"{n//1_000_000}M"
    else:
        n_tag = str(n)
    return EXP_DIR / f"{contract}_god_alpha0_{n_tag}.npz"


def gen_for_contract(contract: str, n_deals: int) -> Path:
    cfg = CONTRACT_CONFIGS[contract]
    out_path = data_path(contract, n_deals)

    if out_path.exists():
        d = np.load(out_path)
        if len(d['y']) >= n_deals:
            print(f"  [skip] {out_path.name} already has {len(d['y'])} records",
                  flush=True)
            return out_path

    print(f"\n=== α=0 datagen: contract={contract} ===", flush=True)
    print(f"  dealer:    {cfg.dealer.__name__}", flush=True)
    print(f"  has_trump: {cfg.has_trump}  input_dim: {input_dim(cfg)}", flush=True)
    print(f"  N_deals:   {n_deals}  workers: {N_WORKERS}", flush=True)
    print(f"  out:       {out_path.name}", flush=True)

    seeds = [SEED_BASE + i for i in range(n_deals)]
    t0 = time.perf_counter()
    Xs = []; ys = []
    report_every = max(5000, n_deals // 40)
    done = 0
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(contract,)) as pool:
        for hv, label in pool.imap_unordered(worker, seeds, chunksize=256):
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
    print(f"  saved {X.shape[0]} records  wall={wall:.0f}s "
          f"({1000*wall/X.shape[0]:.2f} ms/rec)", flush=True)
    print(f"  positive rate: {y.mean():.4f}  ({int(y.sum())} pos)", flush=True)
    return out_path


def main():
    n_deals = int(sys.argv[1]) if len(sys.argv) >= 2 else N_DEFAULT
    print(f"=== Exp 17 α=0 datagen ===")
    print(f"  N per contract: {n_deals}")
    print(f"  workers:        {N_WORKERS}", flush=True)
    for contract in ('durchmars', 'ulti', 'betli', 'parti'):
        gen_for_contract(contract, n_deals)
    print("\n=== Done all contracts ===")


if __name__ == "__main__":
    main()
