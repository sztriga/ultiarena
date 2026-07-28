"""Fit isotonic calibration per contract.

For each contract:
  1. Generate N=50k random α=0 deals
  2. Random discard per deal → 10-card final hand + god label (0/1)
  3. v2 net predicts p̂
  4. Fit IsotonicRegression(p̂ → label)
  5. Save calibrator to {contract}_calib.pkl in exp 15 dir

Wall: ~5-15 minutes for all 4 contracts (mostly god-solver labeling).
"""
from __future__ import annotations

import itertools, pickle, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.vnet.pickup import CONTRACT_CONFIGS, PickupNetV2, featurize, input_dim
from ulti.card import DECK, Card

EXP15_DIR = Path(__file__).parent.parent / "15_vnet_pickup"

N_DEALS    = 50_000
SEED_BASE  = 950_000_000
N_WORKERS  = 4
ALPHA      = 0.0  # uniform random — matches test distribution

_WORKER_CONTRACT = None


def _init_worker(contract_name: str):
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = contract_name


def worker(seed: int):
    """Returns (32 or 36-dim feature vector, binary label)."""
    cfg = CONTRACT_CONFIGS[_WORKER_CONTRACT]
    deal = cfg.dealer(seed=seed, alpha=ALPHA)
    sol12 = list(deal.sol_hand) + list(deal.talon)
    if len(sol12) != 12:
        raise RuntimeError(f"expected 12 cards, got {len(sol12)}")
    d1 = list(deal.def1_hand); d2 = list(deal.def2_hand)
    trump = deal.trump if cfg.has_trump else None

    rng = random.Random(seed ^ 0xC3C3C3C3)
    idx = rng.sample(range(12), 2)
    discard = [sol12[i] for i in idx]
    remaining = [sol12[i] for i in range(12) if i not in idx]
    pos = pis.build_position(
        hands=[remaining, d1, d2], soloist=0, leader=0,
        contract=cfg.solver, trump=trump, talon=discard,
    )
    label = 1 if god_says_soloist_wins(pos, contract=cfg.solver) else 0
    return featurize(remaining, trump, cfg.has_trump), label


def _v2_weights(name: str) -> Path:
    return EXP15_DIR / f"{name}_vnet_v2.pt"


def _calib_path(name: str) -> Path:
    return EXP15_DIR / f"{name}_calib.pkl"


def main():
    for name in CONTRACT_CONFIGS:
        cfg = CONTRACT_CONFIGS[name]
        print(f"\n=== {name} ===")
        seeds = [SEED_BASE + i for i in range(N_DEALS)]
        t0 = time.perf_counter()
        Xs = []; ys = []
        with Pool(N_WORKERS, initializer=_init_worker,
                  initargs=(name,)) as pool:
            for hv, label in pool.imap_unordered(worker, seeds, chunksize=128):
                Xs.append(hv); ys.append(label)
        wall = time.perf_counter() - t0
        X = np.stack(Xs)
        y = np.array(ys, dtype=np.float32)
        pos_rate = y.mean()
        print(f"  datagen: {N_DEALS} hands in {wall:.0f}s   "
              f"positive rate: {pos_rate*100:.1f}%")

        # Load v2 net, get predictions
        net = PickupNetV2(in_dim=input_dim(cfg))
        net.load_state_dict(torch.load(_v2_weights(name), weights_only=True))
        net.eval()
        with torch.no_grad():
            p_hat = net(torch.from_numpy(X)).numpy()

        # Calibration error before
        mae_before = float(np.abs(p_hat - y).mean())
        brier_before = float(((p_hat - y) ** 2).mean())

        # Fit isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(p_hat, y)
        p_cal = iso.predict(p_hat)

        # Calibration error after
        mae_after = float(np.abs(p_cal - y).mean())
        brier_after = float(((p_cal - y) ** 2).mean())

        print(f"  v2 raw    : MAE={mae_before:.4f}  Brier={brier_before:.4f}  "
              f"mean_p̂={p_hat.mean():.3f}")
        print(f"  calibrated: MAE={mae_after:.4f}  Brier={brier_after:.4f}  "
              f"mean_p̂={p_cal.mean():.3f}")

        # Calibration table per bin
        bins = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
        print(f"  {'bin':>14}  {'n':>5}  {'raw p̂':>8}  {'cal p̂':>8}  "
              f"{'actual':>8}")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p_hat >= lo) & (p_hat < hi)
            if not m.any():
                continue
            print(f"  [{lo:.2f},{hi:.2f})  {int(m.sum()):>5}  "
                  f"{p_hat[m].mean():>7.3f}  {p_cal[m].mean():>7.3f}  "
                  f"{y[m].mean():>7.3f}")

        # Save
        with open(_calib_path(name), 'wb') as f:
            pickle.dump(iso, f)
        print(f"  saved → {_calib_path(name)}")


if __name__ == "__main__":
    main()
