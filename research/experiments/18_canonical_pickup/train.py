"""Train exp 18 pickup-net variants.

Adapted from experiments/17_clean_pickup_net/train.py. Variants:
  a — canonical 32-dim features, α=0 data only
  b — exp 17 features (36-dim + suit-perm aug), betli/duri biased mix
  c — canonical 32-dim features, betli/duri biased mix

Mix = α=0 1M (exp 17) + α-biased 250k (exp 15) for betli/durchmars
only. Val split is carved from the α=0 portion only, so best-epoch
selection stays on the deployment distribution.

Usage: python train.py <a|b|c>
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.vnet.pickup import (
    CONTRACT_CONFIGS, MultiHeadPickupNet, pad_to_unified, canonicalize,
    CANON_DIM, UNIFIED_DIM, random_suit_perm, apply_suit_perm,
)

EXP_DIR   = Path(__file__).parent
EXP15_DIR = EXP_DIR.parent / "15_vnet_pickup"
EXP17_DIR = EXP_DIR.parent / "17_clean_pickup_net"

VARIANTS = {
    'a': {'canonical': True,  'mix': False},
    'b': {'canonical': False, 'mix': True},
    'c': {'canonical': True,  'mix': True},
}

EPOCHS  = 30
PER_CONTRACT_BATCH = 256   # batch = 1024 total
LR      = 1e-3
DROPOUT = 0.1
VAL_FRAC = 0.02
SEED    = 0
MIX_CONTRACTS = ('betli', 'durchmars')


def weights_path(variant: str) -> Path:
    return EXP_DIR / f"multihead_v18{variant}.pt"


def augment_batch(X, has_trump, rng):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        sigma = random_suit_perm(has_trump, X[i], rng)
        out[i] = apply_suit_perm(X[i], sigma, has_trump)
    return out


def load_contract(contract: str, *, canonical: bool, mix: bool, rng):
    """Returns dict with X_tr/y_tr/X_va/y_va (val = α=0 records only)."""
    cfg = CONTRACT_CONFIGS[contract]
    d = np.load(EXP17_DIR / f"{contract}_god_alpha0_1M.npz")
    X0, y0 = d['X'], d['y']

    n = len(y0)
    perm = rng.permutation(n)
    nv = int(n * VAL_FRAC)
    val_idx, tr_idx = perm[:nv], perm[nv:]
    X_tr, y_tr = X0[tr_idx], y0[tr_idx]
    X_va, y_va = X0[val_idx], y0[val_idx]
    n_biased = 0

    if mix and contract in MIX_CONTRACTS:
        db = np.load(EXP15_DIR / f"{contract}_god_250k.npz")
        X_tr = np.concatenate([X_tr, db['X']])
        y_tr = np.concatenate([y_tr, db['y']])
        n_biased = len(db['y'])

    if canonical:
        X_tr = canonicalize(X_tr, cfg.has_trump)
        X_va = canonicalize(X_va, cfg.has_trump)
    else:
        X_tr = pad_to_unified(X_tr, cfg.has_trump)
        X_va = pad_to_unified(X_va, cfg.has_trump)

    return {
        'X_tr': X_tr, 'y_tr': y_tr,
        'X_va': torch.from_numpy(X_va), 'y_va': torch.from_numpy(y_va),
        'has_trump': cfg.has_trump, 'n_biased': n_biased,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
        print("Usage: train.py <a|b|c>")
        sys.exit(1)
    variant = sys.argv[1]
    canonical = VARIANTS[variant]['canonical']
    mix = VARIANTS[variant]['mix']
    contracts = list(CONTRACT_CONFIGS)
    in_dim = CANON_DIM if canonical else UNIFIED_DIM

    print(f"=== Exp 18 training: variant {variant} "
          f"(canonical={canonical}, mix={mix}) ===")
    rng = np.random.default_rng(SEED)
    splits = {}
    for c in contracts:
        s = load_contract(c, canonical=canonical, mix=mix, rng=rng)
        splits[c] = s
        print(f"  {c:>10}: {len(s['y_tr'])} train "
              f"({s['n_biased']} biased), {len(s['y_va'])} val (α=0), "
              f"train pos {s['y_tr'].mean()*100:.2f}%", flush=True)
    print(f"  arch: {in_dim}→128→64 body, 64→32→1 per head")
    print(f"  epochs={EPOCHS}  per-contract-batch={PER_CONTRACT_BATCH}  "
          f"lr={LR}  aug={'none (canonical)' if canonical else 'suit-perm'}",
          flush=True)

    torch.manual_seed(SEED)
    model = MultiHeadPickupNet(dropout=DROPOUT, contracts=contracts,
                               input_dim=in_dim)
    print(f"  total params: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    aug_rng = np.random.default_rng(SEED + 1)
    n_train = {c: len(splits[c]['y_tr']) for c in contracts}
    steps_per_epoch = min(n_train.values()) // PER_CONTRACT_BATCH

    t0 = time.perf_counter()
    best_val_brier = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perms = {c: rng.permutation(n_train[c]) for c in contracts}
        epoch_loss = {c: 0.0 for c in contracts}
        n_seen = {c: 0 for c in contracts}
        for step in range(steps_per_epoch):
            total_loss = 0.0
            for c in contracts:
                idx = perms[c][step * PER_CONTRACT_BATCH:
                               (step + 1) * PER_CONTRACT_BATCH]
                xb = splits[c]['X_tr'][idx]
                if not canonical:
                    xb = augment_batch(xb, splits[c]['has_trump'], aug_rng)
                xb_t = torch.from_numpy(np.ascontiguousarray(xb))
                yb_t = torch.from_numpy(splits[c]['y_tr'][idx])
                pred = model(xb_t, c)
                loss = loss_fn(pred, yb_t)
                total_loss = total_loss + loss
                epoch_loss[c] += loss.item() * len(idx)
                n_seen[c] += len(idx)
            opt.zero_grad(); total_loss.backward(); opt.step()

        model.eval()
        v_briers = {}
        with torch.no_grad():
            for c in contracts:
                vp = model(splits[c]['X_va'], c)
                v_briers[c] = ((vp - splits[c]['y_va']) ** 2).mean().item()
        v_brier_mean = sum(v_briers.values()) / len(v_briers)
        if v_brier_mean < best_val_brier:
            best_val_brier = v_brier_mean
            torch.save(model.state_dict(), weights_path(variant))

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            wall = time.perf_counter() - t0
            losses = "  ".join(f"{c[:3]}={epoch_loss[c]/n_seen[c]:.4f}"
                               for c in contracts)
            briers = "  ".join(f"{c[:3]}={v_briers[c]:.4f}"
                               for c in contracts)
            print(f"  ep {epoch:>3} | tr {losses} | val_brier  {briers} | "
                  f"mean {v_brier_mean:.5f} | {wall:.0f}s", flush=True)

    wall = time.perf_counter() - t0
    print(f"\nBest val_brier (mean, α=0 val): {best_val_brier:.5f}")
    print(f"Wall: {wall:.1f}s")
    print(f"Weights → {weights_path(variant)}")


if __name__ == "__main__":
    main()
