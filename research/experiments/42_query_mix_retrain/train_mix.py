"""Train a head on the exp42 mixture; calibrate on the QUERY-distribution holdout.

Division of labour (same as exp17's net+isotonic stack): the class-weighted BCE +
positive oversampling make the net a good RANKER (it must see makes to rank them),
which deliberately distorts its raw probabilities; the isotonic — fitted ONLY on the
query-distribution holdout — restores calibration where the auction actually asks.

Env: HEAD, EPOCHS (30), BATCH (1024), LR (1e-3), VAL (0.1), SEED (7).
Writes candidates/{HEAD}_baseline.pt + candidates/{HEAD}_isotonic.npz.
"""
import os, sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from ulti.bidding.base_head import Head, auc  # noqa: E402

HEAD = os.environ.get("HEAD", "duri_colored")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = int(os.environ.get("BATCH", "1024"))
LR = float(os.environ.get("LR", "1e-3"))
VAL = float(os.environ.get("VAL", "0.1"))
SEED = int(os.environ.get("SEED", "7"))
HERE = Path(__file__).resolve().parent
OUT = HERE / "candidates"
OUT.mkdir(exist_ok=True)


def isotonic_grid(pred, y, nbins=25):
    """exp23-style monotone bin calibration: quantile bins → (mean pred, mean y),
    ys forced non-decreasing. Provider applies it by np.interp."""
    order = np.argsort(pred)
    pred, y = pred[order], y[order]
    edges = np.linspace(0, len(pred), nbins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(pred[a:b].mean()); ys.append(y[a:b].mean())
    return np.array(xs), np.maximum.accumulate(np.array(ys))


def main():
    d = np.load(HERE / f"{HEAD}_mix.npz")
    X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    print(f"=== train {HEAD} on mixture ===  N={len(y)}  base_rate={y.mean():.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    nv = int(len(y) * VAL)
    vi, ti = perm[:nv], perm[nv:]
    Xtr, ytr = torch.from_numpy(X[ti]), torch.from_numpy(y[ti])
    Xva, yva = torch.from_numpy(X[vi]), torch.from_numpy(y[vi])

    torch.manual_seed(SEED)
    model = Head(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_w = min(float((ytr == 0).sum() / max(1, (ytr == 1).sum())), 20.0)
    print(f"  pos_weight={pos_w:.2f}  train={len(ti)}  val={len(vi)}", flush=True)

    n = len(ti)
    best = {"auc": -1.0}
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            p = model(xb).clamp(1e-6, 1 - 1e-6)
            w = torch.where(yb > 0.5, pos_w, 1.0)
            loss = (-(w * (yb * torch.log(p) + (1 - yb) * torch.log(1 - p)))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).numpy()
        a = auc(yva.numpy(), pv)
        if a > best["auc"]:
            best = {"auc": a, "ep": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:>3}  val_auc={a:.4f}", flush=True)
    model.load_state_dict(best["state"])
    print(f"  best ep{best['ep']} val_auc={best['auc']:.4f}", flush=True)

    # ── isotonic on the QUERY-ONLY holdout ──
    c = np.load(HERE / f"{HEAD}_calib_query.npz")
    Xc, yc = c["X"].astype(np.float32), c["y"].astype(np.float32)
    with torch.no_grad():
        pc = model(torch.from_numpy(Xc)).numpy()
    xs, ys = isotonic_grid(pc, yc)
    pcal = np.interp(pc, xs, ys)
    print(f"  calib(query, n={len(yc)}): raw mean={pc.mean():.4f} → iso mean={pcal.mean():.4f} "
          f"| god rate={yc.mean():.4f}", flush=True)
    top = np.argsort(pc)[-max(1, len(pc) // 20):]      # top 5% — the argmax-hunted tail
    print(f"  top-5% tail: raw={pc[top].mean():.3f} iso={np.interp(pc[top], xs, ys).mean():.3f} "
          f"god={yc[top].mean():.3f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1],
                "head": HEAD, "data": f"{HEAD}_mix.npz", "val_auc": best["auc"]},
               OUT / f"{HEAD}_baseline.pt")
    np.savez(OUT / f"{HEAD}_isotonic.npz", x=xs, y=ys)
    print(f"  saved candidates/{HEAD}_baseline.pt + isotonic", flush=True)


if __name__ == "__main__":
    main()
