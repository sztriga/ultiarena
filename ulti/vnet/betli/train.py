"""Train the defender value-net on a generated dataset.

Loads an .npz from data_gen.py, splits into train/val, fits with MSE loss.
Reports per-epoch train/val loss and validation AUC (vs. binary outcome label
where +1 → 1, −1 → 0).

Usage:
    PYTHONPATH=. python3 -m vnet.betli.train \\
        --data vnet/betli/data/train_a03.npz \\
        --out  vnet/betli/models/v_a03.pt \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _ensure_npy_cache(npz_path: Path):
    """Stream-decompress an .npz into sibling .npy files (one-time) so we
    can mmap them instead of pulling the whole tensor into RAM. Big
    datasets (10M+ rows) blow past memory otherwise.

    Reads the inner .npy headers, copies the body bytes via shutil
    (zipfile decompresses on the fly, no full-RAM materialisation), and
    leaves cached .npy files next to the .npz for reuse.
    """
    base    = npz_path.parent / npz_path.stem
    x_path  = base.with_suffix(".X.npy")
    y_path  = base.with_suffix(".y.npy")
    if not x_path.exists() or not y_path.exists():
        print(f"  extracting {npz_path.name} → cache .npy (one-time)…", flush=True)
        with zipfile.ZipFile(npz_path) as zf:
            for inner_name, out_path in (("X.npy", x_path), ("y.npy", y_path)):
                with zf.open(inner_name) as fin, open(out_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)
    return x_path, y_path


def _read_meta_only(npz_path: Path):
    """Pull just the `meta` dict from an .npz without loading X/y."""
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if "meta.npy" not in zf.namelist():
                return {}
            with zf.open("meta.npy") as f:
                arr = np.load(f, allow_pickle=True)
        return dict(arr.item())
    except Exception:
        return {}

from .features import FEATURE_DIM
from .net import ValueNet
from .net_nnue import NnueValueNet


class _NpyIndexDataset(Dataset):
    """Wraps a numpy array + index permutation; converts rows to tensors
    lazily so we never materialise the whole dataset in torch memory.
    Cheap enough to use for both train and val on 10M+ rows."""

    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.idx = idx

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        return (
            torch.from_numpy(np.ascontiguousarray(self.X[j])),
            torch.tensor(float(self.y[j]), dtype=torch.float32),
        )


def _auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    pos_s = y_score[y_true_bin == 1]
    neg_s = y_score[y_true_bin == 0]
    if len(pos_s) == 0 or len(neg_s) == 0:
        return float("nan")
    paired = ((pos_s[:, None] > neg_s[None, :]).sum()
              + 0.5 * (pos_s[:, None] == neg_s[None, :]).sum())
    return float(paired / (len(pos_s) * len(neg_s)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--net-type", choices=("value", "nnue"), default="value")
    ap.add_argument("--nnue-embed", type=int, default=256)
    ap.add_argument("--nnue-trunk", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--save-best", action="store_true",
                    help="Keep the checkpoint with the highest val AUC across epochs.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="If >0, randomly subsample the dataset to this many rows "
                         "(cap memory pressure on very large datasets).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Stream-extract the .npz to sibling .npy files (cached) and mmap them
    # so big datasets (10M+ rows × 132 dims = ~8 GB) don't materialise the
    # whole tensor in RAM.
    x_path, y_path = _ensure_npy_cache(Path(args.data))
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    assert X.shape[1] == FEATURE_DIM, (X.shape[1], FEATURE_DIM)
    meta = _read_meta_only(Path(args.data))

    n_total = X.shape[0]
    if args.max_samples and args.max_samples < n_total:
        # Pre-shuffle and cap so subsequent indexing only touches a small slice
        # of the mmap'd file → much lighter page-cache footprint.
        pool = np.random.permutation(n_total)[: args.max_samples]
        pool.sort()    # sequential reads from mmap are dramatically cheaper
        n = args.max_samples
        print(f"data: subsampling {n_total} → {n} examples (--max-samples)")
    else:
        pool = np.arange(n_total, dtype=np.int64)
        n = n_total
    idx = pool[np.random.permutation(n)]
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    yva_bin = (np.asarray(y[val_idx]) > 0).astype(int)
    print(f"data: {n} examples → {len(tr_idx)} train / {len(val_idx)} val")

    if args.net_type == "value":
        net = ValueNet(input_dim=FEATURE_DIM, hidden=tuple(args.hidden),
                       dropout=args.dropout)
        net_desc = f"value  hidden={args.hidden}"
    else:
        net = NnueValueNet(input_dim=FEATURE_DIM,
                          embed_dim=args.nnue_embed,
                          trunk=tuple(args.nnue_trunk),
                          dropout=args.dropout)
        net_desc = f"nnue  embed={args.nnue_embed}  trunk={args.nnue_trunk}"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"net: {net_desc}  dropout={args.dropout}  params={n_params:,}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    train_ds = _NpyIndexDataset(X, y, tr_idx)
    val_ds   = _NpyIndexDataset(X, y, val_idx)
    loader   = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch, 4096), shuffle=False)

    history = []
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        net.train()
        tr_loss = 0.0
        n_tr = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            bs = len(xb)
            tr_loss += loss.item() * bs
            n_tr += bs
        tr_loss /= max(n_tr, 1)

        net.eval()
        with torch.no_grad():
            va_loss_sum = 0.0
            n_va = 0
            preds_all = []
            for xb, yb in val_loader:
                pred = net(xb)
                preds_all.append(pred.numpy())
                va_loss_sum += loss_fn(pred, yb).item() * len(xb)
                n_va += len(xb)
            va_loss = va_loss_sum / max(n_va, 1)
            auc = _auc(yva_bin, np.concatenate(preds_all))

        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_auc": auc})
        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  ep {ep:>3}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
              f"va_auc={auc:.4f}  {'*' if is_best else ' '}  "
              f"wall={time.perf_counter() - t0:.1f}s",
              flush=True)

    if args.save_best and best_state is not None:
        net.load_state_dict(best_state)
        print(f"\nbest val_auc={best_auc:.4f} at epoch {best_epoch} — checkpoint saved.")
    else:
        print(f"\nfinal epoch checkpoint saved (best was {best_auc:.4f} @ ep {best_epoch}).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_payload = {
        "state_dict": net.state_dict(),
        "input_dim": FEATURE_DIM,
        "net_type":  args.net_type,
        "hidden":    tuple(args.hidden),
        "dropout":   args.dropout,
        "history":   history,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "data_meta": meta,
    }
    if args.net_type == "nnue":
        ckpt_payload["nnue_embed"] = args.nnue_embed
        ckpt_payload["nnue_trunk"] = tuple(args.nnue_trunk)
    torch.save(ckpt_payload, out)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"history": history, "args": vars(args)}, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
