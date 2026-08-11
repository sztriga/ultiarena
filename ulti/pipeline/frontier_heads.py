"""End-to-end retraining of the frontier BIDDING heads (docs: module __init__).

One command, all seven heads, one recipe — the best known from five weeks of
experiments, consolidated:

  datagen    query-mixture states (random deal × random keep-10 × random trump —
             where the auction sweep actually queries, exp41's lesson) PLUS
             strong-dealer positives with SMART discards (the current net's best
             keep-10 — exp42's recipe), god-labeled on a process pool.
  train      one Head per contract, class-weighted BCE (imbalance-aware, exp42)
             and SUIT-PERMUTATION AUGMENTATION (exp17's 6×/24× — the step exp42
             forgot), early-stop on validation Brier.
  calibrate  isotonic with FIXED probability edges on a held-out slice (exp42's
             fix for the mute query-only calibrator).
  audit      ulti.eval.head_audit report card on the candidate directory.

The output directory is a drop-in weights_dir for NetProvider. PROMOTION IS NOT
AUTOMATIC: the audit judges each head, but only a gate tournament judges the
system (exp42's two-head candidate looked better and lost). Run the gate before
copying anything into models/ulti/bidding.

Run (full, overnight-scale on 3 workers):
  python -m ulti.pipeline.frontier_heads all --out models/candidates/DATE \
      --n 8000 --n-pos 2500 --workers 3 --log data/logs/pipeline.log
Stages are resumable: existing per-head files are skipped; rerun any stage alone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

HEADS = ("parti", "ulti", "reach100_40", "reach100_20", "duri_colored",
         "betli", "colorless_duri")

# strong-dealer for the positive oversample, per head (ulti.eval.dojo)
_POS_DEALER = {
    "parti": ("deal_parti", 1.2), "ulti": ("deal_ulti_biased", 1.3),
    "reach100_40": ("deal_parti", 1.6), "reach100_20": ("deal_parti", 1.6),
    "duri_colored": ("deal_durchmars_colored", 1.4),
    "betli": ("deal_betli", 2.2), "colorless_duri": ("deal_durchmars_colorless", 2.2),
}

_log_fh = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()


def _head_cfg(head: str) -> dict:
    from ulti.eval.head_audit import HEADS as AUDIT_HEADS
    return AUDIT_HEADS[head]


# ── datagen ─────────────────────────────────────────────────────────────────────

def _sample_query(head: str, cfg, rng, prov):
    """One query-distribution state: random deal, random drop-2, random trump —
    honouring the marriage gate (a gated-off query never reaches the net)."""
    from ulti.bidding.deal import deal_12_10_10
    from ulti.bidding.recipe import sol_marriages
    from ulti.eval.head_audit import _mk_job, TRUMPS
    for _ in range(60):
        sol12, d1, d2 = deal_12_10_10(rng.getrandbits(31))
        drop = rng.sample(range(12), 2)
        keep10 = [c for i, c in enumerate(sol12) if i not in drop]
        discard = [sol12[i] for i in drop]
        trump = rng.choice(TRUMPS) if cfg["colored"] else None
        if cfg["gate"] is not None:
            has40, has20 = sol_marriages(keep10, trump)
            if cfg["gate"] == "40" and not has40:
                continue
            if cfg["gate"] == "20" and not has20:
                continue
        return keep10, trump, _mk_job(cfg, keep10, d1, d2, discard, trump)
    return None


def _sample_positive(head: str, cfg, rng, prov):
    """One strong-dealer state with the SMART discard: deal a biased 12, keep the
    net's best 10 (argmax over the 66 discards — how deployment actually discards)."""
    from ulti.bidding.recipe import sol_marriages
    from ulti.eval import dojo
    from ulti.eval.head_audit import _mk_job
    from ulti.vnet.pickup.features import featurize

    fname, alpha = _POS_DEALER[head]
    d = getattr(dojo, fname)(seed=rng.getrandbits(31), alpha=alpha)
    cards12 = list(d.sol_hand) + list(d.talon)
    trump = d.trump if cfg["colored"] else None
    best = None
    for i, j in itertools.combinations(range(12), 2):
        keep10 = [c for k, c in enumerate(cards12) if k not in (i, j)]
        if cfg["gate"] is not None:
            has40, has20 = sol_marriages(keep10, trump)
            if (cfg["gate"] == "40" and not has40) or (cfg["gate"] == "20" and not has20):
                continue
        p = prov._p(head, featurize(keep10, trump, cfg["colored"]))
        if best is None or p > best[0]:
            best = (p, keep10, [cards12[i], cards12[j]])
    if best is None:
        return None
    _p, keep10, discard = best
    return keep10, trump, _mk_job(cfg, keep10, d.def1_hand, d.def2_hand, discard, trump)


def stage_datagen(out: Path, heads, n: int, n_pos: int, workers: int, seed: int) -> None:
    from ulti.bidding.provider import NetProvider
    from ulti.eval.head_audit import _god_label
    from ulti.vnet.pickup.features import featurize

    (out / "data").mkdir(parents=True, exist_ok=True)
    prov = NetProvider(calibrate=False)          # ranker only — smart discards
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    try:
        for head in heads:
            path = out / "data" / f"{head}.npz"
            if path.exists():
                log(f"datagen {head}: exists, skipped"); continue
            cfg = _head_cfg(head)
            rng = random.Random(seed + hash(head) % 10000)
            t0 = time.time()
            samples = []
            while len(samples) < n:
                s = _sample_query(head, cfg, rng, prov)
                if s:
                    samples.append(s)
            while len(samples) < n + n_pos:
                s = _sample_positive(head, cfg, rng, prov)
                if s:
                    samples.append(s)
            futs = [pool.submit(_god_label, job) for (_k, _t, job) in samples]
            X = np.stack([featurize(k, t, cfg["colored"]) for (k, t, _j) in samples])
            y = np.zeros(len(samples), dtype=np.float32)
            for i, f in enumerate(futs):
                y[i] = f.result()
                if (i + 1) % 500 == 0:
                    done, total = i + 1, len(futs)
                    eta = (time.time() - t0) / done * (total - done)
                    log(f"datagen {head}: {done}/{total} labeled, "
                        f"pos-rate {y[:done].mean():.3f}, ETA {eta/60:.0f}m")
            np.savez(path, X=X, y=y, n_query=n)
            log(f"datagen {head}: DONE n={len(y)} pos={y.mean():.3f} "
                f"({(time.time()-t0)/60:.0f}m)")
    finally:
        pool.shutdown()


# ── train ───────────────────────────────────────────────────────────────────────

def stage_train(out: Path, heads, seed: int) -> None:
    import torch
    from torch import nn
    from ulti.bidding.base_head import Head
    from ulti.vnet.pickup.augment import random_suit_perm, apply_suit_perm

    (out / "weights").mkdir(parents=True, exist_ok=True)
    for head in heads:
        wpath = out / "weights" / f"{head}_baseline.pt"
        if wpath.exists():
            log(f"train {head}: exists, skipped"); continue
        cfg = _head_cfg(head)
        d = np.load(out / "data" / f"{head}.npz")
        X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_val = max(500, len(y) // 10)
        vi, ti = idx[:n_val], idx[n_val:]
        pos_w = float((1 - y[ti].mean()) / max(y[ti].mean(), 1e-3))

        torch.manual_seed(seed)
        model = Head(in_dim=X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        bce = nn.BCELoss(reduction="none")
        Xv, yv = torch.from_numpy(X[vi]), torch.from_numpy(y[vi])
        best = float("inf")
        t0 = time.time()
        for epoch in range(1, 61):
            model.train()
            order = rng.permutation(len(ti))
            for s in range(0, len(ti), 1024):
                bidx = ti[order[s:s + 1024]]
                xb = X[bidx].copy()
                for r in range(len(xb)):          # the augmentation exp42 forgot
                    sigma = random_suit_perm(cfg["colored"], xb[r], rng)
                    xb[r] = apply_suit_perm(xb[r], sigma, cfg["colored"])
                yb = torch.from_numpy(y[bidx])
                pred = model(torch.from_numpy(xb))
                w = torch.where(yb > 0.5, torch.tensor(pos_w), torch.tensor(1.0))
                loss = (bce(pred, yb) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                brier = float(((model(Xv) - yv) ** 2).mean())
            if brier < best:
                best = brier
                torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1],
                            "head": head, "data": "ulti.pipeline.frontier_heads",
                            "base_rate": float(y.mean())}, wpath)
            if epoch % 10 == 0 or epoch == 1:
                log(f"train {head}: epoch {epoch} val-brier {brier:.4f} "
                    f"(best {best:.4f}, {(time.time()-t0)/60:.1f}m)")
        log(f"train {head}: DONE best val-brier {best:.4f}")


# ── calibrate ───────────────────────────────────────────────────────────────────

_EDGES = np.array([0, .01, .02, .04, .07, .12, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0])


def _pava(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Isotonic regression (pool adjacent violators)."""
    y = y.copy(); w = w.copy()
    blocks = [[i] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        ya = np.average(y[a], weights=w[a]); yb = np.average(y[b], weights=w[b])
        if ya <= yb:
            i += 1
        else:
            blocks[i] = a + b
            del blocks[i + 1]
            i = max(0, i - 1)
    out = np.empty_like(y, dtype=np.float64)
    for blk in blocks:
        out[blk] = np.average(y[blk], weights=w[blk])
    return out


def stage_calibrate(out: Path, heads, seed: int) -> None:
    import torch
    from ulti.bidding.base_head import Head

    for head in heads:
        cpath = out / "weights" / f"{head}_isotonic.npz"
        if cpath.exists():
            log(f"calibrate {head}: exists, skipped"); continue
        d = np.load(out / "data" / f"{head}.npz")
        X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
        rng = np.random.default_rng(seed)
        vi = rng.permutation(len(y))[:max(500, len(y) // 10)]   # same holdout size class
        ck = torch.load(out / "weights" / f"{head}_baseline.pt",
                        map_location="cpu", weights_only=False)
        model = Head(in_dim=ck["in_dim"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(X[vi])).numpy()
        yv = y[vi]
        mids, gods, ws = [], [], []
        for lo, hi in zip(_EDGES[:-1], _EDGES[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 8:                     # fixed edges, minimum occupancy
                mids.append(p[m].mean()); gods.append(yv[m].mean()); ws.append(m.sum())
        xs = np.array(mids); ys = _pava(xs, np.array(gods), np.array(ws, dtype=float))
        np.savez(cpath, x=xs, y=ys)
        log(f"calibrate {head}: {len(xs)} bins, "
            f"range {ys.min():.3f}-{ys.max():.3f}")


# ── audit ───────────────────────────────────────────────────────────────────────

def stage_audit(out: Path, heads, workers: int) -> None:
    import shutil
    from ulti.eval.head_audit import audit

    # the provider loads EVERY head — overlay untouched ones from the deployed
    # models so a partial retrain still yields a complete, drop-in weights dir
    deployed = _REPO / "models" / "ulti" / "bidding"
    for h in HEADS:
        for suffix in ("_baseline.pt", "_isotonic.npz"):
            dst = out / "weights" / f"{h}{suffix}"
            src = deployed / f"{h}{suffix}"
            if not dst.exists() and src.exists():
                shutil.copy(src, dst)
    res = audit(n_uniform=400, n_argmax=150, workers=workers,
                weights_dir=str(out / "weights"), calibrate=True,
                only=list(heads), log=log)
    json.dump(res, open(out / "audit.json", "w"), indent=1)
    log(f"audit: written {out / 'audit.json'}")
    for head, r in res.items():
        gap = r["argmax_mean_pred"] - r["argmax_mean_god"]
        verdict = ("HEALTHY" if abs(gap) < 0.10 else
                   "MISCALIBRATED" if abs(gap) < 0.30 else "SICK")
        log(f"  {head:15s} argmax pred {r['argmax_mean_pred']:.3f} "
            f"god {r['argmax_mean_god']:.3f} gap {gap:+.2f}  {verdict}")
    log("REMINDER: the audit judges heads; only a gate tournament judges the "
        "system. Do not promote without one.")


def main():
    global _log_fh
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "datagen", "train", "calibrate", "audit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", default=",".join(HEADS))
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--n-pos", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(a.log, "a")
    out = Path(a.out)
    heads = [h.strip() for h in a.heads.split(",") if h.strip()]
    log(f"pipeline {a.stage}: heads={heads} n={a.n}+{a.n_pos} workers={a.workers} -> {out}")
    if a.stage in ("all", "datagen"):
        stage_datagen(out, heads, a.n, a.n_pos, a.workers, a.seed)
    if a.stage in ("all", "train"):
        stage_train(out, heads, a.seed)
    if a.stage in ("all", "calibrate"):
        stage_calibrate(out, heads, a.seed)
    if a.stage in ("all", "audit"):
        stage_audit(out, heads, a.workers)
    log("pipeline: done")


if __name__ == "__main__":
    main()
