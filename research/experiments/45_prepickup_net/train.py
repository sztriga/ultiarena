"""exp45 — fit and judge the pre-pickup model.

The target is `E[EV of the game you announce after picking up | your 10 cards]`. What
matters is not regression error but DECISION quality: how often the model puts a hand on
the correct side of the pass/bid threshold. Three candidates are compared on exactly that:

  raw          blind_ev > threshold            what the cheat-free bidder does today, and
                                               it under-bids: the pickup is worth ~+3 GP
                                               that this quantity does not contain
  offset       blind_ev + c > threshold        the cheap patch — one constant, fitted
  model        f(features) > threshold         the real thing

`oracle` (true y > threshold) is the ceiling: a perfect predictor of the post-pickup
value still misjudges nothing, so the gap between `model` and `oracle` is what remains to
win, and the gap between `raw` and `oracle` is what the bug costs.

Judged out-of-fold throughout — the offset constant is fitted inside each training fold
too, otherwise it reads its own answer off the test set.

Run:  python3 train.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "prepickup.jsonl")
MODEL = os.path.join(_HERE, "prepickup_model.joblib")

OPEN_THRESHOLD = -2.0        # the forehand's pass penalty, per defender


def load():
    xs, ys, cur = [], [], []
    names = None
    with open(DATA) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r:
                continue
            if names is None:
                names = sorted(r["x"])
            xs.append([r["x"][k] for k in names])
            ys.append(r["y"])
            cur.append(r["cur_rung"])
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float64),
            np.asarray(cur, dtype=np.int32), names)


def decision_stats(pred, true, thr):
    """Agreement with the decision a perfect predictor would make."""
    p, t = pred > thr, true > thr
    return {
        "rate": float(p.mean()),
        "acc": float((p == t).mean()),
        # a miss is a hand you should have played and passed; a false alarm is the reverse
        "missed": float((~p & t).mean()),
        "false_alarm": float((p & ~t).mean()),
        # GP actually left on the table: for every hand you got wrong, the difference
        # between what you took and what the right call was worth
        "regret": float(np.where(p == t, 0.0, np.abs(true - thr)).mean()),
    }


def main():
    X, y, cur, names = load()
    print(f"exp45: {len(y)} examples, {X.shape[1]} features")
    print(f"  target: mean {y.mean():+.2f}  sd {y.std():.2f}")
    bi = names.index("blind_ev")
    blind = X[:, bi]
    print(f"  blind_ev: mean {blind.mean():+.2f}   uplift E[y]-E[blind] = "
          f"{(y - blind).mean():+.2f}   corr {np.corrcoef(blind, y)[0, 1]:.3f}\n")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    oof = np.zeros(len(y))
    oof_off = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=45).split(X):
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=45)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        oof_off[te] = X[te, bi] + (y[tr] - X[tr, bi]).mean()   # offset fitted in-fold

    ss = ((y - y.mean()) ** 2).sum()
    print(f"{'candidate':10s} {'MAE':>6s} {'R2':>7s}")
    for lab, p in (("raw", blind), ("offset", oof_off), ("model", oof)):
        mae = np.abs(p - y).mean()
        r2 = 1 - ((y - p) ** 2).sum() / ss
        print(f"{lab:10s} {mae:6.2f} {r2:7.3f}")

    for lab, mask in (("OPENING (threshold -2)", cur < 0), ("OVERCALL", cur >= 0)):
        if not mask.any():
            continue
        print(f"\n{lab}   n={int(mask.sum())}")
        print(f"  {'candidate':10s} {'picks up':>9s} {'accuracy':>9s} "
              f"{'missed':>8s} {'false':>8s} {'regret GP':>10s}")
        for name, p in (("raw", blind), ("offset", oof_off), ("model", oof),
                        ("oracle", y)):
            s = decision_stats(p[mask], y[mask], OPEN_THRESHOLD)
            print(f"  {name:10s} {100*s['rate']:8.1f}% {100*s['acc']:8.1f}% "
                  f"{100*s['missed']:7.1f}% {100*s['false_alarm']:7.1f}% "
                  f"{s['regret']:10.3f}")

    # refit on everything and save
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=45)
    m.fit(X, y)
    try:
        import joblib
        joblib.dump({"model": m, "names": names}, MODEL)
        print(f"\nsaved: {MODEL}")
    except ImportError:
        print("\n(joblib unavailable — model not saved)")


if __name__ == "__main__":
    main()
