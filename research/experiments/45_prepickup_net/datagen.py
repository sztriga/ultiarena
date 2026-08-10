"""exp45 — a pre-pickup model: is this hand worth taking the talon up for?

THE PROBLEM (2026-08-02). Ulti's auction has two decisions with two different
information sets, and we were serving them from one stack of models:

    PICKUP    from your own 10 cards      "is this worth committing to?"
    ANNOUNCE  from the real 12            "which game, and what do I bury?"

The bidding heads answer the second. Applying them to the first — by feeding them a
10-card hand as though it were final — is the wrong instrument, and measurably so:
picking up is worth **+2.92 GP on average**, because you get to keep the best 10 of 12.
Ignoring that uplift made the fixed (blind) bidder pass 67% of deals against the leaky
bidder's 32%. The answer is not to peek at the talon again; it is to predict the uplift.

TARGET. For a 10-card hand H and a standing rung R:

    y(H, R) = E_talon[ EV of the game you would announce after picking up ]

computed by actually running the announce-stage search on H+talon for sampled talons.
It includes the COMMITMENT FALLBACK — having picked up you must announce something, so
when the confidence floor leaves nothing biddable the search is repeated with the floor
dropped. Predicting the value of a pickup means predicting what really happens after one,
including the times it goes badly.

Across-talon sd for a fixed hand is ~1.4 GP, so N_TALONS=16 pins y to about ±0.35.

FEATURES are the blind head outputs themselves — the seven probabilities per candidate
trump, the blind best EV, marriage gates, and hand shape. So the model is literally a
calibration of the quantity we were misusing, which is the smallest honest change: it
cannot invent information the blind hand does not contain, and `assert_blind()` proves
the featuriser never touches the talon.

No solver anywhere in this file — every label is net arithmetic, ~0.7 s per example.

Run:  WORKERS=6 python3 datagen.py build 6000
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

from ulti.bidding.auction import DEBIAS_PCTL, _best_pickup, _blind_best   # noqa: E402
from ulti.bidding.ladder import LADDER, GPTable                          # noqa: E402
from ulti.bidding.recipe import sol_marriages                            # noqa: E402
from ulti.card import DECK, RANK_POINTS, SUITS                           # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "prepickup.jsonl")
SEED_BASE = 450_000_000
WORKERS = int(os.environ.get("WORKERS", "6"))
N_TALONS = int(os.environ.get("N_TALONS", "16"))
TRUMPS = ("hearts", "acorns", "leaves", "bells")

# Standing-rung distribution: half the decisions are openings (nothing to beat), the
# rest are overcalls, weighted toward the low rungs the auction actually reaches.
_OVERCALL_RUNGS = [r.index for r in LADDER][:10]


def _hand_shape(hand):
    by_suit = {s: [c for c in hand if c.suit == s] for s in SUITS}
    lens = sorted((len(v) for v in by_suit.values()), reverse=True)
    return {
        "sh_len0": lens[0], "sh_len1": lens[1], "sh_len2": lens[2], "sh_len3": lens[3],
        "sh_ace": sum(1 for c in hand if c.rank == "ace"),
        "sh_ten": sum(1 for c in hand if c.rank == "10"),
        "sh_king": sum(1 for c in hand if c.rank == "king"),
        "sh_upper": sum(1 for c in hand if c.rank == "upper"),
        "sh_marr": sum(1 for v in by_suit.values()
                       if any(c.rank == "king" for c in v)
                       and any(c.rank == "upper" for c in v)),
        "sh_pts": sum(RANK_POINTS[c.rank] for c in hand),
    }


def featurize(hand10, current_index, provider, gp) -> dict:
    """Everything the pickup decision is entitled to know. `hand10` only — there is no
    talon parameter here, by construction (see ulti.bidding.auction._blind_best)."""
    f = {"cur_rung": float(current_index)}
    current = None
    if current_index >= 0:
        current = next((r for r in LADDER if r.index == current_index), None)
    # floor=0.0 on purpose. The confidence FLOOR is an announce-stage guard — it stops
    # you declaring a contract the net is unsure of. Applying it to the pickup decision
    # punishes twice, and a raw dealt 10 is exactly where head probabilities are lowest.
    # (Measured: it only moves the pickup rate 36%→40%, so it is not the main leak — but
    # it is the wrong quantity, and the feature must be a clean EV, not a censored one.)
    blind = _blind_best(hand10, lambda h, t, tal: provider.base_probs(h, t),
                        current, gp, floor=0.0)
    f["blind_none"] = float(blind is None)
    f["blind_ev"] = float(blind[0]) if blind else -15.0
    f["blind_rung"] = float(blind[1].index) if blind else -1.0
    for t in TRUMPS:
        bp = provider.base_probs(hand10, t)
        k = t[:2]
        f[f"p_parti_{k}"] = float(bp.p_parti)
        f[f"p_ulti_{k}"] = float(bp.p_ulti)
        f[f"p_r40_{k}"] = float(bp.p_reach100_40)
        f[f"p_r20_{k}"] = float(bp.p_reach100_20)
        f[f"p_duri_{k}"] = float(bp.p_duri_colored)
        has40, has20 = sol_marriages(hand10, t)
        f[f"has40_{k}"] = float(has40)
        f[f"has20_{k}"] = float(has20)
        f[f"ntr_{k}"] = float(sum(1 for c in hand10 if c.suit == t))
    bp0 = provider.base_probs(hand10, "hearts")
    f["p_betli"] = float(bp0.p_betli)
    f["p_duri_cl"] = float(bp0.p_duri_colorless)
    f.update({k: float(v) for k, v in _hand_shape(hand10).items()})
    return f


def realized_ev(hand10, talon, current, provider, gp) -> float:
    """What the announce stage would actually be worth on these 12 cards.

    Mirrors ulti.bidding.auction.net_bid_fn's second stage exactly, INCLUDING the
    commitment fallback: you picked up, so you announce the best thing available even if
    the confidence floor would rather you had not."""
    pf = lambda h, t, tal: provider.base_probs(h, t)
    twelve = list(hand10) + list(talon)
    r = _best_pickup(twelve, pf, current, gp, pctl=DEBIAS_PCTL)
    if r is None:
        r = _best_pickup(twelve, pf, current, gp, pctl=DEBIAS_PCTL, floor=0.0)
    if r is None:
        # No rung above `current` exists at all (the standing bid is the top of the
        # ladder). Not a bad pickup — an impossible one; clipped rather than left as a
        # −99 sentinel that would dominate both the mean target and the feature scale.
        return -15.0
    return float(r[0])


_PROV = None
_GP = None


def _init():
    global _PROV, _GP
    from ulti.bidding.frontier import frontier_provider
    _PROV = frontier_provider()
    _GP = GPTable()


def _worker(seed):
    try:
        rng = random.Random(seed)
        d = list(DECK)
        rng.shuffle(d)
        hand10, pool = d[:10], d[10:]
        cur_ix = -1 if rng.random() < 0.5 else rng.choice(_OVERCALL_RUNGS)
        current = None if cur_ix < 0 else next(r for r in LADDER if r.index == cur_ix)
        f = featurize(hand10, cur_ix, _PROV, _GP)
        vals = []
        for k in range(N_TALONS):
            vals.append(realized_ev(hand10, pool[2 * k:2 * k + 2], current, _PROV, _GP))
        return {"seed": seed, "x": f, "y": float(np.mean(vals)),
                "y_sd": float(np.std(vals)), "n_talons": N_TALONS,
                "hand": [c.id for c in hand10], "cur_rung": cur_ix}
    except Exception as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {e}"}


def _seen(path):
    s = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    s.add(json.loads(line)["seed"])
                except Exception:
                    pass
    return s


def build(n):
    todo = [s for s in (SEED_BASE + i for i in range(n)) if s not in _seen(OUT)]
    print(f"exp45 pre-pickup datagen: {len(todo)} examples, {WORKERS} workers, "
          f"{N_TALONS} talons each", flush=True)
    t0 = time.perf_counter()
    errs = 0
    ys = []
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool, open(OUT, "a") as o:
        for i, r in enumerate(pool.imap_unordered(_worker, todo, chunksize=4), 1):
            o.write(json.dumps(r) + "\n")
            o.flush()
            if r.get("error"):
                errs += 1
                if errs <= 3:
                    print(f"  ! {r['seed']}: {r['error']}", flush=True)
            else:
                ys.append(r["y"])
            if i % 50 == 0:
                el = time.perf_counter() - t0
                rate = i / el
                print(f"[prepickup] {i}/{len(todo)}  {el:.0f}s  "
                      f"eta {(len(todo)-i)/rate/60:.1f}m  ({rate*60:.0f}/min)  "
                      f"mean y={np.mean(ys):+.2f}  err={errs}", flush=True)
    print(f"done: {len(ys)} examples, {errs} errors, mean y={np.mean(ys):+.2f}", flush=True)


def assert_blind():
    """The featuriser must be a pure function of the 10 cards and the standing rung.
    Vary the other 22 cards; the feature vector must be byte-identical."""
    from ulti.bidding.frontier import frontier_provider
    prov, gp = frontier_provider(), GPTable()
    rng = random.Random(0)
    d = list(DECK)
    rng.shuffle(d)
    hand10 = d[:10]
    base = featurize(hand10, -1, prov, gp)
    for _ in range(5):
        rng.shuffle(d)                       # the other 22 cards move; the hand does not
        assert featurize(hand10, -1, prov, gp) == base, "featuriser is not blind!"
    print("blind-featuriser check: OK (features depend only on the 10 cards)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
    elif cmd == "check":
        assert_blind()
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
