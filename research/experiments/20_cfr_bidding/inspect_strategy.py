"""Show whether the CFR policy actually conditions on the auction history —
i.e. does belief-updating. The decisive view: hold a player's own hand-bucket
FIXED and vary what the opponent bid; if the response distribution shifts, the
policy is reading the bid (the non-uniform talon/opponent prior you wanted).

Usage: STRATEGY=strategy.pkl python inspect_strategy.py
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from common import ACTIONS                       # noqa: E402
from buckets import EDGES, _BASES                # noqa: E402

STRATEGY = Path(__file__).parent / os.environ.get("STRATEGY", "strategy.pkl")
_LVL = ['lo', 'md', 'hi']


def decode(bucket: int):
    """int → per-action bin labels (parti, ulti, betli, duri, ulti_piros)."""
    lv = []
    for base in reversed(_BASES):
        lv.append(bucket % base)
        bucket //= base
    lv = lv[::-1]
    return tuple(_LVL[l] if base > 2 else ('lo', 'hi')[l]
                 for l, base in zip(lv, _BASES))


def fmt_bucket(b):
    d = decode(b)
    return " ".join(f"{a[:4]}={v}" for a, v in zip(ACTIONS, d))


def fmt_dist(dist):
    return " ".join(f"{a}:{p:.2f}" for a, p in
                    sorted(dist.items(), key=lambda x: -x[1]) if p > 0.02)


def main():
    with open(STRATEGY, 'rb') as f:
        S = pickle.load(f)
    avg = S['avg']
    print(f"strategy: {STRATEGY.name}  infosets={len(avg)}  "
          f"iters={S.get('iters')}\n")

    # ── P0 opening policy by bucket ──────────────────────────────────────────
    print("=== P0 opening policy, by hand-bucket ===")
    opens = {}
    for (bucket, av, hist), dist in avg.items():
        if hist == ():
            opens[(bucket, av)] = dist
    for (bucket, av), dist in sorted(opens.items()):
        print(f"  [{fmt_bucket(bucket)}] ulti_av={av[0]:d} piros_av={av[1]:d}"
              f"  →  {fmt_dist(dist)}")

    # ── belief-updating: P1 response vs what P0 opened (hand-bucket fixed) ────
    print("\n=== P1 response after P0 opens X  (rows = P1 bucket, fixed) ===")
    print("    if the row changes across columns, P1 is READING P0's bid.\n")
    opens_of_interest = ['parti', 'ulti', 'ulti_piros']
    resp = defaultdict(dict)   # (bucket,av) -> {open_action: dist}
    for (bucket, av, hist), dist in avg.items():
        if len(hist) == 1 and hist[0][0] == 0 and hist[0][1] in opens_of_interest:
            resp[(bucket, av)][hist[0][1]] = dist
    shown = 0
    for (bucket, av), byopen in sorted(resp.items()):
        if len(byopen) < 2:
            continue
        print(f"  P1 [{fmt_bucket(bucket)}] ulti_av={av[0]:d} piros_av={av[1]:d}")
        for o in opens_of_interest:
            if o in byopen:
                d = byopen[o]
                passp = d.get('pass', 0.0)
                print(f"      P0={o:>10} → pass:{passp:.2f}  "
                      f"| {fmt_dist({k: v for k, v in d.items() if k != 'pass'})}")
        shown += 1
        if shown >= 25:
            break
    if shown == 0:
        print("  (no P1 infosets with ≥2 distinct P0-opens — need more training "
              "coverage of contested auctions)")


if __name__ == "__main__":
    main()
