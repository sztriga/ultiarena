"""Is the value-model's betli/ulti confidence (the bucket bin) actually
predictive of god-makeability? Stratify the leaf god-win rate by bucket bin.
Tells us whether CFR's occasional betli-open is justified signal or pure
overconfidence."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from common import ACTIONS
from buckets import _BASES

d = np.load(Path(__file__).parent / "leaves_200000.npz")
EV, AV, BK = d['ev'], d['avail'], d['bucket']
N = EV.shape[0]

# decode bucket → per-action bin level
def decode(b):
    lv = []
    for base in reversed(_BASES):
        lv.append(b % base); b //= base
    return lv[::-1]

bins = np.array([decode(int(b)) for b in BK.reshape(-1)]).reshape(N, 3, len(ACTIONS))
EVf = EV.reshape(N * 3, len(ACTIONS))
AVf = AV.reshape(N * 3, len(ACTIONS))
binf = bins.reshape(N * 3, len(ACTIONS))

print("god-makeable rate by value-model bucket bin (per action):\n")
for ai, a in enumerate(ACTIONS):
    nb = _BASES[ai]
    print(f"  {a}:")
    for lvl in range(nb):
        m = AVf[:, ai] & (binf[:, ai] == lvl)
        if m.sum() < 50:
            print(f"    bin {lvl}: n={m.sum():>6} (sparse)")
            continue
        made = (EVf[m, ai] > 0).mean()
        # EV/def if you bid this action from this bin (god outcome)
        evbin = EVf[m, ai].mean()
        print(f"    bin {lvl}: n={m.sum():>6}  god-make {made*100:5.1f}%  "
              f"mean god-EV/def {evbin:+.2f}")
