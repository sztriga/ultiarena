import os
import sys
for p in ("experiments/23_bidding_integration", "experiments/24_bidding_loop", "."):
    sys.path.insert(0, p)
os.environ.setdefault("FLOOR", "0.7")
os.environ.setdefault("PIMC_N", "8")
os.environ.setdefault("KONTRA_NDET", "6")

from provider import NetProvider
from auction import run_auction, net_bid_fn
from scorers import kontra_pimc_outcome, _hand_makeability, resolve_bidset
from kontra import kontra_level_for

prov = NetProvider(calibrate=True)
bf = net_bid_fn(prov)
n = 0
for seed in range(500000000, 500000060):
    r = run_auction(seed, bf)
    if r["winner"] is None:
        continue
    c = r["contract"]
    if "parti" not in c and c != "ulti" and "piros ulti" not in c:
        continue
    bid = resolve_bidset(r["rung"], r["sol"], r["trump"])
    primary = "ulti" if bid.ulti else "parti"
    pd = _hand_makeability(r["sol"], r["def1"], r["def2"], r["trump"], r["talon"], primary, 1, 6, seed + 11)
    ps = _hand_makeability(r["sol"], r["def1"], r["def2"], r["trump"], r["talon"], primary, 0, 6, seed + 23)
    lvl = kontra_level_for(pd, bid, p_sol=ps)
    gp, made = kontra_pimc_outcome(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"], r["talon"], seed=seed)
    print(f"{c:<14} p_def={pd:.2f} p_sol={ps:.2f} kontra_lvl={lvl} made={made} gp={gp:+.1f}")
    n += 1
    if n >= 12:
        break
print(f"\nsmoke ok — {n} deals, no crash")
