"""exp29 follow-up — is the KONTRA=1 bidder OVER-PASSING?

On the passz deals (frontier opener passed → forehand pays −4 total), FORCE the forehand
to bid piros parti instead, play it out (PIMC) against the promoted lenient defenders, score.
If the realized soloist GP averages BETTER than −4, the bidder is over-passing (its pass
threshold assumes OPTIMAL god defender kontra, but the deployed defenders kontra leniently
post-exp27) → a fixable leak. Cheat-clean; the forehand's piros-parti discard is net-chosen.
Env: WORKERS.
"""
import json
import os
import sys
import time
from multiprocessing import get_context

os.environ["KONTRA"] = "1"
os.environ.setdefault("FLOOR", "0.7")
os.environ.setdefault("DEBIAS_PCTL", "0.80")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/27_kontra_revamp",
           f"{_REPO}/experiments/24_bidding_loop",
           f"{_REPO}/experiments/23_bidding_integration", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SELF = os.path.join(_HERE, "selfplay.jsonl")
OUT = os.path.join(_HERE, "force_parti.jsonl")
WORKERS = int(os.environ.get("WORKERS", "8"))
PARTI_RUNG = 0                                   # ladder rung 0 = piros parti

_PARTI_FN = None


def _init():
    global _PARTI_FN
    from provider import NetProvider
    from auction import net_bid_fn
    _PARTI_FN = net_bid_fn(NetProvider(calibrate=True), allowed={PARTI_RUNG})


def _worker(seed):
    import harness27 as h
    from frontier_selfplay import _kontra_decision
    from _lib import deal_12_10_10
    from scorers import resolve_bidset, _primary_made
    from scoring.oracle import score as osc
    sol12, d1, d2 = deal_12_10_10(seed)
    pick = _PARTI_FN(list(sol12), None, None)    # forced piros-parti pick (net's best discard)
    if pick is None:
        return {"seed": seed, "ok": False}
    _ev, rung, trump, discard, hand10 = pick
    bid = resolve_bidset(rung, hand10, trump)
    pos, bid = h._play_terminal(rung, trump, hand10, d1, d2, discard, seed)
    kontras, lvl = _kontra_decision(bid, trump, hand10, d1, d2, discard, seed)
    pv = osc(final_pos=pos, bid=bid, kontras=kontras)
    return {"seed": seed, "ok": True, "forced_gp": float(pv.total_sol),
            "made": bool(_primary_made(bid, pv)), "kontra": lvl}


def run():
    passz = [json.loads(l)["seed"] for l in open(SELF) if json.loads(l).get("pass")]
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["seed"] for l in open(OUT)}
    seeds = [s for s in passz if s not in seen]
    print(f"force-parti on {len(seeds)} passz deals (vs the −4 passz baseline)", flush=True)
    t0 = time.perf_counter(); done = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(OUT, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=2):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if done % 50 == 0:
                el = time.perf_counter() - t0
                print(f"[force] {done}/{len(seeds)} {el:.0f}s", flush=True)
    print("done", flush=True)


def analyze():
    recs = [json.loads(l) for l in open(OUT) if json.loads(l).get("ok")]
    n = len(recs)
    gp = sum(r["forced_gp"] for r in recs) / n
    made = 100 * sum(1 for r in recs if r["made"]) / n
    better = 100 * sum(1 for r in recs if r["forced_gp"] > -4) / n
    print(f"\nforce-parti on {n} passz deals (forehand bids piros parti instead of passing):")
    print(f"  mean realized soloist GP = {gp:+.2f}   (passz baseline = −4.00)")
    print(f"  piros parti made {made:.0f}%;  bidding beats the −4 passz on {better:.0f}% of deals")
    verdict = "OVER-PASSING — bidding the floor is better on average" if gp > -4 else \
              "passing is correct — bidding the floor is worse"
    print(f"  → {gp - (-4):+.2f} GP/deal vs passing  →  {verdict}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "analyze":
        analyze()
