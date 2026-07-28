"""exp28 — DISCARD-decision headroom for the opener pickup.

The deployed pickup chooses (contract, trump, discard) as the argmax over 4 trumps ×
66 discards, scoring each kept 10-hand with the net. The discard actually played is the
RAW net argmax — never audited against an oracle. Here we measure the CEILING: given the
net's chosen contract+trump for an opener's 12-hand, we god-solve ALL 66 discards (using
the known defender hands) and ask: does the net's discard preserve makeability when a
better discard exists? Per contract (headroom concentrates in ulti/100/duri, not parti).

  build : deal 12/10/10 → net opener decision (contract,trump,discard) → god-make of the
          net's discard + god-make of all 66 discards for that contract   → discard.jsonl

This is the god (perfect-info) ceiling on the DISCARD choice only (contract held = net's).
A separate step measures the contract/trump choice, and a cheat-clean PIMC discard.
Env: N (seeds), WORKERS.  Cheat-clean N/A here — this is an oracle ceiling probe.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from multiprocessing import get_context

os.environ.setdefault("FLOOR", "0.7")
os.environ.setdefault("DEBIAS_PCTL", "0.80")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/24_bidding_loop",
           f"{_REPO}/experiments/23_bidding_integration", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DISCARD = os.path.join(_HERE, "discard.jsonl")
SEED_BASE = 640_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))

_BID_FN = None


def _init():
    global _BID_FN
    from net_bidder import make_net_bid_fn
    _BID_FN = make_net_bid_fn()


def _framing(bid, hand10, trump):
    from scorers import _play_weights
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        return "betli", "betli", None, None
    if bid.durchmars and trump is None and n_trick == 1:
        return "durchmars", "durchmars", None, None
    restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
    return "parti", "multi", _play_weights(bid, hand10, trump), restrict


def _primary(bid):
    if bid.betli: return "betli"
    if bid.forty_hundred and bid.ulti: return "ulti+40_100"
    if bid.twenty_hundred and bid.ulti: return "ulti+20_100"
    if bid.forty_hundred: return "40_100"
    if bid.twenty_hundred: return "20_100"
    if bid.ulti: return "ulti"
    if bid.durchmars: return "durchmars"
    return "parti"


def _god_make(keep, d1, d2, disc, trump, build_c, solve_c, weights, restrict):
    from solvers import pis
    from eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights
    if weights is not None:
        set_multi_weights(**weights)
    pos = pis.build_position(hands=[list(keep), list(d1), list(d2)], soloist=0, leader=0,
                             contract=build_c, trump=trump, talon=list(disc),
                             declare_marriages=(trump is not None), marriage_restrict=restrict)
    try:
        return 1 if god_says_soloist_wins(pos, contract=solve_c) else 0
    except Exception:
        return 0


def _worker(seed):
    from _lib import deal_12_10_10
    from scorers import resolve_bidset
    sol12, d1, d2 = deal_12_10_10(seed)
    pick = _BID_FN(list(sol12), None, None)         # opener decision (current=None)
    if pick is None:
        return {"seed": seed, "bid": False}
    _ev, rung, trump, discard, hand10 = pick
    bid = resolve_bidset(rung, hand10, trump)
    build_c, solve_c, weights, restrict = _framing(bid, hand10, trump)
    net_make = _god_make(hand10, d1, d2, discard, trump, build_c, solve_c, weights, restrict)
    cards12 = list(sol12)
    n_make = 0
    for combo in itertools.combinations(range(12), 2):
        keep = [cards12[i] for i in range(12) if i not in combo]
        disc = [cards12[i] for i in combo]
        n_make += _god_make(keep, d1, d2, disc, trump, build_c, solve_c, weights, restrict)
    return {"seed": seed, "bid": True, "contract": rung.name, "primary": _primary(bid),
            "trump": trump, "net_make": net_make, "any_make": 1 if n_make > 0 else 0,
            "n_making": n_make}


def build(n):
    seen = set()
    if os.path.exists(DISCARD):
        seen = {json.loads(l)["seed"] for l in open(DISCARD)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp28 discard-ceiling: {len(seeds)} seeds (66 god-solves each)", flush=True)
    t0 = time.perf_counter(); done = 0; bids = 0; regret = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(DISCARD, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=1):
            o.write(json.dumps(rec) + "\n"); o.flush()
            done += 1
            if rec.get("bid"):
                bids += 1
                if rec["net_make"] == 0 and rec["any_make"] == 1:
                    regret += 1
            if done % 20 == 0:
                el = time.perf_counter() - t0
                eta = (len(seeds) - done) / (done / el) if done else 0
                print(f"[disc] {done}/{len(seeds)} {el:.0f}s eta {eta/60:.1f}m  "
                      f"bids {bids} discard-regret {regret}", flush=True)
    print(f"done: {bids} bids, {regret} discard-fixable losses", flush=True)


def analyze():
    import collections
    recs = [json.loads(l) for l in open(DISCARD) if json.loads(l).get("bid")]
    print(f"\nexp28 discard ceiling — N={len(recs)} opener bids\n")
    by = collections.defaultdict(lambda: {"n": 0, "net": 0, "any": 0, "regret": 0, "nmk": 0})
    for r in recs:
        b = by[r["primary"]]
        b["n"] += 1; b["net"] += r["net_make"]; b["any"] += r["any_make"]
        b["nmk"] += r["n_making"]
        if r["net_make"] == 0 and r["any_make"] == 1:
            b["regret"] += 1
    tot = {"n": 0, "net": 0, "any": 0, "regret": 0}
    print(f"{'contract':<14}{'n':>5}{'net make%':>11}{'best make%':>12}"
          f"{'discard-regret':>16}{'avg #making/66':>16}")
    for k in sorted(by, key=lambda k: -by[k]["n"]):
        b = by[k]; n = b["n"]
        for kk in ("n", "net", "any", "regret"):
            tot[kk] += b[kk]
        print(f"{k:<14}{n:>5}{100*b['net']/n:>10.0f}%{100*b['any']/n:>11.0f}%"
              f"{b['regret']:>10} ({100*b['regret']/n:>3.0f}%){b['nmk']/n:>15.1f}")
    n = tot["n"]
    print(f"\nOVERALL n={n}  net make {100*tot['net']/n:.0f}%  best-possible make "
          f"{100*tot['any']/n:.0f}%  discard-fixable losses {tot['regret']} "
          f"({100*tot['regret']/n:.1f}%)")
    print("(discard-regret = net's discard god-LOSES the contract but some other discard "
          "god-WINS it — the ceiling headroom from a better put-down, contract held fixed.)")


# ── achievable: cheat-clean PIMC discard vs the net's discard (ulti/duri) ──────
PIMC = os.path.join(_HERE, "pimc_discard.jsonl")
_KDET = int(os.environ.get("KDET", "6"))
_DISC_CONTRACTS = {"ulti", "ulti+40_100", "ulti+20_100", "durchmars"}


def _pimc_worker(seed):
    import random
    from _lib import deal_12_10_10
    from scorers import resolve_bidset
    sol12, d1, d2 = deal_12_10_10(seed)
    pick = _BID_FN(list(sol12), None, None)
    if pick is None:
        return {"seed": seed, "skip": True}
    _ev, rung, trump, discard, hand10 = pick
    bid = resolve_bidset(rung, hand10, trump)
    prim = _primary(bid)
    if prim not in _DISC_CONTRACTS:
        return {"seed": seed, "skip": True}
    build_c, solve_c, weights, restrict = _framing(bid, hand10, trump)
    cards12 = list(sol12)
    combos = list(itertools.combinations(range(12), 2))
    disc_ids = {c.id for c in discard}
    net_combo = tuple(sorted(i for i in range(12) if cards12[i].id in disc_ids))
    # 20 unknown cards = the two real defender hands; sample K splits, SHARED across discards
    unknown = list(d1) + list(d2)
    rng = random.Random(seed * 7 + 1)
    worlds = []
    for _ in range(_KDET):
        u = list(unknown); rng.shuffle(u)
        worlds.append((u[:10], u[10:20]))
    # cheat-clean score of each discard = mean god-make over sampled worlds
    best_combo, best_score = None, -1.0
    actual_make = {}
    for combo in combos:
        keep = [cards12[i] for i in range(12) if i not in combo]
        disc = [cards12[i] for i in combo]
        s = sum(_god_make(keep, w1, w2, disc, trump, build_c, solve_c, weights, restrict)
                for (w1, w2) in worlds) / _KDET
        if s > best_score:
            best_score, best_combo = s, combo
        actual_make[combo] = _god_make(keep, d1, d2, disc, trump, build_c, solve_c, weights, restrict)
    return {"seed": seed, "skip": False, "primary": prim, "contract": rung.name,
            "net_make": actual_make[net_combo], "pimc_make": actual_make[best_combo],
            "best_make": max(actual_make.values())}


def pimc(n):
    seen = set()
    if os.path.exists(PIMC):
        seen = {json.loads(l)["seed"] for l in open(PIMC)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp28 PIMC-discard (ulti/duri only, K={_KDET}): scanning {len(seeds)} seeds", flush=True)
    t0 = time.perf_counter(); done = 0; hits = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(PIMC, "a") as o:
        for rec in pool.imap_unordered(_pimc_worker, seeds, chunksize=2):
            done += 1
            if not rec.get("skip"):
                o.write(json.dumps(rec) + "\n"); o.flush(); hits += 1
            if done % 100 == 0:
                el = time.perf_counter() - t0
                print(f"[pimc] scan {done}/{len(seeds)} {el:.0f}s  ulti/duri hands {hits}", flush=True)
    print(f"done: {hits} ulti/duri hands", flush=True)


def pimc_analyze():
    import collections
    recs = [json.loads(l) for l in open(PIMC) if not json.loads(l).get("skip")]
    by = collections.defaultdict(lambda: {"n": 0, "net": 0, "pimc": 0, "best": 0})
    for r in recs:
        b = by[r["primary"]]
        b["n"] += 1; b["net"] += r["net_make"]; b["pimc"] += r["pimc_make"]; b["best"] += r["best_make"]
    print(f"\nexp28 ACHIEVABLE discard (cheat-clean PIMC pick vs net argmax) — N={len(recs)}\n")
    print(f"{'contract':<14}{'n':>5}{'net make%':>11}{'PIMC make%':>12}{'god-best%':>11}")
    tn = tnet = tp = tb = 0
    for k in sorted(by, key=lambda k: -by[k]["n"]):
        b = by[k]; n = b["n"]; tn += n; tnet += b["net"]; tp += b["pimc"]; tb += b["best"]
        print(f"{k:<14}{n:>5}{100*b['net']/n:>10.0f}%{100*b['pimc']/n:>11.0f}%{100*b['best']/n:>10.0f}%")
    if tn:
        print(f"\nOVERALL n={tn}  net {100*tnet/tn:.1f}%  PIMC-discard {100*tp/tn:.1f}%  "
              f"god-best {100*tb/tn:.1f}%  → cheat-clean gain {100*(tp-tnet)/tn:+.1f}pp "
              f"(ceiling {100*(tb-tnet)/tn:+.1f}pp)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "600")))
    elif cmd == "analyze":
        analyze()
    elif cmd == "pimc":
        pimc(int(os.environ.get("N", "2600")))
    elif cmd == "pimc_analyze":
        pimc_analyze()
    else:
        print(f"unknown cmd {cmd}", flush=True)
