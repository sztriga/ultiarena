"""exp36 — BETLI DEFENSE BENCHMARK: how many DEFENDER-HOLDABLE betlis does the current
frontier (PIMC) defense actually take, vs perfect (god) defense?

The frontier bidder essentially never bids PLAIN betli (it bids terített/other), so we use
realistic BID-WORTHY plain-betli hands (deal_betli, strong soloist = low α) and keep the
DEFENDER-HOLDABLE ones (double-dummy LOST for the soloist). On those, god defense takes 100%;
PIMC defense takes some fraction, and the gap = betlis the soloist STEALS — the headroom a better
defensive policy could recover. Per holdable betli, 4 (soloist, defender) pairs; metric = soloist
make-rate (= stolen). Env: N (deals/α), ALPHAS, WORKERS, PIMC_N.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/31_exploit_play", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ulti.solvers import pis, determinize as _det                         # noqa: E402
from ulti.eval.pimc_matchup import defenders_won, god_says_soloist_wins, pimc_pick  # noqa: E402
from ulti.eval.dojo import deal_betli                                     # noqa: E402

OUT = os.environ.get("OUT") or os.path.join(_HERE, "benchmark.jsonl")
SEED_BASE = 363_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "1.0,1.5").split(",")]
CONTRACT = "betli"


def _god(pos, *_a):
    return pis.solve_best(pos, contract=CONTRACT)[0]


def _pimc(pos, vd, seed):
    return pimc_pick(pos=pos, contract=CONTRACT, n_samples=PIMC_N, seed=seed, voids_dict=vd)


def _play(deal, sol_pick, def_pick, seed):
    pos = pis.build_position(hands=[list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)],
                             soloist=0, leader=0, contract=CONTRACT, trump=None, talon=list(deal.talon))
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        mv = (sol_pick if p == 0 else def_pick)(pos, vd, seed * 131 + mi)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, mv); mi += 1
    return 0 if defenders_won(pos, CONTRACT) else 1


def _worker(job):
    seed, alpha = job
    deal = deal_betli(seed=seed, alpha=alpha)
    pos0 = pis.build_position(hands=[list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)],
                              soloist=0, leader=0, contract=CONTRACT, trump=None, talon=list(deal.talon))
    rec = {"seed": seed, "alpha": alpha, "dd_makeable": int(god_says_soloist_wins(pos0, contract=CONTRACT))}
    if not rec["dd_makeable"]:
        rec["pp"] = _play(deal, _pimc, _pimc, seed)
        rec["pg"] = _play(deal, _pimc, _god, seed)
        rec["gp"] = _play(deal, _god, _pimc, seed)
        rec["gg"] = _play(deal, _god, _god, seed)
    return rec


def build(n):
    seen = set()
    if os.path.exists(OUT):
        seen = {(json.loads(l)["seed"], json.loads(l)["alpha"]) for l in open(OUT)}
    jobs = [(SEED_BASE + i, a) for a in ALPHAS for i in range(n) if (SEED_BASE + i, a) not in seen]
    random.Random(0).shuffle(jobs)
    print(f"exp36 betli-defense benchmark: {len(jobs)} deals ({n}/α, α∈{ALPHAS})", flush=True)
    t0 = time.perf_counter(); done = 0; hold = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS) as pool, open(OUT, "a") as o:
        for rec in pool.imap_unordered(_worker, jobs, chunksize=4):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if not rec["dd_makeable"]:
                hold += 1
            if done % 100 == 0:
                el = time.perf_counter() - t0
                print(f"[bench] {done}/{len(jobs)} {el:.0f}s eta {(len(jobs)-done)/(done/el)/60:.1f}m "
                      f"holdable {hold}", flush=True)
    print(f"done: {hold} holdable", flush=True)


def analyze():
    import collections
    recs = [json.loads(l) for l in open(OUT)]
    hold = [r for r in recs if not r["dd_makeable"] and "pp" in r]
    by_a = collections.Counter(r["alpha"] for r in recs)
    ddl_a = collections.Counter(r["alpha"] for r in hold)
    out = ["# exp36 — BETLI DEFENSE BENCHMARK (frontier PIMC defense vs god)\n",
           f"Realistic bid-worthy plain-betli hands (deal_betli, α∈{ALPHAS}). "
           f"{len(recs)} deals, {len(hold)} DEFENDER-HOLDABLE (double-dummy-lost).\n"]
    if hold:
        def rate(sub, k):
            return 100 * sum(r[k] for r in sub) / len(sub) if sub else 0
        pp, pg, gp, gg = (rate(hold, k) for k in ("pp", "pg", "gp", "gg"))
        out += ["## On DEFENDER-HOLDABLE betlis — soloist STEAL rate (made a dd-lost betli):",
                "| soloist \\ defender | PIMC def | god def |", "|---|---|---|",
                f"| **PIMC soloist** | **{pp:.1f}%** (realistic) | {pg:.1f}% |",
                f"| god soloist | {gp:.1f}% (max) | {gg:.1f}% |", "",
                f"- **The frontier PIMC defense lets the soloist STEAL {pp:.1f}% of the holdable betlis** "
                f"→ it takes only {100-pp:.1f}% of what perfect defense takes (god=100%). **{pp:.0f}pp of "
                f"defensive headroom.**",
                f"- vs a perfect attacker the leak is {gp:.1f}%; sanity pimc/god-def {pg:.1f}%, god/god {gg:.1f}%.",
                "\n### by α (soloist strength — lower α = stronger, more bid-worthy):"]
        for a in sorted(ddl_a):
            sub = [r for r in hold if r["alpha"] == a]
            out.append(f"- α={a}: {ddl_a[a]}/{by_a[a]} holdable ({100*ddl_a[a]/by_a[a]:.0f}%) · "
                       f"PIMC-def steal {rate(sub,'pp'):.0f}% · god-sol-vs-pimc {rate(sub,'gp'):.0f}%")
    txt = "\n".join(out) + "\n"
    open(os.path.join(_HERE, "BENCHMARK.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "2500")))
    elif cmd == "analyze":
        analyze()
    else:
        print(f"unknown cmd {cmd}", flush=True)
