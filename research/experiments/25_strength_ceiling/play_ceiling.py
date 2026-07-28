"""Phase 1c — the PLAY-quality ceiling.

Fix the bidder (net agent). Play each won contract move-by-move under mixed
policies and score with the oracle:
  (pimc-sol, pimc-def)  = fully realistic outcome (what the agent actually gets)
  (god-sol,  pimc-def)  = perfect soloist play vs realistic defense
  (god-sol,  god-def)   = both perfect (the god_outcome ceiling)

  soloist-play headroom = (god-sol,pimc-def) − (pimc-sol,pimc-def)
      → how much a better PLAY policy could buy (god play is a CEILING, not
        deployable — it sees all hands; the agent stays PIMC).
  defender-strength     = (god-sol,pimc-def) − (god-sol,god-def)
      → how much the soloist loses when defenders are perfect vs realistic.

Env: N, PIMC_N, WORKERS.
"""
import os
import sys
import time
from multiprocessing import get_context

_HERE = os.path.dirname(os.path.abspath(__file__))
_E24 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/24_bidding_loop"
_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_HERE, _E24, _E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

N = int(os.environ.get("N", "120"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
WORKERS = int(os.environ.get("WORKERS", "8"))

_BID = None


def _init():
    global _BID
    from provider import NetProvider
    from auction import net_bid_fn
    _BID = net_bid_fn(NetProvider(calibrate=True))


def mixed_outcome(rung, trump, sol, d1, d2, talon, sol_god, def_god, seed):
    import random
    from ulti.solvers import pis, determinize as _det
    from ulti.eval.pimc_matchup import pimc_pick, god_pick
    from ulti.scoring.oracle import score as score_oracle
    from trickster._solver_core import set_multi_weights
    from scorers import resolve_bidset, _play_weights, _primary_made

    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        contract, t, restrict = "betli", None, None
    elif bid.durchmars and rung.colorless and n_trick == 1:
        contract, t, restrict = "durchmars", None, None
    else:
        contract, t = "multi", trump
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        set_multi_weights(**_play_weights(bid, sol, trump))
    bkw = dict(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
               contract=("parti" if contract == "multi" else contract), trump=t,
               talon=list(talon), declare_marriages=(t is not None),
               marriage_restrict=restrict)
    pos = pis.build_position(**bkw)
    voids = _det.Voids()
    vd = voids.as_dict()
    rng = random.Random(seed)
    mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        use_god = sol_god if p == 0 else def_god
        if use_god:
            ch = god_pick(pos=pos, contract=contract)
        else:
            ch = pimc_pick(pos=pos, contract=contract, n_samples=PIMC_N,
                           seed=seed * 31337 + mi, voids_dict=vd)
        if ch is None:
            ch = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch)
        vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, ch)
        mi += 1
    pvec = score_oracle(final_pos=pos, bid=bid)
    return pvec.total_per_def, _primary_made(bid, pvec)


def worker(seed):
    from auction import run_auction
    r = run_auction(seed, _BID)
    if r["winner"] is None:
        return None
    a = mixed_outcome(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                      r["talon"], False, False, seed)[0]   # pimc-sol, pimc-def
    b = mixed_outcome(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                      r["talon"], True, False, seed)[0]    # god-sol,  pimc-def
    c = mixed_outcome(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                      r["talon"], True, True, seed)[0]     # god-sol,  god-def
    return (a, b, c)


def main():
    print(f"=== PLAY CEILING | N={N} PIMC_N={PIMC_N} ===", flush=True)
    seeds = [500_000_000 + i for i in range(N)]
    rows = []
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(worker, seeds, chunksize=2):
            if r is not None:
                rows.append(r)
            if len(rows) % 25 == 0 and rows:
                print(f"  {len(rows)}  {time.perf_counter()-t0:.0f}s", flush=True)
    n = len(rows)
    pp = sum(x[0] for x in rows) / n
    gp_ = sum(x[1] for x in rows) / n
    gg = sum(x[2] for x in rows) / n
    print(f"\n=== results | played {n} | wall={time.perf_counter()-t0:.0f}s ===")
    print(f"  (pimc-sol, pimc-def) realistic     {pp:+.3f} GP/def")
    print(f"  (god-sol,  pimc-def) perfect play  {gp_:+.3f} GP/def")
    print(f"  (god-sol,  god-def)  both perfect  {gg:+.3f} GP/def")
    print(f"\n  SOLOIST-PLAY headroom (pimc→god sol) {gp_-pp:+.3f} GP/def")
    print(f"  defender-strength (pimc→god def)     {gp_-gg:+.3f} GP/def")


if __name__ == "__main__":
    main()
