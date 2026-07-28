"""exp26 — defender-kontra fix harness.

Kontra is a payoff MULTIPLIER (play unchanged), so we:
  build   : run the champion auction, keep colored-simple (parti/ulti) wins, play
            each ONCE (PIMC both sides, cheat-clean), cache oracle GP at kontra
            level 0/1/2 + made flag                          -> played.jsonl
  pools   : per (deal, viewer in 0/1/2) sample a POOL of worlds from that seat's
            OWN-HAND info set, god-solve each (P(soloist makes)), and tag each
            world with whether the CHAMPION opener would declare >= the observed
            contract with the sampled soloist 12-hand (auction conditioning) ->
            pools.jsonl   (policy-agnostic; the expensive part; resumable)
  policies: derive every candidate defender-kontra policy (N, conditioning, margin)
            INSTANTLY from the pools + played GP, write results.md

All heavy phases are resumable by seed (append-only jsonl, skip seen seeds) and
log a flushed heartbeat (count + elapsed + ETA + live stat). N>=500 for headlines.

Champion config (must match the deployed engine): FLOOR=0.7, DEBIAS_PCTL=0.80,
NetProvider(calibrate=True).  Env: N (seeds), POOL (worlds/viewer), WORKERS.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import get_context

# ── champion config BEFORE any exp import (provider/auction read env at import) ──
os.environ.setdefault("FLOOR", "0.7")
os.environ.setdefault("DEBIAS_PCTL", "0.80")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE,
           f"{_REPO}/experiments/24_bidding_loop",
           f"{_REPO}/experiments/23_bidding_integration",
           _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PLAYED = os.path.join(_HERE, "played.jsonl")
POOLS = os.path.join(_HERE, "pools.jsonl")
RESULTS = os.path.join(_HERE, "results.md")
SEED_BASE = 700_000_000
POOL = int(os.environ.get("POOL", "40"))          # worlds god-solved per (deal, viewer)
WORKERS = int(os.environ.get("WORKERS", "8"))


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _ids(cards):
    return [c.id for c in cards]


def _cards(ids):
    from ulti.card import card_from_id
    return [card_from_id(i) for i in ids]


def _seen_seeds(path):
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["seed"])
                except Exception:
                    continue
    return seen


def _heartbeat(tag, done, total, t0, extra=""):
    el = time.perf_counter() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"[{tag}] {done}/{total}  {el:.0f}s  eta {eta/60:.1f}m  "
          f"({rate*60:.1f}/min)  {extra}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# build: auction + one play-out + GP at kontra level 0/1/2
# ────────────────────────────────────────────────────────────────────────────
_BID_FN = None


def _init_build():
    global _BID_FN
    from net_bidder import make_net_bid_fn
    _BID_FN = make_net_bid_fn()


def _play_score3(rung, trump, sol, d1, d2, talon, seed):
    """Play the hand once with PIMC (both sides), return (made, [spd0,spd1,spd2])
    where spdL = soloist total_per_def GP scored at kontra level L (colored shared)."""
    import random
    from dataclasses import replace
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import pimc_pick
    from trickster._solver_core import set_multi_weights
    from scoring.oracle import score as score_oracle
    from scorers import resolve_bidset, _play_weights, _primary_made

    bid = resolve_bidset(rung, sol, trump)
    pimc_n = int(os.environ.get("PIMC_N", "16"))
    set_multi_weights(**_play_weights(bid, sol, trump))
    pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                             leader=0, contract="parti", trump=trump,
                             talon=list(talon), declare_marriages=True,
                             marriage_restrict=None)
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        ch = pimc_pick(pos=pos, contract="multi", n_samples=pimc_n,
                       seed=seed * 31337 + mi, voids_dict=vd)
        if ch is None:
            ch = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, ch); mi += 1
    spd = []
    made = None
    for lvl in (0, 1, 2):
        pvec = score_oracle(final_pos=pos, bid=replace(bid, kontra_level=lvl))
        spd.append(float(pvec.total_per_def))
        if lvl == 0:
            made = bool(_primary_made(bid, pvec))
    return made, spd


def _build_worker(seed):
    from auction import run_auction
    from scorers import resolve_bidset
    r = run_auction(seed, _BID_FN)
    if r["winner"] is None:
        return {"seed": seed, "kept": False, "why": "pass"}
    trump = r["trump"]
    bid = resolve_bidset(r["rung"], r["sol"], trump)
    colored_simple = (trump is not None and not bid.betli and not bid.durchmars
                      and not (bid.forty_hundred or bid.twenty_hundred or bid.teritett))
    if not colored_simple:
        return {"seed": seed, "kept": False, "why": r["contract"]}
    primary = "ulti" if bid.ulti else "parti"
    made, spd = _play_score3(r["rung"], trump, r["sol"], r["def1"], r["def2"],
                             r["talon"], seed)
    return {"seed": seed, "kept": True, "winner": r["winner"],
            "rung_index": r["rung"].index, "contract": r["contract"],
            "trump": trump, "primary": primary, "made": made, "spd": spd,
            "sol": _ids(r["sol"]), "d1": _ids(r["def1"]), "d2": _ids(r["def2"]),
            "talon": _ids(r["talon"])}


def build(n_seeds):
    seeds = [SEED_BASE + i for i in range(n_seeds) if (SEED_BASE + i) not in _seen_seeds(PLAYED)]
    print(f"build: {len(seeds)} new seeds (target {n_seeds}); POOL n/a here", flush=True)
    t0 = time.perf_counter(); kept = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init_build) as pool, open(PLAYED, "a") as out:
        for i, rec in enumerate(pool.imap_unordered(_build_worker, seeds, chunksize=2), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            if rec.get("kept"):
                kept += 1
            if i % 20 == 0:
                _heartbeat("build", i, len(seeds), t0, f"kept {kept} ({100*kept/i:.0f}%)")
    print(f"build done: {kept} colored-simple deals kept", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# pools: per (deal, viewer) god-solve a pool of worlds + auction-conditioning tag
# ────────────────────────────────────────────────────────────────────────────
def _viewer_pool(sol, d1, d2, talon, trump, primary, viewer, rung_index, n_pool, seed):
    """Sample n_pool worlds from `viewer`'s own-hand info set. Per world record
    (god_win, passes) where passes = the champion opener would declare a rung of
    index >= rung_index with the sampled soloist 12-hand (auction conditioning).
    Cheat-clean: the viewer only ever sees its own hand."""
    import random
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import god_says_soloist_wins
    root = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                              leader=0, contract=primary, trump=trump,
                              talon=list(talon), declare_marriages=(trump is not None))
    iset = _det.build_info_set(root, viewer, primary, voids=None)
    rng = random.Random(seed)
    out = []
    for _ in range(n_pool):
        hands, tal = _det.sample_world(iset, rng)
        spos = (pis.clone_with_hands_and_talon(root, hands, tal)
                if iset.talon_known is None else pis.clone_with_hands(root, hands))
        god_win = 1 if god_says_soloist_wins(spos, contract=primary) else 0
        passes = 1
        if viewer != 0:                        # only defenders condition on the auction
            sol12 = list(hands[0]) + list(tal)  # imagined soloist's bid-time 12
            try:
                pick = _BID_FN(sol12, None, None)
                passes = 1 if (pick is not None and pick[1].index >= rung_index) else 0
            except Exception:
                passes = 1
        out.append((god_win, passes))
    return out


def _pool_worker(rec):
    seed = rec["seed"]
    sol, d1, d2 = _cards(rec["sol"]), _cards(rec["d1"]), _cards(rec["d2"])
    talon = _cards(rec["talon"]); trump = rec["trump"]; primary = rec["primary"]
    ri = rec["rung_index"]
    pools = {}
    for v in (0, 1, 2):
        pools[v] = _viewer_pool(sol, d1, d2, talon, trump, primary, v, ri,
                                POOL, seed + 101 * v + 7)
    return {"seed": seed, "pools": pools}


def pools():
    kept = []
    with open(PLAYED) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kept"):
                kept.append(r)
    seen = _seen_seeds(POOLS)
    todo = [r for r in kept if r["seed"] not in seen]
    print(f"pools: {len(todo)} deals to solve (POOL={POOL}/viewer, {len(kept)} kept total)", flush=True)
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init_build) as pool, open(POOLS, "a") as out:
        for i, rec in enumerate(pool.imap_unordered(_pool_worker, todo, chunksize=1), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            if i % 10 == 0:
                _heartbeat("pools", i, len(todo), t0)
    print("pools done", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# policies: derive every candidate policy instantly from played + pools
# ────────────────────────────────────────────────────────────────────────────
def _p_from_pool(worlds, n_det, conditioned):
    """P(soloist makes) from the first worlds; if conditioned, restrict to worlds
    that pass the auction filter. Falls back to unconditioned if too few pass."""
    sub = worlds[:n_det]
    if conditioned:
        passed = [g for (g, p) in sub if p]
        if len(passed) >= max(3, n_det // 4):
            return sum(passed) / len(passed)
        # too few passing worlds — widen to the full pool's passing set
        passed = [g for (g, p) in worlds if p]
        if passed:
            return sum(passed) / len(passed)
    return sum(g for (g, _p) in sub) / max(1, len(sub))


# Signal configs: (name, n_det, conditioned). Signal per deal = min(p_d1, p_d2)
# (either defender kontra-ing triggers the shared colored kontra).
SIGNALS = [
    ("god_n6",   6,  False),
    ("god_n20",  20, False),
    ("god_n40",  40, False),
    ("cond_n20", 20, True),
    ("cond_n40", 40, True),
]
_TAU_GRID = [i / 100.0 for i in range(0, 101, 2)]   # 0.00 .. 1.00


def _load():
    played = {}
    with open(PLAYED) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kept"):
                played[r["seed"]] = r
    pools_by = {}
    with open(POOLS) as f:
        for line in f:
            r = json.loads(line)
            pools_by[r["seed"]] = {int(k): v for k, v in r["pools"].items()}
    return played, pools_by


def policies():
    from scorers import resolve_bidset
    from kontra import _sol_ev
    played, pools_by = _load()
    seeds = [s for s in played if s in pools_by]
    print(f"policies: {len(seeds)} deals with pools", flush=True)

    # Build a per-deal row: contract, made, spd0, spd_kontra (level with FIXED soloist
    # rekontra rule), and the min-defender signal for each config.
    rows = []
    for s in seeds:
        rec = played[s]; pl = pools_by[s]
        bid = resolve_bidset(_rung(rec["rung_index"]), _cards(rec["sol"]), rec["trump"])
        p_sol = _p_from_pool(pl[0], 40, False)                 # fixed soloist rekontra signal
        rekontra = _sol_ev(p_sol, bid, 0) > 0
        lvl = 2 if rekontra else 1
        row = {"seed": s, "primary": rec["primary"], "made": rec["made"],
               "spd0": rec["spd"][0], "spdK": rec["spd"][lvl], "bid": bid, "sig": {}}
        for nm, n_det, cond in SIGNALS:
            row["sig"][nm] = min(_p_from_pool(pl[1], n_det, cond),
                                 _p_from_pool(pl[2], n_det, cond))
        rows.append(row)

    # train/test split by seed parity — τ is tuned on TRAIN, reported on TEST.
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def _mean_gp(sub, decide):
        g = f = 0.0
        for r in sub:
            kon = decide(r)
            g += r["spdK"] if kon else r["spd0"]
            f += 1 if kon else 0
        n = max(1, len(sub))
        return g / n, 100 * f / n

    def _best_tau(train_sub, nm):          # τ minimising TRAIN soloist GP
        best = None
        for tau in _TAU_GRID:
            gp, _f = _mean_gp(train_sub, lambda r, t=tau: r["sig"][nm] < t)
            if best is None or gp < best[1]:
                best = (tau, gp)
        return best[0]

    lines = [f"# exp26 defender-kontra — N={len(rows)} colored-simple deals "
             f"(train {len(train)} / test {len(test)})\n",
             "Soloist GP/deal, reported on the HELD-OUT test half (LOWER = better "
             "defense). `never`=no kontra. `oracle`=kontra iff the hand was actually "
             "bukott (ceiling). `current`=deployed rule (_sol_ev(sig)<0). "
             "`τ*`=kontra iff signal<τ*, τ* tuned on TRAIN, gain measured on TEST.\n"]
    for grp in ("all", "parti", "ulti"):
        te = test if grp == "all" else [r for r in test if r["primary"] == grp]
        tr = train if grp == "all" else [r for r in train if r["primary"] == grp]
        if not te:
            continue
        never, _ = _mean_gp(te, lambda r: False)
        oracle, _ = _mean_gp(te, lambda r: not r["made"])
        made_rate = 100 * sum(r["made"] for r in te) / len(te)
        lines.append(f"\n## {grp}  (test n={len(te)}, soloist made {made_rate:.0f}%)")
        lines.append(f"- never  : {never:+.3f}")
        lines.append(f"- oracle : {oracle:+.3f}   (defender ceiling, gain {never-oracle:+.3f})")
        lines.append("\n| signal | current GP | current fire% | curr gain | τ* (train) | τ* test GP | τ* gain | τ* fire% |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for nm, _n, _c in SIGNALS:
            cur_gp, cur_fire = _mean_gp(te, lambda r: _sol_ev(r["sig"][nm], r["bid"], 0) < 0)
            tau = _best_tau(tr, nm)
            t_gp, t_fire = _mean_gp(te, lambda r, t=tau: r["sig"][nm] < t)
            lines.append(f"| {nm} | {cur_gp:+.3f} | {cur_fire:.0f}% | {never-cur_gp:+.3f} "
                         f"| {tau:.2f} | {t_gp:+.3f} | {never-t_gp:+.3f} | {t_fire:.0f}% |")

    # calibration diagnostic — the god→real gap that drives the bad kontras.
    lines.append("\n## calibration: god-makeability (god_n40 signal) vs ACTUAL make rate")
    lines.append("If actual make >> signal, god underestimates the soloist → over-kontra.\n")
    lines.append("| contract | signal bin | n | mean signal | actual make% |")
    lines.append("|---|---|---|---|---|")
    for grp in ("parti", "ulti"):
        sub = [r for r in rows if r["primary"] == grp]
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
            hi = lo + 0.2
            b = [r for r in sub if lo <= r["sig"]["god_n40"] < hi or (hi == 1.0 and r["sig"]["god_n40"] == 1.0)]
            if not b:
                continue
            ms = sum(r["sig"]["god_n40"] for r in b) / len(b)
            mk = 100 * sum(r["made"] for r in b) / len(b)
            lines.append(f"| {grp} | [{lo:.1f},{hi:.1f}) | {len(b)} | {ms:.2f} | {mk:.0f}% |")

    txt = "\n".join(lines) + "\n"
    with open(RESULTS, "w") as f:
        f.write(txt)
    print(txt, flush=True)


def _rung(idx):
    from ladder import LADDER
    return LADDER[idx]


def _max_def_trumps(rec):
    from ulti.card import card_from_id
    t = rec["trump"]
    return max(sum(1 for i in rec["d1"] if card_from_id(i).suit == t),
               sum(1 for i in rec["d2"] if card_from_id(i).suit == t))


def analyze_combined():
    """The final head-to-head on the FULL colored-simple mix, out-of-sample:
    deployed kontra rule vs the recommended contract-specific policy
    (parti: keep the makeability rule; ulti: kontra only if a defender holds >=4
    trumps — the only cheat-clean ulti signal that beats never-kontra)."""
    from scorers import resolve_bidset
    from kontra import _sol_ev
    played, pools_by = _load()
    seeds = [s for s in played if s in pools_by]
    rows = []
    for s in seeds:
        rec = played[s]; pl = pools_by[s]
        bid = resolve_bidset(_rung(rec["rung_index"]), _cards(rec["sol"]), rec["trump"])
        lvl = 2 if _sol_ev(_p_from_pool(pl[0], 40, False), bid, 0) > 0 else 1
        rows.append({"seed": s, "primary": rec["primary"], "made": rec["made"],
                     "spd0": rec["spd"][0], "spdK": rec["spd"][lvl], "bid": bid,
                     "p6": min(_p_from_pool(pl[1], 6, False), _p_from_pool(pl[2], 6, False)),
                     "p40": min(_p_from_pool(pl[1], 40, False), _p_from_pool(pl[2], 40, False)),
                     "nt": _max_def_trumps(rec)})
    tr = [r for r in rows if r["seed"] % 2 == 0]
    te = [r for r in rows if r["seed"] % 2 == 1]

    def mgp(sub, dec):
        g = f = 0.0
        for r in sub:
            k = dec(r); g += r["spdK"] if k else r["spd0"]; f += 1 if k else 0
        return g / max(1, len(sub)), 100 * f / max(1, len(sub))

    # tune the parti threshold on TRAIN (p40 signal); ulti uses the 4-trump gate.
    parti_tr = [r for r in tr if r["primary"] == "parti"]
    best = None
    for tau in _TAU_GRID:
        gp, _ = mgp(parti_tr, lambda r, t=tau: r["p40"] < t)
        if best is None or gp < best[1]:
            best = (tau, gp)
    tau_parti = best[0]

    def deployed(r):
        return _sol_ev(r["p6"], r["bid"], 0) < 0                    # current: both contracts
    def recommended(r):
        if r["primary"] == "ulti":
            return r["nt"] >= 4                                     # ulti: 4-trump gate
        return r["p40"] < tau_parti                                 # parti: tuned makeability

    lines = ["# exp26 — FINAL combined policy (full colored-simple mix, held-out test)\n",
             f"parti threshold τ*={tau_parti:.2f} (tuned on train); ulti gate = defender holds >=4 trumps.",
             "Soloist GP/deal, LOWER = better defense. Δ = defender gain vs deployed.\n",
             "| policy | soloist GP/deal | fire% | vs never | vs deployed |",
             "|---|---|---|---|---|"]
    never, _ = mgp(te, lambda r: False)
    dep_gp, dep_f = mgp(te, deployed)
    for name, dec in (("never", lambda r: False),
                      ("deployed", deployed),
                      ("recommended", recommended),
                      ("oracle (ceiling)", lambda r: not r["made"])):
        gp, f = mgp(te, dec)
        lines.append(f"| {name} | {gp:+.3f} | {f:.0f}% | {never-gp:+.3f} | {dep_gp-gp:+.3f} |")
    # split the recommended gain by contract
    for grp in ("parti", "ulti"):
        sub = [r for r in te if r["primary"] == grp]
        nv, _ = mgp(sub, lambda r: False)
        dp, _ = mgp(sub, deployed)
        rc, rf = mgp(sub, recommended)
        lines.append(f"\n- {grp} (test n={len(sub)}): never {nv:+.3f}  deployed {dp:+.3f}  "
                     f"recommended {rc:+.3f} (fire {rf:.0f}%)  → def gain vs deployed {dp-rc:+.3f}")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(_HERE, "results_combined.md"), "w") as f:
        f.write(txt)
    print(txt, flush=True)


# ────────────────────────────────────────────────────────────────────────────
# realpools: REALISTIC-defense makeability for ULTI (the deals god can't read).
# Per sampled world we PIMC-PLAY it out (imperfect defense) and record whether the
# soloist actually made it — the signal god-makeability lacks.
# ────────────────────────────────────────────────────────────────────────────
POOLS_R = os.path.join(_HERE, "poolsR.jsonl")
POOL_R = int(os.environ.get("POOL_R", "20"))


def _viewer_pool_real(rung, sol, d1, d2, talon, trump, primary, viewer, n_pool, seed):
    """Sample n_pool worlds from `viewer`'s own-hand info set; PIMC-play each and
    record (made, passes) — made vs REALISTIC defense (not god). Cheat-clean."""
    import random
    from solvers import pis, determinize as _det
    root = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                              leader=0, contract=primary, trump=trump,
                              talon=list(talon), declare_marriages=(trump is not None))
    iset = _det.build_info_set(root, viewer, primary, voids=None)
    rng = random.Random(seed)
    out = []
    for k in range(n_pool):
        hands, tal = _det.sample_world(iset, rng)
        passes = 1
        try:
            pick = _BID_FN(list(hands[0]) + list(tal), None, None)
            passes = 1 if (pick is not None and pick[1].index >= rung.index) else 0
        except Exception:
            passes = 1
        try:
            made, _spd = _play_score3(rung, trump, hands[0], hands[1], hands[2], tal,
                                      seed * 97 + k)
            out.append((1 if made else 0, passes))
        except Exception:
            continue
    return out


def _realpool_worker(rec):
    seed = rec["seed"]
    rung = _rung(rec["rung_index"])
    sol, d1, d2 = _cards(rec["sol"]), _cards(rec["d1"]), _cards(rec["d2"])
    talon = _cards(rec["talon"]); trump = rec["trump"]; primary = rec["primary"]
    pools = {}
    for v in (1, 2):
        pools[v] = _viewer_pool_real(rung, sol, d1, d2, talon, trump, primary, v,
                                     POOL_R, seed + 313 * v + 5)
    return {"seed": seed, "poolsR": pools}


def realpools(primary_filter="ulti"):
    kept = []
    with open(PLAYED) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kept") and (primary_filter is None or r["primary"] == primary_filter):
                kept.append(r)
    seen = _seen_seeds(POOLS_R)
    todo = [r for r in kept if r["seed"] not in seen]
    print(f"realpools[{primary_filter}]: {len(todo)} deals × 2 viewers × {POOL_R} PIMC "
          f"playouts ({len(kept)} total)", flush=True)
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init_build) as pool, open(POOLS_R, "a") as out:
        for i, rec in enumerate(pool.imap_unordered(_realpool_worker, todo, chunksize=1), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            if i % 10 == 0:
                _heartbeat("realpools", i, len(todo), t0)
    print("realpools done", flush=True)


def analyze_real():
    """Compare god-makeability vs REALISTIC-defense makeability on the ULTI deals,
    out-of-sample (τ tuned on train, gain on test)."""
    from scorers import resolve_bidset
    from kontra import _sol_ev
    played, pools_by = _load()
    realR = {}
    with open(POOLS_R) as f:
        for line in f:
            r = json.loads(line)
            realR[r["seed"]] = {int(k): v for k, v in r["poolsR"].items()}
    seeds = [s for s in played if s in pools_by and s in realR
             and played[s]["primary"] == "ulti"]
    print(f"analyze_real: {len(seeds)} ulti deals with real pools", flush=True)
    rows = []
    for s in seeds:
        rec = played[s]; pl = pools_by[s]; rr = realR[s]
        bid = resolve_bidset(_rung(rec["rung_index"]), _cards(rec["sol"]), rec["trump"])
        p_sol = _p_from_pool(pl[0], 40, False)
        lvl = 2 if _sol_ev(p_sol, bid, 0) > 0 else 1
        p_god = min(_p_from_pool(pl[1], 40, False), _p_from_pool(pl[2], 40, False))
        p_real = min(sum(g for g, _ in rr[1]) / max(1, len(rr[1])),
                     sum(g for g, _ in rr[2]) / max(1, len(rr[2])))
        rows.append({"seed": s, "made": rec["made"], "spd0": rec["spd"][0],
                     "spdK": rec["spd"][lvl], "god": p_god, "real": p_real})
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def _mgp(sub, dec):
        g = f = 0.0
        for r in sub:
            k = dec(r); g += r["spdK"] if k else r["spd0"]; f += 1 if k else 0
        return g / max(1, len(sub)), 100 * f / max(1, len(sub))

    def _best(tr, key):
        b = None
        for tau in _TAU_GRID:
            gp, _ = _mgp(tr, lambda r, t=tau: r[key] < t)
            if b is None or gp < b[1]:
                b = (tau, gp)
        return b[0]

    never, _ = _mgp(test, lambda r: False)
    oracle, _ = _mgp(test, lambda r: not r["made"])
    lines = [f"\n# realistic-makeability on ULTI (test n={len(test)})",
             f"- never  : {never:+.3f}",
             f"- oracle : {oracle:+.3f}   (gain {never-oracle:+.3f})", ""]
    for key in ("god", "real"):
        tau = _best(train, key)
        gp, fire = _mgp(test, lambda r, t=tau, k=key: r[k] < t)
        lines.append(f"- {key:<4} τ*={tau:.2f}: test GP {gp:+.3f}  gain {never-gp:+.3f}  fire {fire:.0f}%")
    # calibration of the realistic signal
    lines.append("\nreal-signal calibration (does it track actual make?):")
    lines.append("| bin | n | mean real-sig | actual make% |")
    lines.append("|---|---|---|---|")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        b = [r for r in rows if lo <= r["real"] < lo + 0.2]
        if b:
            lines.append(f"| [{lo:.1f},{lo+0.2:.1f}) | {len(b)} | "
                         f"{sum(r['real'] for r in b)/len(b):.2f} | "
                         f"{100*sum(r['made'] for r in b)/len(b):.0f}% |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(_HERE, "results_real.md"), "w") as f:
        f.write(txt)
    print(txt, flush=True)


# ────────────────────────────────────────────────────────────────────────────
def smoke(n_seeds=12):
    global POOL
    POOL = 8
    build(n_seeds); pools(); policies()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    N = int(os.environ.get("N", "2500"))
    if cmd == "smoke":
        smoke(int(os.environ.get("N", "12")))
    elif cmd == "build":
        build(N)
    elif cmd == "pools":
        pools()
    elif cmd == "policies":
        policies()
    elif cmd == "realpools":
        realpools(os.environ.get("PRIMARY", "ulti"))
    elif cmd == "analyze_real":
        analyze_real()
    elif cmd == "analyze_combined":
        analyze_combined()
    elif cmd == "all":
        build(N); pools(); policies()
    else:
        print(f"unknown cmd {cmd}", flush=True)
