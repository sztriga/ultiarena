"""exp44 — frontier self-play table for the CHEAT-FREE bidder (2026-08-02).

The exp29 table, re-measured. It has to be re-measured, because every number in it was
produced by a bidder that decided whether to enter the auction while looking at the
talon — two cards it had not picked up (fixed today in ulti.bidding.auction; guard in
tests/ulti/test_information_boundaries.py). This run is the first honest frontier table.

Faithful to the DEPLOYED engine, and deliberately so — the point is to measure what
ships, not an idealisation:
  * FULL auction — any seat may open after a forehand pass, the forehand buries two junk
    cards when it declines, three consecutive passes end it (mirrors
    apps.api.auction_flow._advance_auction, including an AI holder declining to re-raise)
  * BLIND pickup — the bid/pass decision sees 10 cards; the talon only arrives once the
    seat has committed to announcing a game
  * DEPLOYED play — exp31 exploit soloist, exp36 betli-defense net, terített pinning,
    PIMC, anti-tell mixer (shared with exp43 so the two corpora cannot drift)
  * DEPLOYED per-unit kontra — exp27 rules (ulti trumps>=4, colored duri trumps>=3,
    parti makeability<0.10, 40-100 holding the trump marriage card, else abstain) and the
    post-trick-1 rekontra
  * oracle scoring including silents and the per-unit kontra multipliers

Kontra never changes play — it is a payoff multiplier — so each deal is played once and
the kontra vector is computed afterwards from the recorded state. The defender rules read
the initial 10-card hands, and the rekontra reads the post-trick-1 position, both of which
survive in the transcript.

Run:
    WORKERS=8 python3 selfplay.py run 6000
    python3 selfplay.py table
Resumable: `run` skips seeds already in selfplay.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import get_context

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "43_kontra_signals"))

OUT = os.path.join(_HERE, "selfplay.jsonl")
SEED_BASE = 440_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
PASS_PENALTY = 2.0

# exp27 deployed defender gates (kept in sync with apps/api/kontra_flow.py)
_KONTRA_ULTI_TRUMPS = 4
_KONTRA_DURI_TRUMPS = 3
# parti kontra REMOVED 2026-08-03 (exp47) — kept in sync with apps/api/kontra_flow.py


def _hb(tag, done, total, t0, extra=""):
    el = time.perf_counter() - t0
    rate = done / el if el > 0 else 0.0
    print(f"[{tag}] {done}/{total}  {el:.0f}s  eta {(total-done)/rate/60 if rate else 0:.1f}m  "
          f"({rate*60:.0f}/min)  {extra}", flush=True)


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


# ── the full auction (mirrors apps/api/auction_flow._advance_auction) ───────────

def _weakest_two(cards12, trump):
    """The 2 cards the forehand buries when it declines to open. Keeps 7s — a buried
    trump 7 is the ulti card and would hand it to the next picker-up."""
    def junk(c):
        return (1 if (trump is not None and c.suit == trump) else 0,
                c.points, 1 if c.rank == "7" else 0, c.rank_index)
    o = sorted(cards12, key=junk)
    return o[:2], o[2:]


def _full_auction(seed, bid_fn, open_thresholds=None):
    """`bid_fn` is one callable for all seats, or a list of 3 (per-seat, for head-to-head
    runs where two bidder configs share a table).

    `open_thresholds` (list of 3) overrides the value a seat must beat TO OPEN. The
    deployed engine uses −PASS_PENALTY for every seat, but only the forehand actually
    forfeits it: on a passz the forehand pays −4 while the other two seats COLLECT +2
    (measured, exp44/6000). For seats 1 and 2 the alternative to opening is +2 if the deal
    dies and about −5 if someone else opens — never −2. This hook exists to test whether
    that stand-in costs GP; None keeps the deployed behaviour exactly."""
    from ulti.bidding.deal import deal_12_10_10
    fns = list(bid_fn) if isinstance(bid_fn, (list, tuple)) else [bid_fn] * 3
    opens = list(open_thresholds) if open_thresholds else [-PASS_PENALTY] * 3
    sol12, d1, d2 = deal_12_10_10(seed)
    hands = [list(sol12[:10]), list(d1), list(d2)]
    talon = list(sol12[10:])
    current = None
    history = []
    passes = 0
    turn = 0
    while passes < 3:
        if current is not None and turn == current["pid"]:
            passes += 1                       # an AI holder never re-raises its own bid
            turn = (turn + 1) % 3
            continue
        if current is None:
            pick = fns[turn](hands[turn], talon, None, opens[turn], None)
        else:
            pick = fns[turn](hands[turn], talon, current["rung"], -current["ev"], None)
        if pick is not None:
            ev, rung, trump, discard, hand10 = pick
            hands[turn] = list(hand10)
            talon = list(discard)
            current = {"pid": turn, "rung": rung, "trump": trump, "ev": ev}
            history.append([turn, rung.name, trump])
            passes = 0
        else:
            if current is None and not history:        # only the forehand holds the talon
                disc, keep = _weakest_two(list(hands[turn]) + list(talon), None)
                hands[turn] = keep
                talon = disc
            passes += 1
        turn = (turn + 1) % 3
    if current is None:
        return None
    w = current["pid"]
    return {"winner": w, "rung": current["rung"], "trump": current["trump"],
            "sol": hands[w], "def1": hands[(w + 1) % 3], "def2": hands[(w + 2) % 3],
            "talon": talon, "bid_seq": history}


# ── deployed kontra, computed after the fact ───────────────────────────────────

def _defender_kontras(unit, own, trump, mk):
    """apps/api/kontra_flow._ai_defender_kontras_unit, on plain arguments. `mk` is unused
    since the parti rule was removed; kept in the signature so the two stay call-compatible."""
    if unit == "ulti":
        return sum(1 for c in own if c.suit == trump) >= _KONTRA_ULTI_TRUMPS
    if unit == "durchmars" and trump is not None:
        return sum(1 for c in own if c.suit == trump) >= _KONTRA_DURI_TRUMPS
    if unit == "40_100" and trump is not None:
        return any(c.suit == trump and c.rank in ("king", "upper") for c in own)
    return False                    # 20_100 / betli / colorless durchmars → abstain


def _kontra_vector(bid, trump, hands0, talon, hist, recipe, seed):
    """{unit: level | (l1, l2)} exactly as apps/api/kontra_flow._kontra_dict builds it."""
    from apps.api import ai_worker
    from ulti.bidding.kontra import _sol_ev
    from ulti.scoring.units import kontra_units
    units = kontra_units(bid)
    if not units:
        return {}, 0
    colorless = trump is None
    ids = [[c.id for c in h] for h in hands0]
    tal = [c.id for c in talon]

    def _mk(unit, viewer, salt):
        return ai_worker.op_unit_makeability({
            "hands0": ids, "talon": tal, "trump": trump,
            "unit": unit, "viewer": viewer, "seed": seed + salt})

    kdef = {U: {1: False, 2: False} for U in units}
    for pidx in (1, 2):
        for U in units:
            # colored units are SHARED — once either defender kontras, it is taken
            if not colorless and (kdef[U][1] or kdef[U][2]):
                continue
            if _defender_kontras(U, hands0[pidx], trump,
                                 lambda u=U, p=pidx: _mk(u, p, 100 + p)):
                kdef[U][pidx] = True

    rek = {}
    for U in units:
        if not (kdef[U][1] or kdef[U][2]):
            continue
        job = dict(recipe)
        job.update(unit=U, viewer=0, seed=seed + 200, history=hist[:3])
        p = ai_worker.op_unit_makeability_post1(job)
        rek[U] = _sol_ev(p, bid, 0) > 0

    out = {}
    level = 0
    for U in units:
        def lvl(pidx):
            if not kdef[U][pidx]:
                return 0
            return 2 if rek.get(U) else 1
        l1, l2 = lvl(1), lvl(2)
        if l1 == 0 and l2 == 0:
            continue
        out[U] = (l1, l2) if colorless else max(l1, l2)
        level = max(level, l1, l2)
    return out, level


# ── one deal ───────────────────────────────────────────────────────────────────

_BID_FN = None


def _init():
    global _BID_FN
    from ulti.bidding.frontier import frontier_bid_fn
    _BID_FN = frontier_bid_fn()


def play_and_score(a, seed, play_cfg=None):
    """Play a resolved auction with the deployed stack and score it. Shared with every
    gate (exp45, exp47) so a candidate is always judged on the same pipeline.

    `play_cfg` (list of 3, by PLAY index) lets an ablation seat two play configurations at
    one table — see exp43 `_play_deployed`. None = the deployed globals."""
    from datagen import _framing, _play_deployed              # exp43, shared on purpose
    from ulti.bidding.scorers import _primary_made
    from ulti.scoring.oracle import score as score_oracle
    if True:
        bid, build_c, solve_c, t, restrict, weights = _framing(a["rung"], a["trump"], a["sol"])
        pos, hist = _play_deployed(bid, build_c, solve_c, t, restrict, weights,
                                   a["sol"], a["def1"], a["def2"], a["talon"],
                                   a["trump"], seed, play_cfg=play_cfg)
        hands0 = [a["sol"], a["def1"], a["def2"]]
        recipe = {"hands0": [[c.id for c in h] for h in hands0],
                  "talon": [c.id for c in a["talon"]],
                  "build_c": build_c, "solve_c": solve_c, "trump": t,
                  "restrict": restrict, "has_ulti": bool(bid.ulti), "weights": weights,
                  "voids": None}
        kontras, level = _kontra_vector(bid, a["trump"], hands0, a["talon"],
                                        hist, recipe, seed)
        pvec = score_oracle(final_pos=pos, bid=bid, kontras=kontras)
        sol_gp = float(pvec.total_sol)
        w = a["winner"]
        seat_gp = [0.0, 0.0, 0.0]
        seat_gp[w] = sol_gp
        per_def = float(pvec.total_per_def)
        seat_gp[(w + 1) % 3] = -per_def
        seat_gp[(w + 2) % 3] = -(sol_gp - per_def)
        # Per-UNIT levels, not just the game maximum: a bid ulti also exposes the párti,
        # and "kontra%" must mean the same thing it meant in exp29 — did the CONTRACT
        # itself get doubled — or the column silently counts kontra párti as kontra ulti.
        primary = ("betli" if bid.betli else "durchmars" if bid.durchmars else
                   "ulti" if bid.ulti else "40_100" if bid.forty_hundred else
                   "20_100" if bid.twenty_hundred else "parti")
        kv = kontras.get(primary, 0)
        return {"seed": seed, "contract": a["rung"].name, "trump": a["trump"],
                "winner": w, "sol_gp": sol_gp, "seat_gp": seat_gp,
                "made": bool(_primary_made(bid, pvec)), "kontra_level": level,
                "kontra_primary": int(max(kv) if isinstance(kv, tuple) else kv),
                "kontras": {u: (list(v) if isinstance(v, tuple) else v)
                            for u, v in kontras.items()},
                "n_bids": len(a["bid_seq"]), "bid_seq": a["bid_seq"]}


def _worker(seed):
    try:
        a = _full_auction(seed, _BID_FN)
        if a is None:
            # nobody opened: the forehand pays the pass penalty to each defender
            return {"seed": seed, "contract": "passz",
                    "seat_gp": [-2.0 * 2, 2.0, 2.0], "winner": None}
        return play_and_score(a, seed)
    except Exception as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {e}"}


def run(n):
    todo = [s for s in (SEED_BASE + i for i in range(n)) if s not in _seen(OUT)]
    print(f"exp44 self-play: {len(todo)} seeds, {WORKERS} workers", flush=True)
    t0 = time.perf_counter()
    errs = played = passz = 0
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool, open(OUT, "a") as o:
        for i, r in enumerate(pool.imap_unordered(_worker, todo, chunksize=2), 1):
            o.write(json.dumps(r) + "\n")
            o.flush()
            if r.get("error"):
                errs += 1
                if errs <= 3:
                    print(f"  ! {r['seed']}: {r['error']}", flush=True)
            elif r["contract"] == "passz":
                passz += 1
            else:
                played += 1
            if i % 25 == 0:
                _hb("selfplay", i, len(todo), t0,
                    f"played={played} passz={passz} err={errs}")
    print(f"done: played {played}  passz {passz}  errors {errs}", flush=True)


# ── the table ──────────────────────────────────────────────────────────────────

def table():
    recs = []
    with open(OUT) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error"):
                recs.append(r)
    n = len(recs)
    played = [r for r in recs if r["contract"] != "passz"]
    passz = n - len(played)
    agg = defaultdict(lambda: {"n": 0, "gp": 0.0, "made": 0, "k": 0, "kp": 0, "bids": 0})
    for r in played:
        a = agg[r["contract"]]
        a["n"] += 1
        a["gp"] += r["sol_gp"]
        a["made"] += int(r["made"])
        a["k"] += int(r.get("kontra_primary", 0) > 0)      # the contract itself
        a["kp"] += int(r["kontra_level"] > 0)              # anything in the game
        a["bids"] += r["n_bids"]

    print(f"\n# exp44 frontier self-play (CHEAT-FREE bidder) — {n} deals\n")
    print("| contract | count | freq | avg soloist GP | made% | kontra% | any-k% | avg /def | avg bids |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        k = a["n"]
        print(f"| {c} | {k} | {100*k/n:.1f}% | {a['gp']/k:+.2f} | {100*a['made']/k:.0f}% | "
              f"{100*a['k']/k:.0f}% | {100*a['kp']/k:.0f}% | {a['gp']/k/2:+.2f} | "
              f"{a['bids']/k:.2f} |")
    print(f"| **passz** | {passz} | {100*passz/n:.1f}% | — | — | — | — | — | — |")

    if played:
        tot = sum(r["sol_gp"] for r in played)
        print("\n## Overall")
        print(f"- deals: {n} | played: {len(played)} ({100*len(played)/n:.0f}%) | "
              f"passz: {passz} ({100*passz/n:.0f}%)")
        print(f"- soloist made {100*sum(r['made'] for r in played)/len(played):.0f}% "
              f"of played contracts")
        print(f"- mean soloist GP across played contracts: {tot/len(played):+.2f}")
        print(f"- auction: avg {sum(r['n_bids'] for r in played)/len(played):.2f} "
              f"bids/played-deal; "
              f"{100*sum(1 for r in played if r['n_bids']>1)/len(played):.0f}% contested")

    print("\n## Per-seat (seat 0 = forehand/opener)\n")
    print("| seat | mean GP/deal | won bid (soloist) | GP as soloist | GP as defender |")
    print("|---|---|---|---|---|")
    names = ["P0 forehand", "P1 middle", "P2 rear"]
    for s in range(3):
        tot = sum(r["seat_gp"][s] for r in recs)
        sol = [r for r in played if r["winner"] == s]
        d = [r for r in played if r["winner"] is not None and r["winner"] != s]
        print(f"| {names[s]} | {tot/n:+.3f} | {len(sol)} ({100*len(sol)/n:.0f}%) | "
              f"{sum(r['sol_gp'] for r in sol)/len(sol) if sol else 0:+.2f} | "
              f"{sum(r['seat_gp'][s] for r in d)/len(d) if d else 0:+.2f} |")
    zs = sum(sum(r["seat_gp"]) for r in recs) / max(1, n)
    print(f"\n- zero-sum check: seat GP sums to {zs:+.3f} per deal (should be 0.000)")

    print("\n## Bleeding check\n")
    bad = [(c, a) for c, a in agg.items() if a["gp"] < 0]
    if not bad:
        print("No contract loses GP on average.")
    for c, a in sorted(bad, key=lambda kv: kv[1]["gp"] / kv[1]["n"]):
        tag = "FREQUENT — a real leak" if a["n"] >= 30 else "rare"
        print(f"- **{c}**: {a['gp']/a['n']:+.2f} GP/deal over {a['n']} deals ({tag})")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    elif cmd == "table":
        table()
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
