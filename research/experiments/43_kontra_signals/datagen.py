"""exp43 — what kontra signals can a defender actually get?

RESEARCH QUESTION: for each contract unit, how well can a DEFENDER predict "does the
soloist make this?" from what they legitimately see at their kontra moment — and what
is that prediction worth in GP against the deployed rule (trump counts)?

The payoff algebra says the bet is the same for every unit: kontra is +EV for the
defenders iff P(soloist makes) < 0.5 (true even for the asymmetric bid-ulti bukott,
where made=+4·2^k and bukott=−(2^k+1)·4 — the halves cancel). So the whole problem is
ESTIMATION, and this module produces the labelled corpus for it.

INFORMATION SETS (milan 2026-08-02) — the kontra comes interleaved with trick 1:
  def1 plays 2nd, then decides  → auction + soloist's LEAD + own hand
  def2 plays 3rd, then decides  → auction + soloist's lead + DEF1's card + own hand
def2 is strictly better informed; def1's card is genuine partner information. Neither
sees the other's hand. Both are recorded here and split at featurisation time.

LABELS ARE REALISTIC, NOT GOD. A defender bets on the game that actually gets played,
so the label is the outcome under the DEPLOYED play stack — exp31 exploit soloist,
exp36 betli-defense net, terített pinning, PIMC, anti-tell mixer. (This is the opposite
of exp40's bidding lesson, where god labels won: there the target was the contract's
true worth, here it is what will actually happen.) `_play_deployed` mirrors
apps.api.ai_play._ai_play_pick decision-for-decision and calls the very same
apps.api.ai_worker.op_ai_pick, so the corpus cannot drift from deployment.

Kontra never changes play — it is a pure payoff multiplier — so each deal is played
ONCE and we cache, per live unit U:
    made_U   : did the soloist make U
    iso_U[L] : U's ISOLATED soloist per-defender GP at kontra level L∈{0,1,2}
Any per-unit kontra vector's GP is then additive over units, exactly (exp27's trick).

This file only PLAYS and stores raw material — hands, auction, full play history,
per-unit outcomes. No features. Featurisation is cheap and gets iterated a hundred
times; play costs ~0.5 s/deal and must never be repeated.

Run:
    WORKERS=8 python3 -m datagen build 20000        # natural (auction-selected)
    WORKERS=8 python3 -m datagen forced 8000        # rare-unit coverage
Resumable: every phase skips seeds already present in its output file.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

# The corpus must describe the DEPLOYED frontier, so install its profile before the
# bidding stack is imported (exactly what apps/api does).
from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAYED = os.path.join(_HERE, "played.jsonl")
FORCED = os.path.join(_HERE, "forced.jsonl")
SEED_BASE = 430_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
UNITS = ("parti", "ulti", "40_100", "20_100", "durchmars", "betli")


def _hb(tag, done, total, t0, extra=""):
    el = time.perf_counter() - t0
    rate = done / el if el > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(f"[{tag}] {done}/{total}  {el:.0f}s  eta {eta/60:.1f}m  ({rate*60:.0f}/min)  {extra}",
          flush=True)


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


# ── play framing: byte-identical to apps.api.auction_flow._setup_play ────────────

def _framing(rung, trump, sol):
    """(bid, build_c, solve_c, restrict, weights) for this contract — the same five
    values the deployed session computes, so the position and solver objective match."""
    from ulti.bidding.scorers import resolve_bidset, _play_weights
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        return bid, "betli", "betli", None, None, None
    if bid.durchmars and rung.colorless and n_trick == 1:
        return bid, "durchmars", "durchmars", None, None, None
    restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
    return bid, "parti", "multi", trump, restrict, _play_weights(bid, sol, trump)


def _play_deployed(bid, build_c, solve_c, t, restrict, weights,
                   sol, d1, d2, talon, trump, seed, play_cfg=None):
    """Play the deal with the DEPLOYED stack and return (final_pos, history).

    `play_cfg` is an optional list of 3 dicts, one per PLAY INDEX (0=soloist), each with
    any of `exploit` / `betli_def` / `mix_equiv` / `pimc_n`. It exists so an ablation can
    seat two play configurations at one table: the deployed values are module globals read
    at import, so without a per-seat override the only possible comparison is
    everyone-vs-everyone, which measures "is the game different", not "is A better than B".
    None → the deployed globals, unchanged.

    Mirrors apps.api.ai_play._ai_play_pick branch-for-branch: exp36 betli-defense net
    for plain-betli defenders, exp31 exploit soloist, terített pinning for defenders
    once trick 1 is settled, PIMC otherwise, then the anti-tell equivalence mixer.
    The solver work goes through ai_worker.op_ai_pick — the same function the serving
    worker pool calls — so there is one implementation of "what card does the AI play".
    """
    from apps.api import ai_worker
    from apps.api.engine import _BETLI_DEF, _EXPLOIT, _MIX_EQUIV, _exp36
    from ulti.card import card_from_id
    from ulti.solvers import determinize as _det, pis
    from ulti.solvers.blocks import equivalent_moves

    pos = pis.build_position(
        hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
        contract=build_c, trump=t, talon=list(talon),
        declare_marriages=(t is not None), marriage_restrict=restrict,
        has_ulti=bool(bid.ulti))          # 7esre tartás — exp27's harness missed this
    voids = _det.Voids()
    is_terit = bool(getattr(bid, "teritett", False))
    hist = []
    ctr = seed * 31337

    def _cfg(p, key, default):
        if not play_cfg:
            return default
        return play_cfg[p].get(key, default)

    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        ctr += 1
        ch = None
        if (_cfg(p, "betli_def", _BETLI_DEF) and p != 0 and solve_c == "betli"
                and not is_terit and _exp36 is not None and _exp36.available()):
            ch = _exp36.betli_defense_pick(pos, p)
        if ch is None:
            if _cfg(p, "exploit", _EXPLOIT) and p == 0 and not is_terit:
                mode = "exploit"
            elif p != 0 and is_terit and len(hist) >= 3:
                # terített reveal: the soloist's hand is face-up once trick 1 and the
                # whole kontra round are settled — in an AI-only deal that is exactly
                # the end of trick 1.
                mode = "pimc_pinned"
            else:
                mode = "pimc"
            cid = ai_worker.op_ai_pick({
                "hands0": [[c.id for c in h] for h in (sol, d1, d2)],
                "talon": [c.id for c in talon],
                "build_c": build_c, "solve_c": solve_c, "trump": t,
                "restrict": restrict, "has_ulti": bool(bid.ulti),
                "weights": weights, "voids": voids.as_dict(),
                "history": list(hist), "mode": mode, "seed": ctr, "bid": bid,
                "pimc_n": _cfg(p, "pimc_n", None),
            })
            ch = card_from_id(cid) if cid is not None else None
        if ch is None:
            ch = random.Random(ctr).choice(pis.legal_actions(pos))
        # anti-tell mixer (value-neutral by construction; kept so the corpus is a
        # faithful sample of deployed play). Not applied to a colorless soloist,
        # whose legal set has already been dominance-culled.
        if _cfg(p, "mix_equiv", _MIX_EQUIV) and not (trump is None and p == 0):
            try:
                block = equivalent_moves(pos, p, ch, colorless=(trump is None), trump=trump)
                if len(block) > 1:
                    ch = random.Random(ctr * 7919 + ch.id).choice(block)
            except Exception:
                pass
        voids.observe(pos, p, ch)
        pis.apply_move(pos, ch)
        hist.append((p, ch.id))
    return pos, hist


def _unit_table(final_pos, bid):
    """Per live unit: made + isolated soloist GP at kontra level 0/1/2 (exp27)."""
    from ulti.scoring.oracle import score as osc
    from ulti.scoring.units import unit_of
    base = osc(final_pos=final_pos, bid=bid)
    live = sorted({unit_of(k) for k in base.components if unit_of(k)}, key=UNITS.index)
    out = {}
    for U in live:
        keys = [k for k in base.components if unit_of(k) == U]
        made = sum(base.components[k] for k in keys) > 0
        iso = []
        for L in (0, 1, 2):
            pv = osc(final_pos=final_pos, bid=bid, kontras={U: L})
            iso.append(sum(v for k, v in pv.components.items() if unit_of(k) == U))
        out[U] = {"made": bool(made), "iso": iso}
    return out


# ── build: the natural corpus (frontier auction picks the contract) ─────────────

_BID_FN = None


def _init():
    global _BID_FN
    from ulti.bidding.frontier import frontier_bid_fn
    _BID_FN = frontier_bid_fn()


def _build_worker(seed):
    # The FULL auction (exp44), not ulti.bidding.auction.run_auction. run_auction ends the
    # moment the forehand passes, so it never lets seats 1 or 2 open — it passed 79% of
    # deals here against the deployed engine's 39%, which would have skewed every contract
    # frequency in this corpus. The deployed flow lets any seat open after a forehand pass.
    import sys, os
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "44_frontier_table")
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from selfplay import _full_auction
    try:
        r = _full_auction(seed, _BID_FN)
        if r is None:
            return {"seed": seed, "kept": False}
        bid, build_c, solve_c, t, restrict, weights = _framing(r["rung"], r["trump"], r["sol"])
        pos, hist = _play_deployed(bid, build_c, solve_c, t, restrict, weights,
                                   r["sol"], r["def1"], r["def2"], r["talon"],
                                   r["trump"], seed)
        return {
            "seed": seed, "kept": True, "src": "auction",
            "winner": r["winner"], "contract": r["rung"].name,
            "rung_index": r["rung"].index, "trump": r["trump"],
            "colorless": bool(r["rung"].colorless),
            # PUBLIC auction record only — (seat, rung name, trump). The announced EV
            # that run_auction also returns is NOT public knowledge and is dropped.
            "bid_seq": [[b[0], b[1], b[2]] for b in r["bid_seq"]],
            "units": _unit_table(pos, bid),
            "hist": hist,
            "sol": [c.id for c in r["sol"]], "d1": [c.id for c in r["def1"]],
            "d2": [c.id for c in r["def2"]], "talon": [c.id for c in r["talon"]],
        }
    except Exception as e:                       # one bad deal must not kill a 3h run
        return {"seed": seed, "kept": False, "error": f"{type(e).__name__}: {e}"}


def build(n_seeds, out_path=PLAYED):
    todo = [s for s in (SEED_BASE + i for i in range(n_seeds)) if s not in _seen(out_path)]
    print(f"build: {len(todo)} new seeds, {WORKERS} workers → {out_path}", flush=True)
    t0 = time.perf_counter()
    kept = 0
    errs = 0
    ucount = {}
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool, open(out_path, "a") as o:
        for i, rec in enumerate(pool.imap_unordered(_build_worker, todo, chunksize=2), 1):
            o.write(json.dumps(rec) + "\n")
            o.flush()
            if rec.get("error"):
                errs += 1
                if errs <= 3:
                    print(f"  ! seed {rec['seed']}: {rec['error']}", flush=True)
            if rec.get("kept"):
                kept += 1
                for U in rec["units"]:
                    ucount[U] = ucount.get(U, 0) + 1
            if i % 25 == 0:
                _hb("build", i, len(todo), t0, f"kept={kept} err={errs} {ucount}")
    print(f"build done: kept {kept}/{len(todo)}  errors {errs}  units {ucount}", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if cmd == "build":
        build(n)
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
