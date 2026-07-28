"""exp37 — WHY do the higher betli rungs (rebetli 10p, terított 20p) bleed?

The bidder scores a terített betli by `(2·p − 1)·gp.betli·4` with p = the net's calibrated `p_betli`
on the argmax-over-66-discards hand. terített REVEALS the soloist's hand → the defenders effectively
have full information → the true make ≈ the double-dummy (god) make. So the bleed per bid is
    bleed ≈ 40 · (god_make − p_used)
and the higher rungs bleed precisely when the net's p_betli OVER-states the true god make. Two suspects:
  (1) the head is miscalibrated-optimistic vs god, and/or
  (2) argmax-over-66-discards SELECTS the discard where the net is most optimistic (selection inflation);
      the DEBIAS 0.85-percentile is meant to counter this at the DECISION — does it?

PART A — `calib`: over uniform deals (deal_12_10_10, the deployed distribution), reproduce the bidder's
betli read: for all 66 discards score the calibrated head, take the argmax (net_p, the played hand) and
the 0.85-percentile (pctl_p, the DEBIAS decision value). Compute the TRUE god make on the argmax hand.
Report the calibration curve + the net_p−god / pctl_p−god gaps, globally and in the high-p region where
terített is actually bid → this is the "why", cheaply.

PART B — `bleed`: faithful self-play auction (deployed baseline config, betli_real OFF). Collect every
betli-FAMILY win (betli / rebetli / terített betli), record the net p used, the god make, and the REALISED
oracle GP → the ground-truth bleed and its magnitude per rung.

Env: N, WORKERS, PIMC_N. Cmd: calib | bleed | analyze.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

os.environ["KONTRA"] = "1"
os.environ.setdefault("FLOOR", "0.80")
os.environ.setdefault("DEBIAS_PCTL", "0.85")
os.environ.setdefault("DURI_TERIT_MULT", "0.3")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/29_frontier_table", f"{_REPO}/experiments/27_kontra_revamp",
           f"{_REPO}/experiments/24_bidding_loop", f"{_REPO}/experiments/23_bidding_integration",
           f"{_REPO}/experiments/14_minigame_bid_eval", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ulti.solvers import pis, determinize as _det                       # noqa: E402
from ulti.eval.pimc_matchup import god_says_soloist_wins                # noqa: E402
from ulti.vnet.pickup import featurize                                  # noqa: E402
from _lib import deal_12_10_10                                     # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "8"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
DEBIAS = float(os.environ.get("DEBIAS_PCTL", "0.85"))
SEED_BASE = 540_000_000
CALIB_OUT = os.path.join(_HERE, "bleed_calib.jsonl")
BLEED_OUT = os.path.join(_HERE, "bleed_selfplay.jsonl")
_COMBOS = list(itertools.combinations(range(12), 2))

_PROV = None
_BID_FN = None
import torch  # noqa: E402


def _init():
    global _PROV, _BID_FN
    from provider import NetProvider
    from auction import net_bid_fn
    _PROV = NetProvider(calibrate=True)
    _BID_FN = net_bid_fn(_PROV, pctl=DEBIAS, floor=0.80, duri_mult=0.3, betli_real=False)


# ── PART A: head calibration vs god on the argmax-discard betli hand ──────────
def _betli_ps(sol12):
    """Calibrated p_betli for all 66 discards of a 12-card hand (batched forward)."""
    feats = np.stack([featurize([sol12[i] for i in range(12) if i not in set(c)], None, False)
                      for c in _COMBOS]).astype(np.float32)
    with torch.no_grad():
        raw = _PROV.heads["betli"](torch.from_numpy(feats)).numpy()
    if "betli" in _PROV.calib:
        xs, ys = _PROV.calib["betli"]
        return np.interp(raw, xs, ys)
    return raw


def _calib_worker(seed):
    sol12, d1, d2 = deal_12_10_10(seed)
    ps = _betli_ps(sol12)
    i = int(ps.argmax())
    net_p = float(ps[i]); pctl_p = float(np.quantile(ps, DEBIAS))
    disc = set(_COMBOS[i])
    hand10 = [sol12[j] for j in range(12) if j not in disc]
    talon = [sol12[j] for j in disc]
    pos = pis.build_position(hands=[hand10, list(d1), list(d2)], soloist=0, leader=0,
                             contract="betli", trump=None, talon=talon)
    god = int(god_says_soloist_wins(pos, contract="betli"))
    return {"net_p": net_p, "pctl_p": pctl_p, "god": god}


def build_calib(n):
    seen = 0
    if os.path.exists(CALIB_OUT):
        seen = sum(1 for _ in open(CALIB_OUT))
    seeds = [SEED_BASE + i for i in range(seen, n)]
    print(f"exp37 betli-bleed CALIB: {len(seeds)} deals (argmax-discard head vs god)", flush=True)
    t0 = time.perf_counter(); done = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(CALIB_OUT, "a") as o:
        for rec in pool.imap_unordered(_calib_worker, seeds, chunksize=32):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if done % 2000 == 0:
                el = time.perf_counter() - t0
                print(f"[calib] {done}/{len(seeds)} {el:.0f}s eta {(len(seeds)-done)/(done/el)/60:.0f}m", flush=True)
    print("calib done", flush=True)


# ── PART B: faithful self-play → realised bleed on betli-family bids ──────────
def _full_auction(seed):
    from frontier_selfplay import _weakest_two
    sol12, d1, d2 = deal_12_10_10(seed)
    hands = [list(sol12[:10]), list(d1), list(d2)]; talon = list(sol12[10:])
    current = None; passes = 0; turn = 0; n_bids = 0
    while passes < 3:
        if current is not None and turn == current["pid"]:
            passes += 1; turn = (turn + 1) % 3; continue
        cards = list(hands[turn]) + list(talon)
        pick = _BID_FN(cards, current["rung"] if current else None, None)
        thresh = -2.0 if current is None else -current["ev"]
        if pick is not None and pick[0] > thresh:
            ev, rung, trump, discard, hand10 = pick
            hands[turn] = hand10; talon = discard
            current = {"pid": turn, "rung": rung, "trump": trump, "ev": ev}; n_bids += 1; passes = 0
        else:
            if current is None and n_bids == 0 and turn == 0:
                disc, hand10 = _weakest_two(list(hands[turn]) + list(talon), None)
                hands[turn] = hand10; talon = disc
            passes += 1
        turn = (turn + 1) % 3
    if current is None:
        return None
    w = current["pid"]
    return {"winner": w, "rung": current["rung"], "contract": current["rung"].name,
            "trump": current["trump"], "sol": hands[w], "def1": hands[(w + 1) % 3],
            "def2": hands[(w + 2) % 3], "talon": talon}


def _bleed_worker(seed):
    r = _full_auction(seed)
    if r is None or "betli" not in r["contract"]:
        return {"seed": seed, "betli": False}
    import harness27 as h
    from frontier_selfplay import _kontra_decision
    from ulti.scoring.oracle import score as osc
    sol, d1, d2, talon = r["sol"], r["def1"], r["def2"], r["talon"]
    # net p_betli the bidder used on the played hand
    xb = featurize(sol, None, False).astype(np.float32)
    with torch.no_grad():
        raw = float(_PROV.heads["betli"](torch.from_numpy(xb).unsqueeze(0))[0])
    net_p = float(np.interp(raw, *_PROV.calib["betli"])) if "betli" in _PROV.calib else raw
    pos0 = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
                              contract="betli", trump=None, talon=list(talon))
    god = int(god_says_soloist_wins(pos0, contract="betli"))
    pos, bid = h._play_terminal(r["rung"], r["trump"], sol, d1, d2, talon, seed)
    kontras, _ = _kontra_decision(bid, r["trump"], sol, d1, d2, talon, seed)
    gp = float(osc(final_pos=pos, bid=bid, kontras=kontras).total_sol)
    return {"seed": seed, "betli": True, "contract": r["contract"], "net_p": net_p, "god": god, "gp": gp}


def build_bleed(n):
    seen = set()
    if os.path.exists(BLEED_OUT):
        seen = {json.loads(l)["seed"] for l in open(BLEED_OUT)}
    seeds = [SEED_BASE + 100_000_000 + i for i in range(n) if SEED_BASE + 100_000_000 + i not in seen]
    print(f"exp37 betli-bleed SELF-PLAY: {len(seeds)} deals (collect betli-family bids + realised GP)", flush=True)
    t0 = time.perf_counter(); done = 0; nb = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(BLEED_OUT, "a") as o:
        for rec in pool.imap_unordered(_bleed_worker, seeds, chunksize=4):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if rec.get("betli"):
                nb += 1
            if done % 500 == 0:
                el = time.perf_counter() - t0
                print(f"[bleed] {done}/{len(seeds)} {el:.0f}s eta {(len(seeds)-done)/(done/el)/60:.0f}m "
                      f"betli-bids {nb}", flush=True)
    print(f"bleed done: {nb} betli-family bids", flush=True)


def analyze():
    out = ["# exp37 — why the higher betli rungs bleed\n"]
    # PART A
    if os.path.exists(CALIB_OUT):
        R = [json.loads(l) for l in open(CALIB_OUT)]
        net = np.array([r["net_p"] for r in R]); pct = np.array([r["pctl_p"] for r in R])
        god = np.array([r["god"] for r in R], dtype=float)
        out += [f"## PART A — head calibration vs god  (n={len(R)} uniform deals, argmax-discard)",
                f"- overall: net_p(argmax) mean {net.mean():.3f} · pctl_p(0.85) mean {pct.mean():.3f} · "
                f"true god-make {god.mean():.3f}",
                f"- **argmax inflation** net_p−god = {net.mean()-god.mean():+.3f} ; "
                f"DEBIAS'd pctl_p−god = {pct.mean()-god.mean():+.3f}",
                "\n### calibration curve (net_p bin → actual god-make rate):",
                "| net_p bin | n | mean net_p | god-make | gap | terit EV/bid = 40·(god−net_p) |",
                "|---|---|---|---|---|---|"]
        for lo in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95):
            hi = {0.0: 0.3, 0.3: 0.5, 0.5: 0.7, 0.7: 0.85, 0.85: 0.95, 0.95: 1.01}[lo]
            m = (net >= lo) & (net < hi)
            if m.sum() == 0:
                continue
            g = god[m].mean(); p = net[m].mean()
            out.append(f"| [{lo:.2f},{hi:.2f}) | {int(m.sum())} | {p:.3f} | {g:.3f} | {p-g:+.3f} | {40*(g-p):+.1f} |")
        # the region where terített actually gets bid (decision pctl high)
        for thr in (0.5, 0.7):
            m = pct >= thr
            if m.sum():
                out.append(f"- where DEBIAS decision pctl_p≥{thr} (terített-bid region): n={int(m.sum())} "
                           f"({100*m.mean():.1f}%), true god-make {god[m].mean():.3f} → "
                           f"terített EV/bid ≈ {40*(god[m].mean()-pct[m].mean()):+.1f}, plain-betli EV ≈ "
                           f"{10*(god[m].mean()-pct[m].mean()):+.1f}")
    # PART B
    if os.path.exists(BLEED_OUT):
        R = [json.loads(l) for l in open(BLEED_OUT)]
        bl = [r for r in R if r.get("betli")]
        out += [f"\n## PART B — realised bleed in faithful self-play  (n={len(R)} deals, {len(bl)} betli-family bids)"]
        if bl:
            import collections
            by = collections.defaultdict(list)
            for r in bl:
                by[r["contract"]].append(r)
            out.append("| rung | n | net_p | god-make | realised GP/bid |")
            out.append("|---|---|---|---|---|")
            for c, sub in sorted(by.items(), key=lambda kv: -len(kv[1])):
                gp = np.mean([r["gp"] for r in sub]); npv = np.mean([r["net_p"] for r in sub])
                gd = np.mean([r["god"] for r in sub])
                out.append(f"| {c} | {len(sub)} | {npv:.3f} | {gd:.3f} | **{gp:+.2f}** |")
            allgp = np.mean([r["gp"] for r in bl])
            out.append(f"- ALL betli-family bids: mean realised GP/bid **{allgp:+.2f}** "
                       f"(net_p {np.mean([r['net_p'] for r in bl]):.3f} vs god-make {np.mean([r['god'] for r in bl]):.3f})")
    txt = "\n".join(out) + "\n"
    open(os.path.join(_HERE, "BLEED.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calib"
    if cmd == "calib":
        build_calib(int(os.environ.get("N", "20000")))
    elif cmd == "bleed":
        build_bleed(int(os.environ.get("N", "20000")))
    elif cmd == "analyze":
        analyze()
    else:
        print(f"unknown cmd {cmd}", flush=True)
