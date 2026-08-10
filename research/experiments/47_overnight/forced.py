"""Track A, phase 2 — forced-contract corpus, because the kontra-able units never occur.

THE PROBLEM. Kontra is +EV for the defenders iff P(soloist makes) < 0.5, and on the honest
frontier the only units under that bar are betli (~38% make) and durchmars (~33%). Those
are exactly the units the deployed rule ABSTAINS on — and exactly the units natural
self-play almost never produces. In 6,000 frontier deals (exp44):

    ulti 1269 · piros parti 940 · piros ulti 822 · ... · betli 14 · durchmars 6

Reaching 500 durchmars deals naturally would take ~500,000 deals. So the signal search runs
on a FORCED corpus: hands where the contract is plausible, forced to that contract, played
out with the deployed stack.

THE BIAS, AND HOW IT IS HANDLED. Forcing changes which hands hold the contract, so the
forced corpus may NOT be used for frequency-weighted GP claims — only for "given this
contract on this hand, what predicts the outcome". The natural corpus supplies the real
frequencies, and the gate is the final arbiter either way. The screen is deliberately loose
(a low head-probability floor, not the bidder's FLOOR) so the corpus spans the marginal
hands where a kontra decision is actually difficult; screening at the deployed floor would
only produce hands nobody would kontra.

Output shares exp43's record schema exactly, so features.py and evaluate.py read both
corpora without knowing which is which.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from multiprocessing import get_context

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP43 = os.path.join(os.path.dirname(_HERE), "43_kontra_signals")
if _EXP43 not in sys.path:
    sys.path.insert(0, _EXP43)

OUT = os.path.join(_HERE, "forced.jsonl")
SEED_BASE = 471_000_000
TRUMPS = ("hearts", "acorns", "leaves", "bells")

# unit -> (rung name to force, minimum head probability to bother playing it out)
# The floors are LOW on purpose: we want the marginal hands, not the certainties.
TARGETS = {
    "betli":          ("betli", 0.10),
    "durchmars":      ("duri", 0.05),
    "teritett_betli": ("teritett betli", 0.15),
    "20_100":         ("20-100", 0.05),
    "40_100":         ("40-100", 0.10),
    "ulti":           ("ulti", 0.30),
}

# Card weights for BIASED DEALING, by target unit. Measured on 40 random deals, the best
# head probability over all 66 discards has median 0.004 for betli and 0.000 for durchmars
# and 20-100 — a uniform deal essentially never contains these contracts, so screening a
# uniform deal yields nothing no matter how low the floor goes. The soloist's twelve are
# therefore SAMPLED with a bias toward the shape the contract needs, and the remaining
# twenty are dealt uniformly to the defenders. This biases which hands appear, which is
# why the forced corpus may only answer "given this hand, what predicts the outcome" and
# never "how often does this happen" — the natural corpus owns frequency.

def _weight(unit, card):
    from features import COLORLESS_RANK          # exp43: 10 sits under the jack
    cl = COLORLESS_RANK[card.rank]
    if unit in ("betli", "teritett_betli"):
        return 8.0 if cl <= 2 else (3.0 if cl == 3 else 0.6)      # want 7/8/9
    if unit == "durchmars":
        # Winning all ten tricks with NO trump needs the top of every suit, not just a
        # few honours — a mild bias leaves the head at a 0.016 median (measured), i.e.
        # hopeless hands with no outcome variance to learn from. Concentrate hard.
        return {"ace": 14.0, "10": 8.0, "king": 4.0, "upper": 2.0}.get(card.rank, 0.15)
    if unit == "40_100":
        return 5.0 if card.rank in ("king", "upper") else (3.0 if card.rank in ("ace", "10") else 0.8)
    if unit == "20_100":
        # 20-100 needs a marriage AND 80 of the 90 card points — the points bind harder
        # than the marriage, so weight aces/tens above the king/felso pair.
        return {"ace": 9.0, "10": 7.0, "king": 4.0, "upper": 4.0}.get(card.rank, 0.2)
    if unit == "ulti":
        return 4.0 if card.rank in ("7", "ace", "10") else 1.0
    return 1.0


def _biased_deal(unit, rng):
    """(sol12, def1, def2) with the soloist's twelve drawn toward `unit`'s shape."""
    from ulti.card import DECK
    pool = list(DECK)
    weights = [_weight(unit, c) for c in pool]
    sol12 = []
    for _ in range(12):                       # weighted sampling without replacement
        tot = sum(weights)
        x = rng.random() * tot
        acc = 0.0
        for k, w in enumerate(weights):
            acc += w
            if acc >= x:
                break
        sol12.append(pool.pop(k))
        weights.pop(k)
    rng.shuffle(pool)
    return sol12, pool[:10], pool[10:20]

_PROV = None
_GP = None


def _init():
    global _PROV, _GP
    from ulti.bidding.frontier import frontier_provider
    from ulti.bidding.ladder import GPTable
    _PROV = frontier_provider()
    _GP = GPTable()


def _head_p(unit, probs):
    return {"betli": probs.p_betli, "teritett_betli": probs.p_betli,
            "durchmars": probs.p_duri_colorless, "ulti": probs.p_ulti,
            "40_100": probs.p_reach100_40, "20_100": probs.p_reach100_20}[unit]


def _screen(unit, sol12, floor, target=None):
    """(head_p, hand10, talon, trump) for this unit, or None if nothing clears `floor`.

    With `target=None` this returns the BEST discard — the announce-stage search restricted
    to one unit. But always taking the argmax manufactures easy hands: measured make rates
    came out 90% for betli, 88% for 20-100, 89% for ulti, all far from the 50% breakeven
    where a kontra decision is actually hard. So the corpus builder passes a difficulty
    TARGET drawn uniformly over [floor, 1] and this picks the discard whose head
    probability is closest to it, spreading the corpus across the range that matters."""
    from ulti.bidding.recipe import sol_marriages
    colorless = unit in ("betli", "teritett_betli", "durchmars")
    best = None
    cands = []
    for i, j in itertools.combinations(range(12), 2):
        hand10 = [c for k, c in enumerate(sol12) if k not in (i, j)]
        talon = [sol12[i], sol12[j]]
        for trump in (("hearts",) if colorless else TRUMPS):
            pr = _PROV.base_probs(hand10, trump)
            if unit == "40_100" and not sol_marriages(hand10, trump)[0]:
                continue
            if unit == "20_100" and not sol_marriages(hand10, trump)[1]:
                continue
            p = _head_p(unit, pr)
            cands.append((p, hand10, talon, None if colorless else trump))
            if best is None or p > best[0]:
                best = cands[-1]
    if best is None or best[0] < floor:
        return None
    if target is None:
        return best
    ok = [c for c in cands if c[0] >= floor]
    return min(ok, key=lambda c: abs(c[0] - target)) if ok else None


def _worker(job):
    seed, unit = job
    from datagen import _framing, _play_deployed, _unit_table
    from ulti.bidding.ladder import LADDER
    try:
        rung_name, floor = TARGETS[unit]
        rung = next(r for r in LADDER if r.name == rung_name)
        rng = random.Random(seed * 7919 + (abs(hash(unit)) % 9973))
        sol12, d1, d2 = _biased_deal(unit, rng)
        pick = _screen(unit, list(sol12), floor, target=rng.uniform(floor, 1.0))
        if pick is None:
            return {"seed": seed, "kept": False, "unit": unit}
        p, hand10, talon, trump = pick
        bid, build_c, solve_c, t, restrict, weights = _framing(rung, trump, hand10)
        pos, hist = _play_deployed(bid, build_c, solve_c, t, restrict, weights,
                                   hand10, list(d1), list(d2), talon, trump, seed)
        return {
            "seed": seed, "kept": True, "src": "forced", "forced_unit": unit,
            "screen_p": float(p),
            "winner": 0, "contract": rung.name, "rung_index": rung.index,
            "trump": trump, "colorless": bool(rung.colorless),
            # No auction happened. An empty bid_seq is honest: the auction FEATURES
            # (a_nbids, a_partner_max, ...) are absent for forced rows, and the analysis
            # must never compare an auction-feature rule across the two corpora.
            "bid_seq": [],
            "units": _unit_table(pos, bid),
            "hist": hist,
            "sol": [c.id for c in hand10], "d1": [c.id for c in d1],
            "d2": [c.id for c in d2], "talon": [c.id for c in talon],
        }
    except Exception as e:
        return {"seed": seed, "kept": False, "unit": unit,
                "error": f"{type(e).__name__}: {e}"}


def _seen():
    s = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    s.add((r["seed"], r.get("forced_unit") or r.get("unit")))
                except Exception:
                    pass
    return s


def build(per_unit: int, workers: int = 8, log=print, units=None):
    seen = _seen()
    jobs = []
    for u in (units or TARGETS):
        for i in range(per_unit):
            key = (SEED_BASE + i, u)
            if key not in seen:
                jobs.append(key)
    log(f"[forced] {len(jobs)} jobs ({per_unit}/unit), {workers} workers")
    t0 = time.perf_counter()
    kept = {}
    errs = 0
    with get_context("fork").Pool(workers, initializer=_init) as pool, open(OUT, "a") as o:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4), 1):
            o.write(json.dumps(r) + "\n")
            o.flush()
            if r.get("error"):
                errs += 1
                if errs <= 3:
                    log(f"  ! forced {r['seed']}: {r['error']}")
            elif r.get("kept"):
                kept[r["forced_unit"]] = kept.get(r["forced_unit"], 0) + 1
            if i % 100 == 0:
                el = time.perf_counter() - t0
                log(f"[forced] {i}/{len(jobs)}  {el:.0f}s  "
                    f"eta {(len(jobs)-i)/(i/el)/60:.1f}m  kept={kept} err={errs}")
    log(f"[forced] done: kept {kept}  errors {errs}")
    return kept


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    build(n, workers=int(os.environ.get("WORKERS", "8")))
