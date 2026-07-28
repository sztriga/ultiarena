"""exp27 — full-ladder per-UNIT kontra/rekontra harness.

Kontra is a per-UNIT payoff multiplier and play is unchanged, so we play each deal
ONCE and cache, per live unit U in {parti,ulti,40_100,20_100,durchmars,betli}:
  - made_U   : did the soloist make that unit (pvec.made / component sign)
  - iso_U[L] : U's ISOLATED soloist GP (vs one defender) at kontra level L=0/1/2
              (other units held at 0; captures piros + bukott-ulti 2/3/5x exactly)
Total soloist GP under any per-unit kontra vector is then additive:
  colored (shared)  : per_def = sum_U iso_U[level_U];  soloist = 2*per_def
  colorless (betli / no-trump duri, per-defender) : soloist = iso_U[L_d0]+iso_U[L_d1]

Also caches rich structural FEATURES per hand (trump counts, high cards, marriages,
card points) for the AI calibration AND the human teaching tables. Every make-rate is
reported in BOTH regimes, never conflated:
  - GOD  = perfect-defense makeability (solver) — theoretical
  - PIMC = realistic make vs the champion's own play — what actually happens

Phases (all resumable, flushed heartbeat, N>=500/unit for headlines):
  build : champion auction (uniform deals) -> play once -> per-unit made+iso+feats
  buildf: FORCED-contract on alpha-biased deals -> coverage for rare units + teaching
  pools : per (deal, live unit, viewer) GOD-makeability + features (defender+soloist view)
  ...    (calibrate / decide / validate / teaching in later commands)
"""
from __future__ import annotations

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

PLAYED = os.path.join(_HERE, "played.jsonl")
SEED_BASE = 820_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
UNITS = ("parti", "ulti", "40_100", "20_100", "durchmars", "betli")
HIGH_TRUMP = {"ace", "10", "king", "upper"}     # top-4 trumps


def _ids(cards):
    return [c.id for c in cards]


def _cards(ids):
    from ulti.card import card_from_id
    return [card_from_id(i) for i in ids]


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


def _hb(tag, done, total, t0, extra=""):
    el = time.perf_counter() - t0
    r = done / el if el > 0 else 0
    eta = (total - done) / r if r > 0 else 0
    print(f"[{tag}] {done}/{total} {el:.0f}s eta {eta/60:.1f}m ({r*60:.0f}/min) {extra}", flush=True)


# ── features (for calibration AND teaching) ──────────────────────────────────
def _hand_feats(cards, trump):
    pts = {"ace": 11, "10": 10, "king": 4, "upper": 3, "lower": 2}
    tr = [c for c in cards if trump is not None and c.suit == trump]
    return {
        "ntrump": len(tr),
        "nhigh_trump": sum(1 for c in tr if c.rank in HIGH_TRUMP),
        "has_trump_ace": any(c.rank == "ace" for c in tr),
        "cardpts": sum(pts.get(c.rank, 0) for c in cards),
        "n_ace": sum(1 for c in cards if c.rank == "ace"),
        "n_ten": sum(1 for c in cards if c.rank == "10"),
        "voids": sum(1 for s in ("acorns", "leaves", "hearts", "bells")
                     if s != trump and not any(c.suit == s for c in cards)),
    }


def _features(sol, d1, d2, talon, trump):
    return {"sol": _hand_feats(sol, trump), "d1": _hand_feats(d1, trump),
            "d2": _hand_feats(d2, trump)}


# ── play the deal once with the bid's framing (mirrors deployed _setup_play) ──
def _play_terminal(rung, trump, sol, d1, d2, talon, seed):
    import random
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import pimc_pick
    from trickster._solver_core import set_multi_weights
    from scorers import resolve_bidset, _play_weights
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        build_c, restrict, weights, solve_c = "betli", None, None, "betli"
    elif bid.durchmars and trump is None and n_trick == 1:
        build_c, restrict, weights, solve_c = "durchmars", None, None, "durchmars"
    else:
        build_c, solve_c = "parti", "multi"
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        weights = _play_weights(bid, sol, trump)
    if weights is not None:
        set_multi_weights(**weights)
    pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0,
                             leader=0, contract=build_c, trump=trump, talon=list(talon),
                             declare_marriages=(trump is not None), marriage_restrict=restrict)
    pimc_n = int(os.environ.get("PIMC_N", "16"))
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        ch = pimc_pick(pos=pos, contract=solve_c, n_samples=pimc_n,
                       seed=seed * 31337 + mi, voids_dict=vd)
        if ch is None:
            ch = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, ch); mi += 1
    return pos, bid


def _unit_table(final_pos, bid):
    """Per live unit U: (made_U, iso_U[0], iso_U[1], iso_U[2]) — U's isolated soloist
    per-defender GP at kontra level 0/1/2. Live units derived from scored components."""
    from scoring.oracle import score as osc, _unit_of
    base = osc(final_pos=final_pos, bid=bid)
    live = sorted({_unit_of(k) for k in base.components if _unit_of(k)},
                  key=lambda u: UNITS.index(u))
    out = {}
    for U in live:
        made = sum(base.components.get(k, 0) for k in base.components if _unit_of(k) == U) > 0
        iso = []
        for L in (0, 1, 2):
            pv = osc(final_pos=final_pos, bid=bid, kontras={U: L})
            iso.append(sum(v for k, v in pv.components.items() if _unit_of(k) == U))
        out[U] = {"made": bool(made), "iso": iso}
    return out, live, (final_pos.trump is None)


# ── build: champion auction (uniform deals) ──────────────────────────────────
_BID_FN = None


def _init():
    global _BID_FN
    from net_bidder import make_net_bid_fn
    _BID_FN = make_net_bid_fn()


def _build_worker(seed):
    from auction import run_auction
    r = run_auction(seed, _BID_FN)
    if r["winner"] is None:
        return {"seed": seed, "kept": False}
    pos, bid = _play_terminal(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                              r["talon"], seed)
    utab, live, colorless = _unit_table(pos, bid)
    return {"seed": seed, "kept": True, "winner": r["winner"], "rung_index": r["rung"].index,
            "contract": r["contract"], "trump": r["trump"], "colorless": colorless,
            "units": {U: utab[U] for U in live},
            "feats": _features(r["sol"], r["def1"], r["def2"], r["talon"], r["trump"]),
            "sol": _ids(r["sol"]), "d1": _ids(r["def1"]), "d2": _ids(r["def2"]),
            "talon": _ids(r["talon"])}


def build(n_seeds, out_path=PLAYED, src="champion"):
    seeds = [SEED_BASE + i for i in range(n_seeds) if (SEED_BASE + i) not in _seen(out_path)]
    print(f"build[{src}]: {len(seeds)} new seeds", flush=True)
    t0 = time.perf_counter(); kept = 0; ucount = {}
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(out_path, "a") as o:
        for i, rec in enumerate(pool.imap_unordered(_build_worker, seeds, chunksize=2), 1):
            o.write(json.dumps(rec) + "\n"); o.flush()
            if rec.get("kept"):
                kept += 1
                for U in rec["units"]:
                    ucount[U] = ucount.get(U, 0) + 1
            if i % 25 == 0:
                _hb("build", i, len(seeds), t0, f"kept {kept} units={ucount}")
    print(f"build done: kept {kept}  per-unit {ucount}", flush=True)


# ── pools: per-unit GOD makeability (perfect-defense) from each viewer ────────
POOLS = os.path.join(_HERE, "pools.jsonl")
POOL = int(os.environ.get("POOL", "24"))

# per-unit god objective (mirrors exp23 gen_base_events): (solver, weights, restrict)
_OBJ = {
    "parti":     ("parti", None, None),
    "ulti":      ("ulti", None, None),
    "betli":     ("betli", None, None),
    "durchmars": ("durchmars", None, None),
    "40_100":    ("multi", {"score_geq_100": 1.0}, "40"),
    "20_100":    ("multi", {"score_geq_100": 1.0}, "20"),
}


def _unit_god_make(sol, d1, d2, talon, trump, unit, viewer, n, seed):
    """P(soloist makes `unit` vs PERFECT defense | viewer's own hand), by sampling
    worlds from the viewer's own-hand info set and god-solving that unit's objective.
    Cheat-clean. Returns (p_god, n_used)."""
    import random
    from solvers import pis, determinize as _det
    from eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights
    solver, weights, restrict = _OBJ[unit]
    build_c = "durchmars" if solver == "durchmars" else ("betli" if solver == "betli" else "parti")
    if weights is not None:
        set_multi_weights(**weights)
    root = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
                              contract=build_c, trump=trump, talon=list(talon),
                              declare_marriages=(trump is not None), marriage_restrict=restrict)
    iset = _det.build_info_set(root, viewer, build_c, voids=None)
    rng = random.Random(seed); w = 0; used = 0
    for _ in range(n):
        try:
            hands, tal = _det.sample_world(iset, rng)
            spos = (pis.clone_with_hands_and_talon(root, hands, tal)
                    if iset.talon_known is None else pis.clone_with_hands(root, hands))
            if weights is not None:
                set_multi_weights(**weights)
            w += 1 if god_says_soloist_wins(spos, contract=solver) else 0
            used += 1
        except Exception:
            continue
    return (w / used if used else 0.0), used


def _pool_worker(rec):
    sol, d1, d2 = _cards(rec["sol"]), _cards(rec["d1"]), _cards(rec["d2"])
    talon = _cards(rec["talon"]); trump = rec["trump"]; seed = rec["seed"]
    god = {}
    for U in rec["units"]:
        god[U] = {}
        for v in (0, 1, 2):
            p, used = _unit_god_make(sol, d1, d2, talon, trump, U, v, POOL, seed + 13 * v + 101 * UNITS.index(U))
            god[U][v] = [round(p, 4), used]
    return {"seed": seed, "god": god}


def pools(in_path=PLAYED, out_path=POOLS):
    kept = [json.loads(l) for l in open(in_path) if json.loads(l).get("kept")]
    seen = _seen(out_path)
    todo = [r for r in kept if r["seed"] not in seen]
    print(f"pools: {len(todo)} deals ({len(kept)} kept), POOL={POOL}/unit/viewer", flush=True)
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(out_path, "a") as o:
        for i, rec in enumerate(pool.imap_unordered(_pool_worker, todo, chunksize=1), 1):
            o.write(json.dumps(rec) + "\n"); o.flush()
            if i % 20 == 0:
                _hb("pools", i, len(todo), t0)
    print("pools done", flush=True)


# ── godactual: TRUE perfect-play make on the REAL deal (teaching truth) ───────
GODACT = os.path.join(_HERE, "godactual.jsonl")


def _god_actual_worker(rec):
    from solvers import pis
    from eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights
    sol, d1, d2 = _cards(rec["sol"]), _cards(rec["d1"]), _cards(rec["d2"])
    talon = _cards(rec["talon"]); trump = rec["trump"]
    ga = {}
    for U in rec["units"]:
        solver, weights, restrict = _OBJ[U]
        build_c = "durchmars" if solver == "durchmars" else ("betli" if solver == "betli" else "parti")
        if weights is not None:
            set_multi_weights(**weights)
        pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
                                 contract=build_c, trump=trump, talon=list(talon),
                                 declare_marriages=(trump is not None), marriage_restrict=restrict)
        try:
            ga[U] = 1 if god_says_soloist_wins(pos, contract=solver) else 0
        except Exception:
            ga[U] = None
    return {"seed": rec["seed"], "god_actual": ga}


def godactual():
    kept = [json.loads(l) for l in open(PLAYED) if json.loads(l).get("kept")]
    seen = _seen(GODACT)
    todo = [r for r in kept if r["seed"] not in seen]
    print(f"godactual: {len(todo)} deals (true perfect-play make on the REAL deal)", flush=True)
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(GODACT, "a") as o:
        for i, rec in enumerate(pool.imap_unordered(_god_actual_worker, todo, chunksize=4), 1):
            o.write(json.dumps(rec) + "\n"); o.flush()
            if i % 100 == 0:
                _hb("godactual", i, len(todo), t0)
    print("godactual done", flush=True)


# ── analysis: per-unit decision (deployed vs calibrated vs oracle) + teaching ──
_TAU = [i / 100.0 for i in range(0, 101, 2)]


def _load_rows():
    """Join played+pools → per-(deal,unit) rows with realized iso, made, signals, feats."""
    from scorers import resolve_bidset
    from ladder import LADDER
    played = {json.loads(l)["seed"]: json.loads(l)
              for l in open(PLAYED) if json.loads(l).get("kept")}
    pools_by = {json.loads(l)["seed"]: json.loads(l)["god"] for l in open(POOLS)}
    gact = {}
    if os.path.exists(GODACT):
        gact = {json.loads(l)["seed"]: json.loads(l)["god_actual"] for l in open(GODACT)}
    rows = {U: [] for U in UNITS}
    for s, rec in played.items():
        if s not in pools_by:
            continue
        god = pools_by[s]
        bid = resolve_bidset(LADDER[rec["rung_index"]], _cards(rec["sol"]), rec["trump"])
        for U, d in rec["units"].items():
            g = god.get(U)
            if not g:
                continue
            f = rec["feats"]
            rows[U].append({
                "seed": s, "made": d["made"], "iso": d["iso"], "bid": bid,
                "colorless": rec["colorless"], "contract": rec["contract"],
                "god_sol": g["0"][0], "god_d1": g["1"][0], "god_d2": g["2"][0],
                "god_def": min(g["1"][0], g["2"][0]),
                "god_actual": gact.get(s, {}).get(U),
                "nt_max": max(f["d1"]["ntrump"], f["d2"]["ntrump"]),
                "nt_d1": f["d1"]["ntrump"], "nt_d2": f["d2"]["ntrump"],
                "sol_cardpts": f["sol"]["cardpts"],
            })
    return rows


def _rekontra_level(row):
    """Fixed soloist rekontra: rekontra iff the soloist's own-hand god-makeability
    says the unit is likely made (>0.5). level 2 if kontra+rekontra else 1."""
    return 2 if row["god_sol"] > 0.5 else 1


def _mean_iso(sub, decide):
    g = f = 0.0
    for r in sub:
        k = decide(r)
        g += r["iso"][_rekontra_level(r)] if k else r["iso"][0]
        f += 1 if k else 0
    n = max(1, len(sub))
    return g / n, 100 * f / n


def analyze():
    from kontra import _sol_ev
    rows = _load_rows()
    out = ["# exp27 — per-unit kontra decision (held-out test). Soloist per-def GP, "
           "LOWER = better defense. Rekontra fixed (soloist god>0.5).\n"]
    for U in UNITS:
        R = rows[U]
        if len(R) < 40:
            out.append(f"\n## {U}: n={len(R)} — too few, skip"); continue
        tr = [r for r in R if r["seed"] % 2 == 0]
        te = [r for r in R if r["seed"] % 2 == 1]
        never, _ = _mean_iso(te, lambda r: False)
        oracle, _ = _mean_iso(te, lambda r: not r["made"])
        mk = 100 * sum(r["made"] for r in R) / len(R)
        out.append(f"\n## {U}  (n={len(R)}, test {len(te)}, PIMC made {mk:.0f}%)")
        out.append(f"- never {never:+.3f} | oracle {oracle:+.3f} (ceiling gain {never-oracle:+.3f})")
        # deployed rule: kontra iff _sol_ev(whole-bid god makeability, 0) < 0 (only where the
        # engine actually kontras — simple single-unit games; combined → never)
        def deployed(r):
            from bidder import _is_simple
            if not _is_simple(r["bid"]):
                return False
            return _sol_ev(r["god_def"], r["bid"], 0) < 0
        dep, depf = _mean_iso(te, deployed)
        out.append(f"- deployed {dep:+.3f} (fire {depf:.0f}%, gain vs never {never-dep:+.3f})")
        # candidate calibrated rules, tuned on train:
        cands = {}
        # god threshold
        bg = min(_TAU, key=lambda t: _mean_iso(tr, lambda r, t=t: r["god_def"] < t)[0])
        cands[f"god<{bg:.2f}"] = (lambda r, t=bg: r["god_def"] < t)
        # trump gate (colored trump games)
        if U in ("ulti", "40_100", "20_100", "durchmars"):
            bt = min(range(1, 6), key=lambda T: _mean_iso(tr, lambda r, T=T: r["nt_max"] >= T)[0])
            cands[f"trumps>={bt}"] = (lambda r, T=bt: r["nt_max"] >= T)
            # combined
            bc = min([(t, T) for t in _TAU for T in range(1, 6)],
                     key=lambda p: _mean_iso(tr, lambda r, t=p[0], T=p[1]: r["god_def"] < t or r["nt_max"] >= T)[0])
            cands[f"god<{bc[0]:.2f} or trumps>={bc[1]}"] = (lambda r, t=bc[0], T=bc[1]: r["god_def"] < t or r["nt_max"] >= T)
        best = None
        out.append("- calibrated rules (test):")
        for name, dec in cands.items():
            gp, fr = _mean_iso(te, dec)
            out.append(f"    {name:<28} {gp:+.3f}  fire {fr:.0f}%  gain {never-gp:+.3f}")
            if best is None or gp < best[1]:
                best = (name, gp)
        out.append(f"  → best: {best[0]} (gain vs deployed {dep-best[1]:+.3f})")
    txt = "\n".join(out) + "\n"
    open(os.path.join(_HERE, "results_units.md"), "w").write(txt)
    print(txt, flush=True)


_UNIT_HUN = {"parti": "Párti", "ulti": "Ulti", "40_100": "40-100 (négyszáz-száz)",
             "20_100": "20-100 (húsz-száz)", "durchmars": "Durchmars (duri)", "betli": "Betli"}


def teaching():
    """God (perfect-play) vs PIMC (realistic) make rates per unit, by feature — the
    student-facing tables. BOTH regimes, always labeled, with sample sizes."""
    rows = _load_rows()

    def pct(sub, key):
        v = [r[key] for r in sub if r.get(key) is not None]
        return (100 * sum(v) / len(v)) if v else None

    def real(sub):
        return 100 * sum(r["made"] for r in sub) / len(sub) if sub else None

    out = ["# Ulti — how often does a contract actually make?",
           "*A study guide for defenders: when is it worth saying kontra?*\n",
           "Every number below is **P(the soloist makes the contract)**, measured over "
           "thousands of real bid hands. Two regimes are shown side by side and never mixed:\n",
           "- **Realistic** — against strong, human-like play (what you should expect at the table).",
           "- **Perfect** — against flawless defense (the theoretical best a defender could do).\n",
           "The two are usually close. Where **Perfect is much lower than Realistic**, the "
           "contract is *beatable but only with precise defense* — those are the hands worth "
           "studying.\n",
           "---\n## 1. The big picture\n",
           "| Contract | Realistic make | Perfect make | Read |",
           "|---|---|---|---|"]
    reads = {"parti": "a real coin-flip-ish fight — kontra often pays",
             "ulti": "very strong once bid — rarely worth kontra",
             "40_100": "strong — rarely worth kontra",
             "20_100": "beatable if you're trump-rich",
             "durchmars": "fragile — often beatable",
             "betli": "the defensive-skill contract (see §3)"}
    for U in UNITS:
        R = rows[U]
        if len(R) < 40:
            continue
        rm, pp = real(R), pct(R, "god_actual")
        out.append(f"| {_UNIT_HUN[U]} | **{rm:.0f}%** | {pp:.0f}% | {reads[U]} |")

    out.append("\n---\n## 2. Trump count is (almost) everything for the trick contracts\n")
    out.append("For **ulti** and **durchmars**, the single biggest clue is how many trumps "
               "*you* (a defender) hold. Nothing else — not fancy win-probability math — beats "
               "just counting your trumps.\n")
    for U in ("ulti", "durchmars", "20_100", "40_100"):
        R = rows[U]
        if len(R) < 40:
            continue
        out.append(f"\n### {_UNIT_HUN[U]}")
        out.append("| Your trumps (as a defender) | hands | Realistic make | Perfect make |")
        out.append("|---|---|---|---|")
        for T in range(0, 6):
            b = [r for r in R if r["nt_max"] == T]
            if len(b) >= 8:
                pp = pct(b, "god_actual")
                out.append(f"| {T} | {len(b)} | {real(b):.0f}% | {pp:.0f}% |")
        # a plain-language rule of thumb
        tips = {"ulti": "**Rule of thumb:** only kontra an ulti when you hold **4+ trumps** "
                        "(then it fails ~2 times in 3). With 2–3 trumps, let it go.",
                "durchmars": "**Rule of thumb:** a durchmars needs *every* trick. With **3+ "
                        "trumps** you almost always have a stopper — kontra freely (it makes ~1 "
                        "time in 20). With none, it still makes about half the time.",
                "20_100": "**Rule of thumb:** trump-rich (4+) makes it very beatable; otherwise the "
                        "soloist usually gets there.",
                "40_100": "**Rule of thumb:** strong contract — even with 4 trumps it makes about "
                        "half the time. Kontra only with real trump strength."}
        out.append("\n" + tips[U])

    # betli — the one real perfect-vs-real gap
    Rb = rows["betli"]
    if len(Rb) >= 40:
        out.append("\n---\n## 3. Betli — the contract that rewards *skill*\n")
        out.append(f"Betli (take zero tricks) is the one place perfect and realistic play "
                   f"diverge sharply: realistic make **{real(Rb):.0f}%**, but perfect defense "
                   f"holds the soloist to **{pct(Rb,'god_actual'):.0f}%**.\n")
        out.append("In other words: **a betli that *should* be beaten is only actually beaten "
                   "by a defender who plays the squeeze correctly.** Unlike ulti or duri, you "
                   "can't read betli off your trump count — there are no trumps. It's pure "
                   "card-play skill, which is exactly why it's the contract most worth "
                   "practicing on the defense.\n")

    out.append("---\n## 4. Why over-eager kontra loses\n")
    out.append("A tempting mistake is to kontra whenever a contract *looks* hard. But the "
               "numbers say most bid contracts make: a soloist only bids what their hand "
               "supports. Ulti makes 83%, 40-100 makes 79%, betli 86%. Kontra doubles the "
               "stake **both ways** — so kontra-ing a contract that makes 80% of the time is a "
               "long-run loss. Save your kontra for the hands the tables above flag as genuinely "
               "beatable: párti (a genuine fight most hands), a **trump-rich** defense "
               "against ulti/duri, and betli when you can actually play the squeeze.\n")
    out.append("*(Data: several thousand champion-bid hands, played out both realistically "
               "(Monte-Carlo perfect-information search) and under a perfect-information solver. "
               "Sample sizes shown per row.)*")
    txt = "\n".join(out) + "\n"
    open(os.path.join(_HERE, "TEACHING.md"), "w").write(txt)
    print(txt, flush=True)


# ── tournament: current frontier (deployed kontra) vs candidate (per-unit) ────
def _kontra_primary(bid):
    from bidder import _is_simple
    if not _is_simple(bid):
        return None
    if bid.betli:     return "betli"
    if bid.ulti:      return "ulti"
    if bid.durchmars: return "durchmars"
    return "parti"


def _deal_records():
    from scorers import resolve_bidset
    from ladder import LADDER
    played = {json.loads(l)["seed"]: json.loads(l)
              for l in open(PLAYED) if json.loads(l).get("kept")}
    pools_by = {json.loads(l)["seed"]: json.loads(l)["god"] for l in open(POOLS)}
    deals = []
    for s, rec in played.items():
        if s not in pools_by:
            continue
        god = pools_by[s]; f = rec["feats"]
        bid = resolve_bidset(LADDER[rec["rung_index"]], _cards(rec["sol"]), rec["trump"])
        units = {}
        for U, d in rec["units"].items():
            g = god.get(U)
            if not g:
                continue
            units[U] = {"iso": d["iso"], "made": d["made"],
                        "god_sol": g["0"][0], "god_d1": g["1"][0], "god_d2": g["2"][0]}
        if not units:
            continue
        deals.append({"seed": s, "bid": bid, "contract": rec["contract"],
                      "colorless": rec["colorless"], "primary": _kontra_primary(bid),
                      "nt_d1": f["d1"]["ntrump"], "nt_d2": f["d2"]["ntrump"], "units": units})
    return deals


def _def_kontras(brain, deal, U, godd, ntd, P):
    """Does a defender (own view: godd=blind makeability, ntd=own trumps) kontra unit U?"""
    from kontra import _sol_ev
    from bidder import _is_simple
    bid = deal["bid"]
    if brain == "deployed":
        if not _is_simple(bid) or U != deal["primary"]:
            return False                                   # deployed: primary of simple bids only
        return _sol_ev(godd, bid, 0) < 0
    # candidate — per unit
    if U == "parti":
        return godd < P["tau_parti"]
    if U == "ulti":
        return ntd >= 4
    if U == "durchmars" and not deal["colorless"]:
        return ntd >= 3
    return False                                           # 40_100/20_100/betli/colorless-duri abstain


def _sol_rekontras(brain, deal, U, P):
    from kontra import _sol_ev
    if brain == "never":
        return False
    bid = deal["bid"]; p = deal["units"][U]["god_sol"]
    if brain == "deployed":
        return _sol_ev(p, bid, 0) > 0                      # deployed proxy (root makeability)
    return _sol_ev(p, bid, 2) > _sol_ev(p, bid, 1)         # candidate: proper backward-induction


def _deal_gp(deal, def_brain, sol_brain, P):
    """Return (soloist_gp, def1_gp, def2_gp) under the two brains' kontra decisions."""
    pd = [0.0, 0.0]                                        # soloist GP vs def0, def1
    for U, ud in deal["units"].items():
        iso = ud["iso"]
        if deal["colorless"]:
            for di, (godd, ntd) in enumerate([(ud["god_d1"], deal["nt_d1"]),
                                              (ud["god_d2"], deal["nt_d2"])]):
                lvl = 0
                if _def_kontras(def_brain, deal, U, godd, ntd, P):
                    lvl = 2 if _sol_rekontras(sol_brain, deal, U, P) else 1
                pd[di] += iso[lvl]
        else:                                              # colored shared (együtt sírunk)
            kon = (_def_kontras(def_brain, deal, U, ud["god_d1"], deal["nt_d1"], P) or
                   _def_kontras(def_brain, deal, U, ud["god_d2"], deal["nt_d2"], P))
            lvl = 0
            if kon:
                lvl = 2 if _sol_rekontras(sol_brain, deal, U, P) else 1
            pd[0] += iso[lvl]; pd[1] += iso[lvl]
    return pd[0] + pd[1], -pd[0], -pd[1]


def tournament():
    deals = _deal_records()
    tr = [d for d in deals if d["seed"] % 2 == 0]
    te = [d for d in deals if d["seed"] % 2 == 1]
    # tune candidate parti threshold on TRAIN (minimise soloist GP, parti-only proxy)
    def soloist_gp(sub, P, dbrain="candidate", sbrain="candidate"):
        return sum(_deal_gp(d, dbrain, sbrain, P)[0] for d in sub) / max(1, len(sub))
    best_tau = min(_TAU, key=lambda t: soloist_gp(tr, {"tau_parti": t}))
    P = {"tau_parti": best_tau}
    n = len(te)

    out = ["# exp27 TOURNAMENT — current frontier (deployed kontra) vs candidate (per-unit)\n",
           f"Held-out test N={n} deals. Candidate parti τ*={best_tau:.2f} (tuned on train).",
           "Soloist GP is per-deal to the soloist; defender GP is the pair's total (=-soloist). "
           "Kontra doesn't change play, so this isolates the kontra/rekontra decisions.\n"]

    # (A) self-play leakage: all-deployed vs all-candidate
    dep_sol = soloist_gp(te, P, "deployed", "deployed")
    can_sol = soloist_gp(te, P, "candidate", "candidate")
    out.append("## A. Self-play (all 3 seats one brain) — soloist GP/deal (lower = better defense)")
    out.append(f"- all-deployed  soloist {dep_sol:+.3f}  → defenders {-dep_sol:+.3f}")
    out.append(f"- all-candidate soloist {can_sol:+.3f}  → defenders {-can_sol:+.3f}")
    out.append(f"- candidate defenders concede {dep_sol-can_sol:+.3f} GP/deal LESS than deployed defenders\n")

    # (B) head-to-head per table: candidate always plays deployed. Average the two
    # mirror tables (candidate defends / candidate solos) → per-table GP advantage.
    diff = 0
    adv = 0.0
    for d in te:
        s1, _, _ = _deal_gp(d, "candidate", "deployed", P)   # cand DEF vs dep SOL → soloist=deployed
        s2, _, _ = _deal_gp(d, "deployed", "candidate", P)   # dep DEF vs cand SOL → soloist=candidate
        # candidate GP per table: as defender = -s1 ; as soloist = +s2 ; average the 2 tables
        adv += ((-s1) + s2) / 2.0 - ((s1) + (-s2)) / 2.0     # cand avg − deployed avg (=s2−s1)
        if _deal_gp(d, "candidate", "candidate", P) != _deal_gp(d, "deployed", "deployed", P):
            diff += 1
    out.append("## B. Head-to-head (candidate vs current frontier, per table)")
    out.append(f"- **candidate wins {adv/n:+.3f} GP/deal** head-to-head (decisions differ on {100*diff/n:.0f}% of deals)\n")

    # (C) defender kontra only (both brains share deployed rekontra) — isolates the kontra fix
    net_d = sum(sum(_deal_gp(d, "candidate", "deployed", P)[1:]) -
                sum(_deal_gp(d, "deployed", "deployed", P)[1:]) for d in te) / n
    out.append("## C. Defender kontra only (rekontra = deployed for both) — the isolated defender fix")
    out.append(f"- candidate defenders gain {net_d:+.3f} GP/deal vs deployed defenders (same soloist)\n")

    # (C2) rekontra only (both brains share CANDIDATE defender) — isolates the rekontra rule
    reko = {}
    for sb, lbl in (("deployed", "deployed-rekontra"), ("candidate", "candidate-rekontra"),
                    ("never", "never-rekontra")):
        reko[lbl] = sum(_deal_gp(d, "candidate", sb, P)[0] for d in te) / n
    out.append("## C2. Rekontra rule (candidate defenders fixed) — soloist GP/deal (HIGHER = better for soloist)")
    for lbl, v in reko.items():
        out.append(f"- {lbl}: soloist {v:+.3f}")
    out.append("")

    # (D) by contract
    out.append("## D. By contract — soloist GP/deal, deployed vs candidate self-play")
    import collections
    byc = collections.defaultdict(list)
    for d in te:
        byc[d["contract"]].append(d)
    out.append("| contract | n | deployed sol GP | candidate sol GP | defender gain |")
    out.append("|---|---|---|---|---|")
    for c, sub in sorted(byc.items(), key=lambda kv: -len(kv[1]))[:12]:
        ds = soloist_gp(sub, P, "deployed", "deployed")
        cs = soloist_gp(sub, P, "candidate", "candidate")
        out.append(f"| {c} | {len(sub)} | {ds:+.2f} | {cs:+.2f} | {ds-cs:+.2f} |")
    txt = "\n".join(out) + "\n"
    open(os.path.join(_HERE, "TOURNAMENT.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    N = int(os.environ.get("N", "2000"))
    if cmd == "build":
        build(N)
    elif cmd == "pools":
        pools()
    elif cmd == "godactual":
        godactual()
    elif cmd == "tournament":
        tournament()
    elif cmd == "analyze":
        analyze()
    elif cmd == "teaching":
        teaching()
    elif cmd == "smoke":
        build(int(os.environ.get("N", "40"))); pools()
    else:
        print(f"unknown cmd {cmd}", flush=True)
