"""exp40 — REALISTIC-defense make-probability datagen for the WHOLE base-event slate.

Generalizes exp37 (betli) to the remaining bidder heads: parti, ulti, reach100_40,
reach100_20, duri_colored, colorless_duri. For each head we deal a boundary-biased
hand, make ONE 2-card discard (mirroring the bidder's pickup), then play the contract
out MOVE-BY-MOVE with PIMC on all three seats and record whether the soloist ACTUALLY
MADE it. The GOD (double-dummy) label is stored alongside (one extra solve) so
training/analysis can measure the god-vs-real gap the deployed heads are blind to.

CRITICAL — the label plays the contract the way PRODUCTION plays it: play.py / exp24
`pimc_outcome` route ulti / parti / colored-duri / 100-games through the `multi` solver
with silent-game weights (build as `parti`), NOT their dedicated solvers. So the
realistic label defense == what the contract faces live (PIMC N=16 = the deployed
defense for every non-betli contract). colorless-duri and betli keep their dedicated
solvers, exactly as production does. The GOD comparison label DOES use the dedicated
per-head solver (matching how each deployed god head was trained) — that's a
perfect-info solve with no world-sampling, so it's exact and can't hit the
determinizer edge case that the dedicated `ulti` PIMC path does.

  soloist   = PIMC (deployed non-terit soloist is EXPLOIT ⇒ slightly stronger ⇒ the
              head is mildly conservative, the safe direction for a bidder)
  defenders = PIMC N=16 = the deployed defense for every non-betli contract

Features match each head's god counterpart EXACTLY (provider.base_probs):
  colored heads → featurize(hand10, trump, True)  (36-dim)
  colorless_duri → featurize(hand10, None, False) (32-dim)

Env: HEAD (required), N (60000), WORKERS (8), ALPHA_MAX (per-head default), PIMC_N (16),
     SEED_BASE (400000000), OUT.
"""
from __future__ import annotations

import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/23_bidding_integration", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ulti.solvers import pis, determinize as _det                                    # noqa: E402
from ulti.eval.pimc_matchup import pimc_pick, defenders_won, god_says_soloist_wins    # noqa: E402
from ulti.eval.dojo import (deal_parti, deal_ulti_biased, deal_durchmars_colored,     # noqa: E402
                       deal_durchmars_colorless, deal_biased, ContractSpec)
from ulti.vnet.pickup import featurize                                               # noqa: E402
from recipe_local import sol_marriages                                          # noqa: E402
from ulti.scoring.oracle import score as score_oracle, BidSet                        # noqa: E402
from trickster._solver_core import set_multi_weights                            # noqa: E402

HEAD       = os.environ["HEAD"]
N          = int(os.environ.get("N", "60000"))
WORKERS    = int(os.environ.get("WORKERS", "8"))
PIMC_N     = int(os.environ.get("PIMC_N", "16"))
SEED_BASE  = int(os.environ.get("SEED_BASE", "400000000"))

# ── Guarantee the trump 40 (King + Felső) in the soloist's hand for reach100_40. ──
_REACH40_SPEC = ContractSpec(
    name="reach100_40", bid_action_offset=-1,
    trump_count_weights={2: .10, 3: .40, 4: .30, 5: .15, 6: .05},
    mandatory_trump_ranks=("king", "upper"),
)

# Per-head config.
#   bid:      the BidSet whose PRODUCTION play path we replicate (via _play_spec, below)
#   colorless: colorless-duri path (dedicated durchmars solver, no trump) vs multi
#   colored:  36-dim colored featurize (else 32-dim colorless)
#   made:     terminal make-label check
#   god:      (solve, build, weights, restrict) for the DEDICATED double-dummy label
#   direct_alpha: dealer uses alpha DIRECTLY (pre-sample the sweep) vs deal_biased (resamples)
#   require:  marriage that must survive the discard ("40"/"20"/None)
_CFG = {
    "parti": dict(
        dealer=lambda s, a: deal_parti(seed=s, alpha=a), colored=True, colorless=False,
        direct_alpha=False, require=None, alpha_max=1.0, bid=BidSet(parti=True),
        made="defenders_won:parti", god=("parti", "parti", None, None)),
    "ulti": dict(
        dealer=lambda s, a: deal_ulti_biased(seed=s, alpha=a), colored=True, colorless=False,
        direct_alpha=False, require=None, alpha_max=1.0, bid=BidSet(ulti=True),
        made="defenders_won:ulti", god=("ulti", "ulti", None, None)),
    "duri_colored": dict(
        dealer=lambda s, a: deal_durchmars_colored(seed=s, alpha=a), colored=True, colorless=False,
        direct_alpha=False, require=None, alpha_max=1.8, bid=BidSet(durchmars=True),
        made="defenders_won:durchmars", god=("durchmars", "durchmars", None, None)),
    "colorless_duri": dict(
        dealer=lambda s, a: deal_durchmars_colorless(seed=s, alpha=a), colored=False, colorless=True,
        direct_alpha=True, require=None, alpha_max=2.5, bid=BidSet(durchmars=True),
        made="defenders_won:durchmars", god=("durchmars", "durchmars", None, None)),
    "reach100_40": dict(
        dealer=lambda s, a: deal_biased(spec=_REACH40_SPEC, seed=s, alpha=a), colored=True, colorless=False,
        direct_alpha=False, require="40", alpha_max=1.0, bid=BidSet(forty_hundred=True),
        made="oracle:40_100", god=("multi", "parti", {"score_geq_100": 1.0}, "40")),
    "reach100_20": dict(
        dealer=lambda s, a: deal_parti(seed=s, alpha=a), colored=True, colorless=False,
        direct_alpha=False, require="20", alpha_max=1.0, bid=BidSet(twenty_hundred=True),
        made="oracle:20_100", god=("multi", "parti", {"score_geq_100": 1.0}, "20")),
}
if HEAD not in _CFG:
    raise SystemExit(f"unknown HEAD={HEAD!r}; choose from {sorted(_CFG)}")
CFG = _CFG[HEAD]
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", str(CFG["alpha_max"])))
OUT = os.environ.get("OUT") or os.path.join(_HERE, f"{HEAD}_real.npz")


def _play_weights(bid, sol, trump):
    """PRODUCTION silent-game weights — verbatim from exp24 scorers._play_weights /
    play.py _play_weights, so the datagen plays a contract exactly as it's played live."""
    has40, has20 = sol_marriages(sol, trump)
    w = dict(parti_pts=1.0, silent_ulti=2.0, silent_durchmars=3.0)
    if bid.ulti:
        w["silent_ulti"] = 8.0
    if bid.durchmars:
        w["silent_durchmars"] = 8.0
    if bid.forty_hundred:
        w["score_geq_100"] = 8.0
    elif bid.twenty_hundred:
        w["score_geq_100"] = 16.0
    else:
        w["score_geq_100"] = (2.0 if has40 else 0.0) + (1.0 if has20 else 0.0)
    return w


def _play_spec(cfg, sol, trump):
    """(solve, build, weights, trump_arg, restrict) for the PRODUCTION play path."""
    bid = cfg["bid"]
    if cfg["colorless"]:                    # dedicated durchmars solver, no trump (== production)
        return "durchmars", "durchmars", None, None, None
    restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
    return "multi", "parti", _play_weights(bid, sol, trump), trump, restrict


def _marriage_cards(cards, trump, which):
    """The (king, felső) of a held marriage to PROTECT from the discard, or None.
    '40' = trump marriage; '20' = any non-trump suit holding both king+felső."""
    def pair(suit):
        k = [c for c in cards if c.suit == suit and c.rank == "king"]
        u = [c for c in cards if c.suit == suit and c.rank == "upper"]
        return (k[0], u[0]) if k and u else None
    if which == "40":
        return pair(trump)
    for s in sorted({c.suit for c in cards if c.suit != trump}):
        p = pair(s)
        if p:
            return p
    return None


def _discard(sol12, trump, require, rng):
    protect = set()
    if require is not None:
        pr = _marriage_cards(sol12, trump, require)
        if pr is None:
            return None                     # dealer failed to plant the marriage → skip
        protect = {id(pr[0]), id(pr[1])}
    droppable = [i for i, c in enumerate(sol12) if id(c) not in protect]
    di = rng.sample(droppable, 2)
    sol10 = [sol12[i] for i in range(12) if i not in di]
    disc = [sol12[i] for i in di]
    return sol10, disc


def _made(pos, cfg):
    kind, _, arg = cfg["made"].partition(":")
    if kind == "defenders_won":
        return 0 if defenders_won(pos, arg) else 1
    bid = BidSet(forty_hundred=(arg == "40_100"), twenty_hundred=(arg == "20_100"))
    pvec = score_oracle(final_pos=pos, bid=bid)
    return 1 if pvec.components.get(arg, 0.0) > 0 else 0


def _playout(sol10, d1, d2, disc, trump, cfg, seed):
    solve, build, weights, t, restrict = _play_spec(cfg, sol10, trump)
    if weights is not None:
        set_multi_weights(**weights)
    pos = pis.build_position(hands=[list(sol10), list(d1), list(d2)], soloist=0, leader=0,
                             contract=build, trump=t, talon=list(disc),
                             declare_marriages=(t is not None), marriage_restrict=restrict)
    # durchmars is MONOTONE: the sweep is lost the instant a defender wins a trick, so
    # stop then — identical label, but most unmakeable duris break in trick 1-2 (big speedup).
    dur_stop = (cfg["made"] == "defenders_won:durchmars")
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        if weights is not None:
            set_multi_weights(**weights)                 # process-global; reassert each move
        mv = pimc_pick(pos=pos, contract=solve, n_samples=PIMC_N, seed=seed * 131 + mi, voids_dict=vd)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, mv); mi += 1
        if dur_stop and (pos.captured[1] or pos.captured[2]):
            return 0                                     # a defender took a trick → sweep failed
    return _made(pos, cfg)


def _god_label(sol10, d1, d2, disc, trump, cfg):
    solve, build, weights, restrict = cfg["god"]
    t = trump if cfg["colored"] else None
    if weights is not None:
        set_multi_weights(**weights)
    pos0 = pis.build_position(hands=[list(sol10), list(d1), list(d2)], soloist=0, leader=0,
                              contract=build, trump=t, talon=list(disc),
                              declare_marriages=(t is not None), marriage_restrict=restrict)
    return 1 if god_says_soloist_wins(pos0, contract=solve) else 0


def worker(seed):
    """One record, or None to skip (marriage not planted, or a rare solver/determinizer
    edge case — skip-on-error so a single bad deal can't kill an unattended long run)."""
    cfg = CFG
    try:
        rng = random.Random(seed ^ 0xA5A5A5A5)
        a = rng.uniform(0.0, ALPHA_MAX)
        # deal_biased resamples U(0,alpha) internally → pass ALPHA_MAX for a clean sweep;
        # direct-alpha dealers (colorless) use alpha as-is → pass the pre-sampled a.
        deal = cfg["dealer"](seed, a if cfg["direct_alpha"] else ALPHA_MAX)
        trump = deal.trump
        sol12 = list(deal.sol_hand) + list(deal.talon)
        dd = _discard(sol12, trump, cfg["require"], rng)
        if dd is None:
            return None
        sol10, disc = dd
        d1, d2 = list(deal.def1_hand), list(deal.def2_hand)

        y_god = _god_label(sol10, d1, d2, disc, trump, cfg)
        y_real = _playout(sol10, d1, d2, disc, trump, cfg, seed)
        x = (featurize(sol10, trump, True) if cfg["colored"]
             else featurize(sol10, None, False)).astype(np.float32)
        return x, int(y_real), int(y_god), float(a if cfg["direct_alpha"] else -1.0)
    except Exception:
        return "ERR"                        # counted separately; does not abort the pool


def main():
    print(f"=== exp40 realistic datagen: HEAD={HEAD} ===", flush=True)
    print(f"  N={N} ALPHA_MAX={ALPHA_MAX} PIMC_N={PIMC_N} workers={WORKERS} "
          f"colored={CFG['colored']} out={os.path.basename(OUT)}", flush=True)
    seeds = (SEED_BASE + i for i in range(N))
    Xs, yr, yg, al = [], [], [], []
    skipped = errors = 0
    t0 = time.perf_counter(); report_every = max(1000, N // 60)
    ctx = get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        for r in pool.imap_unordered(worker, seeds, chunksize=32):
            if r is None:
                skipped += 1; continue
            if r == "ERR":
                errors += 1; continue
            Xs.append(r[0]); yr.append(r[1]); yg.append(r[2]); al.append(r[3])
            k = len(yr)
            if k % report_every == 0:
                wall = time.perf_counter() - t0; rate = k / wall
                mr = sum(yr) / k; gr = 1.0 - sum(yg) / k
                gap = sum(1 for i in range(k) if yr[i] == 1 and yg[i] == 0) / k
                print(f"  {k} kept ({skipped} skip, {errors} err)  wall={wall:.0f}s "
                      f"eta={(N - k - skipped - errors) / max(rate, 1e-9):.0f}s  "
                      f"real-made={mr:.3f}  dd-lost={gr:.3f}  dd-lost&made={gap:.3f}", flush=True)
    X = np.stack(Xs)
    y_real = np.array(yr, dtype=np.float32); y_god = np.array(yg, dtype=np.float32)
    alpha = np.array(al, dtype=np.float32)
    np.savez_compressed(OUT, X=X, y=y_real, y_god=y_god, alpha=alpha)
    wall = time.perf_counter() - t0
    gap = float(((y_real == 1) & (y_god == 0)).mean())
    print(f"\n  saved {X.shape[0]} records ({skipped} skipped, {errors} errored)  wall={wall:.0f}s", flush=True)
    print(f"  realistic make-rate={y_real.mean():.4f}  dd-lost={1 - y_god.mean():.4f}  "
          f"dd-lost&made={gap:.4f}  (= the realistic headroom the god head is blind to)", flush=True)


if __name__ == "__main__":
    main()
