"""Fundamental-contract eval — parti (with binary-objective PIMC A/B).

Per (α, seed):
  1. Build the deal (deal_parti: 10-card sol + 2-card talon, moderate
     trump suit, no mandatory ranks).
  2. God-label the opening — does god predict sol can score ≥ 50?
  3. Compute two value heads at t=0 from sol's seat:
       pred_scalar — mean(per-world god value) — current PIMC averaging
       pred_binary — mean(1_{value ≥ 50}) — P(win) under sampled prior
  4. Play three head-to-head games on the same seed:
       a) PIMC-scalar sol vs god def → sol_hold_scalar
       b) PIMC-binary sol vs god def → sol_hold_binary
       c) god sol vs PIMC def        → def_stop

Hypothesis: the parti score is a proxy; the real objective is the
threshold crossing. When PIMC averages raw scores across worlds it can
pick a high-mean / lower-P(win) move over a low-variance / sure-win
move. Switching the soloist's averaging rule to indicator-thresholding
should lift sol_hold on parti (where vanilla PIMC bottomed out at ~80%
in prior runs).

Defender's PIMC pick stays scalar (defenders' goal is to push raw score
down, and the threshold rotation on the defender side is a separate
question — keep apples-to-apples on def_stop with the other contracts).

Threshold = 50 matches `god_says_soloist_wins` for parti and is the
canonical simple-parti win line (no marriages, no talon credit shift).

Parallel via multiprocessing.Pool. Per-result checkpoint to disk so a
mid-run kill loses nothing.
"""
from __future__ import annotations

import json, random, time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict

from eval.dojo import deal_parti
from eval.pimc_matchup import god_pick, god_says_soloist_wins, defenders_won
from solvers import determinize as _det
from solvers import pimc as _pimc
from solvers import pis as pis_bridge
from ulti.card import Card

_CONTRACT     = "parti"
# Parti uses the uniform binary 0/1 solver scale (1 = sol won, 0 = sol lost).
# The "binary PIMC" indicator counts a solver win when value ≥ midpoint.
_THRESHOLD    = 0.5
_VALUE_SCALE  = 1.0

ALPHAS    = [0.30, 0.60, 1.50]
N         = 300
PIMC_N    = 32
N_WORKERS = 8
SEED_BASE = 84_000_000
OUT_DIR   = Path(__file__).parent
CHECKPOINT_PATH = OUT_DIR / "checkpoint_parti.jsonl"


# ─────────────────────────────────────────────────────────────────────
# Binary-objective PIMC: same as solvers.pimc.pimc_decision but
# averages indicator functions instead of raw values. Defender flip
# (argmin) handled by the caller for non-soloist viewers.
# ─────────────────────────────────────────────────────────────────────
def _pimc_decision_binary(*, true_pos, contract, n_samples, seed, threshold,
                           voids=None, must_hold=None):
    rng    = random.Random(seed)
    viewer = pis_bridge.current_player(true_pos)
    iset   = _det.build_info_set(
        true_pos, viewer, contract, voids=voids, must_hold=must_hold,
    )
    hit_sums:  Dict[Card, float] = {}
    hit_count: Dict[Card, int]   = {}
    for _ in range(n_samples):
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            sample_pos = pis_bridge.clone_with_hands_and_talon(true_pos, hands, talon)
        else:
            sample_pos = pis_bridge.clone_with_hands(true_pos, hands)
        values = pis_bridge.solve_all(sample_pos, contract=contract)
        for card, v in values.items():
            hit = 1.0 if float(v) >= threshold else 0.0
            hit_sums[card]  = hit_sums.get(card, 0.0) + hit
            hit_count[card] = hit_count.get(card, 0) + 1
    if not hit_sums:
        return None, {}
    averaged = {c: hit_sums[c] / hit_count[c] for c in hit_sums}
    best     = max(averaged, key=lambda c: averaged[c])
    return best, averaged


def _pick_pimc_scalar(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def _pick_pimc_binary(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc_decision_binary(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, threshold=_THRESHOLD, voids=voids_dict,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


# Local play loop so we can plug in a custom soloist pick. Mirrors
# eval.pimc_matchup.play_one but parameterised on the per-seat callable.
def _play_one(*, deal, contract, sol_pick, def_pick, pimc_n, seed):
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract=contract,
        trump=getattr(deal, "trump", None), talon=list(deal.talon),
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        picker = sol_pick if p == 0 else def_pick
        chosen = picker(
            pos=pos, contract=contract, n_samples=pimc_n,
            seed=seed * 31337 + move_i, voids_dict=voids_dict,
        )
        if chosen is None:
            chosen = rng.choice(pis_bridge.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis_bridge.apply_move(pos, chosen)
        move_i += 1
    return pos


def _pick_god(*, pos, contract, **_unused):
    return god_pick(pos=pos, contract=contract)


def _value_heads(pos, seed):
    _, scalar_avg = _pimc.pimc_decision(
        true_pos=pos, contract=_CONTRACT, n_samples=PIMC_N, seed=seed,
    )
    _, binary_avg = _pimc_decision_binary(
        true_pos=pos, contract=_CONTRACT, n_samples=PIMC_N, seed=seed,
        threshold=_THRESHOLD,
    )
    scalar = float("nan") if not scalar_avg else max(scalar_avg.values())
    binary = float("nan") if not binary_avg else max(binary_avg.values())
    return scalar, binary


def worker(args):
    alpha, seed = args
    deal = deal_parti(seed=seed, alpha=alpha)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos0 = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract=_CONTRACT,
        trump=deal.trump, talon=list(deal.talon),
    )
    label_sol = god_says_soloist_wins(pos0, _CONTRACT)
    pred_scalar, pred_binary = _value_heads(pos0, seed)

    # (a) PIMC-scalar sol vs god def
    final_a = _play_one(
        deal=deal, contract=_CONTRACT,
        sol_pick=_pick_pimc_scalar, def_pick=_pick_god,
        pimc_n=PIMC_N, seed=seed,
    )
    pimcsol_scalar_won = not defenders_won(final_a, _CONTRACT)

    # (b) PIMC-binary sol vs god def
    final_b = _play_one(
        deal=deal, contract=_CONTRACT,
        sol_pick=_pick_pimc_binary, def_pick=_pick_god,
        pimc_n=PIMC_N, seed=seed,
    )
    pimcsol_binary_won = not defenders_won(final_b, _CONTRACT)

    # (c) god sol vs PIMC-scalar def (apples-to-apples with other contracts)
    final_c = _play_one(
        deal=deal, contract=_CONTRACT,
        sol_pick=_pick_god, def_pick=_pick_pimc_scalar,
        pimc_n=PIMC_N, seed=seed,
    )
    godsol_sol_won = not defenders_won(final_c, _CONTRACT)

    return (alpha, seed, label_sol, pred_scalar, pred_binary,
            pimcsol_scalar_won, pimcsol_binary_won, godsol_sol_won)


def _auc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l]
    neg = [s for l, s in zip(labels, scores) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _se(p, n):
    return (p * (1 - p) / max(n, 1)) ** 0.5


def _load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return []
    out = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(tuple(json.loads(line)))
    return out


def main() -> None:
    all_jobs = [(a, SEED_BASE + int(a * 1e6) + i)
                for a in ALPHAS for i in range(N)]

    cached = _load_checkpoint()
    done_seeds = {(r[0], r[1]) for r in cached}
    jobs = [j for j in all_jobs if (j[0], j[1]) not in done_seeds]
    print(f"Resuming with {len(cached)}/{len(all_jobs)} in checkpoint; "
          f"{len(jobs)} remaining; {N_WORKERS} workers (N={N}/α, pimc_n={PIMC_N})",
          flush=True)

    # rows: (alpha, label, pred_scalar, pred_binary, scalar_won, binary_won, godsol_won)
    rows = [(r[0], r[2], r[3], r[4], r[5], r[6], r[7]) for r in cached]
    t0 = time.perf_counter()
    done = 0
    with open(CHECKPOINT_PATH, "a") as ckpt:
        with Pool(N_WORKERS) as pool:
            for r in pool.imap_unordered(worker, jobs, chunksize=2):
                ckpt.write(json.dumps(list(r)) + "\n")
                ckpt.flush()
                rows.append((r[0], r[2], r[3], r[4], r[5], r[6], r[7]))
                done += 1
                if done % 25 == 0:
                    wall = time.perf_counter() - t0
                    rate = done / wall if wall else 0
                    eta  = (len(jobs) - done) / rate if rate else 0
                    print(f"  {done}/{len(jobs)}  wall={wall:.0f}s  "
                          f"rate={rate:.2f}/s  eta={eta:.0f}s", flush=True)
    wall_total = time.perf_counter() - t0
    print(f"\nDone in {wall_total:.0f}s (checkpoint has {len(rows)} rows)\n",
          flush=True)

    by_alpha: dict = {}
    for r in rows:
        by_alpha.setdefault(r[0], []).append(r)

    summary = []
    for a in ALPHAS:
        rs = by_alpha[a]
        labels    = [r[1] for r in rs]
        scalars   = [r[2] for r in rs]
        binaries  = [r[3] for r in rs]
        s_wins    = [r[4] for r in rs]
        b_wins    = [r[5] for r in rs]
        g_wins    = [r[6] for r in rs]
        n     = len(rs)
        n_sf  = sum(labels); n_df = n - n_sf
        sol_hold_scalar = sum(1 for l, w in zip(labels, s_wins) if l and w) / max(n_sf, 1)
        sol_hold_binary = sum(1 for l, w in zip(labels, b_wins) if l and w) / max(n_sf, 1)
        def_stop = sum(1 for l, w in zip(labels, g_wins) if (not l) and (not w)) / max(n_df, 1)
        auc_scalar = _auc(labels, scalars)
        auc_binary = _auc(labels, binaries)
        m_sf_s = sum(p for l, p in zip(labels, scalars)  if l)     / max(n_sf, 1)
        m_df_s = sum(p for l, p in zip(labels, scalars)  if not l) / max(n_df, 1)
        m_sf_b = sum(p for l, p in zip(labels, binaries) if l)     / max(n_sf, 1)
        m_df_b = sum(p for l, p in zip(labels, binaries) if not l) / max(n_df, 1)
        brier_scalar = sum(((p / _VALUE_SCALE) - (1.0 if l else 0.0)) ** 2
                           for l, p in zip(labels, scalars)) / n
        brier_binary = sum(((p) - (1.0 if l else 0.0)) ** 2
                           for l, p in zip(labels, binaries)) / n
        summary.append({
            "alpha": a, "n": n,
            "n_sol_fav": n_sf, "n_def_fav": n_df,
            "sol_hold_scalar": sol_hold_scalar, "sol_hold_scalar_se": _se(sol_hold_scalar, n_sf),
            "sol_hold_binary": sol_hold_binary, "sol_hold_binary_se": _se(sol_hold_binary, n_sf),
            "def_stop": def_stop, "def_stop_se": _se(def_stop, n_df),
            "value_auc_scalar": auc_scalar, "value_auc_binary": auc_binary,
            "value_mean_solfav_scalar": m_sf_s, "value_mean_deffav_scalar": m_df_s,
            "value_mean_solfav_binary": m_sf_b, "value_mean_deffav_binary": m_df_b,
            "brier_scalar": brier_scalar, "brier_binary": brier_binary,
        })

    print(f"{'α':>5} {'n_sf':>5} {'n_df':>5} "
          f"{'sol_hold_scalar':>15} {'sol_hold_binary':>15} {'def_stop':>15} "
          f"{'AUC_s':>7} {'AUC_b':>7} {'Br_s':>6} {'Br_b':>6}")
    for r in summary:
        print(f"{r['alpha']:>5} {r['n_sol_fav']:>5} {r['n_def_fav']:>5} "
              f"{r['sol_hold_scalar']:>7.3f}±{r['sol_hold_scalar_se']:.3f} "
              f"{r['sol_hold_binary']:>7.3f}±{r['sol_hold_binary_se']:.3f} "
              f"{r['def_stop']:>7.3f}±{r['def_stop_se']:.3f} "
              f"{r['value_auc_scalar']:>7.3f} {r['value_auc_binary']:>7.3f} "
              f"{r['brier_scalar']:>6.3f} {r['brier_binary']:>6.3f}")

    out = {
        "contract": _CONTRACT, "threshold": _THRESHOLD,
        "n_per_alpha": N, "pimc_n": PIMC_N,
        "n_workers": N_WORKERS, "seed_base": SEED_BASE,
        "wall_s": wall_total, "rows": summary,
    }
    (OUT_DIR / "results_parti.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'results_parti.json'}")


if __name__ == "__main__":
    main()
