"""Baseline: PIMC32 pickup vs V-net pickup, same play layer.

Per deal:
  1. Compute every (discard, contract, trump) EV via PIMC32 AND v-net.
  2. PIMC pickup = best by PIMC EV (or pass if best < 0).
  3. V-net pickup = best by v-net EV (or pass if best < 0).
  4. Play out each chosen bid with sol=PIMC32, defs=god solver.
  5. Score via oracle. Report side-by-side.

Goal: isolate pickup quality. Play layer is identical for both → any
GP delta is purely the pickup-decision sacrifice.
"""
from __future__ import annotations

import itertools, random, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, _ev_per_def
from solvers import pis, pimc as _pimc, determinize as _det
from eval.pimc_matchup import god_pick
from ulti.card import SUITS
from _vlib import CONTRACT_CONFIGS, PickupNet, featurize, input_dim, weights_path, EXP_DIR
from vnet.pickup import PickupNetV2
from scoring.oracle import BidSet, score as score_oracle

N         = 300
PIMC_N    = 32
SEED_BASE = 100_000
N_WORKERS = 8
USE_V2    = True   # True → load *_vnet_v2.pt + PickupNetV2(256,128)


def _weights_path(name: str):
    return EXP_DIR / (f"{name}_vnet_v2.pt" if USE_V2 else f"{name}_vnet.pt")


# ─── per-process v-net cache ────────────────────────────────────────
_NETS = None
def _load_nets():
    global _NETS
    if _NETS is not None:
        return _NETS
    nets = {}
    for name, cfg in CONTRACT_CONFIGS.items():
        m = PickupNetV2(in_dim=input_dim(cfg)) if USE_V2 else PickupNet(in_dim=input_dim(cfg))
        m.load_state_dict(torch.load(_weights_path(name), weights_only=True))
        m.eval()
        nets[name] = m
    _NETS = nets
    return nets


def _ev_record(contract, piros, p):
    return _ev_per_def(contract, piros, p)


def _eval_one_deal_both(sol12, d1, d2, *, pimc_n, seed):
    """Return list of records: (discard, contract, trump, p_pimc, p_vnet,
    ev_pimc, ev_vnet). Each record represents one (discard × contract ×
    trump) option."""
    nets = _load_nets()
    records = []
    rng_seed = seed
    discards = list(itertools.combinations(sol12, 2))

    # Pre-batch v-net inputs per contract
    # For trump-aware contracts (parti/ulti): 66 discards × 4 trumps each
    # For trumpless (betli/durchmars): 66 discards each
    vnet_preds = {}  # key = (contract, trump_or_None) → np.ndarray of len 66
    for cname, cfg in CONTRACT_CONFIGS.items():
        if cfg.has_trump:
            for trump in SUITS:
                X = np.stack([
                    featurize([c for c in sol12 if c not in dp], trump, True)
                    for dp in discards
                ])
                with torch.no_grad():
                    vnet_preds[(cname, trump)] = nets[cname](torch.from_numpy(X)).numpy()
        else:
            X = np.stack([
                featurize([c for c in sol12 if c not in dp], None, False)
                for dp in discards
            ])
            with torch.no_grad():
                vnet_preds[(cname, None)] = nets[cname](torch.from_numpy(X)).numpy()

    for i, discard_pair in enumerate(discards):
        remaining = [c for c in sol12 if c not in discard_pair]
        talon = list(discard_pair)

        # parti × 4 trumps + ulti × 4 trumps (ulti only if sol has trump-7)
        for trump in SUITS:
            piros = (trump == 'hearts')
            for cname in ('parti', 'ulti'):
                if cname == 'ulti':
                    has_t7 = any(c.suit == trump and c.rank == '7'
                                 for c in remaining)
                    if not has_t7:
                        records.append((discard_pair, cname, trump,
                                        0.0, 0.0, _ev_record(cname, piros, 0.0),
                                        _ev_record(cname, piros, 0.0)))
                        continue
                pos = pis.build_position(
                    hands=[remaining, d1, d2], soloist=0, leader=0,
                    contract=cname, trump=trump, talon=talon,
                )
                rng_seed += 1
                _, avg = _pimc.pimc_decision(
                    true_pos=pos, contract=cname, n_samples=pimc_n, seed=rng_seed,
                )
                p_pimc = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
                p_vnet = float(vnet_preds[(cname, trump)][i])
                records.append((discard_pair, cname, trump, p_pimc, p_vnet,
                                _ev_record(cname, piros, p_pimc),
                                _ev_record(cname, piros, p_vnet)))

        # betli + durchmars (trumpless)
        for cname in ('betli', 'durchmars'):
            pos = pis.build_position(
                hands=[remaining, d1, d2], soloist=0, leader=0,
                contract=cname, trump=None, talon=talon,
            )
            rng_seed += 1
            _, avg = _pimc.pimc_decision(
                true_pos=pos, contract=cname, n_samples=pimc_n, seed=rng_seed,
            )
            p_pimc = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
            p_vnet = float(vnet_preds[(cname, None)][i])
            records.append((discard_pair, cname, None, p_pimc, p_vnet,
                            _ev_record(cname, False, p_pimc),
                            _ev_record(cname, False, p_vnet)))
    return records


def _best_from(records, ev_idx):
    """Pick max EV. Returns None if best < 0 (pass)."""
    if not records:
        return None
    best = max(records, key=lambda r: r[ev_idx])
    if best[ev_idx] < 0:
        return None
    return best


# ─── Play layer: sol=PIMC32, defs=god ───────────────────────────────
def _play_pimc_vs_god(*, sol10, d1, d2, talon, contract, trump, pimc_n, seed):
    pos = pis.build_position(
        hands=[sol10, d1, d2], soloist=0, leader=0,
        contract=contract, trump=trump, talon=talon,
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        if p == 0:
            chosen, avg = _pimc.pimc_decision(
                true_pos=pos, contract=contract, n_samples=pimc_n,
                seed=seed * 31337 + move_i, voids=voids_dict,
            )
        else:
            chosen = god_pick(pos=pos, contract=contract)
        if chosen is None:
            chosen = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis.apply_move(pos, chosen)
        move_i += 1
    return pos


def _score(pos, *, contract, piros):
    bid = BidSet(
        parti     = (contract == 'parti'),
        ulti      = (contract == 'ulti'),
        durchmars = (contract == 'durchmars'),
        betli     = (contract == 'betli'),
        piros     = piros,
    )
    return score_oracle(
        final_pos=pos, bid=bid,
        score_parti=(contract == 'parti'),
        silents=False,
    ).total_per_def


def worker(seed):
    sol12, d1, d2 = deal_12_10_10(seed)
    t0 = time.perf_counter()
    recs = _eval_one_deal_both(sol12, d1, d2, pimc_n=PIMC_N, seed=seed * 17)
    eval_wall = time.perf_counter() - t0

    # Indices: 5 = ev_pimc, 6 = ev_vnet
    best_pimc = _best_from(recs, 5)
    best_vnet = _best_from(recs, 6)

    out = {'seed': seed, 'eval_wall': eval_wall}

    def _make_result(best, key):
        if best is None:
            out[f'{key}_pass'] = True
            out[f'{key}_contract'] = None
            out[f'{key}_trump'] = None
            out[f'{key}_pred_ev'] = 0.0
            out[f'{key}_actual_gp'] = 0.0
            out[f'{key}_p'] = 0.0
            return 0.0
        discard, contract, trump, p_pimc, p_vnet, ev_p, ev_v = best
        piros = (trump == 'hearts')
        remaining = [c for c in sol12 if c not in discard]
        talon = list(discard)
        t1 = time.perf_counter()
        final = _play_pimc_vs_god(
            sol10=remaining, d1=d1, d2=d2, talon=talon,
            contract=contract, trump=trump, pimc_n=PIMC_N,
            seed=seed * 919,
        )
        play_wall = time.perf_counter() - t1
        actual = _score(final, contract=contract, piros=piros)
        out[f'{key}_pass'] = False
        out[f'{key}_contract'] = contract
        out[f'{key}_trump'] = trump
        out[f'{key}_pred_ev'] = ev_v if key == 'vnet' else ev_p
        out[f'{key}_actual_gp'] = actual
        out[f'{key}_p'] = p_vnet if key == 'vnet' else p_pimc
        return play_wall

    pwall = _make_result(best_pimc, 'pimc')
    vwall = _make_result(best_vnet, 'vnet')
    out['play_wall'] = pwall + vwall
    return out


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m)**2 for x in xs) / n
    return m, v**0.5


def _print_side(rows, prefix):
    n = len(rows)
    n_pass = sum(1 for r in rows if r[f'{prefix}_pass'])
    bid_rows = [r for r in rows if not r[f'{prefix}_pass']]
    if bid_rows:
        pe_m, _ = _mean_std([r[f'{prefix}_pred_ev'] for r in bid_rows])
        ag_m, _ = _mean_std([r[f'{prefix}_actual_gp'] for r in bid_rows])
    else:
        pe_m = ag_m = 0.0
    overall = sum(r[f'{prefix}_actual_gp'] for r in rows) / n
    print(f"  {prefix.upper():>5}  pass={n_pass}/{n} ({n_pass/n*100:.1f}%)  "
          f"pred_ev_bids={pe_m:+.3f}  actual_bids={ag_m:+.3f}  "
          f"GP/deal={overall:+.3f}  sol_total={overall*2*n:+.1f}")


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(worker, seeds):
            rows.append(r)
            if len(rows) % 25 == 0:
                wall = time.perf_counter() - t0
                print(f"  {len(rows)}/{N}  wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    print()
    print(f"=== Pickup baseline: PIMC32 vs V-net (N={N}, sol=PIMC32, def=god) ===")
    _print_side(rows, 'pimc')
    _print_side(rows, 'vnet')

    # Agreement on pickup
    agree = 0
    for r in rows:
        sp = (r['pimc_pass'], r['pimc_contract'], r['pimc_trump'])
        sv = (r['vnet_pass'], r['vnet_contract'], r['vnet_trump'])
        if sp == sv:
            agree += 1
    print(f"  pickup agreement: {agree}/{N} ({agree/N*100:.1f}%)")

    # Per-contract breakdown for each picker
    for prefix in ('pimc', 'vnet'):
        print()
        print(f"=== {prefix.upper()} per-contract performance ===")
        bd = Counter()
        for r in rows:
            if r[f'{prefix}_pass']:
                bd['pass'] += 1
            else:
                bd[f"{r[f'{prefix}_contract']}/{r[f'{prefix}_trump'] or 'colorless'}"] += 1
        print(f"  {'bid':>22}  {'n':>4}  {'pred EV':>8}  {'actual':>8}  "
              f"{'won':>4}  {'won %':>7}")
        for k in sorted(bd, key=lambda x: -bd[x]):
            if k == 'pass':
                print(f"  {'pass':>22}  {bd[k]:>4}")
                continue
            sub = [r for r in rows if not r[f'{prefix}_pass']
                   and f"{r[f'{prefix}_contract']}/{r[f'{prefix}_trump'] or 'colorless'}" == k]
            pe = sum(r[f'{prefix}_pred_ev'] for r in sub) / len(sub)
            ag = sum(r[f'{prefix}_actual_gp'] for r in sub) / len(sub)
            won = sum(1 for r in sub if r[f'{prefix}_actual_gp'] > 0)
            print(f"  {k:>22}  {len(sub):>4}  {pe:>+7.2f}  {ag:>+7.2f}  "
                  f"{won:>4}  {won/len(sub)*100:>6.1f}%")

    # Disagreement deals (where pickups differ)
    print()
    print("=== Deals where PIMC and V-net disagreed ===")
    print(f"  {'seed':>8}  {'PIMC pick':>22}  {'pimc GP':>7}  "
          f"{'V-net pick':>22}  {'vnet GP':>7}  {'Δ(v-p)':>7}")
    n_disagree = 0; total_delta = 0.0
    for r in rows:
        sp = (r['pimc_pass'], r['pimc_contract'], r['pimc_trump'])
        sv = (r['vnet_pass'], r['vnet_contract'], r['vnet_trump'])
        if sp == sv:
            continue
        n_disagree += 1
        def fmt(pas, c, t):
            return 'pass' if pas else f"{c}/{t or 'colorless'}"
        pn = fmt(*sp); vn = fmt(*sv)
        pg = r['pimc_actual_gp']; vg = r['vnet_actual_gp']
        total_delta += (vg - pg)
        print(f"  {r['seed']:>8}  {pn:>22}  {pg:>+6.2f}  "
              f"{vn:>22}  {vg:>+6.2f}  {vg-pg:>+6.2f}")
    if n_disagree:
        print(f"\n  mean Δ (vnet - pimc) on disagreements: {total_delta/n_disagree:+.3f} GP/deal")

    em = sum(r['eval_wall'] for r in rows) / N
    pm = sum(r['play_wall'] for r in rows) / N
    print()
    print(f"=== Timing ===")
    print(f"  wall total: {wall:.0f}s ({wall/60:.1f} min)")
    print(f"  mean eval / deal: {em:.2f}s   (BOTH pimc + vnet predictions)")
    print(f"  mean play / deal: {pm:.2f}s   (2 plays per deal: 1 per picker)")


if __name__ == "__main__":
    main()
