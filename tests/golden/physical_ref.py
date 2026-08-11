"""Physical-equivalence probes for the suit re-encoding (milan 2026-08-11).

Cards are identified by NAME (suit, rank) — never by id — so the same physical
deal can be probed before and after the wire encoding changes. For each deal:

  * every bidding head's calibrated probability (nets + isotonic),
  * exact solver values (betli + the multi/parti objective),
  * the oracle's scoring of a deterministic playout,
  * the betli-defense net's policy vector.

Identical numbers (float-tolerance for nets — weight-column permutation reorders
GEMM summation; exact for solver values and oracle GP) prove the re-encode +
weight surgery changed REPRESENTATION only, not one decision.

Run:  python tests/golden/physical_ref.py capture <out.json>
      python tests/golden/physical_ref.py verify  <ref.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

N_DEALS = 10


def _name(c) -> str:
    return f"{c.suit}:{c.rank}"


def _card(name: str):
    from ulti.card import Card
    suit, rank = name.split(":")
    return Card(suit=suit, rank=rank)


def _physical_deals():
    """Deterministic PHYSICAL deals: dealt under whatever encoding is live, then
    normalized to name-sorted hands so both encodings see identical cards."""
    from ulti.bidding.deal import deal_12_10_10
    deals = []
    for seed in range(1, N_DEALS + 1):
        sol12, d1, d2 = deal_12_10_10(seed * 7919)
        # names sorted lexically = encoding-independent canonical form
        deals.append({
            "sol10": sorted(_name(c) for c in sol12[:10]),
            "talon": sorted(_name(c) for c in sol12[10:]),
            "d1": sorted(_name(c) for c in d1),
            "d2": sorted(_name(c) for c in d2),
        })
    return deals


def _probe(deal: dict) -> dict:
    from ulti.bidding.provider import NetProvider
    from ulti.betli import defense as betli_defense
    from ulti.solvers import pis as pis_bridge

    global _PROV
    if "_PROV" not in globals():
        _PROV = NetProvider(calibrate=True,
                            betli_real_dir=str(Path(_REPO) / "models/ulti/betli"))
    sol10 = [_card(n) for n in deal["sol10"]]
    d1 = [_card(n) for n in deal["d1"]]
    d2 = [_card(n) for n in deal["d2"]]
    talon = [_card(n) for n in deal["talon"]]

    out: dict = {"heads": {}}
    # 1) all bidding heads, for two trumps (piros + a non-piros)
    for trump in ("hearts", "bells"):
        bp = _PROV.base_probs(sol10, trump)
        out["heads"][trump] = {
            "parti": bp.p_parti, "ulti": bp.p_ulti,
            "r40": bp.p_reach100_40, "r20": bp.p_reach100_20,
            "duri_c": bp.p_duri_colored, "betli": bp.p_betli,
            "duri_cl": bp.p_duri_colorless,
        }

    # 2) exact solver values: betli, and the multi objective on a parti build
    pos_b = pis_bridge.build_position(hands=[sol10, d1, d2], soloist=0, leader=0,
                                      contract="betli", trump=None, talon=talon,
                                      declare_marriages=False)
    _mv, v_betli = pis_bridge.solve_best(pos_b, contract="betli")
    pos_p = pis_bridge.build_position(hands=[sol10, d1, d2], soloist=0, leader=0,
                                      contract="parti", trump="hearts", talon=talon,
                                      declare_marriages=True)
    _mv2, v_parti = pis_bridge.solve_best(pos_p, contract="parti")
    out["solver"] = {"betli": float(v_betli), "parti": float(v_parti)}

    # 3) oracle scoring of a deterministic playout (first legal by NAME order)
    from ulti.bidding.ladder import overcalls, contract_name
    bid = None
    for r in overcalls(None):
        for b in r.bids:
            if contract_name(b) == "piros parti":
                bid = b
    pos_s = pis_bridge.build_position(hands=[sol10, d1, d2], soloist=0, leader=0,
                                      contract="parti", trump="hearts", talon=talon,
                                      declare_marriages=True)
    while not pis_bridge.is_terminal(pos_s):
        legal = sorted(pis_bridge.legal_actions(pos_s), key=_name)
        pis_bridge.apply_move(pos_s, legal[0])
    from ulti.scoring.oracle import score as oracle_score
    pvec = oracle_score(final_pos=pos_s, bid=bid, kontras={})
    out["oracle"] = {"total_sol": float(pvec.total_sol),
                     "gp0": float(pvec.gp_vs(0)), "gp1": float(pvec.gp_vs(1))}

    # 4) betli-defense net: policy over the first defender's opening choice
    pos_d = pis_bridge.build_position(hands=[sol10, d1, d2], soloist=0, leader=0,
                                      contract="betli", trump=None, talon=talon,
                                      declare_marriages=False)
    first = sorted(pis_bridge.legal_actions(pos_d), key=_name)[0]
    pis_bridge.apply_move(pos_d, first)                      # soloist leads; def1 to move
    pick = (betli_defense.betli_defense_pick(pos_d, viewer=1)
            if betli_defense.available() else None)
    out["betli_def"] = {_name(pick): 1.0} if pick is not None else {}
    return out


def capture(path: str) -> None:
    deals = _physical_deals()
    ref = {"deals": deals, "probes": [_probe(d) for d in deals]}
    json.dump(ref, open(path, "w"), indent=0, sort_keys=True)
    print(f"captured {len(deals)} physical probes -> {path}")


def verify(path: str) -> None:
    ref = json.load(open(path))
    bad = 0
    for i, (deal, want) in enumerate(zip(ref["deals"], ref["probes"])):
        got = _probe(deal)
        for trump in want["heads"]:
            for head, v in want["heads"][trump].items():
                g = got["heads"][trump][head]
                if v is None or g is None:
                    ok = (v is None) == (g is None)
                else:
                    ok = abs(g - v) < 1e-4
                if not ok:
                    print(f"deal {i} head {head}/{trump}: {v} -> {g}"); bad += 1
        for k, v in want["solver"].items():
            if abs(got["solver"][k] - v) > 1e-9:
                print(f"deal {i} solver {k}: {v} -> {got['solver'][k]}"); bad += 1
        for k, v in want["oracle"].items():
            if abs(got["oracle"][k] - v) > 1e-9:
                print(f"deal {i} oracle {k}: {v} -> {got['oracle'][k]}"); bad += 1
        for name, p in want["betli_def"].items():
            g = got["betli_def"].get(name)
            if g is None or abs(g - p) > 1e-4:
                print(f"deal {i} betli_def {name}: {p} -> {g}"); bad += 1
    print("PHYSICAL EQUIVALENCE: OK" if bad == 0 else f"MISMATCHES: {bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    {"capture": capture, "verify": verify}[mode](path)
