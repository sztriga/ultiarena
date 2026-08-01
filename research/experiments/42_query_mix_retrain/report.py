"""exp42 gate report: lone-candidate GP vs the frontier + contract-mix deltas."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def h2h():
    rows = [json.loads(l) for l in open(HERE / "mu_h2h_CAND_FRONTIER.jsonl")]
    lone_gp = []          # the lone config's GP per game (CAND alone at the table)
    per_contract = defaultdict(list)
    passes = 0
    for r in rows:
        for g in r["games"]:
            cfgs = g["seat_cfgs"]
            lone_seat = cfgs.index("CAND") if cfgs.count("CAND") == 1 else None
            if lone_seat is None:
                continue
            if g.get("pass"):
                passes += 1
                lone_gp.append(0.0)         # all-pass: symmetric, count as 0
                continue
            gp = g["seat_gp"][lone_seat]
            lone_gp.append(gp)
            per_contract[g["contract"]].append(gp)
    n = len(lone_gp)
    mean = sum(lone_gp) / n
    var = sum((x - mean) ** 2 for x in lone_gp) / max(1, n - 1)
    se = math.sqrt(var / n)
    print(f"GATE  lone-CAND GP/game vs 2×FRONTIER: {mean:+.3f} ± {se:.3f}  "
          f"(n={n} games, t={mean/se:.1f}, all-pass {passes})")
    print(f"\n{'contract':24s} {'n':>5s} {'CAND-lone GP':>13s}")
    for c, v in sorted(per_contract.items(), key=lambda kv: -abs(sum(kv[1]))):
        print(f"{c:24s} {len(v):5d} {sum(v)/len(v):+13.2f}")


def selfmix(tag):
    p = HERE / f"mu_self_{tag}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    mix = defaultdict(lambda: [0, 0.0])
    total = 0
    for r in rows:
        for g in r["games"]:
            total += 1
            if g.get("pass"):
                mix["passz"][0] += 1
                continue
            mix[g["contract"]][0] += 1
            mix[g["contract"]][1] += g["seat_gp"][g["winner"]]
    return total, mix


def contracts():
    a, b = selfmix("CAND"), selfmix("FRONTIER")
    if not (a and b):
        print("\n(self runs incomplete)"); return
    (na, ma), (nb, mb) = a, b
    print(f"\n{'contract':24s} {'FRONTIER %':>10s} {'CAND %':>8s} {'FRONTIER solGP':>15s} {'CAND solGP':>11s}")
    for c in sorted(set(ma) | set(mb), key=lambda c: -(mb.get(c, [0, 0])[0])):
        fa, fb = ma.get(c, [0, 0.0]), mb.get(c, [0, 0.0])
        fgp = fb[1] / fb[0] if fb[0] else 0.0
        cgp = fa[1] / fa[0] if fa[0] else 0.0
        print(f"{c:24s} {fb[0]/nb:10.1%} {fa[0]/na:8.1%} {fgp:+15.2f} {cgp:+11.2f}")


if __name__ == "__main__":
    h2h()
    contracts()
