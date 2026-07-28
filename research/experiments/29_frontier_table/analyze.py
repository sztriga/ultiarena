"""exp29 — build the frontier self-play table from selfplay.jsonl.

Contract frequency + avg soloist GP + made/kontra rates; passz; per-seat (positional)
analytics; bleeding detection. Seat 0 = forehand/opener. Writes TABLE.md + prints.
"""
import json
import os
import sys
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "selfplay.jsonl")


def main():
    recs = [json.loads(l) for l in open(OUT)]
    N = len(recs)
    played = [r for r in recs if not r.get("pass")]
    passes = [r for r in recs if r.get("pass")]
    L = []

    def P(s=""):
        L.append(s)

    P(f"# Frontier self-play — {N} deals (3 frontier models bid + play)\n")
    P("KONTRA-aware bidder (opener passes weak hands), full auction (any seat may open), "
      "PIMC play, promoted per-unit kontra, oracle scoring incl. silents. Seat 0 = "
      "forehand/opener.\n")

    # ── contract frequency + soloist GP ──────────────────────────────────────
    by = collections.defaultdict(lambda: {"n": 0, "sol": 0.0, "pdef": 0.0, "made": 0,
                                           "kontra": 0, "nb": 0, "seats": collections.Counter()})
    for r in played:
        b = by[r["contract"]]
        b["n"] += 1; b["sol"] += r["soloist_gp"]; b["pdef"] += r["per_def"]
        b["made"] += 1 if r["made"] else 0; b["kontra"] += 1 if r.get("kontra", 0) else 0
        b["nb"] += r["n_bids"]; b["seats"][r["winner"]] += 1
    P("## Contracts — frequency & soloist GP (sorted by frequency)\n")
    P("| contract | count | freq | avg soloist GP | made% | kontra% | avg /def | avg bids |")
    P("|---|---|---|---|---|---|---|---|")
    for c in sorted(by, key=lambda k: -by[k]["n"]):
        b = by[c]; n = b["n"]
        P(f"| {c} | {n} | {100*n/N:.1f}% | {b['sol']/n:+.2f} | {100*b['made']/n:.0f}% | "
          f"{100*b['kontra']/n:.0f}% | {b['pdef']/n:+.2f} | {b['nb']/n:.2f} |")
    P(f"| **passz** | {len(passes)} | {100*len(passes)/N:.1f}% | — | — | — | — | — |")

    # ── overall ──────────────────────────────────────────────────────────────
    sol_gp_all = sum(r["soloist_gp"] for r in played)
    made_all = sum(1 for r in played if r["made"])
    P(f"\n## Overall")
    P(f"- deals: {N} | played: {len(played)} ({100*len(played)/N:.0f}%) | "
      f"passz: {len(passes)} ({100*len(passes)/N:.0f}%)")
    P(f"- soloist made {100*made_all/max(1,len(played)):.0f}% of played contracts")
    P(f"- mean soloist GP across played contracts: {sol_gp_all/max(1,len(played)):+.2f}")
    nb = [r["n_bids"] for r in played]
    P(f"- auction: avg {sum(nb)/max(1,len(nb)):.2f} bids/played-deal; "
      f"{100*sum(1 for x in nb if x>1)/max(1,len(nb)):.0f}% were contested (overcalled)")

    # ── per-seat (positional) analytics ──────────────────────────────────────
    seat_sum = [0.0, 0.0, 0.0]; seat_sol = [0, 0, 0]; seat_sol_gp = [0.0, 0.0, 0.0]
    seat_def_gp = [0.0, 0.0, 0.0]; seat_def_n = [0, 0, 0]
    pass_payer = 0
    for r in recs:
        for s in (0, 1, 2):
            seat_sum[s] += r["seat_gp"][s]
        if r.get("pass"):
            pass_payer += 1
            continue
        w = r["winner"]; seat_sol[w] += 1; seat_sol_gp[w] += r["seat_gp"][w]
        for s in (0, 1, 2):
            if s != w:
                seat_def_gp[s] += r["seat_gp"][s]; seat_def_n[s] += 1
    P(f"\n## Per-seat (position) — seat 0 = forehand/opener\n")
    P("| seat | mean GP/deal | won bid (soloist) | GP as soloist | GP as defender |")
    P("|---|---|---|---|---|")
    names = {0: "P0 forehand", 1: "P1 middle", 2: "P2 rear"}
    for s in (0, 1, 2):
        P(f"| {names[s]} | {seat_sum[s]/N:+.3f} | {seat_sol[s]} ({100*seat_sol[s]/N:.0f}%) | "
          f"{seat_sol_gp[s]/max(1,seat_sol[s]):+.2f} | {seat_def_gp[s]/max(1,seat_def_n[s]):+.2f} |")
    P(f"\n- zero-sum check: seat means sum to {sum(seat_sum)/N:+.3f} (should be ~0)")
    P(f"- passz: seat 0 is the payer on all {pass_payer} passzes ({100*pass_payer/N:.0f}% of deals), "
      f"−{2*2:.0f} GP each → a structural forehand tax")

    # ── bleeding detection ───────────────────────────────────────────────────
    P(f"\n## Bleeding check\n")
    bleed = [(c, by[c]["sol"]/by[c]["n"], by[c]["n"]) for c in by if by[c]["sol"]/by[c]["n"] < 0]
    bleed.sort(key=lambda x: x[1])
    if bleed:
        P("Contracts where the soloist LOSES GP on average (negative avg soloist GP):")
        for c, gp, n in bleed:
            P(f"- **{c}**: {gp:+.2f} GP/deal over {n} deals "
              f"({'rare' if n < 30 else 'FREQUENT — a real leak' if n*abs(gp) > 100 else 'minor'})")
    else:
        P("- No contract has negative average soloist GP.")
    worst_seat = min((0, 1, 2), key=lambda s: seat_sum[s])
    P(f"\n- worst positional seat: {names[worst_seat]} at {seat_sum[worst_seat]/N:+.3f} GP/deal")
    # contracts a seat over-bids (bids often but low GP)
    P(f"- biggest GP contributors (contract × count × avg): ")
    contrib = sorted(by, key=lambda k: -by[k]["n"] * abs(by[k]["sol"]/by[k]["n"]))[:5]
    for c in contrib:
        b = by[c]; n = b["n"]
        P(f"    {c}: {n} deals × {b['sol']/n:+.2f} = {b['sol']:+.0f} total soloist GP")

    txt = "\n".join(L) + "\n"
    open(os.path.join(_HERE, "TABLE.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
