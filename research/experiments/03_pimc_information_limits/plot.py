"""Render exp 3 plots from the three result JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).parent
opening = json.loads((OUT / "opening_auc_sweep.json").read_text())
god_vs_k = json.loads((OUT / "auc_vs_cards_godplay.json").read_text())
pimc_vs_k = json.loads((OUT / "auc_vs_cards_pimcplay_n4.json").read_text())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: opening AUC vs N
ax = axes[0]
ns       = [r["n"] for r in opening["rows"]]
auc_sol  = [r["auc_sol"] for r in opening["rows"]]
auc_def  = [r["auc_def"] for r in opening["rows"]]
ax.plot(ns, auc_sol, "o-", color="#2c7fb8", label="soloist viewpoint")
ax.plot(ns, auc_def, "s-", color="#d7301f", label="defender viewpoint (avg)")
ax.set_xscale("log", base=2)
ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("PIMC samples N")
ax.set_ylabel("ROC-AUC vs god label")
ax.set_title("Opening AUC vs N  (α=0.7, 50 deals)")
ax.set_ylim(0.45, 1.0)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right")

# Panel 2: AUC vs K under god-optimal play
ax = axes[1]
rows = god_vs_k["rows"]
ks       = [r["k"] for r in rows]
auc_sol  = [r["auc_sol"] for r in rows]
auc_def  = [r["auc_def"] for r in rows]
ax.plot(ks, auc_sol, "o-", color="#2c7fb8", label="soloist viewpoint")
ax.plot(ks, auc_def, "s-", color="#d7301f", label="defender viewpoint (avg)")
ax.invert_xaxis()
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("cards in hand per player (K)")
ax.set_ylabel("ROC-AUC")
ax.set_title("AUC vs K — god-optimal play  (N=32)")
ax.set_ylim(0.45, 1.05)
ax.grid(True, alpha=0.3); ax.legend(loc="lower left")

# Panel 3: AUC vs K under PIMC-played trajectory
ax = axes[2]
rows = pimc_vs_k["auc_rows"]
ks       = [r["k"] for r in rows]
auc_sol  = [r["auc_sol"] for r in rows]
auc_def  = [r["auc_def"] for r in rows]
ax.plot(ks, auc_sol, "o-", color="#2c7fb8", label="soloist viewpoint")
ax.plot(ks, auc_def, "s-", color="#d7301f", label="defender viewpoint (avg)")
ax.invert_xaxis()
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("cards in hand per player (K)")
ax.set_ylabel("ROC-AUC")
ax.set_title(f"AUC vs K — PIMC-played  (N={pimc_vs_k['n_samples']})")
ax.set_ylim(0.40, 1.05)
ax.grid(True, alpha=0.3); ax.legend(loc="lower left")

fig.suptitle("Exp 3 — PIMC information limits at α=0.7", fontsize=13)
fig.tight_layout()
out = OUT / "exp3_summary.png"
fig.savefig(out, dpi=140)
print(f"saved → {out}")
