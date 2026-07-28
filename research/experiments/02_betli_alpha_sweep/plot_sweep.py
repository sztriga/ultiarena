"""Render sweep.json into sweep.png (win rate + timing vs α)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).parent
data = json.loads((OUT_DIR / "sweep.json").read_text())
rows = data["rows"]
alphas = [r["alpha"] for r in rows]
wr = [r["sol_win_rate"] * 100 for r in rows]
mean_s = [r["mean_s"] * 1000 for r in rows]      # ms
median_s = [r["median_s"] * 1000 for r in rows]  # ms
max_s = [r["max_s"] * 1000 for r in rows]        # ms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(alphas, wr, marker="o", color="#2c7fb8", linewidth=2)
ax1.set_xlabel("α (hand bias toward low cards)")
ax1.set_ylabel("soloist win rate (%)")
ax1.set_title(f"Soloist win rate vs α  (N={data['n_per_alpha']}/α)")
ax1.set_ylim(-5, 105)
ax1.grid(True, alpha=0.3)
ax1.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

ax2.plot(alphas, max_s, marker="^", label="max", color="#d7301f", linewidth=1.5)
ax2.plot(alphas, mean_s, marker="o", label="mean", color="#2c7fb8", linewidth=1.5)
ax2.plot(alphas, median_s, marker="s", label="median", color="#41ab5d", linewidth=1.5)
ax2.set_xlabel("α (hand bias toward low cards)")
ax2.set_ylabel("solve time (ms)")
ax2.set_title("Fast solver timing vs α")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

fig.suptitle("Betli fast solver — α sweep", fontsize=13)
fig.tight_layout()
out = OUT_DIR / "sweep.png"
fig.savefig(out, dpi=140)
print(f"saved → {out}")
