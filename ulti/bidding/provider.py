"""NetProvider: a hand (+ chosen trump) → BaseProbs, by running the 7 baseline
value heads. This is what turns the composer into a real bidder.

Colored heads (parti, ulti, reach100_40, reach100_20, duri_colored) take the
36-dim hand+trump featurize; colorless heads (betli, colorless_duri) take the
32-dim hand featurize (trump-independent). reach100_40/20 are gated by actually
holding the marriage. Optional per-head isotonic calibration (load if present).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ulti.config import env_str, repo_root
from ulti.vnet.pickup import featurize
from ulti.bidding.recipe import sol_marriages
from ulti.bidding.base_head import Head

# consolidated deployed bidder heads (repo-root/models/ulti/bidding)
_MODELS_DIR = os.path.join(repo_root(), "models", "ulti", "bidding")
from ulti.bidding.bidder import BaseProbs               # noqa: E402

COLORED = ("parti", "ulti", "reach100_40", "reach100_20", "duri_colored")
COLORLESS = ("betli", "colorless_duri")


class NetProvider:
    def __init__(self, weights_dir=_MODELS_DIR, calibrate=False, betli_real_dir=None):
        self.heads = {}
        self.calib = {}
        for h in COLORED + COLORLESS:
            ckpt = torch.load(os.path.join(weights_dir, f"{h}_baseline.pt"),
                              weights_only=False)
            m = Head(ckpt["in_dim"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.heads[h] = m
            if calibrate:
                cpath = os.path.join(weights_dir, f"{h}_isotonic.npz")
                if os.path.exists(cpath):
                    d = np.load(cpath)
                    self.calib[h] = (d["x"], d["y"])
        # exp37: optional REALISTIC-defense betli head (drop-in, 32-dim colorless). When loaded,
        # `p_betli_real` is populated and the bidder (BETLI_REAL / betli_real=True) scores PLAIN
        # betli by it. Default (betli_real_dir=None) → not loaded → deployed behaviour unchanged.
        betli_real_dir = betli_real_dir or env_str("BETLI_REAL_DIR")
        if betli_real_dir:
            rp = os.path.join(betli_real_dir, "betli_real_baseline.pt")
            if os.path.exists(rp):
                ck = torch.load(rp, weights_only=False)
                m = Head(ck["in_dim"]); m.load_state_dict(ck["state_dict"]); m.eval()
                self.heads["betli_real"] = m
                if calibrate:
                    cp = os.path.join(betli_real_dir, "betli_real_isotonic.npz")
                    if os.path.exists(cp):
                        d = np.load(cp); self.calib["betli_real"] = (d["x"], d["y"])

    def _p(self, head, X):
        with torch.no_grad():
            v = float(self.heads[head](torch.from_numpy(X[None].astype(np.float32)))[0])
        if head in self.calib:
            xs, ys = self.calib[head]
            v = float(np.interp(v, xs, ys))
        return v

    def base_probs(self, hand10, trump) -> BaseProbs:
        xc = featurize(hand10, trump, True)        # 36-dim (colored)
        xb = featurize(hand10, None, False)        # 32-dim (colorless)
        has40, has20 = sol_marriages(hand10, trump)
        return BaseProbs(
            p_parti=self._p("parti", xc),
            p_ulti=self._p("ulti", xc),
            p_reach100_40=self._p("reach100_40", xc) if has40 else 0.0,
            p_reach100_20=self._p("reach100_20", xc) if has20 else 0.0,
            p_duri_colored=self._p("duri_colored", xc),
            p_betli=self._p("betli", xb),
            p_betli_real=(self._p("betli_real", xb) if "betli_real" in self.heads else 0.0),
            p_duri_colorless=self._p("colorless_duri", xb),
            has_40=has40, has_20=has20,
            trump_is_hearts=(trump == "hearts"),
        )


if __name__ == "__main__":
    from ulti.bidding.deal import deal_12_10_10
    prov = NetProvider()
    sol12, d1, d2 = deal_12_10_10(seed=7)
    for trump in ("hearts", "acorns"):
        pr = prov.base_probs(sol12[:10], trump)
        print(f"trump={trump}: parti={pr.p_parti:.2f} ulti={pr.p_ulti:.2f} "
              f"r100_40={pr.p_reach100_40:.2f} r100_20={pr.p_reach100_20:.2f} "
              f"duriC={pr.p_duri_colored:.2f} betli={pr.p_betli:.2f} "
              f"duri0={pr.p_duri_colorless:.2f} has40={pr.has_40} has20={pr.has_20}")
