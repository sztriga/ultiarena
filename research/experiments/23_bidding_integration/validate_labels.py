"""Cross-check gen_base_events labels against the trusted oracle (exp 22).

For each kept deal: compute the solve-at-root god label, THEN play the optimal
(minimax) principal variation to the end and score it with `scoring/oracle`.
The label must equal the oracle's made-verdict — if they disagree, the labeling
mechanism (marriage seeding / threshold / objective) is wrong.

Also reports keep-rate and positive rate so we can judge dealer/alpha before a
long run. Single-process (deterministic, small N).

Env: HEAD (reach100_40|reach100_20|duri_colored), CAND, ALPHA.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, "/Users/milansimity/Cuccok/kodok/oldtawer")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.eval.dojo import deal_ulti_biased, deal_durchmars_colored
from trickster._solver_core import set_multi_weights
from trickster.games.ulti.game import soloist_won_durchmars
from ulti.scoring.oracle import BidSet, score as score_oracle
from recipe_local import sol_marriages

HEAD = os.environ.get("HEAD", "reach100_40")
CAND = int(os.environ.get("CAND", "1500"))
ALPHA = float(os.environ.get("ALPHA", "0.0"))

_CFG = {
    "reach100_40": dict(dealer="ulti", need="has_40", restrict="40",
                        solver="multi", bidkw={"forty_hundred": True},
                        key="40_100", keep_marriage=True),
    "reach100_20": dict(dealer="ulti", need="has_20", restrict="20",
                        solver="multi", bidkw={"twenty_hundred": True},
                        key="20_100", keep_marriage=True),
    "duri_colored": dict(dealer="duri", need=None, restrict=None,
                         solver="durchmars", bidkw={"durchmars": True},
                         key="durchmars", keep_marriage=False),
}


def _deal(kind, seed):
    return (deal_ulti_biased(seed=seed, alpha=ALPHA) if kind == "ulti"
            else deal_durchmars_colored(seed=seed, alpha=ALPHA))


def _oracle_made(final_pos, cfg):
    pv = score_oracle(final_pos=final_pos, bid=BidSet(**cfg["bidkw"]))
    return pv.components.get(cfg["key"], 0) > 0


def main():
    cfg = _CFG[HEAD]
    if cfg["solver"] == "multi":
        set_multi_weights(score_geq_100=1.0)

    kept = pos_lbl = agree = disagree = 0
    examples = []
    for i in range(CAND):
        seed = 770000000 + i
        deal = _deal(cfg["dealer"], seed)
        has_40, has_20 = sol_marriages(deal.sol_hand, deal.trump)
        if cfg["need"] == "has_40" and not has_40:
            continue
        if cfg["need"] == "has_20" and not has_20:
            continue
        sol12 = list(deal.sol_hand) + list(deal.talon)
        if len(sol12) != 12:
            continue
        rng = random.Random(seed ^ 0xA5A5A5A5)
        pool = ([j for j, c in enumerate(sol12) if c.rank not in ("king", "upper")]
                if cfg["keep_marriage"] else list(range(12)))
        if len(pool) < 2:
            continue
        idx = set(rng.sample(pool, 2))
        remaining = [sol12[j] for j in range(12) if j not in idx]
        discard = [sol12[j] for j in idx]

        pos = pis.build_position(
            hands=[remaining, list(deal.def1_hand), list(deal.def2_hand)],
            soloist=0, leader=0,
            contract="parti" if cfg["solver"] == "multi" else "durchmars",
            trump=deal.trump, talon=discard, declare_marriages=True,
            marriage_restrict=cfg["restrict"],
        )
        label = 1 if god_says_soloist_wins(pos, contract=cfg["solver"]) else 0

        # play the minimax principal variation, score the terminal with the oracle
        pv = pis.principal_variation(pos, contract=cfg["solver"])
        sim = pis.build_position(
            hands=[remaining, list(deal.def1_hand), list(deal.def2_hand)],
            soloist=0, leader=0,
            contract="parti" if cfg["solver"] == "multi" else "durchmars",
            trump=deal.trump, talon=discard, declare_marriages=True,
            marriage_restrict=cfg["restrict"],
        )
        for _pid, card in pv:
            if pis.is_terminal(sim):
                break
            pis.apply_move(sim, card)
        if cfg["solver"] == "durchmars":
            oracle_made = soloist_won_durchmars(sim)
        else:
            oracle_made = _oracle_made(sim, cfg)

        kept += 1
        pos_lbl += label
        if bool(label) == bool(oracle_made):
            agree += 1
        else:
            disagree += 1
            if len(examples) < 6:
                examples.append((seed, label, oracle_made))

    print(f"=== validate {HEAD}  ALPHA={ALPHA}  CAND={CAND} ===")
    print(f"  kept {kept}/{CAND}  (keep-rate {kept/CAND:.3f})")
    if kept:
        print(f"  positive rate (label): {pos_lbl/kept:.3f}")
        print(f"  label == oracle-made:  {agree}/{kept}  ({agree/kept:.4f})")
        if disagree:
            print(f"  DISAGREE {disagree}:  {examples}")
        else:
            print("  PASS — every label matches the oracle's made-verdict")
    return 0 if (kept and disagree == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
