"""
Hierarchical training for the 40-100+Ulti composite subgame.

--load-play is REQUIRED.  Warm-start from a 40-100 or Ulti play checkpoint
(same obs/action space, MaskablePPO.load() works directly).

Phase B — Pre-game agent: TRUMP (suit 12-15) + DISCARD × 2.  Obs: 200-dim.  Action: Discrete(32).
Phase A — Play agent:     PLAY phase only.  Obs: 200-dim.  Action: Discrete(32).

Win condition: BOTH _declarer_ulti AND trick_scores[0] >= 60.
Curated hands require a 7 AND a king+upper pair from the same suit.

Typical usage
-------------
    python train_ulti40100.py --load-play models/40100_play_best --save models/ulti40100

    python train_ulti40100.py \\
        --load-play    models/ulti40100_play_best \\
        --load-pregame models/ulti40100_pregame_best \\
        --save         models/ulti40100

    python train_ulti40100.py --eval-only \\
        --load-play    models/ulti40100_play_best \\
        --load-pregame models/ulti40100_pregame_best
"""
from envs.subgames import (
    UltiNegyvenszazPlayEnv,
    UltiNegyvenszazPreGameEnv,
    heuristic_combo_selector,
)
from training.loop import make_hierarchical_parser, train_hierarchical


def main():
    args = make_hierarchical_parser(
        description='Hierarchical 40-100+Ulti training',
        default_save='models/ulti40100',
    ).parse_args()

    train_hierarchical(
        args,
        play_env_cls=UltiNegyvenszazPlayEnv,
        pregame_env_cls=UltiNegyvenszazPreGameEnv,
        heuristic_selector=heuristic_combo_selector,
        trump_lo=12,
        trump_hi=15,
        tb_prefix='ulti40100',
        description='Hierarchical 40-100+Ulti training',
    )


if __name__ == '__main__':
    main()
