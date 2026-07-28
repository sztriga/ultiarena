"""
Hierarchical training for the 40-100 subgame.

--load-play is REQUIRED.  Warm-start from a Parti checkpoint (same obs/action
space, MaskablePPO.load() works directly).

Phase B — Pre-game agent: TRUMP (suit 4-7) + DISCARD × 2.  Obs: 200-dim.  Action: Discrete(32).
Phase A — Play agent:     PLAY phase only.  Obs: 200-dim.  Action: Discrete(32).

Typical usage
-------------
    python train_40100.py --load-play models/parti_best --save models/40100

    python train_40100.py \\
        --load-play    models/40100_play_best \\
        --load-pregame models/40100_pregame_best \\
        --save         models/40100

    python train_40100.py --eval-only \\
        --load-play    models/40100_play_best \\
        --load-pregame models/40100_pregame_best
"""
from envs.subgames import (
    NegyvenszazPlayEnv,
    NegyvenszazPreGameEnv,
    heuristic_trump_selector,
)
from training.loop import make_hierarchical_parser, train_hierarchical


def main():
    args = make_hierarchical_parser(
        description='Hierarchical 40-100 training',
        default_save='models/40100',
    ).parse_args()

    train_hierarchical(
        args,
        play_env_cls=NegyvenszazPlayEnv,
        pregame_env_cls=NegyvenszazPreGameEnv,
        heuristic_selector=heuristic_trump_selector,
        trump_lo=4,
        trump_hi=7,
        tb_prefix='40100',
        description='Hierarchical 40-100 training',
        silent_ulti=True,
    )


if __name__ == '__main__':
    main()
