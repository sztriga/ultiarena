"""
Hierarchical training for the Ulti subgame.

--load-play is REQUIRED.  Warm-start from a Parti checkpoint (same obs/action
space, MaskablePPO.load() works directly).

Phase B — Pre-game agent: TRUMP (suit 8-11) + DISCARD × 2.  Obs: 200-dim.  Action: Discrete(32).
Phase A — Play agent:     PLAY phase only.  Obs: 200-dim.  Action: Discrete(32).

Typical usage
-------------
    python train_ulti.py --load-play models/parti_best --save models/ulti

    python train_ulti.py \\
        --load-play    models/ulti_play_best \\
        --load-pregame models/ulti_pregame_best \\
        --save         models/ulti

    python train_ulti.py --eval-only \\
        --load-play    models/ulti_play_best \\
        --load-pregame models/ulti_pregame_best
"""
from envs.subgames import (
    UltiPlayEnv,
    UltiPreGameEnv,
    heuristic_ulti_selector,
)
from ulti.training.loop import make_hierarchical_parser, train_hierarchical


def main():
    args = make_hierarchical_parser(
        description='Hierarchical Ulti training',
        default_save='models/ulti',
    ).parse_args()

    train_hierarchical(
        args,
        play_env_cls=UltiPlayEnv,
        pregame_env_cls=UltiPreGameEnv,
        heuristic_selector=heuristic_ulti_selector,
        trump_lo=8,
        trump_hi=11,
        tb_prefix='ulti',
        description='Hierarchical Ulti training',
        silent_40100=True,
    )


if __name__ == '__main__':
    main()
