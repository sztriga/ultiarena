"""
train_all.py — End-to-end Ulti RL training pipeline.

Trains all agents in dependency order and saves them into a single namespace
directory (models/{run}/).  Re-running with the same --run flag automatically
resumes from the last incomplete stage — each stage is skipped if its sentinel
checkpoint already exists.

Stages (in order)
-----------------
  parti_play          Parti declarer vs heuristic defenders
  defender_parti      Adversarial Parti defenders (B/A alternating)
  40100               40-100 licit + play agents (hierarchical)
  defender_40100      Adversarial 40-100 defenders
  ulti                Ulti licit + play agents (hierarchical)
  defender_ulti       Adversarial Ulti defenders
  ulti40100           40-100+Ulti licit + play agents (hierarchical)
  defender_ulti40100  Adversarial 40-100+Ulti defenders
  orchestrator        Orchestrator meta-agent

Resume behaviour
----------------
A stage is skipped when its sentinel file already exists, e.g.:

    models/default/parti_play_best.zip  →  parti_play done

If training is interrupted mid-stage, re-running resumes it because the
interrupted stage's sentinel is absent, while all completed stages are skipped.

To force-restart from a specific stage (regardless of sentinels), use
--from-stage.  Stages before it are always skipped; stages from it onward
respect their sentinels as normal.

To force-rerun a single stage that already has a sentinel, delete it:

    rm models/myrun/defender_parti_agent_best.zip

Usage
-----
    python train_all.py                                   # → models/default/
    python train_all.py --run v2                          # → models/v2/
    python train_all.py --run v2 --from-stage defender_40100
    python train_all.py --run v2 --from-stage orchestrator
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import uuid
from dataclasses import dataclass
from typing import Callable

# Maps composite stage name → script filename
COMPOSITE_SCRIPTS = {
    '40100':   'train_40100.py',
    'ulti':    'train_ulti.py',
    'ulti40100': 'train_ulti40100.py',
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _exists(ns: str, name: str) -> bool:
    """True if models/{ns}/{name}.zip is present on disk."""
    return os.path.exists(os.path.join(ns, name + '.zip'))


def _p(ns: str, name: str) -> str:
    """Full path for a model save prefix (no extension)."""
    return os.path.join(ns, name)


def _run(cmd: list) -> None:
    print(f"\n{'=' * 68}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 68}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStage failed (exit {result.returncode}).  "
              f"Fix the error and re-run train_all.py to resume.",
              file=sys.stderr)
        sys.exit(result.returncode)


def _best_defender(ns: str, subgame: str) -> str:
    """
    Return the best available trained defender for training a composite play agent.

    Prefers the most recently trained defender that exists on disk, falling back
    to earlier ones.  Parti defender is always the final fallback.
    """
    candidates = {
        '40100':     ['defender_parti_agent_best'],
        'ulti':      ['defender_40100_agent_best', 'defender_parti_agent_best'],
        'ulti40100': ['defender_ulti_agent_best',  'defender_40100_agent_best',
                      'defender_parti_agent_best'],
    }
    for name in candidates.get(subgame, []):
        if _exists(ns, name):
            return _p(ns, name)
    return _p(ns, 'defender_parti_agent_best')


def _best_pregame(ns: str, subgame: str) -> str:
    """
    Return the best available pre-game agent path for a subgame.

    Prefers the recalibrated pre-game agent produced by defender training when
    it exists, falling back to the base hierarchical-training checkpoint.
    """
    recalibrated = f'defender_{subgame}_pregame_best'
    if _exists(ns, recalibrated):
        return _p(ns, recalibrated)
    return _p(ns, f'{subgame}_pregame_best')


def _best_play(ns: str, subgame: str) -> str:
    """
    Return the best available play-agent path for a subgame.

    Prefers the Phase-A fine-tuned declarer from defender training when it
    exists, falling back to the base play/licit-trained checkpoint.
    """
    fine_tuned = f'defender_{subgame}_declarer_best'
    if _exists(ns, fine_tuned):
        return _p(ns, fine_tuned)
    base = 'parti_play_best' if subgame == 'parti' else f'{subgame}_play_best'
    return _p(ns, base)


# ──────────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────────

def run_parti_play(ns: str, args) -> None:
    cmd = [sys.executable, 'train_parti_play.py',
           '--steps',     str(args.parti_steps),
           '--n-envs',    str(args.n_envs),
           '--save',      _p(ns, 'parti_play'),
           '--patience',  str(args.patience),
           '--seed',      str(args.seed),
           '--tb-suffix', args.tb_uid]
    if _exists(ns, 'parti_play_best'):
        cmd += ['--load', _p(ns, 'parti_play_best')]
    _run(cmd)


def run_parti_pregame(ns: str, args) -> None:
    cmd = [sys.executable, 'train_parti_pregame.py',
           '--load-play',  _p(ns, 'parti_play_best'),
           '--steps',      str(args.parti_pregame_steps),
           '--n-envs',     str(args.n_envs),
           '--save',       _p(ns, 'parti'),
           '--patience',   str(args.patience),
           '--seed',       str(args.seed),
           '--tb-suffix',  args.tb_uid]
    if _exists(ns, 'parti_pregame_best'):
        cmd += ['--load-pregame', _p(ns, 'parti_pregame_best')]
    _run(cmd)


def run_defender_parti(ns: str, args) -> None:
    # Use Phase-A refined declarer as the frozen opponent if available
    load_decl = (_p(ns, 'defender_parti_declarer_best')
                 if _exists(ns, 'defender_parti_declarer_best')
                 else _p(ns, 'parti_play_best'))
    cmd = [sys.executable, 'train_defender.py',
           '--subgame',               'parti',
           '--load-declarer',         load_decl,
           '--save',                  _p(ns, 'defender_parti'),
           '--n-envs',                str(args.n_envs),
           '--turns',                 str(args.defender_turns),
           '--defender-steps-first',  str(args.defender_steps_first),
           '--defender-steps',        str(args.defender_steps),
           '--declarer-steps',        str(args.declarer_finetune_steps),
           '--patience',              str(args.patience),
           '--seed',                  str(args.seed),
           '--tb-suffix',             args.tb_uid]
    if _exists(ns, 'defender_parti_agent_best'):
        cmd += ['--load-defender', _p(ns, 'defender_parti_agent_best')]
    # Parti pregame agent handles discards for the frozen declarer in DefenderEnv
    if _exists(ns, 'defender_parti_pregame_best'):
        cmd += ['--load-pregame', _p(ns, 'defender_parti_pregame_best')]
    elif _exists(ns, 'parti_pregame_best'):
        cmd += ['--load-pregame', _p(ns, 'parti_pregame_best')]
    _run(cmd)


def _run_composite(ns: str, args, subgame: str) -> None:
    """Run a hierarchical (licit + play) training stage for a composite subgame."""
    script  = COMPOSITE_SCRIPTS[subgame]
    # Warm-start chain: resume from own checkpoint → previous subgame → parti
    _PLAY_WARMSTART = {'ulti40100': 'ulti_play_best'}
    if _exists(ns, f'{subgame}_play_best'):
        load_play = _p(ns, f'{subgame}_play_best')
    elif _exists(ns, _PLAY_WARMSTART.get(subgame, '')):
        load_play = _p(ns, _PLAY_WARMSTART[subgame])
    else:
        load_play = _p(ns, 'parti_play_best')
    cmd = [sys.executable, script,
           '--load-play',       load_play,
           '--load-def',        _best_defender(ns, subgame),
           '--save',            _p(ns, subgame),
           '--n-envs',          str(args.n_envs),
           '--n-pregame-envs',  str(args.n_envs),
           '--turns',           str(args.composite_turns),
           '--play-steps',      str(args.composite_play_steps),
           '--pregame-steps',   str(args.composite_pregame_steps),
           '--patience',        str(args.patience),
           '--seed',            str(args.seed),
           '--tb-suffix',       args.tb_uid]
    if _exists(ns, f'{subgame}_pregame_best') or _exists(ns, f'defender_{subgame}_pregame_best'):
        cmd += ['--load-pregame', _best_pregame(ns, subgame)]
    elif subgame == 'ulti40100' and (_exists(ns, 'defender_ulti_pregame_best')
                                     or _exists(ns, 'ulti_pregame_best')):
        cmd += ['--load-pregame', _best_pregame(ns, 'ulti')]
    elif _exists(ns, 'parti_pregame_best'):
        cmd += ['--load-pregame', _p(ns, 'parti_pregame_best')]
    _run(cmd)


def _run_composite_defender(ns: str, args, subgame: str) -> None:
    """Run adversarial defender training for a composite subgame."""
    load_decl = (_p(ns, f'defender_{subgame}_declarer_best')
                 if _exists(ns, f'defender_{subgame}_declarer_best')
                 else _p(ns, f'{subgame}_play_best'))
    cmd = [sys.executable, 'train_defender.py',
           '--subgame',              subgame,
           '--load-declarer',        load_decl,
           '--load-pregame',         _best_pregame(ns, subgame),
           '--save',                 _p(ns, f'defender_{subgame}'),
           '--n-envs',               str(args.n_envs),
           '--turns',                str(args.defender_turns),
           '--defender-steps-first', str(args.defender_steps_first),
           '--defender-steps',       str(args.defender_steps),
           '--declarer-steps',       str(args.declarer_finetune_steps),
           '--pregame-recal-steps',  str(args.pregame_recal_steps),
           '--patience',             str(args.patience),
           '--seed',                 str(args.seed),
           '--tb-suffix',            args.tb_uid]
    if _exists(ns, f'defender_{subgame}_agent_best'):
        cmd += ['--load-defender', _p(ns, f'defender_{subgame}_agent_best')]
    _run(cmd)


def run_40100(ns: str, args) -> None:
    _run_composite(ns, args, '40100')


def run_defender_40100(ns: str, args) -> None:
    _run_composite_defender(ns, args, '40100')


def run_ulti(ns: str, args) -> None:
    _run_composite(ns, args, 'ulti')


def run_defender_ulti(ns: str, args) -> None:
    _run_composite_defender(ns, args, 'ulti')


def run_ulti40100(ns: str, args) -> None:
    _run_composite(ns, args, 'ulti40100')


def run_defender_ulti40100(ns: str, args) -> None:
    _run_composite_defender(ns, args, 'ulti40100')


def run_orchestrator(ns: str, args) -> None:
    cmd = [sys.executable, 'train_orchestrator.py',
           '--load-parti',         _best_play(ns, 'parti'),
           '--load-40100',         _best_play(ns, '40100'),
           '--load-ulti',          _best_play(ns, 'ulti'),
           '--load-ulti40100',     _best_play(ns, 'ulti40100'),
           '--load-def-parti',     _p(ns, 'defender_parti_agent_best'),
           '--load-def-40100',     _p(ns, 'defender_40100_agent_best'),
           '--load-def-ulti',      _p(ns, 'defender_ulti_agent_best'),
           '--load-def-ulti40100', _p(ns, 'defender_ulti40100_agent_best'),
           '--save',               _p(ns, 'orchestrator'),
           '--n-envs',             str(args.n_envs),
           '--steps',              str(args.orch_steps),
           '--patience',           str(args.orch_patience),
           '--seed',               str(args.seed),
           '--tb-suffix',          args.tb_uid]
    if _exists(ns, 'orchestrator_best'):
        cmd += ['--load-orch', _p(ns, 'orchestrator_best')]
    _run(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# Stage registry
# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth: stage order, sentinel filename, and runner are
# colocated. Adding a stage = appending one Stage(...) entry below.

@dataclass(frozen=True)
class Stage:
    name:     str
    sentinel: str                                  # relative to run dir
    runner:   Callable[[str, argparse.Namespace], None]


STAGES: list[Stage] = [
    Stage('parti_play',         'parti_play_best.zip',              run_parti_play),
    Stage('parti_pregame',      'parti_pregame_best.zip',           run_parti_pregame),
    Stage('defender_parti',     'defender_parti_agent_best.zip',    run_defender_parti),
    Stage('40100',              '40100_play_best.zip',              run_40100),
    Stage('defender_40100',     'defender_40100_agent_best.zip',    run_defender_40100),
    Stage('ulti',               'ulti_play_best.zip',               run_ulti),
    Stage('defender_ulti',      'defender_ulti_agent_best.zip',     run_defender_ulti),
    Stage('ulti40100',          'ulti40100_play_best.zip',          run_ulti40100),
    Stage('defender_ulti40100', 'defender_ulti40100_agent_best.zip', run_defender_ulti40100),
    Stage('orchestrator',       'orchestrator_best.zip',            run_orchestrator),
]

STAGE_NAMES = [s.name for s in STAGES]


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='End-to-end Ulti RL training pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--run',         default='default',
                   help='Run namespace — models saved to models/{run}/')
    p.add_argument('--from-stage',  choices=STAGE_NAMES, default=None,
                   help='Skip all stages before this one; resume normally from it')
    p.add_argument('--n-envs',      type=int, default=4,
                   help='Parallel envs for all stages')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--patience',    type=int, default=5,
                   help='Early-stop patience for declarer/defender stages')

    g = p.add_argument_group('Stage step counts')
    g.add_argument('--parti-steps',            type=int, default=2_000_000,
                   help='Parti play training steps')
    g.add_argument('--parti-pregame-steps',    type=int, default=200_000,
                   help='Parti pre-game agent training steps')
    g.add_argument('--composite-turns',        type=int, default=5,
                   help='B/A turns for each composite subgame')
    g.add_argument('--composite-play-steps',   type=int, default=300_000,
                   help='Play-agent steps per turn (composite subgames)')
    g.add_argument('--composite-pregame-steps', type=int, default=50_000,
                   help='Pre-game agent steps per turn (composite subgames)')
    g.add_argument('--defender-turns',         type=int, default=5,
                   help='B/A turns for each defender stage')
    g.add_argument('--defender-steps-first',   type=int, default=500_000,
                   help='Defender-agent steps for turn 0')
    g.add_argument('--defender-steps',         type=int, default=150_000,
                   help='Defender-agent steps for turns 1+')
    g.add_argument('--declarer-finetune-steps', type=int, default=300_000,
                   help='Declarer fine-tune steps per defender turn (Phase A)')
    g.add_argument('--pregame-recal-steps',     type=int, default=50_000,
                   help='Pre-game recalibration steps after defender B/A loop (Phase C)')
    g.add_argument('--orch-steps',             type=int, default=500_000,
                   help='Orchestrator training steps')
    g.add_argument('--orch-patience',          type=int, default=10,
                   help='Early-stop patience for the orchestrator')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ns   = os.path.join('models', args.run)
    os.makedirs(ns, exist_ok=True)

    # Persist the TensorBoard UID so resumed runs share the same suffix.
    uid_path = os.path.join(ns, 'tb_uid')
    if os.path.exists(uid_path):
        with open(uid_path) as f:
            args.tb_uid = f.read().strip()
    else:
        args.tb_uid = uuid.uuid4().hex[:8]
        with open(uid_path, 'w') as f:
            f.write(args.tb_uid)

    # Determine which stages to run
    skip_before = STAGE_NAMES.index(args.from_stage) if args.from_stage else 0

    print(f"\nUlti RL pipeline — run '{args.run}'  (models → {ns}/)")
    print(f"Session UID: {args.tb_uid}  (TensorBoard suffix)")
    print(f"Stages: {', '.join(STAGE_NAMES)}\n")

    for i, stage in enumerate(STAGES):
        sentinel_path = os.path.join(ns, stage.sentinel)

        if i < skip_before:
            print(f"[skip] {stage.name}  (--from-stage)")
            continue

        if os.path.exists(sentinel_path):
            print(f"[done] {stage.name}  ({sentinel_path} exists)")
            continue

        print(f"\n[run]  {stage.name}")
        stage.runner(ns, args)

        # Verify the stage produced its sentinel
        if not os.path.exists(sentinel_path):
            print(f"\nWarning: {stage.name} completed but sentinel not found: {sentinel_path}",
                  file=sys.stderr)

    print(f"\n{'=' * 68}")
    print(f"  Pipeline complete.  Models in {ns}/")
    print(f"  Orchestrator: {os.path.join(ns, 'orchestrator_best.zip')}")
    print(f"{'=' * 68}\n")


if __name__ == '__main__':
    main()
