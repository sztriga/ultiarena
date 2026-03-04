#!/usr/bin/env python3
"""Offline CFR training for Ulti bidding strategy.

Loads pre-trained value heads (trinity model) and runs external-sampling
MCCFR to compute an approximate Nash equilibrium bidding strategy.
The resulting strategy table is saved as a pickle file that can be loaded
at tournament time.

Usage:
    python scripts/cfr_bid.py trinity --iterations 10000 --hands-per-iter 100
    python scripts/cfr_bid.py trinity --resume models/ulti/trinity/cfr_strategy.pkl
    python scripts/cfr_bid.py trinity --iterations 100 --hands-per-iter 10  # quick test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from trickster.bidding.cfr import CFRSolver, NUM_BUCKETS
from trickster.training.model_io import load_wrappers


def _print_strategy_stats(solver: CFRSolver) -> None:
    """Print diagnostic statistics about the learned strategy."""
    n_nodes = len(solver.nodes)
    n_pickup = sum(1 for k in solver.nodes if k.startswith("pickup|"))
    n_bid = sum(1 for k in solver.nodes if k.startswith("bid|"))

    print()
    print(f"  Strategy statistics:")
    print(f"    Total info sets:  {n_nodes:,}")
    print(f"    Pickup nodes:     {n_pickup:,}")
    print(f"    Bid nodes:        {n_bid:,}")

    # Strategy entropy distribution
    entropies = []
    for node in solver.nodes.values():
        strat = node.average_strategy()
        ent = -np.sum(strat * np.log(strat + 1e-10))
        entropies.append(ent)

    if entropies:
        print(f"    Avg entropy:      {np.mean(entropies):.3f}")
        print(f"    Min entropy:      {np.min(entropies):.3f}")
        print(f"    Max entropy:      {np.max(entropies):.3f}")

    # Check non-triviality
    n_deterministic = 0
    n_uniform = 0
    for node in solver.nodes.values():
        strat = node.average_strategy()
        if np.max(strat) > 0.99:
            n_deterministic += 1
        if np.std(strat) < 0.01:
            n_uniform += 1

    print(f"    Near-deterministic: {n_deterministic}/{n_nodes} "
          f"({100 * n_deterministic / max(1, n_nodes):.0f}%)")
    print(f"    Near-uniform:      {n_uniform}/{n_nodes} "
          f"({100 * n_uniform / max(1, n_nodes):.0f}%)")

    # Pickup statistics
    if n_pickup > 0:
        pickup_rates = []
        for key, node in solver.nodes.items():
            if key.startswith("pickup|"):
                strat = node.average_strategy()
                if len(strat) >= 2:
                    pickup_rates.append(strat[1])
        if pickup_rates:
            print(f"    Avg pickup prob:  {np.mean(pickup_rates):.3f}")
            print(f"    Max pickup prob:  {np.max(pickup_rates):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CFR bidding strategy for Ulti.",
    )
    parser.add_argument(
        "source", type=str,
        help="Model source name (e.g. 'trinity')",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of CFR iterations (default: 1000)",
    )
    parser.add_argument(
        "--hands-per-iter", type=int, default=100,
        help="Hands dealt per iteration (default: 100)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing CFR strategy file to resume from",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for strategy file (default: models/ulti/<source>/cfr_strategy.pkl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="Print progress every N iterations (default: 100)",
    )
    args = parser.parse_args()

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"models/ulti/{args.source}/cfr_strategy.pkl")

    # Banner
    print()
    w = 60
    print("=" * w)
    print(f"  CFR BIDDING STRATEGY TRAINER")
    print("=" * w)
    print(f"  Source:          {args.source}")
    print(f"  Iterations:      {args.iterations}")
    print(f"  Hands/iter:      {args.hands_per_iter}")
    print(f"  Total hands:     {args.iterations * args.hands_per_iter:,}")
    print(f"  Hand buckets:    {NUM_BUCKETS}")
    print(f"  Output:          {out_path}")
    if args.resume:
        print(f"  Resume from:     {args.resume}")
    print()

    # Load model wrappers
    print("  Loading model wrappers...", flush=True)
    wrappers = load_wrappers(args.source)
    if not wrappers:
        print(f"  ERROR: No model wrappers found for source '{args.source}'")
        sys.exit(1)
    print(f"  Loaded {len(wrappers)} contract models: {', '.join(wrappers.keys())}")
    print()

    # Create or resume solver
    import random as rng_module
    rng = rng_module.Random(args.seed)

    if args.resume:
        print(f"  Resuming from {args.resume}...", flush=True)
        solver = CFRSolver.load(args.resume, wrappers=wrappers)
        print(f"  Loaded {len(solver.nodes)} existing info sets, "
              f"{solver.iterations} previous iterations")
    else:
        solver = CFRSolver(wrappers=wrappers)

    # Train
    print()
    print(f"  Training CFR for {args.iterations} iterations...", flush=True)
    t0 = time.perf_counter()

    solver.train(
        n_iterations=args.iterations,
        n_hands_per_iter=args.hands_per_iter,
        rng=rng,
        progress_every=args.progress_every,
    )

    elapsed = time.perf_counter() - t0
    total_hands = args.iterations * args.hands_per_iter
    print()
    print(f"  Training complete: {elapsed:.1f}s "
          f"({total_hands / elapsed:.0f} hands/s)")

    # Print strategy diagnostics
    _print_strategy_stats(solver)

    # Save
    print()
    solver.save(out_path)
    print()
    print("  Done!")
    print()


if __name__ == "__main__":
    main()
