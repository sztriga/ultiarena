#!/usr/bin/env python3
"""Analyze CFR strategy pickle to investigate betli overbidding."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.cfr import _BID_ACTIONS, CFRStrategy

# Bid rank labels
BID_LABELS = {
    1: "passz",
    2: "p.passz",
    3: "40-100",
    4: "ulti",
    5: "betli",
    8: "p.40-100",
    10: "p.ulti",
    11: "p.betli",
}

def main():
    pkl_path = Path("models/ulti/trinity/cfr_strategy.pkl")
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        strategy: CFRStrategy = data["strategy"]
        nodes = data.get("nodes", {})
        iterations = data.get("iterations", 0)
        print(f"  Iterations: {iterations}")
        print(f"  Raw nodes: {len(nodes)}")
    else:
        strategy = data
        nodes = {}
        iterations = "?"

    print(f"\n{'='*70}")
    print(f"  CFR STRATEGY ANALYSIS")
    print(f"{'='*70}")

    # --- 1. Overall stats ---
    n_pickup = len(strategy.pickup_table)
    n_bid = len(strategy.bid_table)
    print(f"\n  Pickup nodes: {n_pickup}")
    print(f"  Bid nodes:    {n_bid}")
    print(f"  Total:        {n_pickup + n_bid}")

    # --- 2. Bid action labels ---
    print(f"\n  _BID_ACTIONS = {_BID_ACTIONS}")
    print(f"  Index mapping:")
    for i, rank in enumerate(_BID_ACTIONS):
        print(f"    [{i}] rank {rank} = {BID_LABELS.get(rank, '?')}")

    # --- 3. Average bid probabilities ---
    print(f"\n{'='*70}")
    print(f"  AVERAGE BID PROBABILITIES ACROSS ALL BID NODES")
    print(f"{'='*70}")

    bid_probs_all = []
    for key, probs in strategy.bid_table.items():
        bid_probs_all.append(probs)

    if bid_probs_all:
        avg_probs = np.mean(bid_probs_all, axis=0)
        print(f"\n  {'Rank':<12} {'Label':<12} {'Avg Prob':>10} {'Max Prob':>10} {'Median':>10}")
        print(f"  {'-'*56}")
        max_probs = np.max(bid_probs_all, axis=0)
        med_probs = np.median(bid_probs_all, axis=0)
        for i, rank in enumerate(_BID_ACTIONS):
            label = BID_LABELS.get(rank, "?")
            print(f"  {rank:<12} {label:<12} {avg_probs[i]:>10.4f} {max_probs[i]:>10.4f} {med_probs[i]:>10.4f}")

    # --- 4. Average pickup probability ---
    print(f"\n{'='*70}")
    print(f"  PICKUP NODE STATISTICS")
    print(f"{'='*70}")

    pickup_probs = []
    for key, probs in strategy.pickup_table.items():
        if len(probs) >= 2:
            pickup_probs.append(probs[1])  # prob of picking up

    if pickup_probs:
        print(f"\n  Avg pickup prob:    {np.mean(pickup_probs):.4f}")
        print(f"  Median pickup prob: {np.median(pickup_probs):.4f}")
        print(f"  Max pickup prob:    {np.max(pickup_probs):.4f}")
        print(f"  Min pickup prob:    {np.min(pickup_probs):.4f}")
        print(f"  Std pickup prob:    {np.std(pickup_probs):.4f}")
        # Distribution
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        hist, _ = np.histogram(pickup_probs, bins=bins)
        print(f"\n  Pickup prob distribution:")
        for i in range(len(bins)-1):
            bar = "#" * hist[i]
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>4}  {bar}")

    # --- 5. Top 20 bid nodes by betli probability (rank 5, index 4) ---
    betli_idx = _BID_ACTIONS.index(5)
    pbetli_idx = _BID_ACTIONS.index(11)

    print(f"\n{'='*70}")
    print(f"  TOP 20 BID NODES BY BETLI PROBABILITY (rank 5, idx {betli_idx})")
    print(f"{'='*70}")

    bid_betli_entries = []
    for key, probs in strategy.bid_table.items():
        bid_betli_entries.append((key, probs[betli_idx], probs))
    bid_betli_entries.sort(key=lambda x: -x[1])

    print(f"\n  {'Key':<40} {'Betli':>8} {'Full distribution'}")
    print(f"  {'-'*90}")
    for key, betli_p, probs in bid_betli_entries[:20]:
        dist_str = "  ".join(f"{BID_LABELS.get(_BID_ACTIONS[i], '?')[:4]}={probs[i]:.3f}"
                             for i in range(len(_BID_ACTIONS)))
        print(f"  {key:<40} {betli_p:>8.4f}   {dist_str}")

    # --- 6. Top 20 bid nodes by p.betli probability (rank 11) ---
    print(f"\n{'='*70}")
    print(f"  TOP 20 BID NODES BY P.BETLI PROBABILITY (rank 11, idx {pbetli_idx})")
    print(f"{'='*70}")

    bid_pbetli_entries = []
    for key, probs in strategy.bid_table.items():
        bid_pbetli_entries.append((key, probs[pbetli_idx], probs))
    bid_pbetli_entries.sort(key=lambda x: -x[1])

    print(f"\n  {'Key':<40} {'P.Betli':>8} {'Full distribution'}")
    print(f"  {'-'*90}")
    for key, pbetli_p, probs in bid_pbetli_entries[:20]:
        dist_str = "  ".join(f"{BID_LABELS.get(_BID_ACTIONS[i], '?')[:4]}={probs[i]:.3f}"
                             for i in range(len(_BID_ACTIONS)))
        print(f"  {key:<40} {pbetli_p:>8.4f}   {dist_str}")

    # --- 7. Pattern analysis: parse bucket and auction key ---
    print(f"\n{'='*70}")
    print(f"  PATTERN ANALYSIS: BETLI BIDS BY AUCTION STATE")
    print(f"{'='*70}")

    # Parse bid keys: "bid|<bucket>|<rel_pos>|<bid_rank>|<is_holder>|<awaiting>|<n_pickups>"
    # Aggregate betli prob by auction state components
    by_bucket = defaultdict(list)
    by_rel_pos = defaultdict(list)
    by_cur_bid_rank = defaultdict(list)
    by_is_holder = defaultdict(list)
    by_awaiting = defaultdict(list)
    by_n_pickups = defaultdict(list)

    for key, probs in strategy.bid_table.items():
        betli_p = probs[betli_idx]
        pbetli_p = probs[pbetli_idx]
        combined = betli_p + pbetli_p

        parts = key.split("|")  # bid|bucket|rel_pos|bid_rank|is_holder|awaiting|n_pickups
        if len(parts) == 7:
            _, bucket, rel_pos, bid_rank, is_holder, awaiting, n_pickups = parts
            by_bucket[int(bucket)].append(combined)
            by_rel_pos[rel_pos].append(combined)
            by_cur_bid_rank[bid_rank].append(combined)
            by_is_holder[is_holder].append(combined)
            by_awaiting[awaiting].append(combined)
            by_n_pickups[n_pickups].append(combined)

    print(f"\n  Average betli+p.betli prob by relative position:")
    for k in sorted(by_rel_pos.keys()):
        vals = by_rel_pos[k]
        print(f"    rel_pos={k}: avg={np.mean(vals):.4f}, n={len(vals)}")

    print(f"\n  Average betli+p.betli prob by current bid rank:")
    for k in sorted(by_cur_bid_rank.keys(), key=lambda x: int(x)):
        vals = by_cur_bid_rank[k]
        print(f"    cur_bid_rank={k}: avg={np.mean(vals):.4f}, n={len(vals)}")

    print(f"\n  Average betli+p.betli prob by is_holder:")
    for k in sorted(by_is_holder.keys()):
        vals = by_is_holder[k]
        print(f"    is_holder={k}: avg={np.mean(vals):.4f}, n={len(vals)}")

    print(f"\n  Average betli+p.betli prob by awaiting_bid:")
    for k in sorted(by_awaiting.keys()):
        vals = by_awaiting[k]
        print(f"    awaiting={k}: avg={np.mean(vals):.4f}, n={len(vals)}")

    print(f"\n  Average betli+p.betli prob by n_pickups:")
    for k in sorted(by_n_pickups.keys()):
        vals = by_n_pickups[k]
        print(f"    n_pickups={k}: avg={np.mean(vals):.4f}, n={len(vals)}")

    # Top buckets
    print(f"\n  Top 20 buckets by avg betli+p.betli prob:")
    bucket_avgs = [(b, np.mean(v), len(v)) for b, v in by_bucket.items()]
    bucket_avgs.sort(key=lambda x: -x[1])
    print(f"  {'Bucket':<10} {'Avg Prob':>10} {'N nodes':>10} {'Decoded (trump,str,marr,side)'}")
    print(f"  {'-'*65}")
    for bucket, avg_p, n in bucket_avgs[:20]:
        # Decode bucket: trump_count * 32 + strength_tier * 8 + marriage * 4 + side_len
        side_len = bucket % 4
        rem = bucket // 4
        marriage = rem % 2
        rem = rem // 2
        strength_tier = rem % 4
        trump_count = rem // 4
        print(f"  {bucket:<10} {avg_p:>10.4f} {n:>10}   trump={trump_count}, str_tier={strength_tier}, marr={marriage}, side={side_len}")

    # --- 8. Distribution of betli prob across all bid nodes ---
    print(f"\n{'='*70}")
    print(f"  DISTRIBUTION OF BETLI PROB ACROSS ALL BID NODES")
    print(f"{'='*70}")

    all_betli_probs = [probs[betli_idx] for probs in strategy.bid_table.values()]
    all_pbetli_probs = [probs[pbetli_idx] for probs in strategy.bid_table.values()]

    bins = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.01]
    hist_b, _ = np.histogram(all_betli_probs, bins=bins)
    hist_pb, _ = np.histogram(all_pbetli_probs, bins=bins)

    print(f"\n  Betli (rank 5) probability distribution:")
    for i in range(len(bins)-1):
        bar = "#" * (hist_b[i] // 2 + (1 if hist_b[i] > 0 else 0))
        print(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}): {hist_b[i]:>5}  {bar}")

    print(f"\n  P.Betli (rank 11) probability distribution:")
    for i in range(len(bins)-1):
        bar = "#" * (hist_pb[i] // 2 + (1 if hist_pb[i] > 0 else 0))
        print(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}): {hist_pb[i]:>5}  {bar}")

    # --- 9. Weighted analysis: how much total probability mass goes to betli? ---
    print(f"\n{'='*70}")
    print(f"  WEIGHTED ANALYSIS: STRATEGY SUM MASS ON BETLI")
    print(f"{'='*70}")

    if nodes:
        bid_nodes = {k: v for k, v in nodes.items() if k.startswith("bid|")}
        print(f"\n  Bid nodes with raw regret/strategy sums: {len(bid_nodes)}")

        # Look at strategy_sum to see cumulative weight
        total_strat_sum = np.zeros(len(_BID_ACTIONS), dtype=np.float64)
        for key, node in bid_nodes.items():
            total_strat_sum += node.strategy_sum

        print(f"\n  Total strategy_sum across all bid nodes:")
        grand_total = total_strat_sum.sum()
        for i, rank in enumerate(_BID_ACTIONS):
            label = BID_LABELS.get(rank, "?")
            frac = total_strat_sum[i] / grand_total if grand_total > 0 else 0
            print(f"    {rank:<4} {label:<12} sum={total_strat_sum[i]:>12.1f}  frac={frac:>8.4f}")

        # Also look at regret sums for betli
        print(f"\n  Average regret_sum for betli and p.betli across bid nodes:")
        betli_regrets = []
        pbetli_regrets = []
        for key, node in bid_nodes.items():
            betli_regrets.append(node.regret_sum[betli_idx])
            pbetli_regrets.append(node.regret_sum[pbetli_idx])
        print(f"    Betli  avg regret: {np.mean(betli_regrets):>10.2f}, "
              f"median: {np.median(betli_regrets):>10.2f}, "
              f"pos count: {sum(1 for r in betli_regrets if r > 0)}/{len(betli_regrets)}")
        print(f"    P.Betli avg regret: {np.mean(pbetli_regrets):>10.2f}, "
              f"median: {np.median(pbetli_regrets):>10.2f}, "
              f"pos count: {sum(1 for r in pbetli_regrets if r > 0)}/{len(pbetli_regrets)}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
