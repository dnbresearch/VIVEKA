#!/usr/bin/env python3
"""
Community Stability Analysis
==============================
Tests whether community detection is stable across random seeds.
Runs Louvain 10 times with different seeds, computes NMI between
all pairs of partitions.

Zero API cost — pure graph computation.

Usage:
  python3 community_stability.py \
      --phase1 ./validation_results/scale/phase1_incremental.json
"""

import json, os, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")
from community_hypothesis_v3 import (
    extract_features, normalize_venue, build_graph_batched,
    GENERIC_METHODS
)

RESULTS_DIR = Path("./validation_results/community_stability")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def louvain_partition(G, seed):
    """Run Louvain and return node->community mapping."""
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, resolution=1.2, seed=seed)
    mapping = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            mapping[node] = idx
    return mapping


def normalized_mutual_info(partition1, partition2, nodes):
    """Compute NMI between two partitions."""
    from collections import Counter
    import math

    labels1 = [partition1.get(n, -1) for n in nodes]
    labels2 = [partition2.get(n, -1) for n in nodes]
    n = len(nodes)

    # Contingency table
    contingency = defaultdict(int)
    for l1, l2 in zip(labels1, labels2):
        contingency[(l1, l2)] += 1

    # Marginals
    row_sums = Counter(labels1)
    col_sums = Counter(labels2)

    # Mutual information
    mi = 0
    for (r, c), count in contingency.items():
        if count == 0: continue
        p_rc = count / n
        p_r = row_sums[r] / n
        p_c = col_sums[c] / n
        if p_r > 0 and p_c > 0:
            mi += p_rc * math.log(p_rc / (p_r * p_c))

    # Entropies
    h1 = -sum((c/n) * math.log(c/n) for c in row_sums.values() if c > 0)
    h2 = -sum((c/n) * math.log(c/n) for c in col_sums.values() if c > 0)

    if h1 + h2 == 0:
        return 1.0
    return 2 * mi / (h1 + h2)


def adjusted_rand_index(partition1, partition2, nodes):
    """Compute ARI between two partitions."""
    from collections import Counter
    import math

    labels1 = [partition1.get(n, -1) for n in nodes]
    labels2 = [partition2.get(n, -1) for n in nodes]
    n = len(nodes)

    # Contingency
    contingency = defaultdict(int)
    for l1, l2 in zip(labels1, labels2):
        contingency[(l1, l2)] += 1

    row_sums = Counter(labels1)
    col_sums = Counter(labels2)

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_nij = sum(comb2(v) for v in contingency.values())
    sum_comb_ai = sum(comb2(v) for v in row_sums.values())
    sum_comb_bj = sum(comb2(v) for v in col_sums.values())
    comb_n = comb2(n)

    if comb_n == 0:
        return 1.0

    expected = sum_comb_ai * sum_comb_bj / comb_n
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    denom = max_index - expected

    if denom == 0:
        return 1.0
    return (sum_comb_nij - expected) / denom


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print(f"COMMUNITY STABILITY ({args.n_seeds} seeds)")
    print("=" * 70)

    with open(args.phase1) as f:
        p1 = json.load(f)

    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    papers, venues = {}, {}
    for r in graph_ready:
        title = r.get("title", "")
        ins = r.get("_insights", [])
        if not ins or not title: continue
        f = extract_features(title, ins)
        if len(f["params"]) >= 2 or f["benchmarks"] or (f["methods"] - GENERIC_METHODS):
            papers[title] = f
            venues[title] = normalize_venue(r.get("venue", "?"))
    titles = list(papers.keys())

    print(f"  Papers: {len(titles)}, Threshold: {args.threshold}")

    # Build graph
    G = build_graph_batched(papers, titles, venues, sim_threshold=args.threshold)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nodes = list(G.nodes())

    # Run Louvain with different seeds
    print(f"\n  Running Louvain {args.n_seeds} times...")
    partitions = []
    n_communities = []
    for seed in range(args.n_seeds):
        t0 = time.time()
        part = louvain_partition(G, seed=seed)
        t1 = time.time()
        n_comm = len(set(part.values()))
        partitions.append(part)
        n_communities.append(n_comm)
        print(f"    Seed {seed}: {n_comm} communities ({t1-t0:.3f}s)")

    # Pairwise NMI
    print(f"\n  --- Pairwise NMI ---")
    nmi_matrix = np.zeros((args.n_seeds, args.n_seeds))
    ari_matrix = np.zeros((args.n_seeds, args.n_seeds))

    for i in range(args.n_seeds):
        for j in range(i, args.n_seeds):
            if i == j:
                nmi_matrix[i][j] = 1.0
                ari_matrix[i][j] = 1.0
            else:
                nmi = normalized_mutual_info(partitions[i], partitions[j], nodes)
                ari = adjusted_rand_index(partitions[i], partitions[j], nodes)
                nmi_matrix[i][j] = nmi
                nmi_matrix[j][i] = nmi
                ari_matrix[i][j] = ari
                ari_matrix[j][i] = ari

    # Extract off-diagonal values
    nmi_vals = []
    ari_vals = []
    for i in range(args.n_seeds):
        for j in range(i+1, args.n_seeds):
            nmi_vals.append(nmi_matrix[i][j])
            ari_vals.append(ari_matrix[i][j])

    print(f"\n  NMI statistics:")
    print(f"    Mean: {np.mean(nmi_vals):.4f}")
    print(f"    Std:  {np.std(nmi_vals):.4f}")
    print(f"    Min:  {np.min(nmi_vals):.4f}")
    print(f"    Max:  {np.max(nmi_vals):.4f}")

    print(f"\n  ARI statistics:")
    print(f"    Mean: {np.mean(ari_vals):.4f}")
    print(f"    Std:  {np.std(ari_vals):.4f}")
    print(f"    Min:  {np.min(ari_vals):.4f}")
    print(f"    Max:  {np.max(ari_vals):.4f}")

    print(f"\n  Community count statistics:")
    print(f"    Mean: {np.mean(n_communities):.1f}")
    print(f"    Std:  {np.std(n_communities):.1f}")
    print(f"    Range: {min(n_communities)}-{max(n_communities)}")

    # Stability interpretation
    nmi_mean = np.mean(nmi_vals)
    if nmi_mean > 0.9:
        stability = "HIGHLY STABLE"
    elif nmi_mean > 0.7:
        stability = "STABLE"
    elif nmi_mean > 0.5:
        stability = "MODERATELY STABLE"
    else:
        stability = "UNSTABLE"

    print(f"\n  Interpretation: {stability} (NMI = {nmi_mean:.3f})")

    # Core members analysis: which nodes are always in the same community?
    print(f"\n  --- Core Membership Analysis ---")
    # For each pair of nodes, check how often they're in the same community
    # Sample 500 random pairs
    import random
    random.seed(42)
    
    if len(nodes) > 50:
        sample_nodes = random.sample(nodes, min(100, len(nodes)))
    else:
        sample_nodes = nodes
    
    co_membership = defaultdict(int)
    for part in partitions:
        comm_map = defaultdict(set)
        for n in sample_nodes:
            if n in part:
                comm_map[part[n]].add(n)
        for comm_members in comm_map.values():
            members = list(comm_members)
            for a in range(len(members)):
                for b in range(a+1, len(members)):
                    co_membership[(members[a], members[b])] += 1
    
    # What fraction of pairs are always together?
    always_together = sum(1 for v in co_membership.values() if v == args.n_seeds)
    sometimes = sum(1 for v in co_membership.values() if 0 < v < args.n_seeds)
    total_pairs = len(co_membership)
    
    print(f"  Always in same community: {always_together}/{total_pairs} pairs "
          f"({100*always_together/max(total_pairs,1):.0f}%)")
    print(f"  Sometimes together:       {sometimes}/{total_pairs} pairs")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"""
  Community detection is {stability}.
  NMI = {nmi_mean:.3f} ± {np.std(nmi_vals):.3f} across {args.n_seeds} random seeds.
  ARI = {np.mean(ari_vals):.3f} ± {np.std(ari_vals):.3f}.
  Number of communities: {np.mean(n_communities):.1f} ± {np.std(n_communities):.1f}.
  
  Paper text:
  "Community detection is stable across random initializations:
  Louvain partitioning with {args.n_seeds} different seeds yields
  NMI = {nmi_mean:.2f} ± {np.std(nmi_vals):.2f} and
  ARI = {np.mean(ari_vals):.2f} ± {np.std(ari_vals):.2f},
  with {np.mean(n_communities):.0f} ± {np.std(n_communities):.0f} communities
  per run."
""")

    # Save
    results = {
        "n_seeds": args.n_seeds,
        "threshold": args.threshold,
        "n_nodes": len(nodes),
        "n_edges": G.number_of_edges(),
        "nmi_mean": round(np.mean(nmi_vals), 4),
        "nmi_std": round(np.std(nmi_vals), 4),
        "nmi_min": round(np.min(nmi_vals), 4),
        "nmi_max": round(np.max(nmi_vals), 4),
        "ari_mean": round(np.mean(ari_vals), 4),
        "ari_std": round(np.std(ari_vals), 4),
        "n_communities_mean": round(np.mean(n_communities), 1),
        "n_communities_std": round(np.std(n_communities), 1),
        "stability": stability,
        "always_together_pct": round(100 * always_together / max(total_pairs, 1), 1),
    }
    with open(RESULTS_DIR / "stability_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
