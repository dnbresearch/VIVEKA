#!/usr/bin/env python3
"""
Ablation Study: Similarity Threshold Effect on Community Hypotheses
=====================================================================
Sweeps sim_threshold from 2.0 to 8.0 and measures:
  - Graph density, #edges
  - #communities, avg size, cross-domain ratio
  - Hypothesis quality (specificity, groundedness, actionability, cross-paper novelty)

Produces a comparison table for the ICDM paper.

Usage:
  # Phase 1: Graph ablation only (no LLM cost)
  python3 threshold_ablation.py --phase1 ./validation_results/scale/phase1_incremental.json --skip-llm

  # Phase 2: Full ablation with hypothesis generation
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 threshold_ablation.py --phase1 ./validation_results/scale/phase1_incremental.json --groups-per-threshold 5
"""

import json, os, sys, time
from pathlib import Path
from collections import Counter, defaultdict

# Import from community_hypothesis_v3
sys.path.insert(0, ".")
from community_hypothesis_v3 import (
    extract_features, compute_sim, normalize_venue, build_graph_batched,
    detect_communities, score_community, generate_hypotheses_for_group,
    GENERIC_METHODS, RESULTS_DIR as BASE_RESULTS_DIR
)

RESULTS_DIR = Path("./validation_results/threshold_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


def run_threshold(papers, titles, venues, insights_map, threshold, api_key=None, max_groups=5):
    """Run community detection + optional hypothesis generation at one threshold."""
    import networkx as nx

    result = {"threshold": threshold}

    # Build graph
    G = build_graph_batched(papers, titles, venues, sim_threshold=threshold)
    result["n_nodes"] = G.number_of_nodes()
    result["n_edges"] = G.number_of_edges()
    result["density"] = round(2 * G.number_of_edges() / max(G.number_of_nodes() * (G.number_of_nodes()-1), 1), 4)

    # Connected components
    components = list(nx.connected_components(G))
    result["n_components"] = len(components)
    result["largest_component"] = max(len(c) for c in components) if components else 0
    isolated = sum(1 for c in components if len(c) == 1)
    result["isolated_nodes"] = isolated

    # Community detection
    raw_groups = detect_communities(G, papers, venues, min_community=3)

    # Score communities
    scored = []
    for members in raw_groups:
        info = score_community(members, papers, venues)
        if info and info["coherence"] >= 1.0:
            scored.append(info)
    scored.sort(key=lambda x: -x["score"])

    # Deduplicate
    final_groups = []
    used = set()
    for g in scored:
        members_set = set(g["members"])
        overlap = len(members_set & used) / max(len(members_set), 1)
        if overlap < 0.5:
            final_groups.append(g)
            used.update(members_set)
        if len(final_groups) >= 50:
            break

    result["n_communities"] = len(final_groups)
    sizes = [g["size"] for g in final_groups]
    result["avg_size"] = round(sum(sizes) / max(len(sizes), 1), 1)
    result["min_size"] = min(sizes) if sizes else 0
    result["max_size"] = max(sizes) if sizes else 0
    result["median_size"] = sorted(sizes)[len(sizes)//2] if sizes else 0

    cross_domain = sum(1 for g in final_groups if g.get("cross_domain"))
    result["cross_domain"] = cross_domain
    result["cross_domain_pct"] = round(100 * cross_domain / max(len(final_groups), 1), 1)

    total_papers = sum(g["size"] for g in final_groups)
    unique_papers = len(set(t for g in final_groups for t in g["members"]))
    result["total_papers_in_groups"] = total_papers
    result["unique_papers_in_groups"] = unique_papers

    # Venue diversity stats
    venue_diversities = [g["venue_diversity"] for g in final_groups]
    result["avg_venue_diversity"] = round(sum(venue_diversities) / max(len(venue_diversities), 1), 1)

    # Coherence stats
    coherences = [g["coherence"] for g in final_groups]
    result["avg_coherence"] = round(sum(coherences) / max(len(coherences), 1), 2)

    # Shared feature stats
    with_shared_bench = sum(1 for g in final_groups if g["shared_bench"])
    with_shared_methods = sum(1 for g in final_groups if g["shared_methods"])
    result["groups_with_shared_bench"] = with_shared_bench
    result["groups_with_shared_methods"] = with_shared_methods

    result["groups"] = final_groups

    # Hypothesis generation (if API key provided)
    if api_key and max_groups > 0:
        hyp_results = []
        total_h = 0
        total_grounded = 0
        total_cross_venue = 0
        total_tokens = 0

        # Pick top groups by score, prefer diversity in size
        hyp_groups = final_groups[:max_groups]

        for i, g in enumerate(hyp_groups):
            print(f"    T={threshold} Group {i+1}/{len(hyp_groups)} "
                  f"({g['size']} papers, venues={g['venues']})...")
            hyps, tokens = generate_hypotheses_for_group(g, papers, insights_map, api_key)
            total_tokens += tokens

            grounded = sum(1 for h in hyps if len(h.get("cited_papers", [])) >= 2)
            cross_v = sum(1 for h in hyps if h.get("gap_type") == "cross_venue_transfer")

            total_h += len(hyps)
            total_grounded += grounded
            total_cross_venue += cross_v

            hyp_results.append({
                "group_id": i+1,
                "size": g["size"],
                "venues": g["venues"],
                "n_hypotheses": len(hyps),
                "n_grounded": grounded,
                "n_cross_venue": cross_v,
                "hypotheses": hyps,
            })
            time.sleep(0.5)

        result["n_hypotheses"] = total_h
        result["n_grounded"] = total_grounded
        result["grounded_pct"] = round(100 * total_grounded / max(total_h, 1), 1)
        result["n_cross_venue"] = total_cross_venue
        result["tokens"] = total_tokens
        result["cost"] = round(total_tokens * 6 / 1_000_000, 3)
        result["hyp_per_group"] = round(total_h / max(len(hyp_groups), 1), 1)
        result["hypothesis_details"] = hyp_results

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--groups-per-threshold", type=int, default=5,
                        help="How many groups to generate hypotheses for per threshold")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds, e.g. '2.0,4.0,6.0'")
    args = parser.parse_args()

    api_key = "" if args.skip_llm else os.environ.get("ANTHROPIC_API_KEY", "")
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else THRESHOLDS

    # Load data
    print("=" * 70)
    print("ABLATION: SIMILARITY THRESHOLD EFFECT")
    print("=" * 70)

    with open(args.phase1) as f:
        p1 = json.load(f)
    print(f"Loaded {len(p1)} repos")

    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    print(f"Graph-ready: {len(graph_ready)}")

    # Extract features (once, reuse across thresholds)
    papers = {}
    insights_map = {}
    venues = {}
    for r in graph_ready:
        title = r.get("title", "")
        ins = r.get("_insights", [])
        if not ins or not title: continue
        f = extract_features(title, ins)
        if len(f["params"]) >= 2 or f["benchmarks"] or (f["methods"] - GENERIC_METHODS):
            papers[title] = f
            insights_map[title] = ins
            venues[title] = normalize_venue(r.get("venue", "?"))

    titles = list(papers.keys())
    print(f"Papers with features: {len(titles)}")

    # Run ablation
    all_results = []
    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"THRESHOLD = {threshold}")
        print(f"{'='*70}")

        result = run_threshold(
            papers, titles, venues, insights_map, threshold,
            api_key=api_key if not args.skip_llm else None,
            max_groups=args.groups_per_threshold,
        )
        all_results.append(result)

        # Quick summary
        print(f"\n  T={threshold}: {result['n_edges']:,} edges, "
              f"{result['n_communities']} communities, "
              f"avg_size={result['avg_size']}, "
              f"cross_domain={result['cross_domain']}")

    # Print comparison table
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS TABLE")
    print("="*70)

    header = f"{'Thresh':>7} {'Edges':>8} {'Density':>8} {'#Comm':>6} {'AvgSz':>6} {'MedSz':>6} " \
             f"{'MaxSz':>6} {'X-Dom':>6} {'X-Dom%':>7} {'VenDiv':>7} {'Coher':>7} {'ShBnch':>7} {'ShMeth':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in all_results:
        line = (f"{r['threshold']:>7.1f} {r['n_edges']:>8,} {r['density']:>8.4f} "
                f"{r['n_communities']:>6} {r['avg_size']:>6.1f} {r['median_size']:>6} "
                f"{r['max_size']:>6} {r['cross_domain']:>6} {r['cross_domain_pct']:>6.1f}% "
                f"{r['avg_venue_diversity']:>7.1f} {r['avg_coherence']:>7.2f} "
                f"{r['groups_with_shared_bench']:>7} {r['groups_with_shared_methods']:>7}")
        print(line)

    # If hypotheses were generated, add hypothesis quality table
    if not args.skip_llm:
        print(f"\n{'='*70}")
        print("HYPOTHESIS QUALITY BY THRESHOLD")
        print("="*70)

        h_header = f"{'Thresh':>7} {'#Hyp':>6} {'Grounded':>9} {'Grnd%':>6} {'X-Venue':>8} {'Hyp/Grp':>8} {'Tokens':>8} {'Cost':>7}"
        print(f"\n{h_header}")
        print("-" * len(h_header))

        for r in all_results:
            if "n_hypotheses" in r:
                line = (f"{r['threshold']:>7.1f} {r['n_hypotheses']:>6} {r['n_grounded']:>9} "
                        f"{r['grounded_pct']:>5.1f}% {r['n_cross_venue']:>8} "
                        f"{r['hyp_per_group']:>8.1f} {r['tokens']:>8,} ${r['cost']:>6.3f}")
                print(line)

    # Save
    # Strip heavy fields for summary
    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k not in ("groups", "hypothesis_details")}
        summary.append(s)

    with open(RESULTS_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(RESULTS_DIR / "ablation_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))

    # Generate LaTeX table for paper
    print(f"\n{'='*70}")
    print("LATEX TABLE (for paper)")
    print("="*70)

    print(r"""
\begin{table}[t]
\centering
\caption{Effect of similarity threshold on community structure and hypothesis quality.}
\label{tab:threshold_ablation}
\begin{tabular}{r|rrr|rrr}
\toprule
$\tau$ & Edges & \#Comm & Avg Size & X-Domain\% & Coherence & Sh.Bench \\
\midrule""")
    for r in all_results:
        print(f"{r['threshold']:.1f} & {r['n_edges']:,} & {r['n_communities']} & "
              f"{r['avg_size']:.0f} & {r['cross_domain_pct']:.0f}\\% & "
              f"{r['avg_coherence']:.2f} & {r['groups_with_shared_bench']} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    if not args.skip_llm:
        print(r"""
\begin{table}[t]
\centering
\caption{Hypothesis quality across similarity thresholds.}
\label{tab:threshold_hyp_quality}
\begin{tabular}{r|rrrr}
\toprule
$\tau$ & \#Hyp & Grounded\% & X-Venue & Hyp/Group \\
\midrule""")
        for r in all_results:
            if "n_hypotheses" in r:
                print(f"{r['threshold']:.1f} & {r['n_hypotheses']} & "
                      f"{r['grounded_pct']:.0f}\\% & {r['n_cross_venue']} & "
                      f"{r['hyp_per_group']:.1f} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print(f"\nAll saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
