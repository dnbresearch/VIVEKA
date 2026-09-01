#!/usr/bin/env python3
"""
Q2: What characterizes the 90.2% uncovered papers at τ=6.0?
- Graph isolates (degree=0)?
- In communities that fail coherence filter?
- In communities removed by deduplication?
- In communities too small (<3 members)?
"""
import json, sys, os
import numpy as np
from collections import defaultdict

sys.path.insert(0, ".")
PHASE1_PATH = "validation_results/scale/phase1_incremental.json"

from community_hypothesis_v3 import (
    extract_features, normalize_venue, build_graph_batched,
    detect_communities, score_community, GENERIC_METHODS
)

with open(PHASE1_PATH) as f:
    p1 = json.load(f)

papers, venues = {}, {}
for r in p1:
    if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3:
        title = r.get("title", "")
        ins = r.get("_insights", [])
        if not ins or not title:
            continue
        f = extract_features(title, ins)
        if len(f["params"]) >= 2 or f["benchmarks"] or (f["methods"] - GENERIC_METHODS):
            papers[title] = f
            venues[title] = normalize_venue(r.get("venue", "?"))

titles = list(papers.keys())
N = len(titles)
print(f"Total graph-ready papers: {N}")

# Build graph at τ=6.0
G = build_graph_batched(papers, titles, venues, sim_threshold=6.0)

# Category 1: Isolates (no edges at τ=6.0)
isolates = [t for t in titles if G.degree(t) == 0]
print(f"\n1. Graph isolates (degree=0): {len(isolates)} ({100*len(isolates)/N:.1f}%)")

# Category 2: Connected but in tiny components (<3)
import networkx as nx
components = list(nx.connected_components(G))
tiny_comp_papers = set()
for comp in components:
    if len(comp) < 3:
        tiny_comp_papers.update(comp)
tiny_comp_papers -= set(isolates)  # don't double count
print(f"2. In tiny components (<3 members): {len(tiny_comp_papers)} ({100*len(tiny_comp_papers)/N:.1f}%)")

# Run community detection to get ALL communities (before filtering)
raw_groups = detect_communities(G, papers, venues, min_community=3)

# Category 3: In raw communities but fail coherence filter
all_raw_members = set()
coherent_members = set()
low_coherence_members = set()

scored_all = []
for members in raw_groups:
    all_raw_members.update(members)
    info = score_community(members, papers, venues)
    if info:
        scored_all.append(info)
        if info["coherence"] >= 1.0:
            coherent_members.update(members)
        else:
            low_coherence_members.update(members)

failed_coherence = low_coherence_members - coherent_members
print(f"3. In communities failing coherence (<1.0): {len(failed_coherence)} ({100*len(failed_coherence)/N:.1f}%)")

# Category 4: In coherent communities but removed by deduplication
scored_coherent = [s for s in scored_all if s["coherence"] >= 1.0]
scored_coherent.sort(key=lambda x: -x["score"])

final = []
used = set()
dedup_removed = set()
for g in scored_coherent:
    ms = set(g["members"])
    overlap = len(ms & used) / max(len(ms), 1)
    if overlap < 0.5:
        final.append(g)
        used.update(ms)
    else:
        dedup_removed.update(ms - used)

print(f"4. Removed by deduplication: {len(dedup_removed)} ({100*len(dedup_removed)/N:.1f}%)")

# Category 5: In final communities (covered)
covered = used
print(f"5. In final communities (COVERED): {len(covered)} ({100*len(covered)/N:.1f}%)")

# Summary
other = N - len(isolates) - len(tiny_comp_papers) - len(failed_coherence) - len(dedup_removed) - len(covered)
print(f"\n{'='*55}")
print(f"BREAKDOWN OF {N} PAPERS AT τ=6.0:")
print(f"{'='*55}")
print(f"  Covered (in final communities):  {len(covered):>5} ({100*len(covered)/N:>5.1f}%)")
print(f"  Graph isolates (no edges):       {len(isolates):>5} ({100*len(isolates)/N:>5.1f}%)")
print(f"  Tiny components (<3):            {len(tiny_comp_papers):>5} ({100*len(tiny_comp_papers)/N:>5.1f}%)")
print(f"  Failed coherence filter:         {len(failed_coherence):>5} ({100*len(failed_coherence)/N:>5.1f}%)")
print(f"  Removed by deduplication:        {len(dedup_removed):>5} ({100*len(dedup_removed)/N:>5.1f}%)")
print(f"  Other/unaccounted:               {other:>5} ({100*other/N:>5.1f}%)")
print(f"  {'':->45}")
print(f"  Total:                           {N:>5}")

# Degree distribution of uncovered papers
uncovered = set(titles) - covered
uncov_degrees = [G.degree(t) for t in uncovered]
cov_degrees = [G.degree(t) for t in covered]
print(f"\n  Avg degree (covered):   {np.mean(cov_degrees):.1f}")
print(f"  Avg degree (uncovered): {np.mean(uncov_degrees):.1f}")
print(f"  Median degree (uncov):  {np.median(uncov_degrees):.0f}")

# Can coverage improve with different filtering?
print(f"\n  --- Could coverage improve? ---")
print(f"  Communities before coherence filter: {len(scored_all)}")
print(f"  Communities after coherence filter:  {len(scored_coherent)}")
print(f"  Communities after deduplication:     {len(final)}")
print(f"  Papers in raw communities:          {len(all_raw_members)} ({100*len(all_raw_members)/N:.1f}%)")
print(f"  → Lowering coherence threshold would add {len(failed_coherence)} papers")
print(f"  → Relaxing deduplication would add {len(dedup_removed)} papers")
print(f"  → Max possible at τ=6.0: {len(all_raw_members)} ({100*len(all_raw_members)/N:.1f}%)")

results = {
    "total": N,
    "covered": len(covered),
    "isolates": len(isolates),
    "tiny_components": len(tiny_comp_papers),
    "failed_coherence": len(failed_coherence),
    "dedup_removed": len(dedup_removed),
    "max_possible": len(all_raw_members),
}
with open("q2_uncovered_breakdown.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved to q2_uncovered_breakdown.json")
