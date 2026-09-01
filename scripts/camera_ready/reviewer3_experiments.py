#!/usr/bin/env python3
"""
Reviewer 3 — Quick Experiments
Q2: Extraction rate by venue and year
Q3: Community stability across τ values
"""
import json, os, sys
import numpy as np
from collections import defaultdict

# ---- Adjust this path to your phase1 data ----
PHASE1_PATH = "validation_results/scale/phase1_incremental.json"

sys.path.insert(0, ".")


def run_q2_extraction_by_year():
    """Q2: Do newer papers have lower extraction rates?"""
    print("=" * 65)
    print("Q2: Extraction Rate by Venue and Year")
    print("=" * 65)

    with open(PHASE1_PATH) as f:
        data = json.load(f)

    # Parse year from venue or paper metadata
    by_year = defaultdict(lambda: {"total": 0, "has_insights": 0, "n_insights": 0})
    by_venue = defaultdict(lambda: {"total": 0, "has_insights": 0})
    by_venue_year = defaultdict(lambda: {"total": 0, "has_insights": 0})

    for r in data:
        venue = r.get("venue", "unknown")
        # Try to extract year from venue string or paper metadata
        year = None
        for y in range(2018, 2026):
            if str(y) in venue or str(y) in r.get("title", ""):
                year = y
                break
        if year is None:
            # Try from URL or other fields
            url = r.get("url", "") or r.get("repo_url", "")
            for y in range(2018, 2026):
                if str(y) in url:
                    year = y
                    break
        if year is None:
            year = "unknown"

        has_insights = r.get("n_insights", 0) > 0
        n_insights = r.get("n_insights", 0)

        by_year[year]["total"] += 1
        by_year[year]["has_insights"] += int(has_insights)
        by_year[year]["n_insights"] += n_insights

        # Clean venue name
        venue_clean = venue.split("/")[0].split("_")[0].strip().upper()[:10]
        by_venue[venue_clean]["total"] += 1
        by_venue[venue_clean]["has_insights"] += int(has_insights)

        if year != "unknown":
            by_venue_year[(venue_clean, year)]["total"] += 1
            by_venue_year[(venue_clean, year)]["has_insights"] += int(has_insights)

    # Print by year
    print(f"\n  {'Year':<10} {'Total':>8} {'With Insights':>15} {'Rate':>8} {'Avg Insights':>14}")
    print(f"  {'-'*58}")
    for year in sorted(by_year.keys()):
        d = by_year[year]
        rate = d["has_insights"] / max(d["total"], 1) * 100
        avg = d["n_insights"] / max(d["has_insights"], 1)
        print(f"  {str(year):<10} {d['total']:>8} {d['has_insights']:>15} {rate:>7.1f}% {avg:>13.1f}")

    # Print by venue (top 15)
    print(f"\n  {'Venue':<12} {'Total':>8} {'With Insights':>15} {'Rate':>8}")
    print(f"  {'-'*45}")
    sorted_venues = sorted(by_venue.items(), key=lambda x: -x[1]["total"])
    for venue, d in sorted_venues[:15]:
        rate = d["has_insights"] / max(d["total"], 1) * 100
        print(f"  {venue:<12} {d['total']:>8} {d['has_insights']:>15} {rate:>7.1f}%")

    # Statistical test: is extraction rate declining?
    years = sorted([y for y in by_year.keys() if isinstance(y, int)])
    if len(years) >= 3:
        rates = [by_year[y]["has_insights"] / max(by_year[y]["total"], 1) for y in years]
        from scipy.stats import spearmanr
        rho, p = spearmanr(years, rates)
        print(f"\n  Trend test (Spearman): ρ={rho:.3f}, p={p:.3f}")
        if p > 0.05:
            print(f"  → No significant decline in extraction rates over time")
        elif rho < 0:
            print(f"  → Significant decline: newer papers have lower extraction rates")
        else:
            print(f"  → Significant increase: newer papers have higher extraction rates")

    # Save results
    results = {
        "by_year": {str(k): v for k, v in by_year.items()},
        "by_venue": dict(sorted(by_venue.items(), key=lambda x: -x[1]["total"])[:15]),
        "trend": {"years": years, "rates": rates} if len(years) >= 3 else None,
    }
    with open("q2_extraction_by_year.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to q2_extraction_by_year.json")


def run_q3_cross_tau_stability():
    """Q3: How stable are communities across different τ values?"""
    print(f"\n{'='*65}")
    print("Q3: Community Stability Across τ Values")
    print("=" * 65)

    # Import pipeline components
    try:
        from community_hypothesis_v3 import (
            extract_features, normalize_venue, build_graph_batched,
            detect_communities, score_community, GENERIC_METHODS
        )
    except ImportError:
        print("  ERROR: community_hypothesis_v3.py not found.")
        print("  Make sure you're in the viveka directory.")
        return

    with open(PHASE1_PATH) as f:
        p1 = json.load(f)

    # Build paper features
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
    print(f"  Papers: {len(titles)}")

    # Build communities at multiple τ values
    tau_values = [2.0, 3.0, 4.0, 5.0, 6.0]
    tau_communities = {}
    tau_memberships = {}

    for tau in tau_values:
        G = build_graph_batched(papers, titles, venues, sim_threshold=tau)
        raw_groups = detect_communities(G, papers, venues, min_community=3)
        scored = []
        for members in raw_groups:
            info = score_community(members, papers, venues)
            if info and info["coherence"] >= 1.0:
                scored.append(info)
        scored.sort(key=lambda x: -x["score"])

        # Deduplicate
        final = []
        used = set()
        for g in scored:
            ms = set(g["members"])
            if len(ms & used) / max(len(ms), 1) < 0.5:
                final.append(g)
                used.update(ms)

        # Store membership as dict: paper → community_id
        membership = {}
        for ci, comm in enumerate(final):
            for m in comm["members"]:
                membership[m] = ci

        tau_communities[tau] = final
        tau_memberships[tau] = membership

        n_covered = len(membership)
        n_comms = len(final)
        avg_coh = np.mean([c["coherence"] for c in final]) if final else 0
        print(f"  τ={tau}: {n_comms} communities, {n_covered} papers covered, "
              f"avg coherence={avg_coh:.2f}")

    # Compute NMI between all pairs of τ values
    print(f"\n  --- Pairwise NMI (Normalized Mutual Information) ---")
    print(f"  {'':>8}", end="")
    for tau in tau_values:
        print(f"  τ={tau:>3}", end="")
    print()

    from sklearn.metrics import normalized_mutual_info_score

    nmi_matrix = np.zeros((len(tau_values), len(tau_values)))

    for i, tau_i in enumerate(tau_values):
        print(f"  τ={tau_i:>3.1f}", end="")
        for j, tau_j in enumerate(tau_values):
            # Get papers present in BOTH
            mem_i = tau_memberships[tau_i]
            mem_j = tau_memberships[tau_j]
            common = set(mem_i.keys()) & set(mem_j.keys())

            if len(common) < 5:
                nmi = float('nan')
            else:
                labels_i = [mem_i[p] for p in common]
                labels_j = [mem_j[p] for p in common]
                nmi = normalized_mutual_info_score(labels_i, labels_j)

            nmi_matrix[i, j] = nmi
            if np.isnan(nmi):
                print(f"  {'—':>5}", end="")
            else:
                print(f"  {nmi:>5.3f}", end="")
        print()

    # Check nestedness: are τ=6.0 communities subsets of τ=4.0?
    print(f"\n  --- Nestedness Analysis ---")
    for tau_high, tau_low in [(6.0, 4.0), (5.0, 3.0), (4.0, 2.0)]:
        comms_high = tau_communities.get(tau_high, [])
        mem_low = tau_memberships.get(tau_low, {})

        nested_count = 0
        total = 0
        for comm in comms_high:
            members = set(comm["members"])
            # Check: are all members in the same community at lower τ?
            low_ids = set()
            for m in members:
                if m in mem_low:
                    low_ids.add(mem_low[m])
            if len(low_ids) == 1:
                nested_count += 1
            total += 1

        if total > 0:
            print(f"  τ={tau_high}→{tau_low}: {nested_count}/{total} communities "
                  f"are strict subsets ({100*nested_count/total:.0f}%)")
        else:
            print(f"  τ={tau_high}→{tau_low}: no communities to compare")

    # Jaccard stability: for each pair of adjacent τ, how much do top communities overlap?
    print(f"\n  --- Membership Overlap (Jaccard) ---")
    for i in range(len(tau_values) - 1):
        tau_a = tau_values[i]
        tau_b = tau_values[i + 1]
        mem_a = set(tau_memberships[tau_a].keys())
        mem_b = set(tau_memberships[tau_b].keys())

        intersection = len(mem_a & mem_b)
        union = len(mem_a | mem_b)
        jaccard = intersection / max(union, 1)

        print(f"  τ={tau_a}→{tau_b}: |A|={len(mem_a)}, |B|={len(mem_b)}, "
              f"overlap={intersection}, Jaccard={jaccard:.3f}")

    # Save
    results = {
        "tau_values": tau_values,
        "communities_per_tau": {str(t): len(c) for t, c in tau_communities.items()},
        "coverage_per_tau": {str(t): len(m) for t, m in tau_memberships.items()},
        "nmi_matrix": nmi_matrix.tolist(),
    }
    with open("q3_cross_tau_stability.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to q3_cross_tau_stability.json")


def run_q5_benchmark_coverage():
    """Q5 bonus: Estimate benchmark false-negative rate."""
    print(f"\n{'='*65}")
    print("Q5 (Bonus): Benchmark Vocabulary Coverage Check")
    print("=" * 65)

    KNOWN_BENCHMARKS = {
        "imagenet", "cifar", "coco", "voc", "cityscapes", "ade20k",
        "mnist", "svhn", "celeba", "lfw", "kinetics", "ucf",
        "sst", "glue", "squad", "wmt", "wikitext", "lambada",
        "modelnet", "shapenet", "scannet", "ntu", "kitti",
        "mvtec", "visa", "mpdd", "btad"
    }

    with open(PHASE1_PATH) as f:
        data = json.load(f)

    detected = 0
    missed = 0
    missed_examples = []

    for r in data:
        insights = r.get("_insights", [])
        for ins in insights:
            # Check if insight mentions a benchmark
            ins_str = json.dumps(ins).lower()

            # Check for known benchmarks
            found_known = any(b in ins_str for b in KNOWN_BENCHMARKS)

            # Check for potential benchmarks we're missing
            # Look for dataset-like patterns not in our list
            import re
            potential = re.findall(r'"dataset["\s:]*["\s]*([^"]{3,30})"', ins_str)
            potential += re.findall(r'"data_dir["\s:]*["\s]*([^"]{3,30})"', ins_str)
            potential += re.findall(r'"benchmark["\s:]*["\s]*([^"]{3,30})"', ins_str)

            for p in potential:
                p_clean = p.strip().lower()
                if not any(b in p_clean for b in KNOWN_BENCHMARKS):
                    if len(p_clean) > 3 and p_clean not in {"none", "null", "true", "false", "data", "train", "test", "val"}:
                        missed += 1
                        if len(missed_examples) < 20:
                            missed_examples.append(p_clean)
                else:
                    detected += 1

    total = detected + missed
    if total > 0:
        miss_rate = missed / total * 100
        print(f"  Detected benchmarks: {detected}")
        print(f"  Potential missed: {missed}")
        print(f"  Estimated false-negative rate: {miss_rate:.1f}%")
        print(f"  Sample missed: {missed_examples[:10]}")
    else:
        print("  Could not estimate (no benchmark patterns found in insights)")


if __name__ == "__main__":
    print("Reviewer 3 — Quick Experiments\n")

    if not os.path.exists(PHASE1_PATH):
        print(f"  ERROR: {PHASE1_PATH} not found.")
        print(f"  Please set PHASE1_PATH at the top of this script.")
        sys.exit(1)

    run_q2_extraction_by_year()
    run_q3_cross_tau_stability()

    # Q5 is bonus — uncomment if you want to run it
    # run_q5_benchmark_coverage()

    print(f"\n{'='*65}")
    print("Done! Share q2_extraction_by_year.json and")
    print("q3_cross_tau_stability.json with results.")
    print("=" * 65)
