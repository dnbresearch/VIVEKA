#!/usr/bin/env python3
"""
Edge Weight Component Ablation
================================
Removes one component at a time from the similarity function (Eq 1),
rebuilds the graph, detects communities, generates hypotheses, and judges.

Validates: which components of w(i,j) are essential?
  - Full:       3|B∩B| + 2|M∩M| + α|P∩P| + 2.5|A∩A|
  - No bench:   0|B∩B| + 2|M∩M| + α|P∩P| + 2.5|A∩A|
  - No methods: 3|B∩B| + 0|M∩M| + α|P∩P| + 2.5|A∩A|
  - No params:  3|B∩B| + 2|M∩M| + 0|P∩P| + 2.5|A∩A|
  - No ablation:3|B∩B| + 2|M∩M| + α|P∩P| + 0|A∩A|

Usage:
  python3 edge_weight_ablation.py \
      --phase1 ./validation_results/scale/phase1_incremental.json
"""

import json, os, sys, time, re, copy
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import networkx as nx

sys.path.insert(0, ".")
from community_hypothesis_v3 import (
    extract_features, normalize_venue, 
    detect_communities, score_community, GENERIC_METHODS
)

RESULTS_DIR = Path("./validation_results/edge_weight_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HYPOTHESIS_PROMPT = """You are a research advisor analyzing a cluster of {n_papers} ML papers from {venues}. These papers share {shared_desc}.

WHAT EACH MEMBER EXPLORED (lines with * contain actual measured values):
{member_details}

Generate 5 specific, actionable research hypotheses that combine findings across these papers.
Each must cite specific results from at least 2 papers.

Return ONLY a JSON array:
[{{"hypothesis":"specific experiment","motivation":"Paper A: metric=value, Paper B: metric=value","expected_outcome":"quantitative","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"]}}]"""

JUDGE_PROMPT = """Rate each hypothesis (1-5):
SPECIFICITY: exact params/values named? GROUNDEDNESS: cites specific numbers?
ACTIONABILITY: implementable today? CROSS-PAPER NOVELTY: requires 2+ papers?

{hypotheses}

JSON array: [{{"n":1,"specificity":4,"groundedness":3,"actionability":4,"cross_paper_novelty":3}}]"""


def parse_json_array(text):
    if not text: return []
    try: return json.loads(text)
    except: pass
    blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
    for b in blocks:
        b = b.strip()
        if b.startswith("["): 
            try: return json.loads(b)
            except: pass
    s, e = text.find("["), text.rfind("]")
    if s >= 0 and e > s:
        try: return json.loads(text[s:e+1])
        except: pass
    return []


def build_graph_ablated(papers, titles, venues, sim_threshold, ablate_component=None):
    """Build graph with one component removed."""
    G = nx.Graph()
    common_params = {"learning_rate", "batch_size", "epochs", "seed", "dataset"}
    
    # Inverted index
    bench_idx = defaultdict(set)
    method_idx = defaultdict(set)
    param_idx = defaultdict(set)
    
    for i, t in enumerate(titles):
        f = papers[t]
        for b in f["benchmarks"]: bench_idx[b].add(i)
        for m in f["methods"]: method_idx[m].add(i)
        for p in f["params"]: param_idx[p].add(i)
    
    candidates = set()
    for idx_set in list(bench_idx.values()) + list(method_idx.values()) + list(param_idx.values()):
        lst = list(idx_set)
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                candidates.add((min(lst[a], lst[b]), max(lst[a], lst[b])))
    
    for i, j in candidates:
        f1, f2 = papers[titles[i]], papers[titles[j]]
        
        # Compute weight with ablation
        w_bench = len(f1["benchmarks"] & f2["benchmarks"]) * 3.0
        w_method = len(f1["methods"] & f2["methods"]) * 2.0
        
        specific = (f1["params"] & f2["params"]) - common_params
        common = (f1["params"] & f2["params"]) & common_params
        w_param = len(specific) * 1.0 + len(common) * 0.3
        
        w_ablation = len(f1["ablation_params"] & f2["ablation_params"]) * 2.5
        
        # Apply ablation
        if ablate_component == "benchmark": w_bench = 0
        elif ablate_component == "method": w_method = 0
        elif ablate_component == "param": w_param = 0
        elif ablate_component == "ablation": w_ablation = 0
        
        w = w_bench + w_method + w_param + w_ablation
        
        if w >= sim_threshold:
            v1, v2 = titles[i], titles[j]
            G.add_edge(v1, v2, weight=w)
    
    return G


def generate_and_judge(communities, papers, insights_map, api_key, label):
    """Generate hypotheses for communities and judge them."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    all_scores = []
    total_hyps = 0
    
    for comm in communities[:5]:  # Top 5 communities
        members = comm["members"]
        shared = comm.get("shared_bench", [])
        
        # Build prompt
        member_details = ""
        for t in members[:10]:  # Limit context
            ins = insights_map.get(t, [])
            member_details += f"\n[{t[:60]}]\n"
            for i_item in ins[:4]:
                member_details += f"  [{i_item['type']}] {i_item['description'][:120]}\n"
        
        prompt = HYPOTHESIS_PROMPT.format(
            n_papers=len(members),
            venues=", ".join(comm.get("venues", ["various"])),
            shared_desc=", ".join(shared) if shared else "experimental overlap",
            member_details=member_details[:6000]
        )
        
        # Generate
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            hyps = parse_json_array(resp.content[0].text.strip())
        except Exception as e:
            print(f"      Gen error: {e}")
            hyps = []
        
        if not hyps:
            continue
        total_hyps += len(hyps)
        
        # Judge
        hyp_text = "\n".join(
            f"{i+1}. {h.get('hypothesis','?')[:200]}\n"
            f"   Cited: {h.get('cited_papers',[])[:3]}"
            for i, h in enumerate(hyps)
        )
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=2500,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(hypotheses=hyp_text)}]
            )
            scores = parse_json_array(resp.content[0].text.strip())
            for s in scores:
                dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
                mean = np.mean([s.get(d, 0) for d in dims if s.get(d, 0) > 0])
                if mean > 0:
                    all_scores.append(mean)
        except Exception as e:
            print(f"      Judge error: {e}")
        
        time.sleep(1)
    
    avg = np.mean(all_scores) if all_scores else 0
    return {"label": label, "mean": round(avg, 2), "n_hyps": total_hyps, 
            "n_scored": len(all_scores)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Only analyze graph structure, skip hypothesis generation")
    args = parser.parse_args()
    
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.skip_generate:
        print("Set ANTHROPIC_API_KEY for hypothesis generation, or use --skip-generate")
        sys.exit(1)
    
    print("=" * 70)
    print("EDGE WEIGHT COMPONENT ABLATION")
    print("=" * 70)
    
    with open(args.phase1) as f:
        p1 = json.load(f)
    
    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    papers, insights_map, venues = {}, {}, {}
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
    print(f"  Papers: {len(titles)}")
    
    # Run ablations
    configs = [
        ("Full model", None),
        ("No benchmarks (−3|B∩B|)", "benchmark"),
        ("No methods (−2|M∩M|)", "method"),
        ("No parameters (−α|P∩P|)", "param"),
        ("No ablations (−2.5|A∩A|)", "ablation"),
    ]
    
    results = []
    for label, ablate in configs:
        print(f"\n  --- {label} ---")
        G = build_graph_ablated(papers, titles, venues, args.threshold, ablate)
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"  Graph: {n_nodes} nodes, {n_edges} edges")
        
        if n_edges == 0:
            print(f"  No edges — component is ESSENTIAL at τ={args.threshold}")
            results.append({
                "label": label, "ablate": ablate or "none",
                "nodes": n_nodes, "edges": n_edges,
                "communities": 0, "avg_coherence": 0, "avg_size": 0,
                "mean_quality": 0, "n_hyps": 0
            })
            continue
        
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
            if len(final) >= 10:
                break
        
        n_comm = len(final)
        avg_coh = np.mean([g["coherence"] for g in final]) if final else 0
        avg_size = np.mean([g["size"] for g in final]) if final else 0
        shared = sum(1 for g in final if g.get("shared_bench"))
        
        print(f"  Communities: {n_comm}, Avg coherence: {avg_coh:.2f}, Shared bench: {shared}/{n_comm}")
        
        result = {
            "label": label, "ablate": ablate or "none",
            "nodes": n_nodes, "edges": n_edges,
            "communities": n_comm, "avg_coherence": round(avg_coh, 2),
            "avg_size": round(avg_size, 1), "shared_bench": f"{shared}/{n_comm}",
        }
        
        # Generate and judge hypotheses
        if not args.skip_generate and final:
            print(f"  Generating hypotheses...", end=" ", flush=True)
            gen_result = generate_and_judge(final, papers, insights_map, api_key, label)
            result["mean_quality"] = gen_result["mean"]
            result["n_hyps"] = gen_result["n_hyps"]
            print(f"mean={gen_result['mean']:.2f} ({gen_result['n_hyps']} hyps)")
        
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPONENT ABLATION RESULTS")
    print("="*70)
    print(f"\n  {'Config':<30} {'Edges':>7} {'#C':>4} {'Coh':>6} {'Quality':>8}")
    print(f"  {'-'*58}")
    for r in results:
        q = r.get('mean_quality', 'N/A')
        q_str = f"{q:.2f}" if isinstance(q, float) and q > 0 else "N/A"
        print(f"  {r['label']:<30} {r['edges']:>7} {r['communities']:>4} "
              f"{r['avg_coherence']:>6.2f} {q_str:>8}")
    
    # Find most important component
    full = next((r for r in results if r["ablate"] == "none"), None)
    if full:
        for r in results:
            if r["ablate"] != "none":
                edge_drop = 100 * (1 - r["edges"] / max(full["edges"], 1))
                comm_drop = full["communities"] - r["communities"]
                print(f"\n  Removing {r['ablate']}: {edge_drop:.0f}% edge loss, {comm_drop} fewer communities")
    
    # Save
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # LaTeX
    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print("="*70)
    print(r"""
\begin{table}[t]
\caption{Edge weight component ablation ($\tau$=6.0). Removing benchmarks eliminates all edges; removing methods has the least impact.}
\label{tab:weight_ablation}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|rrrr}
\toprule
\textbf{Configuration} & \textbf{Edges} & \textbf{\#C} & \textbf{Coh} & \textbf{Mean} \\
\midrule""")
    for r in results:
        q = r.get('mean_quality', 0)
        q_str = f"{q:.2f}" if isinstance(q, float) and q > 0 else "--"
        bold = "\\textbf" if r["ablate"] == "none" else ""
        label = r["label"].replace("−", "$-$").replace("∩", "$\\cap$")
        if bold:
            print(f"\\textbf{{{label}}} & \\textbf{{{r['edges']:,}}} & "
                  f"\\textbf{{{r['communities']}}} & \\textbf{{{r['avg_coherence']:.2f}}} & "
                  f"\\textbf{{{q_str}}} \\\\")
        else:
            print(f"{label} & {r['edges']:,} & {r['communities']} & "
                  f"{r['avg_coherence']:.2f} & {q_str} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
