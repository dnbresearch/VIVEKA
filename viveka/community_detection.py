#!/usr/bin/env python3
"""
Community Hypothesis Generation v3: Large-Scale Cross-Venue
=============================================================
Handles 3,000+ graph-ready repos across 28 venues.

Key changes from v2:
  - Reads phase1_incremental.json directly (not baseline_results.json)
  - Batched similarity computation (avoids 13M+ comparisons)
  - Uses Louvain communities + clique refinement (scales to 3000+ nodes)
  - Tracks cross-venue communities (CVPR+EMNLP = novel)
  - Adds venue diversity scoring to community ranking

Pipeline:
  1. Load phase1_incremental.json → extract features from graph-ready repos
  2. Build similarity graph (batched, with venue-aware edges)
  3. Louvain community detection → refine with clique filtering
  4. Rank communities by coherence × venue diversity
  5. LLM generates hypotheses from top communities

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 community_hypothesis_v3.py --phase1 ./validation_results/scale/phase1_incremental.json

  # Without LLM (just build graph + communities)
  python3 community_hypothesis_v3.py --phase1 ./validation_results/scale/phase1_incremental.json --skip-llm
"""

import json, os, re, sys, time, random, math
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

RESULTS_DIR = Path("./validation_results/community_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================
# Feature extraction (same canonical mappings as v2, extended)
# =========================================================================
NOISE_RE = [re.compile(p, re.I) for p in [r".*path$",r".*dir$",r".*file$",r".*root$",
    r".*print_freq.*",r".*log_freq.*",r".*save_freq.*",r".*verbose.*",r".*workers$",
    r".*num_workers.*",r".*exp_name.*",r".*checkpoint.*",r".*output.*"]]

PARAM_CANON = {}
for c, aa in {"learning_rate":["lr","learning_rate","optim.lr","base_lr"],
    "batch_size":["batch_size","bs","train.batch_size","per_device_train_batch_size"],
    "epochs":["epochs","num_epochs","max_epochs","n_epochs"],
    "weight_decay":["weight_decay","wd"],"dropout":["dropout","drop_rate","drop_path_rate"],
    "hidden_dim":["hidden_dim","hidden_size","d_model","dim","embed_dim","width","channels"],
    "num_layers":["num_layers","n_layers","depth","num_blocks"],
    "num_heads":["num_heads","n_heads","num_attention_heads"],
    "backbone":["backbone","encoder","arch","model","model_name","backbone_name","network.name"],
    "optimizer":["optimizer","opt","optim"],
    "image_size":["image_size","img_size","input_size","resolution","img_resize"],
    "dataset":["dataset","data","dataset_name","data.name","data.dataset_name"],
    "seed":["seed","random_seed"],"warmup":["warmup","warmup_steps","warmup_epochs"],
    "temperature":["temperature","temp","tau"],"momentum":["momentum"],
    "scheduler":["scheduler","lr_scheduler"]}.items():
    for a in aa: PARAM_CANON[a.lower()] = c

KNOWN_BENCH = {"mvtec":"MVTec-AD","btad":"BTAD","visa":"VisA","mpdd":"MPDD",
    "cifar10":"CIFAR-10","cifar-10":"CIFAR-10","cifar100":"CIFAR-100",
    "imagenet":"ImageNet","coco":"COCO","voc":"VOC","ade20k":"ADE20K","ucf":"UCF",
    "kinetics":"Kinetics","ntu":"NTU","squad":"SQuAD","glue":"GLUE","superglue":"SuperGLUE",
    "wmt":"WMT","scannet":"ScanNet","shapenet":"ShapeNet","modelnet":"ModelNet",
    "mnist":"MNIST","svhn":"SVHN","celeba":"CelebA","cityscapes":"Cityscapes",
    "lfw":"LFW","qm9":"QM9","ptb":"PTB","wiki":"WikiText"}

METHOD_KW = {"vit":"vision_transformer","resnet":"resnet","transformer":"transformer",
    "unet":"unet","diffusion":"diffusion","contrastive":"contrastive","clip":"clip",
    "anomaly":"anomaly_detection","segmentation":"segmentation","detection":"object_detection",
    "pretrain":"pretraining","self_supervised":"self_supervised","attention":"attention",
    "graph":"graph_nn","reinforcement":"rl","patchcore":"patchcore","normalization":"normalization",
    "gan":"gan","bert":"bert","gpt":"gpt","llama":"llama","lora":"lora",
    "fine_tun":"finetuning","distill":"distillation","pruning":"pruning",
    "quantiz":"quantization","augment":"augmentation","curriculum":"curriculum",
    "adversar":"adversarial","federat":"federated","meta_learn":"meta_learning",
    "few_shot":"few_shot","zero_shot":"zero_shot","prompt":"prompt_tuning"}

GENERIC_METHODS = {"attention", "classification", "transformer"}

def canon_p(p):
    pl = str(p).lower().strip()
    if pl in PARAM_CANON: return PARAM_CANON[pl]
    if "." in pl:
        l = pl.rsplit(".",1)[-1]
        if l in PARAM_CANON: return PARAM_CANON[l]
    return None

def extract_features(title, insights):
    f = {"params":set(),"benchmarks":set(),"methods":set(),"ablation_params":set(),
         "metrics":set(),"param_values":{}}
    for kw, m in METHOD_KW.items():
        if kw in title.lower(): f["methods"].add(m)
    for ins in insights:
        ev = ins.get("evidence",{}); desc = ins.get("description","").lower()
        for p in ins.get("params",[]):
            if any(r.match(str(p)) for r in NOISE_RE): continue
            c = canon_p(p)
            if c: f["params"].add(c)
        for kw, b in KNOWN_BENCH.items():
            if kw in desc: f["benchmarks"].add(b)
            for fld in ["datasets","param","metric"]:
                if kw in str(ev.get(fld,"")).lower(): f["benchmarks"].add(b)
            for row in ev.get("all_results",[]):
                if kw in str(row.get("label","")).lower(): f["benchmarks"].add(b)
        for kw, m in METHOD_KW.items():
            if kw in desc: f["methods"].add(m)
        if ev.get("metric"): f["metrics"].add(str(ev["metric"]).lower())
        if ins["type"] in ("clean_ablation","cross_dataset_ablation"):
            c = canon_p(ev.get("param",""))
            if c: f["ablation_params"].add(c)
        if ins["type"]=="parameter_range":
            c = canon_p(ev.get("param","")); vals = ev.get("values",[])
            if c and vals: f["param_values"][c] = set(str(v) for v in vals[:10])
        if ins["type"]=="experiment_family":
            for p in ev.get("varying",{}).keys():
                c = canon_p(p)
                if c: f["params"].add(c)
        if ins["type"]=="config_result_link":
            for p in ev.get("config_changes",{}).keys():
                c = canon_p(p)
                if c: f["ablation_params"].add(c)
    return f

def compute_sim(f1, f2):
    score = 0.0
    sb = f1["benchmarks"]&f2["benchmarks"]; score += len(sb)*3.0
    sm = f1["methods"]&f2["methods"]; score += len(sm)*2.0
    common = {"learning_rate","batch_size","epochs","seed","dataset"}
    sp = f1["params"]&f2["params"]
    score += len(sp-common)*1.0 + len(sp&common)*0.3
    score += len(f1["ablation_params"]&f2["ablation_params"])*2.5
    score += len(f1["metrics"]&f2["metrics"])*1.5
    for p in set(f1["param_values"])&set(f2["param_values"]):
        score += len(f1["param_values"][p]&f2["param_values"][p])*0.5
    return score

def normalize_venue(v):
    """Normalize venue names for cleaner grouping."""
    v = str(v).strip()
    if "NeurIPS" in v: return "NeurIPS"
    if "ICLR" in v: return "ICLR"
    if "ICML" in v: return "ICML"
    if "CVPR" in v: return "CVPR"
    if "ECCV" in v: return "ECCV"
    if "ACL" in v and "NAACL" not in v: return "ACL"
    if "NAACL" in v: return "NAACL"
    if "EMNLP" in v: return "EMNLP"
    if "AAAI" in v: return "AAAI"
    if "KDD" in v: return "KDD"
    if "ICDM" in v: return "ICDM"
    if "UAI" in v: return "UAI"
    if "OCP" in v: return "OCP"
    return v[:15]

# =========================================================================
# Optimized graph construction
# =========================================================================
def build_graph_batched(papers, titles, venues, sim_threshold=2.0, max_neighbors=50):
    """
    Build similarity graph with O(n × k) instead of O(n²).
    Strategy: bucket papers by shared benchmarks/methods, only compute
    pairwise similarity within buckets.
    """
    print(f"  Building graph for {len(titles)} papers...")

    # Build inverted index: feature → papers that have it
    bench_index = defaultdict(set)
    method_index = defaultdict(set)
    ablation_index = defaultdict(set)
    metric_index = defaultdict(set)

    for i, t in enumerate(titles):
        f = papers[t]
        for b in f["benchmarks"]: bench_index[b].add(i)
        for m in f["methods"]: method_index[m].add(i)
        for a in f["ablation_params"]: ablation_index[a].add(i)
        for m in f["metrics"]: metric_index[m].add(i)

    # Build candidate pairs (papers that share at least one feature)
    candidates = defaultdict(float)  # (i,j) → estimated minimum similarity
    for idx_set in bench_index.values():
        lst = list(idx_set)
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                pair = (min(lst[a],lst[b]), max(lst[a],lst[b]))
                candidates[pair] = max(candidates[pair], 3.0)  # benchmark match = 3.0

    for idx_set in method_index.values():
        lst = list(idx_set)
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                pair = (min(lst[a],lst[b]), max(lst[a],lst[b]))
                candidates[pair] = max(candidates[pair], 2.0)

    for idx_set in ablation_index.values():
        lst = list(idx_set)
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                pair = (min(lst[a],lst[b]), max(lst[a],lst[b]))
                candidates[pair] = max(candidates[pair], 2.5)

    for idx_set in metric_index.values():
        if len(idx_set) > 500: continue  # skip very common metrics
        lst = list(idx_set)
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                pair = (min(lst[a],lst[b]), max(lst[a],lst[b]))
                candidates[pair] = max(candidates[pair], 1.5)

    print(f"  Candidate pairs: {len(candidates):,} (vs {len(titles)*(len(titles)-1)//2:,} brute force)")

    # Compute actual similarity for candidates
    G = nx.Graph()
    for t in titles: G.add_node(t)

    edges_added = 0
    for (i,j), min_sim in candidates.items():
        if min_sim < sim_threshold * 0.5: continue  # skip unlikely pairs
        s = compute_sim(papers[titles[i]], papers[titles[j]])
        if s >= sim_threshold:
            G.add_edge(titles[i], titles[j], weight=s)
            edges_added += 1

    print(f"  Graph: {G.number_of_nodes()} nodes, {edges_added:,} edges")
    return G

# =========================================================================
# Community detection (Louvain + clique refinement)
# =========================================================================
def detect_communities(G, papers, venues, min_community=3, max_communities=100):
    """
    Louvain for initial communities, then refine each by finding
    coherent cliques within the community.
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=1.2)
    except ImportError:
        print("  WARNING: python-louvain not installed, using connected components")
        partition = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for n in comp: partition[n] = i

    # Group by community
    comm_members = defaultdict(list)
    for node, comm_id in partition.items():
        comm_members[comm_id].append(node)

    raw_communities = [m for m in comm_members.values() if len(m) >= min_community]
    print(f"  Louvain: {len(comm_members)} communities, {len(raw_communities)} with ≥{min_community} members")

    # Refine: for each community, find coherent sub-groups
    refined = []
    for members in raw_communities:
        sub = G.subgraph(members)

        # For small communities, treat as one group
        if len(members) <= 8:
            refined.append(members)
            continue

        # For larger communities, find cliques ≥4 within them
        try:
            cliques = [c for c in nx.find_cliques(sub) if len(c) >= 4]
            if cliques:
                # Take top cliques by size, deduplicate
                cliques.sort(key=len, reverse=True)
                used = set()
                for clique in cliques[:10]:
                    clique_set = set(clique)
                    if len(clique_set & used) / len(clique_set) < 0.5:
                        refined.append(list(clique_set))
                        used.update(clique_set)
            else:
                # No cliques ≥4, keep whole community if dense enough
                density = nx.density(sub)
                if density > 0.3:
                    refined.append(members[:20])  # cap size
        except Exception:
            refined.append(members[:15])

    print(f"  Refined to {len(refined)} groups")
    return refined

def score_community(members, papers, venues):
    """Score a community by coherence × venue diversity."""
    feats = [papers[t] for t in members if t in papers]
    if len(feats) < 2: return None

    # Shared features
    sb = set.intersection(*[f["benchmarks"] for f in feats]) if feats else set()
    sm = set.intersection(*[f["methods"] for f in feats]) - GENERIC_METHODS if feats else set()
    sa = set.intersection(*[f["ablation_params"] for f in feats]) if feats else set()
    shared_metrics = set.intersection(*[f["metrics"] for f in feats]) if feats else set()

    # Union features (for coverage stats)
    all_bench = set.union(*[f["benchmarks"] for f in feats]) if feats else set()
    all_methods = set.union(*[f["methods"] for f in feats]) - GENERIC_METHODS if feats else set()
    all_ablations = set.union(*[f["ablation_params"] for f in feats]) if feats else set()

    coherence = len(sb)*3 + len(sm)*2 + len(sa)*1.5 + len(shared_metrics)*1.0

    # Venue diversity
    member_venues = set()
    for t in members:
        if t in venues: member_venues.add(venues[t])
    venue_diversity = len(member_venues)

    # Cross-domain bonus: papers from different domains
    domains = set()
    domain_map = {"CVPR":"CV","ECCV":"CV","ICLR":"ML","NeurIPS":"ML","ICML":"ML",
                  "ACL":"NLP","EMNLP":"NLP","NAACL":"NLP","AAAI":"AI","KDD":"DM","ICDM":"DM"}
    for v in member_venues:
        domains.add(domain_map.get(v, "other"))
    cross_domain = len(domains) > 1

    # Combined score
    score = coherence * (1 + 0.3 * venue_diversity) * (1.5 if cross_domain else 1.0)

    return {
        "members": members,
        "size": len(members),
        "shared_bench": sorted(sb),
        "shared_methods": sorted(sm),
        "shared_ablations": sorted(sa),
        "shared_metrics": sorted(shared_metrics),
        "all_bench": sorted(all_bench),
        "all_methods": sorted(all_methods),
        "all_ablations": sorted(all_ablations),
        "coherence": round(coherence, 2),
        "venue_diversity": venue_diversity,
        "venues": sorted(member_venues),
        "cross_domain": cross_domain,
        "domains": sorted(domains),
        "score": round(score, 2),
    }

# =========================================================================
# Hypothesis generation
# =========================================================================
HYPOTHESIS_PROMPT = """You are a research advisor analyzing a cluster of {n_papers} ML papers from {venues}. These papers are experimentally related — they share {shared_desc}.

WHAT EACH MEMBER EXPLORED:
{member_details}

COMMUNITY EXPERIMENT COVERAGE:
- Shared benchmarks: {shared_bench}
- All benchmarks tested: {all_bench}
- Parameters ablated: {ablation_coverage}
- Key metrics: {metrics}
- Venues represented: {venue_list}

YOUR TASK: Find experiments that the community's work implies would be valuable but that NO member has actually tried. Focus especially on cross-venue transfer opportunities.

Generate 3-5 specific, actionable research hypotheses.

For each hypothesis:
1. State the SPECIFIC experiment (exact parameter, values, benchmark)
2. Cite which members' results motivate it (minimum 2 papers)
3. Predict the expected outcome based on observed patterns
4. Rate confidence: high/medium/low

RULES:
- Every hypothesis must be grounded in at least 2 members' actual results
- Be maximally specific: "Try ViT-L-14 backbone at 518px on VisA dataset" not "try larger models"
- Focus on COMBINATIONS that exist in the group's parameter space but haven't been tested together
- Identify where one paper's finding could transfer to another paper's setting
- Prioritize cross-venue transfer hypotheses (e.g., CVPR technique → EMNLP benchmark)

Return ONLY a JSON array:
[{{"hypothesis":"specific experiment","motivation":"which papers and why","expected_outcome":"predicted effect","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"],"gap_type":"unexplored_combination|cross_venue_transfer|untested_transfer|missing_ablation|contradicted_assumption"}}]"""

def generate_hypotheses_for_group(group, papers, insights_map, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    members = group["members"]
    shared_bench = group.get("shared_bench", [])
    shared_methods = group.get("shared_methods", [])

    shared_parts = []
    if shared_bench: shared_parts.append(f"benchmark(s): {', '.join(shared_bench)}")
    if shared_methods: shared_parts.append(f"method(s): {', '.join(shared_methods)}")
    shared_desc = " and ".join(shared_parts) if shared_parts else "experimental overlap"

    member_details = ""
    bench_coverage = Counter()
    abl_coverage = Counter()
    metrics = Counter()
    for t in members:
        f = papers.get(t, {})
        ins = insights_map.get(t, [])
        venue = group.get("venues", ["?"])
        member_details += f"\n[{t}]\n"
        if f.get("benchmarks"):
            member_details += f"  Benchmarks: {sorted(f['benchmarks'])}\n"
        if f.get("methods"):
            member_details += f"  Methods: {sorted(f['methods'])}\n"
        for b in f.get("benchmarks",[]): bench_coverage[b] += 1
        for a in f.get("ablation_params",[]): abl_coverage[a] += 1
        for m in f.get("metrics",[]): metrics[m] += 1
        # Include top insights (prioritize config_result_link and clean_ablation)
        sorted_ins = sorted(ins, key=lambda x: (
            x["type"] in ("config_result_link","clean_ablation","cross_dataset_ablation"),
            x.get("significance", 0)
        ), reverse=True)
        for i in sorted_ins[:6]:
            member_details += f"  [{i['type']}] {i['description'][:150]}\n"

    prompt = HYPOTHESIS_PROMPT.format(
        n_papers=len(members),
        venues=", ".join(group.get("venues", ["various"])),
        shared_desc=shared_desc,
        member_details=member_details[:8000],
        shared_bench=", ".join(shared_bench) or "none shared (diverse)",
        all_bench=", ".join(f"{b}({n})" for b,n in bench_coverage.most_common()) or "various",
        ablation_coverage=", ".join(f"{a}({n})" for a,n in abl_coverage.most_common()) or "none",
        metrics=", ".join(m for m,_ in metrics.most_common(8)) or "various",
        venue_list=", ".join(group.get("venues", ["?"])),
    )

    try:
        resp = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=3000,
                                       messages=[{"role":"user","content":prompt}])
        text = resp.content[0].text.strip()
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        if "```" in text:
            for part in text.split("```"):
                p = part.strip()
                if p.startswith("json"): p = p[4:].strip()
                if p.startswith("["): text = p; break
        s, e = text.find("["), text.rfind("]")
        if s >= 0 and e > s: text = text[s:e+1]
        hypotheses = json.loads(text)
        return hypotheses if isinstance(hypotheses, list) else [], tokens
    except Exception as ex:
        print(f"    LLM error: {ex}")
        return [], 0

# =========================================================================
# Main
# =========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True, help="Path to phase1_incremental.json")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--sim-threshold", type=float, default=2.0)
    parser.add_argument("--max-groups", type=int, default=50, help="Max groups for hypothesis generation")
    parser.add_argument("--min-community", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key and not args.skip_llm:
        print("Set ANTHROPIC_API_KEY or use --skip-llm"); sys.exit(1)

    # Load phase1 results
    print("="*70)
    print("COMMUNITY HYPOTHESIS GENERATION v3 (LARGE-SCALE)")
    print("="*70)

    with open(args.phase1) as f:
        p1 = json.load(f)
    print(f"Loaded {len(p1)} repos from phase1")

    # Filter to graph-ready repos
    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    print(f"Graph-ready: {len(graph_ready)} repos")

    # Extract features
    print(f"\n--- Feature Extraction ---")
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
    print(f"Papers with extractable features: {len(titles)}")

    # Feature statistics
    all_bench = Counter()
    all_methods = Counter()
    all_ablations = Counter()
    venue_counts = Counter()
    for t in titles:
        f = papers[t]
        for b in f["benchmarks"]: all_bench[b] += 1
        for m in f["methods"]: all_methods[m] += 1
        for a in f["ablation_params"]: all_ablations[a] += 1
        venue_counts[venues.get(t, "?")] += 1

    print(f"\n  Venues: {dict(venue_counts.most_common())}")
    print(f"  Top benchmarks: {dict(all_bench.most_common(10))}")
    print(f"  Top methods: {dict(all_methods.most_common(10))}")
    print(f"  Top ablated params: {dict(all_ablations.most_common(10))}")

    # Build graph
    print(f"\n--- Graph Construction ---")
    G = build_graph_batched(papers, titles, venues, sim_threshold=args.sim_threshold)

    # Community detection
    print(f"\n--- Community Detection ---")
    raw_groups = detect_communities(G, papers, venues, min_community=args.min_community)

    # Score and rank communities
    scored_groups = []
    for members in raw_groups:
        info = score_community(members, papers, venues)
        if info and info["coherence"] >= 1.0:
            scored_groups.append(info)

    scored_groups.sort(key=lambda x: -x["score"])
    print(f"  Scored groups: {len(scored_groups)}")

    # Deduplicate overlapping groups
    final_groups = []
    used = set()
    for g in scored_groups:
        members_set = set(g["members"])
        overlap = len(members_set & used) / max(len(members_set), 1)
        if overlap < 0.5:
            final_groups.append(g)
            used.update(members_set)
        if len(final_groups) >= args.max_groups:
            break

    print(f"  Final groups (deduped, top {args.max_groups}): {len(final_groups)}")

    # Print community summary
    print(f"\n{'='*70}")
    print("COMMUNITY SUMMARY")
    print("="*70)
    cross_venue_count = 0
    for i, g in enumerate(final_groups[:30]):
        cv = " ★CROSS-DOMAIN" if g["cross_domain"] else ""
        if g["cross_domain"]: cross_venue_count += 1
        print(f"\n  Group {i+1} ({g['size']} papers, score={g['score']:.1f}{cv}):")
        print(f"    Venues: {g['venues']}")
        print(f"    Shared bench: {g['shared_bench'] or 'none'}")
        print(f"    Shared methods: {g['shared_methods'] or 'none'}")
        print(f"    All bench: {g['all_bench'][:5]}")
        for t in g["members"][:4]:
            v = venues.get(t, "?")
            print(f"      [{v}] {t[:65]}")
        if len(g["members"]) > 4:
            print(f"      ... +{len(g['members'])-4} more")

    print(f"\n  Cross-domain groups: {cross_venue_count}/{len(final_groups)}")

    # Save groups
    with open(RESULTS_DIR / "groups.json", "w") as f:
        json.dump(final_groups, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))

    # Save graph stats
    graph_stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_papers_featured": len(titles),
        "n_graph_ready": len(graph_ready),
        "n_communities_raw": len(raw_groups),
        "n_communities_scored": len(scored_groups),
        "n_communities_final": len(final_groups),
        "n_cross_domain": cross_venue_count,
        "venue_distribution": dict(venue_counts.most_common()),
        "top_benchmarks": dict(all_bench.most_common(15)),
        "top_methods": dict(all_methods.most_common(15)),
    }
    with open(RESULTS_DIR / "graph_stats.json", "w") as f:
        json.dump(graph_stats, f, indent=2)

    if args.skip_llm:
        print(f"\nSkipping LLM (--skip-llm). Groups saved to {RESULTS_DIR}/")
        return

    # Generate hypotheses
    print(f"\n{'='*70}")
    print("HYPOTHESIS GENERATION")
    print("="*70)

    all_hyp = []
    total_tokens = 0
    for i, g in enumerate(final_groups):
        cv = " ★" if g["cross_domain"] else ""
        print(f"\nGroup {i+1}/{len(final_groups)} ({g['size']} papers, "
              f"venues={g['venues']}{cv}):")
        hyps, tokens = generate_hypotheses_for_group(g, papers, insights_map, api_key)
        total_tokens += tokens
        grounded = sum(1 for h in hyps if len(h.get("cited_papers",[])) >= 2)
        cross_venue_hyps = sum(1 for h in hyps if h.get("gap_type") == "cross_venue_transfer")

        print(f"  → {len(hyps)} hypotheses ({grounded} grounded, {cross_venue_hyps} cross-venue)")

        for h in hyps:
            print(f"\n    H: {h.get('hypothesis','?')[:130]}")
            print(f"       Confidence: {h.get('confidence','?')} | Gap: {h.get('gap_type','?')}")
            print(f"       Cited: {[p[:35] for p in h.get('cited_papers',[])[:3]]}")

        all_hyp.append({
            "group_id": i+1,
            "group_size": g["size"],
            "shared_bench": g["shared_bench"],
            "shared_methods": g["shared_methods"],
            "venues": g["venues"],
            "cross_domain": g["cross_domain"],
            "domains": g["domains"],
            "members": g["members"],
            "hypotheses": hyps,
            "n_grounded": grounded,
            "n_cross_venue": cross_venue_hyps,
            "tokens": tokens,
        })
        time.sleep(0.5)  # rate limit

    # Final summary
    total_h = sum(len(r["hypotheses"]) for r in all_hyp)
    total_grounded = sum(r["n_grounded"] for r in all_hyp)
    total_cross = sum(r["n_cross_venue"] for r in all_hyp)
    cross_domain_groups = sum(1 for r in all_hyp if r["cross_domain"])

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  Communities analyzed: {len(final_groups)}")
    print(f"  Cross-domain communities: {cross_domain_groups}")
    print(f"  Total hypotheses: {total_h}")
    print(f"  Grounded (≥2 citations): {total_grounded} ({100*total_grounded//max(total_h,1)}%)")
    print(f"  Cross-venue transfer: {total_cross}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Cost: ~${total_tokens*6/1_000_000:.2f}")

    # Save
    with open(RESULTS_DIR / "hypotheses.json", "w") as f:
        json.dump(all_hyp, f, indent=2, default=lambda x: list(x) if isinstance(x,set) else str(x))

    report_lines = [
        "="*70,
        "COMMUNITY HYPOTHESIS v3 — LARGE-SCALE CROSS-VENUE",
        "="*70,
        f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges",
        f"Communities: {len(final_groups)} ({cross_domain_groups} cross-domain)",
        f"Hypotheses: {total_h} ({total_grounded} grounded, {total_cross} cross-venue)",
        f"Tokens: {total_tokens:,}, Cost: ~${total_tokens*6/1_000_000:.2f}",
        f"\nVenue coverage: {dict(venue_counts.most_common())}",
    ]
    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
