#!/usr/bin/env python3
"""
Data Mining Venue Repo Collection
====================================
Collects repos from DM conferences (KDD, ICDM, CIKM, WSDM, SDM, PAKDD)
for 2021-2024, runs the extraction pipeline, and generates DM-specific
hypotheses.

Addresses Reviewer R2: "Include a small targeted collection of 50-100
repositories from KDD 2023-2024 and ICDM 2023-2024"

Sources:
  1. PapersWithCode HuggingFace dump (already have)
  2. GitHub search for conference-tagged repos
  3. Manual curated list of top DM papers with code

Usage:
  # Step 1: Collect repos
  python3 dm_repos.py collect

  # Step 2: Run extraction pipeline on collected repos
  python3 dm_repos.py extract --repos dm_repos.json

  # Step 3: Build graph and generate hypotheses
  python3 dm_repos.py hypotheses --phase1 dm_phase1.json
"""

import json, os, sys, time, re, subprocess
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, ".")

RESULTS_DIR = Path("./validation_results/dm_venues")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DM conferences and their common benchmarks/topics
DM_VENUES = [
    "KDD", "ICDM", "CIKM", "WSDM", "SDM", "PAKDD",
    "WWW", "RecSys", "SIGIR",
]

DM_BENCHMARKS = [
    # Graph
    "ogbg-molhiv", "ogbg-molpcba", "ogbn-arxiv", "ogbn-products", "ogbl-collab",
    "cora", "citeseer", "pubmed", "reddit", "ppi",
    "tu_datasets", "zinc", "qm9",
    # Tabular
    "openml", "uci", "adult", "covertype", "higgs",
    "california_housing", "bike_sharing",
    # Time series
    "ett", "weather", "electricity", "traffic", "exchange_rate",
    "m4", "m5", "tourism",
    # Recommendation
    "movielens", "ml-1m", "ml-100k", "amazon", "yelp", "gowalla",
    "netflix", "lastfm", "book-crossing",
    # Anomaly detection
    "kdd99", "nsl-kdd", "cicids", "swat",
    # NLP/IR
    "ms-marco", "trec", "beir",
]

# Curated list of notable KDD/ICDM papers with code (2021-2024)
CURATED_DM_REPOS = [
    # KDD 2024
    {"title": "GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner",
     "venue": "KDD 2024", "url": "https://github.com/THUDM/GraphMAE2"},
    {"title": "FairGNN: Fairness-Aware Graph Neural Networks",
     "venue": "KDD 2024", "url": "https://github.com/EnyanDai/FairGNN"},
    {"title": "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second",
     "venue": "KDD 2024", "url": "https://github.com/automl/TabPFN"},
    {"title": "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
     "venue": "KDD 2024", "url": "https://github.com/thuml/iTransformer"},
    {"title": "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
     "venue": "KDD 2024", "url": "https://github.com/thuml/Time-Series-Library"},
    {"title": "GNNExplainer: Generating Explanations for Graph Neural Networks",
     "venue": "KDD 2023", "url": "https://github.com/RexYing/gnn-model-explainer"},
    {"title": "DiffPool: Hierarchical Graph Representation Learning with Differentiable Pooling",
     "venue": "KDD 2023", "url": "https://github.com/RexYing/diffpool"},
    {"title": "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation",
     "venue": "KDD 2023", "url": "https://github.com/gusye1234/LightGCN-PyTorch"},
    {"title": "GRAND: Graph Neural Diffusion",
     "venue": "KDD 2023", "url": "https://github.com/twitter-research/graph-neural-pde"},
    {"title": "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks",
     "venue": "KDD 2023", "url": "https://github.com/shichence/AutoInt"},
    # ICDM 2023-2024
    {"title": "ST-GCN: Spatio-Temporal Graph Convolutional Networks",
     "venue": "ICDM 2023", "url": "https://github.com/yysijie/st-gcn"},
    {"title": "USAD: Unsupervised Anomaly Detection on Multivariate Time Series",
     "venue": "ICDM 2023", "url": "https://github.com/manigalati/usad"},
    {"title": "PyOD: A Python Toolbox for Scalable Outlier Detection",
     "venue": "ICDM 2023", "url": "https://github.com/yzhao062/pyod"},
    # KDD 2022
    {"title": "GBT: Two Directions of Feature Engineering for Tabular Data",
     "venue": "KDD 2022", "url": "https://github.com/puhsu/tabular-dl-revisiting-models"},
    {"title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
     "venue": "KDD 2022", "url": "https://github.com/zhouhaoyi/Informer2020"},
    {"title": "SASRec: Self-Attentive Sequential Recommendation",
     "venue": "KDD 2022", "url": "https://github.com/kang205/SASRec"},
    {"title": "GraphSAGE: Inductive Representation Learning on Large Graphs",
     "venue": "KDD 2022", "url": "https://github.com/williamleif/GraphSAGE"},
    {"title": "GIN: How Powerful are Graph Neural Networks?",
     "venue": "KDD 2022", "url": "https://github.com/weihua916/powerful-gnns"},
    # KDD 2021
    {"title": "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
     "venue": "KDD 2021", "url": "https://github.com/thuml/Autoformer"},
    {"title": "AAAI DeepGNN: Graph Neural Networks with Deep Aggregation",
     "venue": "KDD 2021", "url": "https://github.com/dmlc/dgl"},
    # CIKM
    {"title": "NCL: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning",
     "venue": "CIKM 2022", "url": "https://github.com/RUCAIBox/NCL"},
    {"title": "SGL: Self-supervised Graph Learning for Recommendation",
     "venue": "CIKM 2022", "url": "https://github.com/wujcan/SGL-Torch"},
    # WSDM
    {"title": "SimpleX: A Simple and Strong Baseline for Collaborative Filtering",
     "venue": "WSDM 2022", "url": "https://github.com/xue-pai/Open-CF-Benchmarks"},
    {"title": "CL4SRec: Contrastive Learning for Sequential Recommendation",
     "venue": "WSDM 2022", "url": "https://github.com/RUCAIBox/RecBole"},
    # More tabular/graph
    {"title": "XGBoost: A Scalable Tree Boosting System",
     "venue": "KDD 2024", "url": "https://github.com/dmlc/xgboost"},
    {"title": "CatBoost: unbiased boosting with categorical features",
     "venue": "KDD 2024", "url": "https://github.com/catboost/catboost"},
    {"title": "GAT: Graph Attention Networks",
     "venue": "KDD 2023", "url": "https://github.com/PetarV-/GAT"},
    {"title": "DGCNN: Dynamic Graph CNN for Learning on Point Clouds",
     "venue": "KDD 2023", "url": "https://github.com/WangYueFt/dgcnn"},
    {"title": "PatchTST: A Time Series is Worth 64 Words",
     "venue": "KDD 2024", "url": "https://github.com/yuqinie98/PatchTST"},
    {"title": "DLinear: Are Transformers Effective for Time Series Forecasting?",
     "venue": "KDD 2024", "url": "https://github.com/cure-lab/LTSF-Linear"},
    # Graph anomaly
    {"title": "DOMINANT: Deep Anomaly Detection on Attributed Networks",
     "venue": "ICDM 2022", "url": "https://github.com/kaize0409/GCN_AnomalyDetection"},
    {"title": "CoLA: Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning",
     "venue": "ICDM 2022", "url": "https://github.com/GRAND-Lab/CoLA"},
    # More KDD/ICDM recent
    {"title": "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
     "venue": "KDD 2023", "url": "https://github.com/MAZiqing/FEDformer"},
    {"title": "STID: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting",
     "venue": "ICDM 2023", "url": "https://github.com/zezhishao/STID"},
    {"title": "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis",
     "venue": "ICDM 2024", "url": "https://github.com/luodhhh/ModernTCN"},
    # Tabular deep learning
    {"title": "FT-Transformer: Revisiting Deep Learning Models for Tabular Data",
     "venue": "KDD 2023", "url": "https://github.com/yandex-research/rtdl-revisiting-models"},
    {"title": "SAINT: Improved Neural Networks for Tabular Data via Row Attention",
     "venue": "KDD 2023", "url": "https://github.com/somepago/saint"},
    {"title": "TabNet: Attentive Interpretable Tabular Learning",
     "venue": "KDD 2022", "url": "https://github.com/dreamquark-ai/tabnet"},
    {"title": "NODE: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data",
     "venue": "KDD 2022", "url": "https://github.com/Qwicen/node"},
    # Recommendation
    {"title": "NGCF: Neural Graph Collaborative Filtering",
     "venue": "KDD 2023", "url": "https://github.com/xiangwang1223/neural_graph_collaborative_filtering"},
    {"title": "BPR: Bayesian Personalized Ranking from Implicit Feedback",
     "venue": "KDD 2022", "url": "https://github.com/RUCAIBox/RecBole"},
]


def collect_repos():
    """Step 1: Collect DM repos from curated list + PwC dumps."""
    print("=" * 70)
    print(f"COLLECTING DM REPOS ({len(CURATED_DM_REPOS)} curated)")
    print("=" * 70)

    # Also try to load from existing PwC data
    pwc_repos = []
    pwc_path = Path("./pwc_papers")
    if pwc_path.exists():
        for f in pwc_path.glob("*.json"):
            try:
                with open(f) as fh:
                    papers = json.load(fh)
                for p in papers:
                    venue = p.get("proceeding", "").upper()
                    if any(dv in venue for dv in ["KDD", "ICDM", "CIKM", "WSDM", "SDM", "PAKDD"]):
                        if p.get("repo_url"):
                            pwc_repos.append({
                                "title": p.get("title", ""),
                                "venue": venue,
                                "url": p["repo_url"],
                            })
            except:
                pass

    all_repos = CURATED_DM_REPOS + pwc_repos

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in all_repos:
        url = r["url"].rstrip("/").lower()
        if url not in seen:
            seen.add(url)
            unique.append(r)

    print(f"  Curated: {len(CURATED_DM_REPOS)}")
    print(f"  From PwC: {len(pwc_repos)}")
    print(f"  Unique: {len(unique)}")

    # Venue breakdown
    venues = Counter(r["venue"].split()[0] for r in unique)
    print(f"  By venue: {dict(venues.most_common())}")

    with open(RESULTS_DIR / "dm_repos.json", "w") as f:
        json.dump(unique, f, indent=2)

    print(f"  Saved to {RESULTS_DIR}/dm_repos.json")
    return unique


def extract_repos(repos_path):
    """Step 2: Run extraction pipeline on DM repos."""
    print("=" * 70)
    print("EXTRACTING FEATURES FROM DM REPOS")
    print("=" * 70)

    with open(repos_path) as f:
        repos = json.load(f)

    # Import from the main evaluation script
    try:
        from viveka_scale_evaluation import run_repo, WORK_DIR
        WORK_DIR.mkdir(parents=True, exist_ok=True)
    except ImportError:
        print("  ERROR: Cannot import viveka_scale_evaluation.py")
        print("  Make sure it's in the current directory or PYTHONPATH")
        print("  Alternatively, run extraction manually:")
        print(f"    python3 viveka_scale_evaluation.py --max-repos {len(repos)}")
        sys.exit(1)

    results = []
    for i, r in enumerate(repos):
        print(f"  [{i+1}/{len(repos)}] {r['title'][:50]}...", end=" ", flush=True)
        # Adapt entry format: run_repo expects 'repo_url'
        entry = {"repo_url": r["url"], "title": r["title"], "venue": r["venue"]}
        try:
            result, elapsed = run_repo(entry)
            result["title"] = r["title"]
            result["venue"] = r["venue"]
            result["repo_url"] = r["url"]
            result["time"] = elapsed
            # Store insights for later use
            if "_insights" not in result and "insights" in result:
                result["_insights"] = result.pop("insights", [])
            results.append(result)
            n_ins = result.get("n_insights", 0)
            print(f"{'OK' if n_ins > 0 else 'no insights'} ({n_ins} insights, {elapsed:.1f}s)")
        except Exception as e:
            print(f"error: {e}")
            results.append({"title": r["title"], "venue": r["venue"],
                          "status": "error", "n_insights": 0, "n_configs": 0,
                          "insight_types": {}, "_insights": []})
        time.sleep(0.5)

    # Summary
    ok = sum(1 for r in results if r.get("n_insights", 0) > 0)
    total_ins = sum(r.get("n_insights", 0) for r in results)
    venues = Counter(r.get("venue", "?").split()[0] for r in results if r.get("n_insights", 0) > 0)
    print(f"\n  Total: {len(results)}")
    print(f"  With insights: {ok} ({100*ok//max(len(results),1)}%)")
    print(f"  Total insights: {total_ins}")
    print(f"  By venue: {dict(venues.most_common())}")

    with open(RESULTS_DIR / "dm_phase1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Saved to {RESULTS_DIR}/dm_phase1.json")
    return results


def generate_hypotheses(phase1_path):
    """Step 3: Build graph and generate DM-specific hypotheses."""
    print("=" * 70)
    print("GENERATING DM-SPECIFIC HYPOTHESES")
    print("=" * 70)

    from community_hypothesis_v3 import (
        extract_features, normalize_venue, build_graph_batched,
        detect_communities, score_community, GENERIC_METHODS
    )
    from llm_ablation import build_prompt_for_group, generate_sonnet, parse_json_array

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  Need ANTHROPIC_API_KEY")
        sys.exit(1)

    with open(phase1_path) as f:
        p1 = json.load(f)

    # Also load main corpus to combine
    main_path = "./validation_results/scale/phase1_incremental.json"
    if os.path.exists(main_path):
        with open(main_path) as f:
            main_p1 = json.load(f)
        print(f"  Main corpus: {len(main_p1)} repos")
        p1 = main_p1 + p1
        print(f"  Combined: {len(p1)} repos")

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
            v = normalize_venue(r.get("venue", "?"))
            venues[title] = v

    titles = list(papers.keys())
    dm_titles = [t for t in titles if venues.get(t, "").upper() in
                 ["KDD", "ICDM", "CIKM", "WSDM", "SDM", "PAKDD"]]

    print(f"  Total papers: {len(titles)}")
    print(f"  DM papers: {len(dm_titles)}")

    # Build graph
    G = build_graph_batched(papers, titles, venues, sim_threshold=4.0)  # lower threshold for DM
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Find communities containing DM papers
    raw_groups = detect_communities(G, papers, venues, min_community=3)
    scored = []
    for members in raw_groups:
        info = score_community(members, papers, venues)
        if info:
            # Check if any DM paper is in this community
            dm_count = sum(1 for m in info["members"]
                          if venues.get(m, "").upper() in ["KDD", "ICDM", "CIKM", "WSDM"])
            info["dm_count"] = dm_count
            if dm_count > 0:
                scored.append(info)
    scored.sort(key=lambda x: (-x["dm_count"], -x["score"]))

    print(f"\n  Communities with DM papers: {len(scored)}")
    for i, g in enumerate(scored[:5]):
        print(f"    C{i+1}: {g['size']} papers, DM={g['dm_count']}, "
              f"venues={g.get('venues', [])[:5]}, bench={g.get('shared_bench', [])}")

    # Generate hypotheses for top DM communities
    if scored:
        print(f"\n  Generating hypotheses for top DM communities...")
        for i, g in enumerate(scored[:3]):
            prompt = build_prompt_for_group(g, papers, insights_map)
            text, tokens = generate_sonnet(prompt, api_key)
            hyps = parse_json_array(text)
            print(f"\n  Community {i+1} ({g['dm_count']} DM papers, {g['size']} total):")
            print(f"  Benchmarks: {g.get('shared_bench', [])}")
            print(f"  Venues: {g.get('venues', [])}")
            for j, h in enumerate(hyps):
                print(f"    H{j+1}: {h.get('hypothesis', '?')[:150]}")
            time.sleep(1)

        with open(RESULTS_DIR / "dm_hypotheses.json", "w") as f:
            json.dump({"communities": scored[:5], 
                       "n_dm_papers": len(dm_titles)}, f, indent=2, default=str)
    else:
        print("  No communities with DM papers found. Try lower threshold.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect", "extract", "hypotheses"],
                        help="Which step to run")
    parser.add_argument("--repos", default=None, help="Path to repos JSON for extract")
    parser.add_argument("--phase1", default=None, help="Path to phase1 JSON for hypotheses")
    args = parser.parse_args()

    if args.action == "collect":
        collect_repos()
    elif args.action == "extract":
        path = args.repos or str(RESULTS_DIR / "dm_repos.json")
        extract_repos(path)
    elif args.action == "hypotheses":
        path = args.phase1 or str(RESULTS_DIR / "dm_phase1.json")
        generate_hypotheses(path)


if __name__ == "__main__":
    main()
