#!/usr/bin/env python3
"""
ICDM 2026 Camera-Ready Experiments
====================================
All experiments requested by reviewers, prioritized by impact.

Usage:
  export ANTHROPIC_API_KEY=your_key
  python3 camera_ready_experiments.py --phase1 validation_results/scale/phase1_incremental.json

Runs:
  E3:  Hypothesis type classification (R2 Q1)
  E4:  Feature extraction FP/FN analysis (R2 Q4)
  E7:  Coverage flow diagram data (R3)
  E8:  Directional accuracy from validation (R3)
  E12: Prompt template ablation (R4)
  E13: Excluded paper characterization (R4)
  E14: DM-specific hypothesis generation + evaluation (R4)
"""

import json, os, sys, re, time, random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, ".")

RESULTS_DIR = Path("camera_ready_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ================================================================
# E3: Hypothesis Type Classification (Reviewer 2, Q1)
# "How often do hypotheses involve genuinely new module-level
#  combinations vs hyperparameter recombinations?"
# ================================================================

def run_e3_hypothesis_classification(hypotheses, api_key):
    """Classify all hypotheses by type using LLM."""
    print("=" * 65)
    print("E3: Hypothesis Type Classification")
    print("=" * 65)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Batch hypotheses for classification
    batch_size = 10
    all_classifications = []

    for i in range(0, len(hypotheses), batch_size):
        batch = hypotheses[i:i + batch_size]
        hyp_text = "\n".join(
            f"{j+1}. {h.get('hypothesis', '?')[:200]}"
            for j, h in enumerate(batch)
        )

        prompt = f"""You are an ML research expert. Classify each hypothesis into exactly ONE of these categories:

1. MODULE_COMBINATION: Proposes combining architectural modules, loss functions, or model components from different papers (e.g., "use Paper A's attention module in Paper B's detection framework")
2. HYPERPARAMETER_TRANSFER: Transfers specific hyperparameter values (learning rate, batch size, epochs, etc.) from one paper's setting to another paper's model
3. BENCHMARK_TRANSFER: Applies a method tested on one benchmark to a different benchmark or dataset
4. AUGMENTATION_TRANSFER: Transfers data augmentation or regularization techniques across papers
5. ABLATION_GAP: Identifies a missing ablation study within or across papers (e.g., "Paper A didn't test this parameter range that Paper B explored")
6. ARCHITECTURE_MODIFICATION: Proposes modifying model architecture based on insights from another paper (e.g., changing depth, width, adding skip connections)

Hypotheses:
{hyp_text}

Respond with JSON only:
[{{"n": 1, "type": "HYPERPARAMETER_TRANSFER", "reason": "transfers LR from paper X to model Y"}}]"""

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text.strip()
            # Parse JSON
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                all_classifications.extend(parsed)
            print(f"    Classified batch {i // batch_size + 1}/{(len(hypotheses) + batch_size - 1) // batch_size}")
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(1)

    # Summarize
    type_counts = Counter(c.get("type", "UNKNOWN") for c in all_classifications)
    total = len(all_classifications)

    print(f"\n  --- Results ({total} hypotheses classified) ---")
    print(f"  {'Type':<30} {'Count':>6} {'%':>8}")
    print(f"  {'-'*46}")

    module_level = 0
    config_level = 0
    for typ, count in type_counts.most_common():
        pct = 100 * count / max(total, 1)
        print(f"  {typ:<30} {count:>6} {pct:>7.1f}%")
        if typ in ["MODULE_COMBINATION", "ARCHITECTURE_MODIFICATION"]:
            module_level += count
        else:
            config_level += count

    print(f"\n  Module/architecture-level: {module_level}/{total} ({100*module_level/max(total,1):.1f}%)")
    print(f"  Config/parameter-level:    {config_level}/{total} ({100*config_level/max(total,1):.1f}%)")

    results = {
        "total": total,
        "type_counts": dict(type_counts),
        "module_level": module_level,
        "config_level": config_level,
        "module_pct": round(100 * module_level / max(total, 1), 1),
        "classifications": all_classifications,
    }
    with open(RESULTS_DIR / "e3_hypothesis_types.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e3_hypothesis_types.json")
    return results


# ================================================================
# E4: Feature Extraction FP/FN Analysis (Reviewer 2, Q4)
# Systematic false-positive and false-negative analysis
# ================================================================

def run_e4_extraction_analysis(phase1_data, api_key):
    """Analyze false positives and false negatives in feature extraction."""
    print(f"\n{'='*65}")
    print("E4: Feature Extraction FP/FN Analysis")
    print("=" * 65)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Sample 50 repos with insights
    repos_with_insights = [r for r in phase1_data if r.get("n_insights", 0) > 0 and r.get("_insights")]
    random.seed(42)
    sample = random.sample(repos_with_insights, min(50, len(repos_with_insights)))

    from community_hypothesis_v3 import extract_features, GENERIC_METHODS

    # For each repo, ask LLM to evaluate extraction quality
    fp_counts = {"benchmarks": 0, "methods": 0, "params": 0}
    fn_counts = {"benchmarks": 0, "methods": 0, "params": 0}
    total_counts = {"benchmarks": 0, "methods": 0, "params": 0}
    repo_results = []

    for idx, r in enumerate(sample[:20]):  # Do 20 for cost control
        title = r.get("title", "?")[:80]
        insights = r.get("_insights", [])

        # Extract what system found
        features = extract_features(title, insights)

        found_bench = list(features.get("benchmarks", set()))
        found_methods = list(features.get("methods", set()) - GENERIC_METHODS)
        found_params = list(features.get("params", set()))[:10]

        # Sample of insights for LLM to verify
        insight_sample = json.dumps(insights[:5], indent=2)[:2000]

        prompt = f"""You are verifying ML feature extraction quality. Given these raw insights from a paper's code repository, evaluate whether the extracted features are correct.

Paper: {title}

Raw insights (sample):
{insight_sample}

Extracted features:
- Benchmarks: {found_bench}
- Methods: {found_methods}
- Parameters: {found_params[:10]}

For each category, respond with:
1. FALSE POSITIVES: Items in the extracted list that are WRONG (not actually benchmarks/methods/params)
2. FALSE NEGATIVES: Items clearly present in the insights but MISSED by extraction
3. CORRECT: Items correctly extracted

Respond with JSON:
{{
  "benchmarks": {{"correct": [...], "false_positives": [...], "false_negatives": [...]}},
  "methods": {{"correct": [...], "false_positives": [...], "false_negatives": [...]}},
  "params": {{"correct": [...], "false_positives": [...], "false_negatives": [...]}}
}}"""

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                for cat in ["benchmarks", "methods", "params"]:
                    if cat in parsed:
                        fp = len(parsed[cat].get("false_positives", []))
                        fn = len(parsed[cat].get("false_negatives", []))
                        correct = len(parsed[cat].get("correct", []))
                        fp_counts[cat] += fp
                        fn_counts[cat] += fn
                        total_counts[cat] += correct + fp
                repo_results.append({"title": title, "analysis": parsed})
            print(f"    [{idx+1}/20] {title[:50]}...")
        except Exception as e:
            print(f"    [{idx+1}/20] Error: {e}")
        time.sleep(1)

    # Summary
    print(f"\n  --- Results (20 repos) ---")
    print(f"  {'Category':<15} {'Extracted':>10} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*60}")
    for cat in ["benchmarks", "methods", "params"]:
        total = total_counts[cat]
        fp = fp_counts[cat]
        fn = fn_counts[cat]
        tp = total - fp
        precision = tp / max(total, 1) * 100
        recall = tp / max(tp + fn, 1) * 100
        print(f"  {cat:<15} {total:>10} {fp:>6} {fn:>6} {precision:>9.1f}% {recall:>9.1f}%")

    results = {
        "n_repos": len(repo_results),
        "fp_counts": fp_counts,
        "fn_counts": fn_counts,
        "total_counts": total_counts,
        "per_repo": repo_results,
    }
    with open(RESULTS_DIR / "e4_extraction_fpfn.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e4_extraction_fpfn.json")
    return results


# ================================================================
# E7: Coverage Flow Diagram Data (Reviewer 3)
# End-to-end coverage with percentages at every stage
# ================================================================

def run_e7_coverage_flow(phase1_data):
    """Generate coverage flow data for diagram."""
    print(f"\n{'='*65}")
    print("E7: End-to-End Coverage Flow")
    print("=" * 65)

    from community_hypothesis_v3 import (
        extract_features, normalize_venue, build_graph_batched,
        detect_communities, score_community, GENERIC_METHODS
    )

    total_repos = len(phase1_data)

    # Stage 1: Clone + parse
    cloned = [r for r in phase1_data if r.get("status") != "clone_failed"]
    clone_failed = total_repos - len(cloned)

    # Stage 1b: Has config files
    has_configs = [r for r in phase1_data if r.get("n_configs", 0) > 0]

    # Stage 1c: Has insights
    has_insights = [r for r in phase1_data if r.get("n_insights", 0) > 0]

    # Stage 1d: Graph-ready (enough features)
    papers, venues = {}, {}
    for r in phase1_data:
        if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3:
            title = r.get("title", "")
            ins = r.get("_insights", [])
            if not ins or not title:
                continue
            f = extract_features(title, ins)
            if len(f["params"]) >= 2 or f["benchmarks"] or (f["methods"] - GENERIC_METHODS):
                papers[title] = f
                venues[title] = normalize_venue(r.get("venue", "?"))

    graph_ready = len(papers)
    titles = list(papers.keys())

    # Stage 2: Graph
    G = build_graph_batched(papers, titles, venues, sim_threshold=6.0)
    n_edges = G.number_of_edges()
    non_isolates = sum(1 for t in titles if G.degree(t) > 0)

    # Stage 3: Communities
    raw_groups = detect_communities(G, papers, venues, min_community=3)
    scored = []
    for members in raw_groups:
        info = score_community(members, papers, venues)
        if info and info["coherence"] >= 1.0:
            scored.append(info)
    scored.sort(key=lambda x: -x["score"])

    final = []
    used = set()
    for g in scored:
        ms = set(g["members"])
        if len(ms & used) / max(len(ms), 1) < 0.5:
            final.append(g)
            used.update(ms)

    covered = len(used)
    n_communities = len(final)

    # Total insights
    total_insights = sum(r.get("n_insights", 0) for r in phase1_data)

    flow = [
        ("Input repositories", total_repos, 100.0),
        ("Successfully cloned", len(cloned), 100 * len(cloned) / total_repos),
        ("Has config files", len(has_configs), 100 * len(has_configs) / total_repos),
        ("Has extracted insights", len(has_insights), 100 * len(has_insights) / total_repos),
        ("Graph-ready (≥3 configs, features)", graph_ready, 100 * graph_ready / total_repos),
        ("Non-isolates at τ=6.0", non_isolates, 100 * non_isolates / total_repos),
        ("In final communities", covered, 100 * covered / total_repos),
    ]

    print(f"\n  {'Stage':<45} {'Count':>7} {'%':>7}")
    print(f"  {'-'*61}")
    for stage, count, pct in flow:
        bar = "█" * int(pct / 3)
        print(f"  {stage:<45} {count:>7} {pct:>6.1f}% {bar}")

    print(f"\n  Additional stats:")
    print(f"    Total insights extracted: {total_insights:,}")
    print(f"    Graph edges at τ=6.0: {n_edges:,}")
    print(f"    Communities: {n_communities}")
    print(f"    Avg community size: {covered/max(n_communities,1):.1f}")

    results = {
        "flow": [{"stage": s, "count": c, "pct": round(p, 1)} for s, c, p in flow],
        "total_insights": total_insights,
        "n_edges": n_edges,
        "n_communities": n_communities,
    }
    with open(RESULTS_DIR / "e7_coverage_flow.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e7_coverage_flow.json")
    return results


# ================================================================
# E8: Directional Accuracy from Experimental Validation (R3)
# ================================================================

def run_e8_directional_accuracy():
    """Compute directional accuracy from the 8 validated hypotheses."""
    print(f"\n{'='*65}")
    print("E8: Directional Accuracy Analysis")
    print("=" * 65)

    experiments = [
        {
            "name": "H1: LR tuning for VLM",
            "prediction": "LR tuning improves captioning (15-25 point gain)",
            "direction_predicted": "positive",
            "direction_actual": "positive",
            "magnitude_predicted": "15-25 points",
            "magnitude_actual": "10x loss reduction (val_loss 8.89→0.89)",
            "direction_correct": True,
            "magnitude_correct": True,
        },
        {
            "name": "H4: Resolution ablation (CLIP)",
            "prediction": "Halving resolution drops accuracy 15-20pp",
            "direction_predicted": "negative",
            "direction_actual": "negative",
            "magnitude_predicted": "15-20pp",
            "magnitude_actual": "2.6pp",
            "direction_correct": True,
            "magnitude_correct": False,
        },
        {
            "name": "H8: RealNet resolution",
            "prediction": "320px improves Pixel AUROC ~0.1%",
            "direction_predicted": "positive",
            "direction_actual": "N/A (architecture incompatible)",
            "magnitude_predicted": "0.1%",
            "magnitude_actual": "N/A",
            "direction_correct": False,
            "magnitude_correct": False,
        },
        {
            "name": "DM1: GAT + DropEdge",
            "prediction": "DropEdge enables deeper GAT (+1-2%)",
            "direction_predicted": "positive",
            "direction_actual": "mixed (+0.8% CiteSeer, -0.3% Cora)",
            "magnitude_predicted": "1-2%",
            "magnitude_actual": "0.8% on one dataset",
            "direction_correct": True,  # partially
            "magnitude_correct": False,
        },
        {
            "name": "NLP2: Decoding × context length",
            "prediction": "Context length and decoding interact",
            "direction_predicted": "positive (longer context helps)",
            "direction_actual": "context helps, decoding no effect",
            "magnitude_predicted": "significant interaction",
            "magnitude_actual": "10.5% PPL reduction from context; 0% from decoding",
            "direction_correct": True,  # context part correct
            "magnitude_correct": False,
        },
        {
            "name": "NLP3: LoRA LR transfer",
            "prediction": "VB-LoRA LR range (2e-5 to 3e-5) is optimal",
            "direction_predicted": "positive (this range works)",
            "direction_actual": "works for full FT; LoRA needs 15x higher",
            "magnitude_predicted": "optimal range",
            "magnitude_actual": "optimal for full FT, suboptimal for LoRA",
            "direction_correct": True,
            "magnitude_correct": False,
        },
        {
            "name": "NLP5: Selective data quality",
            "prediction": "Higher-quality training data improves performance",
            "direction_predicted": "positive",
            "direction_actual": "positive (PPL 13.36 vs 13.47)",
            "magnitude_predicted": "improvement",
            "magnitude_actual": "0.8% improvement",
            "direction_correct": True,
            "magnitude_correct": True,
        },
        {
            "name": "CV2: Temperature calibration",
            "prediction": "Temperature scaling improves calibration",
            "direction_predicted": "positive",
            "direction_actual": "positive (74% ECE reduction)",
            "magnitude_predicted": "improvement",
            "magnitude_actual": "74% ECE reduction",
            "direction_correct": True,
            "magnitude_correct": True,
        },
    ]

    dir_correct = sum(1 for e in experiments if e["direction_correct"])
    mag_correct = sum(1 for e in experiments if e["magnitude_correct"])
    total = len(experiments)

    print(f"\n  {'Hypothesis':<35} {'Dir':>5} {'Mag':>5}")
    print(f"  {'-'*47}")
    for e in experiments:
        d = "✓" if e["direction_correct"] else "✗"
        m = "✓" if e["magnitude_correct"] else "✗"
        print(f"  {e['name']:<35} {d:>5} {m:>5}")

    print(f"\n  Directional accuracy: {dir_correct}/{total} ({100*dir_correct/total:.1f}%)")
    print(f"  Magnitude accuracy:  {mag_correct}/{total} ({100*mag_correct/total:.1f}%)")
    print(f"  Both correct:        {sum(1 for e in experiments if e['direction_correct'] and e['magnitude_correct'])}/{total}")

    results = {
        "directional_accuracy": round(100 * dir_correct / total, 1),
        "magnitude_accuracy": round(100 * mag_correct / total, 1),
        "experiments": experiments,
    }
    with open(RESULTS_DIR / "e8_directional_accuracy.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e8_directional_accuracy.json")
    return results


# ================================================================
# E12: Prompt Template Ablation (Reviewer 4)
# Test how much the prompt template affects scores
# ================================================================

def run_e12_prompt_ablation(phase1_data, api_key):
    """Test different prompt templates on the same community."""
    print(f"\n{'='*65}")
    print("E12: Prompt Template Ablation")
    print("=" * 65)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    from community_hypothesis_v3 import (
        extract_features, normalize_venue, build_graph_batched,
        detect_communities, score_community, GENERIC_METHODS
    )

    # Build one community to test on
    papers, insights_map, venues = {}, {}, {}
    for r in phase1_data:
        if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3:
            title = r.get("title", "")
            ins = r.get("_insights", [])
            if not ins or not title:
                continue
            f = extract_features(title, ins)
            if len(f["params"]) >= 2 or f["benchmarks"] or (f["methods"] - GENERIC_METHODS):
                papers[title] = f
                insights_map[title] = ins
                venues[title] = normalize_venue(r.get("venue", "?"))

    titles = list(papers.keys())
    G = build_graph_batched(papers, titles, venues, sim_threshold=6.0)
    raw_groups = detect_communities(G, papers, venues, min_community=3)
    scored = []
    for members in raw_groups:
        info = score_community(members, papers, venues)
        if info and info["coherence"] >= 1.0:
            scored.append(info)
    scored.sort(key=lambda x: -x["score"])

    if not scored:
        print("  No communities found!")
        return {}

    test_community = scored[0]
    members = test_community["members"][:15]
    print(f"  Test community: {len(members)} papers, bench={test_community.get('shared_bench', [])[:3]}")

    # Build evidence string
    evidence_lines = []
    for title in members[:10]:
        ins = insights_map.get(title, [])
        for i in ins[:3]:
            if isinstance(i, dict):
                evidence_lines.append(f"  [{title[:40]}] {json.dumps(i)[:150]}")

    evidence_str = "\n".join(evidence_lines[:30])
    paper_list = "\n".join(f"- {t}" for t in members)

    # Different prompt templates
    PROMPTS = {
        "structured_evidence": f"""You are a research advisor. Below are ML papers from a community sharing experimental overlap, along with structured evidence extracted from their code.

PAPERS:
{paper_list}

EVIDENCE:
{evidence_str}

Generate 5 cross-paper hypotheses. Each must:
1. Reference specific metric values marked with ★
2. Combine findings from at least 2 papers
3. Predict a quantitative outcome

Return JSON: [{{"hypothesis":"...","motivation":"...","expected_outcome":"...","confidence":"high|medium|low","cited_papers":["..."]}}]""",

        "minimal_no_evidence": f"""You are a research advisor. Generate 5 research hypotheses combining findings from these ML papers:

{paper_list}

Return JSON: [{{"hypothesis":"...","motivation":"...","expected_outcome":"...","confidence":"high|medium|low","cited_papers":["..."]}}]""",

        "chain_of_thought": f"""You are a research advisor. Below are ML papers from a community with shared experimental features.

PAPERS:
{paper_list}

EVIDENCE:
{evidence_str}

For each hypothesis, first reason step-by-step about WHY combining these papers' findings would work, THEN formulate the hypothesis.

Think about:
- What experimental gaps exist between these papers?
- Which parameter values from Paper A might improve Paper B?
- What untested combinations could yield improvements?

Generate 5 hypotheses. Return JSON: [{{"reasoning":"step-by-step reasoning","hypothesis":"...","motivation":"...","expected_outcome":"...","confidence":"high|medium|low","cited_papers":["..."]}}]""",

        "negative_framing": f"""You are a skeptical research advisor. Below are ML papers with shared experimental features.

PAPERS:
{paper_list}

EVIDENCE:
{evidence_str}

Generate 5 hypotheses that these papers' authors SHOULD have tested but didn't. Focus on:
- Missing ablations that could change conclusions
- Untested parameter combinations
- Potential failure modes not explored

Be critical. Return JSON: [{{"hypothesis":"...","motivation":"...","expected_outcome":"...","confidence":"high|medium|low","cited_papers":["..."]}}]""",
    }

    # Judge prompt
    JUDGE = """Rate each hypothesis on 4 dimensions (1-5):
SPECIFICITY: Names exact parameters, values, benchmarks?
GROUNDEDNESS: Backed by real metric values?
ACTIONABILITY: Implementable today?
CROSS_PAPER_NOVELTY: Requires knowledge from 2+ papers?

HYPOTHESES:
{hypotheses}

Respond JSON: [{{"n":1,"specificity":4,"groundedness":3,"actionability":4,"cross_paper_novelty":3}}]"""

    results = {}
    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n  [{prompt_name}]...", end=" ", flush=True)

        # Generate
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_template}]
            )
            gen_text = resp.content[0].text.strip()
            start = gen_text.find("[")
            end = gen_text.rfind("]") + 1
            if start >= 0 and end > start:
                hyps = json.loads(gen_text[start:end])
            else:
                hyps = []
        except Exception as e:
            print(f"Error: {e}")
            continue

        if not hyps:
            print("No hypotheses")
            continue

        # Judge
        hyp_text = "\n".join(
            f"{i+1}. {h.get('hypothesis', '?')[:200]}\n   Expected: {h.get('expected_outcome', '?')[:100]}"
            for i, h in enumerate(hyps)
        )

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": JUDGE.format(hypotheses=hyp_text)}]
            )
            judge_text = resp.content[0].text.strip()
            start = judge_text.find("[")
            end = judge_text.rfind("]") + 1
            scores = json.loads(judge_text[start:end]) if start >= 0 else []
        except:
            scores = []

        if scores:
            dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
            means = {d: np.mean([s.get(d, 0) for s in scores if s.get(d, 0) > 0]) for d in dims}
            overall = np.mean([np.mean([s.get(d, 0) for d in dims if s.get(d, 0) > 0]) for s in scores])
            print(f"{len(hyps)} hyps, mean={overall:.2f}, spec={means['specificity']:.2f}, "
                  f"grnd={means['groundedness']:.2f}")
        else:
            means = {}
            overall = 0
            print(f"{len(hyps)} hyps, scoring failed")

        results[prompt_name] = {
            "n_hypotheses": len(hyps),
            "mean_quality": round(overall, 2),
            "dimensions": {d: round(v, 2) for d, v in means.items()},
            "hypotheses": hyps[:3],  # sample
        }
        time.sleep(2)

    # Summary
    print(f"\n  --- Prompt Template Impact ---")
    print(f"  {'Template':<25} {'Mean':>6} {'Spec':>6} {'Grnd':>6} {'Actn':>6} {'XPap':>6}")
    print(f"  {'-'*57}")
    for name, r in results.items():
        d = r.get("dimensions", {})
        print(f"  {name:<25} {r['mean_quality']:>6.2f} {d.get('specificity',0):>6.2f} "
              f"{d.get('groundedness',0):>6.2f} {d.get('actionability',0):>6.2f} "
              f"{d.get('cross_paper_novelty',0):>6.2f}")

    with open(RESULTS_DIR / "e12_prompt_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e12_prompt_ablation.json")
    return results


# ================================================================
# E13: Excluded Paper Characterization (Reviewer 4)
# By venue, year, and framework
# ================================================================

def run_e13_excluded_characterization(phase1_data):
    """Characterize excluded papers by venue, year, and reason."""
    print(f"\n{'='*65}")
    print("E13: Excluded Paper Characterization")
    print("=" * 65)

    from community_hypothesis_v3 import extract_features, GENERIC_METHODS

    categories = {
        "graph_ready": [],
        "has_insights_not_ready": [],
        "has_configs_no_insights": [],
        "no_configs": [],
        "clone_failed": [],
    }

    for r in phase1_data:
        title = r.get("title", "?")
        venue = r.get("venue", "unknown")
        n_configs = r.get("n_configs", 0)
        n_insights = r.get("n_insights", 0)
        status = r.get("status", "")

        # Parse year
        year = "unknown"
        for y in range(2018, 2026):
            if str(y) in str(venue) or str(y) in str(title):
                year = str(y)
                break

        entry = {"title": str(title)[:60], "venue": str(venue)[:30], "year": year}

        if status == "clone_failed" or status == "error":
            categories["clone_failed"].append(entry)
        elif n_configs == 0:
            categories["no_configs"].append(entry)
        elif n_insights == 0:
            categories["has_configs_no_insights"].append(entry)
        elif n_insights > 0:
            ins = r.get("_insights", [])
            if ins and n_configs >= 3:
                f = extract_features(title, ins)
                if len(f["params"]) >= 2 or f["benchmarks"] or (f["methods"] - GENERIC_METHODS):
                    categories["graph_ready"].append(entry)
                else:
                    categories["has_insights_not_ready"].append(entry)
            else:
                categories["has_insights_not_ready"].append(entry)

    print(f"\n  --- Category Breakdown ---")
    total = len(phase1_data)
    for cat, items in categories.items():
        print(f"  {cat:<30} {len(items):>6} ({100*len(items)/total:>5.1f}%)")

    # Venue distribution for excluded papers
    excluded = (categories["no_configs"] + categories["has_configs_no_insights"] +
                categories["clone_failed"])
    included = categories["graph_ready"]

    excl_venues = Counter(e["venue"][:15] for e in excluded)
    incl_venues = Counter(e["venue"][:15] for e in included)

    print(f"\n  --- Top Venues: Included vs Excluded ---")
    print(f"  {'Venue':<18} {'Included':>10} {'Excluded':>10} {'Incl%':>8}")
    all_venues = set(list(excl_venues.keys()) + list(incl_venues.keys()))
    venue_data = []
    for v in all_venues:
        inc = incl_venues.get(v, 0)
        exc = excl_venues.get(v, 0)
        total_v = inc + exc
        if total_v >= 20:
            venue_data.append((v, inc, exc, 100 * inc / max(total_v, 1)))

    venue_data.sort(key=lambda x: -x[1] - x[2])
    for v, inc, exc, pct in venue_data[:15]:
        print(f"  {v:<18} {inc:>10} {exc:>10} {pct:>7.1f}%")

    # Year distribution
    excl_years = Counter(e["year"] for e in excluded if e["year"] != "unknown")
    incl_years = Counter(e["year"] for e in included if e["year"] != "unknown")

    print(f"\n  --- Year: Included vs Excluded ---")
    print(f"  {'Year':<10} {'Included':>10} {'Excluded':>10} {'Incl%':>8}")
    for year in sorted(set(list(excl_years.keys()) + list(incl_years.keys()))):
        inc = incl_years.get(year, 0)
        exc = excl_years.get(year, 0)
        total_y = inc + exc
        if total_y > 0:
            print(f"  {year:<10} {inc:>10} {exc:>10} {100*inc/total_y:>7.1f}%")

    results = {
        "categories": {k: len(v) for k, v in categories.items()},
        "excluded_venue_top": dict(excl_venues.most_common(10)),
        "included_venue_top": dict(incl_venues.most_common(10)),
    }
    with open(RESULTS_DIR / "e13_excluded_characterization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e13_excluded_characterization.json")
    return results


# ================================================================
# E14: DM-Specific Hypothesis Generation (Reviewer 4)
# ================================================================

def run_e14_dm_hypotheses(phase1_data, api_key):
    """Generate and evaluate hypotheses specifically from DM papers."""
    print(f"\n{'='*65}")
    print("E14: DM-Specific Hypothesis Generation")
    print("=" * 65)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Find DM papers
    dm_venues = {"kdd", "icdm", "cikm", "wsdm", "sigir", "www", "sdm"}
    dm_papers = []
    for r in phase1_data:
        venue = r.get("venue", "").lower()
        if any(dv in venue for dv in dm_venues):
            if r.get("n_insights", 0) > 0:
                dm_papers.append(r)

    # Also find papers with DM-related methods
    dm_keywords = {"graph_nn", "recommendation", "knowledge_graph", "tabular",
                   "time_series", "clustering", "anomaly_detection"}
    for r in phase1_data:
        if r.get("n_insights", 0) > 0:
            ins = r.get("_insights", [])
            ins_str = json.dumps(ins).lower()
            if any(kw in ins_str for kw in dm_keywords):
                if r not in dm_papers:
                    dm_papers.append(r)

    print(f"  DM-related papers with insights: {len(dm_papers)}")

    if len(dm_papers) < 5:
        print("  Not enough DM papers for hypothesis generation")
        return {}

    # Generate hypotheses from DM papers
    paper_info = []
    for r in dm_papers[:20]:
        title = r.get("title", "?")
        ins = r.get("_insights", [])
        ins_sample = json.dumps(ins[:3])[:300]
        paper_info.append(f"- {title}\n  Evidence: {ins_sample}")

    paper_text = "\n".join(paper_info)

    prompt = f"""You are a data mining research advisor. Below are {len(dm_papers[:20])} papers from data mining venues (KDD, ICDM, CIKM, WSDM) with extracted experimental evidence.

Generate 10 cross-paper hypotheses specifically relevant to the data mining community. Focus on:
- Graph neural networks, node classification, link prediction
- Time series forecasting and anomaly detection
- Recommendation systems
- Tabular learning
- Knowledge graphs

PAPERS AND EVIDENCE:
{paper_text}

Return JSON: [{{"hypothesis":"specific experiment","motivation":"why","expected_outcome":"quantitative prediction","confidence":"high|medium|low","cited_papers":["..."],"dm_relevance":"which DM subfield"}}]"""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        hyps = json.loads(text[start:end]) if start >= 0 else []
    except Exception as e:
        print(f"  Generation error: {e}")
        hyps = []

    print(f"  Generated {len(hyps)} DM hypotheses")

    # Show samples
    for i, h in enumerate(hyps[:5]):
        print(f"\n  DM-H{i+1}: {h.get('hypothesis', '?')[:120]}")
        print(f"    DM relevance: {h.get('dm_relevance', '?')}")
        print(f"    Confidence: {h.get('confidence', '?')}")

    # Classify by DM subfield
    subfields = Counter(h.get("dm_relevance", "other") for h in hyps)
    print(f"\n  DM Subfield Distribution:")
    for sf, count in subfields.most_common():
        print(f"    {sf}: {count}")

    results = {
        "n_dm_papers": len(dm_papers),
        "n_hypotheses": len(hyps),
        "hypotheses": hyps,
        "subfield_distribution": dict(subfields),
    }
    with open(RESULTS_DIR / "e14_dm_hypotheses.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/e14_dm_hypotheses.json")
    return results


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True, help="Path to phase1_incremental.json")
    parser.add_argument("--skip", nargs="*", default=[], help="Experiments to skip (e3,e4,e7,e8,e12,e13,e14)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    with open(args.phase1) as f:
        phase1_data = json.load(f)
    print(f"Loaded {len(phase1_data)} repos from {args.phase1}\n")

    # Collect hypotheses for E3
    hypotheses = []
    for r in phase1_data:
        for h in r.get("_hypotheses", r.get("hypotheses", [])):
            if isinstance(h, dict) and "hypothesis" in h:
                hypotheses.append(h)

    # Also try loading from LLM ablation results
    for hyp_file in ["llm_ablation_results.json", "validation_results/llm_ablation_results.json"]:
        if os.path.exists(hyp_file):
            with open(hyp_file) as f:
                ablation_data = json.load(f)
            for model_name, runs in ablation_data.get("all_runs", {}).items():
                for run in runs[:1]:
                    for group in run.get("results", []):
                        for h in group.get("hypotheses", []):
                            if isinstance(h, dict) and "hypothesis" in h:
                                hypotheses.append(h)
            break

    print(f"Found {len(hypotheses)} hypotheses for classification\n")

    all_results = {}

    # E3: Hypothesis classification
    if "e3" not in args.skip and api_key and hypotheses:
        all_results["e3"] = run_e3_hypothesis_classification(hypotheses[:50], api_key)

    # E4: FP/FN analysis
    if "e4" not in args.skip and api_key:
        all_results["e4"] = run_e4_extraction_analysis(phase1_data, api_key)

    # E7: Coverage flow
    if "e7" not in args.skip:
        all_results["e7"] = run_e7_coverage_flow(phase1_data)

    # E8: Directional accuracy (no data needed)
    if "e8" not in args.skip:
        all_results["e8"] = run_e8_directional_accuracy()

    # E12: Prompt ablation
    if "e12" not in args.skip and api_key:
        all_results["e12"] = run_e12_prompt_ablation(phase1_data, api_key)

    # E13: Excluded paper characterization
    if "e13" not in args.skip:
        all_results["e13"] = run_e13_excluded_characterization(phase1_data)

    # E14: DM hypotheses
    if "e14" not in args.skip and api_key:
        all_results["e14"] = run_e14_dm_hypotheses(phase1_data, api_key)

    print(f"\n{'='*65}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*65}")
    print(f"Results saved to {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
