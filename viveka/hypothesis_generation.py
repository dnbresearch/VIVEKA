#!/usr/bin/env python3
"""
Improved Hypothesis Generation: Metric-Grounded
==================================================
Fixes the groundedness weakness by:
1. Formatting insights with actual numbers prominently
2. Separating "evidence-rich" insights from "structure-only" ones
3. Instructing LLM to cite specific metric values in hypotheses

Then re-judges to measure groundedness improvement.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...
  export GEMINI_API_KEY=...
  python3 hypothesis_grounded_v3.py \
      --baseline-results baseline_results.json \
      --groups groups.json
"""

import json, os, re, sys, time, random
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

RESULTS_DIR = Path("./validation_results/hypothesis_grounded_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Format insights WITH actual numbers (the key fix)
# =====================================================================

def format_insight_rich(ins, repo_title):
    """Format a single insight with all available metric data."""
    itype = ins["type"]
    desc = ins["description"]
    ev = ins.get("evidence", {})
    
    lines = []
    
    if itype == "config_result_link":
        # Richest type — has param change + metric delta
        metric = ev.get("metric", "?")
        vfrom = ev.get("value_from")
        vto = ev.get("value_to")
        delta = ev.get("delta")
        changes = ev.get("config_changes", {})
        
        param_str = ", ".join(f"{k}: {v.get('from','?')}→{v.get('to','?')}" 
                              for k, v in changes.items())
        lines.append(f"  ★ CONFIG→RESULT: {param_str}")
        if vfrom is not None and vto is not None:
            lines.append(f"    {metric}: {vfrom} → {vto} (Δ={delta})")
    
    elif itype == "result_comparison":
        metric = ev.get("metric", "?")
        best = ev.get("best", {})
        worst = ev.get("worst", {})
        all_res = ev.get("all_results", [])
        
        lines.append(f"  ★ RESULT: On '{metric}':")
        if all_res:
            for r in all_res[:5]:
                lines.append(f"    {r.get('label','?')}: {r.get('value','?')}")
        elif best and worst:
            lines.append(f"    Best: {best.get('label','?')} = {best.get('value','?')}")
            lines.append(f"    Worst: {worst.get('label','?')} = {worst.get('value','?')}")
    
    elif itype == "clean_ablation" or itype == "cross_dataset_ablation":
        param = ev.get("param", "?")
        values = ev.get("values", [])
        n_pairs = ev.get("n_pairs", 0)
        lines.append(f"  ABLATION: '{param}' tested with values {values[:6]} ({n_pairs} pairs)")
    
    elif itype == "parameter_range":
        param = ev.get("param", "?")
        values = ev.get("values", [])
        if any(isinstance(v, (int, float)) for v in values):
            lines.append(f"  RANGE: '{param}' = {values[:8]}")
        else:
            return None  # skip non-numeric parameter ranges
    
    elif itype == "experiment_family":
        varying = ev.get("varying", {})
        n_configs = ev.get("n_configs", 0)
        lines.append(f"  DESIGN: {n_configs} configs, varying: {dict(list(varying.items())[:5])}")
    
    elif itype == "dataset_comparison":
        datasets = ev.get("datasets", [])
        lines.append(f"  DATASETS: tested on {datasets}")
    
    else:
        lines.append(f"  {desc[:120]}")
    
    return "\n".join(lines) if lines else None


def build_rich_member_context(members, insights_by_title):
    """Build detailed context with actual numbers for each community member."""
    context = ""
    
    for title in members[:12]:  # cap at 12
        insights = insights_by_title.get(title, [])
        if not insights:
            continue
        
        context += f"\n[{title}]\n"
        
        # Separate evidence-rich from structure-only
        rich_insights = []
        structure_insights = []
        
        for ins in insights:
            ev = ins.get("evidence", {})
            has_numbers = (ev.get("delta") is not None or 
                          ev.get("value_from") is not None or
                          ev.get("best") is not None or
                          ev.get("all_results"))
            
            if has_numbers or ins["type"] in ("config_result_link", "result_comparison"):
                rich_insights.append(ins)
            else:
                structure_insights.append(ins)
        
        # Show rich insights first (with all numbers)
        for ins in rich_insights[:6]:
            formatted = format_insight_rich(ins, title)
            if formatted:
                context += formatted + "\n"
        
        # Then structure insights (abbreviated)
        for ins in structure_insights[:4]:
            formatted = format_insight_rich(ins, title)
            if formatted:
                context += formatted + "\n"
    
    return context


# =====================================================================
# Improved hypothesis prompt — demands specific numbers
# =====================================================================

GROUNDED_HYPOTHESIS_PROMPT = """You are a research advisor analyzing a tightly-connected cluster of {n_papers} ML papers. Every paper shares {shared_desc}.

Below is what each paper found experimentally. Lines marked ★ contain ACTUAL MEASURED RESULTS — use these numbers in your hypotheses.

MEMBER EXPERIMENTS:
{member_details}

COMMUNITY COVERAGE:
- Benchmarks: {bench_coverage}
- Parameters ablated: {ablation_coverage}

YOUR TASK: Generate 5 specific, evidence-grounded research hypotheses.

CRITICAL REQUIREMENTS:
1. Each hypothesis MUST cite specific numbers from at least 2 papers above
   GOOD: "Paper A achieved AUROC 94.3 with ResNet50, Paper B got 96.1 with WRN50 — try WRN50 on Paper A's benchmark"
   BAD: "Try a different backbone"
2. State the EXACT experiment: parameter name, values to test, benchmark, expected metric change
3. The expected outcome must be quantitative: "expect +2-3% AUROC" not "expect improvement"
4. Reference the ★ RESULT lines above — those are real measurements

Return ONLY a JSON array:
[{{
  "hypothesis": "specific experiment with numbers",
  "evidence": "Paper A: metric=X on benchmark B; Paper C: metric=Y with param=Z",
  "expected_outcome": "expect metric change of +/- N% based on patterns in Papers A,C",
  "confidence": "high|medium|low",
  "cited_papers": ["Paper A full title", "Paper C full title"],
  "gap_type": "unexplored_combination|untested_transfer|missing_ablation"
}}]"""


def generate_grounded_hypotheses(groups, insights_by_title, api_key):
    """Generate hypotheses with explicit metric grounding."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    all_results = []
    total_tokens = 0
    
    for gi, group in enumerate(groups):
        members = group["members"]
        shared_bench = group.get("shared_bench", [])
        shared_methods = group.get("shared_methods", [])
        
        shared_parts = []
        if shared_bench: shared_parts.append(f"benchmark(s): {', '.join(shared_bench)}")
        if shared_methods: shared_parts.append(f"method(s): {', '.join(shared_methods)}")
        shared_desc = " and ".join(shared_parts) if shared_parts else "experimental overlap"
        
        # Build rich context with numbers
        context = build_rich_member_context(members, insights_by_title)
        
        # Aggregate coverage
        bench_counter = Counter()
        abl_counter = Counter()
        for title in members:
            for ins in insights_by_title.get(title, []):
                ev = ins.get("evidence", {})
                if ins["type"] in ("clean_ablation", "cross_dataset_ablation"):
                    param = ev.get("param", "")
                    if param: abl_counter[param] += 1
        
        bench_str = ", ".join(shared_bench) if shared_bench else "various"
        abl_str = ", ".join(f"{a}({n})" for a, n in abl_counter.most_common(5)) or "none"
        
        prompt = GROUNDED_HYPOTHESIS_PROMPT.format(
            n_papers=len(members),
            shared_desc=shared_desc,
            member_details=context[:7000],
            bench_coverage=bench_str,
            ablation_coverage=abl_str,
        )
        
        print(f"\nGroup {gi+1} ({len(members)} papers, {shared_desc}):")
        
        try:
            resp = client.messages.create(model="claude-sonnet-4-20250514",
                max_tokens=3000, messages=[{"role":"user","content":prompt}])
            text = resp.content[0].text.strip()
            tokens = resp.usage.input_tokens + resp.usage.output_tokens
            total_tokens += tokens
            
            # Parse
            if "```" in text:
                for part in text.split("```"):
                    p = part.strip()
                    if p.startswith("json"): p = p[4:].strip()
                    if p.startswith("["): text = p; break
            s, e = text.find("["), text.rfind("]")
            if s >= 0 and e > s: text = text[s:e+1]
            hyps = json.loads(text) if text.startswith("[") else []
            
            # Check how many have actual numbers
            n_with_numbers = 0
            for h in hyps:
                evidence = h.get("evidence", "") + h.get("hypothesis", "") + h.get("expected_outcome", "")
                numbers = re.findall(r'\d+\.?\d*', evidence)
                if len(numbers) >= 2:
                    n_with_numbers += 1
            
            print(f"  Generated {len(hyps)} hypotheses, {n_with_numbers} with numbers ({tokens} tok)")
            for h in hyps[:2]:
                print(f"    H: {h.get('hypothesis','')[:120]}")
                print(f"    Evidence: {h.get('evidence','')[:100]}")
            
            all_results.append({
                "group_id": gi + 1,
                "group_size": len(members),
                "shared_bench": shared_bench,
                "shared_methods": shared_methods,
                "members": members,
                "hypotheses": hyps,
                "n_with_numbers": n_with_numbers,
                "tokens": tokens,
            })
            
        except Exception as ex:
            print(f"  ERROR: {ex}")
            all_results.append({
                "group_id": gi + 1,
                "group_size": len(members),
                "error": str(ex)[:100],
                "hypotheses": [],
            })
    
    return all_results, total_tokens


# =====================================================================
# Judge with same 4 metrics
# =====================================================================

JUDGE_PROMPT = """Rate each research hypothesis on FOUR dimensions (1-5 scale):

SPECIFICITY: Does it name exact parameters, values, and benchmarks?
  1=Vague 5=Precise (exact param names, values, benchmark)

GROUNDEDNESS: Is it backed by SPECIFIC NUMBERS from real papers?
  1=No numbers cited 3=Mentions papers but vaguely 5=Cites specific metric values (e.g. "AUROC 94.3") from named papers

ACTIONABILITY: Could a researcher run this experiment today?
  1=Too vague 5=Clear enough to write the config file

CROSS-PAPER NOVELTY: Does this require knowledge from multiple papers?
  1=Single paper only 5=Genuinely synthesizes across papers non-obviously

HYPOTHESES:
{hypotheses}

Respond ONLY with JSON array:
[{{"n":1,"specificity":4,"groundedness":5,"actionability":4,"cross_paper_novelty":3,"reasoning":"brief"}}]"""


def judge_hypotheses(hypotheses_flat, api_keys):
    """Rate with available judges."""
    print(f"\n{'='*65}")
    print("JUDGING GROUNDED HYPOTHESES")
    print("="*65)
    
    all_scores = {}
    total_tokens = {}
    batch_size = 8
    
    # Determine which judges are available
    judges = []
    if api_keys.get("anthropic"):
        judges.append(("claude", "anthropic"))
    if api_keys.get("gpt"):
        judges.append(("gpt", "gpt"))
    if api_keys.get("gemini"):
        judges.append(("gemini", "gemini"))
    
    for jname, _ in judges:
        all_scores[jname] = {}
        total_tokens[jname] = 0
    
    for batch_start in range(0, len(hypotheses_flat), batch_size):
        batch = hypotheses_flat[batch_start:batch_start+batch_size]
        hyp_text = "\n".join(
            f"{h['id']}. {h['hypothesis']}\n   Evidence: {h.get('evidence','')[:150]}\n   Expected: {h.get('expected_outcome','')[:100]}"
            for h in batch
        )
        prompt = JUDGE_PROMPT.format(hypotheses=hyp_text)
        bn = batch_start // batch_size + 1
        nb = (len(hypotheses_flat) + batch_size - 1) // batch_size
        print(f"\n  Batch {bn}/{nb}")
        
        for jname, key_name in judges:
            print(f"    {jname}...", end=" ", flush=True)
            
            if jname == "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=api_keys["anthropic"])
                try:
                    resp = client.messages.create(model="claude-opus-4-5",
                        max_tokens=3000, messages=[{"role":"user","content":prompt}])
                    text = resp.content[0].text
                    tokens = resp.usage.input_tokens + resp.usage.output_tokens
                except Exception as e:
                    print(f"FAIL"); continue
            
            elif jname == "gpt":
                from openai import OpenAI
                client = OpenAI(api_key=api_keys["gpt"])
                try:
                    resp = client.chat.completions.create(model="gpt-5.2",
                        messages=[{"role":"user","content":prompt}], max_completion_tokens=3000)
                    text = resp.choices[0].message.content
                    tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
                except Exception as e:
                    print(f"FAIL"); continue
            
            elif jname == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_keys["gemini"], transport="rest")
                try:
                    model = genai.GenerativeModel("gemini-2.5-pro")
                    resp = model.generate_content(prompt,
                        generation_config={"temperature":0.3, "max_output_tokens":4000},
                        request_options={"timeout": 120})
                    text = resp.text if hasattr(resp, 'text') else ""
                    tokens = len(prompt)//4 + len(text)//4
                except Exception as e:
                    print(f"FAIL ({str(e)[:40]})"); continue
            
            total_tokens[jname] += tokens
            
            # Parse scores
            parsed = []
            try:
                t = text.strip()
                if "```" in t:
                    for part in t.split("```"):
                        p = part.strip()
                        if p.startswith("json"): p = p[4:].strip()
                        if p.startswith("["): t = p; break
                s, e = t.find("["), t.rfind("]")
                if s >= 0 and e > s:
                    parsed = json.loads(t[s:e+1])
            except: pass
            
            n_rated = 0
            for item in parsed:
                n = item.get("n", 0)
                if "specificity" in item:
                    all_scores[jname][n] = {
                        "specificity": item.get("specificity", 0),
                        "groundedness": item.get("groundedness", 0),
                        "actionability": item.get("actionability", 0),
                        "cross_paper_novelty": item.get("cross_paper_novelty", 0),
                    }
                    n_rated += 1
            print(f"{n_rated} rated")
            time.sleep(1)
    
    return all_scores, total_tokens


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    
    api_keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "gpt": os.environ.get("OPENAI_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")),
    }
    if not api_keys["anthropic"]:
        print("Set ANTHROPIC_API_KEY"); sys.exit(1)
    
    random.seed(42)
    
    # Load data
    with open(args.baseline_results) as f:
        br = json.load(f)
    with open(args.groups) as f:
        groups = json.load(f)
    
    insights_by_title = {}
    for r in br["viveka"]:
        insights_by_title[r["title"]] = r.get("insights", [])
    
    print(f"Groups: {len(groups)}")
    print(f"Papers with insights: {len(insights_by_title)}")
    
    # Generate grounded hypotheses
    print(f"\n{'='*65}")
    print("GENERATING METRIC-GROUNDED HYPOTHESES")
    print("="*65)
    
    results, gen_tokens = generate_grounded_hypotheses(groups, insights_by_title, api_keys["anthropic"])
    
    # Save hypotheses
    with open(RESULTS_DIR / "hypotheses_grounded.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    total_hyps = sum(len(r.get("hypotheses", [])) for r in results)
    total_with_numbers = sum(r.get("n_with_numbers", 0) for r in results)
    print(f"\n{'='*65}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Total hypotheses: {total_hyps}")
    print(f"  With metric numbers: {total_with_numbers} ({100*total_with_numbers/max(total_hyps,1):.0f}%)")
    print(f"  Tokens: {gen_tokens:,}")
    print(f"  Cost: ~${gen_tokens*6/1_000_000:.2f}")
    
    # Judge
    if args.skip_judge:
        print("\nSkipping judge evaluation")
    else:
        # Flatten for judging
        flat = []
        for r in results:
            for h in r.get("hypotheses", []):
                flat.append({
                    "id": len(flat) + 1,
                    "hypothesis": h.get("hypothesis", "")[:200],
                    "evidence": h.get("evidence", "")[:200],
                    "expected_outcome": h.get("expected_outcome", "")[:100],
                    "cited_papers": h.get("cited_papers", []),
                    "group_id": r["group_id"],
                })
        
        random.shuffle(flat)
        scores, judge_tokens = judge_hypotheses(flat, api_keys)
        
        # Analyze
        active_judges = [j for j in scores if scores[j]]
        print(f"\n{'='*65}")
        print(f"JUDGE RESULTS (judges: {active_judges})")
        print("="*65)
        
        dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
        dim_totals = {d: [] for d in dims}
        
        for h in flat:
            hid = h["id"]
            for dim in dims:
                vals = []
                for j in active_judges:
                    s = scores[j].get(hid) or scores[j].get(str(hid))
                    if s and dim in s:
                        vals.append(s[dim])
                if vals:
                    dim_totals[dim].append(sum(vals)/len(vals))
        
        print(f"\n  {'Metric':<25} {'v2 (old)':>8} {'v3 (new)':>8} {'Change':>8}")
        print(f"  {'-'*50}")
        
        old_scores = {"specificity": 3.83, "groundedness": 2.33, 
                      "actionability": 3.50, "cross_paper_novelty": 3.25}
        
        for dim in dims:
            old = old_scores.get(dim, 0)
            new = sum(dim_totals[dim])/max(len(dim_totals[dim]),1) if dim_totals[dim] else 0
            delta = new - old
            print(f"  {dim:<25} {old:>8.2f} {new:>8.2f} {delta:>+8.2f}")
        
        new_mean = sum(sum(dim_totals[d])/max(len(dim_totals[d]),1) for d in dims) / len(dims)
        old_mean = sum(old_scores.values()) / len(old_scores)
        print(f"  {'MEAN':<25} {old_mean:>8.2f} {new_mean:>8.2f} {new_mean-old_mean:>+8.2f}")
        
        # Save
        with open(RESULTS_DIR / "judge_results.json", "w") as f:
            json.dump({
                "scores": scores,
                "judge_tokens": judge_tokens,
                "hypotheses": flat,
                "dimension_means": {d: round(sum(dim_totals[d])/max(len(dim_totals[d]),1), 2) 
                                    for d in dims},
            }, f, indent=2, default=str)
    
    # Report
    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write(f"Grounded Hypotheses v3\n")
        f.write(f"Total hypotheses: {total_hyps}\n")
        f.write(f"With numbers: {total_with_numbers}\n")
    
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
