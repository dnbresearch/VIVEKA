#!/usr/bin/env python3
"""
LLM Ablation: Which reasoning model produces the best hypotheses?
===================================================================
Tests the same community groups across multiple LLMs to isolate
the contribution of the graph structure vs the LLM reasoning.

Models tested:
  - Claude Sonnet 4     (Anthropic, default generator)
  - Claude Opus 4.5     (Anthropic, strongest)
  - GPT-5.2             (OpenAI)
  - Gemini 2.5 Pro      (Google)
  - Llama 70B           (Meta, via Together/Groq/etc)

All models receive IDENTICAL community evidence as input.
All hypotheses judged by the SAME judge (Claude Sonnet 4).

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...
  export GEMINI_API_KEY=...
  export TOGETHER_API_KEY=...   # for Llama

  python3 llm_ablation.py \
      --phase1 ./validation_results/scale/phase1_incremental.json \
      --threshold 6.0 \
      --groups 5

  # Test subset of models
  python3 llm_ablation.py --phase1 ... --models sonnet,opus,gpt
"""

import json, os, re, sys, time, random
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from community_hypothesis_v3 import (
    extract_features, compute_sim, normalize_venue, build_graph_batched,
    detect_communities, score_community, GENERIC_METHODS
)

RESULTS_DIR = Path("./validation_results/llm_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================
# Hypothesis generation prompt (identical for all models)
# =========================================================================
HYPOTHESIS_PROMPT = """You are a research advisor analyzing a cluster of {n_papers} ML papers from {venues}. These papers are experimentally related — they share {shared_desc}.

WHAT EACH MEMBER EXPLORED (lines with * contain actual measured values):
{member_details}

COMMUNITY EXPERIMENT COVERAGE:
- Shared benchmarks: {shared_bench}
- All benchmarks tested: {all_bench}
- Parameters ablated: {ablation_coverage}
- Key metrics: {metrics}

YOUR TASK: Find experiments that the community's work implies would be valuable but that NO member has actually tried.

Generate 5 specific, actionable research hypotheses.

For each hypothesis you MUST:
1. State the SPECIFIC experiment (exact parameter, values, benchmark)
2. Cite which members' results motivate it — quote SPECIFIC NUMBERS from the * lines above (e.g., "Paper A achieved mAP=78.3 with ResNet-50, Paper B showed +3.2% gain from curriculum learning")
3. Predict a QUANTITATIVE expected outcome (e.g., "expect 1-2% mAP improvement" not "expect improvement")
4. Rate confidence: high/medium/low

GROUNDING RULES (critical):
- Every hypothesis MUST cite at least one specific metric value from each of 2+ papers
- Use the actual numbers shown in the * lines — do NOT make up numbers
- If you cannot find specific numbers to support a hypothesis, do NOT include it
- "Paper A showed X=Y" is grounded. "Paper A explored X" is NOT grounded.

Return ONLY a JSON array:
[{{"hypothesis":"specific experiment","motivation":"Paper A: metric=value, Paper B: metric=value, therefore...","expected_outcome":"quantitative prediction","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"],"gap_type":"unexplored_combination|cross_venue_transfer|untested_transfer|missing_ablation"}}]"""

# =========================================================================
# LLM Judge prompt (same for all)
# =========================================================================
JUDGE_PROMPT = """Rate each research hypothesis on FOUR dimensions (1-5 scale):

SPECIFICITY: Does it name exact parameters, values, and benchmarks?
  1=Vague ("try larger models") 5=Precise ("test ViT-L-14 at 518px on VisA")

GROUNDEDNESS: Is it backed by specific metric values from cited papers?
  1=No evidence 5=Directly motivated by specific results

ACTIONABILITY: Could a researcher implement this experiment today?
  1=Too vague 5=Clear enough to write the config file

CROSS-PAPER NOVELTY: Does this hypothesis REQUIRE knowledge from multiple papers?
  1=Derivable from single paper 5=Genuinely synthesizes across papers

HYPOTHESES TO RATE:
{hypotheses}

Respond ONLY with a JSON array:
[{{"n":1,"specificity":4,"groundedness":5,"actionability":4,"cross_paper_novelty":3}}]"""


# =========================================================================
# Model-specific generation functions
# =========================================================================

def generate_sonnet(prompt, api_key):
    """Claude Sonnet 4"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip(), resp.usage.input_tokens + resp.usage.output_tokens
    except Exception as e:
        print(f"      Sonnet error: {e}")
        return f"ERROR: {e}", 0


def generate_opus(prompt, api_key):
    """Claude Opus 4.5"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model="claude-opus-4-5-20250918",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip(), resp.usage.input_tokens + resp.usage.output_tokens
    except Exception as e:
        print(f"      Opus error: {e}")
        return f"ERROR: {e}", 0


def generate_gpt(prompt, api_key):
    """GPT-5.2"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        text = resp.choices[0].message.content.strip()
        tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
        return text, tokens
    except Exception as e:
        print(f"      GPT error: {e}")
        return f"ERROR: {e}", 0


def generate_gemini(prompt, api_key):
    """Gemini 2.5 Pro — handles thinking blocks and markdown fences."""
    import google.generativeai as genai
    genai.configure(api_key=api_key, transport="rest")
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 4096},
            request_options={"timeout": 180}
        )
        # Extract text from all parts, skipping thinking blocks
        text = ""
        try:
            text = resp.text
        except:
            pass
        if not text:
            for cand in resp.candidates:
                for part in cand.content.parts:
                    # Skip thinking parts (Gemini 2.5 thinking mode)
                    if hasattr(part, "thought") and part.thought:
                        continue
                    if hasattr(part, "text") and part.text:
                        text += part.text
        # Gemini often wraps in markdown fences — aggressively extract JSON
        if text and "[" not in text[:50]:
            # Try to find JSON block in markdown fences
            json_blocks = re.findall(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
            if json_blocks:
                text = json_blocks[0]
        try:
            tokens = resp.usage_metadata.prompt_token_count + resp.usage_metadata.candidates_token_count
        except:
            tokens = len(prompt) // 4
        return text, tokens
    except Exception as e:
        print(f"      Gemini error: {e}")
        return f"ERROR: {e}", 0


def generate_llama(prompt, api_key):
    """Llama 70B via Together AI"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    try:
        resp = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        text = resp.choices[0].message.content.strip()
        tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens if resp.usage else len(prompt) // 4
        return text, tokens
    except Exception as e:
        print(f"      Llama error: {e}")
        return f"ERROR: {e}", 0


# =========================================================================
# Common utilities
# =========================================================================

def parse_json_array(text):
    if not text or text.startswith("ERROR"):
        return []
    # Strategy 1: direct parse
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except:
        pass
    # Strategy 2: extract from markdown fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        # Find all fenced blocks
        blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        for block in blocks:
            block = block.strip()
            if block.startswith("["):
                try:
                    return json.loads(block)
                except:
                    pass
    # Strategy 3: find [ ... ] anywhere in text
    s = text.find("[")
    e = text.rfind("]")
    if s >= 0 and e > s:
        fragment = text[s:e+1]
        try:
            return json.loads(fragment)
        except:
            pass
        # Strategy 4: fix common JSON issues
        fragment = re.sub(r',\s*]', ']', fragment)
        fragment = re.sub(r',\s*}', '}', fragment)
        # Fix unescaped newlines in strings
        fragment = fragment.replace('\n', ' ')
        try:
            return json.loads(fragment)
        except:
            pass
        # Strategy 5: try fixing truncated JSON (add missing brackets)
        open_braces = fragment.count('{') - fragment.count('}')
        open_brackets = fragment.count('[') - fragment.count(']')
        if open_braces > 0 or open_brackets > 0:
            fragment += '}' * open_braces + ']' * open_brackets
            try:
                return json.loads(fragment)
            except:
                pass
    return []


def format_insight_with_evidence(ins):
    """Format a single insight with actual metric values prominently displayed."""
    ev = ins.get("evidence", {})
    desc = ins.get("description", "")[:120]
    itype = ins["type"]
    lines = [f"  [{itype}] {desc}"]

    # Extract actual numbers from evidence
    if itype == "config_result_link":
        changes = ev.get("config_changes", {})
        metric = ev.get("metric", "")
        before = ev.get("before", "")
        after = ev.get("after", "")
        if metric and before and after:
            try:
                delta = float(after) - float(before)
                lines.append(f"    * {metric}: {before} -> {after} (delta={delta:+.2f})")
            except:
                lines.append(f"    * {metric}: {before} -> {after}")
        if changes:
            lines.append(f"    * Changed: {changes}")

    elif itype == "clean_ablation":
        param = ev.get("param", "")
        values = ev.get("values", [])
        metric = ev.get("metric", "")
        if param and values:
            lines.append(f"    * Ablated {param}: {values[:6]}")
        if metric:
            lines.append(f"    * Metric: {metric}")
        # Include result rows if available
        for row in ev.get("all_results", [])[:3]:
            label = row.get("label", "")
            val = row.get("value", "")
            if label and val:
                lines.append(f"    * {label} = {val}")

    elif itype == "cross_dataset_ablation":
        param = ev.get("param", "")
        metric = ev.get("metric", "")
        if param:
            lines.append(f"    * Varied: {param}")
        for row in ev.get("all_results", [])[:4]:
            label = row.get("label", "")
            val = row.get("value", "")
            if label and val:
                lines.append(f"    * {label} = {val}")

    elif itype == "result_comparison":
        metric = ev.get("metric", "")
        for row in ev.get("all_results", [])[:4]:
            label = row.get("label", "")
            val = row.get("value", "")
            if label and val:
                lines.append(f"    * {label}: {metric}={val}")

    elif itype == "parameter_range":
        param = ev.get("param", "")
        values = ev.get("values", [])
        if param and values:
            lines.append(f"    * {param} range: {values[:8]}")

    elif itype == "experiment_family":
        varying = ev.get("varying", {})
        n_configs = ev.get("n_configs", "")
        if varying:
            lines.append(f"    * Varying: {dict(list(varying.items())[:4])}")
        if n_configs:
            lines.append(f"    * {n_configs} configurations")

    return "\n".join(lines)


def build_prompt_for_group(group, papers, insights_map):
    """Build the IDENTICAL prompt for a community group — with metric values."""
    members = group["members"]
    shared_bench = group.get("shared_bench", [])
    shared_methods = group.get("shared_methods", [])

    shared_parts = []
    if shared_bench:
        shared_parts.append(f"benchmark(s): {', '.join(shared_bench)}")
    if shared_methods:
        shared_parts.append(f"method(s): {', '.join(shared_methods)}")
    shared_desc = " and ".join(shared_parts) if shared_parts else "experimental overlap"

    member_details = ""
    bench_coverage = Counter()
    abl_coverage = Counter()
    metrics = Counter()
    for t in members:
        f = papers.get(t, {})
        ins = insights_map.get(t, [])
        member_details += f"\n[{t}]\n"
        if f.get("benchmarks"):
            member_details += f"  Benchmarks: {sorted(f['benchmarks'])}\n"
        if f.get("methods"):
            member_details += f"  Methods: {sorted(f['methods'])}\n"
        for b in f.get("benchmarks", []):
            bench_coverage[b] += 1
        for a in f.get("ablation_params", []):
            abl_coverage[a] += 1
        for m in f.get("metrics", []):
            metrics[m] += 1
        # Prioritize insights with actual result data
        sorted_ins = sorted(ins, key=lambda x: (
            x["type"] in ("config_result_link", "result_comparison"),  # best: has numbers
            x["type"] in ("clean_ablation", "cross_dataset_ablation"),  # good: has ablation data
            bool(x.get("evidence", {}).get("all_results")),  # has result rows
            bool(x.get("evidence", {}).get("metric")),  # has metric name
        ), reverse=True)
        for i in sorted_ins[:6]:
            member_details += format_insight_with_evidence(i) + "\n"

    prompt = HYPOTHESIS_PROMPT.format(
        n_papers=len(members),
        venues=", ".join(group.get("venues", ["various"])),
        shared_desc=shared_desc,
        member_details=member_details[:8000],
        shared_bench=", ".join(shared_bench) or "none shared",
        all_bench=", ".join(f"{b}({n})" for b, n in bench_coverage.most_common()) or "various",
        ablation_coverage=", ".join(f"{a}({n})" for a, n in abl_coverage.most_common()) or "none",
        metrics=", ".join(m for m, _ in metrics.most_common(8)) or "various",
    )
    return prompt


def judge_hypotheses(hypotheses, api_key):
    """Judge using Claude Sonnet 4 (consistent judge across all models)."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    hyp_text = "\n".join(
        f"{i+1}. {h.get('hypothesis', '?')[:200]}\n"
        f"   Cited: {h.get('cited_papers', [])[:3]}\n"
        f"   Expected: {h.get('expected_outcome', '?')[:150]}"
        for i, h in enumerate(hypotheses)
    )
    prompt = JUDGE_PROMPT.format(hypotheses=hyp_text)

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        parsed = parse_json_array(text)
        scores = {}
        for item in parsed:
            n = item.get("n", 0)
            if "specificity" in item:
                scores[n] = {
                    "specificity": item.get("specificity", 0),
                    "groundedness": item.get("groundedness", 0),
                    "actionability": item.get("actionability", 0),
                    "cross_paper": item.get("cross_paper_novelty", item.get("cross_paper", 0)),
                }
        return scores, tokens
    except Exception as e:
        print(f"      Judge error: {e}")
        return {}, 0


GROUNDING_PROMPT = """You are verifying and improving the grounding of research hypotheses.

Below are hypotheses generated from a community of {n_papers} related ML papers, followed by the actual experimental evidence from those papers.

HYPOTHESES:
{hypotheses_text}

AVAILABLE EVIDENCE (lines with * contain actual measured values):
{evidence_text}

For EACH hypothesis:
1. Find the SPECIFIC numbers from the evidence above that support it
2. Rewrite the motivation to cite exact metric values (e.g., "Paper A: mAP=78.3 on COCO with ResNet-50")
3. Make the expected outcome quantitative (e.g., "expect 1.5-3.0% improvement")
4. If no specific numbers support a hypothesis, mark confidence as "low"

Return ONLY a JSON array with the improved hypotheses:
[{{"hypothesis":"same or refined","motivation":"with specific numbers from evidence","expected_outcome":"quantitative","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"],"gap_type":"same"}}]"""


def two_pass_ground(hypotheses, group, papers, insights_map, gen_fn, api_key):
    """Pass 2: Re-ground hypotheses with explicit number citation."""
    members = group["members"]

    # Build evidence text with actual numbers
    evidence_text = ""
    for t in members:
        ins = insights_map.get(t, [])
        if not ins:
            continue
        evidence_text += f"\n[{t}]\n"
        for i in ins[:8]:
            evidence_text += format_insight_with_evidence(i) + "\n"

    hyp_text = ""
    for i, h in enumerate(hypotheses):
        hyp_text += f"\nH{i+1}: {h.get('hypothesis', '?')}\n"
        hyp_text += f"  Motivation: {h.get('motivation', '?')[:200]}\n"
        hyp_text += f"  Expected: {h.get('expected_outcome', '?')[:150]}\n"
        hyp_text += f"  Cited: {h.get('cited_papers', [])}\n"

    prompt = GROUNDING_PROMPT.format(
        n_papers=len(members),
        hypotheses_text=hyp_text[:4000],
        evidence_text=evidence_text[:6000],
    )

    text, tokens = gen_fn(prompt, api_key)
    improved = parse_json_array(text)

    if improved and len(improved) >= len(hypotheses) * 0.5:
        return improved, tokens
    return hypotheses, tokens


# =========================================================================
# Main
# =========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--groups", type=int, default=5, help="Number of groups to test")
    parser.add_argument("--models", type=str, default="sonnet,opus,gpt,gemini,llama",
                        help="Comma-separated list of models to test")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--two-pass", action="store_true",
                        help="Enable two-pass grounding: generate then verify/enhance citations")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of independent runs per model for statistical significance")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # API keys
    api_keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")),
        "together": os.environ.get("TOGETHER_API_KEY", ""),
    }

    # Model registry
    MODEL_REGISTRY = {
        "sonnet": ("Claude Sonnet 4", generate_sonnet, "anthropic"),
        "opus": ("Claude Opus 4.5", generate_opus, "anthropic"),
        "gpt": ("GPT-5.2", generate_gpt, "openai"),
        "gemini": ("Gemini 2.5 Pro", generate_gemini, "gemini"),
        "llama": ("Llama 3.3 70B", generate_llama, "together"),
    }

    models_to_test = [m.strip() for m in args.models.split(",")]
    active_models = {}
    for m in models_to_test:
        if m not in MODEL_REGISTRY:
            print(f"  Unknown model: {m}, skipping")
            continue
        name, gen_fn, key_name = MODEL_REGISTRY[m]
        if not api_keys.get(key_name):
            print(f"  {name}: no API key ({key_name.upper()}_API_KEY), skipping")
            continue
        active_models[m] = (name, gen_fn, api_keys[key_name])

    if not active_models:
        print("No models available. Set API keys and retry.")
        sys.exit(1)

    print("=" * 70)
    print("LLM ABLATION: REASONING MODEL COMPARISON")
    print("=" * 70)
    print(f"  Models: {[name for _, (name, _, _) in active_models.items()]}")

    # Load data
    with open(args.phase1) as f:
        p1 = json.load(f)

    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    papers = {}
    insights_map = {}
    venues = {}
    for r in graph_ready:
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
    print(f"  Papers: {len(titles)}, Threshold: {args.threshold}")

    # Build graph and detect communities
    G = build_graph_batched(papers, titles, venues, sim_threshold=args.threshold)
    raw_groups = detect_communities(G, papers, venues, min_community=3)

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
        ms = set(g["members"])
        if len(ms & used) / max(len(ms), 1) < 0.5:
            final_groups.append(g)
            used.update(ms)
        if len(final_groups) >= args.groups:
            break

    print(f"  Communities: {len(final_groups)} (top {args.groups})")

    # Build prompts ONCE (identical for all models)
    prompts = []
    for i, g in enumerate(final_groups):
        prompt = build_prompt_for_group(g, papers, insights_map)
        prompts.append((i, g, prompt))
        print(f"  Group {i+1}: {g['size']} papers, venues={g.get('venues', [])[:4]}, "
              f"bench={g.get('shared_bench', [])}")

    # Generate hypotheses with each model across multiple runs
    print(f"\n{'='*70}")
    print(f"GENERATING HYPOTHESES ({args.n_runs} run{'s' if args.n_runs > 1 else ''})")
    print("="*70)

    # all_run_results[model_key][run_idx] = {group_results, overall_mean, ...}
    all_run_results = {m: [] for m in active_models}
    judge_key = api_keys["anthropic"]

    for run_idx in range(args.n_runs):
        if args.n_runs > 1:
            print(f"\n{'='*50}")
            print(f"  RUN {run_idx+1}/{args.n_runs}")
            print(f"{'='*50}")

        for model_key, (model_name, gen_fn, api_key) in active_models.items():
            print(f"\n  --- {model_name} (run {run_idx+1}) ---")
            model_results = []
            total_tokens = 0

            for group_idx, group, prompt in prompts:
                print(f"    Group {group_idx+1}...", end=" ", flush=True)
                text, tokens = gen_fn(prompt, api_key)
                total_tokens += tokens
                hyps = parse_json_array(text)
                grounded = sum(1 for h in hyps if len(h.get("cited_papers", [])) >= 2)
                print(f"{len(hyps)} hyps ({grounded} grounded, {tokens} tok)", end="")

                if args.two_pass and hyps:
                    print(" -> pass 2...", end=" ", flush=True)
                    hyps, p2_tokens = two_pass_ground(hyps, group, papers, insights_map, gen_fn, api_key)
                    total_tokens += p2_tokens
                    grounded = sum(1 for h in hyps if len(h.get("cited_papers", [])) >= 2)
                    print(f"{len(hyps)} refined (+{p2_tokens} tok)")
                else:
                    print()

                # Judge immediately if not skipping
                dim_scores = {"specificity": 0, "groundedness": 0,
                              "actionability": 0, "cross_paper": 0}
                mean_score = 0
                if not args.skip_judge and hyps:
                    scores, jtokens = judge_hypotheses(hyps, judge_key)
                    total_tokens += jtokens
                    dims = ["specificity", "groundedness", "actionability", "cross_paper"]
                    for d in dims:
                        vals = [scores.get(i+1, {}).get(d, 0) for i in range(len(hyps))]
                        vals = [v for v in vals if v > 0]
                        dim_scores[d] = round(sum(vals) / max(len(vals), 1), 2)
                    mean_score = round(sum(dim_scores.values()) / 4, 2)

                model_results.append({
                    "group_idx": group_idx + 1,
                    "n_hypotheses": len(hyps),
                    "n_grounded": grounded,
                    "dim_scores": dim_scores,
                    "mean_score": mean_score,
                    "hypotheses": hyps,
                    "tokens": tokens,
                })
                time.sleep(1)

            # Compute run-level aggregates
            dims = ["specificity", "groundedness", "actionability", "cross_paper"]
            run_dims = {}
            for d in dims:
                vals = [gr["dim_scores"].get(d, 0) for gr in model_results if gr["dim_scores"].get(d, 0) > 0]
                run_dims[d] = round(sum(vals) / max(len(vals), 1), 2)
            run_mean = round(sum(run_dims.values()) / 4, 2)

            all_run_results[model_key].append({
                "run": run_idx + 1,
                "model_name": model_name,
                "results": model_results,
                "overall_scores": run_dims,
                "overall_mean": run_mean,
                "total_hypotheses": sum(r["n_hypotheses"] for r in model_results),
                "total_grounded": sum(r["n_grounded"] for r in model_results),
                "total_tokens": total_tokens,
                # Per-group mean scores for significance tests
                "group_means": [gr["mean_score"] for gr in model_results],
            })
            print(f"    Run {run_idx+1}: mean={run_mean:.2f}, "
                  f"{sum(r['n_hypotheses'] for r in model_results)} hyps")

    # =====================================================================
    # ANALYSIS: Aggregate across runs
    # =====================================================================
    print(f"\n\n{'='*70}")
    if args.n_runs > 1:
        print(f"RESULTS ACROSS {args.n_runs} RUNS (mean +/- std)")
    else:
        print("RESULTS")
    print("="*70)

    from scipy import stats as scipy_stats
    import numpy as np

    dims = ["specificity", "groundedness", "actionability", "cross_paper"]
    model_summaries = {}

    for model_key in models_to_test:
        if model_key not in all_run_results or not all_run_results[model_key]:
            continue
        runs = all_run_results[model_key]
        model_name = runs[0]["model_name"]

        # Collect per-run scores
        run_means = [r["overall_mean"] for r in runs]
        run_dims = {d: [r["overall_scores"].get(d, 0) for r in runs] for d in dims}
        run_hyps = [r["total_hypotheses"] for r in runs]

        # Compute mean +/- std
        summary = {
            "model_name": model_name,
            "mean": round(np.mean(run_means), 2),
            "std": round(np.std(run_means, ddof=1), 2) if len(run_means) > 1 else 0,
            "n_runs": len(runs),
            "n_hyp": round(np.mean(run_hyps), 1),
        }
        for d in dims:
            summary[f"{d}_mean"] = round(np.mean(run_dims[d]), 2)
            summary[f"{d}_std"] = round(np.std(run_dims[d], ddof=1), 2) if len(run_dims[d]) > 1 else 0

        # Store per-group means across runs (for significance tests)
        # Flatten: list of all group mean scores across all runs
        summary["all_group_means"] = []
        for r in runs:
            summary["all_group_means"].extend(r["group_means"])

        model_summaries[model_key] = summary

    # Print results table
    if args.n_runs > 1:
        print(f"\n  {'Model':<22} {'Spec':>12} {'Grnd':>12} {'Actn':>12} {'XPap':>12} {'Mean':>12} {'#Hyp':>6}")
        print(f"  {'-'*82}")
        for model_key in models_to_test:
            if model_key not in model_summaries:
                continue
            s = model_summaries[model_key]
            def fmt(d):
                return f"{s[f'{d}_mean']:.2f}+/-{s[f'{d}_std']:.2f}"
            print(f"  {s['model_name']:<22} "
                  f"{fmt('specificity'):>12} {fmt('groundedness'):>12} "
                  f"{fmt('actionability'):>12} {fmt('cross_paper'):>12} "
                  f"{s['mean']:.2f}+/-{s['std']:.2f}  {s['n_hyp']:>5.0f}")
    else:
        print(f"\n  {'Model':<22} {'Spec':>6} {'Grnd':>6} {'Actn':>6} {'XPap':>6} {'Mean':>6} {'#Hyp':>6}")
        print(f"  {'-'*58}")
        for model_key in models_to_test:
            if model_key not in model_summaries:
                continue
            s = model_summaries[model_key]
            print(f"  {s['model_name']:<22} "
                  f"{s['specificity_mean']:>6.2f} {s['groundedness_mean']:>6.2f} "
                  f"{s['actionability_mean']:>6.2f} {s['cross_paper_mean']:>6.2f} "
                  f"{s['mean']:>6.2f} {s['n_hyp']:>6.0f}")

    # Statistical significance (paired t-tests between models)
    if args.n_runs > 1 and len(model_summaries) >= 2:
        print(f"\n{'='*70}")
        print(f"STATISTICAL SIGNIFICANCE (paired t-test, per-group scores)")
        print("="*70)

        model_keys_sorted = sorted(model_summaries.keys(),
                                    key=lambda k: model_summaries[k]["mean"], reverse=True)

        print(f"\n  {'Pair':<40} {'t-stat':>8} {'p-value':>10} {'Sig':>6} {'Cohen d':>8}")
        print(f"  {'-'*72}")

        for i in range(len(model_keys_sorted)):
            for j in range(i+1, len(model_keys_sorted)):
                k1, k2 = model_keys_sorted[i], model_keys_sorted[j]
                s1 = model_summaries[k1]
                s2 = model_summaries[k2]
                scores1 = s1["all_group_means"]
                scores2 = s2["all_group_means"]

                # Ensure same length (trim to shorter)
                n = min(len(scores1), len(scores2))
                if n < 3:
                    continue
                scores1 = scores1[:n]
                scores2 = scores2[:n]

                # Paired t-test
                t_stat, p_val = scipy_stats.ttest_rel(scores1, scores2)

                # Cohen's d (paired)
                diffs = [a - b for a, b in zip(scores1, scores2)]
                d_mean = np.mean(diffs)
                d_std = np.std(diffs, ddof=1)
                cohen_d = d_mean / d_std if d_std > 0 else 0

                # Significance markers
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."

                pair_name = f"{s1['model_name']} vs {s2['model_name']}"
                print(f"  {pair_name:<40} {t_stat:>8.3f} {p_val:>10.4f} {sig:>6} {cohen_d:>8.3f}")

        print(f"\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
        print(f"  Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")

    # Also run Kruskal-Wallis (non-parametric, no normality assumption)
    if args.n_runs > 1 and len(model_summaries) >= 3:
        all_groups_by_model = []
        model_labels = []
        for mk in model_keys_sorted:
            all_groups_by_model.append(model_summaries[mk]["all_group_means"])
            model_labels.append(model_summaries[mk]["model_name"])

        try:
            h_stat, kw_p = scipy_stats.kruskal(*all_groups_by_model)
            print(f"\n  Kruskal-Wallis (non-parametric, all models): H={h_stat:.3f}, p={kw_p:.4f}")
            if kw_p < 0.05:
                print(f"  -> Significant difference exists between models (p<0.05)")
            else:
                print(f"  -> No significant difference between models")
        except:
            pass

    # LaTeX table
    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print("="*70)

    if args.n_runs > 1:
        print(r"""
\begin{table}[t]
\centering
\caption{LLM ablation (mean$\pm$std over """ + str(args.n_runs) + r""" runs). All models receive identical community evidence. Judged by Claude Sonnet 4.}
\label{tab:llm_ablation}
\begin{tabular}{l|rrrr|r}
\toprule
\textbf{Model} & \textbf{Spec} & \textbf{Grnd} & \textbf{Actn} & \textbf{XPap} & \textbf{Mean} \\
\midrule""")
        for model_key in models_to_test:
            if model_key not in model_summaries:
                continue
            s = model_summaries[model_key]
            def ltx(d):
                m = s[f'{d}_mean']
                sd = s[f'{d}_std']
                return f"${m:.2f}\\pm{sd:.2f}$"
            print(f"{s['model_name']} & {ltx('specificity')} & {ltx('groundedness')} & "
                  f"{ltx('actionability')} & {ltx('cross_paper')} & "
                  f"${s['mean']:.2f}\\pm{s['std']:.2f}$ \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")
    else:
        print(r"""
\begin{table}[t]
\centering
\caption{LLM ablation: hypothesis quality across reasoning models. All models receive identical community evidence. Judged by Claude Sonnet 4.}
\label{tab:llm_ablation}
\begin{tabular}{l|rrrr|r|r}
\toprule
\textbf{Model} & \textbf{Spec} & \textbf{Grnd} & \textbf{Actn} & \textbf{XPap} & \textbf{Mean} & \textbf{\#Hyp} \\
\midrule""")
        for model_key in models_to_test:
            if model_key not in model_summaries:
                continue
            s = model_summaries[model_key]
            print(f"{s['model_name']} & {s['specificity_mean']:.2f} & "
                  f"{s['groundedness_mean']:.2f} & {s['actionability_mean']:.2f} & "
                  f"{s['cross_paper_mean']:.2f} & {s['mean']:.2f} & {s['n_hyp']:.0f} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Save final results
    with open(RESULTS_DIR / "llm_ablation_results.json", "w") as f:
        save_data = {
            "n_runs": args.n_runs,
            "summaries": model_summaries,
            "all_runs": {k: v for k, v in all_run_results.items()},
        }
        json.dump(save_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f"\nAll saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
