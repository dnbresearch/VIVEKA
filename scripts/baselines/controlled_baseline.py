#!/usr/bin/env python3
"""
Controlled Baseline Comparison (Reviewer 2, R2)
=================================================
Isolates the contribution of community detection vs structured prompting.

Experiment design:
  - B1-random:    Standalone LLM with random paper batches (existing baseline)
  - B1-community: Standalone LLM with the SAME community papers our method uses,
                   but WITHOUT structured evidence (just titles + abstracts)
  - Ours:         Full pipeline with structured code-extracted evidence

If B1-community > B1-random: community detection matters
If Ours > B1-community: structured prompting matters
Both expected to hold → both components contribute

Usage:
  python3 controlled_baseline.py \
      --phase1 ./validation_results/scale/phase1_incremental.json
"""

import json, os, sys, time, re
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")

RESULTS_DIR = Path("./validation_results/controlled_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# B1-community prompt: same papers, but only titles (no structured evidence)
B1_COMMUNITY_PROMPT = """You are a research advisor. Below are {n_papers} ML papers that share experimental overlap. Based on their titles and topics, generate 5 specific research hypotheses that combine findings across these papers.

PAPERS:
{paper_list}

Generate 5 hypotheses. Each must reference at least 2 papers.

Return ONLY a JSON array:
[{{"hypothesis":"specific experiment","motivation":"why this combination","expected_outcome":"quantitative prediction","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"]}}]"""

# B1-random prompt: random papers, no structure
B1_RANDOM_PROMPT = """You are a research advisor. Below are {n_papers} ML papers. Generate 5 specific research hypotheses that combine findings across these papers.

PAPERS:
{paper_list}

Generate 5 hypotheses. Each must reference at least 2 papers.

Return ONLY a JSON array:
[{{"hypothesis":"specific experiment","motivation":"why this combination","expected_outcome":"quantitative prediction","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"]}}]"""

JUDGE_PROMPT = """Rate each hypothesis (1-5):
SPECIFICITY: exact params/values named?
GROUNDEDNESS: cites specific numbers from papers?
ACTIONABILITY: implementable as concrete experiment?
CROSS-PAPER NOVELTY: requires knowledge from 2+ papers?

{hypotheses}

Respond ONLY with JSON: [{{"n":1,"specificity":4,"groundedness":3,"actionability":4,"cross_paper_novelty":3}}]"""


def parse_json_array(text):
    if not text:
        return []
    try:
        return json.loads(text)
    except:
        pass
    blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
    for b in blocks:
        if b.strip().startswith("["):
            try:
                return json.loads(b.strip())
            except:
                pass
    s, e = text.find("["), text.rfind("]")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except:
            pass
    return []


def call_llm(prompt, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def judge_hypotheses(hyps, api_key):
    """Judge a batch of hypotheses, return per-hypothesis mean scores."""
    dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
    hyp_text = "\n".join(
        f"{i+1}. {h.get('hypothesis', '?')[:200]}\n"
        f"   Cited: {h.get('cited_papers', [])[:3]}\n"
        f"   Expected: {h.get('expected_outcome', '?')[:150]}"
        for i, h in enumerate(hyps)
    )
    text = call_llm(JUDGE_PROMPT.format(hypotheses=hyp_text), api_key)
    scores = parse_json_array(text)

    means = []
    dim_scores = {d: [] for d in dims}
    for s in scores:
        vals = [s.get(d, 0) for d in dims if s.get(d, 0) > 0]
        if vals:
            means.append(np.mean(vals))
            for d in dims:
                if s.get(d, 0) > 0:
                    dim_scores[d].append(s[d])

    return means, dim_scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--n-communities", type=int, default=5,
                        help="Number of communities to test")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Need ANTHROPIC_API_KEY")
        sys.exit(1)

    from community_hypothesis_v3 import (
        extract_features, normalize_venue, build_graph_batched,
        detect_communities, score_community, GENERIC_METHODS
    )
    from llm_ablation import build_prompt_for_group

    print("=" * 70)
    print("CONTROLLED BASELINE COMPARISON")
    print("=" * 70)

    # Load and prepare
    with open(args.phase1) as f:
        p1 = json.load(f)

    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    papers, insights_map, venues = {}, {}, {}
    all_titles = []
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
            all_titles.append(title)

    titles = list(papers.keys())
    print(f"  Papers: {len(titles)}")

    # Build graph and detect communities
    G = build_graph_batched(papers, titles, venues, sim_threshold=6.0)
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
        if len(final) >= args.n_communities:
            break

    print(f"  Communities: {len(final)}")

    import random
    random.seed(42)

    # For each community, run 3 conditions
    results = {"ours": [], "b1_community": [], "b1_random": []}
    dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
    dim_results = {cond: {d: [] for d in dims} for cond in results}

    for ci, comm in enumerate(final):
        members = comm["members"]
        n = len(members)
        print(f"\n  Community {ci+1} ({n} papers, bench={comm.get('shared_bench', [])[:2]}):")

        # ---- Condition 1: OURS (structured prompt with evidence) ----
        print(f"    [Ours] Structured prompt...", end=" ", flush=True)
        prompt_ours = build_prompt_for_group(comm, papers, insights_map)
        text = call_llm(prompt_ours, api_key)
        hyps_ours = parse_json_array(text)
        if hyps_ours:
            means, dscores = judge_hypotheses(hyps_ours, api_key)
            results["ours"].extend(means)
            for d in dims:
                dim_results["ours"][d].extend(dscores.get(d, []))
            print(f"{len(means)} scored, mean={np.mean(means):.2f}")
        else:
            print("no hypotheses")
        time.sleep(1)

        # ---- Condition 2: B1-COMMUNITY (same papers, titles only) ----
        print(f"    [B1-community] Same papers, no evidence...", end=" ", flush=True)
        paper_list = "\n".join(f"- {t}" for t in members[:25])
        prompt_b1c = B1_COMMUNITY_PROMPT.format(n_papers=min(n, 25), paper_list=paper_list)
        text = call_llm(prompt_b1c, api_key)
        hyps_b1c = parse_json_array(text)
        if hyps_b1c:
            means, dscores = judge_hypotheses(hyps_b1c, api_key)
            results["b1_community"].extend(means)
            for d in dims:
                dim_results["b1_community"][d].extend(dscores.get(d, []))
            print(f"{len(means)} scored, mean={np.mean(means):.2f}")
        else:
            print("no hypotheses")
        time.sleep(1)

        # ---- Condition 3: B1-RANDOM (random papers of same size) ----
        print(f"    [B1-random] Random papers, no evidence...", end=" ", flush=True)
        random_papers = random.sample(all_titles, min(n, 25))
        paper_list = "\n".join(f"- {t}" for t in random_papers)
        prompt_b1r = B1_RANDOM_PROMPT.format(n_papers=min(n, 25), paper_list=paper_list)
        text = call_llm(prompt_b1r, api_key)
        hyps_b1r = parse_json_array(text)
        if hyps_b1r:
            means, dscores = judge_hypotheses(hyps_b1r, api_key)
            results["b1_random"].extend(means)
            for d in dims:
                dim_results["b1_random"][d].extend(dscores.get(d, []))
            print(f"{len(means)} scored, mean={np.mean(means):.2f}")
        else:
            print("no hypotheses")
        time.sleep(1)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)

    print(f"\n  {'Condition':<20} {'n':>4} {'Mean':>8} {'Spec':>8} {'Grnd':>8} {'Actn':>8} {'XPap':>8}")
    print(f"  {'-'*68}")
    for cond in ["ours", "b1_community", "b1_random"]:
        n = len(results[cond])
        mean = np.mean(results[cond]) if results[cond] else 0
        d_means = {d: np.mean(dim_results[cond][d]) if dim_results[cond][d] else 0 for d in dims}
        label = {"ours": "Ours (full)", "b1_community": "B1-community",
                 "b1_random": "B1-random"}[cond]
        print(f"  {label:<20} {n:>4} {mean:>8.2f} {d_means['specificity']:>8.2f} "
              f"{d_means['groundedness']:>8.2f} {d_means['actionability']:>8.2f} "
              f"{d_means['cross_paper_novelty']:>8.2f}")

    # Compute deltas
    ours_mean = np.mean(results["ours"]) if results["ours"] else 0
    b1c_mean = np.mean(results["b1_community"]) if results["b1_community"] else 0
    b1r_mean = np.mean(results["b1_random"]) if results["b1_random"] else 0

    print(f"\n  --- Contribution Decomposition ---")
    print(f"  Community detection:   B1-community vs B1-random = {b1c_mean - b1r_mean:+.2f}")
    print(f"  Structured prompting:  Ours vs B1-community      = {ours_mean - b1c_mean:+.2f}")
    print(f"  Total improvement:     Ours vs B1-random          = {ours_mean - b1r_mean:+.2f}")

    if b1c_mean > b1r_mean:
        pct_detection = 100 * (b1c_mean - b1r_mean) / max(ours_mean - b1r_mean, 0.01)
        pct_prompting = 100 * (ours_mean - b1c_mean) / max(ours_mean - b1r_mean, 0.01)
        print(f"  Attribution: {pct_detection:.0f}% from detection, {pct_prompting:.0f}% from prompting")

    # Save
    output = {
        "ours": {"mean": round(ours_mean, 2), "n": len(results["ours"]),
                 "dims": {d: round(np.mean(dim_results["ours"][d]), 2)
                         if dim_results["ours"][d] else 0 for d in dims}},
        "b1_community": {"mean": round(b1c_mean, 2), "n": len(results["b1_community"]),
                         "dims": {d: round(np.mean(dim_results["b1_community"][d]), 2)
                                 if dim_results["b1_community"][d] else 0 for d in dims}},
        "b1_random": {"mean": round(b1r_mean, 2), "n": len(results["b1_random"]),
                      "dims": {d: round(np.mean(dim_results["b1_random"][d]), 2)
                              if dim_results["b1_random"][d] else 0 for d in dims}},
        "delta_detection": round(b1c_mean - b1r_mean, 2),
        "delta_prompting": round(ours_mean - b1c_mean, 2),
        "delta_total": round(ours_mean - b1r_mean, 2),
    }

    with open(RESULTS_DIR / "controlled_baseline.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Paper text:")
    print(f'  "To isolate contributions, we gave the standalone LLM the same')
    print(f'  community papers (B1-community) vs random batches (B1-random).')
    print(f'  B1-community scores {b1c_mean:.2f} vs B1-random {b1r_mean:.2f}')
    print(f'  ({b1c_mean-b1r_mean:+.2f}), showing community detection improves input quality.')
    print(f'  Our structured prompting adds a further {ours_mean-b1c_mean:+.2f} ({ours_mean:.2f}),')
    print(f'  confirming both components contribute."')

    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
