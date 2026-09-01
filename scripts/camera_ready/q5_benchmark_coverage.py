#!/usr/bin/env python3
"""
Q5: Benchmark False-Negative Rate Estimation
=============================================
Uses LLM to identify benchmarks/datasets in repository configs
that the system's fixed vocabulary misses.

Approach:
1. Sample 100 repos with insights
2. Extract ALL dataset-like strings from their configs/insights
3. Ask LLM to classify which are real ML benchmarks
4. Compare against system's 27-keyword vocabulary
5. Compute false-negative rate

Usage:
  export ANTHROPIC_API_KEY=your_key
  python3 q5_benchmark_coverage.py
"""
import json, os, sys, re, time
from collections import defaultdict

PHASE1_PATH = "validation_results/scale/phase1_incremental.json"
RESULTS_FILE = "q5_benchmark_coverage.json"
N_SAMPLE = 100  # repos to sample

# System's current benchmark vocabulary (27 keywords)
SYSTEM_BENCHMARKS = {
    "imagenet", "cifar", "coco", "voc", "cityscapes", "ade20k",
    "mnist", "svhn", "celeba", "lfw", "kinetics", "ucf",
    "sst", "glue", "squad", "wmt", "wikitext", "lambada",
    "modelnet", "shapenet", "scannet", "ntu", "kitti",
    "mvtec", "visa", "mpdd", "btad"
}


def extract_dataset_strings(insights):
    """Extract all potential dataset/benchmark mentions from insights."""
    candidates = set()

    for ins in insights:
        ins_str = json.dumps(ins) if isinstance(ins, dict) else str(ins)

        # Pattern 1: dataset field values
        for pattern in [
            r'"dataset["\s:]*["\s]*([A-Za-z0-9_\-\.]{2,40})"',
            r'"data_dir["\s:]*["\s]*/[^"]*?/([A-Za-z0-9_\-]{3,30})[/"]',
            r'"data_path["\s:]*["\s]*/[^"]*?/([A-Za-z0-9_\-]{3,30})[/"]',
            r'"benchmark["\s:]*["\s]*([A-Za-z0-9_\-]{3,30})"',
            r'"eval_dataset["\s:]*["\s]*([A-Za-z0-9_\-]{3,30})"',
            r'"train_dataset["\s:]*["\s]*([A-Za-z0-9_\-]{3,30})"',
            r'"test_dataset["\s:]*["\s]*([A-Za-z0-9_\-]{3,30})"',
            r'"data_name["\s:]*["\s]*([A-Za-z0-9_\-]{3,30})"',
            r'"corpus["\s:]*["\s]*([A-Za-z0-9_\-]{3,30})"',
        ]:
            matches = re.findall(pattern, ins_str, re.IGNORECASE)
            for m in matches:
                m_clean = m.strip().lower()
                # Filter obvious non-benchmarks
                if m_clean not in {"none", "null", "true", "false", "data", "train",
                                   "test", "val", "default", "custom", "local", "path",
                                   "config", "model", "output", "input", "result",
                                   "checkpoint", "log", "save", "load", "experiment"}:
                    candidates.add(m_clean)

        # Pattern 2: known dataset name patterns (capitalized words)
        caps = re.findall(r'\b([A-Z][A-Za-z0-9]*(?:[-_][A-Z][A-Za-z0-9]*)*)\b', ins_str)
        for c in caps:
            if len(c) >= 3 and c.lower() not in {"true", "false", "none", "adam", "sgd",
                                                    "relu", "gelu", "linear", "conv", "batch",
                                                    "layer", "model", "train", "test", "eval",
                                                    "config", "yaml", "json", "torch", "cuda",
                                                    "float", "int", "bool", "string", "list"}:
                candidates.add(c.lower())

    return candidates


def classify_with_llm(candidates, api_key, batch_size=50):
    """Use LLM to classify which candidates are real ML benchmarks/datasets."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    all_classified = {}
    candidate_list = sorted(candidates)

    for i in range(0, len(candidate_list), batch_size):
        batch = candidate_list[i:i + batch_size]

        prompt = f"""You are an ML research expert. Below is a list of strings extracted from ML code repositories.
For each string, classify whether it is a **real ML benchmark or dataset** (like CIFAR-10, ImageNet, COCO, SQuAD, etc.)
or **not a benchmark** (like a method name, parameter, architecture, random string, etc.).

Respond ONLY with a JSON object mapping each string to either "benchmark" or "not_benchmark".
Include a brief note for benchmarks identifying the domain (CV, NLP, audio, tabular, graph, medical, etc.).

Strings to classify:
{json.dumps(batch, indent=2)}

Respond with JSON only, no markdown:
{{"string1": {{"class": "benchmark", "domain": "CV"}}, "string2": {{"class": "not_benchmark"}}}}"""

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text.strip()

            # Parse JSON
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Try extracting JSON from response
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    parsed = {}

            all_classified.update(parsed)
            print(f"    Classified batch {i//batch_size + 1}: {len(batch)} strings")
        except Exception as e:
            print(f"    Error on batch {i//batch_size + 1}: {e}")

        time.sleep(1)

    return all_classified


def main():
    print("=" * 65)
    print("Q5: Benchmark False-Negative Rate Estimation (LLM-assisted)")
    print("=" * 65)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    with open(PHASE1_PATH) as f:
        data = json.load(f)

    # Sample repos with insights
    repos_with_insights = [r for r in data if r.get("n_insights", 0) > 0 and r.get("_insights")]
    import random
    random.seed(42)
    sample = random.sample(repos_with_insights, min(N_SAMPLE, len(repos_with_insights)))
    print(f"  Sampled {len(sample)} repos (of {len(repos_with_insights)} with insights)")

    # Extract all candidate dataset strings
    print(f"\n  Extracting dataset-like strings...")
    all_candidates = set()
    per_repo_candidates = {}

    for r in sample:
        title = r.get("title", "?")[:60]
        insights = r.get("_insights", [])
        candidates = extract_dataset_strings(insights)
        all_candidates.update(candidates)
        per_repo_candidates[title] = candidates

    print(f"  Found {len(all_candidates)} unique candidate strings across {len(sample)} repos")

    # Check which are detected by system vocabulary
    detected_by_system = set()
    missed_by_system = set()

    for c in all_candidates:
        if any(b in c for b in SYSTEM_BENCHMARKS):
            detected_by_system.add(c)
        else:
            missed_by_system.add(c)

    print(f"  Detected by system vocabulary: {len(detected_by_system)}")
    print(f"  Not in system vocabulary: {len(missed_by_system)}")

    # Use LLM to classify missed candidates
    print(f"\n  Classifying {len(missed_by_system)} unrecognized strings with LLM...")
    classifications = classify_with_llm(missed_by_system, api_key)

    # Analyze results
    true_benchmarks_missed = {}
    not_benchmarks = []

    for string, info in classifications.items():
        if isinstance(info, dict):
            cls = info.get("class", "not_benchmark")
            domain = info.get("domain", "unknown")
        elif isinstance(info, str):
            cls = info
            domain = "unknown"
        else:
            continue

        if cls == "benchmark":
            true_benchmarks_missed[string] = domain
        else:
            not_benchmarks.append(string)

    # Also count system-detected benchmarks
    true_benchmarks_detected = len(detected_by_system)
    true_benchmarks_total = true_benchmarks_detected + len(true_benchmarks_missed)

    false_negative_rate = len(true_benchmarks_missed) / max(true_benchmarks_total, 1) * 100

    # Results
    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"\n  System vocabulary: {len(SYSTEM_BENCHMARKS)} benchmarks")
    print(f"  Candidates extracted: {len(all_candidates)}")
    print(f"  Detected by vocabulary: {true_benchmarks_detected}")
    print(f"  Real benchmarks missed: {len(true_benchmarks_missed)}")
    print(f"  False-negative rate: {false_negative_rate:.1f}%")

    if true_benchmarks_missed:
        print(f"\n  --- Missed Benchmarks (by domain) ---")
        by_domain = defaultdict(list)
        for name, domain in true_benchmarks_missed.items():
            by_domain[domain].append(name)

        for domain in sorted(by_domain.keys()):
            names = sorted(by_domain[domain])
            print(f"    {domain}: {', '.join(names[:10])}")
            if len(names) > 10:
                print(f"      ... and {len(names)-10} more")

    # Per-repo analysis
    repos_with_missed = 0
    for title, candidates in per_repo_candidates.items():
        missed_in_repo = [c for c in candidates if c in true_benchmarks_missed]
        if missed_in_repo:
            repos_with_missed += 1

    print(f"\n  Repos affected by missed benchmarks: {repos_with_missed}/{len(sample)} "
          f"({100*repos_with_missed/len(sample):.1f}%)")

    # Save
    output = {
        "n_repos_sampled": len(sample),
        "n_candidates": len(all_candidates),
        "n_detected": true_benchmarks_detected,
        "n_missed_benchmarks": len(true_benchmarks_missed),
        "false_negative_rate": round(false_negative_rate, 1),
        "missed_benchmarks": true_benchmarks_missed,
        "repos_affected_pct": round(100 * repos_with_missed / len(sample), 1),
        "system_vocabulary_size": len(SYSTEM_BENCHMARKS),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {RESULTS_FILE}")

    # Paper sentence
    print(f"\n  --- For rebuttal ---")
    print(f"  \"We estimated the benchmark false-negative rate by sampling {len(sample)} repos,")
    print(f"  extracting {len(all_candidates)} dataset-like strings, and using an LLM to classify")
    print(f"  those outside our 27-keyword vocabulary. Of {true_benchmarks_total} total benchmarks")
    print(f"  identified, {len(true_benchmarks_missed)} ({false_negative_rate:.1f}%) were missed by")
    print(f"  the fixed vocabulary, affecting {repos_with_missed}/{len(sample)} repos ({100*repos_with_missed/len(sample):.1f}%).\"")


if __name__ == "__main__":
    main()
