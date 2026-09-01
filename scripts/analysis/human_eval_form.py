#!/usr/bin/env python3
"""
Human Evaluation Form Generator (Reviewer 2, R1)
==================================================
Generates a ready-to-share evaluation form for ML researchers.
Selects 10 diverse hypotheses and creates a Google Forms-style sheet.

Usage:
  python3 human_eval_form.py \
      --llm-results ./validation_results/llm_ablation/llm_ablation_results.json
"""

import json, os, sys, random
from pathlib import Path

RESULTS_DIR = Path("./validation_results/human_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-results", required=True)
    parser.add_argument("--n-hypotheses", type=int, default=10)
    args = parser.parse_args()

    with open(args.llm_results) as f:
        data = json.load(f)

    # Extract hypotheses from best model
    all_runs = data.get("all_runs", {})
    hypotheses = []
    for mk, runs in all_runs.items():
        if runs:
            for gr in runs[0].get("results", []):
                gidx = gr.get("group_idx", 0)
                for h in gr.get("hypotheses", []):
                    if h.get("hypothesis"):
                        h["community"] = gidx
                        hypotheses.append(h)
            break

    # Select diverse sample (stratified by community)
    random.seed(42)
    communities = {}
    for h in hypotheses:
        c = h.get("community", 0)
        if c not in communities:
            communities[c] = []
        communities[c].append(h)

    selected = []
    comm_keys = sorted(communities.keys())
    while len(selected) < args.n_hypotheses and comm_keys:
        for c in comm_keys:
            if communities[c] and len(selected) < args.n_hypotheses:
                selected.append(communities[c].pop(random.randint(0, len(communities[c])-1)))
        comm_keys = [c for c in comm_keys if communities[c]]

    print(f"Selected {len(selected)} hypotheses from {len(set(h['community'] for h in selected))} communities")

    # Generate Markdown evaluation form
    form = []
    form.append("# Hypothesis Quality Evaluation Form")
    form.append("")
    form.append("**Instructions:** Please rate each hypothesis on three dimensions using a 1-5 scale.")
    form.append("You do NOT need to run the experiment — rate based on your expert judgment.")
    form.append("")
    form.append("## Rating Dimensions")
    form.append("")
    form.append("**Scientific Plausibility (1-5):** Is this experiment scientifically sound?")
    form.append("Would you expect the predicted outcome to be approximately correct?")
    form.append("- 1 = Nonsensical or contradicts known results")
    form.append("- 3 = Plausible but uncertain")
    form.append("- 5 = Highly likely to produce a meaningful result")
    form.append("")
    form.append("**Practical Value (1-5):** Would you or a colleague choose to run this experiment?")
    form.append("- 1 = No researcher would find this interesting")
    form.append("- 3 = Somewhat interesting but low priority")
    form.append("- 5 = I would run this experiment myself")
    form.append("")
    form.append("**Feasibility (1-5):** Could this experiment be run with reasonable resources?")
    form.append("- 1 = Requires unavailable data/compute/access")
    form.append("- 3 = Feasible but requires significant effort")
    form.append("- 5 = Could be run in a day with standard hardware")
    form.append("")
    form.append("---")
    form.append("")
    form.append("## Your Background")
    form.append("")
    form.append("- **Name (optional):** ____________")
    form.append("- **Research area:** ____________")
    form.append("- **Years in ML research:** ____________")
    form.append("")
    form.append("---")
    form.append("")

    for i, h in enumerate(selected):
        form.append(f"## Hypothesis {i+1}")
        form.append("")
        form.append(f"**Proposed experiment:** {h.get('hypothesis', 'N/A')}")
        form.append("")
        form.append(f"**Motivation:** {h.get('motivation', 'N/A')}")
        form.append("")
        form.append(f"**Expected outcome:** {h.get('expected_outcome', 'N/A')}")
        form.append("")
        form.append(f"**Cited papers:** {', '.join(h.get('cited_papers', []))}")
        form.append("")
        form.append(f"**System confidence:** {h.get('confidence', 'N/A')}")
        form.append("")
        form.append("| Dimension | Score (1-5) | Notes |")
        form.append("|-----------|-------------|-------|")
        form.append("| Scientific Plausibility | | |")
        form.append("| Practical Value | | |")
        form.append("| Feasibility | | |")
        form.append("")
        form.append("---")
        form.append("")

    form.append("## Summary")
    form.append("")
    form.append("**Overall impression of hypothesis quality (1-5):** ____")
    form.append("")
    form.append("**Would you use a system that generates hypotheses like these? (Y/N):** ____")
    form.append("")
    form.append("**Any additional comments:**")
    form.append("")
    form.append("")
    form.append("---")
    form.append("*Thank you for your evaluation. Your responses will be anonymized in the paper.*")

    form_text = "\n".join(form)

    with open(RESULTS_DIR / "evaluation_form.md", "w") as f:
        f.write(form_text)

    # Also save as JSON for programmatic use
    with open(RESULTS_DIR / "selected_hypotheses.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved to:")
    print(f"  {RESULTS_DIR}/evaluation_form.md    — share with evaluators")
    print(f"  {RESULTS_DIR}/selected_hypotheses.json — for analysis")
    print(f"\nDistribute evaluation_form.md to 3-5 ML researchers.")
    print(f"Collect responses, compute inter-rater agreement (Krippendorff's α).")
    print(f"Report: 'X ML researchers rated 10 hypotheses on plausibility,")
    print(f"value, and feasibility. Mean plausibility: Y/5, mean value: Z/5.'")


if __name__ == "__main__":
    main()
