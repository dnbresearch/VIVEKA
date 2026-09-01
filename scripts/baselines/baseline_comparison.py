#!/usr/bin/env python3
"""
Baseline Comparison v3: Large-Scale (3,699 repos)
====================================================
Adapted from v2 to work with phase1_incremental.json directly.

Baselines:
  B1: Standalone LLM          (Si et al. 2024 NeurIPS)
  B2: RAG + LLM               (Radensky et al. 2025 Scideator)
  B3: AblationBench-style     (Abramovich & Chechik 2025)
  B4: Abstract-keyword graph  (Xiong et al. 2024 KG-CoI)
  B5: AI Scientist-style      (Lu et al. 2024/2026 Nature)

Evaluation (4 metrics, LLM-as-judge):
  - Specificity (1-5)
  - Groundedness (1-5)
  - Actionability (1-5)
  - Cross-paper Novelty (1-5)

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 baseline_comparison_v3.py \
      --phase1 ./validation_results/scale/phase1_incremental.json \
      --our-hypotheses ./validation_results/full_ablation/ablation_results.json

  # Generate baselines only (no judging, cheaper)
  python3 baseline_comparison_v3.py \
      --phase1 ./validation_results/scale/phase1_incremental.json \
      --our-hypotheses ./validation_results/full_ablation/ablation_results.json \
      --skip-judge

  # Use specific threshold's hypotheses as "ours"
  python3 baseline_comparison_v3.py \
      --phase1 ... --our-hypotheses ... --our-threshold 4.0
"""

import json, os, re, sys, time, random
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

RESULTS_DIR = Path("./validation_results/baseline_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Feature extraction (same as community_hypothesis_v3)
# =====================================================================
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
    "kinetics":"Kinetics","ntu":"NTU","squad":"SQuAD","glue":"GLUE",
    "wmt":"WMT","scannet":"ScanNet","shapenet":"ShapeNet","modelnet":"ModelNet",
    "mnist":"MNIST","svhn":"SVHN","celeba":"CelebA","cityscapes":"Cityscapes",
    "lfw":"LFW","qm9":"QM9","ptb":"PTB","wiki":"WikiText"}
METHOD_KW = {"vit":"vision_transformer","resnet":"resnet","transformer":"transformer",
    "unet":"unet","diffusion":"diffusion","contrastive":"contrastive","clip":"clip",
    "anomaly":"anomaly_detection","segmentation":"segmentation","detection":"object_detection",
    "pretrain":"pretraining","self_supervised":"self_supervised","attention":"attention",
    "graph":"graph_nn","reinforcement":"rl","patchcore":"patchcore",
    "gan":"gan","bert":"bert","gpt":"gpt","llama":"llama","lora":"lora",
    "fine_tun":"finetuning","distill":"distillation","pruning":"pruning",
    "quantiz":"quantization","augment":"augmentation",
    "adversar":"adversarial","federat":"federated","prompt":"prompt_tuning"}

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

def normalize_venue(v):
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

# =====================================================================
# LLM calls
# =====================================================================
def call_anthropic(prompt, api_key, max_tokens=3000, retries=2):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(retries):
        try:
            resp = client.messages.create(model="claude-sonnet-4-20250514",
                max_tokens=max_tokens, messages=[{"role":"user","content":prompt}])
            return resp.content[0].text.strip(), resp.usage.input_tokens + resp.usage.output_tokens
        except Exception as e:
            if attempt < retries - 1: time.sleep(2)
            else: return f"ERROR: {e}", 0

def parse_json_array(text):
    if not text or text.startswith("ERROR"): return []
    try:
        arr = json.loads(text)
        if isinstance(arr, list): return arr
    except: pass
    cleaned = text
    if "```" in cleaned:
        for part in cleaned.split("```"):
            p = part.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("["): cleaned = p; break
    s, e = cleaned.find("["), cleaned.rfind("]")
    if s >= 0 and e > s:
        try: return json.loads(cleaned[s:e+1])
        except: pass
        fragment = cleaned[s:e+1]
        fragment = re.sub(r',\s*]', ']', fragment)
        fragment = re.sub(r',\s*}', '}', fragment)
        try: return json.loads(fragment)
        except: pass
    return []

# =====================================================================
# B1: Standalone LLM
# =====================================================================
def baseline_B1_standalone(papers, insights_map, api_key):
    print("\n  B1: Standalone LLM...")
    titles = list(papers.keys())
    # Sample 100 papers across the full dataset (LLM context limit per batch)
    sampled = random.sample(titles, min(100, len(titles)))
    all_hyps = []; total_tokens = 0

    for batch_idx in range(5):  # 5 batches of 20
        batch_titles = sampled[batch_idx*20:(batch_idx+1)*20]
        if not batch_titles: break
        context = ""
        for title in batch_titles:
            ins = insights_map.get(title, [])
            context += f"\n[{title}]\n"
            for i in ins[:4]:
                context += f"  [{i['type']}] {i['description'][:100]}\n"

        prompt = f"""You are an ML research advisor. Below are {len(batch_titles)} recent ML papers and their experimental findings extracted from their code repositories.

PAPERS:
{context[:5000]}

Generate 8 specific, actionable research hypotheses. Each must:
1. Name specific parameters, values, and benchmarks
2. Reference at least 2 papers from the list above
3. Predict an expected outcome

IMPORTANT: Return ONLY a valid JSON array. No other text before or after.
[
  {{"hypothesis": "test X on Y", "motivation": "Paper A shows..., Paper B shows...", "expected_outcome": "expect Z", "confidence": "high", "cited_papers": ["Paper A title", "Paper B title"], "gap_type": "unexplored_combination"}}
]"""
        text, tokens = call_anthropic(prompt, api_key)
        hyps = parse_json_array(text)
        if not hyps:
            prompt2 = f"""List 5 ML experiment ideas based on these papers. For each, name 2 papers, specific parameters, and expected outcome. Return JSON array only.
Papers: {', '.join(t[:40] for t in batch_titles[:10])}
[{{"hypothesis":"...","cited_papers":["...","..."],"expected_outcome":"...","confidence":"medium","gap_type":"missing_ablation"}}]"""
            text, tokens2 = call_anthropic(prompt2, api_key)
            hyps = parse_json_array(text)
            tokens += tokens2
        all_hyps.extend(hyps)
        total_tokens += tokens
    print(f"    Generated {len(all_hyps)} hypotheses ({total_tokens} tokens)")
    return all_hyps, total_tokens

# =====================================================================
# B2: RAG + LLM
# =====================================================================
def baseline_B2_rag(papers, insights_map, api_key, n_queries=10):
    print("\n  B2: RAG + LLM...")
    titles = list(papers.keys())
    all_hyps = []; total_tokens = 0
    seeds = random.sample(titles, min(n_queries, len(titles)))
    for seed in seeds:
        sf = papers[seed]
        seed_words = set(" ".join(list(sf["benchmarks"])+list(sf["methods"])+list(sf["params"])).lower().split())
        scored = []
        for t in titles:
            if t == seed: continue
            tf = papers[t]
            tw = set(" ".join(list(tf["benchmarks"])+list(tf["methods"])+list(tf["params"])).lower().split())
            overlap = len(seed_words & tw) / max(len(seed_words | tw), 1)
            scored.append((t, overlap))
        scored.sort(key=lambda x: -x[1])
        retrieved = [seed] + [t for t,_ in scored[:5]]
        context = ""
        for t in retrieved:
            context += f"\n[{t}]\n"
            for i in insights_map.get(t,[])[:5]:
                context += f"  [{i['type']}] {i['description'][:100]}\n"
        prompt = f"""You are an ML research advisor. Below are {len(retrieved)} related papers. Generate 3 specific research hypotheses based on gaps. Each must reference ≥2 papers, name specific params/values/benchmarks.

PAPERS:
{context[:4000]}

Return ONLY a JSON array:
[{{"hypothesis":"...","motivation":"...","expected_outcome":"...","confidence":"high|medium|low","cited_papers":["...","..."],"gap_type":"unexplored_combination|untested_transfer|missing_ablation"}}]"""
        text, tokens = call_anthropic(prompt, api_key)
        all_hyps.extend(parse_json_array(text))
        total_tokens += tokens
    print(f"    Generated {len(all_hyps)} hypotheses ({total_tokens} tokens)")
    return all_hyps, total_tokens

# =====================================================================
# B3: AblationBench-style (single-paper, no cross-paper)
# =====================================================================
def baseline_B3_ablation(papers, insights_map, api_key, n_papers=20):
    print("\n  B3: AblationBench-style...")
    titles = [t for t in papers if len(insights_map.get(t,[])) >= 3]
    selected = random.sample(titles, min(n_papers, len(titles)))
    all_hyps = []; total_tokens = 0
    for title in selected:
        context = f"[{title}]\n"
        for i in insights_map.get(title,[])[:8]:
            context += f"  [{i['type']}] {i['description'][:120]}\n"
        prompt = f"""Review this ML paper's experiments. Suggest 2 specific missing ablations. Name exact parameters and values.

{context}

Return ONLY JSON array:
[{{"hypothesis":"ablate X with values [a,b,c]","motivation":"why","expected_outcome":"predict","confidence":"high|medium|low","cited_papers":["{title}"],"gap_type":"missing_ablation"}}]"""
        text, tokens = call_anthropic(prompt, api_key)
        all_hyps.extend(parse_json_array(text))
        total_tokens += tokens
    print(f"    Generated {len(all_hyps)} hypotheses ({total_tokens} tokens)")
    return all_hyps, total_tokens

# =====================================================================
# B4: Abstract-keyword graph (title words only, no code features)
# =====================================================================
def baseline_B4_abstract_graph(papers, insights_map, api_key):
    print("\n  B4: Abstract-keyword graph...")
    STOPWORDS = {"a","an","the","for","and","or","of","in","on","to","with","via",
                 "from","by","is","are","at","as","its","using","based","towards",
                 "beyond","into","not","all","you","need","more","less","can","how"}
    G = nx.Graph()
    titles = list(papers.keys())
    title_kw = {}
    for t in titles:
        G.add_node(t)
        words = set(re.findall(r'[a-z]+', t.lower())) - STOPWORDS
        title_kw[t] = {w for w in words if len(w) > 2}
    for i in range(len(titles)):
        for j in range(i+1, len(titles)):
            shared = title_kw[titles[i]] & title_kw[titles[j]]
            if len(shared) >= 2:
                G.add_edge(titles[i], titles[j], weight=len(shared))
    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Find groups — use connected components if graph is too dense for cliques
    if G.number_of_edges() > 50000:
        print(f"    Graph too dense for clique finding, using components...")
        cliques = [list(c) for c in nx.connected_components(G) if len(c) >= 3]
    else:
        cliques = [c for c in nx.find_cliques(G) if len(c) >= 3]
        if not cliques:
            cliques = [list(c) for c in nx.connected_components(G) if len(c) >= 3]
    cliques.sort(key=lambda x: -len(x))
    groups = []; used = set()
    for cl in cliques:
        if len(set(cl) & used) / max(len(cl),1) > 0.7: continue
        groups.append(cl); used.update(cl)
        if len(groups) >= 10: break

    all_hyps = []; total_tokens = 0
    for group in groups:
        context = ""
        for t in group[:8]:
            context += f"\n[{t}]\n"
            for i in insights_map.get(t,[])[:4]:
                context += f"  [{i['type']}] {i['description'][:100]}\n"
        shared_kw = set.intersection(*[title_kw[t] for t in group]) if group else set()
        prompt = f"""You are analyzing {len(group)} related ML papers (topic: {list(shared_kw)[:5]}). Generate 3 specific hypotheses referencing ≥2 papers.

PAPERS:
{context[:4000]}

Return ONLY JSON array:
[{{"hypothesis":"...","motivation":"...","expected_outcome":"...","confidence":"high|medium|low","cited_papers":["...","..."],"gap_type":"unexplored_combination|untested_transfer|missing_ablation"}}]"""
        text, tokens = call_anthropic(prompt, api_key)
        all_hyps.extend(parse_json_array(text))
        total_tokens += tokens
    print(f"    Generated {len(all_hyps)} hypotheses ({total_tokens} tokens)")
    return all_hyps, total_tokens

# =====================================================================
# B5: AI Scientist-style (multi-step ideation + refinement)
# =====================================================================
def baseline_B5_ai_scientist(papers, insights_map, api_key):
    print("\n  B5: AI Scientist-style...")
    titles = list(papers.keys())
    # Sample diverse papers across the full dataset
    sampled = random.sample(titles, min(60, len(titles)))
    context = ""
    for t in sampled:
        context += f"- {t}\n"
        for i in insights_map.get(t,[])[:3]:
            context += f"    [{i['type']}] {i['description'][:80]}\n"

    # Step 1: Brainstorm
    prompt1 = f"""You are an AI research scientist. Below are {len(sampled)} ML papers with experimental findings from code.

{context[:6000]}

STEP 1: Brainstorm 10 diverse research ideas. Be creative. Each should combine insights from multiple papers.

Return JSON array:
[{{"idea": "brief description", "papers": ["paper1", "paper2"]}}]"""
    text1, tok1 = call_anthropic(prompt1, api_key)
    ideas = parse_json_array(text1)
    print(f"    Step 1: {len(ideas)} ideas")
    if not ideas: return [], tok1

    # Step 2: Refine
    ideas_text = "\n".join(f"{i+1}. {idea.get('idea','')[:100]} (papers: {idea.get('papers',[])})"
                           for i, idea in enumerate(ideas[:8]))
    prompt2 = f"""You previously brainstormed these research ideas:
{ideas_text}

Now REFINE the top 5 into specific, testable hypotheses. For each:
1. Name exact parameters, values, and benchmarks
2. Reference which papers' results motivate it
3. Predict expected outcome with confidence

Return ONLY JSON array:
[{{"hypothesis":"specific experiment","motivation":"which papers and why","expected_outcome":"predicted effect","confidence":"high|medium|low","cited_papers":["Paper A","Paper B"],"gap_type":"unexplored_combination|untested_transfer|missing_ablation"}}]"""
    text2, tok2 = call_anthropic(prompt2, api_key)
    hyps = parse_json_array(text2)
    total_tokens = tok1 + tok2
    print(f"    Step 2: {len(hyps)} hypotheses ({total_tokens} tokens)")
    return hyps, total_tokens

# =====================================================================
# LLM-as-Judge (4 metrics)
# =====================================================================
JUDGE_PROMPT = """Rate each research hypothesis on FOUR dimensions (1-5 scale):

SPECIFICITY: Does it name exact parameters, values, and benchmarks?
  1=Vague ("try larger models") 5=Precise ("test ViT-L-14 at 518px on VisA with batch_size=32")

GROUNDEDNESS: Is it backed by evidence from real papers' actual results?
  1=No evidence 5=Directly motivated by specific results from cited papers

ACTIONABILITY: Could a researcher implement this experiment today?
  1=Too vague 5=Clear enough to write the config file right now

CROSS-PAPER NOVELTY: Does this hypothesis REQUIRE knowledge from multiple papers?
  1=Could be generated from a single paper alone
  3=References multiple papers but the insight is obvious from either alone
  5=Genuinely synthesizes findings across papers in a non-obvious way
  NOTE: A hypothesis citing only 1 paper should score 1.

HYPOTHESES TO RATE:
{hypotheses}

Respond ONLY with a JSON array:
[{{"n":1,"specificity":4,"groundedness":5,"actionability":4,"cross_paper_novelty":3,"reasoning":"brief"}}]"""


def judge_hypotheses_batch(flat_batch, api_key):
    """Judge a batch of hypotheses using Claude as judge."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    hyp_text = "\n".join(
        f"{h['id']}. {h['hypothesis']}\n   Cited: {h['cited_papers'][:3]}\n   Expected: {h['expected_outcome']}"
        for h in flat_batch)
    prompt = JUDGE_PROMPT.format(hypotheses=hyp_text)

    try:
        resp = client.messages.create(model="claude-sonnet-4-20250514",
            max_tokens=3000, messages=[{"role":"user","content":prompt}])
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
                    "cross_paper_novelty": item.get("cross_paper_novelty", 0),
                }
        return scores, tokens
    except Exception as e:
        print(f"      Judge error: {e}")
        return {}, 0


def evaluate_all(all_methods_hyps, api_key):
    """Rate all hypotheses blind with 4 metrics."""
    print(f"\n{'='*70}")
    print("LLM-AS-JUDGE EVALUATION (4 metrics)")
    print("="*70)

    # Flatten and shuffle blind
    flat = []
    for method, hyps in all_methods_hyps.items():
        for h in hyps:
            flat.append({"id": len(flat)+1, "method": method,
                "hypothesis": h.get("hypothesis","")[:200],
                "cited_papers": h.get("cited_papers",[]),
                "expected_outcome": h.get("expected_outcome","")[:100]})
    random.shuffle(flat)
    print(f"  Total hypotheses to judge: {len(flat)}")

    all_scores = {}
    total_tokens = 0
    batch_size = 8

    for batch_start in range(0, len(flat), batch_size):
        batch = flat[batch_start:batch_start+batch_size]
        bn = batch_start//batch_size + 1
        nb = (len(flat)+batch_size-1)//batch_size
        print(f"  Batch {bn}/{nb}...", end=" ", flush=True)

        scores, tokens = judge_hypotheses_batch(batch, api_key)
        total_tokens += tokens
        all_scores.update(scores)
        print(f"{len(scores)} rated ({tokens} tokens)")
        time.sleep(0.5)

    # Map scores back to methods
    dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
    by_method = defaultdict(lambda: {d: [] for d in dims})

    for h in flat:
        s = all_scores.get(h["id"])
        if not s: continue
        for d in dims:
            v = s.get(d, 0)
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                by_method[h["method"]][d].append(v)

    return flat, by_method, total_tokens


def print_results(by_method, all_methods_hyps):
    """Print final comparison table."""
    dims = ["specificity", "groundedness", "actionability", "cross_paper_novelty"]
    dim_short = {"specificity": "Spec", "groundedness": "Grnd",
                 "actionability": "Actn", "cross_paper_novelty": "XPap"}

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("="*70)

    print(f"\n  {'Method':<22} {'Spec':>6} {'Grnd':>6} {'Actn':>6} {'XPap':>6} {'Mean':>6} {'Count':>6}")
    print(f"  {'-'*60}")

    results = {}
    method_order = ["ours", "B5_ai_scientist", "B1_standalone", "B2_rag",
                    "B3_ablation", "B4_abstract_graph"]

    for method in method_order:
        if method not in by_method: continue
        d = by_method[method]
        avgs = {}
        for dim in dims:
            avgs[dim] = round(sum(d[dim])/max(len(d[dim]),1), 2) if d[dim] else 0
        mean = round(sum(avgs.values())/len(dims), 2)
        n = len(all_methods_hyps.get(method, []))
        rated = max(len(d[dim]) for dim in dims) if any(d[dim] for dim in dims) else 0
        results[method] = {**avgs, "mean": mean, "n": n, "rated": rated}

        marker = " ◄" if method == "ours" else ""
        print(f"  {method:<22} {avgs.get('specificity',0):>6.2f} {avgs.get('groundedness',0):>6.2f} "
              f"{avgs.get('actionability',0):>6.2f} {avgs.get('cross_paper_novelty',0):>6.2f} "
              f"{mean:>6.2f} {n:>6}{marker}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    ours = results.get("ours", {})
    for baseline, label in [("B1_standalone", "Standalone LLM"),
                             ("B4_abstract_graph", "Abstract graph"),
                             ("B3_ablation", "Single-repo ablation"),
                             ("B5_ai_scientist", "AI Scientist")]:
        if baseline in results:
            diff = ours.get("mean", 0) - results[baseline].get("mean", 0)
            xp_diff = ours.get("cross_paper_novelty", 0) - results[baseline].get("cross_paper_novelty", 0)
            print(f"    vs {label:<22}: mean {diff:+.2f}, XPap {xp_diff:+.2f}")

    # Volume × Quality
    print(f"\n  VOLUME × QUALITY:")
    for method in method_order:
        if method not in results: continue
        r = results[method]
        effective = r["n"] * r["mean"]
        print(f"    {method:<22} {r['n']:>3} × {r['mean']:.2f} = {effective:>6.1f}")

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True, help="phase1_incremental.json")
    parser.add_argument("--our-hypotheses", required=True,
                        help="ablation_results.json or hypotheses.json from community_v3")
    parser.add_argument("--our-threshold", type=float, default=None,
                        help="Which threshold's hypotheses to use as 'ours' (default: best)")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key: print("Set ANTHROPIC_API_KEY"); sys.exit(1)

    # Load phase1 data
    print("="*70)
    print("BASELINE COMPARISON v3 (LARGE-SCALE)")
    print("="*70)

    with open(args.phase1) as f:
        p1 = json.load(f)
    print(f"Loaded {len(p1)} repos from phase1")

    # Extract features from graph-ready repos
    graph_ready = [r for r in p1 if r.get("n_insights", 0) > 0 and r.get("n_configs", 0) >= 3]
    papers = {}; insights_map = {}; venues = {}
    for r in graph_ready:
        title = r.get("title", "")
        ins = r.get("_insights", [])
        if not ins or not title: continue
        f = extract_features(title, ins)
        if len(f["params"]) >= 2 or f["benchmarks"] or f["methods"]:
            papers[title] = f
            insights_map[title] = ins
            venues[title] = normalize_venue(r.get("venue", "?"))

    print(f"Graph-ready: {len(graph_ready)}, Featured: {len(papers)}")

    # All baselines get access to ALL featured papers (same as our method)
    # Each baseline handles its own internal batching/sampling
    papers_for_baselines = papers
    insights_for_baselines = insights_map

    # Load our hypotheses
    with open(args.our_hypotheses) as f:
        our_data = json.load(f)

    # Extract our hypotheses - handle both ablation_results.json and hypotheses.json formats
    our_hyps = []
    if isinstance(our_data, list):
        # Could be ablation_results.json (list of threshold results) or hypotheses.json
        for item in our_data:
            if "group_results" in item:
                # ablation_results.json format
                threshold = item.get("threshold")
                if args.our_threshold and threshold != args.our_threshold:
                    continue
                for gr in item["group_results"]:
                    our_hyps.extend(gr.get("hypotheses", []))
            elif "hypotheses" in item:
                # community hypotheses.json format
                our_hyps.extend(item.get("hypotheses", []))
            elif "hypothesis" in item:
                # flat list of hypotheses
                our_hyps.append(item)

    # If no threshold specified and we got from ablation, pick best
    if not our_hyps and isinstance(our_data, list):
        for item in our_data:
            if "group_results" in item:
                for gr in item["group_results"]:
                    our_hyps.extend(gr.get("hypotheses", []))

    print(f"Our hypotheses: {len(our_hyps)}")

    # Generate baselines
    print(f"\n{'='*70}")
    print("GENERATING BASELINE HYPOTHESES")
    print("="*70)
    total_gen_tokens = 0

    b1, t1 = baseline_B1_standalone(papers_for_baselines, insights_for_baselines, api_key)
    total_gen_tokens += t1
    b2, t2 = baseline_B2_rag(papers_for_baselines, insights_for_baselines, api_key)
    total_gen_tokens += t2
    b3, t3 = baseline_B3_ablation(papers_for_baselines, insights_for_baselines, api_key)
    total_gen_tokens += t3
    b4, t4 = baseline_B4_abstract_graph(papers_for_baselines, insights_for_baselines, api_key)
    total_gen_tokens += t4
    b5, t5 = baseline_B5_ai_scientist(papers_for_baselines, insights_for_baselines, api_key)
    total_gen_tokens += t5

    all_methods = {"ours": our_hyps, "B1_standalone": b1, "B2_rag": b2,
                   "B3_ablation": b3, "B4_abstract_graph": b4, "B5_ai_scientist": b5}

    print(f"\n{'='*70}")
    print("HYPOTHESIS COUNTS")
    print("="*70)
    for m, h in all_methods.items():
        gr = sum(1 for x in h if len(x.get("cited_papers",[])) >= 2)
        print(f"  {m:<22} total={len(h):>3}  grounded(≥2)={gr:>3}")
    print(f"\n  Generation cost: ~${total_gen_tokens*6/1_000_000:.2f}")

    # Save baselines
    with open(RESULTS_DIR / "all_hypotheses.json", "w") as f:
        json.dump(all_methods, f, indent=2, default=str)

    # Judge
    if args.skip_judge:
        print("\nSkipping judge (--skip-judge). Hypotheses saved.")
    else:
        # Sample for judging (max 15 per method)
        sampled_methods = {}
        for m, h in all_methods.items():
            if h:
                sampled_methods[m] = random.sample(h, min(15, len(h))) if len(h) > 15 else h

        flat, by_method, judge_tokens = evaluate_all(sampled_methods, api_key)
        results = print_results(by_method, all_methods)

        # Save
        with open(RESULTS_DIR / "final_results.json", "w") as f:
            json.dump({
                "results": results,
                "counts": {m: len(h) for m, h in all_methods.items()},
                "generation_tokens": total_gen_tokens,
                "judge_tokens": judge_tokens,
                "total_cost": round((total_gen_tokens + judge_tokens) * 6 / 1_000_000, 2),
            }, f, indent=2, default=str)

        # LaTeX table
        print(f"\n{'='*70}")
        print("LATEX TABLE")
        print("="*70)
        print(r"""
\begin{table}[t]
\centering
\caption{Baseline comparison on 4 evaluation dimensions. All methods receive the same VIVEKA-extracted insights as input. Ours uses community-based cross-paper grouping.}
\label{tab:baselines}
\begin{tabular}{l|rrrr|r|r}
\toprule
Method & Spec & Grnd & Actn & XPap & Mean & \#Hyp \\
\midrule""")
        for method in ["ours","B5_ai_scientist","B1_standalone","B2_rag","B3_ablation","B4_abstract_graph"]:
            if method not in results: continue
            r = results[method]
            label = {"ours":"\\textbf{Ours}","B1_standalone":"B1: Standalone LLM",
                     "B2_rag":"B2: RAG+LLM","B3_ablation":"B3: AblationBench",
                     "B4_abstract_graph":"B4: Abstract Graph","B5_ai_scientist":"B5: AI Scientist"}.get(method,method)
            bold = method == "ours"
            fmt = "\\textbf{%.2f}" if bold else "%.2f"
            print(f"{label} & {fmt % r['specificity']} & {fmt % r['groundedness']} & "
                  f"{fmt % r['actionability']} & {fmt % r['cross_paper_novelty']} & "
                  f"{fmt % r['mean']} & {r['n']} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print(f"\nAll saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
