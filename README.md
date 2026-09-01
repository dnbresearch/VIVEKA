# VIVEKA: Cross-Paper Hypothesis Generation via Experiment-Similarity Graphs

[![Paper](https://img.shields.io/badge/ICDM%202026-Accepted-brightgreen)](https://icdm2026.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**VIVEKA** mines structured experimental features from ML code repositories, constructs experiment-similarity graphs, detects communities of related papers, and generates cross-paper hypotheses grounded in actual experimental evidence.

> **Accepted at ICDM 2026** — IEEE International Conference on Data Mining

## Overview

Machine learning researchers collectively explore vast experimental spaces — ablation studies, benchmark evaluations, hyperparameter sweeps — yet cross-paper synthesis remains manual. VIVEKA automates this by:

1. **Mining** structured experimental features (benchmarks, methods, parameters, metrics) from 12,303 code repositories across 28 venues
2. **Constructing** an experiment-similarity graph where edges encode shared experimental configurations
3. **Detecting** cross-venue research communities via Louvain + coherence-based clique refinement
4. **Generating** cross-paper hypotheses grounded in members' actual metric values

## Key Results

| Metric | Value |
|---|---|
| Repositories processed | 12,303 |
| Graph-ready papers | 2,530 |
| Hypotheses at τ=6.0 | 50 (mean quality 3.79/5.0) |
| vs. Standalone LLM (B1) | +0.66 quality |
| vs. Abstract-keyword graph (B4) | +1.00 quality |
| Experimental validation | 4/8 validated, 87.5% directional accuracy |
| Human evaluation (6 experts) | 3.0/5 plausibility, 4/6 would use system |

## Installation

```bash
git clone https://github.com/dnbresearch/VIVEKA.git
cd VIVEKA
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- An Anthropic API key (for hypothesis generation and evaluation)
- ~50GB disk space for full repository cloning

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Quick Start

### 1. Mine experimental features from repositories

```bash
# Process a list of repositories
python viveka/scale_evaluation.py \
    --input data/repo_list.json \
    --output results/phase1_incremental.json \
    --max-repos 100
```

### 2. Build experiment-similarity graph and detect communities

```bash
python viveka/community_detection.py \
    --phase1 results/phase1_incremental.json \
    --threshold 6.0 \
    --output results/communities.json
```

### 3. Generate cross-paper hypotheses

```bash
python viveka/hypothesis_generation.py \
    --phase1 results/phase1_incremental.json \
    --communities results/communities.json \
    --output results/hypotheses.json
```

### 4. Evaluate hypotheses

```bash
python viveka/evaluate.py \
    --hypotheses results/hypotheses.json \
    --output results/evaluation.json
```

## Project Structure

```
VIVEKA/
├── viveka/                          # Core pipeline
│   ├── __init__.py
│   ├── scale_evaluation.py          # Stage 1: Repository mining
│   ├── community_detection.py       # Stages 2-3: Graph + communities
│   ├── hypothesis_generation.py     # Stage 4: LLM hypothesis generation
│   └── evaluate.py                  # Evaluation (LLM judge + metrics)
│
├── scripts/                         # Analysis & ablation scripts
│   ├── baselines/
│   │   ├── baseline_comparison.py   # 5-baseline comparison
│   │   ├── controlled_baseline.py   # Controlled comparison (same papers)
│   │   └── controlled_baseline_opus.py  # With Opus judge
│   ├── ablations/
│   │   ├── llm_ablation.py          # 7 LLMs × 5 runs
│   │   ├── threshold_ablation.py    # τ ∈ [2.0, 6.0]
│   │   ├── edge_weight_ablation.py  # Component ablation for Eq. 1
│   │   └── community_stability.py   # NMI across 10 seeds
│   ├── analysis/
│   │   ├── paper_analyses.py        # Gap types, diversity, calibration
│   │   ├── inter_judge.py           # Inter-judge agreement (κ, τ)
│   │   ├── review_fixes.py          # Dual judge, novelty, weight sensitivity
│   │   └── reviewer2_experiments.py # Embedding baseline, Jaccard null
│   ├── validation/                  # Experimental hypothesis validation
│   │   ├── run_h1_hf.py             # H1: BLIP LR ablation (COCO)
│   │   ├── run_h4_resolution.py     # H4: CLIP resolution ablation
│   │   ├── run_h8_ablation.py       # H8: RealNet resolution (MVTec-AD)
│   │   ├── run_dm_hypothesis.py     # DM: GAT+DropEdge (Cora/CiteSeer)
│   │   ├── run_dm_residual.py       # DM: Residual connections in GCN
│   │   ├── run_nlp_hypothesis2.py   # NLP: R-Drop for BERT
│   │   ├── run_batch3.py            # NLP: DoLa, data quality, calibration
│   │   └── run_ts_hypothesis.py     # TS: RevIN for forecasting
│   ├── domain/
│   │   └── dm_repos.py              # DM venue repository collection
│   └── camera_ready/
│       ├── camera_ready_experiments.py  # All camera-ready experiments
│       ├── q2_uncovered.py           # Uncovered paper characterization
│       ├── q5_benchmark_coverage.py  # Benchmark vocabulary FNR
│       └── reviewer3_experiments.py  # Extraction by year, cross-τ NMI
│
├── data/                            # Data files
│   ├── repo_list_sample.json        # Sample repository list (100 repos)
│   └── venue_list.json              # 28 venue definitions
│
├── configs/                         # Configuration files
│   └── default.yaml                 # Default pipeline parameters
│
├── paper/                           # Paper and figures
│   ├── viveka_icdm_paper_v4.tex     # Camera-ready paper
│   └── figures/                     # All paper figures
│       ├── fig1_pipeline_clean.pdf
│       ├── fig2_baselines.pdf
│       ├── fig2_communities.pdf
│       ├── fig3_ablation.pdf
│       ├── fig4_volume_quality.pdf
│       └── fig5_llm_ablation.pdf
│
├── results/                         # Result files (gitignored, regenerable)
│   └── .gitkeep
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline Details

### Stage 1: Experimental Feature Extraction

Mines code repositories for structured experimental data:
- **YAML/JSON/TOML** configuration files
- **argparse** defaults from Python scripts
- **Shell scripts** with hyperparameter sweeps
- **README** result tables

Extracts five feature types per paper:
- **Benchmarks** (B): Canonical datasets (COCO, ImageNet, CIFAR, etc.)
- **Methods** (M): Model architectures and techniques
- **Parameters** (P): Hyperparameter names and ranges
- **Ablations** (A): Parameters varied in controlled experiments
- **Metrics** (μ): Performance values marked with ★ for grounding

### Stage 2: Graph Construction

Computes pairwise experiment similarity:

```
w(i,j) = 3.0·|B_i ∩ B_j| + 2.0·|M_i ∩ M_j| + 1.0·|P_i ∩ P_j| + 2.5·|A_i ∩ A_j|
          + 0.3·I[venue_i = venue_j]
```

Uses inverted-index batching for 91% reduction in pairwise comparisons.

### Stage 3: Community Detection

Louvain partitioning with coherence-based clique refinement:
1. Standard Louvain on the similarity graph
2. Coherence scoring: shared benchmarks × venue diversity
3. Clique refinement to break overly large communities

### Stage 4: Hypothesis Generation

LLM reasoning scoped to each community with structured evidence:
- Papers and their ★-marked experimental evidence
- Shared benchmarks and methods as context
- Prompt requires citing specific numbers from evidence

## Reproducing Paper Results

### Main evaluation (Table I)

```bash
python viveka/scale_evaluation.py --phase1 results/phase1_incremental.json
python scripts/baselines/baseline_comparison.py --phase1 results/phase1_incremental.json
```

### LLM ablation (Table III, Figure 5)

```bash
python scripts/ablations/llm_ablation.py --phase1 results/phase1_incremental.json
```

### Threshold ablation (Table II, Figure 3)

```bash
python scripts/ablations/threshold_ablation.py --phase1 results/phase1_incremental.json
```

### Controlled baseline with judge sensitivity (Section VI)

```bash
python scripts/baselines/controlled_baseline.py --phase1 results/phase1_incremental.json
python scripts/baselines/controlled_baseline_opus.py --phase1 results/phase1_incremental.json
```

### Experimental validation of hypotheses (Table IX)

```bash
# H1: BLIP LR ablation
python scripts/validation/run_h1_hf.py

# H4: CLIP resolution
python scripts/validation/run_h4_resolution.py

# DM: GAT+DropEdge on Cora/CiteSeer
python scripts/validation/run_dm_hypothesis.py

# NLP: R-Drop, DoLa, data quality, calibration
python scripts/validation/run_nlp_hypothesis2.py
python scripts/validation/run_batch3.py
```

### Camera-ready experiments (E3-E14)

```bash
export ANTHROPIC_API_KEY=your_key
python scripts/camera_ready/camera_ready_experiments.py \
    --phase1 results/phase1_incremental.json
```

## Dataset

The full dataset comprises:
- **28 venues**: NeurIPS, ICML, ICLR, CVPR, ECCV, ACL, EMNLP, NAACL, KDD, ICDM, CIKM, WSDM, and 16 others
- **12,303 repositories** linked from PapersWithCode and OpenReview
- **2,530 graph-ready papers** with ≥3 configuration files and sufficient features
- **40,681 structured insights** extracted from 152,965 configuration files

Due to size constraints, the full dataset is not included in this repository. To reproduce:

```bash
# Download repository list
python viveka/fetch_papers.py --output data/repo_list.json

# Run extraction (requires ~50GB disk, ~18 hours on 16 threads)
python viveka/scale_evaluation.py --input data/repo_list.json --output results/phase1_incremental.json
```

A sample of 100 repositories is included in `data/repo_list_sample.json` for testing.

## Citation

If you use VIVEKA in your research, please cite:

```bibtex
@inproceedings{bhatia2026viveka,
  title={Cross-Paper Hypothesis Generation via Experiment-Similarity Graphs},
  author={Bhatia, Divyansh and Sidhaiyan, Dhiyanesh},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the six ML researchers who participated in our human evaluation study, and the anonymous ICDM reviewers whose detailed feedback substantially improved the paper.
