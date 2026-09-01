# VIVEKA: Cross-Paper Hypothesis Generation via Experiment-Similarity Graphs

[![Paper](https://img.shields.io/badge/ICDM%202026-Accepted-brightgreen)](https://icdm2026.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**VIVEKA** mines structured experimental features from ML code repositories, constructs experiment-similarity graphs, detects cross-venue research communities, and generates cross-paper hypotheses grounded in actual experimental evidence.

> **Accepted at ICDM 2026** — IEEE International Conference on Data Mining

## Overview

ML researchers collectively explore vast experimental spaces — ablation studies, benchmark evaluations, hyperparameter sweeps — yet cross-paper synthesis remains manual. VIVEKA automates this via a 4-stage pipeline:

1. **Mine** structured features (benchmarks, methods, parameters, metrics) from 12,303 code repositories
2. **Construct** an experiment-similarity graph (edges = shared experimental configurations)
3. **Detect** cross-venue communities via Louvain + coherence-based clique refinement
4. **Generate** hypotheses grounded in community members' actual metric values

## Key Results

| Metric | Value |
|---|---|
| Repositories processed | 12,303 across 28 venues |
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
pip install -e .
```

Or without install:
```bash
pip install -r requirements.txt
```

### Prerequisites

- Python 3.9+
- Anthropic API key (for hypothesis generation and evaluation)

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Quick Start

### Option 1: Full pipeline (one command)

```bash
export ANTHROPIC_API_KEY=your_key
./run_pipeline.sh            # Full run (~18 hours for Stage 1)
./run_pipeline.sh --sample   # Quick test with 100 repos
```

### Option 2: Step by step

```bash
# Step 1: Collect paper repositories
python viveka/fetch_papers.py \
    --output results/repo_list.json \
    --venues CVPR,NeurIPS,ICLR \
    --years 2023,2024

# Step 2: Mine experimental features (~18h for full corpus, minutes for sample)
python viveka/scale_evaluation.py \
    --input results/repo_list.json \
    --output results/phase1_incremental.json

# Step 3: Build graph + detect communities
python viveka/community_detection.py \
    --phase1 results/phase1_incremental.json \
    --threshold 6.0 \
    --output results/communities.json

# Step 4: Generate hypotheses
python viveka/hypothesis_generation.py \
    --phase1 results/phase1_incremental.json \
    --communities results/communities.json \
    --output results/hypotheses.json

# Step 5: Evaluate
python viveka/evaluate.py \
    --hypotheses results/hypotheses.json \
    --output results/evaluation.json
```

## Project Structure

```
VIVEKA/
├── viveka/                          # Core pipeline
│   ├── __init__.py
│   ├── fetch_papers.py              # Collect repos from PwC + OpenReview
│   ├── scale_evaluation.py          # Stage 1: Mine experimental features
│   ├── community_detection.py       # Stages 2-3: Graph + community detection
│   ├── hypothesis_generation.py     # Stage 4: LLM hypothesis generation
│   └── evaluate.py                  # LLM-as-judge evaluation
│
├── scripts/
│   ├── baselines/                   # Baseline comparisons (B1-B5)
│   │   ├── baseline_comparison.py   # 5-baseline evaluation
│   │   ├── controlled_baseline.py   # Same-papers controlled comparison
│   │   └── controlled_baseline_opus.py  # With stronger judge
│   ├── ablations/                   # Ablation studies
│   │   ├── llm_ablation.py          # 7 LLMs × 5 runs (Table III)
│   │   ├── threshold_ablation.py    # τ ∈ [2.0, 6.0] (Table II)
│   │   ├── edge_weight_ablation.py  # Eq. 1 component ablation (Table VII)
│   │   └── community_stability.py   # NMI across 10 seeds
│   ├── analysis/                    # Paper analyses
│   │   ├── paper_analyses.py        # Gap types, diversity, calibration
│   │   ├── inter_judge.py           # Inter-judge agreement (κ, τ)
│   │   ├── review_fixes.py          # Dual judge, novelty, weight sensitivity
│   │   ├── reviewer2_experiments.py # Embedding baseline, Jaccard null model
│   │   └── human_eval_form.py       # Human evaluation form generator
│   ├── validation/                  # Experimental hypothesis validation
│   │   ├── run_h1_hf.py             # H1: BLIP LR ablation (COCO)
│   │   ├── run_h1_full.py           # H1: Full-scale (2000 train, 500 val)
│   │   ├── run_h4_resolution.py     # H4: CLIP resolution ablation
│   │   ├── run_h8_ablation.py       # H8: RealNet on MVTec-AD
│   │   ├── run_dm_hypothesis.py     # DM: GAT+DropEdge (Cora/CiteSeer)
│   │   ├── run_dm_residual.py       # DM: Residual connections in GCN
│   │   ├── run_nlp_hypothesis2.py   # NLP: R-Drop + Focal Loss
│   │   ├── run_batch3.py            # NLP: DoLa, data quality, calibration
│   │   └── run_ts_hypothesis.py     # TS: RevIN for forecasting
│   ├── domain/
│   │   └── dm_repos.py              # DM venue repo collection
│   └── camera_ready/
│       ├── camera_ready_experiments.py  # E3-E14 reviewer experiments
│       ├── q2_uncovered.py           # Uncovered paper characterization
│       ├── q5_benchmark_coverage.py  # Benchmark vocabulary FNR
│       └── reviewer3_experiments.py  # Extraction by year, cross-τ NMI
│
├── configs/
│   └── default.yaml                 # Pipeline parameters + benchmark list
├── data/
│   ├── repo_list_sample.json        # Sample repos for testing
│   └── venue_list.json              # 28 venue definitions
├── paper/
│   ├── viveka_icdm_paper_v4.tex     # Camera-ready paper
│   └── figures/                     # Paper figures (5 PDFs)
├── results/                         # Output directory (gitignored)
│
├── community_hypothesis_v3.py       # Compatibility wrapper for scripts
├── viveka_scale_evaluation.py       # Compatibility wrapper for scripts
├── run_pipeline.sh                  # One-command full pipeline
├── setup.py                         # pip install -e .
├── requirements.txt
├── LICENSE                          # MIT
└── README.md
```

## Pipeline Details

### Stage 1: Experimental Feature Extraction

Mines five feature types per paper from code repositories:
- **Benchmarks** (B): COCO, ImageNet, CIFAR, etc. (27 canonical + extensible)
- **Methods** (M): Model architectures (ViT, ResNet, CLIP, etc.)
- **Parameters** (P): Hyperparameter names and value ranges
- **Ablations** (A): Parameters varied in controlled experiments
- **Metrics** (μ): Performance values marked with ★ for hypothesis grounding

Sources: YAML/JSON/TOML configs, argparse defaults, shell scripts, README tables.

### Stage 2: Graph Construction

Pairwise experiment similarity (Equation 1):
```
w(i,j) = 3.0·|B_i ∩ B_j| + 2.0·|M_i ∩ M_j| + 1.0·|P_i ∩ P_j|
        + 2.5·|A_i ∩ A_j| + 0.3·I[venue_i = venue_j]
```
Uses inverted-index batching: 280K candidates vs 3.2M brute-force (91% reduction).

### Stage 3: Community Detection

Louvain partitioning + coherence-based clique refinement. Communities must share at least one benchmark. At τ=6.0: 10 communities, 100% shared benchmarks, avg 25 papers.

### Stage 4: Hypothesis Generation

LLM receives community papers with ★-marked evidence. Prompt requires citing specific numbers and predicting quantitative outcomes. 5 hypotheses per community.

## Reproducing Paper Results

All commands assume `results/phase1_incremental.json` exists from Stage 1.

```bash
# Table I: Main baseline comparison
python scripts/baselines/baseline_comparison.py --phase1 results/phase1_incremental.json

# Table II: Threshold ablation
python scripts/ablations/threshold_ablation.py --phase1 results/phase1_incremental.json

# Table III: LLM ablation (7 models × 5 runs)
python scripts/ablations/llm_ablation.py --phase1 results/phase1_incremental.json

# Table VII: Edge weight component ablation
python scripts/ablations/edge_weight_ablation.py --phase1 results/phase1_incremental.json

# Section VI: Controlled baseline + judge sensitivity
python scripts/baselines/controlled_baseline.py --phase1 results/phase1_incremental.json
python scripts/baselines/controlled_baseline_opus.py --phase1 results/phase1_incremental.json

# Table IX: Experimental validation of generated hypotheses
python scripts/validation/run_h1_full.py        # H1: BLIP LR ablation
python scripts/validation/run_h4_resolution.py   # H4: CLIP resolution
python scripts/validation/run_dm_hypothesis.py   # DM: GAT+DropEdge
python scripts/validation/run_nlp_hypothesis2.py # NLP: R-Drop
python scripts/validation/run_batch3.py          # NLP: DoLa, calibration

# Camera-ready experiments (E3-E14)
python scripts/camera_ready/camera_ready_experiments.py \
    --phase1 results/phase1_incremental.json
```

## Dataset

- **28 venues**: NeurIPS, ICML, ICLR, CVPR, ECCV, ACL, EMNLP, KDD, ICDM, CIKM, + 18 more
- **12,303 repositories** from PapersWithCode + OpenReview
- **2,530 graph-ready papers** with ≥3 config files
- **40,681 structured insights** from 152,965 configuration files

The full extracted dataset (`phase1_incremental.json`, ~500MB) is not included due to size. To reproduce, run Stage 1 (~18 hours on 16 threads, ~50GB disk for cloning).

## Citation

```bibtex
@inproceedings{bhatia2026viveka,
  title={Cross-Paper Hypothesis Generation via Experiment-Similarity Graphs},
  author={Bhatia, Divyansh and Sidhaiyan, Dhiyanesh},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

We thank the six ML researchers who participated in our human evaluation study, and the anonymous ICDM reviewers whose feedback substantially improved the paper.
