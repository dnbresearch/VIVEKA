#!/usr/bin/env bash
# =============================================================
# VIVEKA — Full Pipeline Run
# =============================================================
# Usage:
#   export ANTHROPIC_API_KEY=your_key
#   ./run_pipeline.sh [--sample]       # --sample for quick test
# =============================================================

set -e

SAMPLE_MODE=false
if [ "$1" == "--sample" ]; then
    SAMPLE_MODE=true
    echo "Running in SAMPLE mode (100 repos)"
fi

echo "=============================================="
echo "VIVEKA: Cross-Paper Hypothesis Generation"
echo "=============================================="

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Set ANTHROPIC_API_KEY environment variable"
    exit 1
fi

# Create output directories
mkdir -p results

# ---- Stage 0: Fetch papers ----
echo ""
echo "[Stage 0] Fetching paper repositories..."
if [ "$SAMPLE_MODE" = true ]; then
    cp data/repo_list_sample.json results/repo_list.json
    echo "  Using sample data (100 repos)"
else
    python viveka/fetch_papers.py \
        --output results/repo_list.json \
        --venue-file data/venue_list.json
fi

# ---- Stage 1: Mine experimental features ----
echo ""
echo "[Stage 1] Mining experimental features from repositories..."
python viveka/scale_evaluation.py \
    --input results/repo_list.json \
    --output results/phase1_incremental.json

# ---- Stages 2-3: Build graph + detect communities ----
echo ""
echo "[Stages 2-3] Building graph and detecting communities..."
python viveka/community_detection.py \
    --phase1 results/phase1_incremental.json \
    --threshold 6.0 \
    --output results/communities.json

# ---- Stage 4: Generate hypotheses ----
echo ""
echo "[Stage 4] Generating cross-paper hypotheses..."
python viveka/hypothesis_generation.py \
    --phase1 results/phase1_incremental.json \
    --communities results/communities.json \
    --output results/hypotheses.json

# ---- Evaluate ----
echo ""
echo "[Evaluate] Scoring hypotheses..."
python viveka/evaluate.py \
    --hypotheses results/hypotheses.json \
    --output results/evaluation.json

echo ""
echo "=============================================="
echo "Pipeline complete! Results in results/"
echo "=============================================="
ls -la results/
