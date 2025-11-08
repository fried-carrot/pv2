#!/bin/bash
#
# Master script to run all GMVAE4P paper experiments
#
# Usage: ./run_all_experiments.sh
#
# Generates all results for Tables 1-5 and Figures 1-3
#

set -e  # Exit on error

DATA_DIR="processed_data"
DEVICE="cuda"

echo "========================================="
echo "GMVAE4P PAPER - ALL EXPERIMENTS"
echo "========================================="
echo ""
echo "This will run:"
echo "  1. Baseline methods (Table 1 comparison)"
echo "  2. Full pipeline (Tables 1, 2, 3, 5)"
echo "  3. Ablation study (Table 4)"
echo "  4. Figure generation (Figures 1-3)"
echo ""
echo "Expected runtime: 14-24 hours on 2xH200"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Create results directories
mkdir -p results/baselines
mkdir -p results/full_pipeline/models
mkdir -p results/ablation
mkdir -p results/figures

echo ""
echo "========================================="
echo "STEP 1: Training Baselines"
echo "========================================="
python scripts_v2/train_baselines.py \
    --data_dir $DATA_DIR \
    --output_dir results/baselines \
    --seed 42

echo ""
echo "========================================="
echo "STEP 2: Full Pipeline (Tables 1,2,3,5)"
echo "========================================="
python scripts_v2/run_complete_pipeline.py \
    --data_dir $DATA_DIR \
    --output_dir results/full_pipeline \
    --device $DEVICE

echo ""
echo "========================================="
echo "STEP 3: Ablation Study (Table 4)"
echo "========================================="
python scripts_v2/run_ablation.py \
    --data_dir $DATA_DIR \
    --output_dir results/ablation \
    --device $DEVICE \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4

echo ""
echo "========================================="
echo "STEP 4: Generate Figures"
echo "========================================="
python scripts_v2/generate_paper_figures.py \
    --data_dir $DATA_DIR \
    --model_dir results/full_pipeline \
    --output_dir results/figures

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - results/baselines/baseline_results.csv (Table 1 baselines)"
echo "  - results/full_pipeline/results.csv (Tables 1,2,3,5)"
echo "  - results/ablation/ablation_metrics.csv (Table 4)"
echo "  - results/figures/*.pdf (Figures 1-3)"
echo ""
echo "========================================="
