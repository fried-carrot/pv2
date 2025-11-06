#!/bin/bash
set -e

echo "=============================================================================="
echo "GMVAE4P Multi-Modal Training Pipeline"
echo "=============================================================================="

# Configuration
DATA_FILE="data/CLUESImmVar_nonorm.V6.h5ad"
PREPROCESS_DIR="processed_data/gmvae"
MULTIMODAL_DIR="processed_data/multimodal"
MODEL_DIR="models/multimodal"
DEVICE="cuda"
BATCH_SIZE=32
EPOCHS=50
LR=1e-4

# Create directories
mkdir -p processed_data models

echo ""
echo "Step 1/3: Preprocessing scRNA-seq data"
echo "=============================================================================="
python scripts_v2/1_data_preprocessing.py \
  --input "$DATA_FILE" \
  --output "$PREPROCESS_DIR" \
  --min_cells 5

echo ""
echo "Step 2/3: Generating pseudobulk and ground truth modalities"
echo "=============================================================================="
python scripts_v2/generate_pseudobulk.py \
  --input "$PREPROCESS_DIR/processed_data.h5ad" \
  --output "$MULTIMODAL_DIR" \
  --patient_col ind_cov \
  --ct_col ct_cov \
  --label_col disease_cov \
  --state_dim 64 \
  --n_hvg 500

echo ""
echo "Step 3/3: Training multi-modal classifier"
echo "=============================================================================="
python scripts_v2/train_multimodal.py \
  --data_dir "$MULTIMODAL_DIR" \
  --output_dir "$MODEL_DIR" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --embedding_dim 128 \
  --state_dim 64 \
  --test_size 0.2 \
  --device "$DEVICE"

echo ""
echo "=============================================================================="
echo "TRAINING COMPLETE"
echo "=============================================================================="
echo "Results saved to: $MODEL_DIR"
echo ""
echo "Generated files:"
ls -lh "$MODEL_DIR"
echo ""
echo "View summary:"
echo "  cat $MODEL_DIR/summary.json"
echo ""
echo "View training history:"
echo "  cat $MODEL_DIR/history.json"
