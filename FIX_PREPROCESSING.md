# Fix Preprocessing Issue

## Problem

The preprocessing script expects `--output` to be a **directory**, not a file path.

You ran:
```bash
# WRONG
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data/filtered.h5ad  # ❌ This is a FILE path
```

This created `processed_data/filtered.h5ad/` as a directory, not the files the GMVAE script expects.

## Solution

Use `--output` as a **directory** path:

```bash
# CORRECT
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data \
    --min_cells 5
```

## What This Creates

```
processed_data/
├── matrix.mtx           # Sparse gene × cell matrix
├── labels.csv           # Cell type labels + patient IDs
├── genes.txt            # Gene names
├── processed_data.h5ad  # Processed h5ad file
└── metadata.json        # Dataset metadata (needed by GMVAE script)
```

## Complete Corrected Workflow

```bash
# Step 1: Preprocessing (creates directory structure)
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data \
    --min_cells 5

# Step 2: Train GMVAE with DDP
torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_ddp.py \
    --data_dir processed_data \
    --output models/gmvae/gmvae_model.pt \
    --epochs 30 \
    --batch_size 2048 \
    --learning_rate 5e-4

# Step 3: Generate pseudobulk (uses processed_data.h5ad)
python scripts_v2/generate_pseudobulk_tape.py \
    --input processed_data/processed_data.h5ad \
    --output processed_data/multimodal \
    --n_samples 10 \
    --n_cells 500

# Continue with remaining steps...
```

## Quick Fix if You Already Ran Wrong Command

If you already created `processed_data/filtered.h5ad/`, remove it and re-run:

```bash
# Remove incorrectly created directory
rm -rf processed_data/filtered.h5ad

# Run preprocessing correctly
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data \
    --min_cells 5
```

## Verify Preprocessing Succeeded

```bash
ls processed_data/

# Should see:
# matrix.mtx
# labels.csv
# genes.txt
# processed_data.h5ad
# metadata.json

# Check metadata
cat processed_data/metadata.json

# Should show:
# {
#   "n_cells": 834096,
#   "n_genes": 24205,
#   "n_cell_types": 8,
#   ...
# }
```

## Complete One-Line Command (2xH200) - CORRECTED

```bash
python scripts_v2/1_data_preprocessing.py --input data/CLUESImmVar_nonorm.V6.h5ad --output processed_data --min_cells 5 && \
torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_ddp.py --data_dir processed_data --output models/gmvae/gmvae_model.pt --epochs 30 --batch_size 2048 --learning_rate 5e-4 && \
python scripts_v2/generate_pseudobulk_tape.py --input processed_data/processed_data.h5ad --output processed_data/multimodal --n_samples 10 --n_cells 500 && \
python scripts_v2/run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline --device cuda --n_gpus 2 && \
python scripts_v2/run_ablation.py --data_dir processed_data --output_dir results/ablation --device cuda && \
python scripts_v2/generate_paper_figures.py --model_dir results/full_pipeline/models --data_dir processed_data --output_dir figures --device cuda
```
