# GMVAE4P Cloud Setup and Execution Guide

Complete guide to run uncertainty-aware multi-modal bulk RNA-seq classification on cloud.

## Prerequisites

- Cloud GPU instance (H100 80GB recommended, or A100/V100)
- Ubuntu 20.04+
- Python 3.9+
- CUDA 11.8+

## Step 1: Clone Repository

```bash
git clone https://github.com/fried-carrot/pv2.git
cd pv2
```

## Step 2: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scanpy anndata pandas numpy scipy scikit-learn
pip install matplotlib seaborn tqdm
```

## Step 3: Upload Data

Upload your lupus dataset to the server:

```bash
# Create data directory
mkdir -p data

# Upload h5ad file (use scp, rsync, or cloud storage)
scp local/path/to/CLUESImmVar_nonorm.V6.h5ad user@server:~/pv2/data/

# Or download from cloud storage
# gsutil cp gs://your-bucket/lupus.h5ad data/
# aws s3 cp s3://your-bucket/lupus.h5ad data/
```

## Step 4: Data Preprocessing

```bash
# Create output directories
mkdir -p processed_data/gmvae
mkdir -p processed_data/multimodal
mkdir -p models

# Run preprocessing
python scripts_v2/1_data_preprocessing.py \
  --input data/CLUESImmVar_nonorm.V6.h5ad \
  --output processed_data/gmvae \
  --min_cells 5

# Expected output: matrix.mtx, labels.csv, genes.txt, processed_data.h5ad
```

## Step 5: Generate Pseudobulk + Ground Truth Modalities

```bash
python scripts_v2/generate_pseudobulk.py \
  --input processed_data/gmvae/processed_data.h5ad \
  --output processed_data/multimodal \
  --patient_col ind_cov \
  --ct_col ct_cov \
  --label_col disease_cov \
  --state_dim 64 \
  --n_hvg 500

# Expected output:
#   pseudobulk.csv (patients x genes)
#   proportions.csv (patients x cell_types)
#   states.csv (patients x cell_type_states)
#   communication.csv (patients x interactions)
#   labels.csv (patient labels)
#   metadata.json
```

## Step 6: Train Multi-Modal Model

```bash
python scripts_v2/train_multimodal.py \
  --data_dir processed_data/multimodal \
  --output_dir models/multimodal \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --embedding_dim 128 \
  --state_dim 64 \
  --test_size 0.2 \
  --device cuda

# Expected output:
#   best_model.pth
#   history.json
#   summary.json
```

## Step 7: Monitor Training (Optional)

In a separate terminal, monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Monitor training logs:

```bash
tail -f nohup.out  # if running in background
```

## Step 8: Verify Results

```bash
# Check model outputs
ls -lh models/multimodal/

# View training summary
cat models/multimodal/summary.json

# View training history
python -c "import json; print(json.dumps(json.load(open('models/multimodal/history.json')), indent=2))"
```

## Running in Background (Recommended)

For long-running training:

```bash
# Using nohup
nohup python scripts_v2/train_multimodal.py \
  --data_dir processed_data/multimodal \
  --output_dir models/multimodal \
  --batch_size 32 \
  --epochs 50 \
  --device cuda > training.log 2>&1 &

# Or using tmux (recommended)
tmux new -s training
python scripts_v2/train_multimodal.py ...
# Press Ctrl+B then D to detach
# tmux attach -t training  # to reattach
```

## Downloading Results

```bash
# From server to local
scp -r user@server:~/pv2/models/multimodal ./local_results/
scp user@server:~/pv2/training.log ./local_results/
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts_v2/train_multimodal.py --batch_size 16 ...

# Or use gradient accumulation (requires code modification)
```

### CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Data Loading Errors
```bash
# Check file paths
ls -lh processed_data/multimodal/

# Verify data format
python -c "import pandas as pd; print(pd.read_csv('processed_data/multimodal/pseudobulk.csv', nrows=5))"
```

## Complete Workflow Script

Save as `run_all.sh`:

```bash
#!/bin/bash
set -e

echo "=== GMVAE4P Multi-Modal Training Pipeline ==="

# 1. Preprocessing
echo "Step 1: Preprocessing..."
python scripts_v2/1_data_preprocessing.py \
  --input data/CLUESImmVar_nonorm.V6.h5ad \
  --output processed_data/gmvae \
  --min_cells 5

# 2. Generate pseudobulk
echo "Step 2: Generating pseudobulk..."
python scripts_v2/generate_pseudobulk.py \
  --input processed_data/gmvae/processed_data.h5ad \
  --output processed_data/multimodal \
  --state_dim 64 \
  --n_hvg 500

# 3. Train model
echo "Step 3: Training multi-modal model..."
python scripts_v2/train_multimodal.py \
  --data_dir processed_data/multimodal \
  --output_dir models/multimodal \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --device cuda

echo "=== Training complete! ==="
echo "Results saved to: models/multimodal/"
```

Run with:
```bash
chmod +x run_all.sh
./run_all.sh
```

## Expected Timeline

- Preprocessing: 5-10 minutes (depends on dataset size)
- Pseudobulk generation: 2-5 minutes
- Training: 1-2 hours (50 epochs, ~2-3 min/epoch)

## Next Steps

After training completes:
1. Evaluate on test set
2. Compare with baselines
3. Analyze uncertainty predictions
4. Test deployment on bulk RNA-seq
