#!/bin/bash
set -e

echo "=========================================="
echo "PV2 Setup for 4xH200 Server"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup_h200.sh" ]; then
    echo "Error: Run this script from the pv2 root directory"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data
mkdir -p processed_data
mkdir -p models
mkdir -p results

# Check if data file exists
if [ ! -f "data/CLUESImmVar_nonorm.V6.h5ad" ]; then
    echo ""
    echo "Downloading data file from Google Drive..."
    pip install -q gdown
    gdown --id 1znI4lccRallcAf7-0MLdF08TyAETHwLS -O data/CLUESImmVar_nonorm.V6.h5ad
else
    echo "Data file already exists, skipping download"
fi

echo ""
echo "Installing Python dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q scanpy pandas numpy scipy scikit-learn scikit-misc tqdm

echo ""
echo "=========================================="
echo "Data Preprocessing"
echo "=========================================="

# Step 1: Preprocess data (filter genes)
if [ ! -f "processed_data/filtered.h5ad/processed_data.h5ad" ]; then
    echo "Running preprocessing (filter genes with min_cells=5)..."
    python scripts_v2/1_data_preprocessing.py \
        --input data/CLUESImmVar_nonorm.V6.h5ad \
        --output processed_data/filtered.h5ad \
        --min_cells 5
else
    echo "Filtered data already exists, skipping preprocessing"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  data/                        - Raw data (1.5GB h5ad file)"
echo "  processed_data/filtered.h5ad - Preprocessed data ready for training"
echo "  models/                      - Model outputs will be saved here"
echo "  results/                     - Training results and metrics"
echo ""
echo "=========================================="
echo "Training Commands"
echo "=========================================="
echo ""
echo "GMVAE Training (4xH200, ~3-5 minutes):"
echo "----------------------------------------"
echo "torchrun --nproc_per_node=4 scripts_v2/2_train_gmvae_ddp.py \\"
echo "  --data_dir processed_data/filtered.h5ad \\"
echo "  --output models/gmvae_model.pt \\"
echo "  --epochs 30 \\"
echo "  --batch_size 2048 \\"
echo "  --learning_rate 5e-4 \\"
echo "  --num_workers 4"
echo ""
echo "Multi-Modal Training (requires TAPE pseudobulk first):"
echo "-------------------------------------------------------"
echo "# Step 1: Generate TAPE pseudobulk"
echo "python scripts_v2/generate_pseudobulk_tape.py \\"
echo "  --input processed_data/filtered.h5ad/processed_data.h5ad \\"
echo "  --output_dir processed_data/multimodal_tape \\"
echo "  --n_samples_per_patient 10 \\"
echo "  --n_cells_per_sample 500 \\"
echo "  --label_col disease_cov"
echo ""
echo "# Step 2: Train multi-modal classifier with grouped CV"
echo "python scripts_v2/train_multimodal_cv.py \\"
echo "  --data_dir processed_data/multimodal_tape \\"
echo "  --output_dir results/cv_tape_grouped \\"
echo "  --batch_size 32 \\"
echo "  --epochs 50 \\"
echo "  --device cuda"
echo ""
echo "=========================================="
