#!/bin/bash
# Download lupus dataset from Google Drive

set -e

echo "=============================================================================="
echo "Downloading Lupus Dataset from Google Drive"
echo "=============================================================================="

# Create data directory
mkdir -p data

# Install gdown if not already installed
pip install -q gdown

# Download dataset (1.5GB)
echo "Downloading CLUESImmVar_nonorm.V6.h5ad (1.5GB)..."
gdown --id 1znI4lccRallcAf7-0MLdF08TyAETHwLS -O data/CLUESImmVar_nonorm.V6.h5ad

# Verify download
if [ -f "data/CLUESImmVar_nonorm.V6.h5ad" ]; then
    echo ""
    echo "=============================================================================="
    echo "Download Complete!"
    echo "=============================================================================="
    ls -lh data/CLUESImmVar_nonorm.V6.h5ad
    echo ""
    echo "You can now run the training pipeline with:"
    echo "  ./run_all.sh"
    echo "=============================================================================="
else
    echo "ERROR: Download failed"
    exit 1
fi
