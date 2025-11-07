# PV2 Quickstart Guide for 4xH200

## One-Command Setup

```bash
# Clone repo and setup everything
git clone https://github.com/fried-carrot/pv2.git
cd pv2
bash setup_h200.sh
```

This will:
1. Create directory structure (`data/`, `processed_data/`, `models/`, `results/`)
2. Download data from Google Drive (1.5GB)
3. Install dependencies (torch, scanpy, etc.)
4. Preprocess data (filter genes with min_cells=5)

## Training

### Option 1: GMVAE (scRNA-seq clustering)

Train GMVAE on 834K cells with 4xH200 (~3-5 minutes):

```bash
torchrun --nproc_per_node=4 scripts_v2/2_train_gmvae_ddp.py \
  --data_dir processed_data/filtered.h5ad \
  --output models/gmvae_model.pt \
  --epochs 30 \
  --batch_size 2048 \
  --learning_rate 5e-4 \
  --num_workers 4
```

**Output**: `models/gmvae_model.pt` (GMVAE with cell type clustering)

---

### Option 2: Multi-Modal Patient Phenotyping

Two-step process:

#### Step 1: Generate TAPE-style pseudobulk

```bash
python scripts_v2/generate_pseudobulk_tape.py \
  --input processed_data/filtered.h5ad/processed_data.h5ad \
  --output_dir processed_data/multimodal_tape \
  --n_samples_per_patient 10 \
  --n_cells_per_sample 500 \
  --label_col disease_cov
```

**Output**: 1690 synthetic samples (169 patients × 10 samples each)

#### Step 2: Train multi-modal classifier with patient-grouped CV

```bash
python scripts_v2/train_multimodal_cv.py \
  --data_dir processed_data/multimodal_tape \
  --output_dir results/cv_tape_grouped \
  --batch_size 32 \
  --epochs 50 \
  --device cuda \
  --n_folds 5
```

**Output**: 5-fold CV results with patient-grouped splits

---

## Project Structure

```
pv2/
├── data/                              # Raw data
│   └── CLUESImmVar_nonorm.V6.h5ad    # 1.5GB scRNA-seq data
├── processed_data/
│   ├── filtered.h5ad/                 # Preprocessed data (834K cells × 24K genes)
│   └── multimodal_tape/               # TAPE pseudobulk (1690 samples)
├── models/                            # Trained models
│   └── gmvae_model.pt                # GMVAE checkpoint
├── results/                           # Training results
│   └── cv_tape_grouped/              # Multi-modal CV results
└── scripts_v2/
    ├── 1_data_preprocessing.py        # Gene filtering
    ├── 2_train_gmvae.py              # Single-GPU GMVAE
    ├── 2_train_gmvae_ddp.py          # Multi-GPU GMVAE (4xH200)
    ├── generate_pseudobulk_tape.py   # TAPE pseudobulk generation
    ├── multimodal_bulk.py            # Multi-modal architecture
    └── train_multimodal_cv.py        # Patient-grouped CV training
```

---

## Key Differences from Standard Setup

1. **TAPE Pseudobulk**: Random cell sampling with Dirichlet proportions prevents memorization
2. **Patient-Grouped CV**: All 10 synthetic samples from same patient stay together
3. **DDP Training**: 4xH200 = 4x speedup via data parallelism
4. **Mixed Precision**: FP16 AMP for 2x additional speedup

---

## Expected Results

### GMVAE
- Cell type clustering accuracy: ~87%
- Training time: 3-5 minutes on 4xH200

### Multi-Modal
- ROC-AUC: 0.70-0.85 (realistic for 169 patients)
- Training time: ~30-60 minutes (5 folds × 50 epochs)

---

## Troubleshooting

**OOM errors**: Reduce `--batch_size` (current: 2048 for GMVAE, 32 for multi-modal)

**Slow training**: Check GPU utilization with `nvidia-smi`

**Data not found**: Run `bash setup_h200.sh` to download and preprocess data

**Import errors**: Install dependencies: `pip install -r requirements.txt`
