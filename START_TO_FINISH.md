# GMVAE4P: Complete Start-to-Finish Execution Guide

Complete step-by-step instructions from fresh instance to full paper results.

---

## Prerequisites (One-Time Setup)

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch scanpy pandas numpy scipy scikit-learn scikit-misc tqdm

# For paper figures
pip install umap-learn matplotlib seaborn
```

### 2. Download Dataset

```bash
# Create directories
mkdir -p data processed_data models results figures

# Download CLUESImmVar dataset (lupus, 834k cells, 169 patients)
pip install gdown
gdown --id 1znI4lccRallcAf7-0MLdF08TyAETHwLS -O data/CLUESImmVar_nonorm.V6.h5ad
```

---

## Step-by-Step Execution

### STEP 1: Preprocess scRNA-seq Data

**What it does:** Filters cells and genes, prepares data for GMVAE training

```bash
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data/filtered.h5ad \
    --min_cells 5 \
    --min_genes 200
```

**Time:** 2-5 minutes

**Output:** `processed_data/filtered.h5ad` (filtered dataset)

**Expected:** ~834k cells, ~24k genes after filtering

---

### STEP 2: Train GMVAE (Stage 1)

**What it does:** Pretrains GMVAE on all 834k cells (unsupervised, no labels)

```bash
python scripts_v2/2_train_gmvae_ddp.py \
    --input processed_data/filtered.h5ad \
    --output_dir models/gmvae \
    --epochs 30 \
    --batch_size 2048 \
    --learning_rate 5e-4 \
    --latent_dim 64 \
    --n_mixtures 8 \
    --device cuda
```

**For 4xH200 multi-GPU:**
```bash
torchrun --nproc_per_node=4 scripts_v2/2_train_gmvae_ddp.py \
    --input processed_data/filtered.h5ad \
    --output_dir models/gmvae \
    --epochs 30 \
    --batch_size 2048 \
    --learning_rate 5e-4 \
    --latent_dim 64 \
    --n_mixtures 8 \
    --device cuda
```

**Time:** 3-5 minutes on 4xH200, 15-20 minutes on single GPU

**Output:**
- `models/gmvae/gmvae_final.pth` (trained GMVAE)
- `models/gmvae/training_history.json` (loss curves)

**Expected final loss:** ~0.8-1.2 (reconstruction + KL)

---

### STEP 3: Generate TAPE-Style Pseudobulk

**What it does:** Creates synthetic bulk RNA-seq with patient mapping for multi-modal training

```bash
python scripts_v2/generate_pseudobulk_tape.py \
    --input processed_data/filtered.h5ad \
    --output processed_data/multimodal \
    --n_samples 10 \
    --n_cells 500 \
    --sparse_prob 0.3 \
    --patient_col patient_id \
    --celltype_col cell_type \
    --label_col phenotype
```

**Time:** 5-10 minutes

**Outputs:**
```
processed_data/multimodal/
├── pseudobulk.csv              # 1690 samples (169 patients × 10)
├── proportions.csv             # Cell type proportions (8 cell types)
├── states.csv                  # Cell state embeddings (4000 features)
├── communication.csv           # Cell-cell communication scores
├── labels.csv                  # Patient labels (SLE vs healthy)
├── patient_mapping.csv         # Maps samples to patients (CRITICAL for CV)
└── metadata.json               # Dataset info
```

**Expected:** 1690 samples from 169 patients (10 synthetic samples per patient)

---

### STEP 4: Train GMVAE4P and Generate Paper Results

**What it does:**
- Trains GMVAE4P with 5-fold patient-grouped CV
- Trains on small cohorts (50, 100, 150 patients)
- Simulates bulk deployment
- Generates Tables 1, 2, 3, 5

```bash
python scripts_v2/run_complete_pipeline.py \
    --data_dir processed_data \
    --output_dir results/full_pipeline \
    --device cuda \
    --n_gpus 4 \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4 \
    --seed 42
```

**Time:** 1.5-2.5 hours on 4xH200

**Outputs:**
```
results/full_pipeline/
├── main_metrics.csv              # Table 1: Main results
├── small_cohort_metrics.csv      # Table 2: 50/100/150/169 patients
├── bulk_deployment_metrics.csv   # Table 3: Bulk vs scRNA
├── compute_times.csv             # Table 5: Wall-clock time & cost
└── models/
    ├── fold_1_best.pth           # Best model (each fold)
    ├── fold_1_epoch_1.pth        # Every epoch checkpoint
    ├── fold_1_epoch_2.pth
    ...
    ├── fold_5_best.pth
```

**Expected results:**
- ROC-AUC: 0.79-0.82 (full dataset, 169 patients)
- ROC-AUC: 0.76-0.79 (100 patients)
- Bulk deployment: 0.75-0.78 (95% retention)
- ECE: 0.08-0.10 (well-calibrated)

**If you get 1.00 ROC-AUC:** Data leakage - check patient-grouped CV and validation settings

---

### STEP 5: Run Ablation Study

**What it does:** Trains 6 ablated GMVAE4P configs to quantify each component's contribution

```bash
python scripts_v2/run_ablation.py \
    --data_dir processed_data \
    --output_dir results/ablation \
    --device cuda \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4 \
    --seed 42
```

**Time:** 9-15 hours (6 configs × 5 folds × ~20 min)

**Output:**
```
results/ablation/
└── ablation_metrics.csv          # Table 4: Ablation results
```

**Expected deltas (vs full model):**
- w/o uncertainty weighting: -2.3pp
- w/o multi-modal fusion: -1.9pp
- w/o z-score normalization: -2.7pp
- w/o contrastive alignment: -1.3pp
- w/o attention: -1.6pp

---

### STEP 6: Generate All Paper Figures

**What it does:** Extracts embeddings, trains single-modality models, creates all 3 figures

```bash
python scripts_v2/generate_paper_figures.py \
    --model_dir results/full_pipeline/models \
    --data_dir processed_data \
    --output_dir figures \
    --device cuda \
    --fold 1
```

**Time:** 30-60 minutes

**Outputs:**
```
figures/
├── figure1_attention.pdf         # 4 panels: heatmap, bar, t-SNE, entropy
├── figure2_embeddings.pdf        # 4 panels: UMAP before/after, patient, attention
└── figure3_modalities.pdf        # 4 panels: ROC, fusion, weights, correlation
```

---

## Verify All Results Generated

```bash
# Check tables exist
ls results/full_pipeline/main_metrics.csv
ls results/full_pipeline/small_cohort_metrics.csv
ls results/full_pipeline/compute_times.csv
ls results/ablation/ablation_metrics.csv

# Check figures exist
ls figures/figure1_attention.pdf
ls figures/figure2_embeddings.pdf
ls figures/figure3_modalities.pdf

# Check models exist
ls results/full_pipeline/models/fold_*_best.pth
```

---

## Extract Numbers for Abstract

```bash
# Full dataset ROC-AUC and ECE
cat results/full_pipeline/main_metrics.csv

# Example output:
# method,roc_auc,roc_auc_std,f1_macro,f1_macro_std,accuracy,accuracy_std,ece,time_seconds
# GMVAE4P,0.802,0.015,0.784,0.012,0.808,0.011,0.087,5421.3

# 100-patient ROC-AUC
cat results/full_pipeline/small_cohort_metrics.csv | grep "100"

# Example output:
# 100,0.768,0.018
```

**Update abstract** ([paper/GMVAE4P_COMPLETE_FRAMEWORK.tex](paper/GMVAE4P_COMPLETE_FRAMEWORK.tex) line 37):

```latex
% OLD (placeholder)
GMVAE4P achieves 0.79-0.82 ROC-AUC using 100 patients

% NEW (with exact values from above)
GMVAE4P achieves 0.768 ROC-AUC using 100 patients
```

---

## Copy Results to Paper

### Table 1: Main Results (Line 314)

```bash
cat results/full_pipeline/main_metrics.csv
```

Copy `roc_auc`, `f1_macro`, `accuracy`, `ece` to LaTeX table.

### Table 2: Small Cohort (Line 345)

```bash
cat results/full_pipeline/small_cohort_metrics.csv
```

Copy ROC-AUC values for 50, 100, 150, 169 patients.

### Table 3: Bulk Deployment (Line 374)

```bash
cat results/full_pipeline/bulk_deployment_metrics.csv
```

Copy bulk ROC-AUC and compute retention %.

### Table 4: Ablation (Line 401)

```bash
cat results/ablation/ablation_metrics.csv
```

Copy all configurations with Δ AUC values.

### Table 5: Compute Cost (Line 487)

```bash
cat results/full_pipeline/compute_times.csv
```

Copy wall-clock time and cost.

### Figures (Lines 430, 448, 465)

Insert figures in LaTeX:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{../figures/figure1_attention.pdf}
\caption{Attention analysis...}
\label{fig:attention}
\end{figure}
```

---

## Compile Paper

```bash
cd paper
pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
bibtex GMVAE4P_COMPLETE_FRAMEWORK
pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
```

**Output:** `GMVAE4P_COMPLETE_FRAMEWORK.pdf` (complete 10-page paper)

---

## Timeline Summary

| Step | Task | Time |
|------|------|------|
| 1 | Preprocess data | 2-5 min |
| 2 | Train GMVAE | 3-5 min (4xH200) |
| 3 | Generate pseudobulk | 5-10 min |
| 4 | Train GMVAE4P + tables | 1.5-2.5 hours |
| 5 | Ablation study | 9-15 hours |
| 6 | Generate figures | 30-60 min |
| **Total** | | **~12-18 hours** |

---

## One-Line Execution (All Steps)

```bash
# Step 1: Preprocess
python scripts_v2/1_data_preprocessing.py --input data/CLUESImmVar_nonorm.V6.h5ad --output processed_data/filtered.h5ad --min_cells 5 && \

# Step 2: Train GMVAE
torchrun --nproc_per_node=4 scripts_v2/2_train_gmvae_ddp.py --input processed_data/filtered.h5ad --output_dir models/gmvae --epochs 30 --batch_size 2048 && \

# Step 3: Generate pseudobulk
python scripts_v2/generate_pseudobulk_tape.py --input processed_data/filtered.h5ad --output processed_data/multimodal --n_samples 10 --n_cells 500 && \

# Step 4: Train GMVAE4P
python scripts_v2/run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline --device cuda --n_gpus 4 && \

# Step 5: Ablation
python scripts_v2/run_ablation.py --data_dir processed_data --output_dir results/ablation --device cuda && \

# Step 6: Figures
python scripts_v2/generate_paper_figures.py --model_dir results/full_pipeline/models --data_dir processed_data --output_dir figures --device cuda
```

Run this single command and wait ~12-18 hours. Everything will be ready.

---

## Troubleshooting

### Issue: ROC-AUC = 1.00 (unrealistic)

**Fix:** Data leakage. Check:
1. Validation uses `true_proportions=None, true_states=None, true_communication=None`
2. Patient-grouped CV keeps all 10 samples from same patient together
3. No bulk encoder in architecture

**Files to check:**
- [scripts_v2/train_multimodal_cv.py](scripts_v2/train_multimodal_cv.py) lines 126-133, 299-344
- [scripts_v2/multimodal_bulk.py](scripts_v2/multimodal_bulk.py) lines 192-198

### Issue: UMAP import error

```bash
pip install umap-learn matplotlib seaborn
```

### Issue: OOM errors

```bash
# Reduce batch size
python scripts_v2/run_complete_pipeline.py ... --batch_size 16
```

### Issue: Missing patient_mapping.csv

```bash
# Regenerate pseudobulk
python scripts_v2/generate_pseudobulk_tape.py --input processed_data/filtered.h5ad --output processed_data/multimodal --n_samples 10 --n_cells 500
```

---

## Critical Files

| File | Purpose |
|------|---------|
| `scripts_v2/1_data_preprocessing.py` | Filter cells/genes |
| `scripts_v2/2_train_gmvae_ddp.py` | GMVAE pretraining (Stage 1) |
| `scripts_v2/generate_pseudobulk_tape.py` | Create TAPE pseudobulk + patient mapping |
| `scripts_v2/run_complete_pipeline.py` | Train GMVAE4P, generate Tables 1-5 |
| `scripts_v2/run_ablation.py` | Generate Table 4 |
| `scripts_v2/generate_paper_figures.py` | Generate Figures 1-3 |
| `processed_data/multimodal/patient_mapping.csv` | **CRITICAL:** Maps samples to patients for CV |
| `paper/GMVAE4P_COMPLETE_FRAMEWORK.tex` | Paper template with all content |

---

## Final Checklist

- [ ] Step 1: Data preprocessed (`filtered.h5ad` exists)
- [ ] Step 2: GMVAE trained (`models/gmvae/gmvae_final.pth` exists)
- [ ] Step 3: Pseudobulk generated (`patient_mapping.csv` exists)
- [ ] Step 4: GMVAE4P trained (5 `fold_*_best.pth` files exist)
- [ ] Step 4: Tables 1, 2, 3, 5 generated (4 CSV files)
- [ ] Step 5: Ablation complete (`ablation_metrics.csv` exists)
- [ ] Step 6: All 3 figures generated (3 PDF files)
- [ ] Abstract updated with exact ROC-AUC values
- [ ] All tables copied to LaTeX
- [ ] All figures inserted to LaTeX
- [ ] Paper compiled (`GMVAE4P_COMPLETE_FRAMEWORK.pdf` exists)

Done. Ready for submission.
