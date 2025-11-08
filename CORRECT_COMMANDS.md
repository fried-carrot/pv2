# Corrected Commands for 2xH200 Setup

## Complete Pipeline (Start to Finish)

### STEP 1: Preprocess Data (2-5 min)

```bash
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data/filtered.h5ad \
    --min_cells 5 \
    --min_genes 200
```

---

### STEP 2: Train GMVAE with DDP (5-8 min on 2xH200)

**CORRECT ARGUMENTS:**
```bash
torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_ddp.py \
    --data_dir processed_data \
    --output models/gmvae/gmvae_model.pt \
    --epochs 30 \
    --batch_size 2048 \
    --learning_rate 5e-4 \
    --num_workers 4
```

**Key points:**
- Uses `--data_dir` (not `--input`)
- Uses `--output` (not `--output_dir`)
- `--batch_size` is PER GPU (total effective batch = 2048 × 2 = 4096)
- Creates `models/gmvae/` directory automatically

---

### STEP 3: Generate TAPE Pseudobulk (5-10 min)

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

---

### STEP 4: Train GMVAE4P (2-3 hours on 2xH200)

```bash
python scripts_v2/run_complete_pipeline.py \
    --data_dir processed_data \
    --output_dir results/full_pipeline \
    --device cuda \
    --n_gpus 2 \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4 \
    --seed 42
```

**Outputs:**
- `results/full_pipeline/main_metrics.csv` (Table 1)
- `results/full_pipeline/small_cohort_metrics.csv` (Table 2)
- `results/full_pipeline/bulk_deployment_metrics.csv` (Table 3)
- `results/full_pipeline/compute_times.csv` (Table 5)
- `results/full_pipeline/models/fold_*_best.pth` (trained models)

---

### STEP 5: Run Ablation Study (10-18 hours on 2xH200)

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

**Output:**
- `results/ablation/ablation_metrics.csv` (Table 4)

---

### STEP 6: Generate All Figures (30-60 min)

```bash
python scripts_v2/generate_paper_figures.py \
    --model_dir results/full_pipeline/models \
    --data_dir processed_data \
    --output_dir figures \
    --device cuda \
    --fold 1
```

**Outputs:**
- `figures/figure1_attention.pdf`
- `figures/figure2_embeddings.pdf`
- `figures/figure3_modalities.pdf`

---

## One-Line Complete Execution (2xH200)

```bash
python scripts_v2/1_data_preprocessing.py --input data/CLUESImmVar_nonorm.V6.h5ad --output processed_data/filtered.h5ad --min_cells 5 && \
torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_ddp.py --data_dir processed_data --output models/gmvae/gmvae_model.pt --epochs 30 --batch_size 2048 --learning_rate 5e-4 && \
python scripts_v2/generate_pseudobulk_tape.py --input processed_data/filtered.h5ad --output processed_data/multimodal --n_samples 10 --n_cells 500 && \
python scripts_v2/run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline --device cuda --n_gpus 2 --epochs 100 --batch_size 32 --lr 2e-4 && \
python scripts_v2/run_ablation.py --data_dir processed_data --output_dir results/ablation --device cuda --epochs 100 --batch_size 32 --lr 2e-4 && \
python scripts_v2/generate_paper_figures.py --model_dir results/full_pipeline/models --data_dir processed_data --output_dir figures --device cuda
```

**Total time:** ~14-22 hours on 2xH200

---

## Timeline for 2xH200

| Step | Task | Time |
|------|------|------|
| 1 | Preprocess data | 2-5 min |
| 2 | Train GMVAE (DDP) | 5-8 min |
| 3 | Generate pseudobulk | 5-10 min |
| 4 | Train GMVAE4P + tables | 2-3 hours |
| 5 | Ablation study | 10-18 hours |
| 6 | Generate figures | 30-60 min |
| **Total** | | **~14-22 hours** |

---

## Quick Reference: Argument Names

| Script | Data Input | Model Output | Other |
|--------|-----------|--------------|-------|
| `1_data_preprocessing.py` | `--input` | `--output` | `--min_cells`, `--min_genes` |
| `2_train_gmvae_ddp.py` | `--data_dir` | `--output` | `--epochs`, `--batch_size`, `--learning_rate` |
| `generate_pseudobulk_tape.py` | `--input` | `--output` | `--n_samples`, `--n_cells` |
| `run_complete_pipeline.py` | `--data_dir` | `--output_dir` | `--epochs`, `--batch_size`, `--lr`, `--n_gpus` |
| `run_ablation.py` | `--data_dir` | `--output_dir` | `--epochs`, `--batch_size`, `--lr` |
| `generate_paper_figures.py` | `--data_dir`, `--model_dir` | `--output_dir` | `--fold` |

---

## Verification After Each Step

### After Step 1 (Preprocessing)
```bash
ls processed_data/filtered.h5ad
# Should exist
```

### After Step 2 (GMVAE Training)
```bash
ls models/gmvae/gmvae_model.pt
# Should exist (~500-800 MB)
```

### After Step 3 (Pseudobulk)
```bash
ls processed_data/multimodal/
# Should see: pseudobulk.csv, proportions.csv, states.csv, communication.csv,
#            labels.csv, patient_mapping.csv, metadata.json
```

### After Step 4 (GMVAE4P)
```bash
ls results/full_pipeline/models/fold_*_best.pth
# Should see 5 files (fold_1_best.pth through fold_5_best.pth)

cat results/full_pipeline/main_metrics.csv
# Should show ROC-AUC around 0.79-0.82
```

### After Step 5 (Ablation)
```bash
cat results/ablation/ablation_metrics.csv
# Should show 6 configurations with delta_auc values
```

### After Step 6 (Figures)
```bash
ls figures/*.pdf
# Should see: figure1_attention.pdf, figure2_embeddings.pdf, figure3_modalities.pdf
```

---

## Troubleshooting

### Issue: "error: the following arguments are required: --data_dir, --output"

**Fix:** Use correct argument names:
- ❌ `--input processed_data/filtered.h5ad`
- ✅ `--data_dir processed_data`

- ❌ `--output_dir models/gmvae`
- ✅ `--output models/gmvae/gmvae_model.pt`

### Issue: OOM during GMVAE training

**Fix:** Reduce batch size per GPU:
```bash
torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_ddp.py \
    --data_dir processed_data \
    --output models/gmvae/gmvae_model.pt \
    --batch_size 1024  # Reduced from 2048
```

### Issue: OOM during GMVAE4P training

**Fix:** Reduce batch size:
```bash
python scripts_v2/run_complete_pipeline.py \
    --data_dir processed_data \
    --output_dir results/full_pipeline \
    --device cuda \
    --n_gpus 2 \
    --batch_size 16  # Reduced from 32
```

---

## Expected Output Example

### GMVAE Training Output
```
==============================================================
GMVAE DDP training
==============================================================
Loading data from: processed_data
Found 834096 cells, 24205 genes, 8 cell types
Creating DataLoader with DistributedSampler...
World size: 2, Local rank: 0
Effective batch size: 4096 (2048 per GPU × 2 GPUs)

Epoch 1/30: loss=1247.3456, recon=1198.2341, kld=49.1115, acc=0.7234
Epoch 2/30: loss=894.5678, recon=862.3421, kld=32.2257, acc=0.8012
...
Epoch 30/30: loss=312.4567, recon=289.1234, kld=23.3333, acc=0.9156

GMVAE saved to: models/gmvae/gmvae_model.pt
```

### GMVAE4P Training Output
```
Fold 1/5 Training:
------------------------------------------------------------
  Epoch   1/100: Val AUC=0.6234, Acc=0.6154, F1=0.6012, Loss=1.2345
  Epoch   2/100: Val AUC=0.6789, Acc=0.6615, F1=0.6543, Loss=0.9876
  ...
  Epoch  35/100: Val AUC=0.8012, Acc=0.7846, F1=0.7712, Loss=0.4123
  Early stopping at epoch 50
  Fold 1 Best AUC: 0.8012
```

---

## Next Steps After Completion

1. **Extract numbers for abstract:**
   ```bash
   cat results/full_pipeline/main_metrics.csv
   cat results/full_pipeline/small_cohort_metrics.csv
   ```

2. **Copy tables to LaTeX:**
   - Table 1: `main_metrics.csv` → line 314 of paper
   - Table 2: `small_cohort_metrics.csv` → line 345
   - Table 3: `bulk_deployment_metrics.csv` → line 374
   - Table 4: `ablation_metrics.csv` → line 401
   - Table 5: `compute_times.csv` → line 487

3. **Insert figures to LaTeX:**
   - Figure 1: line 430
   - Figure 2: line 448
   - Figure 3: line 465

4. **Compile paper:**
   ```bash
   cd paper
   pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
   bibtex GMVAE4P_COMPLETE_FRAMEWORK
   pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
   pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
   ```
