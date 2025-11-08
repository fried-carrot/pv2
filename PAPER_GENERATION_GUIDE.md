# GMVAE4P Paper Generation Guide

Complete guide for generating all paper results, tables, and figures.

## Overview

Three scripts generate everything needed for the paper:

1. **run_complete_pipeline.py** → Tables 1, 2, 3, 5
2. **run_ablation.py** → Table 4
3. **generate_paper_figures.py** → Figures 1, 2, 3

---

## Prerequisites

Install additional dependencies:

```bash
pip install umap-learn matplotlib seaborn
```

---

## Execution Order

### 1. Train GMVAE4P and Generate Main Tables

This trains GMVAE4P with 5-fold patient-grouped CV, small cohort experiments, and bulk deployment simulation.

**Command:**
```bash
python scripts_v2/run_complete_pipeline.py \
    --data_dir processed_data \
    --output_dir results/full_pipeline \
    --device cuda \
    --n_gpus 4 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed 42
```

**Expected time:** 1-2 hours on 4xH200

**Outputs:**
- `results/full_pipeline/main_metrics.csv` → **Table 1** (ROC-AUC, F1, Accuracy, ECE)
- `results/full_pipeline/small_cohort_metrics.csv` → **Table 2** (50, 100, 150, 169 patients)
- `results/full_pipeline/bulk_deployment_metrics.csv` → **Table 3** (scRNA vs bulk)
- `results/full_pipeline/compute_times.csv` → **Table 5** (wall-clock time, cost)
- `results/full_pipeline/models/fold_*_best.pth` → trained model checkpoints

---

### 2. Run Ablation Study

Trains 7 ablated configurations to quantify each component's contribution.

**Command:**
```bash
python scripts_v2/run_ablation.py \
    --data_dir processed_data \
    --output_dir results/ablation \
    --device cuda \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed 42
```

**Expected time:** 3-4 hours on single GPU (7 configs × ~30 min each)

**Output:**
- `results/ablation/ablation_metrics.csv` → **Table 4** (ROC-AUC + Δ for each config)

**Table 4 Configurations:**
1. Full GMVAE4P (baseline)
2. w/o uncertainty weighting (equal weights)
3. w/o multi-modal fusion (proportions only)
4. w/o z-score normalization (raw embeddings)
5. w/o contrastive alignment
6. w/o attention (mean pooling)

---

### 3. Generate All Paper Figures

Creates 3 main figures with 4 panels each.

**Command:**
```bash
python scripts_v2/generate_paper_figures.py \
    --model_dir results/full_pipeline/models \
    --data_dir processed_data \
    --output_dir figures \
    --device cuda \
    --fold 1
```

**Expected time:** 30-60 minutes (includes training 3 single-modality models)

**Outputs:**
- `figures/figure1_attention.pdf` → **Figure 1: Attention Analysis**
  - Panel A: Heatmap of attention weights across modalities
  - Panel B: Top-attended modalities per phenotype
  - Panel C: t-SNE of patient embeddings with decision boundary
  - Panel D: Attention entropy (correct vs incorrect predictions)

- `figures/figure2_embeddings.pdf` → **Figure 2: Embedding Visualization**
  - Panel A: UMAP before z-score normalization (dominated by cell type)
  - Panel B: UMAP after z-score normalization (disease signal emerges)
  - Panel C: UMAP of patient embeddings (colored by label)
  - Panel D: UMAP colored by max attention weight

- `figures/figure3_modalities.pdf` → **Figure 3: Modality Contributions**
  - Panel A: ROC curves for individual modalities
  - Panel B: Fusion performance vs # modalities
  - Panel C: Learned uncertainty weights distribution
  - Panel D: Correlation heatmap between modality predictions

---

## Complete One-Line Execution

Run all three scripts sequentially:

```bash
python scripts_v2/run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline --device cuda --n_gpus 4 && \
python scripts_v2/run_ablation.py --data_dir processed_data --output_dir results/ablation --device cuda && \
python scripts_v2/generate_paper_figures.py --model_dir results/full_pipeline/models --data_dir processed_data --output_dir figures --device cuda
```

**Total time:** 5-7 hours on 4xH200

---

## What You Get for the Abstract

After running Script 1 (`run_complete_pipeline.py`), check:

```bash
cat results/full_pipeline/main_metrics.csv
```

Extract these numbers for the abstract:

1. **Full dataset ROC-AUC** → Currently says "0.79-0.82" (line 37 of paper framework)
2. **Small cohort (100 patients) ROC-AUC** → Check `small_cohort_metrics.csv`
3. **Bulk deployment ROC-AUC** → Currently says "0.75-0.78" (line 37)

Update [paper/GMVAE4P_COMPLETE_FRAMEWORK.tex](paper/GMVAE4P_COMPLETE_FRAMEWORK.tex) lines 37-38 with exact values.

---

## Expected Realistic Metrics

Based on TAPE pseudobulk + patient-grouped CV + no data leakage:

- **Full dataset (169 patients):** 0.79-0.82 ROC-AUC
- **100 patients:** 0.76-0.79 ROC-AUC
- **Bulk deployment:** 0.75-0.78 ROC-AUC (95% retention)

If you get 1.00 ROC-AUC, there's still data leakage.

---

## Troubleshooting

### Issue: "UMAP not installed"
```bash
pip install umap-learn
```

### Issue: "Model checkpoint not found"
Make sure you ran Script 1 first. The models should be at:
```
results/full_pipeline/models/fold_1_best.pth
results/full_pipeline/models/fold_2_best.pth
...
```

### Issue: "Data directory not found"
Make sure you have:
```
processed_data/multimodal/pseudobulk.csv
processed_data/multimodal/proportions.csv
processed_data/multimodal/states.csv
processed_data/multimodal/communication.csv
processed_data/multimodal/labels.csv
processed_data/multimodal/patient_mapping.csv
processed_data/multimodal/metadata.json
```

Generate these with:
```bash
python scripts_v2/generate_pseudobulk_tape.py \
    --input processed_data/filtered.h5ad \
    --output processed_data/multimodal \
    --n_samples 10 \
    --n_cells 500
```

### Issue: OOM errors
Reduce batch size:
```bash
python scripts_v2/run_complete_pipeline.py ... --batch_size 8
```

---

## File Structure After Completion

```
results/
├── full_pipeline/
│   ├── main_metrics.csv              # Table 1
│   ├── small_cohort_metrics.csv      # Table 2
│   ├── bulk_deployment_metrics.csv   # Table 3
│   ├── compute_times.csv             # Table 5
│   └── models/
│       ├── fold_1_best.pth
│       ├── fold_2_best.pth
│       └── ...
├── ablation/
│   └── ablation_metrics.csv          # Table 4
└── figures/
    ├── figure1_attention.pdf         # Figure 1
    ├── figure2_embeddings.pdf        # Figure 2
    └── figure3_modalities.pdf        # Figure 3
```

---

## Next Steps for Paper Writing

1. **Update abstract** with exact ROC-AUC values from `main_metrics.csv`

2. **Insert tables** into LaTeX:
   - Table 1: Copy values from `main_metrics.csv`
   - Table 2: Copy values from `small_cohort_metrics.csv`
   - Table 3: Copy values from bulk deployment results
   - Table 4: Copy values from `ablation_metrics.csv`
   - Table 5: Copy values from `compute_times.csv`

3. **Insert figures** into LaTeX:
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=\textwidth]{figures/figure1_attention.pdf}
   \caption{Attention analysis...}
   \label{fig:attention}
   \end{figure}
   ```

4. **Write Results section** using the framework in [paper/GMVAE4P_COMPLETE_FRAMEWORK.tex](paper/GMVAE4P_COMPLETE_FRAMEWORK.tex) as guide

5. **Compile LaTeX**:
   ```bash
   cd paper
   pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
   bibtex GMVAE4P_COMPLETE_FRAMEWORK
   pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
   pdflatex GMVAE4P_COMPLETE_FRAMEWORK.tex
   ```

---

## Questions?

- Check [scripts_v2/train_multimodal_cv.py](scripts_v2/train_multimodal_cv.py) for patient-grouped CV implementation
- Check [scripts_v2/multimodal_bulk.py](scripts_v2/multimodal_bulk.py) for architecture details
- Check [paper/GMVAE4P_COMPLETE_FRAMEWORK.tex](paper/GMVAE4P_COMPLETE_FRAMEWORK.tex) for paper structure

All three scripts are fully self-contained and ready to run.
