# GMVAE4P Paper Quick Reference

## One-Command Execution

```bash
# Generate everything (Tables 1-5, Figures 1-3)
python scripts_v2/run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline --device cuda --n_gpus 4 && \
python scripts_v2/run_ablation.py --data_dir processed_data --output_dir results/ablation --device cuda && \
python scripts_v2/generate_paper_figures.py --model_dir results/full_pipeline/models --data_dir processed_data --output_dir figures --device cuda
```

Total time: 5-7 hours on 4xH200

---

## What Gets Generated

### Tables (CSV files)

| Table | File | Content |
|-------|------|---------|
| **Table 1** | `results/full_pipeline/main_metrics.csv` | Main results (ROC-AUC, F1, Accuracy, ECE) |
| **Table 2** | `results/full_pipeline/small_cohort_metrics.csv` | Small cohort (50, 100, 150, 169 patients) |
| **Table 3** | `results/full_pipeline/bulk_deployment_metrics.csv` | Bulk deployment vs scRNA-seq |
| **Table 4** | `results/ablation/ablation_metrics.csv` | Ablation study (7 configs + Δ AUC) |
| **Table 5** | `results/full_pipeline/compute_times.csv` | Wall-clock time & cost |

### Figures (PDF files)

| Figure | File | Panels |
|--------|------|--------|
| **Figure 1** | `figures/figure1_attention.pdf` | A: Attention heatmap<br>B: Top modalities<br>C: t-SNE<br>D: Entropy |
| **Figure 2** | `figures/figure2_embeddings.pdf` | A: UMAP before norm<br>B: UMAP after norm<br>C: Patient UMAP<br>D: Attention UMAP |
| **Figure 3** | `figures/figure3_modalities.pdf` | A: ROC curves<br>B: Fusion performance<br>C: Uncertainty weights<br>D: Correlation |

---

## For the Abstract

After running `run_complete_pipeline.py`, get these numbers:

```bash
# Main ROC-AUC
cat results/full_pipeline/main_metrics.csv | grep "roc_auc"

# Small cohort ROC-AUC (100 patients)
cat results/full_pipeline/small_cohort_metrics.csv | grep "100"

# Bulk deployment retention
cat results/full_pipeline/bulk_deployment_metrics.csv
```

Update abstract (lines 36-37 in [paper/GMVAE4P_COMPLETE_FRAMEWORK.tex](paper/GMVAE4P_COMPLETE_FRAMEWORK.tex)):
- "0.79-0.82 ROC-AUC using 100 patients" → use exact value
- "0.75-0.78 ROC-AUC" for bulk → use exact value

---

## Expected Realistic Values

| Metric | Expected Range | Current Abstract |
|--------|----------------|------------------|
| Full dataset (169 pts) | 0.79-0.82 | 0.79-0.82 ✓ |
| 100 patients | 0.76-0.79 | 0.79-0.82 (update) |
| Bulk deployment | 0.75-0.78 | 0.75-0.78 ✓ |
| ECE | 0.08-0.10 | 0.087 (update) |

**If you get 1.00 ROC-AUC, there's data leakage.**

---

## Critical Implementation Details

### Patient-Grouped CV
All 10 synthetic samples from same patient stay together in train/test splits.

Implementation: [scripts_v2/train_multimodal_cv.py](scripts_v2/train_multimodal_cv.py) lines 299-344

### No Data Leakage in Validation
Validation passes `true_proportions=None, true_states=None, true_communication=None`.

Implementation: [scripts_v2/train_multimodal_cv.py](scripts_v2/train_multimodal_cv.py) lines 126-133

### No Bulk Encoder
Model only uses 3 predicted modalities (proportions, states, communication).

Implementation: [scripts_v2/multimodal_bulk.py](scripts_v2/multimodal_bulk.py) lines 192-198 (no `self.encode_bulk`)

---

## File Locations

### Input Data
```
processed_data/multimodal/
├── pseudobulk.csv
├── proportions.csv
├── states.csv
├── communication.csv
├── labels.csv
├── patient_mapping.csv
└── metadata.json
```

### Scripts
```
scripts_v2/
├── run_complete_pipeline.py      # Tables 1, 2, 3, 5
├── run_ablation.py               # Table 4
└── generate_paper_figures.py     # Figures 1, 2, 3
```

### Outputs
```
results/full_pipeline/            # Tables 1, 2, 3, 5 + models
results/ablation/                 # Table 4
figures/                          # Figures 1, 2, 3
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| UMAP not installed | `pip install umap-learn` |
| Model checkpoint not found | Run `run_complete_pipeline.py` first |
| Data directory not found | Run `generate_pseudobulk_tape.py` first |
| OOM errors | Reduce `--batch_size 8` |
| 1.00 ROC-AUC (unrealistic) | Check validation uses `None` for ground truth modalities |

---

## Complete Paper Checklist

- [ ] Run `run_complete_pipeline.py` → get Tables 1, 2, 3, 5
- [ ] Run `run_ablation.py` → get Table 4
- [ ] Run `generate_paper_figures.py` → get Figures 1, 2, 3
- [ ] Update abstract with exact ROC-AUC from `main_metrics.csv`
- [ ] Update ECE with exact value from `main_metrics.csv`
- [ ] Insert all 5 tables into LaTeX
- [ ] Insert all 3 figures into LaTeX
- [ ] Write Results section using framework as guide
- [ ] Compile LaTeX: `pdflatex → bibtex → pdflatex → pdflatex`

---

## Time Estimates (4xH200)

| Task | Time | Cost (@$6/hr) |
|------|------|---------------|
| GMVAE4P training (5-fold) | 1-2 hours | $6-12 |
| Ablation study (7 configs) | 3-4 hours | $18-24 |
| Figure generation | 0.5-1 hour | $3-6 |
| **Total** | **5-7 hours** | **$30-42** |

All within clinical deployment constraints (<4 hours per method).
