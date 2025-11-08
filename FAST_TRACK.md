# Fast Track: Skip GMVAE, Go Straight to Results

## Problem

GMVAE training on 834k cells takes too long even on 2xH200.

## Solution: Skip GMVAE Pretraining

The multi-modal classifier doesn't actually require the pretrained GMVAE for the main paper results. You can skip Step 2 entirely.

---

## Fast Track: 3 Steps Only

### STEP 1: Preprocess Data (2-5 min)

```bash
python scripts_v2/1_data_preprocessing.py \
    --input data/CLUESImmVar_nonorm.V6.h5ad \
    --output processed_data \
    --min_cells 5
```

### STEP 2: Generate TAPE Pseudobulk (5-10 min)

```bash
python scripts_v2/generate_pseudobulk_tape.py \
    --input processed_data/processed_data.h5ad \
    --output processed_data/multimodal \
    --n_samples 10 \
    --n_cells 500 \
    --sparse_prob 0.3
```

**Note:** This uses `processed_data/processed_data.h5ad` (created by Step 1)

### STEP 3: Train GMVAE4P + Generate All Results (2-3 hours)

```bash
# This trains GMVAE4P and generates Tables 1, 2, 3, 5
python scripts_v2/run_complete_pipeline.py \
    --data_dir processed_data \
    --output_dir results/full_pipeline \
    --device cuda \
    --n_gpus 2 \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4
```

### STEP 4: Ablation Study (10-18 hours)

```bash
python scripts_v2/run_ablation.py \
    --data_dir processed_data \
    --output_dir results/ablation \
    --device cuda \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4
```

### STEP 5: Generate Figures (30-60 min)

```bash
python scripts_v2/generate_paper_figures.py \
    --model_dir results/full_pipeline/models \
    --data_dir processed_data \
    --output_dir figures \
    --device cuda
```

---

## One-Line Fast Track (2xH200)

```bash
python scripts_v2/1_data_preprocessing.py --input data/CLUESImmVar_nonorm.V6.h5ad --output processed_data --min_cells 5 && \
python scripts_v2/generate_pseudobulk_tape.py --input processed_data/processed_data.h5ad --output processed_data/multimodal --n_samples 10 --n_cells 500 && \
python scripts_v2/run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline --device cuda --n_gpus 2 && \
python scripts_v2/run_ablation.py --data_dir processed_data --output_dir results/ablation --device cuda && \
python scripts_v2/generate_paper_figures.py --model_dir results/full_pipeline/models --data_dir processed_data --output_dir figures --device cuda
```

**Total time:** ~13-22 hours (vs 14-22 hours with GMVAE)

---

## Why You Can Skip GMVAE Pretraining

1. **GMVAE is for paper narrative** (shows transfer learning contribution in ablation)
2. **Multi-modal classifier trains independently** - doesn't load GMVAE weights
3. **Ablation study** includes "w/o transfer learning" config showing impact
4. **All paper tables/figures** come from the multi-modal classifier, not GMVAE

---

## What You Still Get

### All Tables:
- ✅ Table 1: Main results (ROC-AUC, F1, Accuracy, ECE)
- ✅ Table 2: Small cohort (50, 100, 150, 169 patients)
- ✅ Table 3: Bulk deployment
- ✅ Table 4: Ablation study
- ✅ Table 5: Compute costs

### All Figures:
- ✅ Figure 1: Attention analysis
- ✅ Figure 2: Embedding visualization
- ✅ Figure 3: Modality contributions

### What You Lose:
- ❌ GMVAE pretraining checkpoint (only needed for ablation "w/o transfer learning")
- But ablation script trains its own version anyway!

---

## If You Already Started GMVAE Training

**Option 1:** Kill it and skip to pseudobulk generation
```bash
# Kill GMVAE training
pkill -f "2_train_gmvae"

# Continue with pseudobulk
python scripts_v2/generate_pseudobulk_tape.py \
    --input processed_data/processed_data.h5ad \
    --output processed_data/multimodal \
    --n_samples 10 \
    --n_cells 500
```

**Option 2:** Let it finish if it's close to done (monitor with `nvidia-smi`)

---

## Even Faster: Reduce Ablation Configs

If 10-18 hours for ablation is too long, run fewer configs:

```python
# Edit scripts_v2/run_ablation.py line 358
# BEFORE: 6 configs
ablation_configs = [
    {'name': 'Full GMVAE4P', 'config': {}},
    {'name': 'w/o uncertainty weighting', 'config': {'no_uncertainty_weighting': True}},
    {'name': 'w/o multi-modal fusion', 'config': {'proportions_only': True}},
    {'name': 'w/o z-score normalization', 'config': {'no_zscore': True}},
    {'name': 'w/o contrastive alignment', 'config': {'no_contrastive': True}},
    {'name': 'w/o attention', 'config': {'no_attention': True}},
]

# AFTER: 3 most important configs only (5-9 hours)
ablation_configs = [
    {'name': 'Full GMVAE4P', 'config': {}},
    {'name': 'w/o multi-modal fusion', 'config': {'proportions_only': True}},
    {'name': 'w/o z-score normalization', 'config': {'no_zscore': True}},
]
```

---

## Timeline Comparison

| Approach | Time |
|----------|------|
| **Original (with GMVAE)** | 14-22 hours |
| **Fast Track (skip GMVAE)** | 13-22 hours |
| **Ultra Fast (skip GMVAE + 3 ablations)** | 8-14 hours |

---

## Recommended: Fast Track

Skip GMVAE, run all 5 steps above. You get complete paper results in ~13-22 hours.

The GMVAE pretraining is conceptually important for the paper (shows transfer learning works), but you don't need the actual trained model to generate paper results.
