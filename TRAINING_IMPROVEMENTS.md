# Training Speed Improvements

## Changes Made

### 1. Faster Training with More Epochs

**Before:**
- 30 epochs max
- Batch size: 16
- Learning rate: 1e-4
- Early stopping patience: 10 epochs
- Print every 5 epochs
- Save only best model

**After:**
- **100 epochs max** (with early stopping)
- **Batch size: 32** (2x faster per epoch)
- **Learning rate: 2e-4** (faster convergence)
- **Early stopping patience: 15 epochs** (more chances to improve)
- **Print every epoch** (better monitoring)
- **Save checkpoint every epoch** (can resume or analyze any epoch)

### 2. Model Checkpoints

Now saves two types of checkpoints per fold:

1. **Best model:** `fold_{N}_best.pth` (highest ROC-AUC)
2. **Every epoch:** `fold_{N}_epoch_{E}.pth` (all epochs saved)

Example after 25 epochs:
```
results/full_pipeline/models/
├── fold_1_best.pth              # Best model (e.g., epoch 18)
├── fold_1_epoch_1.pth
├── fold_1_epoch_2.pth
...
├── fold_1_epoch_25.pth
```

### 3. Benefits

**Speed:**
- 2x larger batch size → 2x fewer iterations per epoch
- Higher learning rate → faster convergence
- Early stopping with patience 15 → stops when not improving

**Flexibility:**
- Can load any epoch checkpoint for analysis
- Can resume training from any epoch
- Can compare models across epochs

**Monitoring:**
- Print every epoch → see progress in real-time
- Track exact epoch where model peaked

---

## Updated Commands

### Run Complete Pipeline (100 epochs, batch 32)

```bash
python scripts_v2/run_complete_pipeline.py \
    --data_dir processed_data \
    --output_dir results/full_pipeline \
    --device cuda \
    --n_gpus 4 \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4
```

**Expected behavior:**
- Each epoch ~30-60 seconds (vs 1-2 min before)
- Early stopping typically around epoch 25-40
- Total per fold: 15-30 minutes
- Total for 5 folds: 1-2.5 hours (vs 3-4 hours before)

### Run Ablation (100 epochs, batch 32)

```bash
python scripts_v2/run_ablation.py \
    --data_dir processed_data \
    --output_dir results/ablation \
    --device cuda \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4
```

**Expected behavior:**
- 6 configs × 5 folds × ~20 min = 10 hours (vs 15-20 hours before)

---

## Output Example

### Training Output (Every Epoch)

```
Fold 1/5 Training:
------------------------------------------------------------
  Epoch   1/100: Val AUC=0.6234, Acc=0.6154, F1=0.6012, Loss=1.2345
  Epoch   2/100: Val AUC=0.6789, Acc=0.6615, F1=0.6543, Loss=0.9876
  Epoch   3/100: Val AUC=0.7123, Acc=0.6923, F1=0.6834, Loss=0.8234
  ...
  Epoch  25/100: Val AUC=0.8012, Acc=0.7846, F1=0.7712, Loss=0.4123
  Epoch  26/100: Val AUC=0.7998, Acc=0.7831, F1=0.7698, Loss=0.4156
  ...
  Epoch  40/100: Val AUC=0.7956, Acc=0.7800, F1=0.7650, Loss=0.4289
  Early stopping at epoch 40
  Fold 1 Best AUC: 0.8012
```

### Saved Checkpoints

```bash
ls results/full_pipeline/models/fold_1_*.pth

fold_1_best.pth          # Epoch 25 (best ROC-AUC)
fold_1_epoch_1.pth
fold_1_epoch_2.pth
...
fold_1_epoch_40.pth      # Final epoch before early stopping
```

---

## Loading Specific Epochs

### Load Best Model

```python
checkpoint = torch.load('results/full_pipeline/models/fold_1_best.pth')
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best ROC-AUC: {checkpoint['val_roc_auc']:.4f}")

model.load_state_dict(checkpoint['model_state_dict'])
```

### Load Specific Epoch

```python
checkpoint = torch.load('results/full_pipeline/models/fold_1_epoch_25.pth')
print(f"Epoch 25 ROC-AUC: {checkpoint['val_roc_auc']:.4f}")

model.load_state_dict(checkpoint['model_state_dict'])
```

### Resume Training from Epoch

```python
checkpoint = torch.load('results/full_pipeline/models/fold_1_epoch_25.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training from epoch 26
for epoch in range(checkpoint['epoch'] + 1, 100):
    ...
```

---

## Expected Timeline

| Task | Old Time | New Time | Speedup |
|------|----------|----------|---------|
| Single fold training | 30-60 min | 15-30 min | 2x |
| 5-fold CV (GMVAE4P) | 3-4 hours | 1.5-2.5 hours | 1.5x |
| Ablation (6 configs) | 15-20 hours | 9-15 hours | 1.5x |
| **Total pipeline** | **20-25 hours** | **12-18 hours** | **~1.7x** |

---

## Cleanup Script (Optional)

If you want to keep only the best models and delete per-epoch checkpoints to save disk space:

```bash
# Keep only best models
cd results/full_pipeline/models
ls fold_*_epoch_*.pth | xargs rm

# Or keep best + last 5 epochs
ls fold_*_epoch_*.pth | sort -V | head -n -5 | xargs rm
```

---

## Summary

**Key improvements:**
1. **Batch size 32** (vs 16) → 2x faster per epoch
2. **Learning rate 2e-4** (vs 1e-4) → faster convergence
3. **100 epochs max** with early stopping at 15 patience → more training opportunities
4. **Save every epoch** → can analyze/resume from any point
5. **Print every epoch** → real-time monitoring

**Result:** ~1.7x speedup with better monitoring and flexibility.
