#!/usr/bin/env python3
"""
5-Fold Cross-Validation for uncertainty-aware multi-modal bulk RNA-seq classifier.

More robust evaluation for small datasets (169 patients).

Usage:
    python train_multimodal_cv.py --data_dir processed_data/multimodal --output_dir models/multimodal_cv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import sys
import os

sys.path.append(os.path.dirname(__file__))
from multimodal_bulk import MultiModalClassifier


class MultiModalDataset(Dataset):
    """Dataset for multi-modal bulk RNA-seq training"""

    def __init__(
        self,
        pseudobulk: pd.DataFrame,
        proportions: pd.DataFrame,
        states: pd.DataFrame,
        communication: pd.DataFrame,
        labels: pd.Series
    ):
        self.pseudobulk = torch.FloatTensor(pseudobulk.values)
        self.proportions = torch.FloatTensor(proportions.values)
        self.states = torch.FloatTensor(states.values)
        self.communication = torch.FloatTensor(communication.values)

        # Convert labels to numeric
        label_map = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
        self.labels = torch.LongTensor([label_map[l] for l in labels.values])
        self.label_map = label_map
        self.label_names = {v: k for k, v in label_map.items()}

    def __len__(self):
        return len(self.pseudobulk)

    def __getitem__(self, idx):
        return {
            'bulk': self.pseudobulk[idx],
            'proportions': self.proportions[idx],
            'states': self.states[idx],
            'communication': self.communication[idx],
            'label': self.labels[idx]
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_cls = 0
    total_recon = 0
    total_kl = 0
    total_contrastive = 0

    for batch in dataloader:
        bulk = batch['bulk'].to(device)
        proportions = batch['proportions'].to(device)
        states = batch['states'].to(device)
        communication = batch['communication'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        output = model(
            bulk=bulk,
            labels=labels,
            true_proportions=proportions,
            true_states=states,
            true_communication=communication,
            training=True
        )

        loss = output['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += output['loss_components']['classification']
        total_recon += output['loss_components']['reconstruction']
        total_kl += output['loss_components']['kl']
        total_contrastive += output['loss_components']['contrastive']

    n_batches = len(dataloader)
    return {
        'total': total_loss / n_batches,
        'classification': total_cls / n_batches,
        'reconstruction': total_recon / n_batches,
        'kl': total_kl / n_batches,
        'contrastive': total_contrastive / n_batches
    }


def validate(model, dataloader, device):
    """Validate model and return detailed metrics"""
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            bulk = batch['bulk'].to(device)
            proportions = batch['proportions'].to(device)
            states = batch['states'].to(device)
            communication = batch['communication'].to(device)
            labels = batch['label'].to(device)

            output = model(
                bulk=bulk,
                labels=labels,
                true_proportions=proportions,
                true_states=states,
                true_communication=communication,
                training=True
            )

            all_logits.append(output['logits'].cpu())
            all_labels.append(labels.cpu())
            total_loss += output['loss'].item()

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

    accuracy = accuracy_score(labels.numpy(), preds.numpy())
    precision = precision_score(labels.numpy(), preds.numpy(), average='macro', zero_division=0)
    recall = recall_score(labels.numpy(), preds.numpy(), average='macro', zero_division=0)
    f1_macro = f1_score(labels.numpy(), preds.numpy(), average='macro', zero_division=0)

    # ROC-AUC
    if probs.shape[1] == 2:
        roc_auc = roc_auc_score(labels.numpy(), probs[:, 1].numpy())
    else:
        roc_auc = roc_auc_score(labels.numpy(), probs.numpy(), multi_class='ovr')

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'predictions': preds.numpy(),
        'labels': labels.numpy(),
        'probabilities': probs.numpy()
    }


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    fold_num,
    output_dir
):
    """Train a single fold"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_auc = 0
    patience_counter = 0
    max_patience = 10

    fold_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
        'val_f1_macro': []
    }

    print(f"\n  Fold {fold_num} Training:")
    print("  " + "-" * 60)

    for epoch in range(epochs):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_metrics['roc_auc'])

        # Save history
        fold_history['train_loss'].append(train_losses['total'])
        fold_history['val_loss'].append(val_metrics['loss'])
        fold_history['val_accuracy'].append(val_metrics['accuracy'])
        fold_history['val_roc_auc'].append(val_metrics['roc_auc'])
        fold_history['val_f1_macro'].append(val_metrics['f1_macro'])

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"Val AUC={val_metrics['roc_auc']:.4f}, "
                  f"Acc={val_metrics['accuracy']:.4f}, "
                  f"F1={val_metrics['f1_macro']:.4f}")

        # Track best
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0

            # Save best model for this fold
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_roc_auc': best_val_auc,
                'fold': fold_num
            }, output_dir / f'fold_{fold_num}_best.pth')

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Fold {fold_num} Best AUC: {best_val_auc:.4f}")

    # Load best model and get final validation metrics
    checkpoint = torch.load(output_dir / f'fold_{fold_num}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = validate(model, val_loader, device)

    return final_metrics, fold_history


def main():
    parser = argparse.ArgumentParser(description='5-Fold CV for multi-modal classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with pseudobulk data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (smaller for CV)')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs per fold')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--state_dim', type=int, default=64, help='State dimension per cell type')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-MODAL BULK RNA-SEQ: 5-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Folds: {args.n_folds}, Epochs/fold: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Seed: {args.seed}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data_dir = Path(args.data_dir)

    pseudobulk = pd.read_csv(data_dir / 'pseudobulk.csv', index_col=0)
    proportions = pd.read_csv(data_dir / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'communication.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'labels.csv', index_col=0).squeeze()

    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(pseudobulk)} patients")
    print(f"  Classes: {labels.value_counts().to_dict()}")

    # Create full dataset
    full_dataset = MultiModalDataset(
        pseudobulk, proportions, states, communication, labels
    )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print(f"\n{'=' * 80}")
    print(f"STARTING {args.n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 80}")

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(full_dataset)), full_dataset.labels.numpy()), 1):
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_num}/{args.n_folds}")
        print(f"{'=' * 80}")
        print(f"  Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")

        # Create data loaders
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        # Initialize fresh model for each fold
        model = MultiModalClassifier(
            n_genes=pseudobulk.shape[1],
            n_cell_types=proportions.shape[1],
            n_interactions=communication.shape[1],
            n_classes=metadata['n_classes'],
            embedding_dim=args.embedding_dim,
            state_dim=args.state_dim
        ).to(device)

        # Train fold
        fold_metrics, fold_history = train_fold(
            model, train_loader, val_loader, device,
            args.epochs, args.lr, fold_num, output_dir
        )

        # Store results
        fold_results.append({
            'fold': fold_num,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'accuracy': fold_metrics['accuracy'],
            'precision': fold_metrics['precision'],
            'recall': fold_metrics['recall'],
            'f1_macro': fold_metrics['f1_macro'],
            'roc_auc': fold_metrics['roc_auc'],
            'history': fold_history
        })

        all_predictions.extend(fold_metrics['predictions'])
        all_labels.extend(fold_metrics['labels'])
        all_probabilities.append(fold_metrics['probabilities'])

    # Aggregate results
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 80}")

    results_df = pd.DataFrame(fold_results)

    print("\nPer-Fold Results:")
    print(results_df[['fold', 'roc_auc', 'accuracy', 'f1_macro', 'precision', 'recall']].to_string(index=False))

    print(f"\n{'=' * 80}")
    print("AGGREGATE METRICS (Mean ± Std)")
    print(f"{'=' * 80}")

    metrics_summary = {
        'roc_auc': {
            'mean': results_df['roc_auc'].mean(),
            'std': results_df['roc_auc'].std(),
            'min': results_df['roc_auc'].min(),
            'max': results_df['roc_auc'].max()
        },
        'accuracy': {
            'mean': results_df['accuracy'].mean(),
            'std': results_df['accuracy'].std(),
            'min': results_df['accuracy'].min(),
            'max': results_df['accuracy'].max()
        },
        'f1_macro': {
            'mean': results_df['f1_macro'].mean(),
            'std': results_df['f1_macro'].std(),
            'min': results_df['f1_macro'].min(),
            'max': results_df['f1_macro'].max()
        },
        'precision': {
            'mean': results_df['precision'].mean(),
            'std': results_df['precision'].std()
        },
        'recall': {
            'mean': results_df['recall'].mean(),
            'std': results_df['recall'].std()
        }
    }

    for metric, stats in metrics_summary.items():
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(range: {stats.get('min', 0):.4f}-{stats.get('max', 0):.4f})")

    # Save results
    results_summary = {
        'n_folds': args.n_folds,
        'n_patients': len(full_dataset),
        'metrics_summary': metrics_summary,
        'per_fold_results': [
            {k: v for k, v in fold.items() if k != 'history'}
            for fold in fold_results
        ],
        'config': vars(args)
    }

    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save detailed results
    results_df.to_csv(output_dir / 'cv_fold_results.csv', index=False)

    print(f"\n{'=' * 80}")
    print("Results saved:")
    print(f"  {output_dir / 'cv_results.json'}")
    print(f"  {output_dir / 'cv_fold_results.csv'}")
    print(f"  {output_dir / 'fold_*_best.pth'} (model checkpoints)")
    print(f"{'=' * 80}")

    # Final summary for paper
    print(f"\n{'=' * 80}")
    print("FOR PAPER:")
    print(f"{'=' * 80}")
    print(f"ROC-AUC: {metrics_summary['roc_auc']['mean']:.3f} ± {metrics_summary['roc_auc']['std']:.3f}")
    print(f"Accuracy: {metrics_summary['accuracy']['mean']:.3f} ± {metrics_summary['accuracy']['std']:.3f}")
    print(f"F1-Macro: {metrics_summary['f1_macro']['mean']:.3f} ± {metrics_summary['f1_macro']['std']:.3f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
