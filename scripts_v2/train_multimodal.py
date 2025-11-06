#!/usr/bin/env python3
"""
Training script for uncertainty-aware multi-modal bulk RNA-seq classifier.

Usage:
    python train_multimodal.py --data_dir processed_data/multimodal --output_dir models/multimodal
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
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
    """Validate model"""
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

    # ROC-AUC (handle binary and multiclass)
    if probs.shape[1] == 2:
        roc_auc = roc_auc_score(labels.numpy(), probs[:, 1].numpy())
    else:
        roc_auc = roc_auc_score(labels.numpy(), probs.numpy(), multi_class='ovr')

    f1_macro = f1_score(labels.numpy(), preds.numpy(), average='macro')

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_macro': f1_macro
    }


def main():
    parser = argparse.ArgumentParser(description='Train multi-modal classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with pseudobulk data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--state_dim', type=int, default=64, help='State dimension per cell type')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-MODAL BULK RNA-SEQ TRAINING")
    print("=" * 80)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data_dir = Path(args.data_dir)

    pseudobulk = pd.read_csv(data_dir / 'pseudobulk.csv', index_col=0)
    proportions = pd.read_csv(data_dir / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'communication.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'labels.csv', index_col=0, squeeze=True)

    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(pseudobulk)} patients")
    print(f"  Pseudobulk: {pseudobulk.shape}")
    print(f"  Proportions: {proportions.shape}")
    print(f"  States: {states.shape}")
    print(f"  Communication: {communication.shape}")

    # Train/val split
    indices = np.arange(len(pseudobulk))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=42,
        stratify=labels.values
    )

    train_dataset = MultiModalDataset(
        pseudobulk.iloc[train_idx],
        proportions.iloc[train_idx],
        states.iloc[train_idx],
        communication.iloc[train_idx],
        labels.iloc[train_idx]
    )

    val_dataset = MultiModalDataset(
        pseudobulk.iloc[val_idx],
        proportions.iloc[val_idx],
        states.iloc[val_idx],
        communication.iloc[val_idx],
        labels.iloc[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nTrain: {len(train_dataset)} patients, Val: {len(val_dataset)} patients")

    # Initialize model
    print("\nInitializing model...")
    model = MultiModalClassifier(
        n_genes=pseudobulk.shape[1],
        n_cell_types=proportions.shape[1],
        n_interactions=communication.shape[1],
        n_classes=metadata['n_classes'],
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_auc = 0
    patience_counter = 0
    max_patience = 10

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
        'val_f1_macro': []
    }

    for epoch in range(args.epochs):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_metrics['roc_auc'])

        # Save history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Total: {train_losses['total']:.4f}, "
              f"Cls: {train_losses['classification']:.4f}, "
              f"Recon: {train_losses['reconstruction']:.4f}, "
              f"KL: {train_losses['kl']:.4f}, "
              f"Contrast: {train_losses['contrastive']:.4f}")
        print(f"  Val - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['roc_auc']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")

        # Save best model
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_roc_auc': best_val_auc,
                'metadata': metadata
            }, output_dir / 'best_model.pth')

            print(f"  *** New best model (AUC: {best_val_auc:.4f}) ***")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save final summary
    summary = {
        'best_val_roc_auc': best_val_auc,
        'best_val_acc': max(history['val_accuracy']),
        'best_val_f1': max(history['val_f1_macro']),
        'n_epochs': len(history['train_loss']),
        'n_params': n_params,
        'config': vars(args)
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
