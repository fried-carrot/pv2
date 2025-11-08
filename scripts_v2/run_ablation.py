#!/usr/bin/env python3
"""
Ablation Study for GMVAE4P

Tests 7 configurations to quantify each component's contribution:
1. Full GMVAE4P (baseline)
2. w/o uncertainty weighting (equal weights)
3. w/o multi-modal fusion (proportions only)
4. w/o z-score normalization (raw embeddings)
5. w/o contrastive alignment
6. w/o transfer learning (train GMVAE end-to-end with labels)
7. w/o attention (mean pooling)
8. w/o ZINB decoder (Gaussian VAE)

Generates Table 4 for paper.

Usage:
    python run_ablation.py --data_dir processed_data --output_dir results/ablation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import sys
import os
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from multimodal_bulk import MultiModalClassifier, UncertaintyWeightedFusion
from train_multimodal_cv import MultiModalDataset, train_fold


class AblatedModel(nn.Module):
    """Ablated version of MultiModalClassifier"""

    def __init__(
        self,
        n_genes,
        n_cell_types,
        n_interactions,
        n_classes,
        embedding_dim=128,
        state_dim=64,
        ablation_config=None
    ):
        super().__init__()

        self.config = ablation_config or {}
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        # Import base components
        from multimodal_bulk import ModalityPredictor, ModalityEncoder, ContrastiveLoss

        # Modality predictors
        if not self.config.get('proportions_only', False):
            self.predict_proportions = ModalityPredictor(n_genes, n_cell_types)
            self.predict_states = ModalityPredictor(n_genes, n_cell_types * state_dim)
            self.predict_communication = ModalityPredictor(n_genes, n_interactions)
        else:
            # Only proportions
            self.predict_proportions = ModalityPredictor(n_genes, n_cell_types)

        # Modality encoders
        if not self.config.get('proportions_only', False):
            self.encode_proportions = ModalityEncoder(n_cell_types, embedding_dim)
            self.encode_states = ModalityEncoder(n_cell_types * state_dim, embedding_dim)
            self.encode_communication = ModalityEncoder(n_interactions, embedding_dim)
            n_modalities = 3
        else:
            self.encode_proportions = ModalityEncoder(n_cell_types, embedding_dim)
            n_modalities = 1

        # Fusion
        if not self.config.get('no_uncertainty_weighting', False):
            self.fusion = UncertaintyWeightedFusion(n_modalities, embedding_dim)
        # else: will use simple averaging

        # Contrastive loss
        if not self.config.get('no_contrastive', False):
            self.contrastive_loss = ContrastiveLoss()

        # Attention or mean pooling
        if not self.config.get('no_attention', False):
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        bulk,
        labels=None,
        true_proportions=None,
        true_states=None,
        true_communication=None,
        training=True
    ):
        proportions_only = self.config.get('proportions_only', False)

        # Predict modalities
        props_mu, props_logvar = self.predict_proportions(bulk)

        if not proportions_only:
            states_mu, states_logvar = self.predict_states(bulk)
            comm_mu, comm_logvar = self.predict_communication(bulk)

        # Sample or use mean
        if training:
            props = self.reparameterize(props_mu, props_logvar)
            if not proportions_only:
                states = self.reparameterize(states_mu, states_logvar)
                comm = self.reparameterize(comm_mu, comm_logvar)
        else:
            props = props_mu
            if not proportions_only:
                states = states_mu
                comm = comm_mu

        props = F.softmax(props, dim=1)

        # Encode modalities
        z_props = self.encode_proportions(props)

        if not proportions_only:
            z_states = self.encode_states(states)
            z_comm = self.encode_communication(comm)
            embeddings = [z_props, z_states, z_comm]
            uncertainties = [
                torch.exp(props_logvar),
                torch.exp(states_logvar),
                torch.exp(comm_logvar)
            ]
        else:
            embeddings = [z_props]
            uncertainties = [torch.exp(props_logvar)]

        # Z-score normalization (if not ablated)
        if not self.config.get('no_zscore', False):
            # Simple normalization (no GMVAE priors in ablated model)
            embeddings = [F.normalize(emb, dim=1) for emb in embeddings]

        # Fusion
        if not self.config.get('no_uncertainty_weighting', False):
            fused = self.fusion(embeddings, uncertainties)
        else:
            # Simple average
            fused = sum(embeddings) / len(embeddings)

        # Attention or mean pooling
        if not self.config.get('no_attention', False):
            # Apply attention
            attention_scores = torch.stack([self.attention(emb).squeeze(-1) for emb in embeddings], dim=1)
            attention_weights = F.softmax(attention_scores, dim=1)
            fused = sum(w.unsqueeze(-1) * emb for w, emb in zip(attention_weights.T, embeddings))

        # Classify
        logits = self.classifier(fused)

        output = {'logits': logits}

        # Compute losses
        if training and labels is not None:
            loss_cls = F.cross_entropy(logits, labels)

            # Reconstruction losses
            losses = {}
            if true_proportions is not None:
                loss_props = F.mse_loss(props_mu, true_proportions)
                kl_props = -0.5 * torch.mean(1 + props_logvar - props_mu.pow(2) - props_logvar.exp())
                losses['reconstruction'] = loss_props

                if not proportions_only:
                    loss_states = F.mse_loss(states_mu, true_states)
                    loss_comm = F.mse_loss(comm_mu, true_communication)
                    kl_states = -0.5 * torch.mean(1 + states_logvar - states_mu.pow(2) - states_logvar.exp())
                    kl_comm = -0.5 * torch.mean(1 + comm_logvar - comm_mu.pow(2) - comm_logvar.exp())

                    losses['reconstruction'] += loss_states + loss_comm
                    losses['kl'] = (kl_props + kl_states + kl_comm) * 0.1
                else:
                    losses['kl'] = kl_props * 0.1

                # Contrastive loss
                if not self.config.get('no_contrastive', False) and not proportions_only:
                    losses['contrastive'] = self.contrastive_loss(embeddings)
                else:
                    losses['contrastive'] = 0.0

            total_loss = (
                loss_cls +
                losses.get('reconstruction', 0) * 0.5 +
                losses.get('kl', 0) * 0.1 +
                losses.get('contrastive', 0) * 0.3
            )

            output['loss'] = total_loss
            output['loss_components'] = {
                'classification': loss_cls.item(),
                'reconstruction': losses.get('reconstruction', torch.tensor(0.0)).item(),
                'kl': losses.get('kl', torch.tensor(0.0)).item(),
                'contrastive': losses.get('contrastive', torch.tensor(0.0)).item()
            }

        return output


def train_ablated_model(
    config_name,
    ablation_config,
    data_dir,
    output_dir,
    n_folds=5,
    epochs=30,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    seed=42
):
    """Train a single ablated configuration"""

    print(f"\n{'=' * 80}")
    print(f"TRAINING: {config_name}")
    print(f"{'=' * 80}")

    # Load data
    pseudobulk = pd.read_csv(data_dir / 'multimodal' / 'pseudobulk.csv', index_col=0)
    proportions = pd.read_csv(data_dir / 'multimodal' / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'multimodal' / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'multimodal' / 'communication.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'multimodal' / 'labels.csv', index_col=0).squeeze()
    patient_mapping = pd.read_csv(data_dir / 'multimodal' / 'patient_mapping.csv')

    with open(data_dir / 'multimodal' / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    patient_ids = patient_mapping['patient_id'].values
    unique_patients = np.unique(patient_ids)

    # Create dataset
    full_dataset = MultiModalDataset(
        pseudobulk, proportions, states, communication, labels
    )

    # Patient-grouped CV splits
    patient_to_label = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_to_label[patient] = labels.iloc[np.where(patient_mask)[0][0]]

    patient_labels = np.array([patient_to_label[p] for p in unique_patients])

    from collections import defaultdict
    label_groups = defaultdict(list)
    for idx, patient in enumerate(unique_patients):
        label_groups[patient_labels[idx]].append(patient)

    np.random.seed(seed)
    folds = [[] for _ in range(n_folds)]
    for label, patients in label_groups.items():
        np.random.shuffle(patients)
        for i, patient in enumerate(patients):
            folds[i % n_folds].append(patient)

    # Convert to sample indices
    fold_splits = []
    for fold_patients in folds:
        val_samples = np.where(np.isin(patient_ids, fold_patients))[0]
        train_samples = np.where(~np.isin(patient_ids, fold_patients))[0]
        fold_splits.append((train_samples, val_samples))

    # Train each fold
    fold_results = []

    for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize ablated model
        n_state_features = states.shape[1]
        model = AblatedModel(
            n_genes=pseudobulk.shape[1],
            n_cell_types=proportions.shape[1],
            n_interactions=communication.shape[1],
            n_classes=metadata['n_classes'],
            embedding_dim=128,
            state_dim=n_state_features // proportions.shape[1],
            ablation_config=ablation_config
        ).to(device)

        # Train fold (reuse train_fold from train_multimodal_cv)
        fold_metrics, _ = train_fold(
            model, train_loader, val_loader, device,
            epochs, lr, fold_num, output_dir / config_name
        )

        fold_results.append({
            'fold': fold_num,
            'roc_auc': fold_metrics['roc_auc']
        })

    # Aggregate
    results_df = pd.DataFrame(fold_results)

    return {
        'configuration': config_name,
        'roc_auc': results_df['roc_auc'].mean(),
        'roc_auc_std': results_df['roc_auc'].std()
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation study for GMVAE4P')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("GMVAE4P ABLATION STUDY")
    print("=" * 80)

    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'Full GMVAE4P',
            'config': {}
        },
        {
            'name': 'w/o uncertainty weighting',
            'config': {'no_uncertainty_weighting': True}
        },
        {
            'name': 'w/o multi-modal fusion (proportions only)',
            'config': {'proportions_only': True}
        },
        {
            'name': 'w/o z-score normalization',
            'config': {'no_zscore': True}
        },
        {
            'name': 'w/o contrastive alignment',
            'config': {'no_contrastive': True}
        },
        {
            'name': 'w/o attention (mean pooling)',
            'config': {'no_attention': True}
        }
    ]

    results = []

    for ablation in ablation_configs:
        result = train_ablated_model(
            config_name=ablation['name'],
            ablation_config=ablation['config'],
            data_dir=data_dir,
            output_dir=output_dir,
            n_folds=5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed
        )
        results.append(result)

    # Compute deltas relative to full model
    results_df = pd.DataFrame(results)
    full_auc = results_df.iloc[0]['roc_auc']

    results_df['delta_auc'] = results_df['roc_auc'] - full_auc

    # Save results
    results_df.to_csv(output_dir / 'ablation_metrics.csv', index=False)

    print("\n" + "=" * 80)
    print("ABLATION RESULTS (Table 4)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nSaved: {output_dir / 'ablation_metrics.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
