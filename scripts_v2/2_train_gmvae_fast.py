#!/usr/bin/env python3
"""
FAST GMVAE Training - Simplified for Speed

Key optimizations:
1. Simpler VAE (no ZINB, just MSE reconstruction)
2. Smaller batch size to reduce memory and time
3. Fewer epochs needed
4. Stronger learning signal for cell type classification

Usage:
    python scripts_v2/2_train_gmvae_fast.py \
        --data_dir processed_data \
        --output models/gmvae/gmvae_fast.pt \
        --epochs 10 \
        --batch_size 4096 \
        --learning_rate 1e-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class SimpleGMVAE(nn.Module):
    """Simplified GMVAE with MSE reconstruction - much faster than ZINB"""

    def __init__(self, input_dim, n_cell_types, z_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.n_cell_types = n_cell_types
        self.z_dim = z_dim

        # Cell type classifier (from gene expression)
        self.cell_type_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cell_types)
        )

        # Encoder (gene expression → latent z)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

        # Decoder (latent z → gene expression)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # Learnable priors per cell type
        self.mu_prior = nn.Parameter(torch.randn(n_cell_types, z_dim) * 0.1)
        self.logvar_prior = nn.Parameter(torch.zeros(n_cell_types, z_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cell_types=None):
        # Cell type prediction
        cell_type_logits = self.cell_type_classifier(x)

        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar, cell_type_logits

    def get_embeddings(self, x):
        """Get embeddings for downstream tasks"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        return mu


def compute_loss(x, x_recon, mu, logvar, cell_type_logits, cell_types, mu_prior, logvar_prior):
    """
    Simplified loss:
    1. MSE reconstruction (much faster than ZINB)
    2. KL divergence to cell-type-specific priors
    3. Cell type classification
    """
    batch_size = x.size(0)

    # 1. Reconstruction loss (MSE on log1p transformed counts)
    x_log = torch.log1p(x)
    x_recon_log = torch.log1p(torch.relu(x_recon))  # Ensure non-negative
    recon_loss = F.mse_loss(x_recon_log, x_log, reduction='sum') / batch_size

    # 2. KL divergence to cell-type-specific priors
    # Get prior for each sample's cell type
    prior_mu = mu_prior[cell_types]  # [batch, z_dim]
    prior_logvar = logvar_prior[cell_types]  # [batch, z_dim]

    kl_loss = -0.5 * torch.sum(
        1 + logvar - prior_logvar -
        ((mu - prior_mu).pow(2) + logvar.exp()) / prior_logvar.exp()
    ) / batch_size

    # 3. Cell type classification (strong supervision signal)
    ce_loss = F.cross_entropy(cell_type_logits, cell_types)

    # Total loss with weights
    total_loss = recon_loss + 0.1 * kl_loss + 1.0 * ce_loss

    return total_loss, recon_loss, kl_loss, ce_loss


def train_fast_gmvae(data_loader, input_dim, n_cell_types, save_path,
                     epochs=10, learning_rate=1e-3, device='cuda'):
    """Train simplified GMVAE - much faster"""

    model = SimpleGMVAE(input_dim, n_cell_types, z_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining Fast GMVAE:")
    print(f"  Input dim: {input_dim}")
    print(f"  Cell types: {n_cell_types}")
    print(f"  Latent dim: 64")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {data_loader.batch_size}")
    print(f"  Device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_ce = 0
        correct = 0
        total = 0

        for batch_idx, (x, cell_types) in enumerate(data_loader):
            x = x.to(device)
            cell_types = cell_types.to(device)

            optimizer.zero_grad()

            # Forward
            x_recon, mu, logvar, cell_type_logits = model(x, cell_types)

            # Loss
            loss, recon_loss, kl_loss, ce_loss = compute_loss(
                x, x_recon, mu, logvar, cell_type_logits, cell_types,
                model.mu_prior, model.logvar_prior
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_ce += ce_loss.item()

            # Cell type accuracy
            preds = cell_type_logits.argmax(dim=1)
            correct += (preds == cell_types).sum().item()
            total += cell_types.size(0)

        # Epoch stats
        n_batches = len(data_loader)
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        avg_ce = total_ce / n_batches
        accuracy = correct / total

        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"loss={avg_loss:.4f}, "
              f"recon={avg_recon:.4f}, "
              f"kl={avg_kl:.4f}, "
              f"ce={avg_ce:.4f}, "
              f"acc={accuracy:.4f}")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'mu_prior': model.mu_prior.data,
        'logvar_prior': model.logvar_prior.data,
        'input_dim': input_dim,
        'n_cell_types': n_cell_types,
        'z_dim': 64,
    }, save_path)

    print(f"\nModel saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fast GMVAE')
    parser.add_argument('--data_dir', required=True, help='Preprocessed data directory')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("FAST GMVAE TRAINING")
    print("=" * 60)

    # Load metadata
    with open(os.path.join(args.data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    n_genes = metadata['n_genes']
    n_cell_types = metadata['n_cell_types']

    print(f"Dataset: {metadata['n_cells']} cells × {n_genes} genes")
    print(f"Cell types: {n_cell_types}")

    # Load data
    print("\nLoading data...")
    matrix = sio.mmread(os.path.join(args.data_dir, "matrix.mtx")).T.tocsr()
    labels_df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    cell_type_labels = torch.LongTensor(labels_df['cluster'].values)

    # Convert to tensor
    if hasattr(matrix, 'toarray'):
        X = torch.FloatTensor(matrix.toarray())
    else:
        X = torch.FloatTensor(matrix)

    print(f"Loaded: {X.shape[0]} cells × {X.shape[1]} genes")

    # Create dataset and dataloader
    dataset = TensorDataset(X, cell_type_labels)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Train
    model = train_fast_gmvae(
        data_loader,
        input_dim=n_genes,
        n_cell_types=n_cell_types,
        save_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    print("\nDone!")
