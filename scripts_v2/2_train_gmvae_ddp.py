#!/usr/bin/env python3
"""
GMVAE training with Distributed Data Parallel (DDP) for multi-GPU training.

Usage for 4xH200:
    torchrun --nproc_per_node=4 scripts_v2/2_train_gmvae_ddp.py \
        --data_dir processed_data/filtered.h5ad \
        --output models/gmvae_model.pt \
        --epochs 30 \
        --batch_size 2048 \
        --learning_rate 5e-4
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import sys

# Import everything from the base training script
sys.path.append(os.path.dirname(__file__))
from pathlib import Path

# Import the base GMVAE components
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

# Import GMVAE model and losses from base script
import importlib.util
spec = importlib.util.spec_from_file_location("gmvae_base", "scripts_v2/2_train_gmvae.py")
gmvae_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmvae_base)

GMVAE_ZINB = gmvae_base.GMVAE_ZINB
gmvae_losses = gmvae_base.gmvae_losses
contrastive_loss = gmvae_base.contrastive_loss
Args = gmvae_base.Args


def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_gmvae_ddp(data_loader, input_dim, n_cell_types, save_path, epochs=100,
                    learning_rate=1e-3, local_rank=0, world_size=1):
    """
    Train GMVAE with DDP across multiple GPUs.

    Args:
        local_rank: GPU rank on this node
        world_size: Total number of GPUs
    """
    device = f'cuda:{local_rank}'
    is_main_process = local_rank == 0

    # Create args for model
    args = Args(input_dim, n_cell_types, device)

    # Create model and wrap with DDP
    model = GMVAE_ZINB(args).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler with warmup and cosine annealing
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if is_main_process:
        print(f"\nDDP Training on {world_size} GPUs")
        print(f"training GMVAE on {device}")
        print(f"input dimension: {input_dim}")
        print(f"cell types: {n_cell_types}")
        print(f"mixture components (K): {args.K}")
        print(f"learning rate schedule: warmup {warmup_epochs} epochs, then cosine annealing")

    # Mixed precision training
    use_amp = True
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if is_main_process:
        print(f"Mixed precision training (AMP): {use_amp}")
        print(f"Effective batch size: {data_loader.batch_size} × {world_size} = {data_loader.batch_size * world_size}")
        print()

    model.train()
    for epoch in range(epochs):
        # Set epoch for distributed sampler
        data_loader.sampler.set_epoch(epoch)

        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        n_batches = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (batch, labels) in enumerate(data_loader):
            x = batch.to(device)
            targets = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward and backward
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # forward pass
                    pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_zerop, x_recons_mean, \
                        x_recons_disper, x_recon_zerop, x_recon_mean, x_recon_disper = model(x, targets)

                    # losses
                    total_loss_batch, KLD_gaussian, KLD_pi, zinb_loss = gmvae_losses(
                        x, targets, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,
                        x_recons_zerop, x_recons_mean, x_recons_disper, epoch=epoch)

                    # contrastive regularization
                    z_embeddings = model.module.get_embeddings(x)  # .module for DDP
                    contrast_loss = contrastive_loss(z_embeddings, targets)

                    total_loss_batch = total_loss_batch + 0.1 * contrast_loss

                # backward pass with gradient scaling
                scaler.scale(total_loss_batch).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # forward pass (no AMP)
                pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_zerop, x_recons_mean, \
                    x_recons_disper, x_recon_zerop, x_recon_mean, x_recon_disper = model(x, targets)

                total_loss_batch, KLD_gaussian, KLD_pi, zinb_loss = gmvae_losses(
                    x, targets, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,
                    x_recons_zerop, x_recons_mean, x_recons_disper, epoch=epoch)

                z_embeddings = model.module.get_embeddings(x)
                contrast_loss = contrastive_loss(z_embeddings, targets)
                total_loss_batch = total_loss_batch + 0.1 * contrast_loss

                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += zinb_loss.item()
            total_kld_loss += (KLD_gaussian + KLD_pi).item()
            n_batches += 1

            # cell type classification accuracy
            predicted_cell_types = pi_x.argmax(dim=1)
            correct_predictions += (predicted_cell_types == targets).sum().item()
            total_predictions += targets.size(0)

        avg_loss = total_loss / n_batches
        avg_recon = total_recon_loss / n_batches
        avg_kld = total_kld_loss / n_batches
        cell_type_acc = correct_predictions / total_predictions

        # Synchronize metrics across GPUs
        metrics = torch.tensor([avg_loss, avg_recon, avg_kld, cell_type_acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size

        scheduler.step()

        if is_main_process and epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:3d}: loss={metrics[0]:.4f}, recon={metrics[1]:.4f}, "
                  f"kld={metrics[2]:.4f}, cell_type_acc={metrics[3]:.4f}, lr={current_lr:.6f}")

    # Only save on main process
    if is_main_process:
        model.eval()
        with torch.no_grad():
            # compute global priors
            mu_genz, logvar_genz = model.module.musig_of_genz(model.module.onehot, batchsize=1)
            model.module.mu_prior = mu_genz.squeeze(0).transpose(0, 1)
            model.module.logvar_prior = logvar_genz.squeeze(0).transpose(0, 1)
            print(f"stored global priors: mu_prior {model.module.mu_prior.shape}, "
                  f"logvar_prior {model.module.logvar_prior.shape}")

        # Save model (unwrap from DDP)
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'args': args,
            'mu_prior': model.module.mu_prior,
            'logvar_prior': model.module.logvar_prior,
            'training_config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'final_loss': metrics[0].item(),
                'world_size': world_size,
            }
        }, save_path)

        print(f"GMVAE saved to: {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train GMVAE-4P with DDP')
    parser.add_argument('--data_dir', required=True, help='preprocessed data directory')
    parser.add_argument('--output', required=True, help='output model path')
    parser.add_argument('--epochs', type=int, default=30, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size PER GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers per GPU')

    args = parser.parse_args()

    # Setup DDP
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    if is_main_process:
        print("=" * 60)
        print("GMVAE DDP training")
        print("=" * 60)

    # Load metadata
    with open(os.path.join(args.data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    n_genes = metadata['n_genes']
    n_cell_types = metadata['n_cell_types']

    if is_main_process:
        print(f"dataset: {metadata['n_cells']} cells x {n_genes} genes")
        print(f"cell types: {n_cell_types}")
        print(f"training: {args.epochs} epochs, batch size {args.batch_size} per GPU")
        print()

    # Load data
    if is_main_process:
        print("loading preprocessed data")
    matrix = sio.mmread(os.path.join(args.data_dir, "matrix.mtx")).T.tocsr()

    if is_main_process:
        print("loading cell type labels")
    labels_df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    cell_type_labels = torch.LongTensor(labels_df['cluster'].values)

    if is_main_process:
        print(f"loaded {len(cell_type_labels)} cell type labels")

    # Convert to tensor
    if hasattr(matrix, 'toarray'):
        X = torch.FloatTensor(matrix.toarray())
    else:
        X = torch.FloatTensor(matrix)

    if is_main_process:
        print(f"data shape: {X.shape}")
        print(f"labels shape: {cell_type_labels.shape}")

    # Create dataset and distributed sampler
    dataset = TensorDataset(X, cell_type_labels)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=42
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use distributed sampler instead of shuffle
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Train GMVAE
    model = train_gmvae_ddp(
        data_loader=data_loader,
        input_dim=n_genes,
        n_cell_types=n_cell_types,
        save_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        local_rank=local_rank,
        world_size=world_size
    )

    if is_main_process:
        print(f"\nTraining complete. Model saved to {args.output}")

    cleanup_ddp()
