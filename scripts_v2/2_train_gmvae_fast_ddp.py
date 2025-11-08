#!/usr/bin/env python3
"""
FAST GMVAE Training with DDP - Much faster than original

Usage for 2xH200:
    torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_fast_ddp.py \
        --data_dir processed_data \
        --output models/gmvae/gmvae_fast.pt \
        --epochs 10 \
        --batch_size 4096
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import sys

sys.path.append(os.path.dirname(__file__))
from pathlib import Path
import scipy.io as sio
import pandas as pd
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset

# Import from fast version
try:
    from train_gmvae_fast import SimpleGMVAE, compute_loss
except ImportError:
    # If running as script, import from same directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("gmvae_fast", "scripts_v2/2_train_gmvae_fast.py")
    gmvae_fast = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gmvae_fast)
    SimpleGMVAE = gmvae_fast.SimpleGMVAE
    compute_loss = gmvae_fast.compute_loss


def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_fast_gmvae_ddp(data_loader, input_dim, n_cell_types, save_path,
                         epochs=10, learning_rate=1e-3, local_rank=0, world_size=1):
    """Train Fast GMVAE with DDP"""

    device = f'cuda:{local_rank}'
    is_main_process = local_rank == 0

    # Create model
    model = SimpleGMVAE(input_dim, n_cell_types, z_dim=64).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if is_main_process:
        print(f"\nTraining Fast GMVAE (DDP):")
        print(f"  Input dim: {input_dim}")
        print(f"  Cell types: {n_cell_types}")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {data_loader.batch_size}")
        print(f"  Effective batch size: {data_loader.batch_size * world_size}")
        print(f"  Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        data_loader.sampler.set_epoch(epoch)

        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_ce = 0
        correct = 0
        total = 0
        n_batches = 0

        for batch_idx, (x, cell_types) in enumerate(data_loader):
            x = x.to(device)
            cell_types = cell_types.to(device)

            optimizer.zero_grad()

            # Forward
            x_recon, mu, logvar, cell_type_logits = model(x, cell_types)

            # Loss
            loss, recon_loss, kl_loss, ce_loss = compute_loss(
                x, x_recon, mu, logvar, cell_type_logits, cell_types,
                model.module.mu_prior, model.module.logvar_prior
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
            n_batches += 1

        # Aggregate metrics across GPUs
        metrics = torch.tensor([total_loss, total_recon, total_kl, total_ce, correct, total, n_batches],
                               dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        if is_main_process:
            avg_loss = metrics[0].item() / metrics[6].item()
            avg_recon = metrics[1].item() / metrics[6].item()
            avg_kl = metrics[2].item() / metrics[6].item()
            avg_ce = metrics[3].item() / metrics[6].item()
            accuracy = metrics[4].item() / metrics[5].item()

            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"loss={avg_loss:.4f}, "
                  f"recon={avg_recon:.4f}, "
                  f"kl={avg_kl:.4f}, "
                  f"ce={avg_ce:.4f}, "
                  f"acc={accuracy:.4f}")

    # Save model (only on main process)
    if is_main_process:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'mu_prior': model.module.mu_prior.data,
            'logvar_prior': model.module.logvar_prior.data,
            'input_dim': input_dim,
            'n_cell_types': n_cell_types,
            'z_dim': 64,
        }, save_path)
        print(f"\nModel saved to: {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fast GMVAE with DDP')
    parser.add_argument('--data_dir', required=True, help='Preprocessed data directory')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size PER GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers per GPU')

    args = parser.parse_args()

    # Setup DDP
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    if is_main_process:
        print("=" * 60)
        print("FAST GMVAE DDP TRAINING")
        print("=" * 60)

    # Load metadata
    with open(os.path.join(args.data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    n_genes = metadata['n_genes']
    n_cell_types = metadata['n_cell_types']

    if is_main_process:
        print(f"Dataset: {metadata['n_cells']} cells × {n_genes} genes")
        print(f"Cell types: {n_cell_types}")

    # Load data
    if is_main_process:
        print("\nLoading data...")

    matrix = sio.mmread(os.path.join(args.data_dir, "matrix.mtx")).T.tocsr()
    labels_df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    cell_type_labels = torch.LongTensor(labels_df['cluster'].values)

    # Convert to tensor
    if hasattr(matrix, 'toarray'):
        X = torch.FloatTensor(matrix.toarray())
    else:
        X = torch.FloatTensor(matrix)

    if is_main_process:
        print(f"Loaded: {X.shape[0]} cells × {X.shape[1]} genes")

    # Create dataset with DistributedSampler
    dataset = TensorDataset(X, cell_type_labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Train
    model = train_fast_gmvae_ddp(
        data_loader,
        input_dim=n_genes,
        n_cell_types=n_cell_types,
        save_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        local_rank=local_rank,
        world_size=world_size
    )

    cleanup_ddp()

    if is_main_process:
        print("\nDone!")
