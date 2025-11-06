"""
Cost profiling for ScRAT baseline
Transformer-based model from Mao et al. 2023
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../repositories/ScRAT-main'))

import torch
from utils.cost_profiler import CostProfiler

# import ScRAT model
try:
    from Transformer import TransformerPredictor
except ImportError:
    print("error: cannot import TransformerPredictor from repositories/ScRAT-main/Transformer.py")
    sys.exit(1)


def create_scrat_model(
    input_dim=1000,
    model_dim=128,
    num_classes=1,  # binary classification with BCE
    num_heads=8,
    num_layers=1,
    dropout=0.3,
    input_dropout=0.0,
    pca=True,
    norm_first=False,
    device='cuda'
):
    """Create ScRAT model for profiling"""
    model = TransformerPredictor(
        input_dim=input_dim,
        model_dim=model_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        input_dropout=input_dropout,
        pca=pca,
        norm_first=norm_first
    ).to(device)

    return model


def profile_scrat(
    input_dim=1000,
    model_dim=128,
    num_classes=1,
    num_heads=8,
    num_layers=1,
    dropout=0.3,
    batch_size=16,  # ScRAT uses small batches
    num_training_samples=834000,
    epochs=100,
    output_dir="cost_profiles"
):
    """
    Profile ScRAT costs

    Args:
        input_dim: Number of genes
        model_dim: Transformer hidden dimension
        num_classes: Number of output classes (1 for binary BCE)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        batch_size: Training batch size (ScRAT uses small batches)
        num_training_samples: Total cells in dataset
        epochs: Training epochs
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling ScRAT on {device}")
    print(f"input: {input_dim} genes")
    print(f"architecture: model_dim={model_dim}, heads={num_heads}, layers={num_layers}")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    model = create_scrat_model(
        input_dim=input_dim,
        model_dim=model_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )

    # ScRAT operates on cell sequences (batch_size, seq_len, features)
    # for profiling, assume average of 500 cells per patient sample
    seq_len = 500

    profile = profiler.profile_model(
        model=model,
        input_shape=(seq_len, input_dim),  # transformer input
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=epochs,
        use_deepspeed=False
    )

    profile['model_name'] = 'ScRAT'
    profile['hyperparameters'] = {
        'input_dim': input_dim,
        'model_dim': model_dim,
        'num_classes': num_classes,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': epochs,
        'seq_len': seq_len
    }

    profiler.save_profile(profile, f"{output_dir}/scrat.json")

    print(f"training cost: ${profile['training_cost_usd']:.2f}")
    print(f"training time: {profile['estimated_training_time_hours']:.2f} hours")
    print(f"inference time: {profile['measured_inference_time_ms']:.2f} ms/batch")
    print()

    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile ScRAT costs')
    parser.add_argument('--input_dim', type=int, default=1000, help='number of genes')
    parser.add_argument('--model_dim', type=int, default=128, help='transformer hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=1, help='number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_scrat(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    print("profiling complete")
