"""
Cost profiling for ProtoCell4P baseline
Original P4P implementation from Xiong et al. 2023
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../repositories/ProtoCell4P/src'))

import torch
from utils.cost_profiler import CostProfiler

# import ProtoCell model
try:
    from model import ProtoCell
except ImportError:
    print("error: cannot import ProtoCell from repositories/ProtoCell4P/src/model.py")
    sys.exit(1)


def create_protocell4p_model(
    input_dim=1000,
    h_dim=512,
    z_dim=64,
    n_layers=3,
    n_proto=16,
    n_classes=2,
    n_ct=8,
    device='cuda'
):
    """Create ProtoCell4P model for profiling"""
    lambdas = {
        'lambda_1': 0.0,
        'lambda_2': 0.0,
        'lambda_3': 0.0,
        'lambda_4': 0.0,
        'lambda_5': 0.0,
        'lambda_6': 1.0,
    }

    model = ProtoCell(
        input_dim=input_dim,
        h_dim=h_dim,
        z_dim=z_dim,
        n_layers=n_layers,
        n_proto=n_proto,
        n_classes=n_classes,
        lambdas=lambdas,
        n_ct=n_ct,
        device=device
    ).to(device)

    return model


def profile_protocell4p(
    input_dim=1000,
    h_dim=512,
    z_dim=64,
    n_layers=3,
    n_proto=16,
    n_classes=2,
    n_ct=8,
    batch_size=512,
    num_training_samples=834000,
    epochs=100,
    output_dir="cost_profiles"
):
    """
    Profile ProtoCell4P costs

    Args:
        input_dim: Number of genes
        h_dim: Hidden dimension
        z_dim: Embedding dimension
        n_layers: Number of encoder/decoder layers
        n_proto: Number of prototypes
        n_classes: Number of patient classes
        n_ct: Number of cell types
        batch_size: Training batch size
        num_training_samples: Total cells in dataset
        epochs: Training epochs
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling ProtoCell4P on {device}")
    print(f"input: {input_dim} genes, {n_ct} cell types, {n_classes} patient classes")
    print(f"architecture: h_dim={h_dim}, z_dim={z_dim}, n_layers={n_layers}, n_proto={n_proto}")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    model = create_protocell4p_model(
        input_dim=input_dim,
        h_dim=h_dim,
        z_dim=z_dim,
        n_layers=n_layers,
        n_proto=n_proto,
        n_classes=n_classes,
        n_ct=n_ct,
        device=device
    )

    profile = profiler.profile_model(
        model=model,
        input_shape=(input_dim,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=epochs,
        use_deepspeed=False
    )

    profile['model_name'] = 'ProtoCell4P'
    profile['hyperparameters'] = {
        'input_dim': input_dim,
        'h_dim': h_dim,
        'z_dim': z_dim,
        'n_layers': n_layers,
        'n_proto': n_proto,
        'n_classes': n_classes,
        'n_ct': n_ct,
        'batch_size': batch_size,
        'epochs': epochs
    }

    profiler.save_profile(profile, f"{output_dir}/protocell4p.json")

    print(f"training cost: ${profile['training_cost_usd']:.2f}")
    print(f"training time: {profile['estimated_training_time_hours']:.2f} hours")
    print(f"inference time: {profile['measured_inference_time_ms']:.2f} ms/batch")
    print()

    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile ProtoCell4P costs')
    parser.add_argument('--input_dim', type=int, default=1000, help='number of genes')
    parser.add_argument('--h_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--z_dim', type=int, default=64, help='embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--n_proto', type=int, default=16, help='number of prototypes')
    parser.add_argument('--n_classes', type=int, default=2, help='number of patient classes')
    parser.add_argument('--n_ct', type=int, default=8, help='number of cell types')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_protocell4p(
        input_dim=args.input_dim,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_layers=args.n_layers,
        n_proto=args.n_proto,
        n_classes=args.n_classes,
        n_ct=args.n_ct,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    print("profiling complete")
