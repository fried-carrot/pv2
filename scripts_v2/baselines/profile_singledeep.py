"""
Cost profiling for singleDeep baseline
4-layer feedforward network from Garcia-Ruiz et al. 2024
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../repositories/singleDeep-main'))

import torch
import torch.nn as nn
from utils.cost_profiler import CostProfiler


class SingleDeepModel(nn.Module):
    """singleDeep neural network architecture"""
    def __init__(self, n_genes, Hs1=500, Hs2=250, Hs3=125, Hs4=50, out_neurons=2):
        super(SingleDeepModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_genes, Hs1),
            nn.ReLU(),
            nn.Linear(Hs1, Hs2),
            nn.ReLU(),
            nn.Linear(Hs2, Hs3),
            nn.ReLU(),
            nn.Linear(Hs3, Hs4),
            nn.ReLU(),
            nn.Linear(Hs4, out_neurons),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def create_singledeep_model(
    n_genes=1000,
    Hs1=500,
    Hs2=250,
    Hs3=125,
    Hs4=50,
    out_neurons=2,
    device='cuda'
):
    """Create singleDeep model for profiling"""
    model = SingleDeepModel(
        n_genes=n_genes,
        Hs1=Hs1,
        Hs2=Hs2,
        Hs3=Hs3,
        Hs4=Hs4,
        out_neurons=out_neurons
    ).to(device)

    return model


def profile_singledeep(
    n_genes=1000,
    Hs1=500,
    Hs2=250,
    Hs3=125,
    Hs4=50,
    out_neurons=2,
    batch_size=64,
    num_training_samples=834000,
    epochs=250,
    output_dir="cost_profiles"
):
    """
    Profile singleDeep costs

    Args:
        n_genes: Number of genes
        Hs1-Hs4: Hidden layer sizes
        out_neurons: Number of output classes
        batch_size: Training batch size
        num_training_samples: Total cells in dataset
        epochs: Training epochs
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling singleDeep on {device}")
    print(f"input: {n_genes} genes")
    print(f"architecture: {n_genes} -> {Hs1} -> {Hs2} -> {Hs3} -> {Hs4} -> {out_neurons}")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    model = create_singledeep_model(
        n_genes=n_genes,
        Hs1=Hs1,
        Hs2=Hs2,
        Hs3=Hs3,
        Hs4=Hs4,
        out_neurons=out_neurons,
        device=device
    )

    profile = profiler.profile_model(
        model=model,
        input_shape=(n_genes,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=epochs,
        use_deepspeed=False
    )

    profile['model_name'] = 'singleDeep'
    profile['hyperparameters'] = {
        'n_genes': n_genes,
        'Hs1': Hs1,
        'Hs2': Hs2,
        'Hs3': Hs3,
        'Hs4': Hs4,
        'out_neurons': out_neurons,
        'batch_size': batch_size,
        'epochs': epochs
    }

    profiler.save_profile(profile, f"{output_dir}/singledeep.json")

    print(f"training cost: ${profile['training_cost_usd']:.2f}")
    print(f"training time: {profile['estimated_training_time_hours']:.2f} hours")
    print(f"inference time: {profile['measured_inference_time_ms']:.2f} ms/batch")
    print()

    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile singleDeep costs')
    parser.add_argument('--n_genes', type=int, default=1000, help='number of genes')
    parser.add_argument('--Hs1', type=int, default=500, help='hidden layer 1 size')
    parser.add_argument('--Hs2', type=int, default=250, help='hidden layer 2 size')
    parser.add_argument('--Hs3', type=int, default=125, help='hidden layer 3 size')
    parser.add_argument('--Hs4', type=int, default=50, help='hidden layer 4 size')
    parser.add_argument('--out_neurons', type=int, default=2, help='number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--epochs', type=int, default=250, help='training epochs')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_singledeep(
        n_genes=args.n_genes,
        Hs1=args.Hs1,
        Hs2=args.Hs2,
        Hs3=args.Hs3,
        Hs4=args.Hs4,
        out_neurons=args.out_neurons,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    print("profiling complete")
