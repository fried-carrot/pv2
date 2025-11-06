"""
Cost profiling for GMVAE4P (our method)
Profiles the enhanced GMVAE + P4P classifier pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from utils.cost_profiler import CostProfiler
import json

# import GMVAE from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_gmvae import GMVAE_ZINB, Args as GMVAEArgs
from train_classifier import P4PClassifier


def create_gmvae_model(input_dim=1000, n_cell_types=8, device='cuda'):
    """Create GMVAE model for profiling"""
    args = GMVAEArgs(input_dim, n_cell_types, device)
    model = GMVAE_ZINB(args).to(device)
    return model, args


def create_p4p_model(z_dim=64, n_classes=2, n_cell_types=8, device='cuda'):
    """Create P4P classifier for profiling"""
    model = P4PClassifier(z_dim=z_dim, n_classes=n_classes, n_cell_types=n_cell_types).to(device)
    return model


def profile_gmvae4p(
    input_dim=1000,
    n_cell_types=8,
    n_classes=2,
    batch_size=512,
    num_training_samples=834000,
    gmvae_epochs=100,
    classifier_epochs=50,
    output_dir="cost_profiles"
):
    """
    Profile GMVAE4P pipeline

    Args:
        input_dim: Number of genes
        n_cell_types: Number of cell types (K mixture components)
        n_classes: Number of patient classes
        batch_size: Training batch size
        num_training_samples: Total cells in dataset
        gmvae_epochs: GMVAE training epochs
        classifier_epochs: Classifier training epochs
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling GMVAE4P on {device}")
    print(f"input: {input_dim} genes, {n_cell_types} cell types, {n_classes} patient classes")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    # profile GMVAE
    print("=" * 60)
    print("profiling GMVAE (step 1/2)")
    print("=" * 60)

    gmvae_model, gmvae_args = create_gmvae_model(input_dim, n_cell_types, device)

    gmvae_profile = profiler.profile_model(
        model=gmvae_model,
        input_shape=(input_dim,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=gmvae_epochs,
        use_deepspeed=False  # manual timing fallback
    )

    gmvae_profile['model_name'] = 'GMVAE4P_step1_GMVAE'
    gmvae_profile['hyperparameters'] = {
        'input_dim': input_dim,
        'n_cell_types': n_cell_types,
        'z_dim': gmvae_args.z_dim,
        'h_dim': gmvae_args.h_dim,
        'h_dim1': gmvae_args.h_dim1,
        'h_dim2': gmvae_args.h_dim2,
        'batch_size': batch_size,
        'epochs': gmvae_epochs
    }

    profiler.save_profile(gmvae_profile, f"{output_dir}/gmvae4p_step1_gmvae.json")

    print(f"\nGMVAE training cost: ${gmvae_profile['training_cost_usd']:.2f}")
    print(f"GMVAE training time: {gmvae_profile['estimated_training_time_hours']:.2f} hours")
    print()

    # profile P4P classifier
    print("=" * 60)
    print("profiling P4P classifier (step 2/2)")
    print("=" * 60)

    p4p_model = create_p4p_model(
        z_dim=gmvae_args.z_dim,
        n_classes=n_classes,
        n_cell_types=n_cell_types,
        device=device
    )

    p4p_profile = profiler.profile_model(
        model=p4p_model,
        input_shape=(gmvae_args.z_dim,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=classifier_epochs,
        use_deepspeed=False
    )

    p4p_profile['model_name'] = 'GMVAE4P_step2_P4P'
    p4p_profile['hyperparameters'] = {
        'z_dim': gmvae_args.z_dim,
        'n_classes': n_classes,
        'n_cell_types': n_cell_types,
        'n_prototypes': n_cell_types * n_classes,
        'batch_size': batch_size,
        'epochs': classifier_epochs
    }

    profiler.save_profile(p4p_profile, f"{output_dir}/gmvae4p_step2_p4p.json")

    print(f"\nP4P training cost: ${p4p_profile['training_cost_usd']:.2f}")
    print(f"P4P training time: {p4p_profile['estimated_training_time_hours']:.2f} hours")
    print()

    # combined profile
    print("=" * 60)
    print("GMVAE4P total pipeline")
    print("=" * 60)

    total_profile = {
        'model_name': 'GMVAE4P_full_pipeline',
        'training_cost_usd': gmvae_profile['training_cost_usd'] + p4p_profile['training_cost_usd'],
        'training_time_hours': gmvae_profile['estimated_training_time_hours'] + p4p_profile['estimated_training_time_hours'],
        'inference_time_ms': gmvae_profile['measured_inference_time_ms'] + p4p_profile['measured_inference_time_ms'],
        'total_flops': gmvae_profile['total_training_flops'] + p4p_profile['total_training_flops'],
        'step1_gmvae': gmvae_profile,
        'step2_p4p': p4p_profile,
        'hyperparameters': {
            'input_dim': input_dim,
            'n_cell_types': n_cell_types,
            'n_classes': n_classes,
            'gmvae_epochs': gmvae_epochs,
            'classifier_epochs': classifier_epochs,
            'batch_size': batch_size
        }
    }

    profiler.save_profile(total_profile, f"{output_dir}/gmvae4p_full.json")

    print(f"total training cost: ${total_profile['training_cost_usd']:.2f}")
    print(f"total training time: {total_profile['training_time_hours']:.2f} hours")
    print(f"total inference time: {total_profile['inference_time_ms']:.2f} ms/batch")
    print()

    return total_profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile GMVAE4P costs')
    parser.add_argument('--input_dim', type=int, default=1000, help='number of genes')
    parser.add_argument('--n_cell_types', type=int, default=8, help='number of cell types')
    parser.add_argument('--n_classes', type=int, default=2, help='number of patient classes')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--gmvae_epochs', type=int, default=100, help='GMVAE epochs')
    parser.add_argument('--classifier_epochs', type=int, default=50, help='classifier epochs')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_gmvae4p(
        input_dim=args.input_dim,
        n_cell_types=args.n_cell_types,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        gmvae_epochs=args.gmvae_epochs,
        classifier_epochs=args.classifier_epochs,
        output_dir=args.output_dir
    )

    print("profiling complete")
