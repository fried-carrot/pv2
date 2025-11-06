"""
Cost profiling for baseline GMVAE (no enhancements)
Original bulk2sc GMVAE without the 6 SOTA enhancements
For comparison to show enhancement benefits
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../repositories/bulk2sc_GMVAE/_model_source_codes'))

import torch
import torch.nn as nn
from utils.cost_profiler import CostProfiler


class BaselineGMVAE_ZINB(nn.Module):
    """
    Baseline GMVAE without enhancements:
    - No adaptive loss weighting (fixed alpha=1)
    - No contrastive loss
    - No batch normalization or dropout
    - No learning rate scheduling
    """
    def __init__(self, input_dim, K, z_dim=64, h_dim=512, h_dim1=512, h_dim2=256, device='cuda'):
        super(BaselineGMVAE_ZINB, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.K = K
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2

        self.onehot = torch.nn.functional.one_hot(
            torch.arange(0, self.K), num_classes=self.K
        ).to(self.device) * 1.0

        # pi network
        self.pi1 = nn.Linear(self.input_dim, self.h_dim)
        self.pi2 = nn.Linear(self.h_dim, self.K)

        # encoder mu path
        self.mu_x1 = nn.Linear(self.input_dim + self.K, self.h_dim1)
        self.mu_x2 = nn.Linear(self.h_dim1, self.h_dim2)
        self.mu_x3 = nn.Linear(self.h_dim2, self.z_dim)

        # encoder logvar path
        self.logvar_x1 = nn.Linear(self.input_dim + self.K, self.h_dim1)
        self.logvar_x2 = nn.Linear(self.h_dim1, self.h_dim2)
        self.logvar_x3 = nn.Linear(self.h_dim2, self.z_dim)

        # mixture component parameters
        self.mu_w1 = nn.Linear(self.K, self.h_dim)
        self.mu_w2 = nn.Linear(self.h_dim, self.z_dim)
        self.logvar_w1 = nn.Linear(self.K, self.h_dim)
        self.logvar_w2 = nn.Linear(self.h_dim, self.z_dim)

        # decoder layers
        self.recon1 = nn.Linear(self.z_dim, self.h_dim2)
        self.recon2z = nn.Linear(self.h_dim2, self.h_dim1)
        self.recon3z = nn.Linear(self.h_dim1, self.input_dim)
        self.recon2m = nn.Linear(self.h_dim2, self.h_dim1)
        self.recon3m = nn.Linear(self.h_dim1, self.input_dim)
        self.recon2d = nn.Linear(self.h_dim2, self.h_dim1)
        self.recon3d = nn.Linear(self.h_dim1, self.input_dim)

    def forward(self, x):
        # simplified forward for profiling
        batchsize = x.size(0)
        x = x.to(self.device)

        # pi network
        h = torch.relu(self.pi1(x))
        pi_x = torch.softmax(self.pi2(h), dim=1)

        # encoder (one component for simplicity)
        xy = torch.cat((x, self.onehot[0, :].expand(x.size(0), self.K)), 1)
        h1_mu = torch.relu(self.mu_x1(xy))
        h2_mu = torch.relu(self.mu_x2(h1_mu))
        mu_z = self.mu_x3(h2_mu)

        # decoder
        h = torch.relu(self.recon1(mu_z))
        hz = torch.relu(self.recon2z(h))
        recon = self.recon3z(hz)

        return recon


class SimpleClassifier(nn.Module):
    """Simple classifier for baseline comparison"""
    def __init__(self, z_dim=64, n_classes=2):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(z_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def create_baseline_gmvae_model(
    input_dim=1000,
    K=8,
    z_dim=64,
    device='cuda'
):
    """Create baseline GMVAE model for profiling"""
    model = BaselineGMVAE_ZINB(
        input_dim=input_dim,
        K=K,
        z_dim=z_dim,
        device=device
    ).to(device)

    return model


def create_baseline_classifier_model(z_dim=64, n_classes=2, device='cuda'):
    """Create baseline classifier"""
    model = SimpleClassifier(
        z_dim=z_dim,
        n_classes=n_classes
    ).to(device)

    return model


def profile_baseline_gmvae(
    input_dim=1000,
    K=8,
    z_dim=64,
    n_classes=2,
    batch_size=512,
    num_training_samples=834000,
    gmvae_epochs=100,
    classifier_epochs=50,
    output_dir="cost_profiles"
):
    """
    Profile baseline GMVAE costs (no enhancements)

    Args:
        input_dim: Number of genes
        K: Number of mixture components
        z_dim: Latent dimension
        n_classes: Number of patient classes
        batch_size: Training batch size
        num_training_samples: Total cells in dataset
        gmvae_epochs: GMVAE training epochs
        classifier_epochs: Classifier training epochs
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling baseline GMVAE on {device}")
    print(f"input: {input_dim} genes, K={K}, z_dim={z_dim}")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    # profile GMVAE
    print("=" * 60)
    print("profiling baseline GMVAE (step 1/2)")
    print("=" * 60)

    gmvae_model = create_baseline_gmvae_model(
        input_dim=input_dim,
        K=K,
        z_dim=z_dim,
        device=device
    )

    gmvae_profile = profiler.profile_model(
        model=gmvae_model,
        input_shape=(input_dim,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=gmvae_epochs,
        use_deepspeed=False
    )

    gmvae_profile['model_name'] = 'Baseline_GMVAE_step1'
    gmvae_profile['hyperparameters'] = {
        'input_dim': input_dim,
        'K': K,
        'z_dim': z_dim,
        'batch_size': batch_size,
        'epochs': gmvae_epochs
    }

    profiler.save_profile(gmvae_profile, f"{output_dir}/baseline_gmvae_step1.json")

    print(f"\nbaseline GMVAE training cost: ${gmvae_profile['training_cost_usd']:.2f}")
    print(f"baseline GMVAE training time: {gmvae_profile['estimated_training_time_hours']:.2f} hours")
    print()

    # profile classifier
    print("=" * 60)
    print("profiling baseline classifier (step 2/2)")
    print("=" * 60)

    classifier_model = create_baseline_classifier_model(
        z_dim=z_dim,
        n_classes=n_classes,
        device=device
    )

    classifier_profile = profiler.profile_model(
        model=classifier_model,
        input_shape=(z_dim,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=classifier_epochs,
        use_deepspeed=False
    )

    classifier_profile['model_name'] = 'Baseline_GMVAE_step2'
    classifier_profile['hyperparameters'] = {
        'z_dim': z_dim,
        'n_classes': n_classes,
        'batch_size': batch_size,
        'epochs': classifier_epochs
    }

    profiler.save_profile(classifier_profile, f"{output_dir}/baseline_gmvae_step2.json")

    print(f"\nbaseline classifier training cost: ${classifier_profile['training_cost_usd']:.2f}")
    print(f"baseline classifier training time: {classifier_profile['estimated_training_time_hours']:.2f} hours")
    print()

    # combined profile
    print("=" * 60)
    print("baseline GMVAE total pipeline")
    print("=" * 60)

    total_profile = {
        'model_name': 'Baseline_GMVAE_full_pipeline',
        'training_cost_usd': gmvae_profile['training_cost_usd'] + classifier_profile['training_cost_usd'],
        'training_time_hours': gmvae_profile['estimated_training_time_hours'] + classifier_profile['estimated_training_time_hours'],
        'inference_time_ms': gmvae_profile['measured_inference_time_ms'] + classifier_profile['measured_inference_time_ms'],
        'total_flops': gmvae_profile['total_training_flops'] + classifier_profile['total_training_flops'],
        'step1_gmvae': gmvae_profile,
        'step2_classifier': classifier_profile,
        'hyperparameters': {
            'input_dim': input_dim,
            'K': K,
            'z_dim': z_dim,
            'n_classes': n_classes,
            'gmvae_epochs': gmvae_epochs,
            'classifier_epochs': classifier_epochs,
            'batch_size': batch_size
        }
    }

    profiler.save_profile(total_profile, f"{output_dir}/baseline_gmvae_full.json")

    print(f"total training cost: ${total_profile['training_cost_usd']:.2f}")
    print(f"total training time: {total_profile['training_time_hours']:.2f} hours")
    print(f"total inference time: {total_profile['inference_time_ms']:.2f} ms/batch")
    print()

    return total_profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile baseline GMVAE costs')
    parser.add_argument('--input_dim', type=int, default=1000, help='number of genes')
    parser.add_argument('--K', type=int, default=8, help='number of mixture components')
    parser.add_argument('--z_dim', type=int, default=64, help='latent dimension')
    parser.add_argument('--n_classes', type=int, default=2, help='number of patient classes')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--gmvae_epochs', type=int, default=100, help='GMVAE epochs')
    parser.add_argument('--classifier_epochs', type=int, default=50, help='classifier epochs')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_baseline_gmvae(
        input_dim=args.input_dim,
        K=args.K,
        z_dim=args.z_dim,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        gmvae_epochs=args.gmvae_epochs,
        classifier_epochs=args.classifier_epochs,
        output_dir=args.output_dir
    )

    print("profiling complete")
