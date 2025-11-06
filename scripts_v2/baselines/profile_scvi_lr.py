"""
Cost profiling for scVI + Logistic Regression baseline
Two-step approach: scVI VAE for embeddings + simple logistic regression classifier
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from utils.cost_profiler import CostProfiler


class scVIModel(nn.Module):
    """
    Simplified scVI architecture for profiling
    Based on Lopez et al. 2018
    """
    def __init__(
        self,
        n_genes=1000,
        n_latent=10,
        n_hidden=128,
        n_layers=1
    ):
        super(scVIModel, self).__init__()

        # encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(n_genes, n_hidden))
        encoder_layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            encoder_layers.append(nn.Linear(n_hidden, n_hidden))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # latent space
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)

        # decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(n_latent, n_hidden))
        decoder_layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            decoder_layers.append(nn.Linear(n_hidden, n_hidden))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(n_hidden, n_genes))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        # reparameterize
        z = self.reparameterize(mu, logvar)

        # decode
        recon = self.decoder(z)

        return recon, mu, logvar


class LogisticRegressionClassifier(nn.Module):
    """Simple logistic regression on scVI embeddings"""
    def __init__(self, n_latent=10, n_classes=2):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(n_latent, n_classes)

    def forward(self, x):
        return self.linear(x)


def create_scvi_model(
    n_genes=1000,
    n_latent=10,
    n_hidden=128,
    n_layers=1,
    device='cuda'
):
    """Create scVI model for profiling"""
    model = scVIModel(
        n_genes=n_genes,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers
    ).to(device)

    return model


def create_lr_model(n_latent=10, n_classes=2, device='cuda'):
    """Create logistic regression classifier"""
    model = LogisticRegressionClassifier(
        n_latent=n_latent,
        n_classes=n_classes
    ).to(device)

    return model


def profile_scvi_lr(
    n_genes=1000,
    n_latent=10,
    n_hidden=128,
    n_layers=1,
    n_classes=2,
    batch_size=128,
    num_training_samples=834000,
    scvi_epochs=400,
    lr_epochs=50,
    output_dir="cost_profiles"
):
    """
    Profile scVI + Logistic Regression costs

    Args:
        n_genes: Number of genes
        n_latent: Latent dimension
        n_hidden: Hidden dimension
        n_layers: Number of layers
        n_classes: Number of output classes
        batch_size: Training batch size
        num_training_samples: Total cells in dataset
        scvi_epochs: scVI training epochs
        lr_epochs: LR training epochs
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling scVI + LR on {device}")
    print(f"input: {n_genes} genes")
    print(f"architecture: n_hidden={n_hidden}, n_latent={n_latent}, n_layers={n_layers}")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    # profile scVI
    print("=" * 60)
    print("profiling scVI (step 1/2)")
    print("=" * 60)

    scvi_model = create_scvi_model(
        n_genes=n_genes,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        device=device
    )

    scvi_profile = profiler.profile_model(
        model=scvi_model,
        input_shape=(n_genes,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=scvi_epochs,
        use_deepspeed=False
    )

    scvi_profile['model_name'] = 'scVI_step1_VAE'
    scvi_profile['hyperparameters'] = {
        'n_genes': n_genes,
        'n_latent': n_latent,
        'n_hidden': n_hidden,
        'n_layers': n_layers,
        'batch_size': batch_size,
        'epochs': scvi_epochs
    }

    profiler.save_profile(scvi_profile, f"{output_dir}/scvi_step1_vae.json")

    print(f"\nscVI training cost: ${scvi_profile['training_cost_usd']:.2f}")
    print(f"scVI training time: {scvi_profile['estimated_training_time_hours']:.2f} hours")
    print()

    # profile logistic regression
    print("=" * 60)
    print("profiling logistic regression (step 2/2)")
    print("=" * 60)

    lr_model = create_lr_model(
        n_latent=n_latent,
        n_classes=n_classes,
        device=device
    )

    lr_profile = profiler.profile_model(
        model=lr_model,
        input_shape=(n_latent,),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=lr_epochs,
        use_deepspeed=False
    )

    lr_profile['model_name'] = 'scVI_step2_LR'
    lr_profile['hyperparameters'] = {
        'n_latent': n_latent,
        'n_classes': n_classes,
        'batch_size': batch_size,
        'epochs': lr_epochs
    }

    profiler.save_profile(lr_profile, f"{output_dir}/scvi_step2_lr.json")

    print(f"\nLR training cost: ${lr_profile['training_cost_usd']:.2f}")
    print(f"LR training time: {lr_profile['estimated_training_time_hours']:.2f} hours")
    print()

    # combined profile
    print("=" * 60)
    print("scVI + LR total pipeline")
    print("=" * 60)

    total_profile = {
        'model_name': 'scVI_LR_full_pipeline',
        'training_cost_usd': scvi_profile['training_cost_usd'] + lr_profile['training_cost_usd'],
        'training_time_hours': scvi_profile['estimated_training_time_hours'] + lr_profile['estimated_training_time_hours'],
        'inference_time_ms': scvi_profile['measured_inference_time_ms'] + lr_profile['measured_inference_time_ms'],
        'total_flops': scvi_profile['total_training_flops'] + lr_profile['total_training_flops'],
        'step1_scvi': scvi_profile,
        'step2_lr': lr_profile,
        'hyperparameters': {
            'n_genes': n_genes,
            'n_latent': n_latent,
            'n_hidden': n_hidden,
            'n_layers': n_layers,
            'n_classes': n_classes,
            'scvi_epochs': scvi_epochs,
            'lr_epochs': lr_epochs,
            'batch_size': batch_size
        }
    }

    profiler.save_profile(total_profile, f"{output_dir}/scvi_lr_full.json")

    print(f"total training cost: ${total_profile['training_cost_usd']:.2f}")
    print(f"total training time: {total_profile['training_time_hours']:.2f} hours")
    print(f"total inference time: {total_profile['inference_time_ms']:.2f} ms/batch")
    print()

    return total_profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile scVI + LR costs')
    parser.add_argument('--n_genes', type=int, default=1000, help='number of genes')
    parser.add_argument('--n_latent', type=int, default=10, help='latent dimension')
    parser.add_argument('--n_hidden', type=int, default=128, help='hidden dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--n_classes', type=int, default=2, help='number of output classes')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--scvi_epochs', type=int, default=400, help='scVI epochs')
    parser.add_argument('--lr_epochs', type=int, default=50, help='LR epochs')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_scvi_lr(
        n_genes=args.n_genes,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        scvi_epochs=args.scvi_epochs,
        lr_epochs=args.lr_epochs,
        output_dir=args.output_dir
    )

    print("profiling complete")
