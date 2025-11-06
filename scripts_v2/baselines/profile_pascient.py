"""
Cost profiling for PaSCient baseline
Hierarchical attention model from Genentech (Lotfollahi et al. 2024)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from utils.cost_profiler import CostProfiler


class PaSCientModel(nn.Module):
    """
    Simplified PaSCient architecture for profiling
    gene2cell -> cell2cell -> aggregation -> patient_encoder -> predictor
    """
    def __init__(
        self,
        n_genes=1000,
        gene2cell_dim=128,
        cell2cell_dim=128,
        patient_dim=64,
        n_classes=2,
        n_layers_gene2cell=2,
        n_layers_cell2cell=2,
        n_layers_patient=2
    ):
        super(PaSCientModel, self).__init__()

        # gene to cell encoder (MLP)
        layers = []
        layers.append(nn.Linear(n_genes, gene2cell_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers_gene2cell - 1):
            layers.append(nn.Linear(gene2cell_dim, gene2cell_dim))
            layers.append(nn.GELU())
        self.gene2cell_encoder = nn.Sequential(*layers)

        # cell to cell encoder (attention-based, simplified as MLP for profiling)
        layers = []
        layers.append(nn.Linear(gene2cell_dim, cell2cell_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers_cell2cell - 1):
            layers.append(nn.Linear(cell2cell_dim, cell2cell_dim))
            layers.append(nn.GELU())
        self.cell2cell_encoder = nn.Sequential(*layers)

        # aggregation (mean pooling)
        self.aggregation = nn.Identity()  # placeholder

        # patient encoder
        layers = []
        layers.append(nn.Linear(cell2cell_dim, patient_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers_patient - 1):
            layers.append(nn.Linear(patient_dim, patient_dim))
            layers.append(nn.GELU())
        self.patient_encoder = nn.Sequential(*layers)

        # predictor
        self.predictor = nn.Linear(patient_dim, n_classes)

    def forward(self, x):
        """
        x: (batch, n_cells, n_genes)
        """
        batch_size, n_cells, n_genes = x.shape

        # reshape to (batch * n_cells, n_genes)
        x = x.reshape(-1, n_genes)

        # gene to cell
        cell_embeds = self.gene2cell_encoder(x)

        # cell to cell
        cell_cross_embeds = self.cell2cell_encoder(cell_embeds)

        # reshape back to (batch, n_cells, dim)
        cell_cross_embeds = cell_cross_embeds.reshape(batch_size, n_cells, -1)

        # aggregate cells to patient (mean pooling)
        patient_embeds = cell_cross_embeds.mean(dim=1)

        # patient encoder
        patient_embeds = self.patient_encoder(patient_embeds)

        # predict
        preds = self.predictor(patient_embeds)

        return preds


def create_pascient_model(
    n_genes=1000,
    gene2cell_dim=128,
    cell2cell_dim=128,
    patient_dim=64,
    n_classes=2,
    n_layers_gene2cell=2,
    n_layers_cell2cell=2,
    n_layers_patient=2,
    device='cuda'
):
    """Create PaSCient model for profiling"""
    model = PaSCientModel(
        n_genes=n_genes,
        gene2cell_dim=gene2cell_dim,
        cell2cell_dim=cell2cell_dim,
        patient_dim=patient_dim,
        n_classes=n_classes,
        n_layers_gene2cell=n_layers_gene2cell,
        n_layers_cell2cell=n_layers_cell2cell,
        n_layers_patient=n_layers_patient
    ).to(device)

    return model


def profile_pascient(
    n_genes=1000,
    gene2cell_dim=128,
    cell2cell_dim=128,
    patient_dim=64,
    n_classes=2,
    n_layers_gene2cell=2,
    n_layers_cell2cell=2,
    n_layers_patient=2,
    batch_size=16,
    num_training_samples=834000,
    epochs=100,
    avg_cells_per_patient=500,
    output_dir="cost_profiles"
):
    """
    Profile PaSCient costs

    Args:
        n_genes: Number of genes
        gene2cell_dim: Gene to cell encoder dimension
        cell2cell_dim: Cell to cell encoder dimension
        patient_dim: Patient encoder dimension
        n_classes: Number of output classes
        n_layers_gene2cell: Layers in gene2cell encoder
        n_layers_cell2cell: Layers in cell2cell encoder
        n_layers_patient: Layers in patient encoder
        batch_size: Training batch size
        num_training_samples: Total cells in dataset
        epochs: Training epochs
        avg_cells_per_patient: Average cells per patient
        output_dir: Directory to save profiles
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"profiling PaSCient on {device}")
    print(f"input: {n_genes} genes")
    print(f"architecture: gene2cell={gene2cell_dim}, cell2cell={cell2cell_dim}, patient={patient_dim}")
    print(f"layers: gene2cell={n_layers_gene2cell}, cell2cell={n_layers_cell2cell}, patient={n_layers_patient}")
    print(f"dataset: {num_training_samples} cells, batch size {batch_size}, avg cells/patient {avg_cells_per_patient}")
    print()

    profiler = CostProfiler()
    os.makedirs(output_dir, exist_ok=True)

    model = create_pascient_model(
        n_genes=n_genes,
        gene2cell_dim=gene2cell_dim,
        cell2cell_dim=cell2cell_dim,
        patient_dim=patient_dim,
        n_classes=n_classes,
        n_layers_gene2cell=n_layers_gene2cell,
        n_layers_cell2cell=n_layers_cell2cell,
        n_layers_patient=n_layers_patient,
        device=device
    )

    # PaSCient operates on (batch, n_cells, n_genes)
    profile = profiler.profile_model(
        model=model,
        input_shape=(avg_cells_per_patient, n_genes),
        batch_size=batch_size,
        num_training_samples=num_training_samples,
        num_epochs=epochs,
        use_deepspeed=False
    )

    profile['model_name'] = 'PaSCient'
    profile['hyperparameters'] = {
        'n_genes': n_genes,
        'gene2cell_dim': gene2cell_dim,
        'cell2cell_dim': cell2cell_dim,
        'patient_dim': patient_dim,
        'n_classes': n_classes,
        'n_layers_gene2cell': n_layers_gene2cell,
        'n_layers_cell2cell': n_layers_cell2cell,
        'n_layers_patient': n_layers_patient,
        'batch_size': batch_size,
        'epochs': epochs,
        'avg_cells_per_patient': avg_cells_per_patient
    }

    profiler.save_profile(profile, f"{output_dir}/pascient.json")

    print(f"training cost: ${profile['training_cost_usd']:.2f}")
    print(f"training time: {profile['estimated_training_time_hours']:.2f} hours")
    print(f"inference time: {profile['measured_inference_time_ms']:.2f} ms/batch")
    print()

    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='profile PaSCient costs')
    parser.add_argument('--n_genes', type=int, default=1000, help='number of genes')
    parser.add_argument('--gene2cell_dim', type=int, default=128, help='gene2cell dimension')
    parser.add_argument('--cell2cell_dim', type=int, default=128, help='cell2cell dimension')
    parser.add_argument('--patient_dim', type=int, default=64, help='patient dimension')
    parser.add_argument('--n_classes', type=int, default=2, help='number of output classes')
    parser.add_argument('--n_layers_gene2cell', type=int, default=2, help='gene2cell layers')
    parser.add_argument('--n_layers_cell2cell', type=int, default=2, help='cell2cell layers')
    parser.add_argument('--n_layers_patient', type=int, default=2, help='patient layers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_samples', type=int, default=834000, help='number of training samples')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--avg_cells_per_patient', type=int, default=500, help='avg cells per patient')
    parser.add_argument('--output_dir', type=str, default='cost_profiles', help='output directory')

    args = parser.parse_args()

    profile = profile_pascient(
        n_genes=args.n_genes,
        gene2cell_dim=args.gene2cell_dim,
        cell2cell_dim=args.cell2cell_dim,
        patient_dim=args.patient_dim,
        n_classes=args.n_classes,
        n_layers_gene2cell=args.n_layers_gene2cell,
        n_layers_cell2cell=args.n_layers_cell2cell,
        n_layers_patient=args.n_layers_patient,
        batch_size=args.batch_size,
        num_training_samples=args.num_samples,
        epochs=args.epochs,
        avg_cells_per_patient=args.avg_cells_per_patient,
        output_dir=args.output_dir
    )

    print("profiling complete")
