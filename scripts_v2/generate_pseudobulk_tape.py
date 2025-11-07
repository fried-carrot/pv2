#!/usr/bin/env python3
"""
TAPE-style pseudobulk generation with synthetic mixing.

Key differences from naive aggregation:
1. Random cell sampling (not all cells from a patient)
2. Dirichlet-distributed proportions (not true patient proportions)
3. Sparse cell types (randomly zero out cell types)
4. Multiple synthetic samples per patient with different mixtures

This prevents memorization and tests true deconvolution ability.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def generate_tape_pseudobulk(
    adata: sc.AnnData,
    patient_col: str,
    ct_col: str,
    n_samples_per_patient: int = 10,
    n_cells_per_sample: int = 500,
    sparse: bool = True,
    sparse_prob: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Generate TAPE-style pseudobulk with synthetic mixing.

    Args:
        adata: AnnData with single cells
        patient_col: Patient ID column
        ct_col: Cell type column
        n_samples_per_patient: Number of synthetic samples per patient
        n_cells_per_sample: Number of cells to sample per mixture
        sparse: Whether to randomly zero out cell types
        sparse_prob: Probability of zeroing cell types
        random_state: Random seed

    Returns:
        pseudobulk_df: [samples x genes]
        proportions_df: [samples x cell_types]
        sample_to_patient: [samples] mapping sample index to patient
    """
    print("Generating TAPE-style pseudobulk...")
    print(f"  {n_samples_per_patient} samples per patient")
    print(f"  {n_cells_per_sample} cells per sample")
    print(f"  Sparse cell types: {sparse} (prob={sparse_prob})")

    np.random.seed(random_state)

    patients = sorted(adata.obs[patient_col].unique())
    cell_types = sorted(adata.obs[ct_col].unique())
    genes = adata.var_names.tolist()

    n_patients = len(patients)
    n_cell_types = len(cell_types)
    n_total_samples = n_patients * n_samples_per_patient

    # Pre-allocate
    pseudobulk = np.zeros((n_total_samples, len(genes)), dtype=np.float32)
    proportions = np.zeros((n_total_samples, n_cell_types), dtype=np.float32)
    sample_to_patient = np.zeros(n_total_samples, dtype=int)

    # Group cells by patient and cell type
    print("Indexing cells by patient and cell type...")
    cell_indices = {}
    for patient in patients:
        cell_indices[patient] = {}
        patient_mask = adata.obs[patient_col] == patient

        for ct in cell_types:
            ct_mask = (adata.obs[ct_col] == ct) & patient_mask
            indices = np.where(ct_mask)[0]
            cell_indices[patient][ct] = indices

    # Generate synthetic mixtures
    print("Generating synthetic mixtures...")
    sample_idx = 0

    for patient_idx, patient in enumerate(tqdm(patients)):
        for sample_num in range(n_samples_per_patient):
            # Generate random proportions using Dirichlet
            props = np.random.dirichlet(np.ones(n_cell_types))

            # Apply sparsity (randomly zero out cell types)
            if sparse:
                n_zero = int(n_cell_types * sparse_prob)
                if n_zero > 0:
                    zero_idx = np.random.choice(n_cell_types, size=n_zero, replace=False)
                    props[zero_idx] = 0

            # Renormalize
            if props.sum() > 0:
                props = props / props.sum()
            else:
                # Edge case: all zero, uniform instead
                props = np.ones(n_cell_types) / n_cell_types

            # Calculate number of cells per type
            cell_counts = np.floor(n_cells_per_sample * props).astype(int)

            # Adjust to exactly n_cells_per_sample
            diff = n_cells_per_sample - cell_counts.sum()
            if diff > 0:
                # Add cells to non-zero types
                non_zero = np.where(props > 0)[0]
                add_idx = np.random.choice(non_zero, size=diff, replace=True)
                for idx in add_idx:
                    cell_counts[idx] += 1

            # Update proportions based on actual cell counts
            props = cell_counts / cell_counts.sum()

            # Sample cells and aggregate
            mixture = np.zeros(len(genes), dtype=np.float32)

            for ct_idx, ct in enumerate(cell_types):
                n_cells = cell_counts[ct_idx]
                if n_cells == 0:
                    continue

                available_cells = cell_indices[patient][ct]
                if len(available_cells) == 0:
                    continue

                # Sample with replacement
                sampled_idx = np.random.choice(available_cells, size=n_cells, replace=True)

                # Sum expression
                if sp.issparse(adata.X):
                    mixture += adata.X[sampled_idx].sum(axis=0).A1
                else:
                    mixture += adata.X[sampled_idx].sum(axis=0)

            # Store
            pseudobulk[sample_idx] = mixture
            proportions[sample_idx] = props
            sample_to_patient[sample_idx] = patient_idx
            sample_idx += 1

    # Create DataFrames
    sample_names = [f"{patients[patient_idx]}_syn{i}"
                   for patient_idx in range(n_patients)
                   for i in range(n_samples_per_patient)]

    pseudobulk_df = pd.DataFrame(pseudobulk, index=sample_names, columns=genes)
    proportions_df = pd.DataFrame(proportions, index=sample_names, columns=cell_types)

    print(f"  Generated: {len(sample_names)} synthetic samples")
    print(f"  Pseudobulk: {pseudobulk_df.shape}")
    print(f"  Proportions: {proportions_df.shape}")

    return pseudobulk_df, proportions_df, sample_to_patient


def compute_cell_states_tape(
    adata: sc.AnnData,
    sample_to_patient: np.ndarray,
    patient_col: str,
    ct_col: str,
    n_hvg: int = 500,
    state_dim: int = 64
) -> pd.DataFrame:
    """
    Compute cell state features for TAPE-style samples.
    Uses patient-level cell type HVG means (same for all synthetic samples of a patient).
    """
    print("Computing cell states...")

    patients = sorted(adata.obs[patient_col].unique())
    cell_types = sorted(adata.obs[ct_col].unique())
    n_samples = len(sample_to_patient)

    # Select HVGs
    print(f"  Selecting top {n_hvg} HVGs per cell type...")
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(adata_copy, flavor='seurat_v3', n_top_genes=n_hvg, subset=True)
    hvg_genes = adata_copy.var_names.tolist()

    # Compute patient-level cell type means
    print("  Computing patient x cell type HVG means...")
    patient_ct_means = {}

    for patient in patients:
        patient_ct_means[patient] = {}
        patient_mask = adata_copy.obs[patient_col] == patient

        for ct in cell_types:
            ct_mask = (adata_copy.obs[ct_col] == ct) & patient_mask
            cells = adata_copy[ct_mask]

            if cells.n_obs > 0:
                if sp.issparse(cells.X):
                    mean_expr = cells.X.mean(axis=0).A1
                else:
                    mean_expr = cells.X.mean(axis=0)
                patient_ct_means[patient][ct] = mean_expr
            else:
                patient_ct_means[patient][ct] = np.zeros(len(hvg_genes))

    # Assign states to synthetic samples based on parent patient
    states = np.zeros((n_samples, len(cell_types) * len(hvg_genes)))

    for sample_idx, patient_idx in enumerate(sample_to_patient):
        patient = patients[patient_idx]
        for ct_idx, ct in enumerate(cell_types):
            start = ct_idx * len(hvg_genes)
            end = start + len(hvg_genes)
            states[sample_idx, start:end] = patient_ct_means[patient][ct]

    # Apply z-score normalization per feature
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    states = scaler.fit_transform(states)

    # Create feature names: celltype_gene
    feature_names = []
    for ct in cell_types:
        for gene in hvg_genes:
            feature_names.append(f"{ct}_{gene}")

    # Samples as rows, features as columns
    sample_names = [f"sample_{i}" for i in range(n_samples)]
    states_df = pd.DataFrame(states, index=sample_names, columns=feature_names)

    print(f"  Cell states: {states_df.shape}")

    return states_df


def compute_communication_tape(
    adata: sc.AnnData,
    sample_to_patient: np.ndarray,
    patient_col: str,
    ct_col: str
) -> pd.DataFrame:
    """
    Compute L-R communication scores for TAPE samples.
    Uses patient-level averages (same for all synthetic samples of a patient).
    """
    print("Computing cell-cell communication...")

    # Immune L-R pairs
    lr_pairs = [
        ('CD40LG', 'CD40'),
        ('IFNG', 'IFNGR1'),
        ('IL2', 'IL2RA'),
        ('TNF', 'TNFRSF1A'),
        ('TGFB1', 'TGFBR1'),
        ('IL10', 'IL10RA'),
        ('IL6', 'IL6R'),
        ('CXCL12', 'CXCR4'),
        ('CCL5', 'CCR5'),
        ('IL1B', 'IL1R1')
    ]

    patients = sorted(adata.obs[patient_col].unique())
    cell_types = sorted(adata.obs[ct_col].unique())
    n_samples = len(sample_to_patient)

    # Get available L-R genes
    available_genes = set(adata.var_names)
    valid_pairs = [(l, r) for l, r in lr_pairs if l in available_genes and r in available_genes]

    print(f"  Using {len(valid_pairs)}/{len(lr_pairs)} L-R pairs")

    # Compute patient-level L-R scores
    patient_scores = {}

    for patient in patients:
        patient_mask = adata.obs[patient_col] == patient
        patient_cells = adata[patient_mask]

        scores = []
        for ligand, receptor in valid_pairs:
            l_idx = adata.var_names.tolist().index(ligand)
            r_idx = adata.var_names.tolist().index(receptor)

            # Average across all cell pairs
            if sp.issparse(patient_cells.X):
                l_expr = patient_cells.X[:, l_idx].toarray().flatten()
                r_expr = patient_cells.X[:, r_idx].toarray().flatten()
            else:
                l_expr = patient_cells.X[:, l_idx]
                r_expr = patient_cells.X[:, r_idx]

            # Interaction score: product of means
            score = l_expr.mean() * r_expr.mean()
            scores.append(score)

        patient_scores[patient] = np.array(scores)

    # Assign to synthetic samples
    comm = np.zeros((n_samples, len(valid_pairs)))

    for sample_idx, patient_idx in enumerate(sample_to_patient):
        patient = patients[patient_idx]
        comm[sample_idx] = patient_scores[patient]

    sample_names = [f"sample_{i}" for i in range(n_samples)]
    pair_names = [f"{l}_{r}" for l, r in valid_pairs]
    comm_df = pd.DataFrame(comm, index=sample_names, columns=pair_names)

    print(f"  Communication: {comm_df.shape}")

    return comm_df


def main():
    parser = argparse.ArgumentParser(description='Generate TAPE-style pseudobulk')
    parser.add_argument('--input', type=str, required=True, help='Input h5ad file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--patient_col', type=str, default='ind_cov', help='Patient ID column')
    parser.add_argument('--ct_col', type=str, default='ct_cov', help='Cell type column')
    parser.add_argument('--label_col', type=str, default='cov_scDRS_label', help='Label column')
    parser.add_argument('--n_samples_per_patient', type=int, default=10, help='Synthetic samples per patient')
    parser.add_argument('--n_cells_per_sample', type=int, default=500, help='Cells per synthetic sample')
    parser.add_argument('--sparse_prob', type=float, default=0.3, help='Probability of sparse cell types')
    parser.add_argument('--n_hvg', type=int, default=500, help='Number of HVGs for states')
    parser.add_argument('--state_dim', type=int, default=64, help='State dimension (not used, for compatibility)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TAPE-STYLE PSEUDOBULK GENERATION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Samples per patient: {args.n_samples_per_patient}")
    print(f"Cells per sample: {args.n_cells_per_sample}")
    print(f"Sparse probability: {args.sparse_prob}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(args.input)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Generate TAPE pseudobulk
    pseudobulk, proportions, sample_to_patient = generate_tape_pseudobulk(
        adata=adata,
        patient_col=args.patient_col,
        ct_col=args.ct_col,
        n_samples_per_patient=args.n_samples_per_patient,
        n_cells_per_sample=args.n_cells_per_sample,
        sparse=True,
        sparse_prob=args.sparse_prob,
        random_state=args.seed
    )

    # Generate states and communication (patient-level)
    states = compute_cell_states_tape(
        adata=adata,
        sample_to_patient=sample_to_patient,
        patient_col=args.patient_col,
        ct_col=args.ct_col,
        n_hvg=args.n_hvg,
        state_dim=args.state_dim
    )

    communication = compute_communication_tape(
        adata=adata,
        sample_to_patient=sample_to_patient,
        patient_col=args.patient_col,
        ct_col=args.ct_col
    )

    # Extract labels (map synthetic samples to patient labels)
    print("Extracting patient labels...")
    patients = sorted(adata.obs[args.patient_col].unique())
    patient_labels = {}
    for patient in patients:
        patient_mask = adata.obs[args.patient_col] == patient
        label = adata.obs.loc[patient_mask, args.label_col].iloc[0]
        patient_labels[patient] = label

    labels = pd.Series(
        [patient_labels[patients[patient_idx]] for patient_idx in sample_to_patient],
        index=pseudobulk.index,
        name='label'
    )

    print(f"  Labels: {len(labels)}, Classes: {labels.unique()}")

    # Save patient mapping for grouped CV
    patient_names = [patients[idx] for idx in sample_to_patient]
    patient_mapping = pd.DataFrame({
        'sample_id': pseudobulk.index,
        'patient_id': patient_names
    })

    # Save
    print("\nSaving outputs...")
    pseudobulk.to_csv(output_dir / 'pseudobulk.csv')
    proportions.to_csv(output_dir / 'proportions.csv')
    states.to_csv(output_dir / 'states.csv')
    communication.to_csv(output_dir / 'communication.csv')
    labels.to_csv(output_dir / 'labels.csv')
    patient_mapping.to_csv(output_dir / 'patient_mapping.csv', index=False)

    # Save metadata
    metadata = {
        'n_synthetic_samples': len(pseudobulk),
        'n_patients': len(patients),
        'n_samples_per_patient': args.n_samples_per_patient,
        'n_cells_per_sample': args.n_cells_per_sample,
        'n_genes': pseudobulk.shape[1],
        'n_cell_types': proportions.shape[1],
        'cell_types': proportions.columns.tolist(),
        'n_states_features': states.shape[1],
        'n_interactions': communication.shape[1],
        'n_classes': len(labels.unique()),
        'classes': labels.unique().tolist(),
        'sparse_prob': args.sparse_prob,
        'random_seed': args.seed
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nGenerated files:")
    print(f"  pseudobulk.csv: {pseudobulk.shape}")
    print(f"  proportions.csv: {proportions.shape}")
    print(f"  states.csv: {states.shape}")
    print(f"  communication.csv: {communication.shape}")
    print(f"  labels.csv: {len(labels)}")
    print(f"  patient_mapping.csv: {len(patient_mapping)} samples")
    print(f"  metadata.json")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
