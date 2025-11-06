#!/usr/bin/env python3
"""
Generate pseudobulk and ground truth modalities from scRNA-seq.

Creates:
1. Pseudobulk expression (aggregate counts per patient)
2. Cell type proportions (ground truth)
3. Cell state distributions (per-cell-type embeddings)
4. Cell-cell communication scores (ligand-receptor interactions)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def aggregate_to_pseudobulk(adata: sc.AnnData, patient_col: str = 'ind_cov') -> pd.DataFrame:
    """
    Aggregate cell counts to pseudobulk per patient.

    Args:
        adata: AnnData object with cells
        patient_col: Column name for patient IDs

    Returns:
        DataFrame: [patients x genes] pseudobulk expression
    """
    print("Generating pseudobulk expression...")

    patients = sorted(adata.obs[patient_col].unique())
    genes = adata.var_names.tolist()

    pseudobulk = np.zeros((len(patients), len(genes)))

    for i, patient in enumerate(patients):
        mask = adata.obs[patient_col] == patient
        patient_cells = adata[mask, :]

        # Sum counts across cells (pseudobulk)
        if sp.issparse(patient_cells.X):
            pseudobulk[i] = patient_cells.X.sum(axis=0).A1
        else:
            pseudobulk[i] = patient_cells.X.sum(axis=0)

    pseudobulk_df = pd.DataFrame(
        pseudobulk,
        index=patients,
        columns=genes
    )

    print(f"  Generated pseudobulk: {pseudobulk_df.shape[0]} patients x {pseudobulk_df.shape[1]} genes")

    return pseudobulk_df


def compute_cell_proportions(adata: sc.AnnData, patient_col: str = 'ind_cov', ct_col: str = 'ct_cov') -> pd.DataFrame:
    """
    Compute cell type proportions per patient.

    Args:
        adata: AnnData object
        patient_col: Patient ID column
        ct_col: Cell type column

    Returns:
        DataFrame: [patients x cell_types] proportions (sum to 1 per row)
    """
    print("Computing cell type proportions...")

    patients = sorted(adata.obs[patient_col].unique())
    cell_types = sorted(adata.obs[ct_col].unique())

    proportions = np.zeros((len(patients), len(cell_types)))

    for i, patient in enumerate(patients):
        patient_cells = adata.obs[adata.obs[patient_col] == patient]
        total_cells = len(patient_cells)

        for j, ct in enumerate(cell_types):
            n_cells = (patient_cells[ct_col] == ct).sum()
            proportions[i, j] = n_cells / total_cells

    props_df = pd.DataFrame(
        proportions,
        index=patients,
        columns=cell_types
    )

    print(f"  Computed proportions: {props_df.shape[0]} patients x {props_df.shape[1]} cell types")

    return props_df


def compute_cell_state_distributions(
    adata: sc.AnnData,
    patient_col: str = 'ind_cov',
    ct_col: str = 'ct_cov',
    n_hvg: int = 500,
    state_dim: int = 64
) -> pd.DataFrame:
    """
    Compute cell state distributions per patient per cell type.

    For each patient and cell type:
    - Take mean expression of top variable genes
    - This captures cell state (e.g., activated vs resting)

    Args:
        adata: AnnData object
        patient_col: Patient ID column
        ct_col: Cell type column
        n_hvg: Number of highly variable genes to use
        state_dim: Output dimension (will be n_cell_types * state_dim)

    Returns:
        DataFrame: [patients x (cell_types * state_dim)]
    """
    print("Computing cell state distributions...")

    # Select highly variable genes for state representation
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', subset=False)
    hvg_mask = adata.var['highly_variable'].values
    hvg_genes = adata.var_names[hvg_mask].tolist()

    patients = sorted(adata.obs[patient_col].unique())
    cell_types = sorted(adata.obs[ct_col].unique())

    n_cell_types = len(cell_types)
    n_features = min(state_dim, len(hvg_genes))

    # Initialize states: [patients x (cell_types * n_features)]
    states = np.zeros((len(patients), n_cell_types * n_features))

    for i, patient in enumerate(patients):
        patient_mask = adata.obs[patient_col] == patient

        for j, ct in enumerate(cell_types):
            ct_mask = patient_mask & (adata.obs[ct_col] == ct)

            if ct_mask.sum() > 0:
                # Get cells of this type for this patient
                ct_cells = adata[ct_mask, hvg_mask]

                # Compute mean expression across top HVG
                if sp.issparse(ct_cells.X):
                    mean_expr = ct_cells.X.mean(axis=0).A1[:n_features]
                else:
                    mean_expr = ct_cells.X.mean(axis=0)[:n_features]

                # Store in flattened format
                start_idx = j * n_features
                end_idx = start_idx + n_features
                states[i, start_idx:end_idx] = mean_expr

    # Column names: celltype_feature0, celltype_feature1, ...
    columns = []
    for ct in cell_types:
        for f in range(n_features):
            columns.append(f"{ct}_state{f}")

    states_df = pd.DataFrame(
        states,
        index=patients,
        columns=columns
    )

    print(f"  Computed states: {states_df.shape[0]} patients x {states_df.shape[1]} features")

    return states_df


def compute_ligand_receptor_scores(
    adata: sc.AnnData,
    patient_col: str = 'ind_cov',
    ct_col: str = 'ct_cov',
    lr_pairs: list = None
) -> pd.DataFrame:
    """
    Compute cell-cell communication scores via ligand-receptor interactions.

    For each L-R pair:
    - Score = mean(ligand expression in sender) * mean(receptor expression in receiver)
    - Aggregate across cell type pairs per patient

    Args:
        adata: AnnData object
        patient_col: Patient ID column
        ct_col: Cell type column
        lr_pairs: List of (ligand, receptor, sender_ct, receiver_ct) tuples

    Returns:
        DataFrame: [patients x interactions]
    """
    print("Computing ligand-receptor communication scores...")

    # Default L-R pairs if none provided (common immune interactions)
    if lr_pairs is None:
        lr_pairs = get_default_lr_pairs()

    patients = sorted(adata.obs[patient_col].unique())
    all_genes = adata.var_names.tolist()

    # Filter pairs to genes present in dataset
    valid_pairs = []
    for ligand, receptor, sender, receiver in lr_pairs:
        if ligand in all_genes and receptor in all_genes:
            valid_pairs.append((ligand, receptor, sender, receiver))

    if len(valid_pairs) == 0:
        print("  WARNING: No valid L-R pairs found in dataset")
        # Return dummy scores
        return pd.DataFrame(
            np.zeros((len(patients), 1)),
            index=patients,
            columns=['dummy_interaction']
        )

    n_interactions = len(valid_pairs)
    scores = np.zeros((len(patients), n_interactions))

    for i, patient in enumerate(patients):
        patient_mask = adata.obs[patient_col] == patient

        for j, (ligand, receptor, sender_ct, receiver_ct) in enumerate(valid_pairs):
            # Get sender cells
            sender_mask = patient_mask & (adata.obs[ct_col] == sender_ct)
            if sender_mask.sum() == 0:
                continue

            sender_cells = adata[sender_mask, :]
            ligand_expr = sender_cells[:, ligand].X
            if sp.issparse(ligand_expr):
                ligand_mean = ligand_expr.mean()
            else:
                ligand_mean = ligand_expr.mean()

            # Get receiver cells
            receiver_mask = patient_mask & (adata.obs[ct_col] == receiver_ct)
            if receiver_mask.sum() == 0:
                continue

            receiver_cells = adata[receiver_mask, :]
            receptor_expr = receiver_cells[:, receptor].X
            if sp.issparse(receptor_expr):
                receptor_mean = receptor_expr.mean()
            else:
                receptor_mean = receptor_expr.mean()

            # Interaction score: ligand * receptor
            scores[i, j] = ligand_mean * receptor_mean

    columns = [f"{l}_{r}_{s}_{rec}" for l, r, s, rec in valid_pairs]
    scores_df = pd.DataFrame(
        scores,
        index=patients,
        columns=columns
    )

    print(f"  Computed communication: {scores_df.shape[0]} patients x {scores_df.shape[1]} interactions")

    return scores_df


def get_default_lr_pairs() -> list:
    """
    Return default ligand-receptor pairs for immune cells.
    Format: (ligand, receptor, sender_cell_type, receiver_cell_type)
    """
    # Common immune L-R interactions (case-sensitive gene names)
    pairs = [
        # T cell - B cell interactions
        ('CD40LG', 'CD40', 'CD4 T', 'B'),
        ('IL21', 'IL21R', 'CD4 T', 'B'),

        # T cell - Monocyte interactions
        ('IFNG', 'IFNGR1', 'CD8 T', 'Mono'),
        ('TNF', 'TNFRSF1A', 'CD4 T', 'Mono'),

        # Monocyte - T cell interactions
        ('IL12B', 'IL12RB1', 'Mono', 'CD4 T'),
        ('IL1B', 'IL1R1', 'Mono', 'CD4 T'),

        # NK - other interactions
        ('IFNG', 'IFNGR1', 'NK', 'Mono'),
        ('GZMB', 'IGFBP7', 'NK', 'CD8 T'),

        # Costimulatory signals
        ('CD86', 'CD28', 'Mono', 'CD4 T'),
        ('CD80', 'CD28', 'Mono', 'CD8 T'),
    ]

    return pairs


def extract_patient_labels(adata: sc.AnnData, patient_col: str = 'ind_cov', label_col: str = 'disease_cov') -> pd.Series:
    """Extract patient-level labels"""
    patients = sorted(adata.obs[patient_col].unique())
    labels = {}

    for patient in patients:
        patient_labels = adata.obs[adata.obs[patient_col] == patient][label_col].unique()
        assert len(patient_labels) == 1, f"Patient {patient} has multiple labels"
        labels[patient] = patient_labels[0]

    return pd.Series(labels, name='label')


def main():
    parser = argparse.ArgumentParser(description='Generate pseudobulk and ground truth modalities')
    parser.add_argument('--input', type=str, required=True, help='Input h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--patient_col', type=str, default='ind_cov', help='Patient ID column')
    parser.add_argument('--ct_col', type=str, default='ct_cov', help='Cell type column')
    parser.add_argument('--label_col', type=str, default='disease_cov', help='Label column')
    parser.add_argument('--state_dim', type=int, default=64, help='State dimension per cell type')
    parser.add_argument('--n_hvg', type=int, default=500, help='Number of HVG for states')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSEUDOBULK AND MODALITY GENERATION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load data
    print("\nLoading scRNA-seq data...")
    adata = sc.read_h5ad(args.input)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Generate all modalities
    pseudobulk = aggregate_to_pseudobulk(adata, args.patient_col)
    proportions = compute_cell_proportions(adata, args.patient_col, args.ct_col)
    states = compute_cell_state_distributions(adata, args.patient_col, args.ct_col, args.n_hvg, args.state_dim)
    communication = compute_ligand_receptor_scores(adata, args.patient_col, args.ct_col)
    labels = extract_patient_labels(adata, args.patient_col, args.label_col)

    # Save all
    print("\nSaving outputs...")
    pseudobulk.to_csv(output_dir / 'pseudobulk.csv')
    proportions.to_csv(output_dir / 'proportions.csv')
    states.to_csv(output_dir / 'states.csv')
    communication.to_csv(output_dir / 'communication.csv')
    labels.to_csv(output_dir / 'labels.csv')

    # Save metadata
    metadata = {
        'n_patients': len(pseudobulk),
        'n_genes': pseudobulk.shape[1],
        'n_cell_types': proportions.shape[1],
        'cell_types': proportions.columns.tolist(),
        'state_dim': args.state_dim,
        'n_states_features': states.shape[1],
        'n_interactions': communication.shape[1],
        'n_classes': len(labels.unique()),
        'classes': labels.unique().tolist()
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nGenerated files:")
    print(f"  pseudobulk.csv: {pseudobulk.shape}")
    print(f"  proportions.csv: {proportions.shape}")
    print(f"  states.csv: {states.shape}")
    print(f"  communication.csv: {communication.shape}")
    print(f"  labels.csv: {len(labels)}")
    print(f"  metadata.json")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
