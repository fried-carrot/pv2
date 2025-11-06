#!/usr/bin/env python3
"""
Data preprocessing - barebones filtering and formatting only
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
from scipy.sparse import csr_matrix
import os
from pathlib import Path
import argparse
import json


def prepare_gmvae_data(input_h5ad, output_dir, subsample_fraction=None, min_cells_per_gene=5):
    """
    Prepare data for GMVAE training - barebones filtering and formatting only.

    Filtering (ProtoCell4P approach):
    - Filter genes with min_cells threshold
    - No normalization, scaling, or feature selection

    Args:
        input_h5ad: Path to input h5ad file
        output_dir: Output directory for processed files
        subsample_fraction: Optional fraction to subsample data
        min_cells_per_gene: Minimum cells expressing a gene (default: 5)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    if subsample_fraction is not None:
        n_cells = adata.shape[0]
        n_subsample = int(n_cells * subsample_fraction)
        np.random.seed(42)
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        adata = adata[indices, :]
        print(f"Subsampled to {subsample_fraction*100}%: {n_subsample} cells")

    # Basic quality filtering (ProtoCell4P approach)
    print(f"\nBefore filtering: {adata.shape[1]} genes")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    print(f"After filtering (min_cells={min_cells_per_gene}): {adata.shape[1]} genes")

    # Extract and format data
    X = adata.X
    X_transposed = X.T.tocsr()  # genes x cells for GMVAE
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)

    # Extract labels
    if "ct_cov" in adata.obs.columns:
        cell_types = adata.obs["ct_cov"]
    elif "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"]
    else:
        raise ValueError("No cell type column found")

    if "ind_cov" in adata.obs.columns:
        patient_ids = adata.obs["ind_cov"]
    else:
        raise ValueError("No patient ID column found")

    if "disease_cov" in adata.obs.columns:
        disease_labels = adata.obs["disease_cov"]
    else:
        raise ValueError("No disease label column found")

    # Create label mappings
    ct_id = sorted(set(cell_types))
    mapping_ct = {c: idx for idx, c in enumerate(ct_id)}
    cell_type_codes = [mapping_ct[ct] for ct in cell_types]

    disease_id = sorted(set(disease_labels))
    mapping_disease = {d: idx for idx, d in enumerate(disease_id)}
    disease_codes = [mapping_disease[d] for d in disease_labels]

    # Save labels
    labels_df = pd.DataFrame({
        'cluster': cell_type_codes,
        'patient_id': patient_ids,
        'disease': disease_codes
    })
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    # Save gene names
    genes = adata.var_names.tolist()
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for gene in genes:
            f.write(f"{gene}\n")

    # Save processed h5ad
    adata.write(os.path.join(output_dir, "processed_data.h5ad"))

    # Save metadata
    metadata = {
        'n_cells': X_transposed.shape[1],
        'n_genes': X_transposed.shape[0],
        'n_cell_types': len(ct_id),
        'cell_types': ct_id,
        'n_patients': len(set(patient_ids)),
        'n_diseases': len(disease_id),
        'diseases': disease_id,
        'disease_mapping': mapping_disease,
        'subsample_fraction': subsample_fraction
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: {X_transposed.shape[0]} genes x {X_transposed.shape[1]} cells")
    print(f"Cell types: {len(ct_id)}, Patients: {len(set(patient_ids))}")
    print(f"Output: {output_dir}")

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for GMVAE4P')
    parser.add_argument('--input', type=str, required=True, help='Input h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--subsample', type=float, default=None,
                       help='Subsample fraction (e.g., 0.1 for 10%)')
    parser.add_argument('--min_cells', type=int, default=5,
                       help='Minimum cells per gene (default: 5)')

    args = parser.parse_args()

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.subsample:
        print(f"Subsample: {args.subsample*100}%")
    print(f"Min cells per gene: {args.min_cells}")

    prepare_gmvae_data(
        args.input,
        args.output,
        subsample_fraction=args.subsample,
        min_cells_per_gene=args.min_cells
    )

    print("\nDone.")
