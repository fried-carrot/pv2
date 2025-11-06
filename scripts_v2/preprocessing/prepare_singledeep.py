"""
Preprocessing for singleDeep.

Input format:
- Per-patient aggregated expression profiles by cell type
- Concatenated cell type profiles for each patient
- Shape: (n_patients, n_genes * n_cell_types)

Output:
- processed_data/singledeep/
  - X_train.npy, X_test.npy (aggregated profiles)
  - y_train.npy, y_test.npy (patient labels)
  - metadata.json
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import json
from sklearn.model_selection import train_test_split
from scipy import sparse

def prepare_singledeep_data(
    input_h5ad: str,
    output_dir: str,
    task: str = "disease",
    test_size: float = 0.2,
    random_state: int = 42,
    min_cells_per_gene: int = 5,
    aggregation_method: str = "mean",
):
    """
    Prepare data for singleDeep - barebones filtering and formatting only.

    Args:
        input_h5ad: Path to processed h5ad file
        output_dir: Output directory
        task: "disease" or "population"
        test_size: Fraction for test split
        random_state: Random seed
        min_cells_per_gene: Filter genes with fewer cells
        aggregation_method: How to aggregate cells per cell type
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    adata = sc.read_h5ad(input_h5ad)

    print("Filtering...")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    print("Aggregating by patient and cell type...")

    genes = adata.var_names.tolist()
    cell_types = adata.obs["ct_cov"]
    ct_id = sorted(set(cell_types))
    n_cell_types = len(ct_id)
    n_genes = len(genes)

    patient_ids = sorted(set(adata.obs["ind_cov"]))
    X = []
    y = []

    for ind in patient_ids:
        patient_mask = adata.obs["ind_cov"] == ind

        # Get label
        if task.lower() == "disease":
            label = list(set(adata.obs[patient_mask]["disease_cov"]))[0]
        elif task.lower() in ["population", "pop"]:
            label = list(set(adata.obs[patient_mask]["pop_cov"]))[0]

        y.append(label)

        # Aggregate by cell type
        patient_profile = []
        for ct in ct_id:
            ct_mask = patient_mask & (adata.obs["ct_cov"] == ct)

            if np.sum(ct_mask) == 0:
                # No cells of this type for this patient - use zeros
                ct_profile = np.zeros(n_genes)
            else:
                ct_data = adata.X[ct_mask]

                # Convert sparse to dense if needed
                if sparse.issparse(ct_data):
                    ct_data = ct_data.toarray()

                # Aggregate
                if aggregation_method == "mean":
                    ct_profile = np.mean(ct_data, axis=0).flatten()
                elif aggregation_method == "sum":
                    ct_profile = np.sum(ct_data, axis=0).flatten()
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            patient_profile.extend(ct_profile)

        X.append(np.array(patient_profile))

    X = np.array(X)  # Shape: (n_patients, n_genes * n_cell_types)

    # Encode labels
    class_id = sorted(set(y))
    mapping = {c: idx for idx, c in enumerate(class_id)}
    y = np.array([mapping[c] for c in y])

    print(f"Total patients: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Classes: {class_id}")
    for i, cls in enumerate(class_id):
        print(f"  {cls}: {np.sum(y == i)} patients")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {len(X_train)} patients")
    print(f"Test: {len(X_test)} patients")

    # Save data
    print("Saving...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Metadata
    metadata = {
        "task": task,
        "n_genes": n_genes,
        "n_cell_types": n_cell_types,
        "feature_dim": n_genes * n_cell_types,
        "n_classes": len(class_id),
        "class_names": class_id,
        "cell_type_names": ct_id,
        "gene_names": genes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "aggregation_method": aggregation_method,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data saved to {output_dir}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--task", type=str, default="disease", help="disease or population")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--aggregation", type=str, default="mean", help="mean or sum")

    args = parser.parse_args()

    prepare_singledeep_data(
        input_h5ad=args.input,
        output_dir=args.output,
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
        aggregation_method=args.aggregation,
    )
