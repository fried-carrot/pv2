"""
Preprocessing for PaSCient (hierarchical model).

Input format:
- Hierarchical structure: Patient -> Cell types -> Cells
- Per-patient data with cell type information
- Similar to ProtoCell4P but may require different aggregation

Output:
- processed_data/pascient/
  - X_train.pkl, X_test.pkl (list of sparse matrices per patient)
  - y_train.npy, y_test.npy (patient labels)
  - ct_train.pkl, ct_test.pkl (cell type labels)
  - metadata.json
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import pickle
import json
from sklearn.model_selection import train_test_split

def prepare_pascient_data(
    input_h5ad: str,
    output_dir: str,
    task: str = "disease",
    test_size: float = 0.2,
    random_state: int = 42,
    min_cells_per_gene: int = 5,
):
    """
    Prepare data for PaSCient - barebones filtering and formatting only.

    Args:
        input_h5ad: Path to processed h5ad file
        output_dir: Output directory
        task: "disease" or "population"
        test_size: Fraction for test split
        random_state: Random seed
        min_cells_per_gene: Filter genes with fewer cells
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    adata = sc.read_h5ad(input_h5ad)

    print("Filtering...")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    print("Organizing by patient...")
    X = []
    y = []
    ct = []

    genes = adata.var_names.tolist()
    cell_types = adata.obs["ct_cov"]

    ct_id = sorted(set(cell_types))
    mapping_ct = {c: idx for idx, c in enumerate(ct_id)}

    patient_ids = sorted(set(adata.obs["ind_cov"]))

    for ind in patient_ids:
        patient_mask = adata.obs["ind_cov"] == ind

        # Get label
        if task.lower() == "disease":
            label = list(set(adata.obs[patient_mask]["disease_cov"]))[0]
        elif task.lower() in ["population", "pop"]:
            label = list(set(adata.obs[patient_mask]["pop_cov"]))[0]

        y.append(label)

        # Get patient data
        x = adata.X[patient_mask]
        X.append(x)

        # Get cell type labels
        ct.append([mapping_ct[c] for c in cell_types[patient_mask]])

    # Encode labels
    class_id = sorted(set(y))
    mapping = {c: idx for idx, c in enumerate(class_id)}
    y = np.array([mapping[c] for c in y])

    print(f"Total patients: {len(X)}")
    print(f"Classes: {class_id}")
    for i, cls in enumerate(class_id):
        print(f"  {cls}: {np.sum(y == i)} patients")

    # Train/test split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    ct_train = [ct[i] for i in train_idx]
    ct_test = [ct[i] for i in test_idx]

    print(f"Train: {len(X_train)} patients")
    print(f"Test: {len(X_test)} patients")

    # Save data
    print("Saving...")

    with open(os.path.join(output_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(output_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)

    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    with open(os.path.join(output_dir, "ct_train.pkl"), "wb") as f:
        pickle.dump(ct_train, f)
    with open(os.path.join(output_dir, "ct_test.pkl"), "wb") as f:
        pickle.dump(ct_test, f)

    # Metadata
    metadata = {
        "task": task,
        "n_genes": len(genes),
        "n_cell_types": len(ct_id),
        "n_classes": len(class_id),
        "class_names": class_id,
        "cell_type_names": ct_id,
        "gene_names": genes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
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

    args = parser.parse_args()

    prepare_pascient_data(
        input_h5ad=args.input,
        output_dir=args.output,
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
    )
