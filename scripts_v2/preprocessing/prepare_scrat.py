"""
Preprocessing for ScRAT (transformer-based).

Input format:
- Sequence of gene expression vectors per patient
- Each cell is a token in the sequence
- Padding/truncation to fixed max_seq_length
- Gene expression tokenized/discretized

Output:
- processed_data/scrat/
  - sequences_train.npy, sequences_test.npy (tokenized sequences)
  - lengths_train.npy, lengths_test.npy (original sequence lengths)
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

def tokenize_expression(expr, n_bins=100, log_transform=True):
    """
    Tokenize gene expression values into discrete bins.

    Args:
        expr: Expression matrix (cells x genes)
        n_bins: Number of bins for discretization
        log_transform: Whether to log1p transform before binning

    Returns:
        Tokenized expression matrix (cells x genes)
    """
    if sparse.issparse(expr):
        expr = expr.toarray()

    if log_transform:
        expr = np.log1p(expr)

    # Discretize each gene independently
    tokenized = np.zeros_like(expr, dtype=np.int32)

    for g in range(expr.shape[1]):
        gene_expr = expr[:, g]
        # Use percentile-based binning
        bins = np.percentile(gene_expr, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicate bin edges
        tokenized[:, g] = np.digitize(gene_expr, bins[1:-1])

    return tokenized


def prepare_scrat_data(
    input_h5ad: str,
    output_dir: str,
    task: str = "disease",
    test_size: float = 0.2,
    random_state: int = 42,
    max_seq_length: int = 1024,
    min_cells_per_gene: int = 5,
    n_expression_bins: int = 100,
):
    """
    Prepare data for ScRAT (transformer) - barebones filtering and formatting only.

    Args:
        input_h5ad: Path to processed h5ad file
        output_dir: Output directory
        task: "disease" or "population"
        test_size: Fraction for test split
        random_state: Random seed
        max_seq_length: Max cells per patient (pad/truncate)
        min_cells_per_gene: Filter genes with fewer cells
        n_expression_bins: Number of bins for expression tokenization
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    adata = sc.read_h5ad(input_h5ad)

    print("Filtering...")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    print("Organizing by patient...")

    genes = adata.var_names.tolist()
    patient_ids = sorted(set(adata.obs["ind_cov"]))

    sequences = []
    lengths = []
    y = []

    for ind in patient_ids:
        patient_mask = adata.obs["ind_cov"] == ind

        # Get label
        if task.lower() == "disease":
            label = list(set(adata.obs[patient_mask]["disease_cov"]))[0]
        elif task.lower() in ["population", "pop"]:
            label = list(set(adata.obs[patient_mask]["pop_cov"]))[0]

        y.append(label)

        # Get patient cells
        patient_data = adata.X[patient_mask]
        n_cells = patient_data.shape[0]
        lengths.append(min(n_cells, max_seq_length))

        # Tokenize expression
        tokenized = tokenize_expression(
            patient_data, n_bins=n_expression_bins, log_transform=True
        )

        # Pad or truncate to max_seq_length
        if n_cells > max_seq_length:
            # Randomly sample max_seq_length cells
            np.random.seed(random_state)
            indices = np.random.choice(n_cells, max_seq_length, replace=False)
            sequence = tokenized[indices]
        else:
            # Pad with zeros
            sequence = np.zeros((max_seq_length, tokenized.shape[1]), dtype=np.int32)
            sequence[:n_cells] = tokenized

        sequences.append(sequence)

    sequences = np.array(sequences)  # Shape: (n_patients, max_seq_length, n_genes)
    lengths = np.array(lengths)

    # Encode labels
    class_id = sorted(set(y))
    mapping = {c: idx for idx, c in enumerate(class_id)}
    y = np.array([mapping[c] for c in y])

    print(f"Total patients: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Classes: {class_id}")
    for i, cls in enumerate(class_id):
        print(f"  {cls}: {np.sum(y == i)} patients")

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(sequences)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    sequences_train = sequences[train_idx]
    sequences_test = sequences[test_idx]
    lengths_train = lengths[train_idx]
    lengths_test = lengths[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Train: {len(sequences_train)} patients")
    print(f"Test: {len(sequences_test)} patients")

    # Save data
    print("Saving...")
    np.save(os.path.join(output_dir, "sequences_train.npy"), sequences_train)
    np.save(os.path.join(output_dir, "sequences_test.npy"), sequences_test)
    np.save(os.path.join(output_dir, "lengths_train.npy"), lengths_train)
    np.save(os.path.join(output_dir, "lengths_test.npy"), lengths_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Metadata
    metadata = {
        "task": task,
        "n_genes": len(genes),
        "n_classes": len(class_id),
        "class_names": class_id,
        "gene_names": genes,
        "n_train": len(sequences_train),
        "n_test": len(sequences_test),
        "max_seq_length": max_seq_length,
        "n_expression_bins": n_expression_bins,
        "vocab_size": n_expression_bins + 1,  # +1 for padding token
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
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--n_bins", type=int, default=100)

    args = parser.parse_args()

    prepare_scrat_data(
        input_h5ad=args.input,
        output_dir=args.output,
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
        max_seq_length=args.max_seq_length,
        n_expression_bins=args.n_bins,
    )
