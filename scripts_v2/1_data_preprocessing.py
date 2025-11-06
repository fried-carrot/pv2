#!/usr/bin/env python3
"""
data preprocessing
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
from scipy.sparse import csr_matrix
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def exploratory_data_analysis(adata, output_dir):
    """
    comprehensive EDA following experimental design requirements

    includes:
    - data description and quality metrics
    - missing value analysis
    - feature relationships and distributions
    - dimensional reduction (PCA)
    - batch effect analysis
    """
    eda_dir = os.path.join(output_dir, "eda")
    Path(eda_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. DATA DESCRIPTION
    print("\n1. DATA DESCRIPTION")
    print(f"Total cells: {adata.n_obs:,}")
    print(f"Total genes: {adata.n_vars:,}")

    # calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # QC statistics
    print(f"\nQuality Control Metrics:")
    print(f"Genes per cell - mean: {adata.obs['n_genes_by_counts'].mean():.0f}, median: {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"UMI counts per cell - mean: {adata.obs['total_counts'].mean():.0f}, median: {adata.obs['total_counts'].median():.0f}")
    print(f"Mitochondrial % - mean: {adata.obs['pct_counts_mt'].mean():.2f}%, median: {adata.obs['pct_counts_mt'].median():.2f}%")

    # cells failing QC thresholds
    low_gene_cells = (adata.obs['n_genes_by_counts'] < 200).sum()
    high_mt_cells = (adata.obs['pct_counts_mt'] > 20).sum()
    print(f"\nCells below QC thresholds:")
    print(f"Cells with <200 genes: {low_gene_cells:,} ({100*low_gene_cells/adata.n_obs:.2f}%)")
    print(f"Cells with >20% MT: {high_mt_cells:,} ({100*high_mt_cells/adata.n_obs:.2f}%)")

    # cell type distribution
    if 'ct_cov' in adata.obs.columns:
        ct_col = 'ct_cov'
    elif 'cell_type' in adata.obs.columns:
        ct_col = 'cell_type'
    else:
        ct_col = None

    if ct_col:
        print(f"\nCell type distribution:")
        ct_counts = adata.obs[ct_col].value_counts()
        for ct, count in ct_counts.items():
            print(f"  {ct}: {count:,} ({100*count/adata.n_obs:.2f}%)")

    # disease distribution
    if 'disease_cov' in adata.obs.columns:
        print(f"\nDisease distribution:")
        disease_counts = adata.obs['disease_cov'].value_counts()
        for disease, count in disease_counts.items():
            print(f"  {disease}: {count:,} ({100*count/adata.n_obs:.2f}%)")

    # patient distribution
    if 'ind_cov' in adata.obs.columns:
        n_patients = adata.obs['ind_cov'].nunique()
        print(f"\nNumber of unique patients: {n_patients}")
        print(f"Cells per patient - mean: {adata.n_obs/n_patients:.0f}")

    # 2. MISSING VALUE ANALYSIS
    print("\n2. MISSING VALUE ANALYSIS")
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    zero_rate = (X_dense == 0).sum() / X_dense.size
    print(f"Zero-inflation rate: {100*zero_rate:.2f}%")
    print(f"Non-zero entries: {100*(1-zero_rate):.2f}%")

    genes_zero_rate = (X_dense == 0).sum(axis=0) / X_dense.shape[0]
    print(f"Genes with >90% zeros: {(genes_zero_rate > 0.9).sum():,}")
    print(f"Genes with >95% zeros: {(genes_zero_rate > 0.95).sum():,}")
    print(f"Genes with >99% zeros: {(genes_zero_rate > 0.99).sum():,}")

    # 3. FEATURE RELATIONSHIPS AND DISTRIBUTIONS
    print("\n3. FEATURE RELATIONSHIPS")

    # QC metric correlations
    qc_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    qc_corr = adata.obs[qc_metrics].corr()
    print("\nQC metric correlations:")
    print(qc_corr)

    # visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # QC distributions
    axes[0, 0].hist(adata.obs['n_genes_by_counts'], bins=50, edgecolor='black')
    axes[0, 0].axvline(200, color='red', linestyle='--', label='QC threshold')
    axes[0, 0].set_xlabel('Genes per cell')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of genes per cell')
    axes[0, 0].legend()

    axes[0, 1].hist(adata.obs['total_counts'], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('UMI counts per cell')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of UMI counts')

    axes[0, 2].hist(adata.obs['pct_counts_mt'], bins=50, edgecolor='black')
    axes[0, 2].axvline(20, color='red', linestyle='--', label='QC threshold')
    axes[0, 2].set_xlabel('Mitochondrial %')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of mitochondrial content')
    axes[0, 2].legend()

    # QC relationships
    axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.3, s=1)
    axes[1, 0].set_xlabel('Total counts')
    axes[1, 0].set_ylabel('Genes detected')
    axes[1, 0].set_title('UMI counts vs genes detected')

    axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], alpha=0.3, s=1)
    axes[1, 1].set_xlabel('Total counts')
    axes[1, 1].set_ylabel('Mitochondrial %')
    axes[1, 1].set_title('UMI counts vs mitochondrial content')

    # zero-inflation per gene
    axes[1, 2].hist(genes_zero_rate, bins=50, edgecolor='black')
    axes[1, 2].set_xlabel('Fraction of zeros')
    axes[1, 2].set_ylabel('Number of genes')
    axes[1, 2].set_title('Zero-inflation distribution across genes')

    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'qc_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. HIGHLY VARIABLE GENES (SEURAT V3)
    print("\n4. HIGHLY VARIABLE GENE SELECTION")
    print("Selecting highly variable genes using Seurat v3 method...")

    # normalize for HVG selection if not already done
    if 'log1p' not in adata.uns:
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=min(20000, adata.n_vars), flavor='seurat_v3', layer=None)
    n_hvg = adata.var['highly_variable'].sum()
    print(f"Selected {n_hvg:,} highly variable genes")

    # HVG visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sc.pl.highly_variable_genes(adata, show=False, ax=ax)
    plt.savefig(os.path.join(eda_dir, 'highly_variable_genes.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. DIMENSIONAL REDUCTION (PCA)
    print("\n5. DIMENSIONAL REDUCTION (PCA)")

    # subset to HVG for PCA
    adata_hvg = adata[:, adata.var['highly_variable']].copy()

    # scale data
    sc.pp.scale(adata_hvg, max_value=10)

    # run PCA
    sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=min(50, adata_hvg.n_vars-1))

    # explained variance
    print(f"Variance explained by first 10 PCs: {adata_hvg.uns['pca']['variance_ratio'][:10].sum()*100:.2f}%")
    print(f"Variance explained by first 20 PCs: {adata_hvg.uns['pca']['variance_ratio'][:20].sum()*100:.2f}%")
    print(f"Variance explained by first 30 PCs: {adata_hvg.uns['pca']['variance_ratio'][:30].sum()*100:.2f}%")

    # PCA visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scree plot
    axes[0].plot(range(1, len(adata_hvg.uns['pca']['variance_ratio'])+1),
                 adata_hvg.uns['pca']['variance_ratio'], 'o-')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Explained')
    axes[0].set_title('PCA Scree Plot')
    axes[0].axhline(0.01, color='red', linestyle='--', alpha=0.5)

    # cumulative variance
    cumvar = np.cumsum(adata_hvg.uns['pca']['variance_ratio'])
    axes[1].plot(range(1, len(cumvar)+1), cumvar, 'o-')
    axes[1].axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% variance')
    axes[1].axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% variance')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'pca_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # PCA scatter plots
    if ct_col:
        fig = sc.pl.pca(adata_hvg, color=ct_col, return_fig=True)
        plt.savefig(os.path.join(eda_dir, 'pca_by_celltype.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'disease_cov' in adata.obs.columns:
        adata_hvg.obs['disease_cov'] = adata.obs['disease_cov'].values
        fig = sc.pl.pca(adata_hvg, color='disease_cov', return_fig=True)
        plt.savefig(os.path.join(eda_dir, 'pca_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 6. BATCH EFFECT ANALYSIS
    print("\n6. BATCH EFFECT ANALYSIS")
    if 'ind_cov' in adata.obs.columns:
        adata_hvg.obs['ind_cov'] = adata.obs['ind_cov'].values

        # check if patient correlates with PC1/PC2
        patient_numeric = pd.factorize(adata_hvg.obs['ind_cov'])[0]
        pc1_corr = np.corrcoef(adata_hvg.obsm['X_pca'][:, 0], patient_numeric)[0, 1]
        pc2_corr = np.corrcoef(adata_hvg.obsm['X_pca'][:, 1], patient_numeric)[0, 1]
        print(f"Patient correlation with PC1: {pc1_corr:.3f}")
        print(f"Patient correlation with PC2: {pc2_corr:.3f}")

    # save processed adata with PCA
    adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
    adata.uns['pca'] = adata_hvg.uns['pca']

    print(f"\nEDA complete. Results saved to {eda_dir}")
    print("Generated files:")
    print("  qc_metrics.png")
    print("  highly_variable_genes.png")
    print("  pca_variance.png")
    print("  pca_by_celltype.png")
    print("  pca_by_disease.png")

    return adata


def prepare_gmvae_data(input_h5ad, output_dir, subsample_fraction=None):
    """
    prepare data for GMVAE training

    args:
        input_h5ad: Path to input h5ad file
        output_dir: Output directory for processed files
        subsample_fraction: Optional fraction to subsample data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # from: ProtoCell4P/src/load_data.py
    # og: adata = sc.read_h5ad(data_path)
    print(f"loading data from: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    print(f"loaded data: {adata.n_obs} cells x {adata.n_vars} genes")

    # subsampling (new)
    if subsample_fraction is not None:
        n_cells = adata.shape[0]
        n_subsample = int(n_cells * subsample_fraction)
        np.random.seed(42)
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        adata = adata[indices, :]
        print(f"subsampled to {subsample_fraction*100}%: {n_subsample} cells from {n_cells}")

    # run EDA
    adata = exploratory_data_analysis(adata, output_dir)

    # from: ProtoCell4P/src/load_data.py line 31-32
    # og: sc.pp.filter_genes(adata, min_cells=5)
    print(f"\nbefore filtering: {adata.shape[1]} genes")
    sc.pp.filter_genes(adata, min_cells=5)
    print(f"after filtering (min_cells=5): {adata.shape[1]} genes")

    # from: ProtoCell4P/src/load_data.py line 33
    # og: sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Normalized to target_sum=1e4")

    # from: ProtoCell4P/src/load_data.py lines 36-37
    # og: if keep_sparse is False: adata.X = adata.X.toarray()
    # keeping sparse format (keep_sparse=True equivalent)
    X = adata.X

    # convert to genes x cells format for GMVAE
    X_transposed = X.T.tocsr()
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)

    # from: ProtoCell4P/src/load_data.py line 46
    # og: cell_types = adata.obs["ct_cov"]
    if "ct_cov" in adata.obs.columns:
        cell_types = adata.obs["ct_cov"]
    elif "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"]
    else:
        raise ValueError("no cell type column found. expected 'ct_cov' or 'cell_type'")

    # extract patient information
    if "ind_cov" in adata.obs.columns:
        patient_ids = adata.obs["ind_cov"]
    else:
        raise ValueError("no patient ID column found. expected 'ind_cov'")

    if "disease_cov" in adata.obs.columns:
        disease_labels = adata.obs["disease_cov"]
    else:
        raise ValueError("no disease label column found. expected 'disease_cov'")

    # from: ProtoCell4P/src/load_data.py lines 48-49
    # og: ct_id = sorted(set(cell_types))
    # og: mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
    ct_id = sorted(set(cell_types))
    mapping_ct = {c: idx for idx, c in enumerate(ct_id)}
    cell_type_codes = [mapping_ct[ct] for ct in cell_types]

    # create disease label mapping
    disease_id = sorted(set(disease_labels))
    mapping_disease = {d: idx for idx, d in enumerate(disease_id)}
    disease_codes = [mapping_disease[d] for d in disease_labels]

    # save all labels
    labels_df = pd.DataFrame({
        'cluster': cell_type_codes,
        'patient_id': patient_ids,
        'disease': disease_codes
    })
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    # from: ProtoCell4P/src/load_data.py line 44
    # og: genes = adata.var_names.tolist()
    genes = adata.var_names.tolist()
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for gene in genes:
            f.write(f"{gene}\n")

    # save processed adata
    adata.write(os.path.join(output_dir, "processed_data.h5ad"))

    # save metadata
    metadata = {
        'n_cells': X_transposed.shape[1],
        'n_genes': X_transposed.shape[0],
        'n_cell_types': len(np.unique(cell_type_codes)),
        'cell_types': list(pd.Categorical(cell_types).categories),
        'n_patients': len(np.unique(patient_ids)),
        'patients': list(pd.Categorical(patient_ids).categories),
        'n_diseases': len(np.unique(disease_codes)),
        'diseases': list(pd.Categorical(disease_labels).categories),
        'disease_mapping': mapping_disease,
        'subsample_fraction': subsample_fraction
    }

    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"prepared data: {X_transposed.shape[0]} genes x {X_transposed.shape[1]} cells")
    print(f"cell types: {len(np.unique(cell_type_codes))}")
    print(f"files saved to: {output_dir}")
    print("generated files:")
    print("matrix.mtx (genes x cells)")
    print("labels.csv (cell type labels)")
    print("genes.txt (gene names)")
    print("processed_data.h5ad (AnnData object)")
    print("metadata.json (dataset info)")

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for GMVAE-4P')

    # from: ProtoCell4P/src/load_data.py line 25
    # og: "../data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad"
    parser.add_argument('--input', type=str, required=True,
                       help='input h5ad file path')
    parser.add_argument('--output', type=str, required=True,
                       help='output directory for processed data')
    parser.add_argument('--subsample', type=float, default=None,
                       help='fraction to subsample (e.g., 0.1 for 10%)')

    args = parser.parse_args()

    print("=" * 60)
    print("data preprocessing")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.subsample:
        print(f"subsampling: {args.subsample*100}%")
    print()

    prepare_gmvae_data(args.input, args.output, subsample_fraction=args.subsample)

    print("\ndata preprocessed")
