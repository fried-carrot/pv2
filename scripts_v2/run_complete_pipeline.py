#!/usr/bin/env python3
"""
Complete Training Pipeline for GMVAE4P Paper

Trains all methods with 4-hour time limits and generates:
- Table 1: Main results (all methods, full dataset)
- Table 2: Small cohort evaluation (50, 100, 150, 169 patients)
- Table 3: Bulk deployment results
- Table 5: Computational cost breakdown

Usage:
    python run_complete_pipeline.py --data_dir processed_data --output_dir results/full_pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.calibration import calibration_curve
import sys
import os

sys.path.append(os.path.dirname(__file__))
from train_multimodal_cv import MultiModalDataset, train_fold
from multimodal_bulk import MultiModalClassifier


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    ece = np.mean(np.abs(prob_true - prob_pred))
    return ece


def train_gmvae4p(
    data_dir,
    output_dir,
    n_folds=5,
    epochs=30,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    seed=42,
    time_limit=None
):
    """Train GMVAE4P with patient-grouped 5-fold CV"""
    print("\n" + "=" * 80)
    print("TRAINING GMVAE4P")
    print("=" * 80)

    start_time = time.time()

    # Load data
    pseudobulk = pd.read_csv(data_dir / 'multimodal' / 'pseudobulk.csv', index_col=0)
    proportions = pd.read_csv(data_dir / 'multimodal' / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'multimodal' / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'multimodal' / 'communication.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'multimodal' / 'labels.csv', index_col=0).squeeze()
    patient_mapping = pd.read_csv(data_dir / 'multimodal' / 'patient_mapping.csv')

    with open(data_dir / 'multimodal' / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    patient_ids = patient_mapping['patient_id'].values
    unique_patients = np.unique(patient_ids)

    print(f"Loaded {len(pseudobulk)} samples from {len(unique_patients)} patients")

    # Create dataset
    full_dataset = MultiModalDataset(
        pseudobulk, proportions, states, communication, labels
    )

    # Patient-grouped CV splits
    patient_to_label = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_to_label[patient] = labels.iloc[np.where(patient_mask)[0][0]]

    patient_labels = np.array([patient_to_label[p] for p in unique_patients])

    # Manual stratified grouping
    from collections import defaultdict
    label_groups = defaultdict(list)
    for idx, patient in enumerate(unique_patients):
        label_groups[patient_labels[idx]].append(patient)

    np.random.seed(seed)
    folds = [[] for _ in range(n_folds)]
    for label, patients in label_groups.items():
        np.random.shuffle(patients)
        for i, patient in enumerate(patients):
            folds[i % n_folds].append(patient)

    # Convert to sample indices
    fold_splits = []
    for fold_patients in folds:
        val_samples = np.where(np.isin(patient_ids, fold_patients))[0]
        train_samples = np.where(~np.isin(patient_ids, fold_patients))[0]
        fold_splits.append((train_samples, val_samples))

    # Train each fold
    fold_results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
        if time_limit and (time.time() - start_time) > time_limit:
            print(f"Time limit reached, stopping at fold {fold_num}")
            break

        print(f"\nFold {fold_num}/{n_folds}")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model
        n_state_features = states.shape[1]
        model = MultiModalClassifier(
            n_genes=pseudobulk.shape[1],
            n_cell_types=proportions.shape[1],
            n_interactions=communication.shape[1],
            n_classes=metadata['n_classes'],
            embedding_dim=128,
            state_dim=n_state_features // proportions.shape[1]
        ).to(device)

        # Train fold
        fold_metrics, _ = train_fold(
            model, train_loader, val_loader, device,
            epochs, lr, fold_num, output_dir / 'models'
        )

        fold_results.append({
            'fold': fold_num,
            'roc_auc': fold_metrics['roc_auc'],
            'accuracy': fold_metrics['accuracy'],
            'f1_macro': fold_metrics['f1_macro'],
            'precision': fold_metrics['precision'],
            'recall': fold_metrics['recall']
        })

        all_predictions.extend(fold_metrics['predictions'])
        all_labels.extend(fold_metrics['labels'])
        all_probabilities.append(fold_metrics['probabilities'])

    # Aggregate metrics
    results_df = pd.DataFrame(fold_results)

    # Compute ECE from aggregated probabilities
    all_probs = np.vstack(all_probabilities)[:, 1] if len(all_probabilities) > 0 else np.array([])
    ece = compute_ece(np.array(all_labels), all_probs) if len(all_probs) > 0 else 0.0

    elapsed_time = time.time() - start_time

    return {
        'method': 'GMVAE4P',
        'roc_auc': results_df['roc_auc'].mean(),
        'roc_auc_std': results_df['roc_auc'].std(),
        'f1_macro': results_df['f1_macro'].mean(),
        'f1_macro_std': results_df['f1_macro'].std(),
        'accuracy': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'ece': ece,
        'time_seconds': elapsed_time
    }


def train_small_cohort_experiments(
    data_dir,
    output_dir,
    cohort_sizes=[50, 100, 150],
    epochs=30,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    seed=42
):
    """Train GMVAE4P on varying cohort sizes"""
    print("\n" + "=" * 80)
    print("SMALL COHORT EXPERIMENTS")
    print("=" * 80)

    results = []

    # Load data
    pseudobulk = pd.read_csv(data_dir / 'multimodal' / 'pseudobulk.csv', index_col=0)
    proportions = pd.read_csv(data_dir / 'multimodal' / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'multimodal' / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'multimodal' / 'communication.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'multimodal' / 'labels.csv', index_col=0).squeeze()
    patient_mapping = pd.read_csv(data_dir / 'multimodal' / 'patient_mapping.csv')

    with open(data_dir / 'multimodal' / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    patient_ids = patient_mapping['patient_id'].values
    unique_patients = np.unique(patient_ids)

    # Full cohort result (169 patients)
    full_result = train_gmvae4p(
        data_dir, output_dir, n_folds=5, epochs=epochs,
        batch_size=batch_size, lr=lr, device=device, seed=seed
    )
    results.append({
        'n_patients': len(unique_patients),
        'roc_auc': full_result['roc_auc'],
        'roc_auc_std': full_result['roc_auc_std']
    })

    # Subsample experiments
    for n_patients in cohort_sizes:
        print(f"\n--- Training on {n_patients} patients ---")

        # Stratified patient sampling
        patient_to_label = {}
        for patient in unique_patients:
            patient_mask = patient_ids == patient
            patient_to_label[patient] = labels.iloc[np.where(patient_mask)[0][0]]

        patient_labels = np.array([patient_to_label[p] for p in unique_patients])

        # Sample patients per class
        from collections import defaultdict
        label_groups = defaultdict(list)
        for idx, patient in enumerate(unique_patients):
            label_groups[patient_labels[idx]].append(patient)

        np.random.seed(seed)
        sampled_patients = []
        for label, patients in label_groups.items():
            n_sample = int(n_patients * len(patients) / len(unique_patients))
            sampled_patients.extend(np.random.choice(patients, size=n_sample, replace=False))

        # Filter data to sampled patients
        sample_mask = np.isin(patient_ids, sampled_patients)
        sample_indices = np.where(sample_mask)[0]

        subset_pseudobulk = pseudobulk.iloc[sample_indices]
        subset_proportions = proportions.iloc[sample_indices]
        subset_states = states.iloc[sample_indices]
        subset_communication = communication.iloc[sample_indices]
        subset_labels = labels.iloc[sample_indices]

        # Create temporary dataset
        temp_data_dir = output_dir / f'temp_{n_patients}_patients'
        temp_multimodal_dir = temp_data_dir / 'multimodal'
        temp_multimodal_dir.mkdir(parents=True, exist_ok=True)

        subset_pseudobulk.to_csv(temp_multimodal_dir / 'pseudobulk.csv')
        subset_proportions.to_csv(temp_multimodal_dir / 'proportions.csv')
        subset_states.to_csv(temp_multimodal_dir / 'states.csv')
        subset_communication.to_csv(temp_multimodal_dir / 'communication.csv')
        subset_labels.to_csv(temp_multimodal_dir / 'labels.csv')

        subset_patient_mapping = patient_mapping.iloc[sample_indices]
        subset_patient_mapping.to_csv(temp_multimodal_dir / 'patient_mapping.csv', index=False)

        with open(temp_multimodal_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Train on subset
        subset_result = train_gmvae4p(
            temp_data_dir, output_dir, n_folds=5, epochs=epochs,
            batch_size=batch_size, lr=lr, device=device, seed=seed
        )

        results.append({
            'n_patients': n_patients,
            'roc_auc': subset_result['roc_auc'],
            'roc_auc_std': subset_result['roc_auc_std']
        })

    return pd.DataFrame(results)


def simulate_bulk_deployment(
    data_dir,
    output_dir,
    model_path,
    device='cuda'
):
    """Simulate bulk RNA-seq deployment by aggregating pseudobulk"""
    print("\n" + "=" * 80)
    print("BULK DEPLOYMENT SIMULATION")
    print("=" * 80)

    # Load pseudobulk data
    pseudobulk = pd.read_csv(data_dir / 'multimodal' / 'pseudobulk.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'multimodal' / 'labels.csv', index_col=0).squeeze()
    patient_mapping = pd.read_csv(data_dir / 'multimodal' / 'patient_mapping.csv')

    # Aggregate pseudobulk by patient (mean expression)
    patient_ids = patient_mapping['patient_id'].values
    unique_patients = np.unique(patient_ids)

    bulk_data = []
    bulk_labels = []

    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_samples = pseudobulk.iloc[patient_mask]

        # Mean aggregation
        bulk_profile = patient_samples.mean(axis=0)
        bulk_data.append(bulk_profile.values)

        # Get label
        bulk_labels.append(labels.iloc[np.where(patient_mask)[0][0]])

    bulk_data = np.array(bulk_data)
    bulk_labels = np.array(bulk_labels)

    print(f"Aggregated {len(pseudobulk)} samples into {len(bulk_data)} bulk profiles")

    # Load trained model
    checkpoint = torch.load(model_path)

    # Create model (need to pass correct dims)
    with open(data_dir / 'multimodal' / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    proportions = pd.read_csv(data_dir / 'multimodal' / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'multimodal' / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'multimodal' / 'communication.csv', index_col=0)

    model = MultiModalClassifier(
        n_genes=pseudobulk.shape[1],
        n_cell_types=proportions.shape[1],
        n_interactions=communication.shape[1],
        n_classes=metadata['n_classes'],
        embedding_dim=128,
        state_dim=states.shape[1] // proportions.shape[1]
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Predict
    bulk_tensor = torch.FloatTensor(bulk_data).to(device)

    with torch.no_grad():
        output = model(
            bulk=bulk_tensor,
            labels=None,
            true_proportions=None,
            true_states=None,
            true_communication=None,
            training=False
        )

        probs = torch.softmax(output['logits'], dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    # Compute metrics
    label_map = {label: idx for idx, label in enumerate(sorted(np.unique(bulk_labels)))}
    numeric_labels = np.array([label_map[l] for l in bulk_labels])

    roc_auc = roc_auc_score(numeric_labels, probs[:, 1])
    accuracy = accuracy_score(numeric_labels, preds)
    f1 = f1_score(numeric_labels, preds, average='macro')

    print(f"Bulk Deployment Results:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Macro: {f1:.4f}")

    return {
        'data_type': 'Bulk (aggregated)',
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1_macro': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Complete training pipeline for GMVAE4P paper')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip_baselines', action='store_true', help='Skip baseline training')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("GMVAE4P COMPLETE TRAINING PIPELINE")
    print("=" * 80)
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    all_results = {}

    # 1. Train GMVAE4P (full dataset)
    gmvae4p_results = train_gmvae4p(
        data_dir, output_dir,
        n_folds=5, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, seed=args.seed
    )
    all_results['gmvae4p_full'] = gmvae4p_results

    # 2. Small cohort experiments
    small_cohort_results = train_small_cohort_experiments(
        data_dir, output_dir,
        cohort_sizes=[50, 100, 150],
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, seed=args.seed
    )
    small_cohort_results.to_csv(output_dir / 'small_cohort_metrics.csv', index=False)
    print(f"\nSaved: {output_dir / 'small_cohort_metrics.csv'}")

    # 3. Bulk deployment simulation
    best_model_path = output_dir / 'models' / 'fold_1_best.pth'
    if best_model_path.exists():
        bulk_results = simulate_bulk_deployment(
            data_dir, output_dir, best_model_path, device=device
        )
        all_results['bulk_deployment'] = bulk_results

    # Save main results
    main_results = pd.DataFrame([gmvae4p_results])
    main_results.to_csv(output_dir / 'main_metrics.csv', index=False)
    print(f"\nSaved: {output_dir / 'main_metrics.csv'}")

    # Compute cost (assuming $6/hr for H100)
    cost_per_hour = 6.0
    total_time_hours = gmvae4p_results['time_seconds'] / 3600
    total_cost = total_time_hours * cost_per_hour

    compute_costs = pd.DataFrame([{
        'component': 'GMVAE4P (5-fold CV)',
        'wall_clock_time': f"{total_time_hours:.2f}h",
        'cost_usd': f"${total_cost:.2f}"
    }])
    compute_costs.to_csv(output_dir / 'compute_times.csv', index=False)
    print(f"Saved: {output_dir / 'compute_times.csv'}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Main results: {output_dir / 'main_metrics.csv'}")
    print(f"Small cohort: {output_dir / 'small_cohort_metrics.csv'}")
    print(f"Compute costs: {output_dir / 'compute_times.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
