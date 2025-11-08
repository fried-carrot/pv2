#!/usr/bin/env python3
"""
Train all baseline methods for GMVAE4P paper comparison (Table 1)

Trains:
- Logistic Regression (bulk RNA)
- Random Forest (bulk RNA)
- XGBoost (bulk RNA)
- Simple Concatenation (all modalities)
- Proportions-only (cell type proportions)

Uses patient-grouped 5-fold CV matching GMVAE4P setup.

Usage:
    python train_baselines.py --data_dir processed_data --output_dir results/baselines
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def patient_grouped_cv_splits(patient_ids, labels, n_folds=5, seed=42):
    """Create patient-grouped stratified CV splits"""
    unique_patients = np.unique(patient_ids)

    # Map patients to labels
    patient_to_label = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_to_label[patient] = labels.iloc[np.where(patient_mask)[0][0]]

    patient_labels = np.array([patient_to_label[p] for p in unique_patients])

    # Group by label
    label_groups = defaultdict(list)
    for idx, patient in enumerate(unique_patients):
        label_groups[patient_labels[idx]].append(patient)

    # Create stratified folds
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

    return fold_splits


def train_baseline(model_name, model_class, X_train, X_val, y_train, y_val, **model_kwargs):
    """Train a single baseline model"""
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    # Metrics
    metrics = {
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING BASELINE METHODS")
    print("=" * 80)

    # Load data
    pseudobulk = pd.read_csv(data_dir / 'multimodal' / 'pseudobulk.csv', index_col=0)
    proportions = pd.read_csv(data_dir / 'multimodal' / 'proportions.csv', index_col=0)
    states = pd.read_csv(data_dir / 'multimodal' / 'states.csv', index_col=0)
    communication = pd.read_csv(data_dir / 'multimodal' / 'communication.csv', index_col=0)
    labels = pd.read_csv(data_dir / 'multimodal' / 'labels.csv', index_col=0).squeeze()
    patient_mapping = pd.read_csv(data_dir / 'multimodal' / 'patient_mapping.csv')

    # Encode labels to integers
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels = pd.Series(labels_encoded, index=labels.index)

    patient_ids = patient_mapping['patient_id'].values
    unique_patients = np.unique(patient_ids)

    print(f"Loaded {len(pseudobulk)} samples from {len(unique_patients)} patients")

    # Create patient-grouped CV splits
    fold_splits = patient_grouped_cv_splits(patient_ids, labels, n_folds=5, seed=args.seed)

    # Define baseline configurations
    baselines = {
        'Logistic Regression (bulk)': {
            'model': LogisticRegression,
            'data': pseudobulk.values,
            'kwargs': {'max_iter': 1000, 'random_state': args.seed}
        },
        'Random Forest (bulk)': {
            'model': RandomForestClassifier,
            'data': pseudobulk.values,
            'kwargs': {'n_estimators': 100, 'max_depth': 10, 'random_state': args.seed, 'n_jobs': -1}
        },
        'XGBoost (bulk)': {
            'model': XGBClassifier,
            'data': pseudobulk.values,
            'kwargs': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': args.seed, 'eval_metric': 'logloss'}
        },
        'Proportions-only': {
            'model': LogisticRegression,
            'data': proportions.values,
            'kwargs': {'max_iter': 1000, 'random_state': args.seed}
        },
        'Concatenation (all modalities)': {
            'model': LogisticRegression,
            'data': np.concatenate([proportions.values, states.values, communication.values], axis=1),
            'kwargs': {'max_iter': 1000, 'random_state': args.seed}
        }
    }

    all_results = []

    for baseline_name, config in baselines.items():
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {baseline_name}")
        print(f"{'=' * 80}")

        X = config['data']
        y = labels.values

        fold_results = []

        for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
            print(f"  Fold {fold_num}/5...", end=" ", flush=True)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            metrics = train_baseline(
                baseline_name,
                config['model'],
                X_train, X_val,
                y_train, y_val,
                **config['kwargs']
            )

            fold_results.append(metrics)
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        # Aggregate
        avg_metrics = {
            'method': baseline_name,
            'roc_auc': np.mean([f['roc_auc'] for f in fold_results]),
            'roc_auc_std': np.std([f['roc_auc'] for f in fold_results]),
            'f1': np.mean([f['f1'] for f in fold_results]),
            'f1_std': np.std([f['f1'] for f in fold_results]),
            'accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'precision': np.mean([f['precision'] for f in fold_results]),
            'recall': np.mean([f['recall'] for f in fold_results])
        }

        all_results.append(avg_metrics)

        print(f"\nAverage ROC-AUC: {avg_metrics['roc_auc']:.4f} ± {avg_metrics['roc_auc_std']:.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'baseline_results.csv', index=False)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(results_df[['method', 'roc_auc', 'roc_auc_std', 'f1', 'f1_std']].to_string(index=False))
    print(f"\nSaved: {output_dir / 'baseline_results.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
