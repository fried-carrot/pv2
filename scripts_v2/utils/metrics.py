"""
Unified metrics tracking for all baseline methods.
Records ROC-AUC and macro F1 score.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import torch


class MetricsTracker:
    """Track and save training/validation metrics."""

    def __init__(self, method_name, output_dir, num_classes=2):
        self.method_name = method_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_roc_auc': [],
            'val_f1_macro': [],
            'epoch': []
        }

        self.best_metrics = {
            'best_epoch': -1,
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_val_roc_auc': 0.0,
            'best_val_f1_macro': 0.0,
        }

        self.test_metrics = {}

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc,
               val_probs, val_labels, val_preds):
        """
        Update metrics for current epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
            val_probs: Validation probabilities (N, num_classes) or (N,) for binary
            val_labels: True labels (N,)
            val_preds: Predicted labels (N,)
        """
        # Convert to numpy if needed
        if torch.is_tensor(val_probs):
            val_probs = val_probs.cpu().numpy()
        if torch.is_tensor(val_labels):
            val_labels = val_labels.cpu().numpy()
        if torch.is_tensor(val_preds):
            val_preds = val_preds.cpu().numpy()

        # Calculate ROC-AUC
        try:
            if self.num_classes == 2:
                # Binary classification - use probability of positive class
                if val_probs.ndim == 2:
                    val_probs_binary = val_probs[:, 1]
                else:
                    val_probs_binary = val_probs
                roc_auc = roc_auc_score(val_labels, val_probs_binary)
            else:
                # Multi-class - one-vs-rest macro average
                roc_auc = roc_auc_score(val_labels, val_probs,
                                       multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            roc_auc = 0.0

        # Calculate macro F1
        try:
            f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        except Exception as e:
            print(f"Warning: Could not calculate F1 macro: {e}")
            f1_macro = 0.0

        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['train_acc'].append(float(train_acc))
        self.history['val_acc'].append(float(val_acc))
        self.history['val_roc_auc'].append(float(roc_auc))
        self.history['val_f1_macro'].append(float(f1_macro))

        # Update best metrics
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_epoch'] = epoch
            self.best_metrics['best_val_loss'] = float(val_loss)
            self.best_metrics['best_val_acc'] = float(val_acc)
            self.best_metrics['best_val_roc_auc'] = float(roc_auc)
            self.best_metrics['best_val_f1_macro'] = float(f1_macro)

        return roc_auc, f1_macro

    def evaluate_test(self, test_probs, test_labels, test_preds):
        """
        Evaluate on test set and save final metrics.

        Args:
            test_probs: Test probabilities (N, num_classes) or (N,)
            test_labels: True labels (N,)
            test_preds: Predicted labels (N,)
        """
        # Convert to numpy if needed
        if torch.is_tensor(test_probs):
            test_probs = test_probs.cpu().numpy()
        if torch.is_tensor(test_labels):
            test_labels = test_labels.cpu().numpy()
        if torch.is_tensor(test_preds):
            test_preds = test_preds.cpu().numpy()

        # Calculate all metrics
        accuracy = accuracy_score(test_labels, test_preds)

        # ROC-AUC
        try:
            if self.num_classes == 2:
                if test_probs.ndim == 2:
                    test_probs_binary = test_probs[:, 1]
                else:
                    test_probs_binary = test_probs
                roc_auc = roc_auc_score(test_labels, test_probs_binary)
            else:
                roc_auc = roc_auc_score(test_labels, test_probs,
                                       multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not calculate test ROC-AUC: {e}")
            roc_auc = 0.0

        # F1 scores
        f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

        # Precision and Recall
        precision_macro = precision_score(test_labels, test_preds, average='macro', zero_division=0)
        recall_macro = recall_score(test_labels, test_preds, average='macro', zero_division=0)

        self.test_metrics = {
            'test_accuracy': float(accuracy),
            'test_roc_auc': float(roc_auc),
            'test_f1_macro': float(f1_macro),
            'test_f1_weighted': float(f1_weighted),
            'test_precision_macro': float(precision_macro),
            'test_recall_macro': float(recall_macro),
        }

        return self.test_metrics

    def save(self):
        """Save all metrics to JSON files."""
        # Save training history
        history_path = self.output_dir / f"{self.method_name}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save best validation metrics
        best_path = self.output_dir / f"{self.method_name}_best_val_metrics.json"
        with open(best_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)

        # Save test metrics if available
        if self.test_metrics:
            test_path = self.output_dir / f"{self.method_name}_test_metrics.json"
            with open(test_path, 'w') as f:
                json.dump(self.test_metrics, f, indent=2)

        # Save comprehensive summary
        summary = {
            'method': self.method_name,
            'best_validation': self.best_metrics,
            'test': self.test_metrics,
            'final_epoch': self.history['epoch'][-1] if self.history['epoch'] else 0,
        }

        summary_path = self.output_dir / f"{self.method_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nMetrics saved to {self.output_dir}")
        print(f"  - Training history: {history_path.name}")
        print(f"  - Best validation: {best_path.name}")
        if self.test_metrics:
            print(f"  - Test metrics: {test_path.name}")
        print(f"  - Summary: {summary_path.name}")

    def print_summary(self):
        """Print summary of metrics."""
        print("\n" + "=" * 60)
        print(f"{self.method_name.upper()} TRAINING SUMMARY")
        print("=" * 60)

        print("\nBest Validation (Epoch {})".format(self.best_metrics['best_epoch']))
        print(f"  Loss:     {self.best_metrics['best_val_loss']:.4f}")
        print(f"  Accuracy: {self.best_metrics['best_val_acc']:.4f}")
        print(f"  ROC-AUC:  {self.best_metrics['best_val_roc_auc']:.4f}")
        print(f"  F1 Macro: {self.best_metrics['best_val_f1_macro']:.4f}")

        if self.test_metrics:
            print("\nTest Set Performance")
            print(f"  Accuracy:  {self.test_metrics['test_accuracy']:.4f}")
            print(f"  ROC-AUC:   {self.test_metrics['test_roc_auc']:.4f}")
            print(f"  F1 Macro:  {self.test_metrics['test_f1_macro']:.4f}")
            print(f"  Precision: {self.test_metrics['test_precision_macro']:.4f}")
            print(f"  Recall:    {self.test_metrics['test_recall_macro']:.4f}")

        print("=" * 60)


def load_metrics(method_name, output_dir):
    """Load saved metrics for a method."""
    output_dir = Path(output_dir)

    summary_path = output_dir / f"{method_name}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Metrics not found: {summary_path}")

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    return summary


def compare_methods(methods, output_dir):
    """
    Compare metrics across multiple methods.

    Args:
        methods: List of method names
        output_dir: Directory containing metrics files

    Returns:
        DataFrame with comparison
    """
    import pandas as pd

    results = []
    for method in methods:
        try:
            summary = load_metrics(method, output_dir)

            result = {
                'Method': method,
                'Best Val ROC-AUC': summary['best_validation']['best_val_roc_auc'],
                'Best Val F1': summary['best_validation']['best_val_f1_macro'],
                'Best Val Acc': summary['best_validation']['best_val_acc'],
            }

            if summary['test']:
                result['Test ROC-AUC'] = summary['test']['test_roc_auc']
                result['Test F1'] = summary['test']['test_f1_macro']
                result['Test Acc'] = summary['test']['test_accuracy']

            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load metrics for {method}: {e}")

    df = pd.DataFrame(results)
    return df
