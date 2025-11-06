#!/usr/bin/env python3
"""
Compare ROC-AUC and macro F1 across all trained methods.
Generates comparison tables and plots.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from metrics import compare_methods, load_metrics


def plot_comparison(df, output_dir, metric='Test ROC-AUC'):
    """Create bar plot comparing methods."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if m == 'GMVAE4P' else '#3498db' for m in df['Method']]

    plt.bar(df['Method'], df[metric], color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} Comparison (4 hours training each)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Add value labels on bars
    for i, (method, value) in enumerate(zip(df['Method'], df[metric])):
        plt.text(i, value + 0.01, f'{value:.4f}', ha='center', fontsize=10)

    filename = metric.lower().replace(' ', '_') + '.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_dir / filename}")
    plt.close()


def generate_latex_table(df, output_dir):
    """Generate LaTeX table for paper."""
    output_dir = Path(output_dir)

    # Select columns for table
    table_df = df[[
        'Method',
        'Test ROC-AUC',
        'Test F1',
        'Test Acc'
    ]].copy()

    # Highlight best values
    best_roc = table_df['Test ROC-AUC'].max()
    best_f1 = table_df['Test F1'].max()
    best_acc = table_df['Test Acc'].max()

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison (4 hours training per method)}")
    latex.append("\\label{tab:comparison}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Method & ROC-AUC & Macro F1 & Accuracy \\\\")
    latex.append("\\midrule")

    for _, row in table_df.iterrows():
        method = row['Method']
        roc = row['Test ROC-AUC']
        f1 = row['Test F1']
        acc = row['Test Acc']

        # Bold best values
        roc_str = f"\\textbf{{{roc:.4f}}}" if roc == best_roc else f"{roc:.4f}"
        f1_str = f"\\textbf{{{f1:.4f}}}" if f1 == best_f1 else f"{f1:.4f}"
        acc_str = f"\\textbf{{{acc:.4f}}}" if acc == best_acc else f"{acc:.4f}"

        latex.append(f"{method} & {roc_str} & {f1_str} & {acc_str} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    # Save
    latex_file = output_dir / "comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_str)

    print(f"\nLaTeX table saved to: {latex_file}")
    print("\nLaTeX code:")
    print(latex_str)

    return latex_str


def print_detailed_comparison(methods, models_dir):
    """Print detailed comparison with all metrics."""
    print("\n" + "=" * 100)
    print("DETAILED COMPARISON (4 HOURS TRAINING PER METHOD)")
    print("=" * 100)

    for method in methods:
        try:
            summary = load_metrics(method, models_dir / method)

            print(f"\n{method}")
            print("-" * 100)

            # Best validation
            print("Best Validation:")
            best_val = summary['best_validation']
            print(f"  Epoch:    {best_val['best_epoch']}")
            print(f"  Loss:     {best_val['best_val_loss']:.4f}")
            print(f"  Accuracy: {best_val['best_val_acc']:.4f}")
            print(f"  ROC-AUC:  {best_val['best_val_roc_auc']:.4f}")
            print(f"  F1 Macro: {best_val['best_val_f1_macro']:.4f}")

            # Test
            if summary['test']:
                print("\nTest Set:")
                test = summary['test']
                print(f"  Accuracy:  {test['test_accuracy']:.4f}")
                print(f"  ROC-AUC:   {test['test_roc_auc']:.4f}")
                print(f"  F1 Macro:  {test['test_f1_macro']:.4f}")
                print(f"  Precision: {test['test_precision_macro']:.4f}")
                print(f"  Recall:    {test['test_recall_macro']:.4f}")

        except Exception as e:
            print(f"\n{method}: ERROR - {e}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Compare all trained methods')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for plots and tables')
    parser.add_argument('--methods', nargs='+',
                       default=['GMVAE4P', 'ProtoCell4P', 'ScRAT', 'singleDeep', 'PaSCient'],
                       help='Methods to compare')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("COMPARING ALL METHODS")
    print("=" * 100)
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {', '.join(args.methods)}")
    print("=" * 100)

    # Load and compare metrics
    df = compare_methods(args.methods, models_dir)

    if df.empty:
        print("\nERROR: No metrics found. Train methods first.")
        return

    # Sort by Test ROC-AUC
    df = df.sort_values('Test ROC-AUC', ascending=False)

    # Save comparison table
    csv_path = output_dir / 'comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")

    # Print table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(df, output_dir, metric='Test ROC-AUC')
    plot_comparison(df, output_dir, metric='Test F1')
    plot_comparison(df, output_dir, metric='Test Acc')

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(df, output_dir)

    # Detailed comparison
    print_detailed_comparison(args.methods, models_dir)

    # Compute improvements
    if 'GMVAE4P' in df['Method'].values:
        gmvae4p_row = df[df['Method'] == 'GMVAE4P'].iloc[0]
        print("\n" + "=" * 100)
        print("GMVAE4P IMPROVEMENTS OVER BASELINES")
        print("=" * 100)

        for _, row in df.iterrows():
            if row['Method'] == 'GMVAE4P':
                continue

            roc_diff = gmvae4p_row['Test ROC-AUC'] - row['Test ROC-AUC']
            f1_diff = gmvae4p_row['Test F1'] - row['Test F1']
            acc_diff = gmvae4p_row['Test Acc'] - row['Test Acc']

            print(f"\n{row['Method']}:")
            print(f"  ROC-AUC: {roc_diff:+.4f} ({roc_diff/row['Test ROC-AUC']*100:+.2f}%)")
            print(f"  F1:      {f1_diff:+.4f} ({f1_diff/row['Test F1']*100:+.2f}%)")
            print(f"  Acc:     {acc_diff:+.4f} ({acc_diff/row['Test Acc']*100:+.2f}%)")

        print("=" * 100)

    print(f"\n\nAll results saved to: {output_dir}")
    print("Files:")
    print(f"  - {csv_path.name}")
    print(f"  - test_roc_auc.png")
    print(f"  - test_f1.png")
    print(f"  - test_acc.png")
    print(f"  - comparison_table.tex")


if __name__ == "__main__":
    main()
