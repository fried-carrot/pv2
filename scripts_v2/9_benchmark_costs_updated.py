#!/usr/bin/env python3
"""
Master benchmarking script with updated 4-8 hour configurations
Uses training_configs.py for realistic profiling
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'baselines'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'configs'))

from profile_gmvae4p import profile_gmvae4p
from profile_protocell4p import profile_protocell4p
from profile_scrat import profile_scrat
from profile_singledeep import profile_singledeep
from profile_pascient import profile_pascient
from profile_scvi_lr import profile_scvi_lr
from profile_baseline_gmvae import profile_baseline_gmvae

from cost_profiler import CostProfiler
from training_configs import (
    GMVAE4P_CONFIG, PROTOCELL4P_CONFIG, SCRAT_CONFIG,
    SINGLEDEEP_CONFIG, PASCIENT_CONFIG, SCVI_LR_CONFIG,
    BASELINE_GMVAE_CONFIG, LUPUS_CELLS, LUPUS_PATIENTS,
    LUPUS_CELL_TYPES, LUPUS_GENES
)


def benchmark_with_configs(output_dir="cost_profiles_4to8hrs"):
    """
    Profile all 7 methods using 4-8 hour configurations
    """
    print("=" * 80)
    print("COST BENCHMARKING WITH 4-8 HOUR CONFIGURATIONS")
    print("=" * 80)
    print(f"Dataset: Lupus")
    print(f"  Cells: {LUPUS_CELLS:,}")
    print(f"  Patients: {LUPUS_PATIENTS}")
    print(f"  Cell types: {LUPUS_CELL_TYPES}")
    print(f"  Genes: {LUPUS_GENES}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print()

    os.makedirs(output_dir, exist_ok=True)
    profiles = {}

    # 1. GMVAE4P
    print("\n" + "=" * 80)
    print("1/7: GMVAE4P (our method)")
    print("=" * 80)
    cfg = GMVAE4P_CONFIG
    try:
        profiles['GMVAE4P'] = profile_gmvae4p(
            input_dim=LUPUS_GENES,
            n_cell_types=LUPUS_CELL_TYPES,
            n_classes=2,
            batch_size=cfg['gmvae']['batch_size'],
            num_training_samples=LUPUS_CELLS,
            gmvae_epochs=cfg['gmvae']['num_epochs'],
            classifier_epochs=cfg['classifier']['num_epochs'],
            output_dir=output_dir
        )
        print(f"GMVAE4P: {profiles['GMVAE4P']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['GMVAE4P'] = None

    # 2. ProtoCell4P
    print("\n" + "=" * 80)
    print("2/7: ProtoCell4P")
    print("=" * 80)
    cfg = PROTOCELL4P_CONFIG
    try:
        profiles['ProtoCell4P'] = profile_protocell4p(
            input_dim=LUPUS_GENES,
            n_proto=cfg['n_prototypes'],
            n_classes=2,
            n_ct=LUPUS_CELL_TYPES,
            num_training_samples=LUPUS_CELLS,
            output_dir=output_dir
        )
        print(f"ProtoCell4P: {profiles['ProtoCell4P']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['ProtoCell4P'] = None

    # 3. ScRAT
    print("\n" + "=" * 80)
    print("3/7: ScRAT")
    print("=" * 80)
    cfg = SCRAT_CONFIG
    try:
        profiles['ScRAT'] = profile_scrat(
            input_dim=LUPUS_GENES,
            num_training_samples=LUPUS_CELLS,
            output_dir=output_dir
        )
        print(f"ScRAT: {profiles['ScRAT']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['ScRAT'] = None

    # 4. singleDeep
    print("\n" + "=" * 80)
    print("4/7: singleDeep")
    print("=" * 80)
    cfg = SINGLEDEEP_CONFIG
    try:
        profiles['singleDeep'] = profile_singledeep(
            n_genes=LUPUS_GENES,
            out_neurons=2,
            num_training_samples=LUPUS_PATIENTS,  # Patient-level
            output_dir=output_dir
        )
        print(f"singleDeep: {profiles['singleDeep']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['singleDeep'] = None

    # 5. PaSCient
    print("\n" + "=" * 80)
    print("5/7: PaSCient")
    print("=" * 80)
    cfg = PASCIENT_CONFIG
    try:
        profiles['PaSCient'] = profile_pascient(
            n_genes=LUPUS_GENES,
            n_classes=2,
            num_training_samples=LUPUS_CELLS,
            output_dir=output_dir
        )
        print(f"PaSCient: {profiles['PaSCient']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['PaSCient'] = None

    # 6. scVI + LR
    print("\n" + "=" * 80)
    print("6/7: scVI + LR")
    print("=" * 80)
    cfg = SCVI_LR_CONFIG
    try:
        profiles['scVI_LR'] = profile_scvi_lr(
            n_genes=LUPUS_GENES,
            n_classes=2,
            num_training_samples=LUPUS_CELLS,
            output_dir=output_dir
        )
        print(f"scVI+LR: {profiles['scVI_LR']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['scVI_LR'] = None

    # 7. Baseline GMVAE
    print("\n" + "=" * 80)
    print("7/7: Baseline GMVAE")
    print("=" * 80)
    cfg = BASELINE_GMVAE_CONFIG
    try:
        profiles['Baseline_GMVAE'] = profile_baseline_gmvae(
            input_dim=LUPUS_GENES,
            K=LUPUS_CELL_TYPES,
            n_classes=2,
            num_training_samples=LUPUS_CELLS,
            output_dir=output_dir
        )
        print(f"Baseline GMVAE: {profiles['Baseline_GMVAE']['training_time_hours']:.2f} hours\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        profiles['Baseline_GMVAE'] = None

    # Generate report
    print("\n" + "=" * 80)
    print("COST COMPARISON REPORT")
    print("=" * 80)

    valid_profiles = {k: v for k, v in profiles.items() if v is not None}

    if len(valid_profiles) == 0:
        print("No profiles generated successfully")
        return None, None

    # Create comparison table
    comparison = []
    for method_name, profile in valid_profiles.items():
        comparison.append({
            'method': method_name,
            'training_cost_usd': profile['training_cost_usd'],
            'training_time_hours': profile['training_time_hours'],
            'inference_time_ms': profile['inference_time_ms'],
            'total_flops': profile['total_flops']
        })

    # Sort by training time
    comparison.sort(key=lambda x: x['training_time_hours'])

    # Print table
    print(f"\n{'Method':<20} {'Time (hrs)':<12} {'Cost ($)':<12} {'Inference (ms)':<15}")
    print("-" * 65)
    for entry in comparison:
        print(f"{entry['method']:<20} "
              f"{entry['training_time_hours']:<12.2f} "
              f"${entry['training_cost_usd']:<11.2f} "
              f"{entry['inference_time_ms']:<15.2f}")

    # Check if all methods meet 4-8 hour target
    print("\n" + "=" * 80)
    print("VALIDATION AGAINST 4-8 HOUR TARGET")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Hours':<12} {'Status':<15}")
    print("-" * 50)

    all_valid = True
    for entry in comparison:
        hours = entry['training_time_hours']
        if 4 <= hours <= 8:
            status = "✓ VALID"
        elif hours < 4:
            status = "⚠ TOO FAST"
        else:
            status = "✗ TOO SLOW"
            all_valid = False
        print(f"{entry['method']:<20} {hours:<12.2f} {status:<15}")

    if all_valid:
        print("\n✓ All methods meet 4-8 hour target")
    else:
        print("\n✗ Some methods exceed 8 hour limit")
        print("  Action: Reduce batch size, epochs, or model dimensions")

    # Save comparison
    comparison_path = os.path.join(output_dir, 'comparison_4to8hrs.json')
    with open(comparison_path, 'w') as f:
        json.dump({
            'configurations': {
                'GMVAE4P': GMVAE4P_CONFIG,
                'ProtoCell4P': PROTOCELL4P_CONFIG,
                'ScRAT': SCRAT_CONFIG,
                'singleDeep': SINGLEDEEP_CONFIG,
                'PaSCient': PASCIENT_CONFIG,
                'scVI_LR': SCVI_LR_CONFIG,
                'Baseline_GMVAE': BASELINE_GMVAE_CONFIG,
            },
            'results': comparison,
            'validation': {
                'target_hours_min': 4,
                'target_hours_max': 8,
                'all_valid': all_valid
            }
        }, f, indent=2)

    print(f"\nComparison saved to: {comparison_path}")
    print("=" * 80)

    return profiles, comparison


if __name__ == "__main__":
    profiles, comparison = benchmark_with_configs()
