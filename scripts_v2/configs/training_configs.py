"""
Training configurations for all baseline methods.
Target: 4-8 hours on single 20GB GPU (H100/A100).

Assumptions:
- Lupus dataset: 834k cells, 214 patients, 8 cell types, 1000 genes
- Single GPU training
- 20GB VRAM constraint
"""

import torch

# Hardware assumptions
GPU_MEMORY_GB = 20
TARGET_HOURS = 4  # All methods normalized to 4 hours for fair comparison

# Dataset specs (Lupus)
LUPUS_CELLS = 834000
LUPUS_PATIENTS = 214
LUPUS_CELL_TYPES = 8
LUPUS_GENES = 1000

# GMVAE4P Configuration (two-stage)
GMVAE4P_CONFIG = {
    # Stage 1: GMVAE training
    'gmvae': {
        'input_dim': LUPUS_GENES,
        'h_dim': 256,
        'z_dim': 64,
        'n_components': LUPUS_CELL_TYPES,
        'batch_size': 512,
        'num_epochs': 35,  # Adjusted for 4h total
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'gradient_clip': 5.0,
        'early_stopping_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
    },
    # Stage 2: P4P classifier
    'classifier': {
        'z_dim': 64,
        'n_classes': 2,
        'n_prototypes': 16,
        'n_cell_types': LUPUS_CELL_TYPES,
        'batch_size': 32,
        'num_epochs': 20,  # Adjusted for 4h total
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'gradient_clip': 5.0,
        'early_stopping_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    },
    'estimated_hours': 4.0,
}

# ProtoCell4P Configuration
PROTOCELL4P_CONFIG = {
    'input_dim': LUPUS_GENES,
    'h_dim': 256,
    'z_dim': 64,
    'n_layers': 3,
    'n_prototypes': 16,  # Default from paper (tested 4, 8, 16)
    'n_classes': 2,
    'n_cell_types': LUPUS_CELL_TYPES,
    'batch_size': 32,  # Patient-level batching
    'num_epochs': 80,  # Adjusted for 4h
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'lambda_proto': 1.0,
    'lambda_sep': 1.0,
    'gradient_clip': 5.0,
    'early_stopping_patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'estimated_hours': 4.0,
}

# ScRAT Configuration
SCRAT_CONFIG = {
    'vocab_size': LUPUS_GENES + 10,  # Genes + special tokens
    'max_seq_length': 1024,  # Max cells per patient (will pad/truncate)
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1,
    'n_classes': 2,
    'batch_size': 16,  # Sequence-level batching (memory intensive)
    'num_epochs': 28,  # Adjusted for 4h (slower due to transformer)
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'warmup_steps': 500,  # Reduced proportionally
    'gradient_clip': 1.0,
    'early_stopping_patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'estimated_hours': 4.0,
}

# singleDeep Configuration
SINGLEDEEP_CONFIG = {
    'input_dim': LUPUS_GENES * LUPUS_CELL_TYPES,  # Concatenated cell type profiles
    'hidden_dims': [512, 256, 128],
    'n_classes': 2,
    'batch_size': 64,
    'num_epochs': 90,  # Adjusted for 4h (already fast)
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'gradient_clip': 5.0,
    'early_stopping_patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'estimated_hours': 4.0,
}

# PaSCient Configuration
PASCIENT_CONFIG = {
    'input_dim': LUPUS_GENES,
    'n_cell_types': LUPUS_CELL_TYPES,
    'h_dim': 256,
    'z_dim': 64,
    'n_classes': 2,
    'batch_size': 32,
    'num_epochs': 53,  # Adjusted for 4h
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'dropout': 0.2,
    'gradient_clip': 5.0,
    'early_stopping_patience': 12,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'estimated_hours': 4.0,
}

# Summary
ALL_CONFIGS = {
    'GMVAE4P': GMVAE4P_CONFIG,
    'ProtoCell4P': PROTOCELL4P_CONFIG,
    'ScRAT': SCRAT_CONFIG,
    'singleDeep': SINGLEDEEP_CONFIG,
    'PaSCient': PASCIENT_CONFIG,
}

def get_config(method_name: str):
    """Get configuration for a specific method."""
    if method_name not in ALL_CONFIGS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[method_name]

def print_summary():
    """Print summary of all configurations."""
    print("=" * 80)
    print("TRAINING CONFIGURATIONS SUMMARY")
    print("=" * 80)
    print(f"Target: {TARGET_HOURS} hours per method (fair comparison)")
    print(f"GPU: {GPU_MEMORY_GB}GB VRAM")
    print(f"Dataset: {LUPUS_CELLS:,} cells, {LUPUS_PATIENTS} patients, {LUPUS_CELL_TYPES} cell types")
    print("=" * 80)

    total_hours = 0
    for name, config in ALL_CONFIGS.items():
        est_hours = config.get('estimated_hours', 0)
        total_hours += est_hours
        print(f"{name:20s}: {est_hours:.1f} hours")

    print("=" * 80)
    print(f"TOTAL: {total_hours:.1f} hours ({len(ALL_CONFIGS)} methods × {TARGET_HOURS}h)")
    print(f"Sequential: ~{total_hours/24:.1f} days")
    print(f"Parallel (5 GPUs): ~{TARGET_HOURS}h")
    print("=" * 80)

if __name__ == '__main__':
    print_summary()
