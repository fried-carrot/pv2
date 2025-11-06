"""
Cost profiling utility using DeepSpeed Flops Profiler
Estimates computational costs for model training and inference

References:
- Aminabadi et al. 2022 (DeepSpeed Inference): 537+ citations
- Hardware: 2x H200 80GB @ $6/hour, 60 TFLOPS
"""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: DeepSpeed not available. Install with: pip install deepspeed")


class CostProfiler:
    """
    Profiles model computational costs and estimates training/inference time
    """

    def __init__(
        self,
        gpu_price_per_hour: float = 6.0,  # H200 80GB
        gpu_tflops: float = 60.0,  # H200 peak TFLOPS
        num_gpus: int = 2
    ):
        """
        Args:
            gpu_price_per_hour: Cost per GPU hour in dollars
            gpu_tflops: Peak TFLOPS for the GPU
            num_gpus: Number of GPUs available
        """
        self.gpu_price_per_hour = gpu_price_per_hour
        self.gpu_tflops = gpu_tflops
        self.num_gpus = num_gpus

    def profile_model(
        self,
        model: nn.Module,
        input_shape: tuple,
        batch_size: int = 64,
        num_training_samples: int = 834000,
        num_epochs: int = 100,
        use_deepspeed: bool = True
    ) -> Dict[str, Any]:
        """
        Profile model FLOPs and estimate costs

        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Training batch size
            num_training_samples: Total number of training samples
            num_epochs: Number of training epochs
            use_deepspeed: Use DeepSpeed profiler if available

        Returns:
            Dictionary containing cost metrics
        """
        device = next(model.parameters()).device

        # create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)

        # profile FLOPs
        if use_deepspeed and DEEPSPEED_AVAILABLE:
            flops_per_forward = self._profile_with_deepspeed(model, dummy_input)
        else:
            flops_per_forward = self._profile_with_manual_timing(model, dummy_input)

        # calculate total operations
        num_batches = int(np.ceil(num_training_samples / batch_size))

        # training: forward + backward (3x forward FLOPs)
        flops_per_training_step = 3 * flops_per_forward
        total_training_flops = flops_per_training_step * num_batches * num_epochs

        # inference: only forward pass
        flops_per_inference = flops_per_forward
        total_inference_flops = flops_per_inference * num_batches

        # estimate time (TFLOPS = 10^12 FLOPs/sec)
        # accounting for GPU utilization (~70% efficiency)
        effective_tflops = self.gpu_tflops * self.num_gpus * 0.7

        training_time_seconds = total_training_flops / (effective_tflops * 1e12)
        inference_time_seconds = total_inference_flops / (effective_tflops * 1e12)

        # estimate costs
        training_cost = (training_time_seconds / 3600) * self.gpu_price_per_hour * self.num_gpus
        inference_cost = (inference_time_seconds / 3600) * self.gpu_price_per_hour * self.num_gpus

        # measure actual inference time
        actual_inference_time = self._measure_inference_time(model, dummy_input, num_runs=100)

        return {
            # FLOPs
            "flops_per_forward": flops_per_forward,
            "flops_per_training_step": flops_per_training_step,
            "total_training_flops": total_training_flops,
            "total_inference_flops": total_inference_flops,

            # Time estimates
            "estimated_training_time_hours": training_time_seconds / 3600,
            "estimated_inference_time_seconds": inference_time_seconds,
            "measured_inference_time_ms": actual_inference_time * 1000,

            # Cost estimates
            "training_cost_usd": training_cost,
            "inference_cost_usd": inference_cost,
            "cost_per_sample_training_usd": training_cost / num_training_samples,
            "cost_per_sample_inference_usd": inference_cost / num_training_samples,

            # Configuration
            "batch_size": batch_size,
            "num_training_samples": num_training_samples,
            "num_epochs": num_epochs,
            "num_gpus": self.num_gpus,
            "gpu_price_per_hour": self.gpu_price_per_hour,
            "gpu_tflops": self.gpu_tflops
        }

    def _profile_with_deepspeed(self, model: nn.Module, dummy_input: torch.Tensor) -> float:
        """Use DeepSpeed Flops Profiler"""
        model.eval()

        with torch.no_grad():
            flops, macs, params = get_model_profile(
                model=model,
                input_res=(dummy_input.shape[1:],),
                input_constructor=lambda shape: {"x": torch.randn(*shape).to(dummy_input.device)},
                print_profile=False,
                detailed=False
            )

        return flops

    def _profile_with_manual_timing(self, model: nn.Module, dummy_input: torch.Tensor) -> float:
        """Fallback: estimate FLOPs from timing"""
        inference_time = self._measure_inference_time(model, dummy_input, num_runs=100)

        # rough estimate: assume 70% GPU utilization
        estimated_flops = inference_time * self.gpu_tflops * 1e12 * 0.7

        return estimated_flops

    def _measure_inference_time(self, model: nn.Module, dummy_input: torch.Tensor, num_runs: int = 100) -> float:
        """Measure actual inference time"""
        model.eval()
        device = dummy_input.device

        # warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append(end - start)

        return np.median(times)

    def save_profile(self, profile: Dict[str, Any], output_path: str):
        """Save profile to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2)

        print(f"Profile saved to {output_path}")

    def compare_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> str:
        """
        Compare multiple model profiles

        Args:
            profiles: Dict mapping model names to their profiles

        Returns:
            Formatted comparison string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL COST COMPARISON")
        lines.append("=" * 80)

        # sort by training cost
        sorted_models = sorted(profiles.items(), key=lambda x: x[1]['training_cost_usd'])

        for model_name, profile in sorted_models:
            lines.append(f"\n{model_name}:")
            lines.append(f"  Training Cost: ${profile['training_cost_usd']:.2f}")
            lines.append(f"  Training Time: {profile['estimated_training_time_hours']:.2f} hours")
            lines.append(f"  Inference Time: {profile['measured_inference_time_ms']:.2f} ms/batch")
            lines.append(f"  Total FLOPs (training): {profile['total_training_flops']:.2e}")
            lines.append(f"  Cost per sample: ${profile['cost_per_sample_training_usd']:.6f}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def load_profile(path: str) -> Dict[str, Any]:
    """Load profile from JSON"""
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # test with simple model
    test_model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    ).cuda()

    profiler = CostProfiler()
    profile = profiler.profile_model(
        model=test_model,
        input_shape=(1000,),
        batch_size=64,
        num_training_samples=10000,
        num_epochs=50,
        use_deepspeed=False
    )

    print(json.dumps(profile, indent=2))
