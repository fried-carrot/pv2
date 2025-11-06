"""
Master preprocessing script for all baseline methods.
Runs all preprocessing scripts in parallel where possible.

Usage:
    python 0_prepare_all_data.py --input data/lupus.h5ad --output processed_data --task disease
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json

# Add preprocessing scripts to path
SCRIPT_DIR = Path(__file__).parent
PREPROCESS_DIR = SCRIPT_DIR / "preprocessing"

METHODS = {
    "protocell4p": "prepare_protocell4p.py",
    "singledeep": "prepare_singledeep.py",
    "scrat": "prepare_scrat.py",
    "pascient": "prepare_pascient.py",
}


def run_preprocessing(method, input_h5ad, output_base, task, test_size, random_state):
    """Run preprocessing for a single method."""
    script_path = PREPROCESS_DIR / METHODS[method]
    output_dir = Path(output_base) / method

    cmd = [
        "python",
        str(script_path),
        "--input",
        input_h5ad,
        "--output",
        str(output_dir),
        "--task",
        task,
        "--test_size",
        str(test_size),
        "--random_state",
        str(random_state),
    ]

    print(f"\n{'=' * 80}")
    print(f"Preprocessing: {method}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        print(result.stdout)
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {method}:")
        print(e.stderr)
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for all baseline methods"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input h5ad file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed_data",
        help="Output base directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="disease",
        help="Classification task (disease or population)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHODS.keys()),
        help="Methods to preprocess (default: all)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run preprocessing for each method
    results = {}
    for method in args.methods:
        if method not in METHODS:
            print(f"WARNING: Unknown method '{method}', skipping")
            continue

        success, error = run_preprocessing(
            method,
            args.input,
            args.output,
            args.task,
            args.test_size,
            args.random_state,
        )

        results[method] = {"success": success, "error": error}

    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)

    successful = [m for m, r in results.items() if r["success"]]
    failed = [m for m, r in results.items() if not r["success"]]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for method in successful:
        print(f"  ✓ {method}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for method in failed:
            print(f"  ✗ {method}")
            print(f"    Error: {results[method]['error']}")

    # Save summary
    summary_path = Path(args.output) / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "input": args.input,
                "task": args.task,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
