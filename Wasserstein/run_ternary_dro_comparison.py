#!/usr/bin/env python3
"""
Run ternary concept training with all three DRO modes for CEBaB and GoEmotions.

This runs 6 experiments total:
- CEBaB: post_hoc, fixed_eps, joint
- GoEmotions: post_hoc, fixed_eps, joint

Results are saved to organized directories.
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
DATASETS = ["cebab", "goemotions"]
DRO_MODES = ["post_hoc", "fixed_eps", "joint"]
EPOCHS = 50
BATCH_SIZE = 16

# Base output directory structure
OUTPUT_BASE = Path("experiments/ternary_dro_comparison")


def run_experiment(dataset: str, dro_mode: str, epochs: int = EPOCHS):
    """Run a single experiment."""
    exp_dir = OUTPUT_BASE / dataset / dro_mode
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = exp_dir / "checkpoints"
    eval_outdir = exp_dir / "eval_outputs"

    cmd = [
        sys.executable, "train_ternary_50epochs.py",
        "--dataset", dataset,
        "--dro_mode", dro_mode,
        "--epochs", str(epochs),
        "--batch_size", str(BATCH_SIZE),
        "--checkpoint_dir", str(checkpoint_dir),
        "--eval_outdir", str(eval_outdir),
        "--save_eval",
    ]

    if dro_mode == "fixed_eps":
        cmd.extend(["--fixed_eps", "0.1"])

    print(f"\n{'='*70}")
    print(f"Running: {dataset} with DRO mode: {dro_mode}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print("TERNARY DRO MODE COMPARISON - ALL EXPERIMENTS")
    print("="*70)
    print(f"\nDatasets: {', '.join(DATASETS)}")
    print(f"DRO Modes: {', '.join(DRO_MODES)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Total experiments: {len(DATASETS) * len(DRO_MODES)}")

    results = {}

    for dataset in DATASETS:
        for dro_mode in DRO_MODES:
            key = f"{dataset}_{dro_mode}"
            success = run_experiment(dataset, dro_mode)

            results[key] = "SUCCESS" if success else "FAILED"

            if not success:
                print(f"\n⚠️  Experiment {key} failed! Continuing...")

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for key, status in results.items():
        symbol = "✅" if status == "SUCCESS" else "❌"
        print(f"{symbol} {key}: {status}")

    success_count = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nCompleted: {success_count}/{len(results)} experiments")

    if success_count == len(results):
        print("\n🎉 All experiments completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - success_count} experiment(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
