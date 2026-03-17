#!/usr/bin/env python
"""
CEBaB Results Viewer and Summary
=================================

Loads and displays all saved results from a CEBaB experiment.
Usage:
    python view_results.py                    # View latest experiment
    python view_results.py --exp_dir outputs/cebab_20260305_203450
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_results(exp_dir: Path):
    """Load all results from experiment directory."""
    results = {
        "config": load_json(exp_dir / "config.json"),
        "split": load_json(exp_dir / "calibration_split.json"),
        "training": load_json(exp_dir / "training_metrics.json"),
        "probes": load_json(exp_dir / "probe_results.json"),
        "calibration": load_json(exp_dir / "calibration_params.json"),
        "credal_test": load_json(exp_dir / "credal_test.json"),
        "credal_shift": load_json(exp_dir / "credal_shift.json"),
        "baselines": load_json(exp_dir / "baseline_results.json"),
    }
    return {k: v for k, v in results.items() if v is not None}


def load_json(filepath: Path):
    """Load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def print_training_summary(training: dict):
    """Print training summary."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Best validation accuracy: {training['best_val_acc']:.4f} (epoch {training['best_epoch']})")
    print(f"Final train accuracy: {training['final_train_acc']:.4f}")
    print(f"Final validation accuracy: {training['final_val_acc']:.4f}")

    # Plot-like visualization
    val_accs = training['history']['val_acc']
    train_accs = training['history']['train_acc']

    print(f"\nEpoch-by-epoch accuracy:")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Δ':>10}")
    print("-"*40)
    for i, (t, v) in enumerate(zip(train_accs, val_accs)):
        delta = v - t
        marker = " ***" if i == training['best_epoch'] else ""
        print(f"{i:>6} {t:>10.4f} {v:>10.4f} {delta:>10.4f}{marker}")


def print_geometric_summary(probes: dict):
    """Print geometric probe summary."""
    print("\n" + "="*70)
    print("GEOMETRIC PROBE RESULTS")
    print("="*70)

    rho = probes['rho']
    print(f"Degeneracy ratio ρ: {rho:.4f}")

    if 'concept_probes' in probes:
        print("\nConcept-wise separability:")
        for concept, values in probes['concept_probes'].items():
            print(f"  {concept}: {values}")

    if 'class_probes' in probes:
        print("\nClass-wise separability:")
        for cls, values in probes['class_probes'].items():
            print(f"  Class {cls}: {values}")


def print_calibration_summary(cal: dict, split: dict):
    """Print calibration summary."""
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)

    print(f"Calibration set size: {split['val_cp_cal_size']:,}")
    print(f"Degeneracy ratio ρ: {cal['rho']:.4f}")
    print(f"Epsilon bounds:")
    print(f"  ε_min: {cal['eps_min']:.4f}")
    print(f"  ε_max: {cal['eps_max']:.4f}")
    print(f"  ε(ρ):  {cal['epsilon']:.4f}")
    print(f"\nEpsilon interpolation: ε = ε_min + (ε_max - ε_min) × ρ")
    print(f"                      = {cal['eps_min']:.4f} + ({cal['eps_max']:.4f} - {cal['eps_min']:.4f}) × {cal['rho']:.4f}")
    print(f"                      = {cal['epsilon']:.4f}")


def print_credal_summary(credal: dict, split_name: str):
    """Print credal evaluation summary."""
    print(f"\n{split_name.upper()} SET:")
    print(f"  Coverage: {credal.get('coverage', 'N/A'):.4f}")
    print(f"  Mean set size: {credal.get('mean_set_size', 'N/A'):.2f}")
    print(f"  Determinacy: {credal.get('determinacy', 'N/A'):.4f}")
    print(f"  Avg entropy: {credal.get('avg_entropy', 'N/A'):.4f}")


def print_baseline_comparison(baselines: dict):
    """Print baseline comparison table."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON (torch-uncertainty)")
    print("="*70)

    # Print table header
    header = f"{'Method':<18} {'Split':<7} {'Cov':>7} {'Size':>7} {'Det':>7} {'Ent':>7}"
    print(header)
    print("-"*len(header))

    # Print each method
    for method, splits in baselines.items():
        for split, metrics in splits.items():
            cov = metrics.get('coverage', float('nan'))
            size = metrics.get('mean_set_size', float('nan'))
            det = metrics.get('determinacy', float('nan'))
            ent = metrics.get('entropy', float('nan'))

            print(f"{method:<18} {split:<7} {cov:>7.3f} {size:>7.2f} {det:>7.3f} {ent:>7.3f}")

    # Compute improvement over baselines
    if 'GACS' in baselines and 'test' in baselines['GACS']:
        gacs_test = baselines['GACS']['test']
        gacs_cov = gacs_test.get('coverage', 0)
        gacs_size = gacs_test.get('mean_set_size', 0)

        print("\nGACS vs Baselines (test set):")
        for method in ['Softmax', 'Temp_scaling', 'MC_dropout', 'Deep_ensemble', 'Fixed_credal']:
            if method in baselines and 'test' in baselines[method]:
                bl = baselines[method]['test']
                bl_cov = bl.get('coverage', 0)
                bl_size = bl.get('mean_set_size', 0)

                cov_diff = gacs_cov - bl_cov
                size_diff = gacs_size - bl_size

                print(f"  vs {method:<15}: Δcov = {cov_diff:+.3f}, Δsize = {size_diff:+.2f}")


def print_full_summary(results: dict):
    """Print complete experiment summary."""
    print("\n" + "="*70)
    print("CEBaB EXPERIMENT SUMMARY")
    print("="*70)

    # Config
    if results.get('config'):
        config = results['config']
        print(f"\nConfiguration:")
        print(f"  Dataset: {config['data']['name']}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Batch size: {config['data']['batch_size']}")
        print(f"  z_dim: {config['model']['z_dim']}")
        print(f"  Seed: {config['training']['seed']}")

    # Training
    if results.get('training'):
        print_training_summary(results['training'])

    # Probes
    if results.get('probes'):
        print_geometric_summary(results['probes'])

    # Calibration
    if results.get('calibration') and results.get('split'):
        print_calibration_summary(results['calibration'], results['split'])

    # Credal
    if results.get('credal_test'):
        print("\n" + "="*70)
        print("CREDAL SET EVALUATION")
        print("="*70)
        print_credal_summary(results['credal_test'], "test")

    if results.get('credal_shift'):
        print_credal_summary(results['credal_shift'], "shift (counterfactuals)")

    # Baselines
    if results.get('baselines'):
        print_baseline_comparison(results['baselines'])

    print("\n" + "="*70)


def save_summary_text(results: dict, exp_dir: Path):
    """Save summary to text file."""
    summary_path = exp_dir / "SUMMARY.txt"

    # Redirect print output to file
    import sys
    original_stdout = sys.stdout

    with open(summary_path, 'w') as f:
        sys.stdout = f
        print_full_summary(results)
        sys.stdout = original_stdout

    print(f"\n✓ Saved summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="View CEBaB experiment results")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Experiment directory (default: latest)")
    parser.add_argument("--save_summary", action="store_true",
                        help="Save summary to SUMMARY.txt")
    args = parser.parse_args()

    # Find experiment directory
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        base = Path("outputs")
        cebab_dirs = sorted([d for d in base.glob("cebab_*") if d.is_dir()])
        if not cebab_dirs:
            print("No CEBaB experiments found in outputs/")
            return
        exp_dir = cebab_dirs[-1]

    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return

    print(f"Loading results from: {exp_dir}\n")

    # Load all results
    results = load_results(exp_dir)

    # Print summary
    print_full_summary(results)

    # Save summary if requested
    if args.save_summary:
        save_summary_text(results, exp_dir)


if __name__ == "__main__":
    main()
