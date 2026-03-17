#!/usr/bin/env python
"""
Monitor CEBaB Experiment Progress
==================================

Usage:
    python monitor_cebab.py
"""

import json
import time
from pathlib import Path
import sys

def tail_log(log_file: Path, n_lines: int = 20):
    """Read last n lines of log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        return ''.join(lines[-n_lines:])
    except:
        return f"Log file not found: {log_file}"

def get_latest_exp_dir(base_dir: str = "outputs") -> Path:
    """Get most recent experiment directory."""
    base = Path(base_dir)
    cebab_dirs = sorted([d for d in base.glob("cebab_*") if d.is_dir()])
    if not cebab_dirs:
        return None
    return cebab_dirs[-1]

def load_json(filepath: Path):
    """Load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def format_size(bytes_size: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def main():
    print("="*70)
    print("CEBaB Experiment Monitor")
    print("="*70)

    # Find latest experiment
    exp_dir = get_latest_exp_dir()
    if not exp_dir:
        print("\nNo CEBaB experiments found in outputs/")
        return

    print(f"\nLatest experiment: {exp_dir.name}")
    print(f"Path: {exp_dir}\n")

    # Check status
    config = load_json(exp_dir / "config.json")
    if config:
        print("Configuration:")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Batch size: {config['data']['batch_size']}")
        print(f"  z_dim: {config['model']['z_dim']}")
        print(f"  Device: {config['training']['device']}")

    # Check calibration split
    split_info = load_json(exp_dir / "calibration_split.json")
    if split_info:
        print(f"\nCalibration Split:")
        print(f"  val_early_stop: {split_info['val_early_stop_size']:,}")
        print(f"  val_cp_cal: {split_info['val_cp_cal_size']:,}")

    # Check training metrics
    train_metrics = load_json(exp_dir / "training_metrics.json")
    if train_metrics:
        print(f"\nTraining:")
        print(f"  Best val accuracy: {train_metrics['best_val_acc']:.4f} (epoch {train_metrics['best_epoch']})")
        print(f"  Final train accuracy: {train_metrics['final_train_acc']:.4f}")
        print(f"  Final val accuracy: {train_metrics['final_val_acc']:.4f}")

    # Check probe results
    probe_results = load_json(exp_dir / "probe_results.json")
    if probe_results:
        print(f"\nGeometric Probes:")
        print(f"  Degeneracy ratio ρ: {probe_results['rho']:.4f}")

    # Check calibration
    cal_data = load_json(exp_dir / "calibration_params.json")
    if cal_data:
        print(f"\nCalibration:")
        print(f"  ρ: {cal_data['rho']:.4f}")
        print(f"  ε: {cal_data['epsilon']:.4f}")
        print(f"  ε_min: {cal_data['eps_min']:.4f}")
        print(f"  ε_max: {cal_data['eps_max']:.4f}")

    # Check credal results
    for split in ['test', 'shift']:
        credal = load_json(exp_dir / f"credal_{split}.json")
        if credal:
            print(f"\nCredal {split}:")
            print(f"  Coverage: {credal.get('coverage', 'N/A')}")
            print(f"  Mean set size: {credal.get('mean_set_size', 'N/A')}")
            print(f"  Determinacy: {credal.get('determinacy', 'N/A')}")

    # Check baselines
    baselines = load_json(exp_dir / "baseline_results.json")
    if baselines:
        print(f"\nBaselines (torch-uncertainty):")
        for method, splits in baselines.items():
            print(f"  {method}:")
            for split, metrics in splits.items():
                print(f"    {split}: cov={metrics.get('coverage', 'N/A'):.3f}, "
                      f"size={metrics.get('mean_set_size', 'N/A'):.2f}")

    # List all files
    print(f"\nSaved Files:")
    total_size = 0
    for file_path in sorted(exp_dir.glob("*")):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            status = "✓" if file_path.exists() else "✗"
            print(f"  {status} {file_path.name:40s} {format_size(size):>10s}")

    print(f"\n  Total: {format_size(total_size):>43s}")

    # Check log file
    log_file = Path("outputs/cebab_run.log")
    if log_file.exists():
        print(f"\nRecent log output (last 20 lines):")
        print("-"*70)
        print(tail_log(log_file, 20))
        print("-"*70)

    print("\n" + "="*70)

    # Check if still running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'gacs.run_cebab'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("Status: EXPERIMENT STILL RUNNING")
        else:
            print("Status: EXPERIMENT COMPLETED")
    except:
        pass

    print("="*70 + "\n")

if __name__ == "__main__":
    main()
