#!/usr/bin/env python3
"""
Monitor training progress for Variational Credal CBM
"""

import os
import time
import subprocess
import json
from pathlib import Path

def get_file_size(filepath):
    """Get file size in MB"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)
        return f"{size:.2f} MB"
    return "N/A"

def check_process(pid):
    """Check if process is running"""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'state='],
                              capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() in ['S', 'R', 'D']
    except:
        return False

def monitor_training(experiment_dir='./experiments/cebab-roberta-100epochs'):
    """Monitor training progress"""

    exp_path = Path(experiment_dir)

    print("=" * 80)
    print("Variational Credal CBM - Training Monitor")
    print("=" * 80)
    print()

    # Check config
    config_file = exp_path / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("Configuration:")
        print(f"  Encoder: {config['encoder_name']}")
        print(f"  Concepts: {config['num_concepts']} × {config['concept_classes']} classes")
        print(f"  Task classes: {config['num_classes']}")
        print(f"  Covariance: {config['covariance_family']}")
        print(f"  MC samples: {config['num_mc_samples']}")
        print()

    # Check checkpoints
    print("Checkpoints:")
    best_model = exp_path / 'best_model.pt'
    final_model = exp_path / 'final_model.pt'

    if best_model.exists():
        print(f"  ✓ best_model.pt: {get_file_size(best_model)}")
    else:
        print(f"  ✗ best_model.pt: Not yet saved (training in progress)")

    if final_model.exists():
        print(f"  ✓ final_model.pt: {get_file_size(final_model)}")
    else:
        print(f"  ✗ final_model.pt: Will be saved at the end")

    print()

    # Check if training process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'python main.py'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"Training process status:")
            for pid in pids:
                if pid and check_process(int(pid)):
                    print(f"  ✓ Process {pid}: Running")
                else:
                    print(f"  ✗ Process {pid}: Not running")
        else:
            print("Training process status:")
            print("  ✗ No training process found")
    except:
        pass

    print()
    print("=" * 80)
    print("Tip: Run 'nvidia-smi' or 'htop' to monitor GPU/CPU usage")
    print("=" * 80)

if __name__ == "__main__":
    import sys

    exp_dir = sys.argv[1] if len(sys.argv) > 1 else './experiments/cebab-roberta-100epochs'
    monitor_training(exp_dir)
