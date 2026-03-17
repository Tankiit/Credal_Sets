#!/usr/bin/env python
"""
CEBaB Experiment Runner with Calibration Split and Baselines
=============================================================

This script runs the complete CEBaB experiment pipeline:
1. Split validation data for early stopping vs calibration
2. Train GACS model with proper calibration split
3. Run geometric probes to compute degeneracy ratio ρ
4. Calibrate ε bounds using val_cp_cal split
5. Evaluate GACS credal sets on test and shift
6. Run all torch-uncertainty baselines
7. Save everything for reproducibility

All results are saved to outputs/cebab_experiment/ with timestamps
so experiments never need to be repeated.

Usage:
    python run_cebab.py --quick              # Quick test (5 epochs)
    python run_cebab.py --epochs 30          # Full training
    python run_cebab.py --epochs 30 --run_baselines  # With all baselines
"""

import argparse
import json
import os
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pickle

from gacs.config import get_config
from gacs.models.vae import GACSModel
from gacs.data.datasets import get_dataloaders
from gacs.trainer import GACSTrainer
from gacs.cal_split import build_cebab_cal_split, report_cal_split_sizes
from gacs.baselines_tu import evaluate_all_baselines


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_dir(base_dir: str = "outputs") -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"cebab_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(config, output_dir: Path):
    """Save configuration to JSON."""
    config_path = output_dir / "config.json"

    # Convert config to dict
    config_dict = {
        "data": {
            "name": config.data.name,
            "num_classes": config.data.num_classes,
            "batch_size": config.data.batch_size,
        },
        "model": {
            "z_dim": config.model.z_dim,
            "concept_dim": config.model.concept_dim,
        },
        "training": {
            "epochs": config.training.epochs,
            "seed": config.training.seed,
            "device": str(config.training.device),
        },
        "probe": {
            "num_directions": config.probe.num_directions,
            "probe_scope": config.probe.probe_scope,
        },
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"✓ Saved config to {config_path}")
    return config_path


def save_split_info(cal_loaders: Dict, output_dir: Path):
    """Save calibration split information."""
    split_info = {
        "val_early_stop_size": len(cal_loaders["val_early_stop"].dataset),
        "val_cp_cal_size": len(cal_loaders["val_cp_cal"].dataset),
        "cp_cal_ratio": len(cal_loaders["val_cp_cal"].dataset) /
                        (len(cal_loaders["val_early_stop"].dataset) +
                         len(cal_loaders["val_cp_cal"].dataset)),
    }

    split_path = output_dir / "calibration_split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"✓ Saved split info to {split_path}")
    return split_info


def save_training_metrics(trainer: GACSTrainer, output_dir: Path):
    """Save training metrics."""
    metrics_path = output_dir / "training_metrics.json"

    # Extract metrics from history
    train_accs = [m.get("accuracy", 0) for m in trainer.history["train"]]
    val_accs = [m.get("accuracy", 0) for m in trainer.history["val"]]
    train_losses = [m.get("loss", 0) for m in trainer.history["train"]]
    val_losses = [m.get("loss", 0) for m in trainer.history["val"]]

    # Find best epoch
    best_epoch = int(np.argmax(val_accs)) if len(val_accs) > 0 else 0

    metrics = {
        "best_val_acc": float(trainer.best_val_acc),
        "best_epoch": int(best_epoch),
        "final_train_acc": float(train_accs[-1]) if len(train_accs) > 0 else 0.0,
        "final_val_acc": float(val_accs[-1]) if len(val_accs) > 0 else 0.0,
        "history": {
            "train_acc": [float(x) for x in train_accs],
            "val_acc": [float(x) for x in val_accs],
            "train_loss": [float(x) for x in train_losses],
            "val_loss": [float(x) for x in val_losses],
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Saved training metrics to {metrics_path}")
    print(f"  Best val accuracy: {trainer.best_val_acc:.4f} (epoch {best_epoch})")
    return metrics


def save_probe_results(probe_results: Dict, output_dir: Path):
    """Save geometric probe results."""
    probe_path = output_dir / "probe_results.json"

    # Convert numpy arrays to lists for JSON serialization
    probe_serializable = {}
    for key, value in probe_results.items():
        if isinstance(value, np.ndarray):
            probe_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            probe_serializable[key] = float(value)
        elif isinstance(value, dict):
            probe_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else
                   float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in value.items()
            }
        else:
            probe_serializable[key] = value

    with open(probe_path, "w") as f:
        json.dump(probe_serializable, f, indent=2)

    print(f"✓ Saved probe results to {probe_path}")
    print(f"  Degeneracy ratio ρ = {probe_results['rho']:.4f}")
    return probe_serializable


def save_credal_results(credal_results: Dict, output_dir: Path, split_name: str):
    """Save credal evaluation results."""
    credal_path = output_dir / f"credal_{split_name}.json"

    credal_serializable = {}
    for key, value in credal_results.items():
        if isinstance(value, np.ndarray):
            credal_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            credal_serializable[key] = float(value)
        else:
            credal_serializable[key] = value

    with open(credal_path, "w") as f:
        json.dump(credal_serializable, f, indent=2)

    print(f"✓ Saved credal results to {credal_path}")
    return credal_serializable


def save_baseline_results(baseline_results: Dict, output_dir: Path):
    """Save baseline comparison results."""
    baseline_path = output_dir / "baseline_results.json"

    # Convert nested dict to JSON-serializable format
    baseline_serializable = {}
    for method, splits in baseline_results.items():
        baseline_serializable[method] = {}
        for split, metrics in splits.items():
            baseline_serializable[method][split] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in metrics.items()
            }

    with open(baseline_path, "w") as f:
        json.dump(baseline_serializable, f, indent=2)

    print(f"✓ Saved baseline results to {baseline_path}")
    return baseline_serializable


def save_model(model: GACSModel, output_dir: Path):
    """Save trained model."""
    model_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved model to {model_path}")
    return model_path


def save_calibration_data(eps_min: float, eps_max: float, rho: float,
                          cal_loaders: Dict, output_dir: Path):
    """Save calibration parameters."""
    cal_path = output_dir / "calibration_params.json"

    cal_data = {
        "rho": float(rho),
        "epsilon": float(eps_min + (eps_max - eps_min) * rho),
        "eps_min": float(eps_min),
        "eps_max": float(eps_max),
        "val_cp_cal_size": len(cal_loaders["val_cp_cal"].dataset),
    }

    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)

    print(f"✓ Saved calibration params to {cal_path}")
    print(f"  ρ = {rho:.4f}")
    print(f"  ε = {cal_data['epsilon']:.4f} (ε_min={eps_min:.4f}, ε_max={eps_max:.4f})")
    return cal_data


def main():
    parser = argparse.ArgumentParser(description="CEBaB Experiment Runner")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (5 epochs, smaller model)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--z_dim", type=int, default=None,
                        help="Latent dimension (overrides config)")
    parser.add_argument("--num_directions", type=int, default=None,
                        help="Number of probe directions (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Run torch-uncertainty baselines (can be slow)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Base output directory")
    parser.add_argument("--cp_cal_ratio", type=float, default=0.30,
                        help="Fraction of val for CP calibration (default: 0.30)")
    args = parser.parse_args()

    # Create output directory
    exp_dir = create_output_dir(args.output_dir)
    print(f"\n{'='*70}")
    print(f"CEBaB Experiment Runner")
    print(f"Output directory: {exp_dir}")
    print(f"{'='*70}\n")

    # Load configuration
    config = get_config(dataset="cebab", quick=args.quick)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.z_dim is not None:
        config.model.z_dim = args.z_dim
    if args.num_directions is not None:
        config.probe.num_directions = args.num_directions
    config.training.seed = args.seed

    set_seed(config.training.seed)

    print(f"Configuration:")
    print(f"  Dataset: {config.data.name}")
    print(f"  Device: {config.training.device}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  z_dim: {config.model.z_dim}")
    print(f"  Seed: {config.training.seed}")
    print(f"  CP cal ratio: {args.cp_cal_ratio}\n")

    # Save configuration
    save_config(config, exp_dir)

    # Load data
    print("Loading CEBaB data...")
    dataloaders, tokenizer = get_dataloaders(config)
    print(f"  Available splits: {list(dataloaders.keys())}")
    print(f"  Train size: {len(dataloaders['train'].dataset):,}")
    print(f"  Val size: {len(dataloaders['val'].dataset):,}")
    print(f"  Test size: {len(dataloaders['test'].dataset):,}")
    if 'shift' in dataloaders:
        print(f"  Shift size: {len(dataloaders['shift'].dataset):,}")

    # Create calibration split
    print(f"\nCreating calibration split (cp_cal_ratio={args.cp_cal_ratio})...")
    cal_loaders = build_cebab_cal_split(
        dataloaders['val'].dataset,
        batch_size=config.data.batch_size,
        cp_cal_ratio=args.cp_cal_ratio,
        seed=config.training.seed,
    )
    report_cal_split_sizes("CEBaB", cal_loaders)
    save_split_info(cal_loaders, exp_dir)

    # Update dataloaders to use calibration split
    dataloaders['val'] = cal_loaders['val_early_stop']
    dataloaders['val_cp_cal'] = cal_loaders['val_cp_cal']

    # Build model
    print("\nBuilding GACS model...")
    model = GACSModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = GACSTrainer(config, model, dataloaders, tokenizer)

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    trainer.train()

    # Save training metrics
    print("\nSaving training metrics...")
    save_training_metrics(trainer, exp_dir)

    # Save model
    print("\nSaving trained model...")
    save_model(model, exp_dir)

    # Run geometric probes
    print("\n" + "="*70)
    print("GEOMETRIC PROBES")
    print("="*70)
    probe_results = trainer.run_geometric_probes()
    save_probe_results(probe_results, exp_dir)

    rho = probe_results['rho']
    print(f"\nDegeneracy ratio ρ = {rho:.4f}")

    # Calibrate epsilon bounds
    print("\n" + "="*70)
    print("EPSILON CALIBRATION")
    print("="*70)

    # Use val_cp_cal for calibration (NOT val_early_stop)
    eps_min, eps_max = trainer.credal_constructor.calibrate_eps_bounds(
        dataloaders['val_cp_cal'],
        rho_target=rho,
    )

    save_calibration_data(eps_min, eps_max, rho, cal_loaders, exp_dir)

    # Evaluate credal sets
    print("\n" + "="*70)
    print("CREDAL EVALUATION")
    print("="*70)

    print("\nTest set:")
    test_results = trainer.run_credal_evaluation(rho, split='test')
    save_credal_results(test_results, exp_dir, 'test')

    if 'shift' in dataloaders:
        print("\nShift set:")
        shift_results = trainer.run_credal_evaluation(rho, split='shift')
        save_credal_results(shift_results, exp_dir, 'shift')

    # Run baselines
    if args.run_baselines:
        print("\n" + "="*70)
        print("TORCH-UNCERTAINTY BASELINES")
        print("="*70)
        print("\nRunning all baselines (this may take a while)...")

        try:
            baseline_results = evaluate_all_baselines(
                gacs_model=model,
                test_loader=dataloaders['test'],
                shift_loader=dataloaders.get('shift', dataloaders['test']),
                rho=rho,
                eps_min=eps_min,
                eps_max=eps_max,
                val_loader=dataloaders['val_cp_cal'],  # Use cal split for fair comparison
                num_classes=config.data.num_classes,
                device=config.training.device,
                mc_num_estimators=20,
                ens_num_estimators=5,
                ens_train_epochs=20,
            )

            save_baseline_results(baseline_results, exp_dir)

        except ImportError as e:
            print(f"\n⚠ Warning: Could not run baselines - {e}")
            print("  Install torch-uncertainty: pip install torch-uncertainty")
        except Exception as e:
            print(f"\n⚠ Warning: Baseline evaluation failed - {e}")

    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {exp_dir}/")
    print(f"\nKey metrics:")
    print(f"  Best val accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Degeneracy ratio ρ: {rho:.4f}")
    print(f"  GACS ε: {eps_min + (eps_max - eps_min) * rho:.4f}")
    print(f"  Test coverage: {test_results.get('coverage', 'N/A')}")
    if 'shift' in dataloaders:
        print(f"  Shift coverage: {shift_results.get('coverage', 'N/A')}")

    # List all saved files
    print(f"\nSaved files:")
    for file_path in sorted(exp_dir.glob("*")):
        size_kb = file_path.stat().st_size / 1024
        print(f"  {file_path.name:40s} ({size_kb:8.1f} KB)")

    print("\n" + "="*70)
    print("All results saved. No need to re-run this experiment.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
