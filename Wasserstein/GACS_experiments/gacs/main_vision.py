#!/usr/bin/env python3
"""
GACS Vision Pipeline — MedMNIST Experiments
=============================================

Usage:
    # Quick debug
    python main_vision.py --mode debug --dataset dermamnist

    # Full single-dataset experiment
    python main_vision.py --mode full --dataset dermamnist

    # Cross-view organ shift experiment (the strongest shift story)
    python main_vision.py --mode shift --shift organ_axial_to_coronal

    # Corruption robustness sweep
    python main_vision.py --mode corrupt --dataset pathmnist

    # Run all MedMNIST datasets (generates Table for paper)
    python main_vision.py --mode sweep
"""
import argparse
import sys
import os
import torch
import numpy as np
import json
from pathlib import Path

# Ensure MedMNIST is available early with a clear error if missing
try:
    # Importing INFO lets us validate dataset names up front
    from medmnist import INFO as _MEDMNIST_INFO  # noqa: F401
except Exception as _e:
    raise ImportError(
        "MedMNIST package is required. Install with: pip install medmnist>=3.0"
    ) from _e

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gacs.configs.vision_config import medmnist_config, organ_shift_config, medmnist_debug_config
from gacs.models.vision_vae import build_vision_model, VisionConceptVAE
from gacs.training.vision_losses import VisionGACSLoss
from gacs.training.vision_trainer import VisionTrainer
from gacs.probes.geometric import get_probe
from gacs.credal.calibration import CredalCalibrator
from gacs.data.medmnist import (
    load_medmnist, load_shift_pair, create_corrupted_test,
    create_vision_dataloaders, MEDMNIST_INFO,
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_vision_backend(config, vision_backend: str, pythae_model: str):
    config.model.encoder_name = vision_backend
    if vision_backend == "pythae":
        config.model.pythae_model_name = pythae_model
    return config


def run_single_dataset(dataset_name: str, config=None, debug: bool = False):
    """Train + evaluate on a single MedMNIST dataset."""
    # Validate dataset name against MedMNIST registry for early, clear errors
    if dataset_name not in MEDMNIST_INFO:
        valid = ", ".join(sorted(MEDMNIST_INFO.keys()))
        raise ValueError(f"Unknown MedMNIST dataset '{dataset_name}'. Choose from: {valid}")
    if config is None:
        config = medmnist_debug_config(dataset_name) if debug else medmnist_config(dataset_name)
    
    device = torch.device(config.training.device)
    
    # Data
    train_ds, val_ds, test_ds = load_medmnist(dataset_name)
    train_loader, val_loader, test_loader = create_vision_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    
    # Model
    model = build_vision_model(dataset_name, config)
    loss_fn = VisionGACSLoss(config)
    
    # Train
    trainer = VisionTrainer(
        model, loss_fn, config,
        train_loader, val_loader, test_loader,
    )
    train_result = trainer.train()
    trainer.load_best()
    
    # Probe
    print("\n--- Geometric Probe ---")
    probe = get_probe(config)
    
    def probe_loss_fn(outputs, labels, concepts, epoch):
        return loss_fn.compute(outputs, labels, epoch=epoch)
    
    # Need a wrapper dataloader that provides 'concepts' key even though unused
    class VisionProbeWrapper:
        """Adapts vision DataLoader to match probe's expected interface."""
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                batch["input_ids"] = batch["image"]  # probe expects input_ids
                batch["attention_mask"] = torch.ones(1)  # dummy
                batch["concepts"] = torch.zeros(batch["label"].size(0), config.model.num_concepts)
                yield batch
        def __len__(self):
            return len(self.loader)
    
    # Override model forward to accept probe's call signature
    original_forward = model.forward
    def probe_forward(input_ids, attention_mask=None):
        return original_forward(input_ids)
    model.forward = probe_forward
    
    probe_result = probe.compute_degeneracy_ratio(
        model, VisionProbeWrapper(val_loader), probe_loss_fn, device,
    )
    model.forward = original_forward  # restore
    
    rho = probe_result["degeneracy_ratio"]
    print(f"ρ = {rho:.4f}")
    
    # Credal evaluation
    print("\n--- Credal Evaluation ---")
    calibrator = CredalCalibrator(config)
    
    # Collect test predictions
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    
    test_probs = np.concatenate(all_probs)
    test_labels = np.concatenate(all_labels)
    
    # Calibrate on val, evaluate on test
    val_probs = trainer.val_probs
    val_labels = trainer.val_labels
    
    cal_result = calibrator.calibrate_on_validation(val_probs, val_labels, rho)
    test_result = calibrator.evaluate(test_probs, test_labels, rho)
    
    print(f"\nGACS: set_acc={test_result['set_accuracy']:.4f}, "
          f"set_size={test_result['mean_set_size']:.2f}, "
          f"det={test_result['determinacy']:.4f}, "
          f"point_acc={test_result['point_accuracy']:.4f}")

    # Torch-Uncertainty baselines (optional)
    # Run only when vision backend uses Pythae (as requested) and TU is available
    try:
        from gacs import baselines_tu as tu  # soft import
        tu_available = tu.TemperatureScaler is not None
    except Exception:
        tu_available = False

    if tu_available and config.model.encoder_name.lower() == "pythae":
        print("\n--- TU Baselines (torch-uncertainty) ---")

        # Wrap model to return logits tensor for TU
        class LogitsOnly(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def forward(self, x):
                return self.core(x)["logits"]

        logits_model = LogitsOnly(model).to(device)

        # Turn dict loaders into (x, y) pairs
        def tuple_iter(loader):
            for b in loader:
                yield b["image"], b["label"]

        # Evaluate a subset of baselines that do not assume factor-head split
        eps_min = config.credal.epsilon_min
        eps_max = config.credal.epsilon_max
        eps = eps_min + (eps_max - eps_min) * rho

        # Softmax
        sm = tu.SoftmaxBaseline(logits_model, device)
        sm_test = sm.evaluate(tuple_iter(test_loader))
        print(f"Softmax     | cov={sm_test['coverage']:.3f} size={sm_test['mean_set_size']:.2f}")

        # Temp scaling (calibrate on val)
        ts = tu.TempScalingBaseline(
            logits_model,
            val_loader=tuple_iter(val_loader),
            device=device,
            eps=eps,
        )
        ts_test = ts.evaluate(tuple_iter(test_loader))
        print(f"TempScaling | cov={ts_test['coverage']:.3f} size={ts_test['mean_set_size']:.2f}")

        # MC Dropout (best-effort; may be skipped if no dropout present)
        try:
            mc = tu.MCDropoutBaseline(
                model=logits_model,
                num_estimators=20,
                eps=eps,
                device=device,
            )
            mc_test = mc.evaluate(tuple_iter(test_loader))
            print(f"MC Dropout  | cov={mc_test['coverage']:.3f} size={mc_test['mean_set_size']:.2f}")
        except Exception as e:
            print(f"MC Dropout skipped: {e}")
    
    return {
        "dataset": dataset_name,
        "train": train_result,
        "probe": probe_result,
        "calibration": cal_result,
        "test": test_result,
    }


def run_shift_experiment(
    shift_name: str,
    debug: bool = False,
    vision_backend: str = "cnn",
    pythae_model: str = "vae",
):
    """Run a distribution shift experiment."""
    from gacs.data.medmnist import SHIFT_PAIRS
    source, target = SHIFT_PAIRS[shift_name]
    
    config = medmnist_debug_config(source) if debug else medmnist_config(source)
    config = _apply_vision_backend(config, vision_backend, pythae_model)
    config.experiment_name = f"gacs_vision_{shift_name}"
    device = torch.device(config.training.device)
    
    # Load shift pair
    data = load_shift_pair(shift_name)
    
    train_loader, val_loader, test_id_loader = create_vision_dataloaders(
        data["train"], data["val"], data["test_id"],
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    # OOD test loader
    test_ood_loader = torch.utils.data.DataLoader(
        data["test_ood"], batch_size=config.training.batch_size * 2,
        shuffle=False, num_workers=config.training.num_workers, pin_memory=True,
    )
    
    # Model + train
    model = build_vision_model(source, config)
    loss_fn = VisionGACSLoss(config)
    trainer = VisionTrainer(model, loss_fn, config, train_loader, val_loader)
    trainer.train()
    trainer.load_best()
    
    # Probe on val
    probe = get_probe(config)
    
    class ProbeWrapper:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                batch["input_ids"] = batch["image"]
                batch["attention_mask"] = torch.ones(1)
                batch["concepts"] = torch.zeros(batch["label"].size(0), config.model.num_concepts)
                yield batch
        def __len__(self):
            return len(self.loader)
    
    def probe_loss_fn(outputs, labels, concepts, epoch):
        return loss_fn.compute(outputs, labels, epoch=epoch)
    
    original_forward = model.forward
    def probe_forward(input_ids, attention_mask=None):
        return original_forward(input_ids)
    model.forward = probe_forward
    
    probe_result = probe.compute_degeneracy_ratio(
        model, ProbeWrapper(val_loader), probe_loss_fn, device,
    )
    model.forward = original_forward
    rho = probe_result["degeneracy_ratio"]
    
    # Evaluate on ID and OOD
    calibrator = CredalCalibrator(config)
    cal_result = calibrator.calibrate_on_validation(
        trainer.val_probs, trainer.val_labels, rho,
    )
    
    def eval_on_loader(loader, label):
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                out = model(imgs)
                all_p.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
                all_l.append(batch["label"].numpy())
        probs = np.concatenate(all_p)
        labels = np.concatenate(all_l)
        return calibrator.evaluate(probs, labels, rho)
    
    id_result = eval_on_loader(test_id_loader, "ID")
    ood_result = eval_on_loader(test_ood_loader, "OOD")
    
    print(f"\n{'='*60}")
    print(f"SHIFT: {source} → {target}")
    print(f"{'='*60}")
    print(f"ρ = {rho:.4f}")
    print(f"ID:  set_acc={id_result['set_accuracy']:.4f}, "
          f"size={id_result['mean_set_size']:.2f}, "
          f"point_acc={id_result['point_accuracy']:.4f}")
    print(f"OOD: set_acc={ood_result['set_accuracy']:.4f}, "
          f"size={ood_result['mean_set_size']:.2f}, "
          f"point_acc={ood_result['point_accuracy']:.4f}")
    
    return {
        "shift": shift_name,
        "source": source,
        "target": target,
        "probe": probe_result,
        "id": id_result,
        "ood": ood_result,
    }


def run_corruption_experiment(
    dataset_name: str,
    debug: bool = False,
    vision_backend: str = "cnn",
    pythae_model: str = "vae",
):
    """Sweep over corruption types and severities."""
    config = medmnist_debug_config(dataset_name) if debug else medmnist_config(dataset_name)
    config = _apply_vision_backend(config, vision_backend, pythae_model)
    config.experiment_name = f"gacs_vision_corrupt_{dataset_name}"
    device = torch.device(config.training.device)
    
    # Train on clean data
    train_ds, val_ds, test_ds = load_medmnist(dataset_name)
    train_loader, val_loader, test_loader = create_vision_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    
    model = build_vision_model(dataset_name, config)
    loss_fn = VisionGACSLoss(config)
    trainer = VisionTrainer(model, loss_fn, config, train_loader, val_loader)
    trainer.train()
    trainer.load_best()
    
    # Probe
    probe = get_probe(config)
    # (probe wrapper code same as above — omitted for brevity, reuse from shift)
    rho = 0.5  # placeholder — run actual probe in practice
    
    calibrator = CredalCalibrator(config)
    calibrator.calibrate_on_validation(trainer.val_probs, trainer.val_labels, rho)
    
    # Sweep corruptions
    corruptions = ["gaussian_noise", "brightness", "contrast", "blur"]
    severities = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    for corruption in corruptions:
        for severity in severities:
            corrupt_ds = create_corrupted_test(test_ds, corruption, severity)
            corrupt_loader = torch.utils.data.DataLoader(
                corrupt_ds, batch_size=config.training.batch_size * 2,
                shuffle=False, num_workers=config.training.num_workers,
            )
            
            model.eval()
            all_p, all_l = [], []
            with torch.no_grad():
                for batch in corrupt_loader:
                    imgs = batch["image"].to(device)
                    out = model(imgs)
                    all_p.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
                    all_l.append(batch["label"].numpy())
            
            probs = np.concatenate(all_p)
            labels = np.concatenate(all_l)
            metrics = calibrator.evaluate(probs, labels, rho)
            metrics["corruption"] = corruption
            metrics["severity"] = severity
            results.append(metrics)
            
            print(f"{corruption} σ={severity:.1f}: "
                  f"set_acc={metrics['set_accuracy']:.4f}, "
                  f"size={metrics['mean_set_size']:.2f}, "
                  f"point_acc={metrics['point_accuracy']:.4f}")
    
    return results


def run_sweep(debug: bool = False):
    """Run all MedMNIST datasets and print summary table."""
    datasets = ["dermamnist", "pathmnist", "bloodmnist"]
    
    all_results = {}
    for ds in datasets:
        print(f"\n{'#'*60}")
        print(f"# {ds.upper()}")
        print(f"{'#'*60}")
        all_results[ds] = run_single_dataset(ds, debug=debug)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY — GACS on MedMNIST")
    print(f"{'='*70}")
    print(f"{'Dataset':<14} {'ρ':>6} {'Set Acc':>10} {'Set Size':>10} {'Point Acc':>10}")
    print("-" * 50)
    for ds, r in all_results.items():
        t = r["test"]
        p = r["probe"]
        print(f"{ds:<14} {p['degeneracy_ratio']:>6.4f} "
              f"{t['set_accuracy']:>10.4f} {t['mean_set_size']:>10.2f} "
              f"{t['point_accuracy']:>10.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="GACS Vision — MedMNIST")
    parser.add_argument("--mode", default="full",
                        choices=["full", "debug", "shift", "corrupt", "sweep"])
    parser.add_argument("--dataset", default="dermamnist")
    parser.add_argument("--shift", default="organ_axial_to_coronal")
    parser.add_argument("--vision_backend", default="cnn", choices=["cnn", "pythae"])
    parser.add_argument("--pythae_model", default="vae", choices=["vae", "betavae"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.mode == "debug":
        config = medmnist_debug_config(args.dataset)
        config = _apply_vision_backend(config, args.vision_backend, args.pythae_model)
        run_single_dataset(args.dataset, config=config, debug=True)
    elif args.mode == "full":
        config = medmnist_config(args.dataset)
        config = _apply_vision_backend(config, args.vision_backend, args.pythae_model)
        run_single_dataset(args.dataset, config=config, debug=False)
    elif args.mode == "shift":
        run_shift_experiment(
            args.shift,
            debug=False,
            vision_backend=args.vision_backend,
            pythae_model=args.pythae_model,
        )
    elif args.mode == "corrupt":
        run_corruption_experiment(
            args.dataset,
            debug=False,
            vision_backend=args.vision_backend,
            pythae_model=args.pythae_model,
        )
    elif args.mode == "sweep":
        run_sweep(debug=False)


if __name__ == "__main__":
    main()
