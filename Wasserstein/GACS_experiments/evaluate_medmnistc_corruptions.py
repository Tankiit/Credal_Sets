#!/usr/bin/env python3
"""
evaluate_medmnistc_corruptions.py
-----------------------------------

Evaluate GACS models on MedMNIST-C corrupted datasets.

This script loads a trained model and evaluates it across all corruption
types and severity levels, computing:
- Accuracy across corruptions
- Relative corruption error (CE)
- Mean corruption error (mCE)

Usage:
    python evaluate_medmnistc_corruptions.py \\
        --dataset pathmnist \\
        --model_path results/best_model.pt \\
        --output_dir results/corruption_eval

Example:
    # Evaluate on single corruption
    python evaluate_medmnistc_corruptions.py \\
        --dataset pathmnist \\
        --corruption gaussian_noise \\
        --severity 3 \\
        --model_path results/best_model.pt

    # Evaluate on all corruptions (full benchmark)
    python evaluate_medmnistc_corruptions.py \\
        --dataset pathmnist \\
        --model_path results/best_model.pt \\
        --all_corruptions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from gacs.data.medmnistc_loader import (
    get_corrupted_medmnist_loaders,
    get_all_corruptions_loader,
    print_available_corruptions,
)
from gacs.models.vision_vae import VisionGACSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load trained GACS model from checkpoint."""
    logger.info(f"Loading model from {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    # Infer model architecture from checkpoint or use default
    # This is a placeholder - adjust based on your actual model config
    model = VisionGACSModel(
        in_channels=3,
        num_classes=ckpt.get("num_classes", 9),
        latent_dim=ckpt.get("latent_dim", 64),
        num_factors=ckpt.get("num_factors", 10),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded | val_acc={ckpt.get('val_acc', 'N/A'):.4f}")
    return model


@torch.no_grad()
def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a single dataloader."""
    model.eval()
    n_correct = 0
    n_total = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device).squeeze(-1).long()

        outputs = model(x)
        logits = outputs["logits"]

        preds = logits.argmax(dim=-1)
        n_correct += (preds == y).sum().item()
        n_total += y.size(0)

        all_probs.append(torch.softmax(logits, dim=-1).cpu())
        all_labels.append(y.cpu())

    accuracy = n_correct / n_total
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return {
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "probs": probs,
        "labels": labels,
    }


def compute_corruption_error(
    clean_acc: float,
    corruption_acc: float,
) -> float:
    """
    Compute relative corruption error.

    CE = (error_corrupted / error_clean) - 1
    """
    error_clean = 1.0 - clean_acc
    error_corr = 1.0 - corruption_acc

    if error_clean == 0:
        return 0.0

    return (error_corr / error_clean) - 1.0


def evaluate_all_corruptions(
    model: nn.Module,
    dataset_name: str,
    severity: int,
    device: torch.device,
    batch_size: int = 128,
    root: str = "./data/medmnistc",
) -> Dict[str, Dict]:
    """
    Evaluate model on all corruptions at a given severity.

    Returns:
        Dict mapping corruption_name -> evaluation_metrics
    """
    logger.info(
        f"Evaluating on all corruptions | "
        f"dataset={dataset_name} | severity={severity}"
    )

    loaders = get_all_corruptions_loader(
        dataset_name=dataset_name,
        severity=severity,
        root=root,
        batch_size=batch_size,
    )

    results = {}
    for corruption_name, loader in loaders.items():
        logger.info(f"  Evaluating {corruption_name}...")
        metrics = evaluate_on_loader(model, loader, device)
        results[corruption_name] = {
            "accuracy": metrics["accuracy"],
            "error": metrics["error"],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GACS on MedMNIST-C corruptions"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="MedMNIST dataset name (e.g., pathmnist)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        help="Specific corruption to evaluate (default: all)",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        help="Corruption severity level 1-5 (default: 3)",
    )
    parser.add_argument(
        "--all_corruptions",
        action="store_true",
        help="Evaluate on all corruptions",
    )
    parser.add_argument(
        "--clean_acc",
        type=float,
        default=None,
        help="Clean test accuracy for CE computation",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/medmnistc",
        help="Root directory for corrupted datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/corruption_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--list_corruptions",
        action="store_true",
        help="List available corruptions and exit",
    )

    args = parser.parse_args()

    # List corruptions if requested
    if args.list_corruptions:
        print_available_corruptions(args.dataset)
        return

    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    if args.all_corruptions:
        results = evaluate_all_corruptions(
            model=model,
            dataset_name=args.dataset,
            severity=args.severity,
            device=device,
            batch_size=args.batch_size,
            root=args.root,
        )

        # Compute mean corruption error if clean accuracy provided
        if args.clean_acc is not None:
            ces = []
            for corr_name, metrics in results.items():
                ce = compute_corruption_error(
                    args.clean_acc, metrics["accuracy"]
                )
                results[corr_name]["corruption_error"] = ce
                ces.append(ce)

            results["mean_corruption_error"] = np.mean(ces)
            results["clean_accuracy"] = args.clean_acc

    elif args.corruption:
        loaders = get_corrupted_medmnist_loaders(
            dataset_name=args.dataset,
            corruption=args.corruption,
            severity=args.severity,
            root=args.root,
            batch_size=args.batch_size,
        )

        metrics = evaluate_on_loader(
            model, loaders["test_shift"], device
        )
        results = {
            args.corruption: {
                "accuracy": metrics["accuracy"],
                "error": metrics["error"],
            }
        }

        if args.clean_acc is not None:
            ce = compute_corruption_error(
                args.clean_acc, metrics["accuracy"]
            )
            results[args.corruption]["corruption_error"] = ce
            results["clean_accuracy"] = args.clean_acc

    else:
        logger.error("Must specify --corruption or --all_corruptions")
        return

    # Save results
    output_file = output_dir / f"{args.dataset}_severity{args.severity}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"MedMNIST-C Evaluation Summary: {args.dataset}")
    print("=" * 70)

    if args.all_corruptions:
        for corr_name, metrics in results.items():
            if corr_name in ["mean_corruption_error", "clean_accuracy"]:
                continue
            print(
                f"  {corr_name:<25} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"error={metrics['error']:.4f}"
            )
            if "corruption_error" in metrics:
                print(f"    CE={metrics['corruption_error']:.4f}")

        if "mean_corruption_error" in results:
            print("\n" + "-" * 70)
            print(f"Mean Corruption Error: {results['mean_corruption_error']:.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
