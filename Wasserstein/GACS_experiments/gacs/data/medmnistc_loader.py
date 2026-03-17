#!/usr/bin/env python3
"""
MedMNIST-C Integration for GACS
=================================

Integration layer for using MedMNIST-C corrupted datasets with GACS.

MedMNIST-C provides 12 datasets with task/modality-specific corruptions:
- Noise: Gaussian, impulse, speckle, shot
- Blur: Defocus, motion, zoom, gaussian
- Compression: JPEG, pixelate
- Enhancement: Brightness, contrast, saturation, gamma
- Microscopy: Bubbles, stain deposits, black corners, characters

Usage:
    # Load corrupted test set
    from gacs.data.medmnistc_loader import get_corrupted_medmnist_loaders

    loaders = get_corrupted_medmnist_loaders(
        dataset_name="pathmnist",
        corruption="gaussian_noise",
        severity=3,  # 1-5
        corruption_type="test_shift"
    )

    # Use with shared trainer
    trainer = GACSSharedTrainer(wrapper, config, loaders["train"], loaders["val"])
    trainer.train()

    # Evaluate on corrupted test
    test_metrics = trainer.evaluate_corrupted(loaders["test_shift"])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add medmnistc-api to path
medmnistc_path = Path(__file__).parent.parent.parent / "medmnistc-api"
if str(medmnistc_path) not in sys.path:
    sys.path.insert(0, str(medmnistc_path))

logger = logging.getLogger(__name__)

# Import medmnistc components
try:
    from medmnistc.dataset import CorruptedMedMNIST
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    MEDMNISTC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"medmnistc-api not available: {e}")
    MEDMNISTC_AVAILABLE = False


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_corruption_list(dataset_name: str) -> List[str]:
    """
    Get list of available corruptions for a dataset.

    Args:
        dataset_name: One of the MedMNIST-C dataset names

    Returns:
        List of corruption names
    """
    if not MEDMNISTC_AVAILABLE:
        raise ImportError("medmnistc-api is not installed")

    if dataset_name not in CORRUPTIONS_DS:
        available = list(CORRUPTIONS_DS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        )

    return list(CORRUPTIONS_DS[dataset_name].keys())


def get_corruption_severity_range(dataset_name: str, corruption: str) -> Tuple[int, int]:
    """
    Get severity range for a corruption.

    All corruptions have 5 severity levels (1-5).

    Args:
        dataset_name: Dataset name
        corruption: Corruption name

    Returns:
        Tuple of (min_severity, max_severity)
    """
    if not MEDMNISTC_AVAILABLE:
        raise ImportError("medmnistc-api is not installed")

    if dataset_name not in CORRUPTIONS_DS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if corruption not in CORRUPTIONS_DS[dataset_name]:
        available = list(CORRUPTIONS_DS[dataset_name].keys())
        raise ValueError(
            f"Unknown corruption '{corruption}' for {dataset_name}. "
            f"Available: {available}"
        )

    return 1, 5  # All corruptions have 5 severity levels


def get_corrupted_medmnist_loaders(
    dataset_name: str,
    corruption: str,
    severity: int = 3,
    root: str = "./data/medmnistc",
    batch_size: int = 128,
    num_workers: int = 4,
    as_rgb: bool = True,
    norm_mean: list = None,
    norm_std: list = None,
) -> Dict[str, DataLoader]:
    """
    Load corrupted MedMNIST dataset for shift evaluation.

    Args:
        dataset_name: Name of MedMNIST dataset (e.g., "pathmnist")
        corruption: Type of corruption (e.g., "gaussian_noise")
        severity: Severity level (1-5, default=3)
        root: Root directory for corrupted data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        as_rgb: Convert greyscale to RGB
        norm_mean: Normalization mean
        norm_std: Normalization std

    Returns:
        Dict with 'test_shift' dataloader for corrupted test set

    Example:
        >>> loaders = get_corrupted_medmnist_loaders(
        ...     "pathmnist", "gaussian_noise", severity=3
        ... )
        >>> # Use with GACS
        >>> model.eval()
        >>> for batch in loaders["test_shift"]:
        ...     with torch.no_grad():
        ...         outputs = model(batch[0])
    """
    if not MEDMNISTC_AVAILABLE:
        raise ImportError("medmnistc-api is not installed")

    # Validate inputs
    if dataset_name not in CORRUPTIONS_DS:
        available = list(CORRUPTIONS_DS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        )

    if corruption not in CORRUPTIONS_DS[dataset_name]:
        available = list(CORRUPTIONS_DS[dataset_name].keys())
        raise ValueError(
            f"Corruption '{corruption}' not available for {dataset_name}. "
            f"Available: {available}"
        )

    if not 1 <= severity <= 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")

    # Use ImageNet normalization by default
    if norm_mean is None:
        norm_mean = [0.485, 0.456, 0.406]
    if norm_std is None:
        norm_std = [0.229, 0.224, 0.225]

    # Load corrupted dataset
    corruption_name = f"{corruption}_{severity}"

    logger.info(
        f"Loading MedMNIST-C: {dataset_name} | "
        f"corruption={corruption} | severity={severity}"
    )

    try:
        dataset = CorruptedMedMNIST(
            dataset_name=dataset_name,
            corruption=corruption_name,
            norm_mean=norm_mean,
            norm_std=norm_std,
            root=root,
            as_rgb=as_rgb,
        )
    except RuntimeError as e:
        logger.error(
            f"Failed to load corrupted dataset. "
            f"Make sure to run create_corrupted_dataset() first."
        )
        raise

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"[MedMNIST-C {dataset_name}] {corruption}_{severity}: "
        f"{len(dataset)} samples | batch_size={batch_size}"
    )

    return {
        "test_shift": loader,
    }


def get_all_corruptions_loader(
    dataset_name: str,
    severity: int = 3,
    root: str = "./data/medmnistc",
    batch_size: int = 128,
) -> Dict[str, DataLoader]:
    """
    Load all corruptions for a dataset at a given severity level.

    Useful for comprehensive robustness evaluation.

    Args:
        dataset_name: Name of MedMNIST dataset
        severity: Severity level (1-5)
        root: Root directory
        batch_size: Batch size

    Returns:
        Dict mapping corruption_name → DataLoader
    """
    corruptions = get_corruption_list(dataset_name)

    loaders = {}
    for corruption in corruptions:
        try:
            corruption_loader = get_corrupted_medmnist_loaders(
                dataset_name=dataset_name,
                corruption=corruption,
                severity=severity,
                root=root,
                batch_size=batch_size,
            )
            loaders[corruption] = corruption_loader["test_shift"]
        except Exception as e:
            logger.warning(f"Failed to load {corruption}: {e}")
            continue

    logger.info(
        f"Loaded {len(loaders)}/{len(corruptions)} corruptions "
        f"for {dataset_name} at severity {severity}"
    )

    return loaders


def print_available_corruptions(dataset_name: str = "all"):
    """
    Print all available corruptions for datasets.

    Args:
        dataset_name: "all" to show all datasets, or specific dataset name
    """
    if not MEDMNISTC_AVAILABLE:
        print("❌ medmnistc-api not installed")
        print("   Install with: pip install medmnistc")
        return

    print("\n" + "=" * 70)
    print("MedMNIST-C Available Corruptions")
    print("=" * 70)

    if dataset_name == "all":
        datasets = list(CORRUPTIONS_DS.keys())
    else:
        datasets = [dataset_name]

    for ds in datasets:
        if ds not in CORRUPTIONS_DS:
            continue

        print(f"\n{ds.upper()}:")
        print("-" * 70)

        corruptions = CORRUPTIONS_DS[ds]
        for i, (corr_name, corr_obj) in enumerate(corruptions.items(), 1):
            print(f"  {i:2d}. {corr_name:<25} — {corr_obj.__class__.__name__}")

    print("\n" + "=" * 70)
    print("Severity levels: 1 (mild) → 5 (severe)")
    print("=" * 70 + "\n")


def create_corrupted_dataset_summary(
    dataset_name: str,
    output_file: str = "medmnistc_corruptions_summary.txt"
):
    """
    Create a text summary of all corruptions for a dataset.

    Args:
        dataset_name: Dataset to summarize
        output_file: Output file path
    """
    if not MEDMNISTC_AVAILABLE:
        print("❌ medmnistc-api not installed")
        return

    corruptions = get_corruption_list(dataset_name)

    with open(output_file, 'w') as f:
        f.write(f"MedMNIST-C Corruptions Summary: {dataset_name}\n")
        f.write("=" * 70 + "\n\n")

        for corr_name in corruptions:
            f.write(f"{corr_name}\n")
            f.write("-" * 70 + "\n")

            corr_obj = CORRUPTIONS_DS[dataset_name][corr_name]
            f.write(f"  Type: {corr_obj.__class__.__name__}\n")
            f.write(f"  Severity levels: 5 (1=mild, 5=severe)\n")
            f.write(f"  Parameters: {corr_obj.severity_params}\n")
            f.write("\n")

    logger.info(f"Summary saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore MedMNIST-C corruptions")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name or 'all' to show all"
    )
    args = parser.parse_args()

    print_available_corruptions(args.dataset)
