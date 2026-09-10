"""CIFAR-10 loaders for frozen-backbone feature extraction."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def pil_collate(batch):
    """Preserve PIL images so the selected backbone owns preprocessing."""
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)


def build_cifar10_test_loader(
    images_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    """Build the ordered CIFAR-10 test loader used by CIFAR-10H.

    Shuffling must remain disabled because CIFAR-10H annotations use the
    original torchvision test-set order.
    """
    dataset = CIFAR10(
        root=str(images_dir),
        train=False,
        download=False,
        transform=None,
    )
    if len(dataset) != 10_000:
        raise RuntimeError(f"expected 10000 test images, got {len(dataset)}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=pil_collate,
    )
