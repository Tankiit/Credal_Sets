"""
gacs/data/medmnist_loader.py
----------------------------
MedMNIST data loading for the full GACS experimental suite.

Three loading modes:

  1. standard(dataset_name)
     Normal train/val/test split from one dataset.
     Used for: all datasets in the appendix sweep table.

  2. cross_view(source_view, target_view)
     Train/val/calibrate on source_view, test (shift) on target_view.
     Used for: OrganMNIST headline experiment.
     OrganMNIST comes in three views: OrganAMNIST (axial),
     OrganCMNIST (coronal), OrganSMNIST (sagittal).

  3. cross_source(dataset_name)
     For datasets with documented cross-source shift
     (DermaMNIST, PathMNIST). Uses the standard split but
     flags the test set as a shift evaluation.

Dataset tier assignments:
  Tier 1 (main results table):
    OrganAMNIST → OrganCMNIST  cross-view shift
    DermaMNIST                 cross-source shift
    PathMNIST                  cross-institutional shift
    BloodMNIST                 minimal shift / sanity check

  Tier 2 (appendix sweep):
    BreastMNIST, PneumoniaMNIST, RetinaMNIST, OCTMNIST,
    TissueMNIST, ChestMNIST, NoduleMNIST

All datasets use 28×28 images (MedMNIST standard).
Normalisation: pixel values ∈ [0,1], ImageNet mean/std NOT applied
(MedMNIST images are greyscale or RGB medical images, not natural images).

D1 compliance: use BCE reconstruction loss, not MSE.
Set reconstruction_loss="bce" in backbone config.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# ── medmnist import guard ──────────────────────────────────────────────────────
try:
    import medmnist
    from medmnist import INFO, Evaluator
    from medmnist.dataset import (
        PathMNIST, DermaMNIST, BloodMNIST,
        OrganAMNIST, OrganCMNIST, OrganSMNIST,
        BreastMNIST, PneumoniaMNIST, RetinaMNIST,
        OCTMNIST, TissueMNIST, ChestMNIST,
    )
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    logger.warning(
        "medmnist not installed. Run: pip install medmnist"
    )


# ── Dataset registry ──────────────────────────────────────────────────────────

# Maps string name → (medmnist class, n_channels, n_classes, tier, shift_type)
DATASET_REGISTRY: Dict[str, dict] = {
    # Tier 1 — main results table
    "organamnist": dict(
        cls=OrganAMNIST,  n_channels=1, n_classes=11,
        tier=1, shift="cross_view",
        description="Abdominal CT — axial view",
    ),
    "organcmnist": dict(
        cls=OrganCMNIST,  n_channels=1, n_classes=11,
        tier=1, shift="cross_view",
        description="Abdominal CT — coronal view",
    ),
    "organsmnist": dict(
        cls=OrganSMNIST,  n_channels=1, n_classes=11,
        tier=1, shift="cross_view",
        description="Abdominal CT — sagittal view",
    ),
    "dermamnist": dict(
        cls=DermaMNIST,   n_channels=3, n_classes=7,
        tier=1, shift="cross_source",
        description="Dermatoscopy — cross-source shift",
    ),
    "pathmnist": dict(
        cls=PathMNIST,    n_channels=3, n_classes=9,
        tier=1, shift="cross_institutional",
        description="Colon pathology — cross-institutional",
    ),
    "bloodmnist": dict(
        cls=BloodMNIST,   n_channels=3, n_classes=8,
        tier=1, shift="minimal",
        description="Blood cell microscopy — minimal shift (sanity check)",
    ),
    # Tier 2 — appendix sweep
    "breastmnist": dict(
        cls=BreastMNIST,      n_channels=1, n_classes=2,
        tier=2, shift="minimal",
        description="Breast ultrasound — binary",
    ),
    "pneumoniamnist": dict(
        cls=PneumoniaMNIST,   n_channels=1, n_classes=2,
        tier=2, shift="minimal",
        description="Chest X-ray pneumonia — binary",
    ),
    "retinamnist": dict(
        cls=RetinaMNIST,      n_channels=3, n_classes=5,
        tier=2, shift="minimal",
        description="Retinal fundus images",
    ),
    "octmnist": dict(
        cls=OCTMNIST,         n_channels=1, n_classes=4,
        tier=2, shift="minimal",
        description="Retinal OCT",
    ),
    "tissuemnist": dict(
        cls=TissueMNIST,      n_channels=1, n_classes=8,
        tier=2, shift="minimal",
        description="Kidney cortex tissue microscopy",
    ),
    "chestmnist": dict(
        cls=ChestMNIST,       n_channels=1, n_classes=14,
        tier=2, shift="minimal",
        description="Chest X-ray multi-label",
        multilabel=True,  # NOTE: multi-label, handle separately
    ),
}

# Cross-view pairs for OrganMNIST
# Each tuple is (source, target) — train on source, shift-test on target
ORGAN_VIEW_PAIRS = [
    ("organamnist", "organcmnist"),   # Axial → Coronal  (headline)
    ("organamnist", "organsmnist"),   # Axial → Sagittal
    ("organcmnist", "organsmnist"),   # Coronal → Sagittal
]

ORGAN_HEADLINE_PAIR = ("organamnist", "organcmnist")   # confirmed headline


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transform(n_channels: int, augment: bool = False) -> transforms.Compose:
    """
    D1-compliant transform: normalise to [0,1], no ImageNet stats.

    MedMNIST images are medical images — applying ImageNet normalisation
    would be incorrect and would violate D1 by introducing a scale shift.

    augment=True adds random horizontal flip + small rotation for training.
    TODO: confirm augmentation is appropriate for your datasets.
          For pathology images, flipping is fine.
          For OrganMNIST (CT anatomy), flipping may be clinically invalid
          (left/right asymmetry matters). Set augment=False for OrganMNIST.
    """
    base = [
        transforms.ToTensor(),  # [0,255] uint8 → [0,1] float32
    ]
    if n_channels == 1:
        # Greyscale: expand to 3 channels for ResNet compatibility
        # TODO: if using a custom 1-channel encoder, remove this
        base.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    if augment:
        base = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ] + base

    return transforms.Compose(base)


# ── Standard loader ────────────────────────────────────────────────────────────

def get_standard_loaders(
    dataset_name: str,
    data_root: str = "./data/medmnist",
    batch_size: int = 128,
    num_workers: int = 0,        # MPS safe default
    pin_memory: bool = False,    # MPS safe default
    download: bool = True,
    probe_size: int = 512,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Standard train/val/test loaders for one MedMNIST dataset.

    Returns dict with keys: train, val, test, probe
    probe: held-out subset of train for geometric probing.
           Must NOT overlap with val (used for eps calibration).

    TODO: for ChestMNIST (multi-label), use a separate loader
          that handles multi-hot labels. Flagged in DATASET_REGISTRY.
    """
    assert MEDMNIST_AVAILABLE, "pip install medmnist"
    assert dataset_name.lower() in DATASET_REGISTRY, (
        f"Unknown dataset '{dataset_name}'. "
        f"Available: {list(DATASET_REGISTRY.keys())}"
    )

    info     = DATASET_REGISTRY[dataset_name.lower()]
    cls      = info["cls"]
    n_ch     = info["n_channels"]

    train_ds = cls(
        split="train", transform=get_transform(n_ch, augment=True),
        download=download, root=data_root,
    )
    val_ds   = cls(
        split="val",   transform=get_transform(n_ch, augment=False),
        download=download, root=data_root,
    )
    test_ds  = cls(
        split="test",  transform=get_transform(n_ch, augment=False),
        download=download, root=data_root,
    )

    # Probe subset from training set
    # Carve first probe_size indices — deterministic, no randomness needed
    # (augmentation is off for probe anyway)
    probe_ds = cls(
        split="train", transform=get_transform(n_ch, augment=False),
        download=False, root=data_root,
    )
    torch.manual_seed(seed)
    probe_indices = torch.randperm(len(probe_ds))[:probe_size].tolist()
    probe_ds      = Subset(probe_ds, probe_indices)

    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

    loaders = {
        "train": make_loader(train_ds, shuffle=True),
        "val":   make_loader(val_ds),
        "test":  make_loader(test_ds),
        "probe": make_loader(probe_ds),
    }

    logger.info(
        f"[{dataset_name}] train={len(train_ds)} | val={len(val_ds)} | "
        f"test={len(test_ds)} | probe={len(probe_ds)} | "
        f"classes={info['n_classes']} | shift={info['shift']}"
    )
    return loaders


def get_cross_view_loaders(
    source: str,
    target: str,
    data_root: str = "./data/medmnist",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = True,
    probe_size: int = 512,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Cross-view loader for OrganMNIST.

    Protocol: train/val/probe on source view, test (shift) on target view.

    Args:
        source: source view name (e.g., "organamnist")
        target: target view name (e.g., "organcmnist")

    Returns dict with keys: train, val, test, probe
    """
    assert MEDMNIST_AVAILABLE, "pip install medmnist"
    assert source.lower() in DATASET_REGISTRY
    assert target.lower() in DATASET_REGISTRY

    source_info = DATASET_REGISTRY[source.lower()]
    target_info = DATASET_REGISTRY[target.lower()]

    n_ch = 1  # OrganMNIST is greyscale

    # Source datasets (train/val with augmentation)
    train_ds = source_info["cls"](
        split="train", transform=get_transform(n_ch, augment=True),
        download=download, root=data_root,
    )
    val_ds = source_info["cls"](
        split="val", transform=get_transform(n_ch, augment=False),
        download=download, root=data_root,
    )

    # Target test set (shift)
    test_ds = target_info["cls"](
        split="test", transform=get_transform(n_ch, augment=False),
        download=download, root=data_root,
    )

    # Probe from source train
    probe_base = source_info["cls"](
        split="train", transform=get_transform(n_ch, augment=False),
        download=False, root=data_root,
    )
    torch.manual_seed(seed)
    probe_indices = torch.randperm(len(probe_base))[:probe_size].tolist()
    probe_ds = Subset(probe_base, probe_indices)

    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    loaders = {
        "train": make_loader(train_ds, shuffle=True),
        "val": make_loader(val_ds),
        "test": make_loader(test_ds),
        "probe": make_loader(probe_ds),
    }

    logger.info(
        f"[Cross-view {source}→{target}] "
        f"train={len(train_ds)} | val={len(val_ds)} | "
        f"test_shift={len(test_ds)} | probe={len(probe_ds)}"
    )
    return loaders


# ── Cross-view loader (OrganMNIST headline) ───────────────────────────────────

def get_organ_leave_one_out_loaders(
    held_out_view: str,
    data_root: str  = "./data/medmnist",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool   = True,
    probe_size: int  = 512,
    seed: int        = 42,
) -> Dict[str, DataLoader]:
    """
    Leave-one-out cross-view loader for OrganMNIST.

    Protocol: train on TWO views, test (shift) on the HELD-OUT view.
    Run three times (hold out A, C, S in turn), report mean ± std.

    This is stronger than a fixed pair — demonstrates the result is
    not specific to one view transition.

    held_out_view : one of "organamnist", "organcmnist", "organsmnist"

    Returns loaders with keys:
        train      — combined train splits of the two source views
        val        — combined val splits of the two source views
        test_iid   — combined test splits of the two source views
        test_shift — test split of the held-out view  (shift evaluation)
        probe      — held-out subset of train (for geometric probing)

    Training size: ~68k (two views × ~34k each)
    Shift test size: ~8k (one view test split)

    TODO: no horizontal flip augmentation for OrganMNIST — CT anatomy
          has left/right semantics (liver is right-sided, spleen left-sided).
          Flipping would create anatomically invalid training examples.
    """
    assert MEDMNIST_AVAILABLE, "pip install medmnist"

    all_views  = ["organamnist", "organcmnist", "organsmnist"]
    held_out   = held_out_view.lower()
    assert held_out in all_views, (
        f"held_out_view must be one of {all_views}, got '{held_out}'"
    )
    source_views = [v for v in all_views if v != held_out]

    n_ch = 1   # OrganMNIST is greyscale
    tf   = get_transform(n_ch, augment=False)   # no flip for CT anatomy

    # ── Build source datasets (two views concatenated) ────────────────────────
    from torch.utils.data import ConcatDataset

    src_train_ds = ConcatDataset([
        DATASET_REGISTRY[v]["cls"](
            split="train", transform=tf,
            download=download, root=data_root,
        )
        for v in source_views
    ])
    src_val_ds = ConcatDataset([
        DATASET_REGISTRY[v]["cls"](
            split="val", transform=tf,
            download=False, root=data_root,
        )
        for v in source_views
    ])
    src_test_ds = ConcatDataset([
        DATASET_REGISTRY[v]["cls"](
            split="test", transform=tf,
            download=False, root=data_root,
        )
        for v in source_views
    ])

    # ── Held-out view: shift test set ─────────────────────────────────────────
    shift_test_ds = DATASET_REGISTRY[held_out]["cls"](
        split="test", transform=tf,
        download=download, root=data_root,
    )

    # ── Probe: from first source view train split only ────────────────────────
    # Using only one view for the probe keeps the probe distribution clean.
    # Mixed-view probe would conflate two distributions.
    probe_base = DATASET_REGISTRY[source_views[0]]["cls"](
        split="train", transform=tf,
        download=False, root=data_root,
    )
    torch.manual_seed(seed)
    probe_indices = torch.randperm(len(probe_base))[:probe_size].tolist()
    probe_ds      = Subset(probe_base, probe_indices)

    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

    loaders = {
        "train":      make_loader(src_train_ds, shuffle=True),
        "val":        make_loader(src_val_ds),
        "test_iid":   make_loader(src_test_ds),
        "test_shift": make_loader(shift_test_ds),
        "probe":      make_loader(probe_ds),
    }

    logger.info(
        f"[OrganMNIST leave-one-out] held_out={held_out} | "
        f"source_views={source_views} | "
        f"train={len(src_train_ds)} | val={len(src_val_ds)} | "
        f"test_iid={len(src_test_ds)} | test_shift={len(shift_test_ds)} | "
        f"probe={len(probe_ds)}"
    )
    return loaders


def get_all_organ_loaders(
    data_root: str  = "./data/medmnist",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool   = True,
    probe_size: int  = 512,
    seed: int        = 42,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Run all three leave-one-out experiments for OrganMNIST.

    Returns dict with keys:
        "holdout_organamnist"  — held out axial
        "holdout_organcmnist"  — held out coronal
        "holdout_organsmnist"  — held out sagittal

    Results should be averaged across the three held-out experiments
    and reported as mean ± std in the paper.

    Usage:
        organ_loaders = get_all_organ_loaders()
        for name, loaders in organ_loaders.items():
            rho  = probe(model, loaders["probe"])
            metrics = evaluate(model, loaders["test_shift"], rho)
            # collect metrics, average at end
    """
    views = ["organamnist", "organcmnist", "organsmnist"]
    return {
        f"holdout_{v}": get_organ_leave_one_out_loaders(
            held_out_view = v,
            data_root     = data_root,
            batch_size    = batch_size,
            num_workers   = num_workers,
            pin_memory    = pin_memory,
            download      = download,
            probe_size    = probe_size,
            seed          = seed,
        )
        for v in views
    }


# ── Full sweep loader ─────────────────────────────────────────────────────────

def get_all_medmnist_loaders(
    data_root: str = "./data/medmnist",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = True,
    tiers: List[int] = [1, 2],
    skip_multilabel: bool = True,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Load all MedMNIST datasets for the appendix sweep.

    Returns dict: {dataset_name: {train, val, test, probe}}

    skip_multilabel=True skips ChestMNIST (14-class multi-label)
    which requires a different evaluation protocol.

    OrganMNIST is loaded in cross-view mode separately via
    get_cross_view_loaders() — it is not included here to avoid
    loading both views as independent standard datasets.

    TODO: after running the sweep, generate the appendix table
          with one row per dataset using:
              gacs/eval/sweep_table.py  (to be written)
    """
    all_loaders = {}
    organ_views = {"organamnist", "organcmnist", "organsmnist"}

    for name, info in DATASET_REGISTRY.items():
        if info["tier"] not in tiers:
            continue
        if name in organ_views:
            continue   # handled separately via get_cross_view_loaders
        if skip_multilabel and info.get("multilabel", False):
            logger.info(f"Skipping {name} (multi-label)")
            continue

        try:
            all_loaders[name] = get_standard_loaders(
                dataset_name = name,
                data_root    = data_root,
                batch_size   = batch_size,
                num_workers  = num_workers,
                pin_memory   = pin_memory,
                download     = download,
            )
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            continue

    # Add OrganMNIST headline cross-view pair
    all_loaders["organ_axial_to_coronal"] = get_cross_view_loaders(
        source    = "organamnist",
        target    = "organcmnist",
        data_root = data_root,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        download    = download,
    )

    logger.info(
        f"Loaded {len(all_loaders)} MedMNIST datasets: "
        f"{list(all_loaders.keys())}"
    )
    return all_loaders


# ── Dataset metadata helpers ──────────────────────────────────────────────────

def get_dataset_info(dataset_name: str) -> dict:
    """Return metadata for a dataset — n_classes, n_channels, shift type."""
    name = dataset_name.lower()
    assert name in DATASET_REGISTRY, f"Unknown: {name}"
    return DATASET_REGISTRY[name]


def get_tier1_names() -> List[str]:
    """Return names of Tier 1 datasets (main results table)."""
    return [
        name for name, info in DATASET_REGISTRY.items()
        if info["tier"] == 1 and name not in {"organamnist", "organcmnist", "organsmnist"}
    ] + ["organ_axial_to_coronal"]


def get_num_classes(dataset_name: str) -> int:
    """Return number of classes for a dataset."""
    return get_dataset_info(dataset_name)["n_classes"]
