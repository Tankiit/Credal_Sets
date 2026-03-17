"""
gacs/data/derm7pt_loader.py
---------------------------
Derm7pt data loader with dermoscopic → clinical modality shift.

Shift design:
    Train/val/test_iid : dermoscopic images  (specialist equipment)
    test_shift         : clinical/macro images (standard camera, same lesions)

This is a genuine acquisition modality shift that occurs in real clinical
deployment — dermoscopes are not always available at triage.

Both image types cover the same 1011 lesions, so label distribution is
identical between test_iid and test_shift. This is a controlled shift:
only the imaging modality changes, not the case mix.

Dataset structure (from official derm7pt release):
    meta/
        meta.csv           — all metadata including criteria labels
        train_indexes.csv  — official train split indexes
        valid_indexes.csv  — official val split indexes
        test_indexes.csv   — official test split indexes
    images/
        derm/              — dermoscopic images
        clinic/            — clinical/macro images

Factor supervision:
    K=7 supervised factors from ground-truth criteria labels in meta.csv.
    Criteria: pigment_network, streaks, pigmentation, regression_structures,
              dots_and_globules, blue_whitish_veil, vascular_structures.
    Each criterion mapped to binary {0=absent, 1=present}.

Reference:
    Kawahara et al., "Seven-Point Checklist and Skin Lesion Classification
    using Multitask Multimodal Neural Nets," IEEE JBHI, 2019.
"""

from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# ── Criteria column names (from meta.csv) ─────────────────────────────────────
# Confirmed column order from meta.csv inspection
# Maps to paper criterion indices 0–6
CRITERIA_COLS = [
    "pigment_network",       # Criterion 0
    "regression_structures", # Criterion 1
    "vascular_structures",   # Criterion 2
    "streaks",               # Criterion 3
    "pigmentation",          # Criterion 4
    "dots_and_globules",     # Criterion 5
    "blue_whitish_veil",     # Criterion 6
]

# Confirmed diagnosis column
DIAGNOSIS_COL = "diagnosis"
DIAGNOSIS_MAP = {
    "melanoma":                  0,
    "melanocytic nevus":         1,
    "basal cell carcinoma":      2,
    "seborrheic keratosis":      3,
    "dermatofibroma":            4,
    "vascular lesion":           5,
    "squamous cell carcinoma":   6,
}

# Image filename columns — confirmed from meta.csv column list
# Column "derm"   → dermoscopic image filename  (e.g. "Aal001.jpg")
# Column "clinic" → clinical image filename     (e.g. "Aal001.jpg")
DERM_FILENAME_COL   = "derm"
CLINIC_FILENAME_COL = "clinic"


# ── Binary criteria mapping ────────────────────────────────────────────────────

# Each criterion has multiple labels. We map to binary: absent=0, present=1.
# For criteria with graded presence (typical/atypical), both map to 1.
CRITERIA_BINARY_MAP = {
    "pigment_network": {
        "absent": 0, "typical": 1, "atypical": 1,
    },
    "regression_structures": {
        "absent": 0, "present": 1,
        "blue areas": 1, "white areas": 1, "combinations": 1,
        "within regression": 1,
    },
    "vascular_structures": {
        "absent": 0, "arborizing": 1, "dotted": 1, "hairpin": 1,
        "within regression": 1, "comma": 1, "linear irregular": 1,
        "wreath": 1,
    },
    "streaks": {
        "absent": 0, "regular": 1, "irregular": 1,
    },
    "pigmentation": {
        "absent": 0,
        "diffuse regular": 1, "localised regular": 1,
        "localized regular": 1, "local regular": 1,
        "diffuse irregular": 1, "localised irregular": 1,
        "localized irregular": 1, "local irregular": 1,
    },
    "dots_and_globules": {
        "absent": 0, "regular": 1, "irregular": 1,
    },
    "blue_whitish_veil": {
        "absent": 0, "present": 1,
    },
}


def criteria_row_to_binary(row: pd.Series) -> List[int]:
    """Convert one row of meta.csv criteria columns to binary list."""
    result = []
    for col in CRITERIA_COLS:
        val = str(row.get(col, "absent")).lower().strip()
        mapping = CRITERIA_BINARY_MAP.get(col, {})
        result.append(mapping.get(val, 0))   # default absent if unknown
    return result


# ── Dataset class ──────────────────────────────────────────────────────────────

class Derm7ptDataset(Dataset):
    """
    Derm7pt dataset for GACS.

    Returns tuples of (image, diagnosis_label, criteria_labels).

    Parameters
    ----------
    meta_df    : pandas DataFrame with metadata (subset of meta.csv)
    image_dir  : path to images/derm/ or images/clinic/
    transform  : torchvision transform
    """

    def __init__(
        self,
        meta_df:      pd.DataFrame,
        image_dir:    str,
        filename_col: str,           # "derm" or "clinic" — confirmed column names
        transform=None,
    ):
        self.meta         = meta_df.reset_index(drop=True)
        self.image_dir    = Path(image_dir)
        self.filename_col = filename_col   # which column holds the image filename
        self.transform    = transform

        # Pre-compute binary criteria labels for all rows
        self.criteria = torch.tensor(
            [criteria_row_to_binary(row) for _, row in self.meta.iterrows()],
            dtype=torch.float32,
        )  # [N, 7]

        # Diagnosis labels
        self.diag_labels = torch.tensor(
            [
                DIAGNOSIS_MAP.get(
                    str(row[DIAGNOSIS_COL]).lower().strip(), -1
                )
                for _, row in self.meta.iterrows()
            ],
            dtype=torch.long,
        )  # [N]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        row = self.meta.iloc[idx]

        # Use confirmed filename column (set at dataset construction time)
        filename = str(row[self.filename_col])
        img_path = self.image_dir / filename

        # Fallback: try without extension, then add .jpg
        if not img_path.exists():
            img_path = self.image_dir / f"{filename}.jpg"
        if not img_path.exists():
            img_path = self.image_dir / f"{filename}.png"

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Check that image_dir='{self.image_dir}' is correct and "
                f"filename column '{self.filename_col}' contains valid filenames."
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.diag_labels[idx], self.criteria[idx]


# ── Transforms ────────────────────────────────────────────────────────────────

def get_derm7pt_transform(img_size: int = 224, augment: bool = False):
    """
    Transform for derm7pt images.

    Uses ImageNet normalisation because ResNet50 encoder was pretrained
    on ImageNet. Both dermoscopic and clinical images are RGB photographs,
    so ImageNet stats are appropriate here (unlike greyscale CT in OrganMNIST).

    Augmentation: horizontal flip only. Vertical flip and rotation are
    clinically valid for dermoscopy — lesion orientation does not carry
    diagnostic meaning. Random colour jitter is useful for modality
    robustness (derm vs clinic have different colour properties).
    """
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ]
    if augment:
        base = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            ),
        ] + base

    return transforms.Compose(base)


# ── Main loader function ───────────────────────────────────────────────────────

def get_derm7pt_loaders(
    dir_release: str,
    img_size:    int   = 224,
    batch_size:  int   = 32,
    num_workers: int   = 0,      # MPS safe
    pin_memory:  bool  = False,  # MPS safe
    probe_size:  int   = 256,    # smaller — derm7pt is ~2k total
    seed:        int   = 42,
) -> Dict[str, DataLoader]:
    """
    Load derm7pt with dermoscopic → clinical modality shift.

    Returns dict with keys:
        train       — derm images, official train split
        val         — derm images, official val split
        test_iid    — derm images, official test split   (in-distribution)
        test_shift  — clinical images, all test indexes  (modality shift)
        probe       — derm images, subset of train       (for geometric probing)

    Directory structure expected:
        {dir_release}/
            meta/
                meta.csv
                train_indexes.csv
                valid_indexes.csv
                test_indexes.csv
            images/
                derm/       ← dermoscopic images
                clinic/     ← clinical/macro images
    """
    dir_release = Path(dir_release)
    dir_meta    = dir_release / "meta"
    dir_derm    = dir_release / "images" / "derm"
    dir_clinic  = dir_release / "images" / "clinic"

    # Load metadata and official split indexes
    meta_df       = pd.read_csv(dir_meta / "meta.csv")
    train_indexes = list(pd.read_csv(dir_meta / "train_indexes.csv")["indexes"])
    valid_indexes = list(pd.read_csv(dir_meta / "valid_indexes.csv")["indexes"])
    test_indexes  = list(pd.read_csv(dir_meta / "test_indexes.csv")["indexes"])

    # Filter out any rows with unknown diagnosis
    known_diag_mask = meta_df[DIAGNOSIS_COL].str.lower().str.strip().isin(
        [k.lower() for k in DIAGNOSIS_MAP.keys()]
    )
    if not known_diag_mask.all():
        n_unknown = (~known_diag_mask).sum()
        logger.warning(
            f"{n_unknown} rows with unknown diagnosis labels — "
            f"these will return label -1 and should be filtered from loaders. "
            f"Unknown values: {meta_df[~known_diag_mask][DIAGNOSIS_COL].unique()}"
        )

    # Transforms
    tf_train = get_derm7pt_transform(img_size, augment=True)
    tf_eval  = get_derm7pt_transform(img_size, augment=False)

    # Build datasets using confirmed filename columns
    # Images are in subdirectories (derm/ and clinic/)
    train_ds = Derm7ptDataset(
        meta_df.iloc[train_indexes], dir_derm, "derm", tf_train
    )
    val_ds = Derm7ptDataset(
        meta_df.iloc[valid_indexes], dir_derm, "derm", tf_eval
    )
    test_iid_ds = Derm7ptDataset(
        meta_df.iloc[test_indexes], dir_derm, "derm", tf_eval
    )
    # Shift: same test cases, clinical image column
    test_shift_ds = Derm7ptDataset(
        meta_df.iloc[test_indexes], dir_clinic, "clinic", tf_eval
    )
    probe_base_ds = Derm7ptDataset(
        meta_df.iloc[train_indexes], dir_derm, "derm", tf_eval
    )
    torch.manual_seed(seed)
    probe_indices = torch.randperm(len(probe_base_ds))[:probe_size].tolist()
    probe_ds      = Subset(probe_base_ds, probe_indices)

    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

    loaders = {
        "train":      make_loader(train_ds, shuffle=True),
        "val":        make_loader(val_ds),
        "test_iid":   make_loader(test_iid_ds),
        "test_shift": make_loader(test_shift_ds),
        "probe":      make_loader(probe_ds),
    }

    logger.info(
        f"[Derm7pt] train={len(train_ds)} | val={len(val_ds)} | "
        f"test_iid={len(test_iid_ds)} | test_shift={len(test_shift_ds)} | "
        f"probe={len(probe_ds)} | "
        f"shift_type=dermoscopic→clinical"
    )

    # Sanity check: warn if test_shift dir is missing
    if not dir_clinic.exists():
        logger.warning(
            f"Clinical image directory not found: {dir_clinic}\n"
            f"test_shift loader will fail at first batch.\n"
            f"Confirm your release includes both derm/ and clinic/ subdirectories."
        )

    return loaders


# ── Criteria label inspection helper ─────────────────────────────────────────

def inspect_criteria_distribution(dir_release: str):
    """
    Print criteria label distribution from meta.csv.
    Run this once before training to verify CRITERIA_BINARY_MAP is correct
    and that class imbalance is not extreme.

    TODO: run this before your first training run:
        from gacs.data.derm7pt_loader import inspect_criteria_distribution
        inspect_criteria_distribution("path/to/derm7pt/release_v0")
    """
    meta_df = pd.read_csv(Path(dir_release) / "meta" / "meta.csv")
    print("\n── Derm7pt criteria label distributions ──────────────────")
    for col in CRITERIA_COLS:
        if col in meta_df.columns:
            counts = meta_df[col].value_counts()
            print(f"\n{col}:")
            for label, count in counts.items():
                binary = CRITERIA_BINARY_MAP.get(col, {}).get(
                    str(label).lower().strip(), "UNKNOWN"
                )
                print(f"  {label:<35} n={count:>4}  → binary={binary}")
        else:
            print(f"\n{col}: COLUMN NOT FOUND — check meta.csv column names")
    print("─" * 55)


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/Users/tanmoy/research/data/derm7pt"
    inspect_criteria_distribution(data_root)
