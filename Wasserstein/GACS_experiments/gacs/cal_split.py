# gacs/cal_split.py
"""
Unified calibration split strategy for GACS + CP baselines.
Works for both CEBaB (NLP) and MedMNIST (vision).

Strategy:
    val  →  val_early_stop (~70%, ~1,170 for CEBaB)  ← early stopping only
         →  val_cp_cal     (~30%, ~500  for CEBaB)   ← GACS ε + CP threshold

Same split object is passed to:
    - calibrate_eps()         in cebab_fixes.py
    - SplitPredictor.calibrate()  in baselines_cp.py  (TorchCP)
    - CrepesPredictor.calibrate() in baselines_cp.py  (crepes)

This eliminates double-dipping: val_early_stop never touches calibration,
val_cp_cal never touches early stopping.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────
# Core split function — works for any Dataset
# ─────────────────────────────────────────────

def split_val_for_cp(
    val_dataset,
    cp_cal_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple:
    """
    Split a val Dataset into (val_early_stop, val_cp_cal).

    Args:
        val_dataset:   any torch Dataset (CEBaB embeddings or MedMNIST)
        cp_cal_ratio:  fraction to reserve for CP/GACS calibration
        seed:          reproducibility — fix this and report in paper §5

    Returns:
        val_early_stop_ds, val_cp_cal_ds

    Sizing guidance:
        CEBaB val ~1,670:
            cp_cal_ratio=0.30 → ~500 cal, ~1,170 early_stop
            cp_cal_ratio=0.25 → ~420 cal, ~1,250 early_stop
            Minimum for stable CP threshold: ~300 (LAC), ~500 (RAPS)

        MedMNIST val sizes vary by dataset:
            DermaMNIST  val ~1,005 → 0.30 gives ~300 cal (borderline)
            PathMNIST   val ~7,180 → 0.30 gives ~2,150 cal (comfortable)
            BloodMNIST  val ~3,421 → 0.30 gives ~1,000 cal (good)
            OrganMNIST  val ~  236 → 0.30 gives ~70  cal (TOO SMALL — see below)

        QUESTION: OrganMNIST val is tiny (~236). Options:
            A) Use a portion of train for CP cal on OrganMNIST only
            B) Use cp_cal_ratio=0.50 (still only ~118 — unstable)
            C) Skip CP baselines for OrganMNIST (it's the shift experiment,
               CP fails anyway — absence of CP baseline is justified)
        Recommendation: Option C — CP failing on OrganMNIST cross-view shift
        is expected and your paper already argues this. Report it as
        "CP baselines excluded for OrganMNIST: val set too small for
        stable threshold; this dataset is used exclusively for shift evaluation."
    """
    n_total = len(val_dataset)
    n_cp_cal = int(n_total * cp_cal_ratio)
    n_early_stop = n_total - n_cp_cal

    generator = torch.Generator().manual_seed(seed)
    val_early_stop_ds, val_cp_cal_ds = random_split(
        val_dataset,
        [n_early_stop, n_cp_cal],
        generator=generator,
    )
    return val_early_stop_ds, val_cp_cal_ds


def make_cal_loaders(
    val_dataset,
    batch_size: int = 32,
    cp_cal_ratio: float = 0.30,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Returns:
        {
          "val_early_stop": DataLoader,   ← pass to trainer for early stopping
          "val_cp_cal":     DataLoader,   ← pass to calibrate_eps() AND CP calibration
        }

    Usage:
        cal_loaders = make_cal_loaders(val_dataset)
        trainer.train(val_loader=cal_loaders["val_early_stop"])
        eps_min, eps_max = calibrate_eps(model, cal_loaders["val_cp_cal"], rho)
        tcp.calibrate(cal_loaders["val_cp_cal"], alpha=0.1)
        ccp.calibrate(cal_probs, cal_labels)   # probs from val_cp_cal
    """
    val_early_stop_ds, val_cp_cal_ds = split_val_for_cp(
        val_dataset, cp_cal_ratio=cp_cal_ratio, seed=seed
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return {
        "val_early_stop": DataLoader(val_early_stop_ds, **loader_kwargs),
        "val_cp_cal":     DataLoader(val_cp_cal_ds,     **loader_kwargs),
    }


# ─────────────────────────────────────────────
# NLP (CEBaB) — integrates with cebab_fixes.py
# ─────────────────────────────────────────────

def build_cebab_cal_split(
    cebab_val_dataset,       # PreCachedEmbeddingDataset from cebab_fixes.py
    batch_size: int = 32,
    cp_cal_ratio: float = 0.30,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    CEBaB-specific wrapper around make_cal_loaders.

    Expected val_dataset type: PreCachedEmbeddingDataset
        (already L2-normalized BERT embeddings — satisfies D1)

    Returns same dict as make_cal_loaders.

    QUESTION: CEBaB labels — are you using binary (2-class) or 5-class?
        This affects CP threshold stability:
        Binary:  large prediction sets less likely, LAC scores cluster near 0/1
        5-class: APS/RAPS more useful, sets can be size 1-5
    """
    # MPS-safe: pin_memory=False, num_workers=0
    return make_cal_loaders(
        cebab_val_dataset,
        batch_size=batch_size,
        cp_cal_ratio=cp_cal_ratio,
        seed=seed,
        num_workers=0,
        pin_memory=False,
    )


# ─────────────────────────────────────────────
# Vision (MedMNIST) — per-dataset sizing logic
# ─────────────────────────────────────────────

# Val set sizes from MedMNIST official splits
MEDMNIST_VAL_SIZES = {
    "dermamnist":  1005,
    "pathmnist":   7180,
    "bloodmnist":  3421,
    "organmnist":   236,   # too small — see below
}

# Minimum cal set size for stable CP thresholds
# LAC needs ~200+, RAPS needs ~400+ for reliable quantile estimation
MIN_CP_CAL_SIZE = {
    "LAC":  200,
    "APS":  300,
    "RAPS": 400,
}

def get_medmnist_cp_cal_ratio(dataset_name: str) -> Optional[float]:
    """
    Returns the cp_cal_ratio to use for a given MedMNIST dataset,
    or None if CP baselines should be skipped.

    Logic:
        - If 0.30 * val_size >= 400 (RAPS minimum): use 0.30
        - If 0.30 * val_size >= 200 (LAC minimum):  use 0.30, skip RAPS
        - If val_size too small:                     return None (skip CP)

    Called by build_medmnist_cal_split() to decide ratio automatically.
    """
    name = dataset_name.lower()
    val_size = MEDMNIST_VAL_SIZES.get(name)

    if val_size is None:
        # TODO: add new datasets to MEDMNIST_VAL_SIZES above
        raise ValueError(f"Unknown MedMNIST dataset: {dataset_name}")

    n_cal = int(val_size * 0.30)

    if n_cal >= MIN_CP_CAL_SIZE["RAPS"]:
        return 0.30   # all three score functions viable
    elif n_cal >= MIN_CP_CAL_SIZE["LAC"]:
        return 0.30   # LAC/APS viable, skip RAPS
    else:
        return None   # skip CP baselines entirely


def build_medmnist_cal_split(
    dataset_name: str,
    medmnist_val_dataset,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    MedMNIST-specific wrapper. Automatically handles per-dataset sizing.

    Args:
        dataset_name:       e.g. "dermamnist", "pathmnist", "organmnist"
        medmnist_val_dataset: standard MedMNIST val split as torch Dataset

    Returns:
        dict with "val_early_stop" and "val_cp_cal" loaders,
        OR raises SkipCPBaselines for OrganMNIST (caller should handle).

    Usage pattern:
        try:
            cal_loaders = build_medmnist_cal_split("organmnist", val_ds)
        except SkipCPBaselines as e:
            print(f"Skipping CP baselines: {e}")
            cal_loaders = {"val_early_stop": full_val_loader, "val_cp_cal": None}
    """
    ratio = get_medmnist_cp_cal_ratio(dataset_name)

    if ratio is None:
        raise SkipCPBaselines(
            f"{dataset_name} val set too small for stable CP threshold "
            f"(n_val={MEDMNIST_VAL_SIZES[dataset_name.lower()]}). "
            f"CP baselines excluded. Report in paper: "
            f"'CP baselines are not reported for {dataset_name} as the "
            f"validation set is insufficient for stable threshold estimation; "
            f"this dataset is used exclusively for distribution shift evaluation.'"
        )

    return make_cal_loaders(
        medmnist_val_dataset,
        batch_size=batch_size,
        cp_cal_ratio=ratio,
        seed=seed,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


class SkipCPBaselines(Exception):
    """Raised when a dataset's val set too small for CP calibration."""
    pass


# ─────────────────────────────────────────────
# Reporting — call before experiments
# Output goes into paper §5 Experimental Setup
# ─────────────────────────────────────────────

def report_cal_split_sizes(
    dataset_name: str,
    cal_loaders: Dict[str, DataLoader],
):
    """
    Prints split sizes for the methods section.

    Example output for CEBaB:
        ┌─────────────────────────────────────────────────────┐
        │ CEBaB calibration split (seed=42, cp_cal_ratio=0.30)│
        ├───────────────────┬─────────┬────────────────────── │
        │ Split             │    Size │ Used for              │
        ├───────────────────┼─────────┼───────────────────────│
        │ val_early_stop    │   1,169 │ Early stopping        │
        │ val_cp_cal        │     501 │ GACS ε + CP threshold │
        └───────────────────┴─────────┴───────────────────────┘
    """
    print(f"\n{dataset_name} calibration split:")
    rows = [
        ("val_early_stop", "Early stopping only"),
        ("val_cp_cal",     "GACS ε calibration + CP threshold (LAC, APS, RAPS)"),
    ]
    for key, purpose in rows:
        if key in cal_loaders and cal_loaders[key] is not None:
            n = len(cal_loaders[key].dataset)
            print(f"  {key:20s}  {n:6,}  ← {purpose}")


def report_viable_score_functions(dataset_name: str):
    """
    Prints which CP score functions are viable for a dataset.
    Call this before running baselines to know what to include in the table.

    Example output:
        DermaMNIST CP baselines:
          LAC   ✓  (n_cal=301 >= 200)
          APS   ✓  (n_cal=301 >= 300)
          RAPS  ✗  (n_cal=301 < 400) — excluded, report in appendix footnote
    """
    name = dataset_name.lower()
    val_size = MEDMNIST_VAL_SIZES.get(name, 1670)  # 1670 = CEBaB default
    n_cal = int(val_size * 0.30)

    print(f"\n{dataset_name} viable CP score functions (n_cal={n_cal}):")
    for fn, min_size in MIN_CP_CAL_SIZE.items():
        viable = n_cal >= min_size
        mark = "✓" if viable else "✗"
        note = "" if viable else f" — excluded (n_cal < {min_size})"
        print(f"  {fn:6s}  {mark}  (min required: {min_size}){note}")
