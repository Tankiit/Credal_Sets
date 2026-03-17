#!/usr/bin/env python3
"""
Test Derm7pt Data Loading
=========================

Quick test to verify Derm7pt loader works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gacs.data.derm7pt_loader import get_derm7pt_loaders
import torch

def test_derm7pt_loading():
    """Test Derm7pt data loading."""

    data_dir = "/Users/tanmoy/research/data/derm7pt"

    print(f"Loading Derm7pt data from: {data_dir}")
    print("-" * 70)

    try:
        loaders = get_derm7pt_loaders(
            dir_release=data_dir,
            batch_size=8,
            num_workers=0,
        )

        print("\n✓ Data loaded successfully!")
        print("\nDataset splits:")
        for split_name, loader in loaders.items():
            if loader is not None:
                print(f"  {split_name:15s}: {len(loader.dataset):4d} samples")
            else:
                print(f"  {split_name:15s}: None")

        # Test a batch
        print("\n" + "-" * 70)
        print("Testing batch loading...")

        for split_name in ["train", "val", "test_iid", "test_shift"]:
            if loaders[split_name] is not None:
                loader = loaders[split_name]
                batch = next(iter(loader))
                x, y = batch

                print(f"\n{split_name.upper()}:")
                print(f"  Input shape:  {x.shape}")
                print(f"  Label shape:  {y.shape}")
                print(f"  Input range:  [{x.min():.3f}, {x.max():.3f}]")
                print(f"  Labels:       {y[:5].tolist()}")
                break  # Just test first available split

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_derm7pt_loading()
    sys.exit(0 if success else 1)
