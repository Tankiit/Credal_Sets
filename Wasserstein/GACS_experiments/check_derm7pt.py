#!/usr/bin/env python3
"""
Check Derm7pt Dataset
=====================

Utility to check if Derm7pt dataset exists and inspect its structure.

Usage:
    python check_derm7pt.py
    python check_derm7pt.py --data-root /path/to/derm7pt/release_v0
"""

import argparse
import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent))


def check_derm7pt_structure(data_root):
    """Check if Derm7pt dataset has the expected structure."""
    data_path = Path(data_root)

    print(f"\n{'='*60}")
    print(f"Checking Derm7pt Dataset Structure")
    print(f"{'='*60}\n")

    print(f"Data root: {data_path}")
    print(f"Exists: {data_path.exists()}\n")

    if not data_path.exists():
        print("❌ Data root not found!")
        print("\nExpected structure:")
        print("  derm7pt/release_v0/")
        print("  ├── images/          # Clinical images")
        print("  ├── meta/            # Metadata JSON files")
        print("  └── labels/          # Optional: label files")
        return False

    # Check subdirectories
    print("Checking subdirectories:")
    required_dirs = ["images", "meta"]
    optional_dirs = ["labels"]

    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {dir_name}/")

        if exists:
            # List some contents
            files = list(dir_path.glob("*"))[:5]
            if files:
                print(f"      Files: {', '.join([f.name for f in files])}")

    for dir_name in optional_dirs:
        dir_path = data_path / dir_name
        exists = dir_path.exists()
        if exists:
            print(f"  ✓ {dir_name}/ (optional)")
            files = list(dir_path.glob("*"))[:3]
            if files:
                print(f"      Files: {', '.join([f.name for f in files])}")

    # Check for metadata files
    print("\nChecking metadata files:")
    meta_dir = data_path / "meta"
    if meta_dir.exists():
        json_files = list(meta_dir.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files")

        # Check for expected split files
        expected_splits = ["train", "val", "test"]
        for split in expected_splits:
            matching = [f for f in json_files if split in f.stem.lower()]
            if matching:
                print(f"    ✓ {split}: {matching[0].name}")

    # Count images
    print("\nChecking images:")
    images_dir = data_path / "images"
    if images_dir.exists():
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))

        print(f"  Found {len(image_files)} images")

        if image_files:
            print(f"  Sample files:")
            for img in image_files[:5]:
                print(f"    - {img.name}")

    print(f"\n{'='*60}\n")

    return True


def inspect_derm7pt(data_root):
    """Use the derm7pt_loader to inspect the dataset."""
    try:
        from gacs.data.derm7pt_loader import inspect_criteria_distribution

        print("Attempting to inspect dataset criteria distribution...\n")

        # Try train split first
        try:
            inspect_criteria_distribution(data_root, split="train")
        except FileNotFoundError as e:
            print(f"Train split not found: {e}")

            # Try other splits
            for split in ["val", "test"]:
                try:
                    print(f"\nTrying {split} split...")
                    inspect_criteria_distribution(data_root, split=split)
                    break
                except FileNotFoundError:
                    continue

        return True

    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Derm7pt dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/derm7pt/release_v0",
        help="Path to Derm7pt data directory"
    )
    args = parser.parse_args()

    # Check structure
    structure_ok = check_derm7pt_structure(args.data_root)

    if not structure_ok:
        print("\n❌ Dataset structure check failed!")
        print("\nNext steps:")
        print("1. Download Derm7pt dataset")
        print("2. Extract it to the expected location")
        print("3. Run this script again")
        return 1

    # Try to inspect
    print("Attempting to inspect dataset content...")
    inspect_ok = inspect_derm7pt(args.data_root)

    if inspect_ok:
        print("\n✅ Dataset inspection complete!")
        print("\nYou can now use Derm7pt for training:")
        print(f"  python train_medmnist.py --dataset derm7pt --data-root {args.data_root}")
    else:
        print("\n⚠️  Dataset exists but inspection failed")
        print("This might be due to missing metadata files or unexpected structure")

    return 0 if inspect_ok else 1


if __name__ == "__main__":
    sys.exit(main())
