#!/usr/bin/env python3
"""
Inspect Derm7pt Criteria Labels
================================

Print detailed criteria label distribution from meta.csv.

Usage:
    python inspect_derm7pt_labels.py
    python inspect_derm7pt_labels.py --data-root /Users/tanmoy/research/data/derm7pt
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


# CSV columns for the 7-point criteria
CRITERIA_COLS = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]

# Map from string labels in CSV to binary (0=absent, 1=present)
CRITERIA_BINARY_MAP = {
    "pigment_network": {
        "typical": 1,
        "atypical": 1,
        "absent": 0,
        "present": 1,
    },
    "streaks": {
        "irregular": 1,
        "regular": 1,
        "absent": 0,
        "present": 1,
    },
    "pigmentation": {
        "diffuse irregular": 1,
        "localized irregular": 1,
        "diffuse regular": 1,
        "localized regular": 1,
        "absent": 0,
        "present": 1,
    },
    "regression_structures": {
        "blue areas": 1,
        "white areas": 1,
        "combinations": 1,
        "within regression": 1,
        "absent": 0,
        "present": 1,
    },
    "dots_and_globules": {
        "irregular": 1,
        "regular": 1,
        "absent": 0,
        "present": 1,
    },
    "blue_whitish_veil": {
        "present": 1,
        "within regression": 1,
        "absent": 0,
    },
    "vascular_structures": {
        "arborizing": 1,
        "dotted": 1,
        "hairpin": 1,
        "absent": 0,
        "present": 1,
    },
}


def inspect_criteria_distribution(dir_release: str):
    """
    Print criteria label distribution from meta.csv.
    Run this once before training to verify CRITERIA_BINARY_MAP is correct
    and that class imbalance is not extreme.

    TODO: run this before your first training run:
        from gacs.data.derm7pt_loader import inspect_criteria_distribution
        inspect_criteria_distribution("path/to/derm7pt/release_v0")
    """
    meta_path = Path(dir_release) / "meta" / "meta.csv"

    if not meta_path.exists():
        print(f"❌ File not found: {meta_path}")
        print(f"\nExpected path structure:")
        print(f"  {dir_release}/")
        print(f"  └── meta/")
        print(f"      └── meta.csv")
        return False

    meta_df = pd.read_csv(meta_path)

    print("\n" + "=" * 60)
    print("Derm7pt Criteria Label Distributions")
    print("=" * 60)
    print(f"\nDataset: {dir_release}")
    print(f"Total samples: {len(meta_df)}")
    print(f"\nAnalyzing {len(CRITERIA_COLS)} criteria columns from meta.csv")
    print("-" * 60)

    for col in CRITERIA_COLS:
        if col in meta_df.columns:
            counts = meta_df[col].value_counts()
            print(f"\n{col.replace('_', ' ').title()}:")
            print("-" * 60)

            for label, count in counts.items():
                # Handle NaN values
                if pd.isna(label):
                    label_str = "NaN"
                    binary = "UNKNOWN"
                else:
                    label_str = str(label).lower().strip()
                    binary = CRITERIA_BINARY_MAP.get(col, {}).get(
                        label_str, "UNKNOWN"
                    )

                # Calculate percentage
                pct = count / len(meta_df) * 100

                # Visual bar
                bar_length = int(pct / 2)
                bar = "█" * bar_length

                print(f"  {label:<35} n={count:>4} ({pct:>5.1f}%) → binary={binary:<3} {bar}")
        else:
            print(f"\n{col}: ❌ COLUMN NOT FOUND — check meta.csv column names")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Calculate binary conversion statistics
    print("\nBinary Conversion Summary (after mapping):")
    print("-" * 60)

    for col in CRITERIA_COLS:
        if col not in meta_df.columns:
            continue

        # Convert to binary using the map
        def to_binary(label):
            if pd.isna(label):
                return None
            label_str = str(label).lower().strip()
            return CRITERIA_BINARY_MAP.get(col, {}).get(label_str, None)

        binary_values = meta_df[col].apply(to_binary)

        # Count present (1) vs absent (0)
        present_count = (binary_values == 1).sum()
        absent_count = (binary_values == 0).sum()
        unknown_count = (binary_values.isnull()).sum()

        total_valid = present_count + absent_count
        present_pct = present_count / total_valid * 100 if total_valid > 0 else 0

        print(f"{col.replace('_', ' ').title():<40}")
        print(f"  Present (1): {present_count:>5} ({present_pct:>5.1f}%)")
        print(f"  Absent (0):  {absent_count:>5}")
        print(f"  Unknown:     {unknown_count:>5}")

        # Class imbalance warning
        if total_valid > 0:
            ratio = max(present_count, absent_count) / min(present_count, absent_count)
            if ratio > 5:
                print(f"  ⚠️  WARNING: High imbalance (ratio: {ratio:.1f}:1)")

    print("\n" + "=" * 60)
    print("✅ Inspection complete!")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Derm7pt criteria label distributions"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/Users/tanmoy/research/data/derm7pt",
        help="Path to derm7pt data directory"
    )
    args = parser.parse_args()

    success = inspect_criteria_distribution(args.data_root)

    if success:
        print("\n✅ All criteria mapped successfully!")
        print("\nNext steps:")
        print("1. Review the binary mappings above")
        print("2. Check for any 'UNKNOWN' mappings")
        print("3. Look for class imbalance warnings")
        print("4. Update CRITERIA_BINARY_MAP if needed\n")
        return 0
    else:
        print("\n❌ Inspection failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
