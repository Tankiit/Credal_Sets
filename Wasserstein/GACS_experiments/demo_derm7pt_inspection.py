#!/usr/bin/env python3
"""
Demo: Derm7pt Criteria Distribution Inspection
==============================================

This script demonstrates how the derm7pt_loader inspection would work
with actual data. Since the dataset isn't downloaded yet, this shows
the expected output format.

Usage:
    python demo_derm7pt_inspection.py
"""

import numpy as np
from typing import Dict


def demo_criteria_distribution():
    """Show what the criteria distribution output looks like."""

    print(f"\n{'='*60}")
    print(f"Derm7pt Criteria Distribution [DEMO]")
    print(f"{'='*60}\n")

    print(f"Total samples: 1,247 (simulated)\n")

    # Simulated criteria distribution (typical values)
    criteria_dist = {
        "pigment_network": 0.68,
        "regression": 0.23,
        "atypical_vascular": 0.15,
        "irregular_streaks": 0.42,
        "atypical_pigment_network": 0.35,
        "atypical_dots_globules": 0.28,
        "blue_white_veil": 0.19,
    }

    print("7-Point Criteria Distribution:")
    print("-" * 60)
    for crit_name, freq in criteria_dist.items():
        bar = "█" * int(freq * 50)
        print(f"  {crit_name:30s} {freq:6.2%}  {bar}")

    # Label distribution
    print("\n" + "=" * 60)
    print("Label Distribution:")
    print("-" * 60)
    label_dist = {0: 0.72, 1: 0.28}  # 72% benign, 28% malignant

    for label, freq in sorted(label_dist.items()):
        label_name = "Malignant" if label == 1 else "Benign"
        bar = "█" * int(freq * 50)
        print(f"  {label_name:10s} ({label}) {freq:6.2%}  {bar}")

    # Co-occurrence (simulated)
    print("\n" + "=" * 60)
    print("Criteria Co-occurrence (Top Pairs):")
    print("-" * 60)

    # Typical co-occurrence patterns in melanoma
    cooccurrence_pairs = [
        ("pigment_network", "atypical_pigment_network", 245, 19.6),
        ("irregular_streaks", "pigment_network", 198, 15.9),
        ("atypical_dots_globules", "atypical_pigment_network", 156, 12.5),
        ("blue_white_veil", "regression", 134, 10.7),
        ("atypical_vascular", "regression", 89, 7.1),
        ("irregular_streaks", "atypical_dots_globules", 78, 6.3),
        ("blue_white_veil", "atypical_pigment_network", 67, 5.4),
        ("pigment_network", "atypical_dots_globules", 145, 11.6),
        ("regression", "atypical_vascular", 56, 4.5),
        ("blue_white_veil", "irregular_streaks", 45, 3.6),
    ]

    for crit1, crit2, count, pct in cooccurrence_pairs:
        print(f"  {crit1:20s} + {crit2:20s} : {count:4d} ({pct:5.1f}%)")

    print(f"\n{'='*60}\n")

    print("Key Insights:")
    print("-" * 60)
    print("1. Pigment network is most common (68%)")
    print("2. Atypical vascular is rarest (15%)")
    print("3. 28% of cases are malignant (higher than typical population)")
    print("4. Pigment + atypical network co-occurs most (19.6%)")
    print("5. Blue-white veil often co-occurs with regression (10.7%)")
    print()

    return {
        "criteria": criteria_dist,
        "labels": label_dist,
        "num_samples": 1247,
    }


def demo_usage():
    """Show how to use the inspect_criteria_distribution function."""

    print("\n" + "=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60 + "\n")

    print("With actual Derm7pt data, use:")
    print("-" * 60)
    print("""
from gacs.data.derm7pt_loader import inspect_criteria_distribution

# Inspect training split
dist = inspect_criteria_distribution(
    "path/to/derm7pt/release_v0",
    split="train"
)

# Access the data programmatically
print(f"Criteria distribution: {dist['criteria']}")
print(f"Label distribution: {dist['labels']}")
print(f"Total samples: {dist['num_samples']}")
""")

    print("\nOr from command line:")
    print("-" * 60)
    print("""
python check_derm7pt.py --data-root path/to/derm7pt/release_v0
""")

    print("\nExpected output structure:")
    print("-" * 60)
    print("""
{
    "criteria": {
        "pigment_network": 0.68,
        "regression": 0.23,
        ...
    },
    "labels": {
        0: 0.72,  # Benign
        1: 0.28,  # Malignant
    },
    "num_samples": 1247
}
""")

    print()


if __name__ == "__main__":
    # Run demo
    result = demo_criteria_distribution()

    # Show usage
    demo_usage()

    print("=" * 60)
    print("📊 Demo complete!")
    print("=" * 60)
    print("\nTo use with real data:")
    print("1. Download Derm7pt dataset")
    print("2. Extract to data/derm7pt/release_v0/")
    print("3. Run: python check_derm7pt.py")
    print()
