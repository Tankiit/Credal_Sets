"""
MAQA Data Loader - Real data from HuggingFace
"""
import numpy as np
from typing import Dict, List


def load_combined_maqa_ambigqa(
    split_ratio=(0.8, 0.1, 0.1),
    seed=42
):
    """
    Load real MAQA + AmbigQA data from HuggingFace.

    Now that datasets library is upgraded to 4.5.0, this should work!
    """
    print("\n" + "="*60)
    print("Loading Combined MAQA + AmbigQA Dataset")
    print("="*60)

    from datasets import load_dataset as hf_load_dataset

    # Load MAQA-Star
    print("\nLoading MAQA-Star...")
    maqa_ds = hf_load_dataset("ttomov/maqa_star")["train"]
    print(f"  ✓ MAQA: {len(maqa_ds)} samples")

    # Load AmbigQA-Star
    print("Loading AmbigQA-Star...")
    ambigqa_ds = hf_load_dataset("ttomov/ambigqa_star")["train"]
    print(f"  ✓ AmbigQA: {len(ambigqa_ds)} samples")

    # Combine datasets
    combined = []
    combined.extend([{"dataset": "maqa", **x} for x in maqa_ds])
    combined.extend([{"dataset": "ambigqa", **x} for x in ambigqa_ds])

    print(f"\n✓ Combined: {len(combined)} total samples")

    # Create splits
    np.random.seed(seed)
    indices = np.random.permutation(len(combined))
    n_train = int(len(combined) * split_ratio[0])
    n_val = int(len(combined) * split_ratio[1])

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        'train': [combined[i] for i in train_indices],
        'validation': [combined[i] for i in val_indices],
        'test': [combined[i] for i in test_indices]
    }

    print(f"\nSplits:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['validation'])}")
    print(f"  Test: {len(splits['test'])}")

    return splits


if __name__ == "__main__":
    data = load_combined_maqa_ambigqa()
    print(f"\n✓ Successfully loaded real MAQA data!")
    print(f"  Total samples: {len(data['train']) + len(data['validation']) + len(data['test'])}")
