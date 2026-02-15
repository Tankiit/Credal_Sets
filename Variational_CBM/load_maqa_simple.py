"""
Simple MAQA data loader - workaround for HuggingFace datasets issue
"""
import torch
import numpy as np
from typing import Dict, List
import json
from pathlib import Path


def create_synthetic_maqa_data(num_samples=3000) -> Dict[str, List[Dict]]:
    """
    Create synthetic MAQA-style data for testing V7b.

    This is a temporary workaround until the HuggingFace dataset issue is resolved.

    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    print("\n" + "="*60)
    print("Creating Synthetic MAQA Data (for testing)")
    print("="*60)

    np.random.seed(42)

    # Generate samples with varying ambiguity levels
    samples = []

    for i in range(num_samples):
        # Random number of answers (2-10)
        num_answers = np.random.randint(2, 11)

        # Generate probability distribution
        # Higher entropy = more uniform
        ambiguity_type = np.random.choice(['low', 'medium', 'high'],
                                          p=[0.3, 0.4, 0.3])

        if ambiguity_type == 'low':
            # Peaked distribution (low entropy)
            probs = np.random.dirichlet(alpha=np.ones(num_answers) * 5)
        elif ambiguity_type == 'medium':
            # Medium entropy
            probs = np.random.dirichlet(alpha=np.ones(num_answers) * 2)
        else:
            # Uniform distribution (high entropy)
            probs = np.random.dirichlet(alpha=np.ones(num_answers) * 0.5)

        # Normalize
        probs = probs / probs.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Generate question text (synthetic)
        questions = [
            f"What is the capital of country {i}?",
            f"Which of the following is true about topic {i}?",
            f"Select the correct answer for question {i}.",
            f"Choose the best option for item {i}.",
        ]
        question = np.random.choice(questions)

        # Get dominant answer
        dominant_answer = int(np.argmax(probs))

        samples.append({
            'question': question,
            'probabilities': probs.tolist(),
            'entropy': float(entropy),
            'num_answers': num_answers,
            'answers': list(range(num_answers)),
            'dominant_answer_idx': dominant_answer,
        })

    # Split into train/val/test
    np.random.shuffle(samples)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))

    splits = {
        'train': samples[:n_train],
        'validation': samples[n_train:n_train + n_val],
        'test': samples[n_train + n_val:]
    }

    print(f"\nCreated {num_samples} synthetic samples:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['validation'])}")
    print(f"  Test: {len(splits['test'])}")

    # Print entropy statistics
    train_entropies = [s['entropy'] for s in splits['train']]
    print(f"\nEntropy statistics:")
    print(f"  Mean: {np.mean(train_entropies):.3f}")
    print(f"  Std: {np.std(train_entropies):.3f}")
    print(f"  Min: {np.min(train_entropies):.3f}")
    print(f"  Max: {np.max(train_entropies):.3f}")

    return splits


def load_combined_maqa_ambigqa(
    split_ratio=(0.8, 0.1, 0.1),
    seed=42,
    use_synthetic=True
):
    """
    Load MAQA dataset (with synthetic fallback).
    """
    if use_synthetic:
        print("\nNote: Using synthetic data (HuggingFace dataset loading issue)")
        return create_synthetic_maqa_data(num_samples=3000)

    # Original implementation (commented out due to datasets library issue)
    try:
        from datasets import load_dataset as hf_load_dataset

        print("\n" + "="*60)
        print("Loading Combined MAQA + AmbigQA Dataset")
        print("="*60)

        # Load MAQA-Star
        print("\nLoading MAQA-Star...")
        maqa_ds = hf_load_dataset("ttomov/maqa_star")["train"]
        print(f"  MAQA: {len(maqa_ds)} samples")

        # Load AmbigQA-Star
        print("Loading AmbigQA-Star...")
        ambigqa_ds = hf_load_dataset("ttomov/ambigqa_star")["train"]
        print(f"  AmbigQA: {len(ambigqa_ds)} samples")

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

        return splits

    except Exception as e:
        print(f"\n⚠️  Error loading from HuggingFace: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_maqa_data(num_samples=3000)


if __name__ == "__main__":
    # Test the loader
    data = load_combined_maqa_ambigqa()
    print(f"\n✓ Successfully loaded MAQA data")
    print(f"  Train: {len(data['train'])} samples")
    print(f"  Validation: {len(data['validation'])} samples")
    print(f"  Test: {len(data['test'])} samples")
