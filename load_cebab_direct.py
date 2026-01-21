"""
CEBaB Dataset Loader
====================

Loads the CEBaB (Counterfactual Explanations for the Bootstrapped Restaurant Dataset)
from HuggingFace with enhanced concept encoding.

This dataset is PERFECT for concept bottleneck models:
- 4 concepts: food, service, ambiance, noise
- 3-level labels: Negative, Positive, unknown
- Multi-annotator disagreement → aleatoric uncertainty
- Counterfactual edits for robustness

Author: Tanmoy
Date: January 2026
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
import json


def load_cebab(
    split_ids_path: Optional[str] = None,
    include_edits: bool = True,
    seed: int = 42
) -> Dict[str, List]:
    """
    Load CEBaB dataset from HuggingFace.

    Args:
        split_ids_path: Path to split IDs JSON (optional)
        include_edits: Whether to include counterfactual edits
        seed: Random seed for reproducibility

    Returns:
        Dict with 'train', 'validation', 'test' splits
    """
    print("\n" + "="*60)
    print("Loading CEBaB Dataset")
    print("="*60)

    # Load from HuggingFace
    print("Loading from HuggingFace...")
    ds = hf_load_dataset("CEBaB/CEBaB")

    print(f"  Loaded splits: {list(ds.keys())}")
    for split_name in ds.keys():
        print(f"    {split_name}: {len(ds[split_name])} samples")

    # Use official splits if available
    if 'train' in ds and 'validation' in ds and 'test' in ds:
        print("\nUsing official dataset splits...")
        splits = {
            'train': list(ds['train']),
            'validation': list(ds['validation']),
            'test': list(ds['test'])
        }
    else:
        # Create splits manually
        print("\nCreating train/val/test splits...")
        all_data = []
        for split in ds.values():
            all_data.extend(list(split))

        # Shuffle
        np.random.seed(seed)
        indices = np.random.permutation(len(all_data))

        n_train = int(len(all_data) * 0.7)
        n_val = int(len(all_data) * 0.15)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        splits = {
            'train': [all_data[i] for i in train_indices],
            'validation': [all_data[i] for i in val_indices],
            'test': [all_data[i] for i in test_indices]
        }

    print(f"\n  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['validation'])} samples")
    print(f"  Test: {len(splits['test'])} samples")

    return splits


def process_cebab_raw(raw_data: List[Dict]) -> List[Dict]:
    """
    Process raw CEBaB data into format expected by VCBM.

    Key features:
    - 4 concepts: food, service, ambiance, noise
    - 3-level labels: Negative (0), unknown (1), Positive (2)
    - Multi-annotator disagreement for aleatoric uncertainty
    - 5-star rating prediction task

    Args:
        raw_data: List of raw CEBaB samples

    Returns:
        List of processed samples
    """
    processed = []

    # Label mapping
    aspect_to_idx = {
        'food': 0,
        'service': 1,
        'ambiance': 2,
        'noise': 3
    }

    label_to_idx = {
        'Negative': 0,
        'unknown': 1,
        'Positive': 2
    }

    for item in tqdm(raw_data, desc="Processing CEBaB"):
        # Extract text
        description = item.get('description', '')
        if not description or len(description.strip()) == 0:
            continue

        # ====================================================================
        # Extract concept labels (4 aspects)
        # ====================================================================
        concepts = np.zeros(4, dtype=np.int64)
        is_unknown = np.zeros(4, dtype=np.float32)
        concept_distributions = []  # For computing disagreement

        for aspect_name, aspect_idx in aspect_to_idx.items():
            majority = item.get(f'{aspect_name}_aspect_majority', 'unknown')

            # Map to 0/1/2
            if majority in label_to_idx:
                concepts[aspect_idx] = label_to_idx[majority]
                is_unknown[aspect_idx] = 1.0 if majority == 'unknown' else 0.0

            # Get distribution for disagreement computation
            dist_str = item.get(f'{aspect_name}_aspect_label_distribution', '{}')
            if dist_str:
                # Parse JSON string
                try:
                    dist = json.loads(dist_str) if isinstance(dist_str, str) else dist_str
                except:
                    dist = {}

                # Convert to probabilities
                total = sum(dist.values()) if isinstance(dist, dict) else 0
                if total > 0:
                    probs = [
                        dist.get('Negative', 0) / total,
                        dist.get('unknown', 0) / total,
                        dist.get('Positive', 0) / total
                    ]
                    concept_distributions.append(probs)
                else:
                    concept_distributions.append([0.0, 1.0, 0.0])  # All unknown
            else:
                concept_distributions.append([0.0, 1.0, 0.0])

        # ====================================================================
        # Extract task label (5-star rating)
        # ====================================================================
        review_majority = item.get('review_majority', 'no majority')

        # Map to 0-4 (for 5 classes)
        if review_majority == 'no majority':
            # Use abstention class or map to middle
            task_label = 2  # Middle rating (3 stars)
        else:
            try:
                rating = int(review_majority)
                task_label = rating - 1  # Convert 1-5 to 0-4
            except:
                task_label = 2  # Default to middle

        # ====================================================================
        # Compute annotator disagreement (aleatoric uncertainty signal)
        # ====================================================================
        # For each concept, compute entropy of label distribution
        concept_entropies = []
        for probs in concept_distributions:
            probs = np.array(probs)
            probs = probs / (probs.sum() + 1e-10)  # Normalize

            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(3)  # 3 classes
            normalized_entropy = entropy / (max_entropy + 1e-10)

            concept_entropies.append(normalized_entropy)

        concept_entropies = np.array(concept_entropies)

        # ====================================================================
        # Compute overall disagreement
        # ====================================================================
        review_dist_str = item.get('review_label_distribution', '{}')
        if review_dist_str:
            # Parse JSON string
            try:
                review_dist = json.loads(review_dist_str) if isinstance(review_dist_str, str) else review_dist_str
            except:
                review_dist = {}

            if review_dist:
                total = sum(review_dist.values()) if isinstance(review_dist, dict) else 0
                if total > 0:
                    rating_probs = [
                        review_dist.get('1', 0) / total,
                        review_dist.get('2', 0) / total,
                        review_dist.get('3', 0) / total,
                        review_dist.get('4', 0) / total,
                        review_dist.get('5', 0) / total
                    ]
                    rating_probs = np.array(rating_probs)
                    rating_probs = rating_probs / (rating_probs.sum() + 1e-10)

                    # Rating entropy
                    rating_entropy = -np.sum(rating_probs * np.log(rating_probs + 1e-10))
                    max_rating_entropy = np.log(5)
                    normalized_rating_entropy = rating_entropy / (max_rating_entropy + 1e-10)
                else:
                    normalized_rating_entropy = 0.0
            else:
                normalized_rating_entropy = 0.0
        else:
            normalized_rating_entropy = 0.0

        # ====================================================================
        # Store additional metadata
        # ====================================================================
        processed.append({
            'text': description,
            'label': task_label,  # 0-4 (5-star rating)
            'concepts': concepts,  # [4] - food, service, ambiance, noise
            'is_unknown': is_unknown,  # [4] - which concepts are unknown

            # For analysis
            '_concept_entropies': concept_entropies.astype(np.float32),
            '_rating_entropy': float(normalized_rating_entropy),
            '_overall_disagreement': float(concept_entropies.mean()),

            # Original data
            '_review_majority': review_majority,
            '_edit_goal': item.get('edit_goal'),
            '_edit_type': item.get('edit_type'),
            '_is_original': item.get('is_original', True),
        })

    return processed


def load_cebab_direct(
    include_edits: bool = True,
    seed: int = 42
) -> Dict:
    """
    Main function to load CEBaB dataset.

    Returns dict with:
        - 'train': list of processed training samples
        - 'val': list of processed validation samples
        - 'test': list of processed test samples
    """
    # Load
    raw_splits = load_cebab(include_edits=include_edits, seed=seed)

    # Process
    processed = {}
    for split, data in raw_splits.items():
        print(f"\nProcessing {split}...")
        processed[split] = process_cebab_raw(data)

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    for split in ["train", "validation", "test"]:
        data = processed.get(split, [])
        print(f"\n{split.capitalize()}:")
        print(f"  Samples: {len(data)}")

        if len(data) > 0:
            # Rating distribution
            labels = [d['label'] for d in data]
            for rating in range(5):
                count = labels.count(rating)
                print(f"    Rating {rating+1}: {count} ({count/len(data)*100:.1f}%)")

            # Concept statistics
            concepts = np.array([d['concepts'] for d in data])
            unknown_rates = np.array([d['is_unknown'] for d in data]).mean(axis=0)

            concept_names = ['food', 'service', 'ambiance', 'noise']
            print(f"  Concept unknown rates:")
            for i, name in enumerate(concept_names):
                print(f"    {name}: {unknown_rates[i]:.2%}")

            # Disagreement statistics
            disagreements = [d['_overall_disagreement'] for d in data]
            print(f"  Mean disagreement: {np.mean(disagreements):.3f} ± {np.std(disagreements):.3f}")

            # Rating entropy
            rating_entropies = [d['_rating_entropy'] for d in data]
            print(f"  Mean rating entropy: {np.mean(rating_entropies):.3f} ± {np.std(rating_entropies):.3f}")

    return processed


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class CEBaBDataset:
    """PyTorch Dataset for CEBaB."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item["label"], dtype=torch.long),
            'concept_labels': torch.tensor(item["concepts"], dtype=torch.long),
            'is_unknown': torch.tensor(item["is_unknown"], dtype=torch.float),
        }

        # Add concept entropies as annotator_entropy (for concept-supervised model)
        if '_concept_entropies' in item:
            result['annotator_entropy'] = torch.tensor(item["_concept_entropies"], dtype=torch.float)

        # Add optional metadata
        for key in ['_rating_entropy', '_overall_disagreement']:
            if key in item:
                result[key] = torch.tensor(item[key], dtype=torch.float)

        return result


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def get_cebab_dataloaders(
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 4,
    include_edits: bool = True
):
    """
    Get PyTorch DataLoaders for CEBaB.

    Returns train_loader, val_loader, test_loader, tokenizer, metadata
    """
    from torch.utils.data import DataLoader

    # Load data
    processed_data = load_cebab_direct(include_edits=include_edits)

    # Create datasets
    train_dataset = CEBaBDataset(
        processed_data["train"], tokenizer, max_length
    )
    val_dataset = CEBaBDataset(
        processed_data["validation"], tokenizer, max_length
    )
    test_dataset = CEBaBDataset(
        processed_data["test"], tokenizer, max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Metadata
    metadata = {
        "dataset_name": "cebab",
        "task": "sentiment_with_concepts",
        "num_classes": 5,  # 5-star ratings
        "num_concepts": 4,  # food, service, ambiance, noise
        "concept_names": ['food', 'service', 'ambiance', 'noise'],
        "class_names": ['1_star', '2_star', '3_star', '4_star', '5_star'],
        "has_concepts": True,
        "has_multi_annotator": True,
        "is_ordinal": True,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, tokenizer, metadata


# ============================================================================
# VALIDATION METRICS FOR CONCEPT BOTTLENECK MODELS
# ============================================================================

def compute_cebab_validation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    concept_probs: np.ndarray,
    concept_labels: np.ndarray,
    unknown_mask: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    concept_entropies: np.ndarray,
    rating_entropy: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive validation metrics for CEBaB.

    Key validations:
    1. Concept accuracy (known concepts only)
    2. Aleatoric correlates with disagreement
    3. Epistemic correlates with concept errors
    4. EU-AU separation
    """
    from scipy import stats

    metrics = {}

    # ========================================================================
    # 1. Task-level metrics
    # ========================================================================
    metrics['accuracy'] = (predictions == labels).mean()

    # Per-class accuracy
    for c in range(5):
        mask = (labels == c)
        if mask.sum() > 0:
            metrics[f'acc_rating_{c+1}'] = (predictions[mask] == labels[mask]).mean()

    # ========================================================================
    # 2. Concept-level metrics
    # ========================================================================
    concept_names = ['food', 'service', 'ambiance', 'noise']

    for k in range(4):
        # Convert to binary (positive vs not positive)
        c_labels_k = (concept_labels[:, k] / 2.0)  # Map 0/1/2 to 0/0.5/1
        c_preds_k = concept_probs[:, k]

        # Known mask (exclude unknown)
        known_mask_k = (concept_labels[:, k] != 1)

        if known_mask_k.sum() > 0:
            # Binary accuracy (positive vs negative)
            c_labels_binary = (c_labels_k[known_mask_k] > 0.5).astype(int)
            c_preds_binary = (c_preds_k[known_mask_k] > 0.5).astype(int)

            c_acc = (c_preds_binary == c_labels_binary).mean()
            metrics[f'concept_{concept_names[k]}_acc'] = c_acc

            # Correlation with disagreement
            if known_mask_k.sum() > 1:
                # Errors on known concepts
                c_errors = (c_preds_binary != c_labels_binary).astype(float)

                if c_errors.std() > 0 and epistemic.std() > 0:
                    rho, p = stats.spearmanr(epistemic, c_errors[known_mask_k])
                    metrics[f'concept_{concept_names[k]}_rho_epi_err'] = rho

    # ========================================================================
    # 3. Uncertainty-Disagreement Correlations
    # ========================================================================

    # 3a. Aleatoric vs Concept Disagreement
    # Aggregate concept entropies
    mean_concept_entropy = concept_entropies.mean(axis=-1) if concept_entropies.ndim > 1 else concept_entropies

    if mean_concept_entropy.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(aleatoric, mean_concept_entropy)
        metrics['rho_ale_concept_disagreement'] = rho
        metrics['p_ale_concept_disagreement'] = p

    # 3b. Aleatoric vs Rating Disagreement
    if rating_entropy.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(aleatoric, rating_entropy)
        metrics['rho_ale_rating_disagreement'] = rho
        metrics['p_ale_rating_disagreement'] = p

    # 3c. Aleatoric vs Unknown Mask
    unknown_rate = unknown_mask.mean(axis=-1) if unknown_mask.ndim > 1 else unknown_mask
    if unknown_rate.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(aleatoric, unknown_rate)
        metrics['rho_ale_unknown'] = rho
        metrics['p_ale_unknown'] = p

    # ========================================================================
    # 4. Epistemic-Aleatoric Separation
    # ========================================================================
    if epistemic.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(epistemic, aleatoric)
        metrics['rho_epi_ale'] = rho
        metrics['p_epi_ale'] = p
        metrics['separation_quality'] = 1 - abs(rho)

    # ========================================================================
    # 5. Epistemic vs Task Error
    # ========================================================================
    errors = (predictions != labels).astype(float)
    if errors.std() > 0 and epistemic.std() > 0:
        rho, p = stats.spearmanr(epistemic, errors)
        metrics['rho_epi_error'] = rho
        metrics['p_epi_error'] = p

    # ========================================================================
    # 6. Statistics
    # ========================================================================
    metrics['mean_epistemic'] = epistemic.mean()
    metrics['std_epistemic'] = epistemic.std()
    metrics['mean_aleatoric'] = aleatoric.mean()
    metrics['std_aleatoric'] = aleatoric.std()
    metrics['mean_concept_entropy'] = mean_concept_entropy.mean()
    metrics['mean_rating_entropy'] = rating_entropy.mean()

    # Per-concept uncertainty
    for k in range(4):
        metrics[f'epistemic_concept_{concept_names[k]}'] = epistemic[:, k].mean() if epistemic.ndim > 1 else epistemic.mean()
        metrics[f'aleatoric_concept_{concept_names[k]}'] = aleatoric[:, k].mean() if aleatoric.ndim > 1 else aleatoric.mean()

    return metrics


if __name__ == "__main__":
    # Test loading
    print("\n" + "="*60)
    print("Testing CEBaB Loader")
    print("="*60)

    data = load_cebab_direct()

    print(f"\n✓ Successfully loaded CEBaB!")
    print(f"  Train: {len(data['train'])}")
    print(f"  Val: {len(data['validation'])}")
    print(f"  Test: {len(data['test'])}")

    # Show sample
    if len(data['train']) > 0:
        sample = data['train'][0]
        print(f"\nSample:")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Label (rating): {sample['label'] + 1} stars")
        print(f"  Concepts: {sample['concepts']}")
        concept_names = ['food', 'service', 'ambiance', 'noise']
        label_names = {0: 'Negative', 1: 'unknown', 2: 'Positive'}
        for i, name in enumerate(concept_names):
            print(f"    {name}: {label_names[sample['concepts'][i]]}")
        print(f"  Unknown mask: {sample['is_unknown']}")
        print(f"  Concept disagreement: {sample['_overall_disagreement']:.3f}")
        print(f"  Rating entropy: {sample['_rating_entropy']:.3f}")
