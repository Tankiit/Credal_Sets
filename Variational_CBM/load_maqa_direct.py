"""
MAQA-Star Dataset Loader
=========================

Loads the Multiple Answer QA (MAQA) dataset with ground-truth answer distributions.
This is the GOLD STANDARD for validating uncertainty decomposition:

- Ground-truth p* (answer distribution) → Validate AU captures ambiguity
- Multiple valid answers → Natural aleatoric uncertainty
- Annotator counts → Compute entropy

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


def load_maqa_star(
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Dict[str, List]:
    """
    Load MAQA-Star dataset and create train/val/test splits.

    Args:
        split_ratio: (train, val, test) ratio, must sum to 1.0
        seed: Random seed for reproducible splits

    Returns:
        Dict with 'train', 'validation', 'test' splits
    """
    print("\n" + "="*60)
    print("Loading MAQA-Star Dataset")
    print("="*60)

    # Load from HuggingFace
    print("Loading from HuggingFace...")
    ds = hf_load_dataset("ttomov/maqa_star")
    raw_data = ds['train']

    print(f"  Loaded {len(raw_data)} samples")

    # Create splits (deterministic)
    np.random.seed(seed)
    indices = np.random.permutation(len(raw_data))
    n_train = int(len(raw_data) * split_ratio[0])
    n_val = int(len(raw_data) * split_ratio[1])

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")

    # Split data
    splits = {
        'train': [raw_data[i] for i in train_indices],
        'validation': [raw_data[i] for i in val_indices],
        'test': [raw_data[i] for i in test_indices]
    }

    return splits


def process_maqa_raw(raw_data: List[Dict]) -> List[Dict]:
    """
    Process raw MAQA-Star data into format expected by VCBM.

    Key features:
    - Extract ground-truth answer distribution p*
    - Compute entropy as ground-truth aleatoric uncertainty
    - Create question+answer encoding for classification

    Expected input format:
    {
        'question': str,
        'rephrased_question': str,
        'answers': List[List[str]],  # Multiple valid answers
        'statement': List[List[str]],  # Answer statements
        'probabilities': List[float],  # Ground-truth distribution!
        'counts': List[int],  # Annotator counts
        'main_keywords': List[str],
        'additional_keywords': List[str]
    }

    Returns:
        List of processed samples with:
        - text: Question text
        - answers: List of valid answers
        - p_star: Ground-truth answer distribution
        - entropy: Ground-truth aleatoric uncertainty
        - num_answers: Number of valid answers
        - keywords: Combined keywords
    """
    processed = []

    for item in tqdm(raw_data, desc="Processing MAQA-Star"):
        # Extract fields
        question = item.get('question', '')
        rephrased = item.get('rephrased_question', '')
        answers = item.get('answers', [])
        statements = item.get('statement', [])
        probabilities = item.get('probabilities', [])
        counts = item.get('counts', [])
        keywords = item.get('main_keywords', []) + item.get('additional_keywords', [])

        # Clean answers (extract from nested lists)
        if answers and isinstance(answers[0], list):
            answers = [a[0] for a in answers if a]

        if statements and isinstance(statements[0], list):
            statements = [s[0] for s in statements if s]

        # Skip if no valid answers
        if not answers:
            continue

        # ====================================================================
        # Compute ground-truth distribution p*
        # ====================================================================
        probs = np.array(probabilities, dtype=np.float64)

        # Normalize (handle all-zero cases)
        if probs.sum() == 0:
            # Use uniform distribution as fallback
            probs = np.ones(len(answers)) / len(answers)
        else:
            probs = probs / probs.sum()

        # ====================================================================
        # Compute ground-truth entropy (AU)
        # ====================================================================
        # H(p*) = -sum_i p_i * log(p_i)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / (max_entropy + 1e-10)  # Normalize to [0, 1]

        # ====================================================================
        # Compute ambiguity level
        # ====================================================================
        # Ambiguity can be measured by:
        # 1. Entropy (high entropy = high ambiguity)
        # 2. Number of answers (more answers = more ambiguous)
        # 3. Probability mass concentration (dominant answer = less ambiguous)

        dominant_prob = probs.max()
        effective_num_answers = np.exp(entropy)  # Perplexity

        # Classify ambiguity level
        if dominant_prob >= 0.8:
            ambiguity_level = 0  # Low (clear answer)
        elif dominant_prob >= 0.5:
            ambiguity_level = 1  # Medium (some ambiguity)
        else:
            ambiguity_level = 2  # High (very ambiguous)

        # ====================================================================
        # Create text input
        # ====================================================================
        # Option 1: Just question
        text = question

        # Option 2: Question + rephrased (can help with understanding)
        # text = f"Question: {question} Rephrased: {rephrased}"

        # Option 3: Question + keywords
        # keywords_str = ", ".join(keywords[:5])  # Top 5 keywords
        # text = f"Question: {question} Keywords: {keywords_str}"

        # ====================================================================
        # Store additional metadata
        # ====================================================================
        processed.append({
            'text': text,
            'answers': answers,
            'statements': statements,
            'p_star': probs.astype(np.float32),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'num_answers': len(answers),
            'ambiguity_level': int(ambiguity_level),
            'effective_num_answers': float(effective_num_answers),
            'dominant_prob': float(dominant_prob),
            'keywords': keywords,
            'counts': counts,
            # For classification: create pseudo-labels
            # (use dominant answer as "label" for supervised training)
            'dominant_answer_idx': int(probs.argmax()),
            # Metadata for analysis
            '_question': question,
            '_rephrased_question': rephrased,
        })

    return processed


def load_maqa_direct(
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Dict:
    """
    Main function to load MAQA-Star dataset.

    Returns dict with:
        - 'train': list of processed training samples
        - 'val': list of processed validation samples
        - 'test': list of processed test samples
    """
    # Load
    raw_splits = load_maqa_star(split_ratio=split_ratio, seed=seed)

    # Process
    processed = {}
    for split, data in raw_splits.items():
        print(f"\nProcessing {split}...")
        processed[split] = process_maqa_raw(data)

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    for split in ["train", "validation", "test"]:
        data = processed.get(split, [])
        print(f"\n{split.capitalize()}:")
        print(f"  Samples: {len(data)}")

        if len(data) > 0:
            # Answer statistics
            num_answers = [d['num_answers'] for d in data]
            print(f"  Num answers: {np.mean(num_answers):.2f} ± {np.std(num_answers):.2f}")

            # Entropy statistics (ambiguity)
            entropies = [d['entropy'] for d in data]
            print(f"  Entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
            print(f"  Normalized entropy: {np.mean([d['normalized_entropy'] for d in data]):.3f}")

            # Ambiguity level distribution
            ambiguity_levels = [d['ambiguity_level'] for d in data]
            level_names = {0: 'Low', 1: 'Medium', 2: 'High'}
            for level in [0, 1, 2]:
                count = ambiguity_levels.count(level)
                print(f"    {level_names[level]} ambiguity: {count} ({count/len(data)*100:.1f}%)")

            # Effective number of answers (perplexity)
            effective_nums = [d['effective_num_answers'] for d in data]
            print(f"  Effective num answers: {np.mean(effective_nums):.2f} ± {np.std(effective_nums):.2f}")

    return processed


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class MAQADataset:
    """PyTorch Dataset for MAQA-Star."""

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

            # For pseudo-supervised training: use dominant answer as label
            'labels': torch.tensor(item["dominant_answer_idx"], dtype=torch.long),

            # Ground-truth distribution (for validation)
            'p_star': torch.tensor(item["p_star"], dtype=torch.float),

            # Ground-truth uncertainty metrics
            'entropy': torch.tensor(item["entropy"], dtype=torch.float),
            'normalized_entropy': torch.tensor(item["normalized_entropy"], dtype=torch.float),
            'num_answers': torch.tensor(item["num_answers"], dtype=torch.long),
            'ambiguity_level': torch.tensor(item["ambiguity_level"], dtype=torch.long),

            # Additional metadata
            'effective_num_answers': torch.tensor(item["effective_num_answers"], dtype=torch.float),
            'dominant_prob': torch.tensor(item["dominant_prob"], dtype=torch.float),
        }

        # Optional: add answer text for analysis
        if 'answers' in item:
            result['_answers'] = item['answers']
        if '_question' in item:
            result['_question'] = item['_question']

        return result


def collate_maqa_batch(batch):
    """
    Custom collate function to handle variable-length p_star vectors.
    """
    # Find max number of answers in batch
    max_answers = max(item['p_star'].shape[0] for item in batch)

    batch_data = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),

        # Pad p_star to max_answers
        'p_star': torch.zeros(len(batch), max_answers),
        'p_star_mask': torch.zeros(len(batch), max_answers, dtype=torch.bool),  # True = valid

        'entropy': torch.stack([item['entropy'] for item in batch]),
        'normalized_entropy': torch.stack([item['normalized_entropy'] for item in batch]),
        'num_answers': torch.stack([item['num_answers'] for item in batch]),
        'ambiguity_level': torch.stack([item['ambiguity_level'] for item in batch]),
        'effective_num_answers': torch.stack([item['effective_num_answers'] for item in batch]),
        'dominant_prob': torch.stack([item['dominant_prob'] for item in batch]),
    }

    # Fill in p_star with padding
    for i, item in enumerate(batch):
        n_answers = item['p_star'].shape[0]
        batch_data['p_star'][i, :n_answers] = item['p_star']
        batch_data['p_star_mask'][i, :n_answers] = True

    return batch_data


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def get_maqa_dataloaders(
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 4
):
    """
    Get PyTorch DataLoaders for MAQA-Star.

    Returns train_loader, val_loader, test_loader, tokenizer, metadata
    """
    from torch.utils.data import DataLoader

    # Load data
    processed_data = load_maqa_direct(split_ratio=split_ratio)

    # Create datasets
    train_dataset = MAQADataset(
        processed_data["train"], tokenizer, max_length
    )
    val_dataset = MAQADataset(
        processed_data["validation"], tokenizer, max_length
    )
    test_dataset = MAQADataset(
        processed_data["test"], tokenizer, max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_maqa_batch,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_maqa_batch,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_maqa_batch,
        pin_memory=True
    )

    # Metadata
    metadata = {
        "dataset_name": "maqa_star",
        "task": "qa_with_ambiguity",
        "num_classes": None,  # Variable: depends on max num answers
        "max_num_answers": max(d['num_answers'] for d in processed_data['train']),
        "avg_num_answers": np.mean([d['num_answers'] for d in processed_data['train']]),
        "has_ground_truth_distribution": True,  # KEY FEATURE!
        "has_multi_annotator": True,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, tokenizer, metadata


# ============================================================================
# VALIDATION METRICS FOR UNCERTAINTY DECOMPOSITION
# ============================================================================

def compute_maqa_validation_metrics(
    predicted_probs: np.ndarray,  # [N, max_answers] model predictions
    p_star: np.ndarray,           # [N, max_answers] ground-truth distribution
    p_star_mask: np.ndarray,      # [N, max_answers] valid mask
    epistemic: np.ndarray,        # [N] epistemic uncertainty
    aleatoric: np.ndarray,        # [N] aleatoric uncertainty
    entropy: np.ndarray,          # [N] ground-truth entropy
    ambiguity_level: np.ndarray   # [N] ground-truth ambiguity level
) -> Dict[str, float]:
    """
    Compute validation metrics for uncertainty decomposition.

    This is the GOLD STANDARD validation:
    - Aleatoric should correlate with H(p*) (ground-truth entropy)
    - Epistemic should correlate with KL(p*||p) (distance to truth)
    - They should be decorrelated from each other
    """
    metrics = {}

    # ========================================================================
    # 1. Mask out padding
    # ========================================================================
    # Only consider valid answer positions
    N = p_star.shape[0]

    # Align predicted_probs with p_star size
    max_p_star_answers = p_star_mask.shape[1]
    if predicted_probs.shape[1] > max_p_star_answers:
        predicted_probs = predicted_probs[:, :max_p_star_answers]

    # Compute per-sample metrics (only over valid answers)
    kl_divergences = []
    for i in range(N):
        valid_mask = p_star_mask[i]
        if valid_mask.sum() > 0:
            # KL(p* || p)
            p_true = p_star[i, valid_mask]
            p_pred = predicted_probs[i, valid_mask]

            # Avoid log(0)
            p_true = np.clip(p_true, 1e-10, 1.0)
            p_pred = np.clip(p_pred, 1e-10, 1.0)

            kl = np.sum(p_true * np.log(p_true / p_pred))
            kl_divergences.append(kl)

    kl_divergences = np.array(kl_divergences)

    # ========================================================================
    # 2. MAIN VALIDATION METRICS
    # ========================================================================
    from scipy import stats

    # 2a. Aleatoric vs Ground-Truth Entropy (should be POSITIVE)
    if entropy.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(aleatoric, entropy)
        metrics['rho_ale_entropy'] = rho
        metrics['p_ale_entropy'] = p
        # Target: rho > 0.3 (aleatoric captures ambiguity)

    # 2b. Aleatoric vs Ambiguity Level (should be POSITIVE)
    if ambiguity_level.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(aleatoric, ambiguity_level)
        metrics['rho_ale_ambiguity'] = rho
        metrics['p_ale_ambiguity'] = p

    # 2c. Epistemic vs KL(p*||p) (should be POSITIVE)
    # Higher epistemic = further from ground-truth distribution
    if kl_divergences.std() > 0 and epistemic.std() > 0:
        rho, p = stats.spearmanr(epistemic, kl_divergences)
        metrics['rho_epi_kl'] = rho
        metrics['p_epi_kl'] = p
        # Target: rho > 0.3 (epistemic captures error)

    # 2d. Epistemic-Aleatoric correlation (should be LOW)
    if epistemic.std() > 0 and aleatoric.std() > 0:
        rho, p = stats.spearmanr(epistemic, aleatoric)
        metrics['rho_epi_ale'] = rho
        metrics['p_epi_ale'] = p
        metrics['separation_quality'] = 1 - abs(rho)
        # Target: abs(rho) < 0.1 (good separation)

    # ========================================================================
    # 3. Additional metrics
    # ========================================================================
    metrics['mean_entropy'] = entropy.mean()
    metrics['mean_kl_divergence'] = kl_divergences.mean()
    metrics['mean_epistemic'] = epistemic.mean()
    metrics['mean_aleatoric'] = aleatoric.mean()

    return metrics


if __name__ == "__main__":
    # Test loading
    print("\n" + "="*60)
    print("Testing MAQA-Star Loader")
    print("="*60)

    data = load_maqa_direct()

    print(f"\n✓ Successfully loaded MAQA-Star!")
    print(f"  Train: {len(data['train'])}")
    print(f"  Val: {len(data['validation'])}")
    print(f"  Test: {len(data['test'])}")

    # Show sample
    if len(data['train']) > 0:
        sample = data['train'][0]
        print(f"\nSample:")
        print(f"  Question: {sample['text'][:100]}...")
        print(f"  Answers: {sample['answers'][:3]}")
        print(f"  p*: {sample['p_star'][:3]}")
        print(f"  Entropy: {sample['entropy']:.3f}")
        print(f"  Ambiguity Level: {sample['ambiguity_level']} (0=Low, 1=Medium, 2=High)")
