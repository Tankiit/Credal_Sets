"""
Direct HateXplain Dataset Loader
==============================

Bypasses the deprecated HuggingFace dataset script and loads data directly.

Author: Tanmoy
Date: January 2026
"""

import json
import requests
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm


def download_hatexplain_data(save_dir: str = "./data/hatexplain") -> Dict[str, List]:
    """
    Load HateXplain dataset from local data directory.

    Expects:
    - dataset.json: Main data file with all samples
    - classes.npy: Label encoder (3 classes)
    - post_id_divisions.json: Train/val/test splits

    Returns dict with train, validation, test splits.
    """
    save_dir = Path(save_dir)

    print(f"Loading HateXplain from local directory: {save_dir}")

    # Load label encoder
    classes_file = save_dir / "classes.npy"
    if not classes_file.exists():
        raise FileNotFoundError(f"Label encoder file not found: {classes_file}")

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.classes_ = np.load(classes_file, allow_pickle=True)
    print(f"  Loaded label encoder with classes: {list(encoder.classes_)}")

    # Load main dataset
    dataset_file = save_dir / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        all_data = json.load(f)

    print(f"  Loaded dataset with {len(all_data)} samples")

    # Load post ID divisions
    divisions_file = save_dir / "post_id_divisions.json"
    if not divisions_file.exists():
        raise FileNotFoundError(f"Post ID divisions file not found: {divisions_file}")

    with open(divisions_file, 'r') as f:
        divisions = json.load(f)

    # Split data according to post_id_divisions
    train_posts = set(divisions.get("train", []))
    val_posts = set(divisions.get("val", []))  # Note: file uses 'val' not 'valid'
    test_posts = set(divisions.get("test", []))

    train_data = []
    val_data = []
    test_data = []

    for post_id, sample in tqdm(all_data.items(), desc="Splitting data"):
        if post_id in train_posts:
            train_data.append(sample)
        elif post_id in val_posts:
            val_data.append(sample)
        elif post_id in test_posts:
            test_data.append(sample)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }


def process_hatexplain_raw(raw_data: List[Dict]) -> List[Dict]:
    """
    Process raw HateXplain data into format expected by VCBM.

    Expected format for each item:
    {
        "post_id": "24198545_gab",
        "annotators": [
            {
                "label": "hatespeech",  # or "offensive", "normal"
                "annotator_id": 4,
                "target": ["African"]
            },
            ...
        ],
        "rationales": [[0,1,0,...], ...],
        "post_tokens": ["and", "this", "is", ...]
    }

    Returns list of processed samples.
    """
    processed = []

    # Label mapping
    label_to_id = {
        "hatespeech": 0,
        "normal": 1,
        "offensive": 2
    }

    for item in tqdm(raw_data, desc="Processing HateXplain"):
        # Get tokens
        tokens = item.get("post_tokens", [])
        if not tokens:
            continue

        # Join tokens into text
        text = " ".join(tokens)
        if len(text.strip()) == 0:
            continue

        # Get annotators
        annotators = item.get("annotators", [])
        if not annotators or len(annotators) == 0:
            continue

        # Extract labels and targets
        labels = []
        targets = []
        for ann in annotators:
            label_str = ann.get("label", "")
            if label_str in label_to_id:
                labels.append(label_to_id[label_str])

            target_list = ann.get("target", [])
            targets.append(target_list)

        if not labels:
            continue

        # ====================================================================
        # Compute majority label
        # ====================================================================
        # Label encoding: 0 = hatespeech, 1 = normal, 2 = offensive
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Get majority label
        majority_label = max(label_counts.items(), key=lambda x: x[1])[0]

        # ====================================================================
        # Compute enhanced concepts
        # ====================================================================
        # Concept 1: has_target (ternary)
        # - 0 = No target identified by any annotator
        # - 1 = Disagreement (some found target, some didn't)
        # - 2 = Clear target (majority/all found target)

        annotators_with_targets = sum(
            1 for t in targets
            if isinstance(t, list) and len(t) > 0
        )

        target_ratio = annotators_with_targets / len(targets)

        if target_ratio == 0:
            target_concept = 0  # No target
        elif target_ratio >= 0.67:
            target_concept = 2  # Clear target
        else:
            target_concept = 1  # Unclear/Disagree

        # Concept 2: is_offensive (ternary)
        # - 0 = Normal (majority say normal)
        # - 1 = Disagreement (mixed labels)
        # - 2 = Offensive (majority say offensive/hatespeech)

        offensive_count = sum(1 for l in labels if l in [0, 2])  # hatespeech or offensive
        normal_count = sum(1 for l in labels if l == 1)

        offensive_ratio = offensive_count / len(labels)

        if offensive_ratio >= 0.67:
            offensive_concept = 2  # Offensive
        elif normal_count / len(labels) >= 0.67:
            offensive_concept = 0  # Normal
        else:
            offensive_concept = 1  # Unclear/Disagree

        # ====================================================================
        # Optional: Compute metadata for analysis
        # ====================================================================
        severity_score = sum(1 for l in labels if l == 0) / len(labels)  # Hatespeech ratio
        consensus_score = max(label_counts.values()) / len(labels)  # Agreement ratio

        # Compute annotator entropy
        label_probs = np.array([label_counts.get(i, 0) / len(labels) for i in range(3)])
        label_probs = label_probs[label_probs > 0]
        if len(label_probs) > 0:
            annotator_entropy = -np.sum(label_probs * np.log(label_probs + 1e-10))
            annotator_entropy = annotator_entropy / np.log(3)  # Normalize to [0, 1]
        else:
            annotator_entropy = 0.0

        # ====================================================================
        # Create processed sample
        # ====================================================================
        processed.append({
            "text": text,
            "label": majority_label,
            "concepts": np.array([target_concept, offensive_concept], dtype=np.int64),
            "is_unknown": np.array([
                target_concept == 1,
                offensive_concept == 1
            ], dtype=np.float32),
            # Optional metadata
            "_severity_score": severity_score,
            "_consensus_score": consensus_score,
            "_annotator_entropy": annotator_entropy,
            "_raw_labels": labels,
        })

    return processed


def load_hatexplain_direct(save_dir: str = "./data/hatexplain") -> Dict:
    """
    Main function to load HateXplain dataset.

    Returns dict with:
        - 'train': list of processed training samples
        - 'val': list of processed validation samples
        - 'test': list of processed test samples
    """
    print("\n" + "="*60)
    print("Loading HateXplain Dataset (Direct)")
    print("="*60)

    # Download
    raw_data = download_hatexplain_data(save_dir)

    # Process
    processed = {}
    for split, data in raw_data.items():
        print(f"\nProcessing {split}...")
        processed[split] = process_hatexplain_raw(data)

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    for split in ["train", "validation", "test"]:
        data = processed.get(split, [])
        print(f"\n{split.capitalize()}:")
        print(f"  Samples: {len(data)}")

        if len(data) > 0:
            # Label distribution
            labels = [d["label"] for d in data]
            label_names = {0: "hatespeech", 1: "normal", 2: "offensive"}
            for label_id, name in label_names.items():
                count = labels.count(label_id)
                print(f"    {name}: {count} ({count/len(data)*100:.1f}%)")

            # Concept statistics
            concepts = np.array([d["concepts"] for d in data])
            unknown_rates = np.array([d["is_unknown"] for d in data]).mean(axis=0)

            print(f"  Target unknown rate: {unknown_rates[0]:.2%}")
            print(f"  Offensive unknown rate: {unknown_rates[1]:.2%}")

    return processed


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class DirectHateXplainDataset:
    """PyTorch Dataset for HateXplain loaded directly."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._cache_tokenized_inputs()

    def __len__(self):
        return len(self.data)

    def _cache_tokenized_inputs(self) -> None:
        texts = [item["text"] for item in self.data]
        if not texts:
            self.input_ids = torch.empty((0, self.max_length), dtype=torch.long)
            self.attention_mask = torch.empty((0, self.max_length), dtype=torch.long)
            return

        encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        self.input_ids = encoding['input_ids']
        self.attention_mask = encoding['attention_mask']

    def __getitem__(self, idx):
        item = self.data[idx]

        result = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(item["label"], dtype=torch.long),
            'concept_labels': torch.tensor(item["concepts"], dtype=torch.long),
            'is_unknown': torch.tensor(item["is_unknown"], dtype=torch.float),
        }

        # Add optional metadata if available
        for key in ['_severity_score', '_consensus_score', '_annotator_entropy']:
            if key in item:
                public_key = key.lstrip('_')
                result[public_key] = torch.tensor(item[key], dtype=torch.float)

        return result


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def get_hatexplain_dataloaders(
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    save_dir: str = "./data/hatexplain",
    num_workers: int = 4,
    subset_fraction: float = 1.0,
):
    """
    Get PyTorch DataLoaders for HateXplain.

    Returns train_loader, val_loader, test_loader, metadata
    """
    import torch
    from torch.utils.data import DataLoader

    # Load data
    processed_data = load_hatexplain_direct(save_dir)
    train_data = processed_data["train"]
    if subset_fraction < 1.0:
        import random
        subset_size = max(1, int(len(train_data) * subset_fraction))
        random.shuffle(train_data)
        train_data = train_data[:subset_size]
        print(f"  Using subset of {subset_size} samples from training set")

    # Create datasets
    train_dataset = DirectHateXplainDataset(
        train_data, tokenizer, max_length
    )
    val_dataset = DirectHateXplainDataset(
        processed_data["validation"], tokenizer, max_length
    )
    test_dataset = DirectHateXplainDataset(
        processed_data["test"], tokenizer, max_length
    )

    # Create dataloaders
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Metadata
    metadata = {
        "dataset_name": "hatexplain",
        "task": "toxicity",
        "num_classes": 3,
        "num_concepts": 2,
        "concept_names": ["has_target", "is_offensive"],
        "class_names": ["hatespeech", "normal", "offensive"],
        "has_concepts": True,
        "has_multi_annotator": True,
        "is_ordinal": False,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, tokenizer, metadata


if __name__ == "__main__":
    # Test loading
    data = load_hatexplain_direct()

    print(f"\n✓ Successfully loaded HateXplain!")
    print(f"  Train: {len(data['train'])}")
    print(f"  Val: {len(data['validation'])}")
    print(f"  Test: {len(data['test'])}")

    # Show sample
    if len(data['train']) > 0:
        sample = data['train'][0]
        print(f"\nSample:")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Label: {sample['label']} ({['hatespeech', 'normal', 'offensive'][sample['label']]})")
        print(f"  Concepts: {sample['concepts']}")
        print(f"  Unknown mask: {sample['is_unknown']}")
