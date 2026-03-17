"""
CEBaB Dataset Loading and Processing
Handles original reviews + counterfactual reviews for distribution shift experiments.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np


# CEBaB concept label mapping: aspect-level sentiment → numeric
CONCEPT_LABEL_MAP = {
    "Negative": 0,
    "unknown": 1,  # treat unknown as neutral
    "Positive": 2,
}

# Overall rating: Use 3-class sentiment (Negative/unknown/Positive) to match original
SENTIMENT_MAP = {
    "Negative": 0,
    "unknown": 1,  # neutral/unknown
    "Positive": 2,
}

# For 5-star ratings, map to 3 classes
STAR_TO_SENTIMENT = {
    1: 0,  # 1 star → Negative
    2: 0,  # 2 stars → Negative
    3: 1,  # 3 stars → unknown/neutral
    4: 2,  # 4 stars → Positive
    5: 2,  # 5 stars → Positive
}

CONCEPT_NAMES = ["food", "service", "ambiance", "noise"]


class CEBaBDataset(Dataset):
    """
    Wraps CEBaB HuggingFace dataset into a PyTorch Dataset.
    
    Each item returns:
        - input_ids, attention_mask (tokenized review)
        - label (0-2, overall sentiment: Negative/unknown/Positive)
        - concepts (tensor of 4 concept labels: food, service, ambiance, noise)
        - has_concepts (bool, whether concept labels are available)
    """
    
    def __init__(
        self,
        hf_dataset,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        include_counterfactuals: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for item in hf_dataset:
            sample = self._process_item(item)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Process a single CEBaB example."""
        text = item.get("description", item.get("review", ""))
        if not text or len(text.strip()) == 0:
            return None

        # Overall rating - try to get sentiment label directly
        rating = item.get("review_majority", item.get("label", None))
        if rating is None:
            return None

        # Handle different rating formats
        if isinstance(rating, str):
            # Check if it's already a sentiment label
            if rating in SENTIMENT_MAP:
                label = SENTIMENT_MAP[rating]
            else:
                # Try to parse as star rating
                try:
                    rating_int = int(rating.split()[0])
                    if rating_int not in STAR_TO_SENTIMENT:
                        return None
                    label = STAR_TO_SENTIMENT[rating_int]
                except (ValueError, IndexError):
                    return None
        elif isinstance(rating, int):
            # Direct star rating
            if rating not in STAR_TO_SENTIMENT:
                return None
            label = STAR_TO_SENTIMENT[rating]
        else:
            return None
        
        # Concept labels
        concepts = []
        has_concepts = True
        for concept_name in CONCEPT_NAMES:
            key = f"{concept_name}_aspect_majority"
            val = item.get(key, "unknown")
            if val is None or val == "":
                val = "unknown"
            concepts.append(CONCEPT_LABEL_MAP.get(val, 1))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "concepts": torch.tensor(concepts, dtype=torch.long),
            "text": text,
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "concepts": sample["concepts"],
        }


def load_cebab_datasets(
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> Tuple[CEBaBDataset, CEBaBDataset, CEBaBDataset]:
    """
    Load CEBaB train/dev/test splits.

    Uses train_inclusive split to match original Wasserstein implementation.
    This provides more training data and better alignment with the paper.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from datasets import load_dataset

    print("Loading CEBaB dataset from HuggingFace...")
    cebab = load_dataset("CEBaB/CEBaB")

    train_ds = CEBaBDataset(
        cebab["train_inclusive"],  # Changed from train_exclusive to train_inclusive
        tokenizer,
        max_length=max_length,
    )
    val_ds = CEBaBDataset(
        cebab["validation"],
        tokenizer,
        max_length=max_length,
    )
    test_ds = CEBaBDataset(
        cebab["test"],
        tokenizer,
        max_length=max_length,
    )

    return train_ds, val_ds, test_ds


def load_cebab_counterfactuals(
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    intervention_concept: Optional[str] = None,
) -> CEBaBDataset:
    """
    Load CEBaB counterfactual reviews for distribution shift evaluation.
    
    These are human-written reviews where one concept (food/service/ambiance/noise)
    has been intervened on — e.g., changing "great food" to "terrible food".
    
    Args:
        intervention_concept: If specified, only load CFs for this concept.
            One of: "food", "service", "ambiance", "noise", or None for all.
    """
    from datasets import load_dataset
    
    print("Loading CEBaB counterfactuals...")
    cebab = load_dataset("CEBaB/CEBaB")
    
    # Counterfactuals are in the test split with edit_ columns
    # We need to filter for rows that have counterfactual edits
    cf_samples = []
    
    for item in cebab["test"]:
        # Check each concept for interventions
        for concept in CONCEPT_NAMES:
            if intervention_concept and concept != intervention_concept:
                continue
            
            # CEBaB stores counterfactual text in edit columns
            edit_key = f"edit_{concept}"
            if edit_key in item and item[edit_key] is not None and len(str(item[edit_key]).strip()) > 0:
                cf_text = item[edit_key]
                if cf_text and len(cf_text.strip()) > 0:
                    cf_samples.append({
                        "text": cf_text,
                        "original_text": item.get("description", ""),
                        "intervened_concept": concept,
                        "label": item.get("review_majority", 3),
                    })
    
    print(f"Found {len(cf_samples)} counterfactual samples")
    return cf_samples


def create_dataloaders(
    train_ds: CEBaBDataset,
    val_ds: CEBaBDataset,
    test_ds: CEBaBDataset,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders with appropriate settings."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
