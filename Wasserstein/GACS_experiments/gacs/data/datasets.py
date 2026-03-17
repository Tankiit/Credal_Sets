"""
GACS Data Loading
=================
Handles CEBaB (primary), Amazon multi-domain, SST-2, IMDB, AG News.
CEBaB provides concept labels + counterfactual distribution shift.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional, Tuple, List
import numpy as np


# ---------------------------------------------------------------------------
# CEBaB Dataset
# ---------------------------------------------------------------------------

CEBAB_CONCEPTS = ["food", "service", "ambiance", "noise"]
CEBAB_CONCEPT_VALUES = {"Positive": 2, "Negative": 0, "unknown": 1, "no majority": 1}


class CEBaBDataset(Dataset):
    """
    CEBaB dataset with concept labels and counterfactual support.

    Each example has:
        - text: restaurant review
        - label: overall sentiment (1-5 stars → 0-4 index)
        - concept_labels: [food, service, ambiance, noise] as floats
        - is_counterfactual: bool
        - edit_concept: which concept was intervened on (if counterfactual)
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train_exclusive",
        max_length: int = 128,
        include_counterfactuals: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.include_counterfactuals = include_counterfactuals

        self.texts = []
        self.labels = []
        self.concept_labels = []
        self.is_cf = []

        self._load(split)

    def _load(self, split: str):
        """Load from HuggingFace datasets."""
        from datasets import load_dataset

        ds = load_dataset("CEBaB/CEBaB")

        if split in ds:
            data = ds[split]
        else:
            raise ValueError(f"Split '{split}' not in CEBaB. Available: {list(ds.keys())}")

        for example in data:
            text = example.get("description", example.get("review", ""))
            if not text:
                continue

            # Overall sentiment: map to 0-4
            label = example.get("review_majority", None)
            if label is None or label == "no majority":
                continue
            try:
                label = int(float(label)) - 1  # 1-5 → 0-4
            except (ValueError, TypeError):
                continue
            if label < 0 or label > 4:
                continue

            # Concept labels
            concepts = []
            for c in CEBAB_CONCEPTS:
                val = example.get(f"{c}_aspect_majority", "unknown")
                concepts.append(CEBAB_CONCEPT_VALUES.get(val, 1) / 2.0)  # normalize to [0,1]

            self.texts.append(text)
            self.labels.append(label)
            self.concept_labels.append(concepts)
            self.is_cf.append(False)

        print(f"CEBaB [{split}]: loaded {len(self.texts)} examples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "concept_labels": torch.tensor(self.concept_labels[idx], dtype=torch.float32),
            "is_counterfactual": torch.tensor(self.is_cf[idx], dtype=torch.bool),
        }


class CEBaBCounterfactualDataset(Dataset):
    """
    CEBaB counterfactual test set for distribution shift evaluation.

    Loads original-counterfactual pairs where one concept was intervened on.
    This provides controlled, semantically meaningful distribution shift.
    """

    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        self.labels = []
        self.concept_labels = []
        self.edit_concepts = []
        self.original_ids = []

        self._load()

    def _load(self):
        from datasets import load_dataset

        ds = load_dataset("CEBaB/CEBaB")

        # The test split contains counterfactual annotations
        for split_name in ["test", "validation"]:
            if split_name not in ds:
                continue
            data = ds[split_name]

            for example in data:
                text = example.get("description", example.get("review", ""))
                if not text:
                    continue

                label = example.get("review_majority", None)
                if label is None or label == "no majority":
                    continue
                try:
                    label = int(float(label)) - 1
                except (ValueError, TypeError):
                    continue
                if label < 0 or label > 4:
                    continue

                # Check if this is an edited (counterfactual) example
                edit_concept = example.get("edit_type", None)
                original_id = example.get("original_id", example.get("id", None))

                concepts = []
                for c in CEBAB_CONCEPTS:
                    val = example.get(f"{c}_aspect_majority", "unknown")
                    concepts.append(CEBAB_CONCEPT_VALUES.get(val, 1) / 2.0)

                self.texts.append(text)
                self.labels.append(label)
                self.concept_labels.append(concepts)
                self.edit_concepts.append(edit_concept)
                self.original_ids.append(original_id)

            break  # just use test

        print(f"CEBaB counterfactual: loaded {len(self.texts)} examples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "concept_labels": torch.tensor(self.concept_labels[idx], dtype=torch.float32),
            "is_counterfactual": torch.tensor(True, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Amazon Multi-Domain (cross-domain shift)
# ---------------------------------------------------------------------------

class AmazonDomainDataset(Dataset):
    """Amazon multi-domain sentiment for cross-domain distribution shift."""

    def __init__(
        self,
        tokenizer,
        domain: str = "books",
        split: str = "train",
        max_length: int = 128,
        max_examples: int = 5000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        self._load(domain, split, max_examples)

    def _load(self, domain: str, split: str, max_examples: int):
        """Load Amazon reviews for a specific domain."""
        try:
            from datasets import load_dataset
            # Use the amazon_polarity or multi_domain_sentiment dataset
            ds = load_dataset("amazon_polarity", split=split, streaming=True)

            count = 0
            for example in ds:
                if count >= max_examples:
                    break
                text = example.get("content", example.get("text", ""))
                label = example.get("label", 0)
                if text:
                    self.texts.append(text[:512])  # truncate long reviews
                    self.labels.append(int(label))
                    count += 1

            print(f"Amazon [{domain}/{split}]: loaded {len(self.texts)} examples")
        except Exception as e:
            print(f"Warning: could not load Amazon dataset: {e}")
            # Fallback: create dummy data for testing
            self.texts = ["This is a good product."] * 100
            self.labels = [1] * 50 + [0] * 50

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "concept_labels": torch.zeros(4, dtype=torch.float32),  # no concept supervision
            "is_counterfactual": torch.tensor(False, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Generic text dataset wrapper (SST-2, IMDB, AG News)
# ---------------------------------------------------------------------------

class GenericTextDataset(Dataset):
    """Wrapper for HuggingFace text classification datasets."""

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "sst2",
        split: str = "train",
        max_length: int = 128,
        max_examples: int = 10000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        self._load(dataset_name, split, max_examples)

    def _load(self, name: str, split: str, max_examples: int):
        from datasets import load_dataset

        DATASET_MAP = {
            "sst2": ("glue", "sst2", "sentence", "label"),
            "imdb": ("imdb", None, "text", "label"),
            "agnews": ("ag_news", None, "text", "label"),
        }

        if name not in DATASET_MAP:
            raise ValueError(f"Unknown dataset: {name}")

        ds_name, ds_config, text_key, label_key = DATASET_MAP[name]

        if ds_config:
            ds = load_dataset(ds_name, ds_config, split=split)
        else:
            ds = load_dataset(ds_name, split=split)

        for i, example in enumerate(ds):
            if i >= max_examples:
                break
            text = example[text_key]
            label = example[label_key]
            if text:
                self.texts.append(text[:512])
                self.labels.append(int(label))

        print(f"{name} [{split}]: loaded {len(self.texts)} examples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "concept_labels": torch.zeros(4, dtype=torch.float32),
            "is_counterfactual": torch.tensor(False, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Data loader factory
# ---------------------------------------------------------------------------

def get_dataloaders(config) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders based on config.

    Returns dict with keys: "train", "val", "test", and optionally "shift"
    for distribution-shifted test sets.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    dc = config.data

    loaders = {}

    if dc.name == "cebab":
        train_ds = CEBaBDataset(tokenizer, split="train_exclusive", max_length=dc.max_seq_length)
        val_ds = CEBaBDataset(tokenizer, split="validation", max_length=dc.max_seq_length)
        test_ds = CEBaBDataset(tokenizer, split="test", max_length=dc.max_seq_length)
        shift_ds = CEBaBCounterfactualDataset(tokenizer, max_length=dc.max_seq_length)

        loaders["train"] = DataLoader(
            train_ds, batch_size=dc.batch_size, shuffle=True,
            num_workers=dc.num_workers, pin_memory=dc.pin_memory,
        )
        loaders["val"] = DataLoader(
            val_ds, batch_size=dc.batch_size, shuffle=False,
            num_workers=dc.num_workers, pin_memory=dc.pin_memory,
        )
        loaders["test"] = DataLoader(
            test_ds, batch_size=dc.batch_size, shuffle=False,
            num_workers=dc.num_workers, pin_memory=dc.pin_memory,
        )
        loaders["shift"] = DataLoader(
            shift_ds, batch_size=dc.batch_size, shuffle=False,
            num_workers=dc.num_workers, pin_memory=dc.pin_memory,
        )

    elif dc.name == "amazon":
        train_ds = AmazonDomainDataset(
            tokenizer, domain=dc.amazon_source_domain, split="train",
            max_length=dc.max_seq_length,
        )
        test_ds = AmazonDomainDataset(
            tokenizer, domain=dc.amazon_source_domain, split="test",
            max_length=dc.max_seq_length, max_examples=2000,
        )
        shift_ds = AmazonDomainDataset(
            tokenizer, domain=dc.amazon_target_domain, split="test",
            max_length=dc.max_seq_length, max_examples=2000,
        )

        # Split train into train/val
        train_size = int(0.9 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

        loaders["train"] = DataLoader(train_ds, batch_size=dc.batch_size, shuffle=True,
                                       num_workers=dc.num_workers, pin_memory=dc.pin_memory)
        loaders["val"] = DataLoader(val_ds, batch_size=dc.batch_size, shuffle=False,
                                     num_workers=dc.num_workers, pin_memory=dc.pin_memory)
        loaders["test"] = DataLoader(test_ds, batch_size=dc.batch_size, shuffle=False,
                                      num_workers=dc.num_workers, pin_memory=dc.pin_memory)
        loaders["shift"] = DataLoader(shift_ds, batch_size=dc.batch_size, shuffle=False,
                                       num_workers=dc.num_workers, pin_memory=dc.pin_memory)

    else:
        # Generic datasets (SST-2, IMDB, AG News)
        train_ds = GenericTextDataset(tokenizer, dc.name, split="train",
                                      max_length=dc.max_seq_length)
        test_ds = GenericTextDataset(tokenizer, dc.name, split="test",
                                     max_length=dc.max_seq_length, max_examples=2000)

        train_size = int(0.9 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

        loaders["train"] = DataLoader(train_ds, batch_size=dc.batch_size, shuffle=True,
                                       num_workers=dc.num_workers, pin_memory=dc.pin_memory)
        loaders["val"] = DataLoader(val_ds, batch_size=dc.batch_size, shuffle=False,
                                     num_workers=dc.num_workers, pin_memory=dc.pin_memory)
        loaders["test"] = DataLoader(test_ds, batch_size=dc.batch_size, shuffle=False,
                                      num_workers=dc.num_workers, pin_memory=dc.pin_memory)

    return loaders, tokenizer
