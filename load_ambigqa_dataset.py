"""
AmbigQA / MAQA dataset helpers.

This module covers two related use cases:
- Raw Hugging Face dataset inspection / DataLoader creation for AmbigQA Star.
- MAQA-style preprocessing that converts ambiguous QA records into the
  processed `text`, `p_star`, `entropy`, and `ambiguity_level` records used by
  the credal QA trainer.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login, whoami
from torch.utils.data import DataLoader, Dataset


def login_to_huggingface(token: Optional[str] = None) -> bool:
    """Login to Hugging Face Hub, preferring existing auth state."""
    try:
        user_info = whoami()
        print(f"Already logged in as: {user_info.get('name', 'Unknown')}")
        return True
    except Exception:
        pass

    if token:
        try:
            login(token=token)
            print("Successfully logged in with provided token")
            return True
        except Exception as e:
            print(f"Failed to login with token: {e}")
            return False

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Successfully logged in with environment token")
            return True
        except Exception as e:
            print(f"Failed to login with environment token: {e}")

    # Do not attempt interactive login in headless environments such as Modal.
    # Public datasets can still be loaded anonymously; private datasets should
    # provide a token through HF_TOKEN/HUGGINGFACE_TOKEN or an explicit arg.
    if sys.stdin is not None and sys.stdin.isatty():
        print("No Hugging Face token found; skipping auth and using anonymous access.")
    return False


class AmbigQADataset(Dataset):
    """
    PyTorch wrapper for the AmbigQA Star dataset.

    This keeps the raw record structure intact and adds a stable `id` field
    plus any requested text fields.
    """

    def __init__(
        self,
        dataset_split,
        id_field: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
    ):
        self.data = dataset_split
        self.id_field = id_field
        if text_fields is None:
            self.text_fields = [
                col for col in self.data.column_names
                if col not in ["id", "idx", "question_id"]
            ]
        else:
            self.text_fields = text_fields

        if self.id_field is None:
            for possible_id in ["id", "idx", "question_id", "example_id"]:
                if possible_id in self.data.column_names:
                    self.id_field = possible_id
                    break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.id_field and self.id_field in sample:
            sample_id = sample[self.id_field]
        else:
            sample_id = idx

        result = {"id": sample_id}
        for field in self.text_fields:
            if field in sample:
                result[field] = sample[field]

        for key, value in sample.items():
            if key not in result and key != self.id_field:
                result[key] = value
        return result


def _safe_text(sample: dict) -> str:
    for key in ("question", "text", "query", "prompt", "input"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if "question" in sample and isinstance(sample["question"], dict):
        q = sample["question"]
        for key in ("text", "stem", "title"):
            value = q.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _safe_probabilities(sample: dict) -> list[float]:
    candidates = (
        sample.get("probabilities"),
        sample.get("p_star"),
        sample.get("answer_distribution"),
        sample.get("answers_probs"),
    )
    for value in candidates:
        if isinstance(value, (list, tuple)) and value:
            probs = [float(v) for v in value]
            total = sum(probs)
            if total > 0:
                return [v / total for v in probs]

    answers = sample.get("answers")
    if isinstance(answers, (list, tuple)) and answers:
        p = [0.0] * len(answers)
        p[0] = 1.0
        return p

    if isinstance(sample.get("answer_label"), int) and sample["answer_label"] >= 0:
        label = int(sample["answer_label"])
        p = [0.0] * (label + 1)
        p[label] = 1.0
        return p

    return [1.0]


def _processed_item(sample: dict) -> dict:
    text = _safe_text(sample)
    probs = _safe_probabilities(sample)
    p = np.asarray(probs, dtype=np.float32)
    p = p / max(float(p.sum()), 1e-12)
    entropy = float(-(p * np.log(p + 1e-10)).sum())
    answers = sample.get("answers")
    if not isinstance(answers, (list, tuple)) or not answers:
        answers = [sample.get("answer", sample.get("gold_answer", ""))]
    dominant = int(np.argmax(p)) if len(p) else 0
    return {
        "text": text,
        "p_star": p.tolist(),
        "entropy": entropy,
        "ambiguity_level": 0 if entropy < 0.1 else 2 if entropy > 0.5 else 1,
        "answers": list(answers),
        "num_answers": int(len(p)),
        "dominant_answer_idx": dominant,
    }


def load_combined_maqa_ambigqa(
    dataset_name: str = "ttomov/ambigqa_star",
) -> Dict[str, List[dict]]:
    """
    Load raw QA splits and convert them into MAQA-style processed records.
    """
    ds = load_dataset(dataset_name)
    splits: Dict[str, List[dict]] = {}
    for split_name, split_data in ds.items():
        processed = []
        for sample in split_data:
            item = _processed_item(sample)
            if not item["text"]:
                continue
            processed.append(item)
        if processed:
            splits[split_name] = processed

    if not splits:
        raise ValueError(f"Dataset '{dataset_name}' produced no usable examples.")
    return splits


def create_dataloaders(
    dataset_name: str = "ttomov/ambigqa_star",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    id_field: Optional[str] = None,
    splits: Optional[List[str]] = None,
    **dataloader_kwargs,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for the raw AmbigQA Star dataset.
    """
    ds = load_dataset(dataset_name)
    if splits is None:
        splits = list(ds.keys())

    dataloaders: Dict[str, DataLoader] = {}
    for split_name in splits:
        if split_name not in ds:
            continue
        dataset = AmbigQADataset(ds[split_name], id_field=id_field)
        should_shuffle = shuffle and (split_name == "train" or "train" in split_name.lower())
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            **dataloader_kwargs,
        )
    return dataloaders


def inspect_dataset_sample(dataloader: DataLoader, num_samples: int = 1) -> None:
    """Print a compact view of a few batches."""
    print("\n" + "=" * 70)
    print("Dataset Sample Inspection")
    print("=" * 70)

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"\nBatch {i + 1}:")
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: list[{len(value)}]")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print(f"  batch type: {type(batch).__name__}")
