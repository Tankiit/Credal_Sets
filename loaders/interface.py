"""
Dataset loader interface.

Every dataset module exports a `LOADER: DatasetLoader` instance. The training
script dispatches via `LOADERS[dataset_name].load(...)` and gets back a
uniform `DatasetBundle`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DatasetBundle:
    """
    Everything the trainer and evaluator need from a loaded dataset.

    Frozen so callers cannot mutate mid-run. A bundle is produced once per
    (dataset, seed, subset_fraction) tuple and passed by reference.
    """

    # Identity
    name: str                               # 'cebab', 'hatexplain', ...

    # Data
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    tokenizer: object                       # HF tokenizer; untyped to avoid transformers import here

    # Schema
    num_concepts: int
    concept_names: tuple[str, ...]
    num_classes: int

    # Metadata for the grad-iso probe and metrics
    #   True  → annotator_entropy present on every batch → AU loss always fires
    #   False → SST-2-like (LLM-generated or absent) → probe may skip some batches
    has_annotator_entropy: bool

    # Sizes — useful for scheduler warmup and logging
    train_size: int
    val_size: int
    test_size: int


@runtime_checkable
class DatasetLoader(Protocol):
    """All four dataset modules implement this."""

    def load(
        self,
        tokenizer_name: str,
        batch_size: int,
        max_length: int,
        seed: int = 42,
        subset_fraction: float = 1.0,
        num_workers: int = 0,
    ) -> DatasetBundle: ...


# --- Contract notes for implementers -------------------------------------
# Every batch yielded by any of the three loaders MUST be a dict with keys:
#
#   input_ids:          LongTensor  [B, L]
#   attention_mask:     LongTensor  [B, L]
#   labels:             LongTensor  [B]
#   concept_labels:     LongTensor  [B, K]          (optional; present iff dataset
#                                                    has concept annotations)
#   annotator_entropy:  FloatTensor [B, K]          (optional; present iff
#                                                    has_annotator_entropy=True)
#
# The trainer uses `.get(key)` for the optional two and passes them to the
# model's forward unchanged. Do not rename these keys.