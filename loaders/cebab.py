"""
CEBaB dataset loader.

Wraps the existing `get_cebab_dataloaders` (copied from load_cebab_direct.py)
in the DatasetBundle interface. Underlying loading logic unchanged — just
the return shape standardised.

CEBaB provides:
- 4 concepts: food, service, ambiance, noise
- 5 sentiment classes (1-5 stars)
- Per-concept annotator distributions → annotator_entropy available
"""
from __future__ import annotations

from typing import Optional
from transformers import AutoTokenizer

from loaders.interface import DatasetBundle, DatasetLoader

# TODO(PR1): paste the contents of load_cebab_direct.py below this line,
# OR import from it while .migrate-src/ still exists:
#     from .migrate-src.load_cebab_direct import get_cebab_dataloaders
# The long-term move is to paste the implementation into this file and
# delete .migrate-src/load_cebab_direct.py once PR 1 is green.
from load_cebab_direct import get_cebab_dataloaders  # noqa


class _CEBaBLoader:
    """Concrete loader. Exposed as module-level `LOADER` below."""

    def load(
        self,
        tokenizer_name: str,
        batch_size: int,
        max_length: int = 256,
        seed: int = 42,
        subset_fraction: float = 1.0,
        num_workers: int = 0,
    ) -> DatasetBundle:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        train_loader, val_loader, test_loader, tokenizer, metadata = \
            get_cebab_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                num_workers=num_workers,
                subset_fraction=subset_fraction,
            )

        if subset_fraction < 1.0:
            metadata["train_size"] = len(train_loader.dataset)

        return DatasetBundle(
            name="cebab",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            num_concepts=metadata["num_concepts"],       # 4
            concept_names=tuple(metadata["concept_names"]),
            num_classes=metadata["num_classes"],         # 5
            has_annotator_entropy=True,                  # CEBaB has explicit per-annotator concept labels
            train_size=metadata["train_size"],
            val_size=metadata["val_size"],
            test_size=metadata["test_size"],
        )


LOADER: DatasetLoader = _CEBaBLoader()
