"""
HateXplain dataset loader.

Wraps `get_hatexplain_dataloaders` from load_hatexplain_direct.py.

HateXplain provides:
- 2 concepts: has_target, is_offensive
- 3 classes: hate / offensive / normal
- Multi-annotator labels → annotator_entropy available
"""
from __future__ import annotations

from transformers import AutoTokenizer

from loaders.interface import DatasetBundle, DatasetLoader
from load_hatexplain_direct import get_hatexplain_dataloaders  # noqa


class _HateXplainLoader:
    def load(
        self,
        tokenizer_name: str,
        batch_size: int,
        max_length: int = 128,
        seed: int = 42,
        subset_fraction: float = 1.0,
        num_workers: int = 0,
    ) -> DatasetBundle:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        train_loader, val_loader, test_loader, tokenizer, metadata = \
            get_hatexplain_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                num_workers=num_workers,
            )

        if subset_fraction < 1.0:
            raise NotImplementedError("subset_fraction < 1.0 not yet wired")

        return DatasetBundle(
            name="hatexplain",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            num_concepts=metadata["num_concepts"],       # 2
            concept_names=tuple(metadata["concept_names"]),
            num_classes=metadata["num_classes"],         # 3
            has_annotator_entropy=True,
            train_size=metadata["train_size"],
            val_size=metadata["val_size"],
            test_size=metadata["test_size"],
        )


LOADER: DatasetLoader = _HateXplainLoader()
