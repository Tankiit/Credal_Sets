"""
GoEmotions dataset loader.

Wraps the multi-dataset `load_dataset_splits` from credence_dataloader.py,
which handles GoEmotions specifically (single-label classification over 28
emotion categories with per-rater annotations).

GoEmotions provides:
- 28 concepts = 28 emotions (each is its own binary "concept")
- 28 classes
- Multi-annotator → annotator_entropy available
"""
from __future__ import annotations

from loaders.interface import DatasetBundle, DatasetLoader

# TODO(PR1): confirm credence_dataloader.py is copied into .migrate_src/
from credence_dataloader import load_dataset_splits, DatasetConfig  # noqa


class _GoEmotionsLoader:
    def load(
        self,
        tokenizer_name: str,
        batch_size: int,
        max_length: int = 64,
        seed: int = 42,
        subset_fraction: float = 1.0,
        num_workers: int = 0,
    ) -> DatasetBundle:
        ds_config = DatasetConfig(
            max_length=max_length,
            batch_size=batch_size,
            tokenizer_name=tokenizer_name,
            num_workers=num_workers,
        )

        train_loader, val_loader, test_loader, tokenizer, metadata = \
            load_dataset_splits(dataset_name="goemotions", config=ds_config)

        if subset_fraction < 1.0:
            raise NotImplementedError("subset_fraction < 1.0 not yet wired")

        # GoEmotions concept names from CREDENCE paper table
        EMOTION_NAMES = (
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral",
        )
        assert len(EMOTION_NAMES) == 28

        return DatasetBundle(
            name="goemotions",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            num_concepts=28,
            concept_names=EMOTION_NAMES,
            num_classes=28,
            has_annotator_entropy=False,  # TODO: credence_dataloader doesn't emit annotator_entropy yet
            train_size=metadata.get("train_size", -1),
            val_size=metadata.get("val_size", -1),
            test_size=metadata.get("test_size", -1),
        )


LOADER: DatasetLoader = _GoEmotionsLoader()
