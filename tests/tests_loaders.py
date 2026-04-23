"""
Smoke test for PR 1: every loader returns a DatasetBundle with the expected
shape, and every batch has the expected keys and dtypes.

Run:  python -m tests.test_loaders
"""
from __future__ import annotations

import torch

from loaders import LOADERS
from loaders.interface import DatasetBundle


REQUIRED_BATCH_KEYS = {"input_ids", "attention_mask", "labels"}
OPTIONAL_BATCH_KEYS = {"concept_labels", "annotator_entropy"}


def _check_bundle(bundle: DatasetBundle) -> None:
    assert isinstance(bundle, DatasetBundle), f"not a DatasetBundle: {type(bundle)}"
    assert bundle.num_concepts > 0
    assert len(bundle.concept_names) == bundle.num_concepts
    assert bundle.num_classes > 0
    for split_name, loader in (
        ("train", bundle.train_loader),
        ("val",   bundle.val_loader),
        ("test",  bundle.test_loader),
    ):
        batch = next(iter(loader))
        missing = REQUIRED_BATCH_KEYS - batch.keys()
        assert not missing, f"{bundle.name}/{split_name}: missing keys {missing}"
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        if bundle.has_annotator_entropy:
            assert "annotator_entropy" in batch, \
                f"{bundle.name} claims annotator entropy but batch is missing it"
            assert batch["annotator_entropy"].dtype == torch.float32
        print(f"  ✓ {bundle.name:12s} / {split_name:5s}  "
              f"batch size {batch['input_ids'].shape[0]:3d}")


def main() -> None:
    # PR 1 scope: CEBaB and HateXplain must pass. GoEmotions is a stretch goal.
    # SST-2 is expected to raise NotImplementedError.
    for name in ["cebab", "hatexplain", "goemotions"]:
        print(f"\n--- {name} ---")
        try:
            bundle = LOADERS[name].load(
                tokenizer_name="distilbert-base-uncased",
                batch_size=4,
                max_length=128,
            )
            _check_bundle(bundle)
        except NotImplementedError as e:
            print(f"  ! skipped: {e}")

    print("\n--- sst2 (expected NotImplementedError) ---")
    try:
        LOADERS["sst2"].load(
            tokenizer_name="distilbert-base-uncased",
            batch_size=4,
            max_length=128,
        )
        raise AssertionError("SST-2 should raise NotImplementedError for now")
    except NotImplementedError:
        print("  ✓ correctly raised NotImplementedError")


if __name__ == "__main__":
    main()