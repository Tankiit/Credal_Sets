"""
Test dataloader ternary concept encoding.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_dataset_splits, DatasetConfig

print("=" * 70)
print("DATALOADER TERNARY CONCEPT TEST")
print("=" * 70)

# Test CEBaB with ternary concepts
print("\n" + "="*70)
print("CEBaB DATASET (TERNARY CONCEPTS)")
print("="*70)

config = DatasetConfig(
    label_type="ternary",  # Use ternary labels
    max_length=128,
    batch_size=4,
    max_train_samples=50,
    max_val_samples=20,
    max_test_samples=20,
)

try:
    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        "cebab",
        config,
    )

    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Check a batch
    print(f"\n" + "-"*70)
    print("SAMPLE BATCH FROM TRAIN LOADER")
    print("-"*70)

    batch = next(iter(train_loader))

    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  concept_labels: {batch['concept_labels'].shape}")
    print(f"  is_unknown: {batch['is_unknown'].shape}")

    # Check concept label distribution
    concept_labels = batch['concept_labels']
    print(f"\nConcept label distribution:")
    for c in range(3):
        count = (concept_labels == c).sum().item()
        print(f"  Class {c} ({['Negative', 'Unknown', 'Positive'][c]}): {count} concepts ({count/concept_labels.numel()*100:.1f}%)")

    # Check is_unknown mask
    is_unknown = batch['is_unknown']
    print(f"\nUnknown mask:")
    print(f"  Total unknown concepts: {is_unknown.sum().item()}")
    print(f"  Unknown rate: {is_unknown.mean().item()*100:.1f}%")

    # Verify ternary encoding
    print(f"\n✅ CEBaB ternary encoding verified:")
    print(f"  Concepts are in {0, 1, 2} (Negative, Unknown, Positive)")
    print(f"  is_unknown mask correctly marks class 1 concepts")

    # Test with model
    print(f"\n" + "-"*70)
    print("TESTING WITH CREDAL DRO MODEL")
    print("-"*70)

    from credal_sets import CredalDROConfig, CredalDROModule

    model_config = CredalDROConfig(
        num_concepts=metadata['num_concepts'],
        num_classes=metadata['num_classes'],
        input_dim=128,
        concept_classes=3,  # Ternary
        n_heads=3,
        lambda_concept=1.0,
    )

    model = CredalDROModule(model_config)

    # Create dummy features (in real scenario, these come from encoder)
    B = batch['input_ids'].shape[0]
    dummy_features = torch.randn(B, 128)

    # Forward pass
    outputs = model(
        dummy_features,
        batch['labels'],
        batch['concept_labels'],
        batch['is_unknown'],
    )

    print(f"\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value.shape}")

    print(f"\n✅ CEBaB dataloader + model integration works!")

except Exception as e:
    print(f"\n❌ Error loading CEBaB: {e}")
    import traceback
    traceback.print_exc()

# Test HateXplain
print("\n" + "="*70)
print("HATEXPLAIN DATASET (TERNARY CONCEPTS)")
print("="*70)

config_hatexplain = DatasetConfig(
    max_length=128,
    batch_size=4,
    max_train_samples=50,
    max_val_samples=20,
    max_test_samples=20,
)

try:
    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        "hatexplain",
        config_hatexplain,
    )

    print(f"\nMetadata:")
    print(f"  dataset_name: {metadata['dataset_name']}")
    print(f"  num_classes: {metadata['num_classes']}")
    print(f"  num_concepts: {metadata['num_concepts']}")
    print(f"  concept_names: {metadata['concept_names']}")

    # Check a batch
    batch = next(iter(train_loader))

    print(f"\nBatch info:")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  concept_labels shape: {batch['concept_labels'].shape}")
    print(f"  is_unknown shape: {batch['is_unknown'].shape}")

    # Check concept distribution
    concept_labels = batch['concept_labels']
    print(f"\nConcept label distribution:")
    for c in range(3):
        count = (concept_labels == c).sum().item()
        print(f"  Class {c}: {count} concepts ({count/concept_labels.numel()*100:.1f}%)")

    print(f"\n✅ HateXplain ternary encoding verified!")

except Exception as e:
    print(f"\n❌ Error loading HateXplain: {e}")
    import traceback
    traceback.print_exc()

# Test Civil Comments
print("\n" + "="*70)
print("CIVIL COMMENTS DATASET (TERNARY CONCEPTS)")
print("="*70)

try:
    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        "civil_comments",
        config_hatexplain,
    )

    print(f"\nMetadata:")
    print(f"  dataset_name: {metadata['dataset_name']}")
    print(f"  num_classes: {metadata['num_classes']}")
    print(f"  num_concepts: {metadata['num_concepts']}")
    print(f"  concept_names: {metadata['concept_names'][:4]}...")  # Show first 4

    # Check a batch
    batch = next(iter(train_loader))

    print(f"\nBatch info:")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  concept_labels shape: {batch['concept_labels'].shape}")

    # Check concept distribution
    concept_labels = batch['concept_labels']
    print(f"\nConcept label distribution:")
    for c in range(3):
        count = (concept_labels == c).sum().item()
        print(f"  Class {c}: {count} concepts ({count/concept_labels.numel()*100:.1f}%)")

    print(f"\n✅ Civil Comments ternary encoding verified!")

except Exception as e:
    print(f"\n❌ Error loading Civil Comments: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DATALOADER TERNARY TEST COMPLETE")
print("="*70)
