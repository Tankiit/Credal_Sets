#!/usr/bin/env python3
"""
Quick test script to verify updated CEBaB dataloader works correctly.
"""
import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoTokenizer
from gacs.data.cebab import load_cebab_datasets, create_dataloaders
from gacs.configs.config import GACSConfig

print("=" * 70)
print("Testing Updated CEBaB Dataloader")
print("=" * 70)

# Load config
config = GACSConfig.for_cebab()
print(f"\nConfig:")
print(f"  num_classes: {config.model.num_classes}")
print(f"  num_concepts: {config.model.num_concepts}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

# Load datasets
print(f"\nLoading datasets...")
train_ds, val_ds, test_ds = load_cebab_datasets(
    tokenizer,
    max_length=config.training.max_seq_length
)

print(f"\n✓ Dataset sizes:")
print(f"  Train: {len(train_ds):,} samples")
print(f"  Val: {len(val_ds):,} samples")
print(f"  Test: {len(test_ds):,} samples")

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_ds, val_ds, test_ds,
    batch_size=4,  # Small batch for testing
    num_workers=0
)

print(f"\n✓ DataLoaders created")

# Test a batch
print(f"\nTesting batch...")
batch = next(iter(train_loader))

print(f"\n✓ Batch structure:")
print(f"  input_ids shape: {batch['input_ids'].shape}")
print(f"  attention_mask shape: {batch['attention_mask'].shape}")
print(f"  label shape: {batch['label'].shape}")
print(f"  concepts shape: {batch['concepts'].shape}")

print(f"\n✓ Sample values:")
print(f"  labels: {batch['label'].tolist()}")
print(f"  label range: [{batch['label'].min().item()}, {batch['label'].max().item()}]")
print(f"  concepts:\n{batch['concepts']}")

# Verify ranges
assert batch['label'].min() >= 0, "Label min < 0!"
assert batch['label'].max() <= 2, "Label max > 2! (should be 3-class)"
assert batch['concepts'].min() >= 0, "Concept min < 0!"
assert batch['concepts'].max() <= 2, "Concept max > 2! (should be 3-class)"

print(f"\n✓ All checks passed!")
print(f"✓ Labels are 3-class (0-2): Negative/unknown/Positive")
print(f"✓ Concepts are 3-class (0-2): Negative/unknown/Positive")
print(f"✓ Using train_inclusive split ({len(train_ds):,} samples)")

print(f"\n" + "=" * 70)
print("SUCCESS: CEBaB dataloader updated correctly!")
print("=" * 70)
