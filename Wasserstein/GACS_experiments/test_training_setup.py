#!/usr/bin/env python3
"""
Test Training Setup
===================

Quick verification that:
1. Cached latents can be loaded
2. Model can be instantiated
3. Forward pass works
4. Training loop can start

Usage:
    python test_training_setup.py --dataset snli
"""

import argparse
import torch
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_with_cached_latents import get_cached_dataloaders, LatentGACSModel
from gacs.config import GACSConfig


def test_dataset_loading(dataset_name):
    """Test that we can load cached latents."""
    print(f"\n{'='*60}")
    print(f"Testing dataset loading: {dataset_name}")
    print(f"{'='*60}")

    try:
        dataloaders, num_classes, num_concepts, latent_dim = get_cached_dataloaders(
            dataset_name, batch_size=4
        )

        print("✓ Dataset loaded successfully!")
        print(f"  Classes: {num_classes}")
        print(f"  Concepts: {num_concepts}")
        print(f"  Latent dim: {latent_dim}")

        # Test loading a batch
        for split_name, loader in dataloaders.items():
            batch = next(iter(loader))
            print(f"  {split_name} batch:")
            print(f"    latents: {batch['latents'].shape}")
            print(f"    labels: {batch['labels'].shape}")
            if "concepts" in batch:
                print(f"    concepts: {batch['concepts'].shape}")
            break  # Only test first split

        return True, dataloaders, num_classes, num_concepts, latent_dim

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None


def test_model_creation(dataset_name, num_classes, num_concepts, latent_dim):
    """Test model instantiation."""
    print(f"\n{'='*60}")
    print(f"Testing model creation")
    print(f"{'='*60}")

    try:
        config = GACSConfig()
        config.data.num_classes = num_classes
        config.model.concept_dim = num_concepts
        config.model.z_dim = 32  # Small for testing

        model = LatentGACSModel(config, latent_dim)

        print("✓ Model created successfully!")
        print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

        return True, model

    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, dataloaders):
    """Test forward pass."""
    print(f"\n{'='*60}")
    print(f"Testing forward pass")
    print(f"{'='*60}")

    try:
        model.eval()
        batch = next(iter(dataloaders["train"]))

        with torch.no_grad():
            outputs = model(batch["latents"])

        print("✓ Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, dataloaders):
    """Test a single training step."""
    print(f"\n{'='*60}")
    print(f"Testing training step")
    print(f"{'='*60}")

    try:
        model.train()
        batch = next(iter(dataloaders["train"]))

        # Forward pass
        outputs = model(batch["latents"])

        # Compute loss
        from gacs.losses.gacs_loss import GACSLoss
        config = GACSConfig()
        loss_fn = GACSLoss(config)

        # Create dummy batch dict for loss
        batch_dict = {
            "labels": batch["labels"],
            "concepts": batch.get("concepts", None),
        }

        loss_dict = loss_fn.compute(outputs, batch_dict, epoch=0)

        print("✓ Training step successful!")
        print(f"  Loss components:")
        for key, value in loss_dict.items():
            print(f"    {key}: {value}")

        # Test backward pass
        loss = loss_dict["total_loss"]
        loss.backward()

        print("✓ Backward pass successful!")

        return True

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test training setup")
    parser.add_argument(
        "--dataset",
        type=str,
        default="snli",
        choices=["snli", "chaosnli"],
        help="Dataset to test",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GACS Training Setup Test")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}")

    # Test 1: Load dataset
    success, dataloaders, num_classes, num_concepts, latent_dim = test_dataset_loading(
        args.dataset
    )
    if not success:
        print("\n❌ Dataset loading failed. Exiting.")
        return 1

    # Test 2: Create model
    success, model = test_model_creation(args.dataset, num_classes, num_concepts, latent_dim)
    if not success:
        print("\n❌ Model creation failed. Exiting.")
        return 1

    # Test 3: Forward pass
    success = test_forward_pass(model, dataloaders)
    if not success:
        print("\n❌ Forward pass failed. Exiting.")
        return 1

    # Test 4: Training step
    success = test_training_step(model, dataloaders)
    if not success:
        print("\n❌ Training step failed. Exiting.")
        return 1

    # All tests passed
    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print(f"{'='*60}")
    print(f"\nYou can now train the model:")
    print(f"  python train_with_cached_latents.py --dataset {args.dataset} --quick")
    print(f"\nOr for full training:")
    print(f"  python train_with_cached_latents.py --dataset {args.dataset} --epochs 30")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
