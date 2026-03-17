#!/usr/bin/env python3
"""
Train GACS Model with Cached Latents
====================================

This script trains the GACS model using pre-extracted cached latents
from the parent directory's latent_cache.

Usage:
    python train_with_cached_latents.py --dataset snli --epochs 30
    python train_with_cached_latents.py --dataset chaosnli --epochs 30 --quick

Available cached datasets:
    - snli: SNLI with CLS latents
    - chaosnli: ChaosNLI with low-rank SVD latents

Output:
    - Checkpoints saved every epoch: checkpoint_epoch_XXX.pt
    - Best model: best_model.pt
    - Training history: training_history.json
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path to use cached data
sys.path.insert(0, str(Path(__file__).parent.parent))

from gacs.models.vae import GACSModel
from gacs.training.trainer import GACSTrainer
from gacs.losses.gacs_loss import GACSLoss
from gacs.config import GACSConfig


class CachedLatentDataset(Dataset):
    """Dataset wrapper for cached latent features."""

    def __init__(self, latents_path, labels_path, concepts_path=None):
        """
        Args:
            latents_path: Path to .npy file with latent features [N, D]
            labels_path: Path to .npy file with labels [N]
            concepts_path: Optional path to concepts [N, K]
        """
        self.latents = np.load(latents_path)
        self.labels = np.load(labels_path)

        # Load concepts if available
        self.concepts = None
        if concepts_path and Path(concepts_path).exists():
            self.concepts = np.load(concepts_path)

        print(f"Loaded {len(self)} samples from {Path(latents_path).parent.name}")
        print(f"  Latents shape: {self.latents.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        if self.concepts is not None:
            print(f"  Concepts shape: {self.concepts.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "latents": torch.tensor(self.latents[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.concepts is not None:
            item["concepts"] = torch.tensor(self.concepts[idx], dtype=torch.float32)
        else:
            # Dummy concepts (will be ignored)
            item["concepts"] = torch.zeros(4, dtype=torch.float32)

        return item


def get_cached_dataloaders(dataset_name, batch_size=32, num_workers=2):
    """
    Create dataloaders from cached latents.

    Args:
        dataset_name: 'snli' or 'chaosnli'
        batch_size: Batch size
        num_workers: Number of dataloader workers

    Returns:
        dict with 'train', 'val', 'test' dataloaders
    """
    latent_cache_dir = Path(__file__).parent.parent / "latent_cache"

    if dataset_name == "snli":
        # SNLI CLS latents
        base_path = latent_cache_dir
        train_data = CachedLatentDataset(
            base_path / "snli_cls_train_latents.npy",
            base_path / "snli_cls_train_labels.npy",
            base_path / "snli_cls_train_concepts.npy",
        )
        val_data = CachedLatentDataset(
            base_path / "snli_cls_val_latents.npy",
            base_path / "snli_cls_val_labels.npy",
            base_path / "snli_cls_val_concepts.npy",
        )
        test_data = CachedLatentDataset(
            base_path / "snli_cls_test_latents.npy",
            base_path / "snli_cls_test_labels.npy",
            base_path / "snli_cls_test_concepts.npy",
        )
        num_classes = 3  # entailment, neutral, contradiction
        num_concepts = 4
        latent_dim = 768  # BERT CLS dimension

    elif dataset_name == "chaosnli":
        # ChaosNLI low-rank SVD latents
        base_path = latent_cache_dir
        train_data = CachedLatentDataset(
            base_path / "chaosnli_low_rank_rank128_svd_train_latents.npy",
            base_path / "chaosnli_cls_train_labels.npy",
            base_path / "chaosnli_low_rank_rank128_svd_train_concepts.npy",
        )
        # For ChaosNLI, we might not have separate val split, use test for both
        test_data = CachedLatentDataset(
            base_path / "chaosnli_cls_test_labels.npy",  # Need to check this path
            base_path / "chaosnli_cls_test_labels.npy",
            base_path / "chaosnli_cls_test_labels.npy",
        )
        num_classes = 3
        num_concepts = 4
        latent_dim = 128  # low-rank dimension

        # Use train/val split
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = torch.utils.data.random_split(
            train_data, [train_size, val_size]
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders, num_classes, num_concepts, latent_dim


def create_config_for_cached(dataset_name, quick=False):
    """Create GACS config for cached latent training."""
    config = GACSConfig()

    # Get dataset info
    dataloaders, num_classes, num_concepts, latent_dim = get_cached_dataloaders(
        dataset_name, batch_size=32
    )

    config.data.name = dataset_name
    config.data.num_classes = num_classes
    config.data.cebab_num_concepts = num_concepts
    config.data.batch_size = 32

    # Model config - we're training on latents, not raw text
    # So we don't need BERT encoder
    config.model.encoder_name = "bert-base-uncased"
    config.model.z_dim = 64
    config.model.concept_dim = num_concepts

    # Adjust for quick run
    if quick:
        config.training.epochs = 5
        config.data.batch_size = 16
        config.model.z_dim = 32
        config.probe.num_directions = 30
    else:
        config.training.epochs = 30

    return config


class LatentGACSModel(nn.Module):
    """
    GACS model that operates directly on cached latents.
    Skips the BERT encoder and works with pre-extracted features.
    """

    def __init__(self, config, latent_dim):
        super().__init__()
        self.config = config
        mc = config.model

        # Input is already latent features, so we start from encoder_head
        # Instead of BERT → h → (μ, logσ²), we have: cached_latent → (μ, logσ²)
        self.input_dim = latent_dim

        # Encoder head: latent → posterior parameters
        self.encoder_head = nn.Sequential(
            nn.Linear(self.input_dim, mc.hidden_dim),
            nn.GELU(),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.hidden_dim, mc.hidden_dim),
        )

        self.mu_head = nn.Linear(mc.hidden_dim, mc.z_dim)
        self.logvar_head = nn.Linear(mc.hidden_dim, mc.z_dim)

        # Factor decoder: z → s
        from gacs.models.vae import FactorDecoder
        self.factor_decoder = FactorDecoder(
            mc.z_dim, mc.concept_dim, mc.hidden_dim, mc.dropout
        )

        # Classifier
        from gacs.models.vae import Classifier
        self.classifier = Classifier(
            mc.concept_dim, config.data.num_classes, mc.dropout
        )

        # Reconstruction head: z → h_recon
        self.recon_head = nn.Sequential(
            nn.Linear(mc.z_dim, mc.hidden_dim),
            nn.GELU(),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.hidden_dim, self.input_dim),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def encode(self, latents):
        """Encode cached latents to posterior."""
        h = self.encoder_head(latents)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        return h, mu, logvar

    def forward(self, latents):
        h, mu, logvar = self.encode(latents)
        z = self.reparameterize(mu, logvar)
        s = self.factor_decoder(z)
        logits, importance = self.classifier(s)
        recon = self.recon_head(z)

        return {
            "h_norm": h,  # Use encoder output as reconstruction target
            "z_mu": mu,
            "z_logvar": logvar,
            "z": z,
            "s": s,
            "logits": logits,
            "importance": importance,
            "recon": recon,
        }


def main():
    parser = argparse.ArgumentParser(description="Train GACS with cached latents")
    parser.add_argument(
        "--dataset",
        type=str,
        default="snli",
        choices=["snli", "chaosnli"],
        help="Dataset to train on",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--z_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_cached",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = create_config_for_cached(args.dataset, quick=args.quick)

    # Override with command line args
    if args.epochs is not None:
        config.training.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.model.z_dim = args.z_dim
    config.training.learning_rate = args.lr
    config.training.head_lr = args.lr
    config.training.output_dir = args.output_dir

    print(f"\n{'='*60}")
    print(f"Training GACS on {args.dataset.upper()} (cached latents)")
    print(f"{'='*60}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"LR: {config.training.learning_rate}")
    print(f"z_dim: {config.model.z_dim}")
    print(f"Output: {config.training.output_dir}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading cached latents...")
    dataloaders, num_classes, num_concepts, latent_dim = get_cached_dataloaders(
        args.dataset, batch_size=config.data.batch_size
    )

    config.data.num_classes = num_classes
    config.model.concept_dim = num_concepts

    print(f"\nDataset splits:")
    for split, loader in dataloaders.items():
        print(f"  {split}: {len(loader.dataset)} samples")

    # Create model
    print("\nBuilding model...")
    model = LatentGACSModel(config, latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Device: {device}")

    # Create dummy tokenizer (not used for cached latents)
    tokenizer = None

    # Create loss function
    from gacs.losses.gacs_loss import GACSLoss
    loss_fn = GACSLoss(config)

    # Create trainer
    from gacs.training.trainer import GACSTrainer
    trainer = GACSTrainer(config, model, dataloaders, tokenizer)

    # Train
    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {config.training.output_dir}")
    print(f"  - Per epoch: checkpoint_epoch_XXX.pt")
    print(f"  - Best model: best_model.pt\n")

    trainer.train()

    # Save results
    trainer.save_results()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best val accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Results saved to: {config.training.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
