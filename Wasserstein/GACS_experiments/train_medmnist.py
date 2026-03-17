#!/usr/bin/env python3
"""
Train GACS on MedMNIST Datasets
================================

Vision-based training for medical imaging datasets.

Usage:
    # Single dataset (standard)
    python train_medmnist.py --dataset pathmnist --epochs 30

    # Cross-view OrganMNIST (headline)
    python train_medmnist.py --dataset organamnist --target organcmnist --epochs 30

    # Tier 1 main results
    python train_medmnist.py --dataset pathmnist --epochs 30
    python train_medmnist.py --dataset dermamnist --epochs 30
    python train_medmnist.py --dataset bloodmnist --epochs 30
    python train_medmnist.py --dataset organamnist --target organcmnist --epochs 30

    # Quick test
    python train_medmnist.py --dataset bloodmnist --quick

Available datasets:
    Tier 1 (main results):
        - pathmnist, dermamnist, bloodmnist
        - organamnist → organcmnist (cross-view)

    Tier 2 (appendix sweep):
        - breastmnist, pneumoniamnist, retinamnist, octmnist, tissuemnist
"""

import argparse
import torch
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from gacs.data.medmnist_loader import (
    get_standard_loaders,
    get_cross_view_loaders,
    get_all_medmnist_loaders,
    get_dataset_info,
    get_tier1_names,
    DATASET_REGISTRY,
)
from gacs.models.vae import GACSModel
from gacs.config import GACSConfig


class VisionGACSModel(torch.nn.Module):
    """
    GACS model for vision data (MedMNIST).

    Similar to text GACS but with CNN encoder instead of BERT.
    """

    def __init__(self, config, input_channels=3, input_size=28):
        super().__init__()
        self.config = config
        mc = config.model

        # CNN encoder
        self.encoder = torch.nn.Sequential(
            # Input: [B, C, 28, 28]
            torch.nn.Conv2d(input_channels, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 14x14

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 7x7

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),  # [B, 128, 1, 1]
        )

        self.encoder_dim = 128

        # Encoder head
        self.mu_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dim, mc.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mc.dropout),
            torch.nn.Linear(mc.hidden_dim, mc.z_dim),
        )

        self.logvar_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dim, mc.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mc.dropout),
            torch.nn.Linear(mc.hidden_dim, mc.z_dim),
        )

        # Factor decoder (from text GACS)
        from gacs.models.vae import FactorDecoder
        self.factor_decoder = FactorDecoder(
            mc.z_dim, mc.concept_dim, mc.hidden_dim, mc.dropout
        )

        # Classifier
        from gacs.models.vae import Classifier
        self.classifier = Classifier(
            mc.concept_dim, config.data.num_classes, mc.dropout
        )

        # Reconstruction head
        self.recon_head = torch.nn.Sequential(
            torch.nn.Linear(mc.z_dim, mc.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mc.dropout),
            torch.nn.Linear(mc.hidden_dim, self.encoder_dim),
        )

    def encode(self, images):
        """Encode images to latent posterior."""
        h = self.encoder(images)  # [B, 128, 1, 1]
        h = h.squeeze(-1).squeeze(-1)  # [B, 128]

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)

        # L2 normalize for D1 compliance
        h_norm = torch.nn.functional.normalize(h, dim=-1)
        return h_norm, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, images):
        h_norm, mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar)
        s = self.factor_decoder(z)
        logits, importance = self.classifier(s)
        recon = self.recon_head(z)

        return {
            "h_norm": h_norm,
            "z_mu": mu,
            "z_logvar": logvar,
            "z": z,
            "s": s,
            "logits": logits,
            "importance": importance,
            "recon": recon,
        }


def train_on_medmnist(
    dataset_name,
    target_view=None,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    z_dim=64,
    quick=False,
    output_dir="outputs_medmnist",
    seed=42,
):
    """Train GACS on a MedMNIST dataset."""

    print(f"\n{'='*60}")
    print(f"Training GACS on {dataset_name.upper()}")
    if target_view:
        print(f"Cross-view: {dataset_name} → {target_view}")
    print(f"{'='*60}\n")

    # Set seed
    torch.manual_seed(seed)

    # Get dataset info
    info = get_dataset_info(dataset_name)
    n_classes = info["n_classes"]
    n_channels = info["n_channels"]

    # Load data
    if target_view:
        # Cross-view loading
        loaders = get_cross_view_loaders(
            source=dataset_name,
            target=target_view,
            batch_size=batch_size,
            download=True,
        )
        print(f"Cross-view mode: {dataset_name} → {target_view}")
    else:
        # Standard loading
        loaders = get_standard_loaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            download=True,
        )

    print(f"\nDataset splits:")
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader.dataset)} samples")

    # Create config
    config = GACSConfig()
    config.data.num_classes = n_classes
    config.model.z_dim = z_dim
    config.model.concept_dim = max(4, n_classes)  # At least 4 concepts

    if quick:
        config.training.epochs = 5
        epochs = 5
    else:
        config.training.epochs = epochs

    config.training.learning_rate = lr
    config.training.head_lr = lr
    config.training.output_dir = output_dir
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    print("\nBuilding model...")
    model = VisionGACSModel(config, input_channels=n_channels, input_size=28)
    model.to(config.training.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Device: {config.training.device}")

    # Create trainer
    from gacs.training.vision_trainer import VisionTrainer
    from gacs.training.vision_losses import VisionGACSLoss

    loss_fn = VisionGACSLoss(config)
    trainer = VisionTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders.get("test"),
    )

    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Checkpoints: {config.training.output_dir}/checkpoint_epoch_*.pt")
    print(f"Best model: {config.training.output_dir}/best_model.pt\n")

    results = trainer.train()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best val accuracy: {results['best_val_acc']:.4f}")
    print(f"  Final epoch: {results['final_epoch']}")
    print(f"  Output: {config.training.output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train GACS on MedMNIST")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bloodmnist",
        help="MedMNIST dataset name",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target view for cross-view (e.g., organcmnist)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--z_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--output_dir", type=str, default="outputs_medmnist")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate dataset
    if args.dataset.lower() not in DATASET_REGISTRY:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(DATASET_REGISTRY.keys())}")
        return 1

    train_on_medmnist(
        dataset_name=args.dataset,
        target_view=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        z_dim=args.z_dim,
        quick=args.quick,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
