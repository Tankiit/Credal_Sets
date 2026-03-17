#!/usr/bin/env python3
"""
Train GACS on Derm7pt Dataset
==============================

Dermatology dataset with 7-point checklist criteria and modality shift.

Key features:
- Supervised factor labels (7-point criteria)
- Modality shift: dermoscopic (train) → clinical (test_shift)
- Image-based classification

Usage:
    # Standard training
    python train_derm7pt.py --data_dir /path/to/derm7pt --epochs 50

    # With evaluation
    python train_derm7pt.py --data_dir /path/to/derm7pt --epochs 50 --eval

    # Quick test
    python train_derm7pt.py --data_dir /path/to/derm7pt --quick

Dataset splits:
    - train: Training set (dermoscopic images)
    - val: Validation set (dermoscopic images)
    - test_iid: Test set (dermoscopic images, same modality)
    - test_shift: Test set (clinical images, MODALITY SHIFT)
    - probe: Probe set for geometric calibration
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from gacs.data.derm7pt_loader import get_derm7pt_loaders
from gacs.training.shared_trainer import GACSSharedTrainer, VisionModelWrapper
from gacs.training.losses import Derm7ptFactorHeadLoss
from gacs.training.evaluator import GACSEvaluator
from gacs.credal.calibration import CredalSetCalibrator
from gacs.credal.geometric_probe import GeometricProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Derm7ptGACSModel(nn.Module):
    """
    GACS model for Derm7pt (dermatology images).

    Architecture:
        - CNN encoder (4 conv layers)
        - VAE latent space (z_dim)
        - Interpretable factors (7 for 7-point criteria)
        - Binary classifier (melanoma vs benign)
        - Decoder (reconstruction)
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        z_dim=64,
        num_factors=7,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        # CNN encoder
        self.encoder = nn.Sequential(
            # Input: [B, 3, 224, 224]
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Flatten(),
        )

        self.encoder_dim = 256

        # VAE encoder head
        self.mu_head = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

        # Reparameterize: z ~ q(z|x)
        self.z_dim = z_dim

        # Factor decoder (interpretable concepts)
        self.factor_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors),
            nn.Sigmoid(),  # s ∈ [0,1]^K
        )

        # Classifier (uses factors)
        self.classifier = nn.Sequential(
            nn.Linear(num_factors, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7)),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.Sigmoid(),  # [0,1] output
        )

    def encode(self, x):
        """Encode x to (z_mu, z_logvar)."""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_factors(self, z):
        """Decode z to interpretable factors s."""
        return self.factor_decoder(z)

    def decode(self, z):
        """Decode z to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        s = self.decode_factors(z)
        logits = self.classifier(s)
        recon = self.decode(z)

        return {
            "logits": logits,
            "z_mu": mu,
            "z_logvar": logvar,
            "s": s,
            "recon": recon,
            "z": z,
        }

    def get_probeable_parameters(self, scope="decoder"):
        """Get parameters for geometric probe."""
        if scope == "decoder":
            return list(self.decoder.parameters())
        elif scope == "factor_decoder":
            return list(self.factor_decoder.parameters())
        else:
            return list(self.parameters())


def create_config(args):
    """Create configuration from args."""
    class Config:
        def __init__(self):
            # Model
            self.z_dim = args.z_dim
            self.num_factors = 7  # 7-point criteria
            self.hidden_dim = args.hidden_dim
            self.dropout = args.dropout

            # Training
            self.lr = args.lr
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.weight_decay = 1e-4
            self.warmup_epochs = 10
            self.patience = 15

            # Loss weights
            self.recon_weight = args.recon_weight
            self.kl_max = args.kl_max
            self.div_weight = args.div_weight
            self.sparse_weight = args.sparse_weight
            self.criteria_weight = args.criteria_weight

            # System
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.output_dir = Path(args.output_dir)
            self.save_every_n_epochs = 5

    return Config()


def train_derm7pt(args):
    """Train GACS on Derm7pt dataset."""

    # Setup
    config = create_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")

    # Load data
    logger.info(f"Loading Derm7pt data from: {args.data_dir}")

    loaders = get_derm7pt_loaders(
        dir_release=args.data_dir,
        batch_size=config.batch_size,
        num_workers=4,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_iid_loader = loaders["test_iid"]
    test_shift_loader = loaders["test_shift"]  # Modality shift!
    probe_loader = loaders["probe"]

    logger.info(
        f"Data loaded | "
        f"train={len(train_loader.dataset)} | "
        f"val={len(val_loader.dataset)} | "
        f"test_iid={len(test_iid_loader.dataset)} | "
        f"test_shift={len(test_shift_loader.dataset)} | "
        f"probe={len(probe_loader.dataset)}"
    )

    # Create model
    model = Derm7ptGACSModel(
        in_channels=3,
        num_classes=2,  # Binary: melanoma vs benign
        z_dim=config.z_dim,
        num_factors=config.num_factors,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )

    logger.info(
        f"Model created | "
        f"z_dim={config.z_dim} | "
        f"num_factors={config.num_factors} | "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    # Wrapper and trainer
    wrapper = VisionModelWrapper(model)

    # Criteria loss (supervised factor labels)
    criteria_loss_fn = Derm7ptFactorHeadLoss(
        num_criteria=7,
        weight=config.criteria_weight,
    )

    trainer = GACSSharedTrainer(
        wrapper=wrapper,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criteria_loss_fn=criteria_loss_fn,
        encoder_lr_scale=1.0,  # CNN trained from scratch
        recon_weight=config.recon_weight,
        kl_max=config.kl_max,
        div_weight=config.div_weight,
        sparse_weight=config.sparse_weight,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Get probe-ready model
    model, probe_params = trainer.get_probe_ready_model(scope="decoder")
    logger.info(f"Probe parameters: {sum(p.numel() for p in probe_params):,}")

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save trainer state for later evaluation
    trainer_state = {
        "model_path": str(config.output_dir / "best_model.pt"),
        "val_acc": trainer.best_val_acc,
        "probe_params": probe_params,
    }

    with open(config.output_dir / "trainer_state.json", "w") as f:
        json.dump({k: str(v) if not isinstance(v, list) else [str(p) for p in v]
                  for k, v in trainer_state.items()}, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best val accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Model saved to: {config.output_dir / 'best_model.pt'}")
    logger.info("\nTo run evaluation:")
    logger.info(f"  python evaluate_derm7pt.py --output_dir {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GACS on Derm7pt")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/tanmoy/research/data/derm7pt",
        help="Path to Derm7pt data directory",
    )

    # Model architecture
    parser.add_argument("--z_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Loss weights
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Reconstruction weight")
    parser.add_argument("--kl_max", type=float, default=1.0, help="Max KL weight")
    parser.add_argument("--div_weight", type=float, default=0.1, help="Diversity weight")
    parser.add_argument("--sparse_weight", type=float, default=0.05, help="Sparsity weight")
    parser.add_argument("--criteria_weight", type=float, default=0.5, help="Criteria loss weight")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Run full evaluation")
    parser.add_argument("--target_coverage", type=float, default=0.9, help="Target coverage for credal sets")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/derm7pt", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test run")

    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.output_dir = "results/derm7pt_quick"

    train_derm7pt(args)


if __name__ == "__main__":
    main()
