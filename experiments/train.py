"""
Train a HybridCredalSLVM on a dataset. Optionally instrument with gradient
isolation probes.

Usage:
    # Plain training on CEBaB, frozen encoder
    python -m experiments.train --dataset cebab --epochs 10

    # Same with gradient isolation logging
    python -m experiments.train --dataset cebab --epochs 10 --grad_iso

    # Unfrozen encoder (Option C stress test)
    python -m experiments.train --dataset cebab --epochs 10 --grad_iso --unfreeze
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from loaders import LOADERS
from models.cbm_body import CBMBody
from models.protop_body import ProtoPBody
from models.senn_body import SENNBody
from models.slvm_base import HybridCredalSLVM, SLVMConfig
from training.trainer import HybridCredalCBMTrainer, InstrumentedTrainer


# Per-dataset hyperparameter defaults. Override with CLI flags.
DATASET_DEFAULTS = {
    "cebab":       {"batch_size":  8, "max_length": 256, "epochs":  50, "lr": 1e-3, "prior_sigma": 0.5},
    "hatexplain":  {"batch_size": 16, "max_length": 128, "epochs": 50, "lr": 1e-3, "prior_sigma": 0.5},
    "goemotions":  {"batch_size": 32, "max_length":  64, "epochs": 50, "lr": 5e-5, "prior_sigma": 0.5},
    "sst2":        {"batch_size": 32, "max_length": 128, "epochs": 50, "lr": 1e-3, "prior_sigma": 0.5},
}


def build_body(model_type: str, config: SLVMConfig):
    """Factory for SLVM body based on model_type."""
    if model_type == "cbm":
        return CBMBody(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
        )
    elif model_type == "senn":
        return SENNBody(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            stability_weight=2e-4,
            stability_epsilon=0.01,
        )
    elif model_type == "protop":
        return ProtoPBody(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            proto_dim=128,
            num_protos_per_concept=1,
            hidden_dim=config.hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    p = argparse.ArgumentParser(description="Train HybridCredalSLVM.")
    p.add_argument("--dataset", required=True,
                   choices=["cebab", "hatexplain", "goemotions", "sst2"])
    p.add_argument("--model", type=str, default="cbm",
                   choices=["cbm", "senn", "protop"],
                   help="Which SLVM architecture to train.")
    p.add_argument("--encoder", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader worker count. Increase for faster batch loading.")
    p.add_argument("--mc_samples", type=int, default=None,
                   help="MC samples for the variational concept encoder. Lower values are faster.")
    p.add_argument("--aleatoric_weight", type=float, default=None,
                   help="Weight on H-supervision (annotator entropy). Default from DATASET_DEFAULTS.")
    p.add_argument("--aleatoric_unknown_weight", type=float, default=0.0,
                   help="Weight on U-supervision (unknown-rate). Default 0 (main text).")
    p.add_argument("--aleatoric_prior", type=float, default=None,
                   help="AleatoricHead prior mean. Default 0.05.")

    p.add_argument("--grad_iso", action="store_true",
                   help="Log per-step gradient isolation records.")
    p.add_argument("--unfreeze", action="store_true",
                   help="Train with encoder unfrozen (Option C stress test).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Training device. 'auto' prefers CUDA, then MPS, then CPU.")
    p.add_argument("--log_dir", type=Path, default=None,
                   help="Directory for TensorBoard/W&B run files. Defaults to <save_dir>/logs.")
    p.add_argument("--no_tensorboard", action="store_true",
                   help="Disable TensorBoard logging.")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging if installed.")
    p.add_argument("--wandb_project", default="neurips-credal",
                   help="W&B project name.")
    p.add_argument("--wandb_run_name", default=None,
                   help="Optional W&B run name.")
    p.add_argument("--wandb_mode", default="offline",
                   choices=["offline", "online", "disabled"],
                   help="W&B mode. Offline keeps runs local by default.")

    p.add_argument("--save_dir", type=Path, default=None,
                   help="Default: checkpoints/hybrid_credal_<dataset>[_unfrozen].")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Resolve hyperparameters
    defaults = DATASET_DEFAULTS[args.dataset]
    batch_size = args.batch_size or defaults["batch_size"]
    max_length = args.max_length or defaults["max_length"]
    epochs = args.epochs or defaults["epochs"]
    lr = args.lr or defaults["lr"]

    torch.manual_seed(args.seed)

    # Load dataset
    bundle = LOADERS[args.dataset].load(
        tokenizer_name=args.encoder,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=args.num_workers if args.num_workers is not None else 0,
    )
    print(f"[bundle] {bundle.name}: "
          f"train={bundle.train_size} val={bundle.val_size} test={bundle.test_size}  "
          f"concepts={bundle.num_concepts}  classes={bundle.num_classes}")

    # Build model
    config = SLVMConfig(
        encoder_name=args.encoder,
        freeze_encoder=(not args.unfreeze),
        num_concepts=bundle.num_concepts,
        concept_names=list(bundle.concept_names),
        num_classes=bundle.num_classes,
        prior_sigma=defaults["prior_sigma"],
        aleatoric_weight=(args.aleatoric_weight if args.aleatoric_weight is not None else 2.0),
        aleatoric_unknown_weight=args.aleatoric_unknown_weight,
        aleatoric_prior=(args.aleatoric_prior if args.aleatoric_prior is not None else 0.05),
        model_type=args.model,
    )
    if args.mc_samples is not None:
        config.num_mc_samples = args.mc_samples
    body = build_body(args.model, config)
    model = HybridCredalSLVM(config, body)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] {total:,} params ({trainable:,} trainable, "
          f"freeze_encoder={config.freeze_encoder})")

    # Save dir
    if args.save_dir is None:
        tag = "unfrozen" if args.unfreeze else "frozen"
        args.save_dir = Path(f"checkpoints/hybrid_credal_{args.model}_{args.dataset}_{tag}")

    # Trainer
    TrainerCls = InstrumentedTrainer if args.grad_iso else HybridCredalCBMTrainer
    trainer = TrainerCls(
        model=model,
        config=config,
        bundle=bundle,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        use_tensorboard=not args.no_tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    # Train
    trainer.fit(num_epochs=epochs, lr=lr)

    # Final test
    print("\n[test] evaluating best checkpoint on test split...")
    ckpt = torch.load(args.save_dir / "best_model.pt",
                      map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = trainer.evaluate(bundle.test_loader)
    print(f"  test acc:      {test_metrics.accuracy:.4f}")
    print(f"  ρ(EU, AU):     {test_metrics.rho_eu_au:+.3f}")
    print(f"  ρ(σ_epi, err): {test_metrics.rho_eu_error:+.3f}")
    print(f"  ρ(σ_ale, H):   {test_metrics.rho_ale_entropy:+.3f}")
    trainer._log_metrics({"test": test_metrics.to_dict()}, step=epochs)

    # Persist test metrics alongside the checkpoint
    import json
    with (args.save_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics.to_dict(), f, indent=2, default=float)

    trainer.close()


if __name__ == "__main__":
    main()
