"""
GACS Main Entry Point
======================

Usage:
    python main.py --dataset cebab --quick          # Quick test
    python main.py --dataset cebab --epochs 30      # Full training
    python main.py --dataset amazon --epochs 20     # Amazon cross-domain
    python main.py --evaluate_only --checkpoint outputs/best_model.pt
"""

import argparse
import torch
import numpy as np
import random

from gacs.config import get_config
from gacs.models.vae import GACSModel
from gacs.data.datasets import get_dataloaders
from gacs.trainer import GACSTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="GACS")
    parser.add_argument("--dataset", type=str, default="cebab",
                        choices=["cebab", "amazon", "sst2", "imdb", "agnews"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--z_dim", type=int, default=None)
    parser.add_argument("--probe_scope", type=str, default="decoder",
                        choices=["all", "concept_layer", "classifier", "decoder"])
    parser.add_argument("--num_directions", type=int, default=None)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_hessian", action="store_true")
    args = parser.parse_args()

    config = get_config(dataset=args.dataset, quick=args.quick)

    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.z_dim is not None:
        config.model.z_dim = args.z_dim
    if args.num_directions is not None:
        config.probe.num_directions = args.num_directions
    config.probe.probe_scope = args.probe_scope
    config.probe.use_hessian = args.use_hessian
    config.training.output_dir = args.output_dir
    config.training.seed = args.seed

    set_seed(config.training.seed)

    print(f"Dataset: {config.data.name}")
    print(f"Device: {config.training.device}")
    print(f"z_dim: {config.model.z_dim}, concept_dim: {config.model.concept_dim}")
    print(f"Classes: {config.data.num_classes}")

    print("\nLoading data...")
    dataloaders, tokenizer = get_dataloaders(config)
    print(f"Splits: {list(dataloaders.keys())}")

    print("\nBuilding model...")
    model = GACSModel(config)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")

    trainer = GACSTrainer(config, model, dataloaders, tokenizer)

    if not args.evaluate_only:
        print("\nTraining...")
        trainer.train()

    print("\nRunning geometric probes...")
    probe_results = trainer.run_geometric_probes()
    rho = probe_results["rho"]

    print("\nRunning credal evaluation...")
    trainer.run_credal_evaluation(rho)

    trainer.save_results()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Best val accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Degeneracy ratio rho: {rho:.4f}")
    print(f"  GACS epsilon: {trainer.credal_constructor.rho_to_epsilon(rho):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
