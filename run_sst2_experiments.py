#!/usr/bin/env python

"""
SST-2 Experiments with Credal CBM
Runs uncertainty correlation experiments on the SST-2 sentiment analysis dataset
"""

import os
import sys
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from credal_cbm_model import CredalCBM, CredalCBMConfig
from exp1_uncertainty_correlation import UncertaintyCorrelationExperiment, load_sst2_data


def run_sst2_credal_cbm_experiment(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_samples: int = 1000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = './results_sst2'
):
    """
    Run complete SST-2 experiment with Credal CBM

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_samples: Maximum samples for inference analysis
        device: Device to run on
        save_dir: Directory to save results
    """

    print("="*80)
    print("SST-2 CREDAL CBM EXPERIMENT")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Max samples for analysis: {max_samples}")
    print(f"Save directory: {save_dir}")
    print("="*80)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load SST-2 data
    print("\nLoading SST-2 dataset...")
    train_loader, val_loader = load_sst2_data(tokenizer, batch_size=batch_size)

    # Create model configuration
    config = CredalCBMConfig(
        base_model_name="distilbert-base-uncased",
        num_concepts=20,
        num_classes=2,
        n_ensemble_heads=5,
        use_credal=True,
        use_rationales=False,
        hidden_dim=256,
        dropout=0.3,
        concept_loss_weight=0.5,
        classification_loss_weight=1.0
    )

    print("\nModel configuration:")
    print(f"  Base model: {config.base_model_name}")
    print(f"  Num concepts: {config.num_concepts}")
    print(f"  Ensemble heads: {config.n_ensemble_heads}")
    print(f"  Use credal sets: {config.use_credal}")
    print(f"  Use rationales: {config.use_rationales}")

    # Initialize model
    print("\nInitializing Credal CBM...")
    model = CredalCBM(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Initialize experiment tracker
    print("\nInitializing uncertainty experiment tracker...")
    experiment = UncertaintyCorrelationExperiment(
        model_class=lambda: model,
        tokenizer=tokenizer,
        device=device
    )

    # Train model with uncertainty tracking
    print("\n" + "="*80)
    print("PHASE 1: TRAINING WITH UNCERTAINTY TRACKING")
    print("="*80)

    trained_model = experiment.train_and_track_uncertainty(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr
    )

    # Run full experiment analysis
    print("\n" + "="*80)
    print("PHASE 2: INFERENCE AND ANALYSIS")
    print("="*80)

    corr_results, correctness_analysis = experiment.run_full_experiment(
        model=trained_model,
        dataloader=val_loader,
        max_samples=max_samples,
        save_dir=save_dir
    )

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\nModel Performance:")
    print(f"  Accuracy: {corr_results['accuracy']:.4f}")

    print(f"\nUncertainty-Error Correlations:")
    print(f"  Epistemic:")
    print(f"    Spearman ρ: {corr_results['epistemic']['spearman_rho']:.4f}")
    print(f"    p-value: {corr_results['epistemic']['spearman_p']:.6f}")
    print(f"  Aleatoric:")
    print(f"    Spearman ρ: {corr_results['aleatoric']['spearman_rho']:.4f}")
    print(f"    p-value: {corr_results['aleatoric']['spearman_p']:.6f}")
    print(f"  Total:")
    print(f"    Spearman ρ: {corr_results['total']['spearman_rho']:.4f}")
    print(f"    p-value: {corr_results['total']['spearman_p']:.6f}")

    print(f"\nUncertainty by Correctness:")
    print(f"  Correct predictions (n={correctness_analysis['correct']['count']}):")
    print(f"    Epistemic: {correctness_analysis['correct']['epistemic_mean']:.4f} ± {correctness_analysis['correct']['epistemic_std']:.4f}")
    print(f"    Aleatoric: {correctness_analysis['correct']['aleatoric_mean']:.4f} ± {correctness_analysis['correct']['aleatoric_std']:.4f}")
    print(f"  Incorrect predictions (n={correctness_analysis['incorrect']['count']}):")
    print(f"    Epistemic: {correctness_analysis['incorrect']['epistemic_mean']:.4f} ± {correctness_analysis['incorrect']['epistemic_std']:.4f}")
    print(f"    Aleatoric: {correctness_analysis['incorrect']['aleatoric_mean']:.4f} ± {correctness_analysis['incorrect']['aleatoric_std']:.4f}")

    print(f"\nResults saved to: {save_dir}")
    print("="*80)

    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, 'credal_cbm_sst2.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': {
            'correlations': corr_results,
            'correctness_analysis': correctness_analysis
        }
    }, checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    return model, experiment, corr_results, correctness_analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run SST-2 Credal CBM experiments')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples for analysis')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./results_sst2', help='Save directory')

    args = parser.parse_args()

    run_sst2_credal_cbm_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        device=args.device,
        save_dir=args.save_dir
    )
