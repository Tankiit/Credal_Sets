"""
Training Script for Gradient-Separated Variational Credal CBM
==============================================================

Key changes from original:
1. Uses GradSeparatedCredalCBM instead of VariationalCredalCBM
2. Calls model.set_epoch() each training epoch
3. Updated config for gradient-separated training
4. Simplified training loop without reg_factor (handled internally)

Author: Tanmoy
Target: ICML 2026
"""

import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import argparse

# Import model
from VCBM import GradSeparatedCredalCBM, GradSeparatedConfig, diagnose_gradient_separation
from load_cebab_direct import get_cebab_dataloaders


# ============================================================================
# UNCERTAINTY METRICS
# ============================================================================

class UncertaintyMetrics:
    """Container for all uncertainty-related metrics."""

    def __init__(self):
        # Task performance
        self.accuracy = 0.0
        self.loss = 0.0

        # Concept performance
        self.concept_accs = {}

        # Uncertainty statistics
        self.mean_epistemic = 0.0
        self.std_epistemic = 0.0
        self.mean_aleatoric = 0.0
        self.std_aleatoric = 0.0

        # Gradient separation metrics
        self.alpha = 0.0
        self.weight_std_mean = 0.0
        self.rho_var_err = 0.0  # Correlation between variational and error-based epistemic

        # Correlations (KEY METRICS!)
        self.rho_eu_au = 0.0           # Should be LOW (~0)
        self.p_eu_au = 1.0
        self.rho_eu_error = 0.0        # Should be HIGH (>0.25)
        self.p_eu_error = 1.0

        # Quadrant analysis
        self.trust_accuracy = 0.0
        self.trust_coverage = 0.0
        self.data_accuracy = 0.0
        self.data_coverage = 0.0
        self.review_accuracy = 0.0
        self.review_coverage = 0.0
        self.abstain_accuracy = 0.0
        self.abstain_coverage = 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        def convert_value(v):
            """Convert numpy/python types to JSON-serializable types."""
            if hasattr(v, 'item'):
                return v.item()
            elif isinstance(v, (np.floating, float)):
                return float(v)
            elif isinstance(v, (np.integer, int)):
                return int(v)
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            else:
                return v

        return {
            'accuracy': convert_value(self.accuracy),
            'loss': convert_value(self.loss),
            'concept_accs': convert_value(self.concept_accs),
            'mean_epistemic': convert_value(self.mean_epistemic),
            'std_epistemic': convert_value(self.std_epistemic),
            'mean_aleatoric': convert_value(self.mean_aleatoric),
            'std_aleatoric': convert_value(self.std_aleatoric),
            'alpha': convert_value(self.alpha),
            'weight_std_mean': convert_value(self.weight_std_mean),
            'rho_var_err': convert_value(self.rho_var_err),
            'rho_eu_au': convert_value(self.rho_eu_au),
            'p_eu_au': convert_value(self.p_eu_au),
            'rho_eu_error': convert_value(self.rho_eu_error),
            'p_eu_error': convert_value(self.p_eu_error),
            'trust_accuracy': convert_value(self.trust_accuracy),
            'trust_coverage': convert_value(self.trust_coverage),
            'data_accuracy': convert_value(self.data_accuracy),
            'data_coverage': convert_value(self.data_coverage),
            'review_accuracy': convert_value(self.review_accuracy),
            'review_coverage': convert_value(self.review_coverage),
            'abstain_accuracy': convert_value(self.abstain_accuracy),
            'abstain_coverage': convert_value(self.abstain_coverage),
        }


# ============================================================================
# TRAINER
# ============================================================================

class GradSeparatedCBMTrainer:
    """Trainer for gradient-separated Credal CBM."""

    def __init__(
        self,
        model: GradSeparatedCredalCBM,
        config: GradSeparatedConfig,
        save_dir: str = "./checkpoints/grad_separated",
        device: str = None
    ):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.current_epoch = 0
        self.best_val_acc = 0.0

    def train_epoch(
        self,
        train_loader,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()
        self.model.set_epoch(self.current_epoch)  # CRITICAL: Set epoch for scheduling

        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Get current alpha for display
        alpha = self.model.alpha_scheduler.get_alpha(self.current_epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')

            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)

            optimizer.zero_grad()

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_labels=concept_labels
            )

            loss = outputs['loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Track
            total_loss += loss.item()
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item(), 'α': f'{alpha:.3f}'})

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': (all_preds == all_labels).mean()
        }

    def evaluate(self, val_loader, compute_quadrants=False) -> UncertaintyMetrics:
        """Evaluate on validation set."""
        self.model.eval()
        self.model.set_epoch(self.current_epoch)  # Set for consistent alpha

        all_preds = []
        all_labels = []
        all_concept_probs = []
        all_concept_labels = []
        all_epistemic = []
        all_aleatoric = []
        all_epistemic_var = []
        all_epistemic_err = []
        all_probs = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                concept_labels = batch.get('concept_labels')

                if concept_labels is not None:
                    concept_labels = concept_labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    concept_labels=concept_labels
                )

                total_loss += outputs['loss'].item()

                all_preds.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_concept_probs.extend(outputs['concept_probs'].cpu().numpy())

                if concept_labels is not None:
                    all_concept_labels.extend(concept_labels.cpu().numpy())

                all_epistemic.append(outputs['epistemic'].cpu())
                all_epistemic_var.append(outputs['epistemic_var'].cpu())
                all_epistemic_err.append(outputs['epistemic_err'].cpu())
                all_aleatoric.append(outputs['aleatoric'].cpu())
                all_probs.append(outputs['probs'].cpu())

        # Concatenate
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_concept_probs = np.array(all_concept_probs)
        if all_concept_labels:
            all_concept_labels = np.array(all_concept_labels)
        else:
            all_concept_labels = None
        all_epistemic = torch.cat(all_epistemic).numpy()
        all_epistemic_var = torch.cat(all_epistemic_var).numpy()
        all_epistemic_err = torch.cat(all_epistemic_err).numpy()
        all_aleatoric = torch.cat(all_aleatoric).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # Task accuracy
        task_acc = (all_preds == all_labels).mean()

        # Metrics
        metrics = UncertaintyMetrics()
        metrics.accuracy = task_acc
        metrics.loss = total_loss / len(val_loader)

        # Uncertainty statistics
        metrics.mean_epistemic = all_epistemic.mean()
        metrics.std_epistemic = all_epistemic.std()
        metrics.mean_aleatoric = all_aleatoric.mean()
        metrics.std_aleatoric = all_aleatoric.std()

        # Alpha and weight std
        metrics.alpha = self.model.alpha_scheduler.get_alpha(self.current_epoch)
        metrics.weight_std_mean = self.model.variational_layer.get_weight_std().mean().item()

        # Correlations
        from scipy import stats

        # Epistemic vs Aleatoric (should be LOW)
        rho, p = stats.spearmanr(all_epistemic.flatten(), all_aleatoric.flatten())
        metrics.rho_eu_au = rho
        metrics.p_eu_au = p

        # Epistemic vs Error (should be HIGH)
        errors = (all_preds != all_labels).astype(float)
        epistemic_per_sample = all_epistemic.mean(axis=1)
        rho, p = stats.spearmanr(epistemic_per_sample, errors)
        metrics.rho_eu_error = rho
        metrics.p_eu_error = p

        # Variational vs Error-based epistemic (should be correlated)
        rho, p = stats.spearmanr(all_epistemic_var.flatten(), all_epistemic_err.flatten())
        metrics.rho_var_err = rho

        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        lr: float = 2e-5,
        warmup_steps: int = 500
    ) -> Dict:
        """Full training loop."""

        print("=" * 80)
        print(f"Starting Training for {num_epochs} Epochs")
        print("=" * 80)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        # Training loop
        history = {'train': [], 'val': []}
        best_metrics = None

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")

            # Print training status
            alpha = self.model.alpha_scheduler.get_alpha(epoch)
            if epoch < self.config.error_warmup_epochs:
                status = f"Warmup (error head inactive, α={alpha:.3f})"
            else:
                status = f"Active (error head training, α={alpha:.3f})"
            print(f"Status: {status}")

            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            history['train'].append(train_metrics)

            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

            # Validate
            val_metrics = self.evaluate(val_loader)
            history['val'].append(val_metrics.to_dict())

            print(f"\nValidation:")
            print(f"  Loss: {val_metrics.loss:.4f}")
            print(f"  Accuracy: {val_metrics.accuracy:.4f}")
            print(f"  Mean Epistemic: {val_metrics.mean_epistemic:.4f} ± {val_metrics.std_epistemic:.4f}")
            print(f"  Mean Aleatoric: {val_metrics.mean_aleatoric:.4f} ± {val_metrics.std_aleatoric:.4f}")
            print(f"  α: {val_metrics.alpha:.3f}")
            print(f"  Weight std: {val_metrics.weight_std_mean:.4f}")
            print(f"  ρ(EU, AU): {val_metrics.rho_eu_au:.3f} (target: ~0)")
            print(f"  ρ(EU, Error): {val_metrics.rho_eu_error:.3f} (target: >0.2)")
            print(f"  ρ(σ_var, σ_err): {val_metrics.rho_var_err:.3f}")

            # Save best model
            if val_metrics.accuracy > self.best_val_acc:
                self.best_val_acc = val_metrics.accuracy
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ New best model saved!")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_metrics)

        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print("=" * 80)

        return {
            'history': history,
            'best_metrics': best_metrics.to_dict() if best_metrics else None
        }

    def save_checkpoint(self, epoch: int, metrics: UncertaintyMetrics, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': metrics.to_dict()
        }

        if is_best:
            path = self.save_dir / "best_model.pt"
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_best_model(self):
        """Load the best model from checkpoint."""
        checkpoint_path = self.save_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
            return checkpoint['metrics']
        else:
            print("⚠ No checkpoint found")
            return None


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function."""

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading CEBaB dataset...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader, val_loader, test_loader, tokenizer, metadata = get_cebab_dataloaders(
        tokenizer=tokenizer,
        batch_size=8,
        max_length=256,
        num_workers=0
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {metadata['train_size']}")
    print(f"  Val: {metadata['val_size']}")
    print(f"  Test: {metadata['test_size']}")
    print(f"  Concepts: {metadata['concept_names']}")
    print(f"  Classes: {metadata['num_classes']}")

    # Create model
    print("\nCreating Gradient-Separated VCBM model...")
    config = GradSeparatedConfig(
        encoder_name='distilbert-base-uncased',
        freeze_encoder=True,
        num_concepts=4,
        concept_names=['food', 'service', 'ambiance', 'noise'],
        concept_classes=3,
        num_classes=5,

        # Variational
        covariance_family=CovarianceFamily.MEAN_FIELD,
        prior_std=1.0,
        num_mc_samples=10,
        min_std=0.05,

        # Loss weights
        kl_weight=1e-3,
        concept_weight=2.0,
        aleatoric_weight=0.2,
        error_pred_weight=1.0,
        alignment_weight=0.5,

        # Scheduling
        error_warmup_epochs=3,
        alpha_start=1.0,
        alpha_end=0.3,
        alpha_warmup_epochs=10,

        # Architecture
        use_orthogonal_projection=True
    )

    print(f"  Config:")
    print(f"    MC samples: {config.num_mc_samples}")
    print(f"    KL weight: {config.kl_weight}")
    print(f"    Error warmup: {config.error_warmup_epochs} epochs")
    print(f"    Alpha schedule: {config.alpha_start} → {config.alpha_end} over {config.alpha_warmup_epochs} epochs")

    model = GradSeparatedCredalCBM(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created: {total_params:,} total, {trainable_params:,} trainable")

    # Create trainer
    trainer = GradSeparatedCBMTrainer(
        model=model,
        config=config,
        save_dir="./checkpoints/grad_separated_cebab",
        device=device
    )

    # Train
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        lr=2e-5
    )

    # Save results
    results_path = trainer.save_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = main()
