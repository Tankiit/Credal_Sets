"""
True Credal CBM Training on CEBaB
==================================

Training script for the True Credal CBM with structural separation.

Key features:
- Three-way orthogonal projection (h_concept, h_epi, h_ale)
- Credal set representation (μ, Σ_epi)
- EU derived from credal geometry: log(Σ_epi)
- AU supervised by annotator entropy

This escapes the Tomov et al. impossibility by deriving EU from
credal set geometry rather than from the predictive distribution.

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.optim as optim
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from scipy import stats
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from VCBM import TrueCredalCBM, TrueCredalConfig
from load_cebab_direct import get_cebab_dataloaders


# ============================================================================
# UNCERTAINTY METRICS
# ============================================================================

class UncertaintyMetrics:
    """Container for uncertainty metrics."""

    def __init__(self):
        self.accuracy = 0.0
        self.loss = 0.0
        self.concept_accs = {}

        # Credal set statistics
        self.mean_sigma_epi = 0.0
        self.std_sigma_epi = 0.0
        self.mean_eu = 0.0
        self.std_eu = 0.0

        # Aleatoric
        self.mean_aleatoric = 0.0
        self.std_aleatoric = 0.0

        # Correlations
        self.rho_eu_au = 0.0
        self.p_eu_au = 0.0
        self.rho_eu_error = 0.0
        self.p_eu_error = 0.0
        self.rho_ale_entropy = 0.0
        self.p_ale_entropy = 0.0

    def to_dict(self):
        return {
            'accuracy': float(self.accuracy),
            'loss': float(self.loss),
            'concept_accs': {k: float(v) for k, v in self.concept_accs.items()},
            'mean_sigma_epi': float(self.mean_sigma_epi),
            'std_sigma_epi': float(self.std_sigma_epi),
            'mean_eu': float(self.mean_eu),
            'std_eu': float(self.std_eu),
            'mean_aleatoric': float(self.mean_aleatoric),
            'std_aleatoric': float(self.std_aleatoric),
            'rho_eu_au': float(self.rho_eu_au),
            'p_eu_au': float(self.p_eu_au),
            'rho_eu_error': float(self.rho_eu_error),
            'p_eu_error': float(self.p_eu_error),
            'rho_ale_entropy': float(self.rho_ale_entropy),
            'p_ale_entropy': float(self.p_ale_entropy),
        }


# ============================================================================
# TRAINER
# ============================================================================

class TrueCredalCBMTrainer:
    """Trainer for True Credal CBM."""

    def __init__(
        self,
        model: TrueCredalCBM,
        config: TrueCredalConfig,
        device: str = "auto",
        save_dir: str = "./checkpoints/true_credal_cebab",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else device
        )
        self.model.to(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.current_epoch = 0

    def train_epoch(
        self,
        train_loader,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')

            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)

            annotator_entropy = batch.get('annotator_entropy')
            if annotator_entropy is not None:
                annotator_entropy = annotator_entropy.to(self.device)

            optimizer.zero_grad()

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_labels=concept_labels,
                annotator_entropy=annotator_entropy
            )

            loss = outputs['loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': np.array(all_preds == all_labels).mean()
        }

    @torch.no_grad()
    def evaluate(self, val_loader) -> UncertaintyMetrics:
        """Full evaluation with uncertainty metrics."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_concept_probs = []
        all_concept_labels = []
        all_sigma_epi = []
        all_eu = []
        all_aleatoric = []
        all_entropies = []
        total_loss = 0.0

        for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')

            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)

            annotator_entropy = batch.get('annotator_entropy')
            if annotator_entropy is not None:
                annotator_entropy = annotator_entropy.to(self.device)

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_labels=concept_labels,
                annotator_entropy=annotator_entropy
            )

            total_loss += outputs['loss'].item()

            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_probs.extend(outputs['concept_probs'].cpu().numpy())

            if concept_labels is not None:
                all_concept_labels.extend(concept_labels.cpu().numpy())

            all_sigma_epi.append(outputs['sigma_epi'].cpu())
            all_eu.append(outputs['epistemic'].cpu())
            all_aleatoric.append(outputs['aleatoric'].cpu())

            if annotator_entropy is not None:
                all_entropies.append(annotator_entropy.cpu())

        # Concatenate
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_concept_probs = np.array(all_concept_probs)
        all_sigma_epi = torch.cat(all_sigma_epi).numpy()
        all_eu = torch.cat(all_eu).numpy()
        all_aleatoric = torch.cat(all_aleatoric).numpy()

        if all_concept_labels:
            all_concept_labels = np.array(all_concept_labels)
        else:
            all_concept_labels = None

        if all_entropies:
            all_entropies = torch.cat(all_entropies).numpy()

        # Task accuracy
        task_acc = (all_preds == all_labels).mean()

        # Concept accuracy
        concept_accs = {}
        concept_names = ['food', 'service', 'ambiance', 'noise']

        if all_concept_labels is not None:
            for k in range(4):
                known_mask = (all_concept_labels[:, k] != 1)
                if known_mask.sum() > 0:
                    c_labels_binary = (all_concept_labels[known_mask, k] / 2.0 > 0.5).astype(int)
                    c_preds_binary = (all_concept_probs[known_mask, k] > 0.5).astype(int)
                    c_acc = (c_preds_binary == c_labels_binary).mean()
                    concept_accs[concept_names[k]] = c_acc
                else:
                    concept_accs[concept_names[k]] = 0.0

        # Credal set statistics
        eu_sample = all_eu.mean(axis=-1)
        aleatoric_sample = all_aleatoric.mean(axis=-1)

        # Correlations
        if eu_sample.std() > 0 and aleatoric_sample.std() > 0:
            rho_eu_au, p_eu_au = stats.spearmanr(eu_sample, aleatoric_sample)
        else:
            rho_eu_au, p_eu_au = 0.0, 1.0

        errors = (all_preds != all_labels).astype(float)
        if errors.std() > 0 and eu_sample.std() > 0:
            rho_eu_error, p_eu_error = stats.spearmanr(eu_sample, errors)
        else:
            rho_eu_error, p_eu_error = 0.0, 1.0

        # Aleatoric vs entropy correlation
        if all_entropies is not None and all_entropies.size > 0:
            entropy_sample = all_entropies.mean(axis=-1)
            if entropy_sample.std() > 0 and aleatoric_sample.std() > 0:
                rho_ale_entropy, p_ale_entropy = stats.spearmanr(aleatoric_sample, entropy_sample)
            else:
                rho_ale_entropy, p_ale_entropy = 0.0, 1.0
        else:
            rho_ale_entropy, p_ale_entropy = 0.0, 1.0

        # Create metrics
        metrics = UncertaintyMetrics()
        metrics.accuracy = task_acc
        metrics.loss = total_loss / len(val_loader)
        metrics.concept_accs = concept_accs
        metrics.mean_sigma_epi = all_sigma_epi.mean()
        metrics.std_sigma_epi = all_sigma_epi.std()
        metrics.mean_eu = eu_sample.mean()
        metrics.std_eu = eu_sample.std()
        metrics.mean_aleatoric = aleatoric_sample.mean()
        metrics.std_aleatoric = aleatoric_sample.std()
        metrics.rho_eu_au = rho_eu_au
        metrics.p_eu_au = p_eu_au
        metrics.rho_eu_error = rho_eu_error
        metrics.p_eu_error = p_eu_error
        metrics.rho_ale_entropy = rho_ale_entropy
        metrics.p_ale_entropy = p_ale_entropy

        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        save_every: int = 5,
    ) -> Dict:
        """Full training loop."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        history = []
        best_metrics = None

        print(f"\n{'='*80}")
        print(f"Training True Credal CBM for {num_epochs} Epochs")
        print(f"{'='*80}")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, scheduler
            )

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Print results
            print(f"\n📊 Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics.loss:.4f}, Val Acc: {val_metrics.accuracy:.4f}")

            print(f"\n🎯 Concept Accuracy:")
            for name, acc in val_metrics.concept_accs.items():
                print(f"  {name.capitalize()}: {acc:.4f}")

            print(f"\n🎯 Credal Set Statistics:")
            print(f"  Mean Σ_epi: {val_metrics.mean_sigma_epi:.4f} ± {val_metrics.std_sigma_epi:.4f}")
            print(f"  Mean EU (log Σ_epi): {val_metrics.mean_eu:.4f} ± {val_metrics.std_eu:.4f}")
            print(f"  Mean AU: {val_metrics.mean_aleatoric:.4f} ± {val_metrics.std_aleatoric:.4f}")

            print(f"\n🎯 Uncertainty Decomposition:")
            print(f"  ρ(EU, AU): {val_metrics.rho_eu_au:.3f} (p={val_metrics.p_eu_au:.3f}) [target: < 0.3]")
            print(f"  ρ(EU, Error): {val_metrics.rho_eu_error:.3f} (p={val_metrics.p_eu_error:.3f}) [target: > 0.2]")
            print(f"  ρ(AU, Entropy): {val_metrics.rho_ale_entropy:.3f} (p={val_metrics.p_ale_entropy:.3f}) [target: > 0.3]")

            # Save best model
            if val_metrics.accuracy > self.best_val_acc:
                self.best_val_acc = val_metrics.accuracy
                best_metrics = val_metrics

                checkpoint_path = self.save_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics.to_dict()
                }, checkpoint_path)
                print(f"\n  ✓ New best model saved! (Val Acc: {self.best_val_acc:.4f})")

            # Periodic checkpoint
            if epoch % save_every == 0:
                checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics.to_dict()
                }, checkpoint_path)

            history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics.to_dict()
            })

        # Save history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=float)

        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*80}")

        return {
            'history': history,
            'best_metrics': best_metrics.to_dict() if best_metrics else None
        }

    def load_best_model(self):
        """Load best model from checkpoint."""
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
# MAIN
# ============================================================================

def main():
    """Main training function."""
    print("\n" + "="*80)
    print("True Credal CBM Training on CEBaB")
    print("="*80)

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
    print("\nCreating True Credal CBM...")
    config = TrueCredalConfig(
        encoder_name='distilbert-base-uncased',
        freeze_encoder=True,
        num_concepts=4,
        concept_names=['food', 'service', 'ambiance', 'noise'],
        num_classes=5,

        # Credal set parameters
        num_mc_samples=10,
        min_sigma=1e-4,
        max_sigma=2.0,
        prior_sigma=1.0,

        # Loss weights
        concept_weight=2.0,
        kl_weight=0.1,
        aleatoric_weight=1.0,
        orth_weight=0.001,

        # Aleatoric prior
        aleatoric_prior=0.3,

        # Architecture
        projection_dim=256,
        hidden_dim=128,
    )

    model = TrueCredalCBM(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total, {trainable_params:,} trainable")

    # Create trainer
    trainer = TrueCredalCBMTrainer(model, config, device=device)

    # Train
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        lr=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_every=5
    )

    # Load best and test
    print("\n" + "="*80)
    print("Evaluating Best Model on Test Set")
    print("="*80)

    trainer.load_best_model()
    test_metrics = trainer.evaluate(test_loader)

    print(f"\n📊 Test Results:")
    print(f"  Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"  Test Loss: {test_metrics.loss:.4f}")

    print(f"\n🎯 Concept Accuracy:")
    for name, acc in test_metrics.concept_accs.items():
        print(f"  {name.capitalize()}: {acc:.4f}")

    print(f"\n🎯 Credal Set Statistics:")
    print(f"  Mean Σ_epi: {test_metrics.mean_sigma_epi:.4f}")
    print(f"  Mean EU: {test_metrics.mean_eu:.4f}")
    print(f"  Mean AU: {test_metrics.mean_aleatoric:.4f}")

    print(f"\n🎯 Uncertainty Decomposition:")
    print(f"  ρ(EU, AU): {test_metrics.rho_eu_au:.3f} [target: < 0.3]")
    print(f"  ρ(EU, Error): {test_metrics.rho_eu_error:.3f} [target: > 0.2]")
    print(f"  ρ(AU, Entropy): {test_metrics.rho_ale_entropy:.3f} [target: > 0.3]")

    # Save results
    results_path = trainer.save_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_metrics.to_dict(), f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")
    print("\n" + "="*80)
    print("All Done!")
    print("="*80)

    return test_metrics.to_dict()


if __name__ == "__main__":
    results = main()
