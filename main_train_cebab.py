"""
Main CEBaB Training Script with VCBM and Comprehensive Metrics
==============================================================

Complete training pipeline for Variational Credal CBM on CEBaB dataset.
Integrates torch-uncertainty library for calibration and selective classification.

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

from VCBM import VariationalCredalCBM, VariationalCredalConfig, CovarianceFamily
from load_cebab_direct import get_cebab_dataloaders

# torch-uncertainty imports
try:
    from torch_uncertainty.metrics.classification import (
        CalibrationError,
        BrierScore,
        AURC,
        AUGRC,
        CategoricalNLL,
    )
    TORCH_UNCERTAINTY_AVAILABLE = True
except ImportError:
    print("⚠ torch-uncertainty not available. Installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'torch-uncertainty'], check=True)
    from torch_uncertainty.metrics.classification import (
        CalibrationError,
        BrierScore,
        AURC,
        AUGRC,
        CategoricalNLL,
    )
    TORCH_UNCERTAINTY_AVAILABLE = True
    print("✓ torch-uncertainty installed successfully")


# ============================================================================
# UNCERTAINTY METRICS CONTAINER
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

        # Correlations (KEY METRICS!)
        self.rho_eu_au = 0.0           # Should be LOW (~0)
        self.p_eu_au = 1.0
        self.rho_eu_error = 0.0        # Should be HIGH (>0.25)
        self.p_eu_error = 1.0

        # Calibration metrics (torch-uncertainty)
        self.ece = 0.0
        self.brier = 0.0
        self.nll = 0.0
        self.aurc = 0.0
        self.augrc = 0.0

        # Quadrant analysis (actionability)
        self.trust_accuracy = 0.0      # Low EU, Low AU
        self.trust_coverage = 0.0
        self.data_accuracy = 0.0       # High EU, Low AU
        self.data_coverage = 0.0
        self.review_accuracy = 0.0     # Low EU, High AU
        self.review_coverage = 0.0
        self.abstain_accuracy = 0.0    # High EU, High AU
        self.abstain_coverage = 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        def convert_value(v):
            """Convert numpy/python types to JSON-serializable types."""
            if hasattr(v, 'item'):  # numpy scalar
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
            'rho_eu_au': convert_value(self.rho_eu_au),
            'p_eu_au': convert_value(self.p_eu_au),
            'rho_eu_error': convert_value(self.rho_eu_error),
            'p_eu_error': convert_value(self.p_eu_error),
            'ece': convert_value(self.ece),
            'brier': convert_value(self.brier),
            'nll': convert_value(self.nll),
            'aurc': convert_value(self.aurc),
            'augrc': convert_value(self.augrc),
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
# VARIATIONAL CREDAL CBM TRAINER
# ============================================================================

class CredalCBMTrainer:
    """
    Trainer for Variational Credal CBM with comprehensive uncertainty metrics.

    Features:
    - Training loop with concept bottleneck supervision
    - Comprehensive evaluation with torch-uncertainty metrics
    - Quadrant analysis for actionable uncertainty
    - Model checkpointing
    """

    def __init__(
        self,
        model: VariationalCredalCBM,
        config: VariationalCredalConfig,
        device: str = "auto",
        save_dir: str = "./checkpoints/cebab",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else device
        )
        self.model.to(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_acc = 0.0
        self.current_epoch = 0

    def get_kl_weight(self, epoch: int, total_epochs: int) -> float:
        """
        Cyclical KL annealing to prevent posterior collapse.

        Starts at 0, ramps to target, helps model learn useful representations
        before regularization kicks in.

        Args:
            epoch: Current epoch
            total_epochs: Total number of training epochs

        Returns:
            KL weight for this epoch
        """
        warmup_epochs = min(5, total_epochs // 3)
        if epoch <= warmup_epochs:
            # Linear warmup
            return self.config.kl_weight * (epoch / warmup_epochs)
        else:
            return self.config.kl_weight

    def train_epoch(
        self,
        train_loader,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Single training epoch with regularization warmup"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Compute regularization warmup factor
        warmup_epochs = 5
        if self.current_epoch <= warmup_epochs:
            reg_factor = self.current_epoch / warmup_epochs
        else:
            reg_factor = 1.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')

            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)

            optimizer.zero_grad()

            # Forward with reg_factor
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_labels=concept_labels,
                reg_factor=reg_factor  # NEW: Warmup regularization
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

            pbar.set_postfix({'loss': loss.item(), 'reg': f'{reg_factor:.2f}'})

            # Clear GPU cache periodically
            if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': (all_preds == all_labels).mean()
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader,
        compute_quadrants: bool = True,
    ) -> UncertaintyMetrics:
        """
        Full evaluation with comprehensive uncertainty metrics.

        Returns:
            UncertaintyMetrics object with all computed metrics
        """
        self.model.eval()

        # Collectors
        all_preds = []
        all_labels = []
        all_concept_probs = []
        all_concept_labels = []
        all_epistemic = []
        all_aleatoric = []
        all_probs = []
        total_loss = 0.0

        # Initialize torch-uncertainty metrics (move to device)
        # NOTE: torch-uncertainty metrics have issues on MPS, so we keep them on CPU
        if TORCH_UNCERTAINTY_AVAILABLE:
            metric_device = 'cpu' if self.device.type == 'mps' else self.device
            ece_metric = CalibrationError(
                task='multiclass',
                num_classes=self.config.num_classes,
                num_bins=15
            ).to(metric_device)
            brier_metric = BrierScore(num_classes=self.config.num_classes).to(metric_device)
            nll_metric = CategoricalNLL().to(metric_device)
            aurc_metric = AURC().to(metric_device)
            augrc_metric = AUGRC().to(metric_device)
        else:
            ece_metric = None

        # Evaluation loop
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')

            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)

            # Forward
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
            all_aleatoric.append(outputs['aleatoric'].cpu())
            all_probs.append(outputs['probs'].cpu())

            # Update torch-uncertainty metrics
            if TORCH_UNCERTAINTY_AVAILABLE:
                probs = outputs['probs']
                preds = outputs['predictions']

                # Move tensors to CPU for MPS compatibility
                if self.device.type == 'mps':
                    probs_cpu = probs.cpu()
                    labels_cpu = labels.cpu()
                    preds_cpu = preds.cpu()
                    ece_metric.update(probs_cpu, labels_cpu)
                    brier_metric.update(probs_cpu, labels_cpu)
                    nll_metric.update(probs_cpu, labels_cpu)

                    # AURC and AUGRC: positional args (confidence, errors)
                    confidence = probs_cpu.max(dim=-1).values
                    errors = (preds_cpu != labels_cpu).long()
                    neg_conf = (-confidence).float()
                    aurc_metric.update(neg_conf, errors)
                    augrc_metric.update(neg_conf, errors)
                else:
                    ece_metric.update(probs, labels)
                    brier_metric.update(probs, labels)
                    nll_metric.update(probs, labels)

                    # AURC and AUGRC: positional args (confidence, errors)
                    confidence = probs.max(dim=-1).values
                    errors = (preds != labels).long()
                    neg_conf = (-confidence).float()
                    aurc_metric.update(neg_conf, errors)
                    augrc_metric.update(neg_conf, errors)

            # Clear GPU cache periodically
            if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_concept_probs = np.array(all_concept_probs)
        if all_concept_labels:
            all_concept_labels = np.array(all_concept_labels)
        else:
            all_concept_labels = None
        all_epistemic = torch.cat(all_epistemic).numpy()
        all_aleatoric = torch.cat(all_aleatoric).numpy()
        all_probs = torch.cat(all_probs).numpy()

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
        else:
            for name in concept_names:
                concept_accs[name] = 0.0

        # Uncertainty statistics
        epistemic_sample = all_epistemic.mean(axis=-1)
        aleatoric_sample = all_aleatoric.mean(axis=-1)

        # Correlations
        if epistemic_sample.std() > 0 and aleatoric_sample.std() > 0:
            rho_eu_au, p_eu_au = stats.spearmanr(epistemic_sample, aleatoric_sample)
        else:
            rho_eu_au, p_eu_au = 0.0, 1.0

        errors = (all_preds != all_labels).astype(float)
        if errors.std() > 0 and epistemic_sample.std() > 0:
            rho_eu_error, p_eu_error = stats.spearmanr(epistemic_sample, errors)
        else:
            rho_eu_error, p_eu_error = 0.0, 1.0

        # Quadrant analysis
        if compute_quadrants:
            eu_thresh = np.median(epistemic_sample)
            au_thresh = np.median(aleatoric_sample)

            trust_mask = (epistemic_sample <= eu_thresh) & (aleatoric_sample <= au_thresh)
            data_mask = (epistemic_sample > eu_thresh) & (aleatoric_sample <= au_thresh)
            review_mask = (epistemic_sample <= eu_thresh) & (aleatoric_sample > au_thresh)
            abstain_mask = (epistemic_sample > eu_thresh) & (aleatoric_sample > au_thresh)

            def accuracy_in_mask(mask):
                if mask.sum() == 0:
                    return float('nan')
                return 1.0 - errors[mask].mean()

            trust_acc = accuracy_in_mask(trust_mask)
            data_acc = accuracy_in_mask(data_mask)
            review_acc = accuracy_in_mask(review_mask)
            abstain_acc = accuracy_in_mask(abstain_mask)
        else:
            trust_acc = data_acc = review_acc = abstain_acc = float('nan')
            trust_mask = data_mask = review_mask = abstain_mask = np.zeros(len(all_preds), dtype=bool)

        # Create metrics object
        metrics = UncertaintyMetrics()
        metrics.accuracy = task_acc
        metrics.loss = total_loss / len(val_loader)
        metrics.concept_accs = concept_accs
        metrics.mean_epistemic = epistemic_sample.mean()
        metrics.std_epistemic = epistemic_sample.std()
        metrics.mean_aleatoric = aleatoric_sample.mean()
        metrics.std_aleatoric = aleatoric_sample.std()
        metrics.rho_eu_au = rho_eu_au
        metrics.p_eu_au = p_eu_au
        metrics.rho_eu_error = rho_eu_error
        metrics.p_eu_error = p_eu_error

        # torch-uncertainty metrics
        if TORCH_UNCERTAINTY_AVAILABLE and ece_metric is not None:
            metrics.ece = ece_metric.compute().item()
            metrics.brier = brier_metric.compute().item()
            metrics.nll = nll_metric.compute().item()
            metrics.aurc = aurc_metric.compute().item()
            metrics.augrc = augrc_metric.compute().item()
        else:
            metrics.ece = float('nan')
            metrics.brier = float('nan')
            metrics.nll = float('nan')
            metrics.aurc = float('nan')
            metrics.augrc = float('nan')

        # Quadrant metrics
        metrics.trust_accuracy = trust_acc
        metrics.trust_coverage = trust_mask.mean()
        metrics.data_accuracy = data_acc
        metrics.data_coverage = data_mask.mean()
        metrics.review_accuracy = review_acc
        metrics.review_coverage = review_mask.mean()
        metrics.abstain_accuracy = abstain_acc
        metrics.abstain_coverage = abstain_mask.mean()

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
        """
        Full training loop.

        Returns:
            Dictionary with training history and best results
        """
        # Optimizer and scheduler
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

        # Training history
        history = []
        best_metrics = None

        print(f"\n{'='*80}")
        print(f"Starting Training for {num_epochs} Epochs")
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
            val_metrics = self.evaluate(val_loader, compute_quadrants=True)

            # Print results
            print(f"\n📊 Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics.loss:.4f}, Val Acc: {val_metrics.accuracy:.4f}")

            print(f"\n🎯 Concept Accuracy:")
            for name, acc in val_metrics.concept_accs.items():
                print(f"  {name.capitalize()}: {acc:.4f}")

            print(f"\n🎯 Uncertainty Decomposition:")
            print(f"  Mean EU: {val_metrics.mean_epistemic:.4f} ± {val_metrics.std_epistemic:.4f}")
            print(f"  Mean AU: {val_metrics.mean_aleatoric:.4f} ± {val_metrics.std_aleatoric:.4f}")
            print(f"  ρ(EU, AU): {val_metrics.rho_eu_au:.3f} (p={val_metrics.p_eu_au:.3f}) [target: < 0.1]")
            print(f"  ρ(EU, Error): {val_metrics.rho_eu_error:.3f} (p={val_metrics.p_eu_error:.3f}) [target: > 0.25]")

            if TORCH_UNCERTAINTY_AVAILABLE:
                print(f"\n📈 Calibration & Selective:")
                print(f"  ECE: {val_metrics.ece:.4f}")
                print(f"  Brier: {val_metrics.brier:.4f}")
                print(f"  NLL: {val_metrics.nll:.4f}")
                print(f"  AURC: {val_metrics.aurc:.4f}")
                print(f"  AUGRC: {val_metrics.augrc:.4f}")

            print(f"\n🎯 Quadrant Analysis:")
            print(f"  Trust (Low EU, Low AU):  {val_metrics.trust_accuracy:.4f} acc, {val_metrics.trust_coverage:.1%} cov")
            print(f"  Data (High EU, Low AU):   {val_metrics.data_accuracy:.4f} acc, {val_metrics.data_coverage:.1%} cov")
            print(f"  Review (Low EU, High AU): {val_metrics.review_accuracy:.4f} acc, {val_metrics.review_coverage:.1%} cov")
            print(f"  Abstain (High EU, High AU): {val_metrics.abstain_accuracy:.4f} acc, {val_metrics.abstain_coverage:.1%} cov")

            # Save best model
            if val_metrics.accuracy > self.best_val_acc:
                self.best_val_acc = val_metrics.accuracy
                best_metrics = val_metrics

                # Save checkpoint
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
                print(f"  Checkpoint saved: {checkpoint_path}")

            # Store history
            history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics.to_dict()
            })

        # Save training history
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
        """Load the best model from checkpoint."""
        checkpoint_path = self.save_dir / "best_model.pt"
        if checkpoint_path.exists():
            # FIX: Add weights_only=False for checkpoints containing numpy arrays
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
    """Main training function"""

    print("\n" + "="*80)
    print("Variational Credal CBM Training on CEBaB")
    print("="*80)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading CEBaB dataset...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader, val_loader, test_loader, tokenizer, metadata = get_cebab_dataloaders(
        tokenizer=tokenizer,
        batch_size=8,  # Reduced from 16 to save memory
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
    print("\nCreating VCBM model...")
    config = VariationalCredalConfig(
        encoder_name='distilbert-base-uncased',
        freeze_encoder=True,
        num_concepts=4,
        concept_names=['food', 'service', 'ambiance', 'noise'],
        concept_classes=3,
        num_classes=5,
        covariance_family=CovarianceFamily.MEAN_FIELD,

        kl_weight=1e-5,  # Very small - let model learn first
        concept_weight=0.0,
        aleatoric_weight=0.2,
        supervision_weight=1.0,
        use_orthogonal_projection=True,
        use_temperature_scaling=True,
        use_aleatoric_prior=True,
        pooling_strategy="cls",
        num_mc_samples=10
    )

    print(f"  Config: MC samples={config.num_mc_samples}, KL weight={config.kl_weight}")
    model = VariationalCredalCBM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total, {trainable_params:,} trainable")

    # Create trainer
    trainer = CredalCBMTrainer(model, config, device=device)

    # Train
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,  # Quick test - reduced from 10
        lr=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_every=1
    )

    # Load best model and evaluate on test set
    print("\n" + "="*80)
    print("Evaluating Best Model on Test Set")
    print("="*80)

    trainer.load_best_model()
    test_metrics = trainer.evaluate(test_loader, compute_quadrants=True)

    print(f"\n📊 Test Results:")
    print(f"  Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"  Test Loss: {test_metrics.loss:.4f}")

    print(f"\n🎯 Concept Accuracy:")
    for name, acc in test_metrics.concept_accs.items():
        print(f"  {name.capitalize()}: {acc:.4f}")

    print(f"\n🎯 Uncertainty Decomposition:")
    print(f"  ρ(EU, AU): {test_metrics.rho_eu_au:.3f} (target: < 0.1) {'✓' if abs(test_metrics.rho_eu_au) < 0.1 else '✗'}")
    print(f"  ρ(EU, Error): {test_metrics.rho_eu_error:.3f} (target: > 0.25) {'✓' if test_metrics.rho_eu_error > 0.25 else '✗'}")

    print(f"\n🎯 Quadrant Analysis:")
    print(f"  Trust (Low EU, Low AU):  {test_metrics.trust_accuracy:.4f} acc, {test_metrics.trust_coverage:.1%} cov")
    print(f"  Data (High EU, Low AU):   {test_metrics.data_accuracy:.4f} acc, {test_metrics.data_coverage:.1%} cov")
    print(f"  Review (Low EU, High AU): {test_metrics.review_accuracy:.4f} acc, {test_metrics.review_coverage:.1%} cov")
    print(f"  Abstain (High EU, High AU): {test_metrics.abstain_accuracy:.4f} acc, {test_metrics.abstain_coverage:.1%} cov")

    # Save final results
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
