"""
HateXplain Training Script with Enhanced VCBM
=============================================

This script implements:
- Enhanced Variational Credal CBM training
- HateXplain dataset with enhanced concept encoding
- Training and validation loops
- Comprehensive metrics and uncertainty evaluation

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from tqdm import tqdm
import wandb

from VCBM import (
    VariationalCredalCBM,
    VariationalCredalConfig,
    CovarianceFamily,
    diagnose_separation,
    diagnose_concept_learning
)
from stochastic_cbm_dataloader import (
    load_dataset_splits,
    DatasetConfig,
    get_recommended_config
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class HateXplainConfig:
    """Configuration for HateXplain experiments"""

    # Dataset
    dataset_name: str = "hatexplain"
    max_length: int = 128
    batch_size: int = 16
    num_workers: int = 4

    # Model
    encoder_name: str = "distilbert-base-uncased"
    num_concepts: int = 2  # has_target, is_offensive
    num_classes: int = 3   # hatespeech, normal, offensive
    concept_classes: int = 3  # ternary concepts

    # Training
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # VCBM specific
    covariance_family: CovarianceFamily = CovarianceFamily.MEAN_FIELD
    kl_weight: float = 1e-5
    concept_weight: float = 0.5
    aleatoric_weight: float = 0.2
    supervision_weight: float = 2.0

    # Enhancements
    use_orthogonal_projection: bool = True
    use_temperature_scaling: bool = True
    use_aleatoric_prior: bool = True
    pooling_strategy: str = "cls"

    # MC samples
    num_mc_samples: int = 20

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    use_wandb: bool = True
    project_name: str = "vcbm-hatexplain"
    run_name: Optional[str] = None

    # Checkpointing
    save_dir: str = "./checkpoints/hatexplain"
    save_every: int = 5


# ============================================================================
# METRICS
# ============================================================================

def compute_uncertainty_metrics(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    errors: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    concept_probs: np.ndarray,
    concept_labels: np.ndarray,
    unknown_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive uncertainty metrics.

    Args:
        epistemic: [N, K] per-concept epistemic uncertainty
        aleatoric: [N, K] per-concept aleatoric uncertainty
        errors: [N] binary error indicator
        predictions: [N] predicted labels
        labels: [N] true labels
        concept_probs: [N, K] concept probabilities
        concept_labels: [N, K] concept labels (ternary)
        unknown_mask: [N, K] unknown concept mask

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Aggregate to sample level
    epi_sample = epistemic.mean(axis=-1) if epistemic.ndim > 1 else epistemic
    ale_sample = aleatoric.mean(axis=-1) if aleatoric.ndim > 1 else aleatoric
    unk_sample = unknown_mask.mean(axis=-1) if unknown_mask.ndim > 1 else unknown_mask

    # ========================================================================
    # Task-level metrics
    # ========================================================================
    # Accuracy
    metrics['accuracy'] = (predictions == labels).mean()

    # Per-class accuracy
    for c in range(3):  # hatespeech, normal, offensive
        mask = (labels == c)
        if mask.sum() > 0:
            metrics[f'acc_class_{c}'] = (predictions[mask] == labels[mask]).mean()

    # ========================================================================
    # Correlation metrics (MAIN PAPER METRICS)
    # ========================================================================
    # 1. Epistemic-Error correlation (should be POSITIVE)
    if epi_sample.std() > 0 and errors.std() > 0:
        rho, p = stats.spearmanr(epi_sample, errors)
        metrics['rho_epi_err'] = rho
        metrics['p_epi_err'] = p

    # 2. Aleatoric-Unknown correlation (should be POSITIVE)
    if ale_sample.std() > 0 and unk_sample.std() > 0:
        rho, p = stats.spearmanr(ale_sample, unk_sample)
        metrics['rho_ale_unk'] = rho
        metrics['p_ale_unk'] = p

    # 3. Epistemic-Aleatoric correlation (should be LOW - main metric!)
    if epi_sample.std() > 0 and ale_sample.std() > 0:
        rho, p = stats.spearmanr(epi_sample, ale_sample)
        metrics['rho_epi_ale'] = rho
        metrics['p_epi_ale'] = p
        metrics['separation_quality'] = 1 - abs(rho)  # Higher is better

    # ========================================================================
    # Concept-level metrics
    # ========================================================================
    num_concepts = concept_probs.shape[1] if concept_probs.ndim > 1 else 1

    for k in range(num_concepts):
        # Convert to binary
        c_labels_k = (concept_labels[:, k] / 2.0) if concept_labels.ndim > 1 else (concept_labels / 2.0)
        c_preds_k = (concept_probs[:, k] > 0.5).astype(int) if concept_probs.ndim > 1 else (concept_probs > 0.5).astype(int)

        # Known mask
        known_mask_k = (concept_labels[:, k] != 1) if concept_labels.ndim > 1 else (concept_labels != 1)

        if known_mask_k.sum() > 0:
            # Concept accuracy
            c_acc = (c_preds_k[known_mask_k] == (c_labels_k[known_mask_k] > 0.5).astype(int)).mean()
            metrics[f'concept_{k}_acc'] = c_acc

            # Epistemic-concept_error correlation
            c_errors = (c_preds_k[known_mask_k] != (c_labels_k[known_mask_k] > 0.5).astype(int)).astype(float)
            if c_errors.std() > 0 and epistemic[known_mask_k, k].std() > 0:
                rho, p = stats.spearmanr(epistemic[known_mask_k, k], c_errors)
                metrics[f'concept_{k}_rho_epi_err'] = rho

    # ========================================================================
    # Uncertainty statistics
    # ========================================================================
    metrics['mean_epistemic'] = epi_sample.mean()
    metrics['mean_aleatoric'] = ale_sample.mean()
    metrics['std_epistemic'] = epi_sample.std()
    metrics['std_aleatoric'] = ale_sample.std()

    # Per-concept statistics
    for k in range(num_concepts):
        metrics[f'epistemic_concept_{k}'] = epistemic[:, k].mean() if epistemic.ndim > 1 else epi_sample.mean()
        metrics[f'aleatoric_concept_{k}'] = aleatoric[:, k].mean() if aleatoric.ndim > 1 else ale_sample.mean()

    return metrics


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Expected Calibration Error (ECE)"""
    # Get predictions and confidences
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (preds == labels).astype(float)

    # Sort by confidence
    indices = np.argsort(confidences)
    confidences = confidences[indices]
    accuracies = accuracies[indices]

    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return {'ece': ece}


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: VariationalCredalCBM,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""

    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_losses = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch.get('concept_labels', None)

        if concept_labels is not None:
            concept_labels = concept_labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            concept_labels=concept_labels
        )

        loss = outputs['loss']
        losses = {k: v.item() if isinstance(v, torch.Tensor) else v
                  for k, v in outputs.items() if 'loss' in k}

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Track
        total_loss += loss.item()
        all_preds.extend(outputs['predictions'].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_losses.append(losses)

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'train_loss': total_loss / len(dataloader),
        'train_accuracy': (all_preds == all_labels).mean()
    }

    # Average all loss components
    for key in all_losses[0].keys():
        metrics[f'train_{key}'] = np.mean([l[key] for l in all_losses])

    return metrics


def validate(
    model: VariationalCredalCBM,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    compute_uncertainty: bool = True
) -> Dict[str, float]:
    """Validation loop with comprehensive metrics"""

    model.eval()

    all_outputs = []
    all_labels = []
    all_concept_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch.get('concept_labels', None)

            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_labels=concept_labels
            )

            total_loss += outputs['loss'].item()

            all_outputs.append({
                'predictions': outputs['predictions'].cpu(),
                'probs': outputs['probs'].cpu(),
                'epistemic': outputs['epistemic'].cpu(),
                'aleatoric': outputs['aleatoric'].cpu(),
                'concept_probs': outputs['concept_probs'].cpu(),
            })

            all_labels.append(labels.cpu())
            if concept_labels is not None:
                all_concept_labels.append(concept_labels.cpu())

    # Concatenate
    labels = torch.cat(all_labels).numpy()
    predictions = torch.cat([o['predictions'] for o in all_outputs]).numpy()
    probs = torch.cat([o['probs'] for o in all_outputs]).numpy()
    epistemic = torch.cat([o['epistemic'] for o in all_outputs]).numpy()
    aleatoric = torch.cat([o['aleatoric'] for o in all_outputs]).numpy()
    concept_probs = torch.cat([o['concept_probs'] for o in all_outputs]).numpy()

    if len(all_concept_labels) > 0:
        concept_labels = torch.cat(all_concept_labels).numpy()
    else:
        concept_labels = None

    # Errors
    errors = (predictions != labels).astype(float)

    # Unknown mask (ternary: 1 = unknown)
    if concept_labels is not None:
        unknown_mask = (concept_labels == 1).astype(float)
    else:
        unknown_mask = np.zeros_like(aleatoric)

    # Base metrics
    metrics = {
        'val_loss': total_loss / len(dataloader),
        'val_accuracy': (predictions == labels).mean()
    }

    # Calibration
    cal_metrics = compute_calibration_metrics(probs, labels)
    metrics.update(cal_metrics)

    # Uncertainty metrics
    if compute_uncertainty:
        uncertainty_metrics = compute_uncertainty_metrics(
            epistemic=epistemic,
            aleatoric=aleatoric,
            errors=errors,
            predictions=predictions,
            labels=labels,
            concept_probs=concept_probs,
            concept_labels=concept_labels if concept_labels is not None else np.array([]),
            unknown_mask=unknown_mask
        )
        metrics.update(uncertainty_metrics)

    return metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_hatexplain(config: HateXplainConfig):
    """Main training function"""

    # ========================================================================
    # Setup
    # ========================================================================
    print("\n" + "="*80)
    print("HateXplain Training with Enhanced VCBM")
    print("="*80)

    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=vars(config)
        )

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\nLoading HateXplain dataset...")

    dataset_config = DatasetConfig(
        label_type="default",
        max_length=config.max_length,
        tokenizer_name=config.encoder_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        dataset_name="hatexplain",
        config=dataset_config
    )

    print(f"\nDataset metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # ========================================================================
    # Create Model
    # ========================================================================
    print("\nCreating model...")

    vcbm_config = VariationalCredalConfig(
        encoder_name=config.encoder_name,
        freeze_encoder=True,
        num_concepts=config.num_concepts,
        concept_names=['has_target', 'is_offensive'],
        concept_classes=config.concept_classes,
        num_classes=config.num_classes,
        covariance_family=config.covariance_family,
        kl_weight=config.kl_weight,
        concept_weight=config.concept_weight,
        aleatoric_weight=config.aleatoric_weight,
        supervision_weight=config.supervision_weight,
        use_orthogonal_projection=config.use_orthogonal_projection,
        use_temperature_scaling=config.use_temperature_scaling,
        use_aleatoric_prior=config.use_aleatoric_prior,
        pooling_strategy=config.pooling_strategy,
        num_mc_samples=config.num_mc_samples
    )

    model = VariationalCredalCBM(vcbm_config).to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # ========================================================================
    # Optimizer and Scheduler
    # ========================================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\nStarting training...")

    best_val_acc = 0.0
    best_rho_epi_ale = float('inf')  # Want LOW correlation
    all_metrics = []

    for epoch in range(1, config.epochs + 1):
        print("\n" + "="*80)
        print(f"Epoch {epoch}/{config.epochs}")
        print("="*80)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, config.device, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, config.device, epoch, compute_uncertainty=True
        )

        # Combine
        epoch_metrics = {**train_metrics, **val_metrics}
        all_metrics.append(epoch_metrics)

        # Print
        print("\nMetrics:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Acc: {train_metrics['train_accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Acc: {val_metrics['val_accuracy']:.4f}")
        print(f"  ρ(EU, AU): {val_metrics.get('rho_epi_ale', 'N/A')}")
        print(f"  ρ(EU, Err): {val_metrics.get('rho_epi_err', 'N/A')}")
        print(f"  ρ(AU, Unk): {val_metrics.get('rho_ale_unk', 'N/A')}")

        # Log to wandb
        if config.use_wandb:
            wandb.log(epoch_metrics, step=epoch)

        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': epoch_metrics
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Track best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            best_path = save_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': epoch_metrics
            }, best_path)
            print(f"  ✓ New best model (acc={best_val_acc:.4f})")

        # Track best separation
        if 'rho_epi_ale' in val_metrics:
            if abs(val_metrics['rho_epi_ale']) < best_rho_epi_ale:
                best_rho_epi_ale = abs(val_metrics['rho_epi_ale'])
                print(f"  ✓ New best separation (ρ={best_rho_epi_ale:.4f})")

    # ========================================================================
    # Final Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    # Load best model
    best_path = save_dir / "best_model.pt"
    checkpoint = torch.load(best_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(
        model, test_loader, config.device, epoch="Test", compute_uncertainty=True
    )

    print("\nTest Results:")
    print(f"  Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"  Test Acc: {test_metrics['val_accuracy']:.4f}")
    print(f"  ECE: {test_metrics.get('ece', 'N/A'):.4f}")
    print(f"  ρ(EU, AU): {test_metrics.get('rho_epi_ale', 'N/A'):.4f}")
    print(f"  ρ(EU, Err): {test_metrics.get('rho_epi_err', 'N/A'):.4f}")
    print(f"  ρ(AU, Unk): {test_metrics.get('rho_ale_unk', 'N/A'):.4f}")

    # Diagnostics
    print("\n" + "="*80)
    print("DIAGNOSTICS")
    print("="*80)

    diagnose_separation(model, val_loader, config.device)
    diagnose_concept_learning(model, val_loader, config.device)

    # Save final results
    results = {
        'config': vars(config),
        'best_val_acc': best_val_acc,
        'best_rho_epi_ale': best_rho_epi_ale,
        'test_metrics': test_metrics,
        'all_epochs': all_metrics
    }

    results_path = save_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to: {results_path}")

    if config.use_wandb:
        wandb.finish()

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = HateXplainConfig()
    config.epochs = 30
    config.batch_size = 16
    config.lr = 1e-4
    config.use_wandb = True
    config.run_name = "enhanced_vcbm_hatexplain"

    # Train
    results = train_hatexplain(config)

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
