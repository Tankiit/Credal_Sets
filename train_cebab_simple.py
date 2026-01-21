"""
Simple CEBaB Training Script with torch-uncertainty Metrics
===========================================================

Quick training script for VCBM on CEBaB dataset with comprehensive metrics.
Integrates torch-uncertainty library for calibration and selective classification.

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from scipy import stats

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


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            concept_labels=concept_labels
        )

        loss = outputs['loss']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Track
        total_loss += loss.item()
        all_preds.extend(outputs['predictions'].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy
    }


def validate(model, dataloader, device, epoch, num_classes=5):
    """Validation with torch-uncertainty metrics"""

    model.eval()

    all_preds = []
    all_labels = []
    all_concept_preds = []
    all_concept_labels = []
    all_epistemic = []
    all_aleatoric = []
    all_probs = []
    all_logits = []
    total_loss = 0.0

    # Initialize torch-uncertainty metrics if available
    if TORCH_UNCERTAINTY_AVAILABLE:
        ece_metric = CalibrationError(task='multiclass', num_classes=num_classes, n_bins=15)
        brier_metric = BrierScore(num_classes=num_classes)
        nll_metric = CategoricalNLL()
        aurc_metric = AURC()
        augrc_metric = AUGRC()
    else:
        ece_metric = None
        brier_metric = None
        nll_metric = None
        aurc_metric = None
        augrc_metric = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_labels=concept_labels
            )

            total_loss += outputs['loss'].item()

            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_preds.extend(outputs['concept_predictions'].cpu().numpy())
            all_concept_labels.extend(concept_labels.cpu().numpy())
            all_epistemic.append(outputs['epistemic'].cpu())
            all_aleatoric.append(outputs['aleatoric'].cpu())
            all_probs.append(outputs['probs'].cpu())
            all_logits.append(outputs['logits'].cpu())

            # Update torch-uncertainty metrics
            if TORCH_UNCERTAINTY_AVAILABLE:
                probs = outputs['probs']
                preds = outputs['predictions']

                # Calibration metrics
                ece_metric.update(probs, labels)
                brier_metric.update(probs, labels)
                nll_metric.update(probs, labels)

                # Selective classification (use negative confidence)
                confidence = probs.max(dim=-1).values
                aurc_metric.update(
                    scores=-confidence,
                    errors=(preds != labels).long()
                )
                augrc_metric.update(
                    scores=-confidence,
                    errors=(preds != labels).long()
                )

    # Concatenate
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_concept_preds = np.array(all_concept_preds)
    all_concept_labels = np.array(all_concept_labels)
    all_epistemic = torch.cat(all_epistemic).numpy()
    all_aleatoric = torch.cat(all_aleatoric).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Task accuracy
    task_acc = (all_preds == all_labels).mean()

    # Concept accuracy (known only)
    concept_accs = []
    concept_names = ['food', 'service', 'ambiance', 'noise']

    for k in range(4):
        known_mask = (all_concept_labels[:, k] != 1)
        if known_mask.sum() > 0:
            # Binary accuracy (positive vs negative)
            c_labels_binary = (all_concept_labels[known_mask, k] / 2.0 > 0.5).astype(int)
            c_preds_binary = (all_concept_preds[known_mask, k] > 0.5).astype(int)
            c_acc = (c_preds_binary == c_labels_binary).mean()
            concept_accs.append(c_acc)
        else:
            concept_accs.append(0.0)

    # Uncertainty statistics
    epistemic_sample = all_epistemic.mean(axis=-1)
    aleatoric_sample = all_aleatoric.mean(axis=-1)

    # Correlations
    if epistemic_sample.std() > 0 and aleatoric_sample.std() > 0:
        rho_epi_ale, p_epi_ale = stats.spearmanr(epistemic_sample, aleatoric_sample)
    else:
        rho_epi_ale, p_epi_ale = 0.0, 1.0

    errors = (all_preds != all_labels).astype(float)
    if errors.std() > 0 and epistemic_sample.std() > 0:
        rho_epi_error, p_epi_error = stats.spearmanr(epistemic_sample, errors)
    else:
        rho_epi_error, p_epi_error = 0.0, 1.0

    # Quadrant analysis
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

    quadrant_results = {
        'trust_accuracy': accuracy_in_mask(trust_mask),
        'data_accuracy': accuracy_in_mask(data_mask),
        'review_accuracy': accuracy_in_mask(review_mask),
        'abstain_accuracy': accuracy_in_mask(abstain_mask),
        'trust_coverage': trust_mask.mean(),
        'data_coverage': data_mask.mean(),
        'review_coverage': review_mask.mean(),
        'abstain_coverage': abstain_mask.mean(),
    }

    # Compute torch-uncertainty metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': task_acc,
        'concept_accs': concept_accs,
        'mean_epistemic': epistemic_sample.mean(),
        'mean_aleatoric': aleatoric_sample.mean(),
        'std_epistemic': epistemic_sample.std(),
        'std_aleatoric': aleatoric_sample.std(),
        'rho_epi_ale': rho_epi_ale,
        'p_epi_ale': p_epi_ale,
        'rho_epi_error': rho_epi_error,
        'p_epi_error': p_epi_error,
        **quadrant_results
    }

    # Add torch-uncertainty metrics
    if TORCH_UNCERTAINTY_AVAILABLE:
        metrics['ece'] = ece_metric.compute().item()
        metrics['brier'] = brier_metric.compute().item()
        metrics['nll'] = nll_metric.compute().item()
        metrics['aurc'] = aurc_metric.compute().item()
        metrics['augrc'] = augrc_metric.compute().item()

    return metrics


def main():
    """Main training function"""

    print("\n" + "="*80)
    print("VCBM Training on CEBaB Dataset")
    print("="*80)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Hyperparameters
    batch_size = 16
    max_length = 256
    num_epochs = 10
    lr = 2e-5
    warmup_steps = 100

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("Loading CEBaB Dataset...")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader, val_loader, test_loader, tokenizer, metadata = get_cebab_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=0
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {metadata['train_size']} samples")
    print(f"  Val: {metadata['val_size']} samples")
    print(f"  Test: {metadata['test_size']} samples")
    print(f"  Concepts: {metadata['concept_names']}")
    print(f"  Classes: {metadata['num_classes']}")

    # ========================================================================
    # Create Model
    # ========================================================================
    print("\n" + "="*80)
    print("Creating VCBM Model...")
    print("="*80)

    config = VariationalCredalConfig(
        encoder_name='distilbert-base-uncased',
        freeze_encoder=True,
        num_concepts=4,
        concept_names=['food', 'service', 'ambiance', 'noise'],
        concept_classes=3,
        num_classes=5,
        covariance_family=CovarianceFamily.MEAN_FIELD,
        kl_weight=1e-5,
        concept_weight=0.5,
        aleatoric_weight=0.2,
        supervision_weight=1.0,
        use_orthogonal_projection=True,
        use_temperature_scaling=True,
        use_aleatoric_prior=True,
        pooling_strategy="cls",
        num_mc_samples=10
    )

    model = VariationalCredalCBM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ========================================================================
    # Optimizer and Scheduler
    # ========================================================================
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\nTraining setup:")
    print(f"  Optimizer: AdamW (lr={lr})")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {num_training_steps}")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)

    best_val_acc = 0.0
    concept_names = ['food', 'service', 'ambiance', 'noise']

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, device, epoch, num_classes=metadata['num_classes'])

        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"\n  Concept Accuracies:")
        for i, name in enumerate(concept_names):
            print(f"    {name}: {val_metrics['concept_accs'][i]:.4f}")
        print(f"\n  Uncertainty Decomposition:")
        print(f"    Mean EU: {val_metrics['mean_epistemic']:.4f} ± {val_metrics['std_epistemic']:.4f}")
        print(f"    Mean AU: {val_metrics['mean_aleatoric']:.4f} ± {val_metrics['std_aleatoric']:.4f}")
        print(f"    ρ(EU, AU): {val_metrics['rho_epi_ale']:.3f} (p={val_metrics['p_epi_ale']:.3f}) [target: < 0.1]")
        print(f"    ρ(EU, Error): {val_metrics['rho_epi_error']:.3f} (p={val_metrics['p_epi_error']:.3f}) [target: > 0.25]")

        # torch-uncertainty metrics
        if 'ece' in val_metrics:
            print(f"\n  Calibration & Selective Classification:")
            print(f"    ECE: {val_metrics['ece']:.4f} [calibration]")
            print(f"    Brier: {val_metrics['brier']:.4f} [proper scoring]")
            print(f"    NLL: {val_metrics['nll']:.4f} [likelihood]")
            print(f"    AURC: {val_metrics['aurc']:.4f} [selective classification]")
            print(f"    AUGRC: {val_metrics['augrc']:.4f} [abstention quality]")

        # Quadrant analysis
        print(f"\n  Quadrant Analysis (actionability):")
        print(f"    Trust (Low EU, Low AU): {val_metrics['trust_accuracy']:.4f} acc, {val_metrics['trust_coverage']:.1%} coverage")
        print(f"    Data (High EU, Low AU): {val_metrics['data_accuracy']:.4f} acc, {val_metrics['data_coverage']:.1%} coverage")
        print(f"    Review (Low EU, High AU): {val_metrics['review_accuracy']:.4f} acc, {val_metrics['review_coverage']:.1%} coverage")
        print(f"    Abstain (High EU, High AU): {val_metrics['abstain_accuracy']:.4f} acc, {val_metrics['abstain_coverage']:.1%} coverage")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']

            save_dir = Path("./checkpoints/cebab_simple")
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {**train_metrics, **val_metrics}
            }, save_dir / "best_model.pt")

            print(f"\n  ✓ New best model saved! (Val Acc: {best_val_acc:.4f})")

    # ========================================================================
    # Final Test Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("Training Complete! Evaluating on Test Set...")
    print("="*80)

    # Load best model
    save_dir = Path("./checkpoints/cebab_simple")
    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test
    test_metrics = validate(model, test_loader, device, "Test", num_classes=5)

    print(f"\n{'='*80}")
    print("TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"\n📊 Task Performance:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")

    print(f"\n🎯 Concept Performance:")
    for i, name in enumerate(concept_names):
        print(f"  {name.capitalize()}: {test_metrics['concept_accs'][i]:.4f}")

    print(f"\n🎯 Uncertainty Decomposition:")
    print(f"  Mean Epistemic: {test_metrics['mean_epistemic']:.4f} ± {test_metrics['std_epistemic']:.4f}")
    print(f"  Mean Aleatoric: {test_metrics['mean_aleatoric']:.4f} ± {test_metrics['std_aleatoric']:.4f}")
    print(f"  ρ(EU, AU): {test_metrics['rho_epi_ale']:.3f} (target: < 0.1) {'✓' if abs(test_metrics['rho_epi_ale']) < 0.1 else '✗'}")
    print(f"  ρ(EU, Error): {test_metrics['rho_epi_error']:.3f} (target: > 0.25) {'✓' if test_metrics['rho_epi_error'] > 0.25 else '✗'}")

    if 'ece' in test_metrics:
        print(f"\n📈 Calibration & Selective Classification (torch-uncertainty):")
        print(f"  ECE: {test_metrics['ece']:.4f} (Expected Calibration Error)")
        print(f"  Brier: {test_metrics['brier']:.4f} (Proper Scoring Rule)")
        print(f"  NLL: {test_metrics['nll']:.4f} (Negative Log Likelihood)")
        print(f"  AURC: {test_metrics['aurc']:.4f} (Area Under Risk Coverage)")
        print(f"  AUGRC: {test_metrics['augrc']:.4f} (AUC with Generalized Risk Coverage)")

    print(f"\n🎯 Quadrant Analysis (Actionability):")
    print(f"  Trust (Low EU, Low AU):  {test_metrics['trust_accuracy']:.4f} acc, {test_metrics['trust_coverage']:.1%} coverage")
    print(f"  Data (High EU, Low AU):   {test_metrics['data_accuracy']:.4f} acc, {test_metrics['data_coverage']:.1%} coverage")
    print(f"  Review (Low EU, High AU): {test_metrics['review_accuracy']:.4f} acc, {test_metrics['review_coverage']:.1%} coverage")
    print(f"  Abstain (High EU, High AU): {test_metrics['abstain_accuracy']:.4f} acc, {test_metrics['abstain_coverage']:.1%} coverage")

    # Save results
    results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_loss': float(test_metrics['loss']),
        'concept_accs': {name: float(acc) for name, acc in zip(concept_names, test_metrics['concept_accs'])},
        'mean_epistemic': float(test_metrics['mean_epistemic']),
        'mean_aleatoric': float(test_metrics['mean_aleatoric']),
        'std_epistemic': float(test_metrics['std_epistemic']),
        'std_aleatoric': float(test_metrics['std_aleatoric']),
        'rho_epi_ale': float(test_metrics['rho_epi_ale']),
        'p_epi_ale': float(test_metrics['p_epi_ale']),
        'rho_epi_error': float(test_metrics['rho_epi_error']),
        'p_epi_error': float(test_metrics['p_epi_error']),
        'best_val_acc': float(best_val_acc),
    }

    # Add torch-uncertainty metrics if available
    if 'ece' in test_metrics:
        results.update({
            'ece': float(test_metrics['ece']),
            'brier': float(test_metrics['brier']),
            'nll': float(test_metrics['nll']),
            'aurc': float(test_metrics['aurc']),
            'augrc': float(test_metrics['augrc']),
        })

    # Add quadrant results
    results.update({
        'trust_accuracy': float(test_metrics['trust_accuracy']),
        'trust_coverage': float(test_metrics['trust_coverage']),
        'data_accuracy': float(test_metrics['data_accuracy']),
        'data_coverage': float(test_metrics['data_coverage']),
        'review_accuracy': float(test_metrics['review_accuracy']),
        'review_coverage': float(test_metrics['review_coverage']),
        'abstain_accuracy': float(test_metrics['abstain_accuracy']),
        'abstain_coverage': float(test_metrics['abstain_coverage']),
    })

    results_path = save_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    print("\n" + "="*80)
    print("All Done!")
    print("="*80)

    return results


if __name__ == "__main__":
    results = main()
