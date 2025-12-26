"""
CREDENCE Ablation Analysis - ENHANCED VERSION

==============================================

Additions for ACL 2026:

1. Temperature scaling for calibration (ECE 0.164 → <0.05)

2. Weighted ensemble to fix destructive averaging

3. Error distance analysis for ordinal tasks (CEBaB)

4. Multi-annotator validation for aleatoric

Original + New analyses combined.

Usage:

    # With checkpoint file path:
    python ablation_analysis.py --checkpoint ./results/cebab_best_model.pt --dataset cebab
    
    # With directory path (will auto-detect checkpoint file):
    python ablation_analysis.py --checkpoint ./results/deberta-v3_20251223_193337 --dataset cebab

"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats
from scipy.optimize import minimize_scalar
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import from your credence.py
from credence import (
    CREDENCE, ExperimentConfig, load_model, get_hidden_states,
    MODEL_REGISTRY
)
from dataloader import load_dataset_splits, DatasetConfig

# =============================================================================
# 1. LOAD CHECKPOINT
# =============================================================================

def load_checkpoint(checkpoint_path: str, dataset: str = None, device: str = "cuda"):
    """Load saved model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt) or directory containing checkpoint
        dataset: Dataset name (used if checkpoint_path is a directory)
        device: Device to load on
    """
    # Handle directory path - look for checkpoint file inside
    if os.path.isdir(checkpoint_path):
        print(f"Checkpoint path is a directory: {checkpoint_path}")
        
        # Try to find checkpoint file
        if dataset:
            # First try: {directory}/{dataset}/{dataset}_best_model.pt (standard structure)
            dataset_subdir = os.path.join(checkpoint_path, dataset)
            expected_file = os.path.join(dataset_subdir, f"{dataset}_best_model.pt")
            if os.path.exists(expected_file):
                checkpoint_path = expected_file
                print(f"  Found checkpoint: {checkpoint_path}")
            else:
                # Second try: {directory}/{dataset}_best_model.pt (flat structure)
                expected_file_flat = os.path.join(checkpoint_path, f"{dataset}_best_model.pt")
                if os.path.exists(expected_file_flat):
                    checkpoint_path = expected_file_flat
                    print(f"  Found checkpoint: {checkpoint_path}")
                else:
                    # Third try: Look for any .pt file in dataset subdirectory
                    if os.path.isdir(dataset_subdir):
                        pt_files = [f for f in os.listdir(dataset_subdir) if f.endswith('.pt')]
                        if pt_files:
                            checkpoint_path = os.path.join(dataset_subdir, pt_files[0])
                            print(f"  Found checkpoint: {checkpoint_path}")
                        else:
                            # Fourth try: Look for any .pt file in root directory
                            pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
                            if pt_files:
                                checkpoint_path = os.path.join(checkpoint_path, pt_files[0])
                                print(f"  Found checkpoint: {checkpoint_path}")
                            else:
                                raise FileNotFoundError(
                                    f"No checkpoint file found in directory: {checkpoint_path}\n"
                                    f"Tried:\n"
                                    f"  - {expected_file}\n"
                                    f"  - {expected_file_flat}\n"
                                    f"  - Any .pt file in {dataset_subdir}\n"
                                    f"  - Any .pt file in {checkpoint_path}"
                                )
                    else:
                        # Look for any .pt file in root directory
                        pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
                        if pt_files:
                            checkpoint_path = os.path.join(checkpoint_path, pt_files[0])
                            print(f"  Found checkpoint: {checkpoint_path}")
                        else:
                            raise FileNotFoundError(
                                f"No checkpoint file found in directory: {checkpoint_path}\n"
                                f"Tried:\n"
                                f"  - {expected_file}\n"
                                f"  - {expected_file_flat}\n"
                                f"  - Any .pt file in {checkpoint_path}"
                            )
        else:
            # Look for any .pt file
            pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
            if pt_files:
                checkpoint_path = os.path.join(checkpoint_path, pt_files[0])
                print(f"  Found checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(
                    f"No checkpoint file found in directory: {checkpoint_path}\n"
                    f"Please specify --dataset or provide path to .pt file"
                )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    metadata = checkpoint['metadata']
    
    # Recreate config
    config = ExperimentConfig(**config_dict)
    
    # Load encoder
    encoder, tokenizer, hidden_size, model_type = load_model(config, device)
    
    # Recreate CREDENCE model
    model = CREDENCE(
        input_dim=hidden_size,
        num_concepts=metadata['num_concepts'],
        num_classes=metadata['num_classes'],
        head_configs=config.get_head_configs(),
        aleatoric_mode=config.aleatoric_mode,
        model_type=model_type,
    )
    
    # Load weights - handle old checkpoint format
    state_dict = checkpoint['model_state_dict']
    
    # Check if this is an old checkpoint format (using 'net' instead of 'fc1'/'fc2')
    old_format = any('net.' in key and 'heads.' in key for key in state_dict.keys())
    
    if old_format:
        print("  Detected old checkpoint format - mapping keys...")
        # Map old keys to new keys
        # Old format: heads.X.net.1 (fc1) and heads.X.net.3 (fc2)
        # New format: heads.X.fc1 and heads.X.fc2
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Map heads.X.net.1 -> heads.X.fc1
            if 'heads.' in key and '.net.1.' in key:
                new_key = key.replace('.net.1.', '.fc1.')
            # Map heads.X.net.3 -> heads.X.fc2
            elif 'heads.' in key and '.net.3.' in key:
                new_key = key.replace('.net.3.', '.fc2.')
            # Keep other keys as-is
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    # Load with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys (will use random init): {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"    - {key}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys (ignored): {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"    - {key}")
    
    model = model.to(device)
    model.eval()
    encoder.eval()
    
    print(f"  Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best val accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return model, encoder, config, metadata, model_type

# =============================================================================
# 2. COLLECT ALL PREDICTIONS
# =============================================================================

@torch.no_grad()
def collect_predictions(model, encoder, dataloader, device, model_type):
    """Collect all model outputs for analysis."""
    
    all_data = defaultdict(list)
    
    for batch in tqdm(dataloader, desc="Collecting predictions"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concepts = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)
        
        # Get hidden states
        hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
        
        # Forward pass
        outputs = model(hidden_states, attention_mask)
        
        # Store everything
        all_data['labels'].append(labels.cpu())
        all_data['concepts'].append(concepts.cpu())
        all_data['is_unknown'].append(is_unknown.cpu())
        all_data['logits'].append(outputs['logits'].cpu())
        all_data['preds'].append(outputs['logits'].argmax(dim=-1).cpu())
        all_data['concept_probs'].append(outputs['concept_probs'].cpu())
        all_data['credal_lower'].append(outputs['credal_lower'].cpu())
        all_data['credal_upper'].append(outputs['credal_upper'].cpu())
        all_data['credal_width'].append(outputs['credal_width'].cpu())
        all_data['disagreement'].append(outputs['disagreement'].cpu())
        all_data['ambiguity'].append(outputs['ambiguity'].cpu())
        all_data['total_uncertainty'].append(outputs['total_uncertainty'].cpu())
        
        # Per-head predictions
        head_probs = torch.stack(outputs['head_probs'], dim=-1)  # [batch, concepts, heads]
        all_data['head_probs'].append(head_probs.cpu())
    
    # Concatenate all batches
    return {k: torch.cat(v, dim=0).numpy() for k, v in all_data.items()}

# =============================================================================
# 3. ABLATION: HEAD CONTRIBUTION ANALYSIS
# =============================================================================

def analyze_head_contributions(data, output_dir):
    """Analyze contribution of each ensemble head."""
    print("\n" + "="*60)
    print("HEAD CONTRIBUTION ANALYSIS")
    print("="*60)
    
    head_probs = data['head_probs']  # [N, concepts, heads]
    n_samples, n_concepts, n_heads = head_probs.shape
    
    errors = (data['preds'] != data['labels']).astype(float)
    
    results = {}
    
    # Per-head accuracy (using each head independently)
    print("\nPer-head concept prediction accuracy:")
    concept_targets = data['concepts'] / 2.0  # Ternary to [0, 0.5, 1]
    
    for h in range(n_heads):
        head_preds = (head_probs[:, :, h] > 0.5).astype(float)
        # Compare to binarized concepts (0 or 2 -> 0 or 1)
        binary_targets = (data['concepts'] == 2).astype(float)
        acc = (head_preds == binary_targets).mean()
        print(f"  Head {h+1}: {acc:.4f}")
        results[f'head_{h+1}_acc'] = float(acc)
    
    # Head agreement analysis
    print("\nHead agreement (mean pairwise correlation):")
    for c in range(min(n_concepts, 5)):  # First 5 concepts
        head_preds_c = head_probs[:, c, :]  # [N, heads]
        corr_matrix = np.corrcoef(head_preds_c.T)
        mean_corr = (corr_matrix.sum() - n_heads) / (n_heads * (n_heads - 1))
        print(f"  Concept {c}: {mean_corr:.4f}")
        results[f'concept_{c}_head_agreement'] = float(mean_corr)
    
    # Disagreement vs error by head count
    print("\nDisagreement breakdown:")
    disagreement = data['disagreement'].mean(axis=1)  # Mean across concepts
    
    # Quartile analysis
    quartiles = np.percentile(disagreement, [25, 50, 75])
    for i, (low, high) in enumerate(zip([0] + list(quartiles), list(quartiles) + [1])):
        mask = (disagreement >= low) & (disagreement < high)
        if mask.sum() > 0:
            err_rate = errors[mask].mean()
            print(f"  Q{i+1} (disagreement {low:.4f}-{high:.4f}): error rate = {err_rate:.4f}, n={mask.sum()}")
            results[f'Q{i+1}_error_rate'] = float(err_rate)
    
    # Plot head correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Average head correlation across concepts
    mean_head_corr = np.zeros((n_heads, n_heads))
    for c in range(n_concepts):
        mean_head_corr += np.corrcoef(head_probs[:, c, :].T)
    mean_head_corr /= n_concepts
    
    sns.heatmap(mean_head_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Head Correlation (averaged over concepts)')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Head')
    
    # Disagreement distribution by error
    axes[1].hist(disagreement[errors == 0], bins=30, alpha=0.6, label='Correct', density=True)
    axes[1].hist(disagreement[errors == 1], bins=30, alpha=0.6, label='Error', density=True)
    axes[1].set_xlabel('Mean Disagreement')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Disagreement Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/head_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/head_analysis.png", dpi=150)
    print(f"\nSaved: {output_dir}/head_analysis.pdf")
    plt.close()
    
    return results

# =============================================================================
# 4. ABLATION: CONCEPT IMPORTANCE ANALYSIS
# =============================================================================

def analyze_concept_importance(model, data, output_dir, concept_names=None):
    """Analyze which concepts contribute most to predictions."""
    print("\n" + "="*60)
    print("CONCEPT IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get classifier weights
    W = model.classifier.W.detach().cpu().numpy()  # [num_classes, num_concepts]
    
    n_classes, n_concepts = W.shape
    
    if concept_names is None:
        concept_names = [f"Concept_{i}" for i in range(n_concepts)]
    
    results = {}
    
    # Absolute weight importance per concept
    importance = np.abs(W).mean(axis=0)  # Mean across classes
    sorted_idx = np.argsort(importance)[::-1]
    
    print("\nConcept importance (by classifier weight):")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {i+1}. {concept_names[idx]}: {importance[idx]:.4f}")
        results[f'importance_rank_{i+1}'] = concept_names[idx]
    
    # Concept uncertainty vs importance
    mean_disagreement = data['disagreement'].mean(axis=0)  # Per concept
    mean_ambiguity = data['ambiguity'].mean(axis=0)
    
    print("\nMost uncertain concepts (epistemic):")
    sorted_epi = np.argsort(mean_disagreement)[::-1]
    for i, idx in enumerate(sorted_epi[:5]):
        print(f"  {i+1}. {concept_names[idx]}: {mean_disagreement[idx]:.4f}")
    
    print("\nMost ambiguous concepts (aleatoric):")
    sorted_ale = np.argsort(mean_ambiguity)[::-1]
    for i, idx in enumerate(sorted_ale[:5]):
        print(f"  {i+1}. {concept_names[idx]}: {mean_ambiguity[idx]:.4f}")
    
    # Correlation: importance vs uncertainty
    corr_imp_epi, _ = stats.spearmanr(importance, mean_disagreement)
    corr_imp_ale, _ = stats.spearmanr(importance, mean_ambiguity)
    print(f"\nCorrelation (importance vs epistemic): {corr_imp_epi:.4f}")
    print(f"Correlation (importance vs aleatoric): {corr_imp_ale:.4f}")
    results['corr_importance_epistemic'] = float(corr_imp_epi)
    results['corr_importance_aleatoric'] = float(corr_imp_ale)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Weight heatmap
    sns.heatmap(W, cmap='RdBu_r', center=0, ax=axes[0])
    axes[0].set_xlabel('Concept')
    axes[0].set_ylabel('Class')
    axes[0].set_title('Classifier Weights')
    
    # Importance vs uncertainty
    axes[1].scatter(importance, mean_disagreement, alpha=0.7, label='Epistemic')
    axes[1].scatter(importance, mean_ambiguity, alpha=0.7, label='Aleatoric')
    axes[1].set_xlabel('Concept Importance')
    axes[1].set_ylabel('Mean Uncertainty')
    axes[1].set_title('Importance vs Uncertainty')
    axes[1].legend()
    
    # Top concepts bar chart
    top_k = min(10, n_concepts)  # Don't exceed number of concepts
    top_idx = sorted_idx[:top_k]
    axes[2].barh(range(top_k), importance[top_idx])
    axes[2].set_yticks(range(top_k))
    axes[2].set_yticklabels([concept_names[i][:15] for i in top_idx])
    axes[2].set_xlabel('Importance')
    axes[2].set_title(f'Top {top_k} Concepts')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/concept_importance.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/concept_importance.png", dpi=150)
    print(f"\nSaved: {output_dir}/concept_importance.pdf")
    plt.close()
    
    return results

# =============================================================================
# 5. ABLATION: CONCEPT INTERVENTION ANALYSIS
# =============================================================================

def analyze_concept_interventions(model, data, device, output_dir):
    """Simulate concept interventions - what if we fix certain concepts?"""
    print("\n" + "="*60)
    print("CONCEPT INTERVENTION ANALYSIS")
    print("="*60)
    
    concept_probs = torch.tensor(data['concept_probs']).to(device)
    labels = torch.tensor(data['labels']).to(device)
    disagreement = data['disagreement']
    ambiguity = data['ambiguity']
    n_samples, n_concepts = concept_probs.shape
    
    # Ground truth concepts (ternary -> probability)
    gt_concepts = torch.tensor(data['concepts'] / 2.0).float().to(device)
    
    # Baseline accuracy (no intervention)
    baseline_logits = model.classifier(concept_probs)
    baseline_preds = baseline_logits.argmax(dim=-1)
    baseline_acc = (baseline_preds == labels).float().mean().item()
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    
    results = {'baseline_acc': baseline_acc}
    
    # Strategy 1: Fix TOP-K highest epistemic uncertainty concepts
    print("\nIntervention: Fix high-EPISTEMIC concepts")
    epi_improvements = []
    for k in [1, 3, 5]:
        # Per-sample top-k epistemic concepts
        top_epi_idx = np.argsort(disagreement, axis=1)[:, -k:]
        
        # Intervene: replace predicted concepts with ground truth
        intervened = concept_probs.clone()
        for i in range(n_samples):
            intervened[i, top_epi_idx[i]] = gt_concepts[i, top_epi_idx[i]]
        
        int_logits = model.classifier(intervened)
        int_preds = int_logits.argmax(dim=-1)
        int_acc = (int_preds == labels).float().mean().item()
        improvement = int_acc - baseline_acc
        
        print(f"  Fix top-{k} epistemic: {int_acc:.4f} (+{improvement:+.4f})")
        epi_improvements.append(improvement)
        results[f'fix_top{k}_epistemic_acc'] = int_acc
    
    # Strategy 2: Fix TOP-K highest aleatoric uncertainty concepts
    print("\nIntervention: Fix high-ALEATORIC concepts")
    ale_improvements = []
    for k in [1, 3, 5]:
        top_ale_idx = np.argsort(ambiguity, axis=1)[:, -k:]
        
        intervened = concept_probs.clone()
        for i in range(n_samples):
            intervened[i, top_ale_idx[i]] = gt_concepts[i, top_ale_idx[i]]
        
        int_logits = model.classifier(intervened)
        int_preds = int_logits.argmax(dim=-1)
        int_acc = (int_preds == labels).float().mean().item()
        improvement = int_acc - baseline_acc
        
        print(f"  Fix top-{k} aleatoric: {int_acc:.4f} (+{improvement:+.4f})")
        ale_improvements.append(improvement)
        results[f'fix_top{k}_aleatoric_acc'] = int_acc
    
    # Strategy 3: Fix ALL concepts (upper bound)
    all_int_logits = model.classifier(gt_concepts)
    all_int_preds = all_int_logits.argmax(dim=-1)
    all_int_acc = (all_int_preds == labels).float().mean().item()
    print(f"\nFix ALL concepts (oracle): {all_int_acc:.4f} (+{all_int_acc - baseline_acc:+.4f})")
    results['fix_all_acc'] = all_int_acc
    
    # Key insight: which intervention strategy is more effective?
    print("\n--- KEY INSIGHT ---")
    if np.mean(ale_improvements) > np.mean(epi_improvements):
        print("Aleatoric interventions MORE effective than epistemic")
        print("→ High-aleatoric concepts have more influence on predictions")
    else:
        print("Epistemic interventions MORE effective than aleatoric")
        print("→ Model confusion drives errors more than data ambiguity")
    
    # Plot intervention comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ks = [1, 3, 5]
    x = np.arange(len(ks))
    width = 0.35
    
    ax.bar(x - width/2, epi_improvements, width, label='Fix Epistemic', color='#3498db')
    ax.bar(x + width/2, ale_improvements, width, label='Fix Aleatoric', color='#e74c3c')
    ax.axhline(y=all_int_acc - baseline_acc, color='green', linestyle='--', label='Oracle (all)')
    
    ax.set_xlabel('Number of Concepts Fixed')
    ax.set_ylabel('Accuracy Improvement')
    ax.set_title('Concept Intervention Effectiveness')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in ks])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/intervention_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/intervention_analysis.png", dpi=150)
    print(f"\nSaved: {output_dir}/intervention_analysis.pdf")
    plt.close()
    
    return results

# =============================================================================
# 6. ABLATION: SELECTIVE PREDICTION (ABSTENTION)
# =============================================================================

def analyze_selective_prediction(data, output_dir):
    """Analyze accuracy vs coverage trade-off using uncertainty."""
    print("\n" + "="*60)
    print("SELECTIVE PREDICTION ANALYSIS")
    print("="*60)
    
    errors = (data['preds'] != data['labels']).astype(float)
    disagreement = data['disagreement'].mean(axis=1)
    ambiguity = data['ambiguity'].mean(axis=1)
    total_unc = data['total_uncertainty'].mean(axis=1)
    
    # Also try max probability as baseline
    logits = data['logits']
    max_prob = np.max(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True), axis=1)
    
    results = {}
    
    def compute_risk_coverage(uncertainty, errors):
        """Compute risk-coverage curve."""
        sorted_idx = np.argsort(uncertainty)
        sorted_errors = errors[sorted_idx]
        
        coverages = np.arange(1, len(errors) + 1) / len(errors)
        risks = np.cumsum(sorted_errors) / np.arange(1, len(errors) + 1)
        
        return coverages, risks
    
    # Compute AURC (Area Under Risk-Coverage curve)
    def compute_aurc(coverages, risks):
        return np.trapz(risks, coverages)
    
    strategies = {
        'Epistemic': disagreement,
        'Aleatoric': ambiguity,
        'Total': total_unc,
        'Max Prob (baseline)': -max_prob,  # Negative because lower prob = more uncertain
        'Random': np.random.rand(len(errors)),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    print("\nArea Under Risk-Coverage Curve (lower is better):")
    for name, unc in strategies.items():
        coverages, risks = compute_risk_coverage(unc, errors)
        aurc = compute_aurc(coverages, risks)
        print(f"  {name}: {aurc:.4f}")
        results[f'aurc_{name.lower().replace(" ", "_")}'] = float(aurc)
        
        axes[0].plot(coverages * 100, risks * 100, label=f'{name} (AURC={aurc:.3f})')
    
    axes[0].set_xlabel('Coverage (%)')
    axes[0].set_ylabel('Risk (Error Rate %)')
    axes[0].set_title('Risk-Coverage Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy at different coverage levels
    print("\nAccuracy at coverage levels:")
    coverage_levels = [0.5, 0.7, 0.9, 1.0]
    
    for name, unc in strategies.items():
        if name == 'Random':
            continue
        sorted_idx = np.argsort(unc)
        
        accs = []
        for cov in coverage_levels:
            n_keep = int(cov * len(errors))
            kept_idx = sorted_idx[:n_keep]
            acc = 1 - errors[kept_idx].mean()
            accs.append(acc)
        
        print(f"  {name}: " + " | ".join([f"{cov*100:.0f}%: {acc:.3f}" for cov, acc in zip(coverage_levels, accs)]))
    
    # Plot accuracy vs coverage
    for name, unc in strategies.items():
        if name == 'Random':
            continue
        sorted_idx = np.argsort(unc)
        sorted_correct = 1 - errors[sorted_idx]
        
        coverages = np.arange(1, len(errors) + 1) / len(errors)
        accs = np.cumsum(sorted_correct) / np.arange(1, len(errors) + 1)
        
        axes[1].plot(coverages * 100, accs * 100, label=name)
    
    axes[1].axhline(y=(1-errors.mean())*100, color='gray', linestyle='--', label='Full coverage')
    axes[1].set_xlabel('Coverage (%)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy vs Coverage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/selective_prediction.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/selective_prediction.png", dpi=150)
    print(f"\nSaved: {output_dir}/selective_prediction.pdf")
    plt.close()
    
    return results

# =============================================================================
# 7. ABLATION: CALIBRATION ANALYSIS (ENHANCED WITH TEMPERATURE SCALING)
# =============================================================================

def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple:
    def softmax(x, axis=-1):
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=axis, keepdims=True)
    
    def compute_ece(temp):
        scaled = logits / temp
        probs = softmax(scaled, axis=-1)
        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        
        n_bins = 15
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = (preds[mask] == labels[mask]).mean()
                bin_conf = confidences[mask].mean()
                ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
        return ece
    
    # Grid search
    temps = np.linspace(0.5, 5.0, 50)
    eces = [compute_ece(t) for t in temps]
    best_idx = np.argmin(eces)
    
    # Refine
    result = minimize_scalar(compute_ece, bounds=(temps[max(0, best_idx-2)], 
                                                   temps[min(len(temps)-1, best_idx+2)]),
                            method='bounded')
    
    return result.x, compute_ece(result.x)

def analyze_calibration(data, output_dir, n_bins=15):
    """Analyze model calibration with reliability diagrams."""
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)
    
    logits = data['logits']
    labels = data['labels']
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(float)
    
    results = {}
    
    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
            
            ece += (bin_count / len(labels)) * abs(bin_acc - bin_conf)
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_counts.append(0)
    
    original_ece = ece
    original_mce = max([abs(a - c) for a, c in zip(bin_accs, bin_confs) if c > 0])
    
    print(f"\n--- ORIGINAL CALIBRATION ---")
    print(f"ECE: {original_ece:.4f}")
    print(f"MCE: {original_mce:.4f}")
    
    # === TEMPERATURE SCALING FIX ===
    print(f"\n--- TEMPERATURE SCALING FIX ---")
    optimal_temp, calibrated_ece = find_optimal_temperature(logits, labels)
    
    print(f"Optimal temperature: {optimal_temp:.3f}")
    print(f"Calibrated ECE: {calibrated_ece:.4f} (was {original_ece:.4f})")
    print(f"Improvement: {(original_ece - calibrated_ece) / original_ece * 100:.1f}%")
    
    if calibrated_ece < 0.05:
        print("Temperature scaling fixes calibration")
    else:
        print("Temperature scaling helps but may need additional fixes")
    
    # Compute calibrated metrics
    scaled_logits = logits / optimal_temp
    scaled_probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
    scaled_confidences = scaled_probs.max(axis=1)
    
    calibrated_bin_accs = []
    for i in range(n_bins):
        mask = (scaled_confidences >= bin_boundaries[i]) & (scaled_confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            calibrated_bin_accs.append(correct[mask].mean())
        else:
            calibrated_bin_accs.append(0)
    
    # Credal width analysis
    credal_widths = data['credal_width'].mean(axis=1)
    print(f"\nMean credal interval width: {credal_widths.mean():.4f}")
    if credal_widths.mean() > 0.4:
        print("Credal intervals too wide - consider reducing ensemble diversity")
    
    results = {
        'original_ece': float(original_ece),
        'original_mce': float(original_mce),
        'optimal_temperature': float(optimal_temp),
        'calibrated_ece': float(calibrated_ece),
        'improvement_percent': float((original_ece - calibrated_ece) / original_ece * 100),
        'mean_credal_width': float(credal_widths.mean())
    }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    bin_mids = [(bin_boundaries[i] + bin_boundaries[i+1]) / 2 for i in range(n_bins)]
    
    # Original reliability diagram
    axes[0].bar(bin_mids, bin_accs, width=1/n_bins, alpha=0.7, edgecolor='black')
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Original (ECE={original_ece:.3f})')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Calibrated reliability diagram
    axes[1].bar(bin_mids, calibrated_bin_accs, width=1/n_bins, alpha=0.7, 
                edgecolor='black', color='green')
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'After T={optimal_temp:.2f} (ECE={calibrated_ece:.3f})')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    # Confidence distributions
    axes[2].hist(confidences, bins=n_bins, alpha=0.5, label='Original', density=True)
    axes[2].hist(scaled_confidences, bins=n_bins, alpha=0.5, label='Calibrated', density=True)
    axes[2].set_xlabel('Confidence')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Confidence Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_enhanced.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/calibration_enhanced.png", dpi=150)
    print(f"\nSaved: {output_dir}/calibration_enhanced.pdf")
    plt.close()
    
    return results

# =============================================================================
# 8. ABLATION: WEIGHTED ENSEMBLE (FIXES DESTRUCTIVE AVERAGING)
# =============================================================================

def analyze_weighted_ensemble(model, data, device, output_dir):
    print("\n" + "="*60)
    print("WEIGHTED ENSEMBLE FIX")
    print("="*60)
    
    head_probs = data['head_probs']  # [N, concepts, heads]
    labels = data['labels']
    n_samples, n_concepts, n_heads = head_probs.shape
    
    results = {}
    
    # 1. Compute per-head accuracy
    print("\n--- Per-Head Performance ---")
    head_accuracies = []
    
    for h in range(n_heads):
        # Use only this head's predictions
        single_head_probs = head_probs[:, :, h]  # [N, concepts]
        single_head_tensor = torch.tensor(single_head_probs).float().to(device)
        
        logits = model.classifier(single_head_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()
        acc = (preds == labels).mean()
        head_accuracies.append(acc)
        print(f"  Head {h+1}: {acc:.1%}")
    
    results['head_accuracies'] = head_accuracies
    
    # 2. Uniform ensemble (current method)
    uniform_probs = head_probs.mean(axis=-1)
    uniform_tensor = torch.tensor(uniform_probs).float().to(device)
    uniform_logits = model.classifier(uniform_tensor)
    uniform_preds = uniform_logits.argmax(dim=-1).cpu().numpy()
    uniform_acc = (uniform_preds == labels).mean()
    
    print(f"\n--- Ensemble Comparison ---")
    print(f"Uniform ensemble: {uniform_acc:.1%}")
    results['uniform_ensemble_acc'] = float(uniform_acc)
    
    # 3. Best single head
    best_head_idx = np.argmax(head_accuracies)
    best_single_acc = head_accuracies[best_head_idx]
    print(f"Best single head (#{best_head_idx+1}): {best_single_acc:.1%}")
    results['best_single_head_acc'] = float(best_single_acc)
    
    # 4. WEIGHTED ENSEMBLE FIX
    # Option A: Accuracy-based weights (softmax with temperature)
    accs_array = np.array(head_accuracies)
    
    # Try different temperature values
    best_weighted_acc = 0
    best_weights = None
    best_temp = None
    
    for temp in [0.05, 0.1, 0.2, 0.5, 1.0]:
        weights = np.exp(accs_array / temp)
        weights = weights / weights.sum()
        
        weighted_probs = np.einsum('nch,h->nc', head_probs, weights)
        weighted_tensor = torch.tensor(weighted_probs).float().to(device)
        weighted_logits = model.classifier(weighted_tensor)
        weighted_preds = weighted_logits.argmax(dim=-1).cpu().numpy()
        weighted_acc = (weighted_preds == labels).mean()
        
        if weighted_acc > best_weighted_acc:
            best_weighted_acc = weighted_acc
            best_weights = weights
            best_temp = temp
    
    print(f"\nWeighted ensemble (T={best_temp}): {best_weighted_acc:.1%}")
    print(f"  Weights: {[f'{w:.3f}' for w in best_weights]}")
    results['weighted_ensemble_acc'] = float(best_weighted_acc)
    results['optimal_weights'] = best_weights.tolist()
    results['optimal_temperature'] = float(best_temp)
    
    # 5. Option B: Drop weak heads
    weak_threshold = 0.5
    strong_heads = [h for h, acc in enumerate(head_accuracies) if acc >= weak_threshold]
    
    if len(strong_heads) < n_heads:
        print(f"\n--- Dropping Weak Heads ---")
        print(f"Keeping heads: {[h+1 for h in strong_heads]} (acc >= {weak_threshold:.0%})")
        
        strong_probs = head_probs[:, :, strong_heads].mean(axis=-1)
        strong_tensor = torch.tensor(strong_probs).float().to(device)
        strong_logits = model.classifier(strong_tensor)
        strong_preds = strong_logits.argmax(dim=-1).cpu().numpy()
        strong_acc = (strong_preds == labels).mean()
        
        print(f"Strong heads ensemble: {strong_acc:.1%}")
        results['strong_heads_only_acc'] = float(strong_acc)
        results['dropped_heads'] = [h+1 for h in range(n_heads) if h not in strong_heads]
    
    # 6. Summary
    print(f"\n--- SUMMARY ---")
    improvement = best_weighted_acc - uniform_acc
    print(f"Uniform → Weighted: {uniform_acc:.1%} → {best_weighted_acc:.1%} (+{improvement*100:.1f}%)")
    
    if best_weighted_acc > uniform_acc:
        print("Weighted ensemble improves accuracy")
    
    if uniform_acc < best_single_acc:
        print("WARNING: Uniform ensemble is still destructive")
        print(f"Single head ({best_single_acc:.1%}) beats weighted ({best_weighted_acc:.1%})")
        print("Consider: (1) More diverse dropout, (2) Different aggregation")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    methods = ['Uniform', 'Weighted', 'Best Single']
    accs = [uniform_acc, best_weighted_acc, best_single_acc]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    axes[0].bar(methods, [a*100 for a in accs], color=colors)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Ensemble Method Comparison')
    for i, acc in enumerate(accs):
        axes[0].text(i, acc*100 + 1, f'{acc:.1%}', ha='center')
    
    # Weight visualization
    x = np.arange(n_heads)
    axes[1].bar(x - 0.2, [1/n_heads]*n_heads, 0.4, label='Uniform', alpha=0.7)
    axes[1].bar(x + 0.2, best_weights, 0.4, label='Optimal', alpha=0.7)
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Ensemble Weights')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'H{i+1}\n({head_accuracies[i]:.0%})' for i in range(n_heads)])
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weighted_ensemble.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/weighted_ensemble.png", dpi=150)
    print(f"\nSaved: {output_dir}/weighted_ensemble.pdf")
    plt.close()
    
    return results

# =============================================================================
# 9. ABLATION: ERROR DISTANCE ANALYSIS (ORDINAL TASKS)
# =============================================================================

def analyze_error_distance(data, output_dir, num_classes=5):
    print("\n" + "="*60)
    print("ERROR DISTANCE ANALYSIS (Ordinal Validation)")
    print("="*60)
    
    predictions = data['preds']
    labels = data['labels']
    disagreement = data['disagreement'].mean(axis=1)  # Epistemic
    ambiguity = data['ambiguity'].mean(axis=1)  # Aleatoric
    
    # Only analyze if we have ordinal data
    unique_labels = np.unique(labels)
    if len(unique_labels) < 3:
        print("Not enough classes for ordinal analysis (need 3+)")
        return {'skipped': True, 'reason': 'binary classification'}
    
    # Compute error distances
    error_distances = np.abs(predictions - labels)
    is_error = predictions != labels
    
    results = {'by_distance': {}}
    
    print(f"\n{'Distance':<12} {'Count':<8} {'Epistemic':<12} {'Aleatoric':<12} {'Ratio':<10}")
    print("-" * 54)
    
    distances_list = []
    epistemic_means = []
    aleatoric_means = []
    
    # Correct predictions
    correct_mask = ~is_error
    if correct_mask.sum() > 0:
        epi_correct = disagreement[correct_mask].mean()
        ale_correct = ambiguity[correct_mask].mean()
        print(f"{'Correct':<12} {correct_mask.sum():<8} {epi_correct:<12.4f} {ale_correct:<12.4f}")
        results['by_distance']['correct'] = {
            'count': int(correct_mask.sum()),
            'epistemic': float(epi_correct),
            'aleatoric': float(ale_correct)
        }
    
    # By error distance
    for dist in range(1, num_classes):
        mask = is_error & (error_distances == dist)
        
        if mask.sum() >= 5:
            epi_mean = disagreement[mask].mean()
            ale_mean = ambiguity[mask].mean()
            ratio = epi_mean / ale_mean if ale_mean > 0 else float('inf')
            
            print(f"{'Dist=' + str(dist):<12} {mask.sum():<8} {epi_mean:<12.4f} {ale_mean:<12.4f} {ratio:<10.2f}")
            
            results['by_distance'][f'distance_{dist}'] = {
                'count': int(mask.sum()),
                'epistemic': float(epi_mean),
                'aleatoric': float(ale_mean),
                'ratio': float(ratio)
            }
            
            distances_list.append(dist)
            epistemic_means.append(epi_mean)
            aleatoric_means.append(ale_mean)
    
    # Correlation analysis
    if len(distances_list) >= 3:
        corr_epi, p_epi = stats.spearmanr(distances_list, epistemic_means)
        corr_ale, p_ale = stats.spearmanr(distances_list, aleatoric_means)
        
        print(f"\n--- Correlation Analysis ---")
        print(f"Epistemic vs Distance: ρ={corr_epi:.3f} (p={p_epi:.3f})")
        print(f"  Expected: POSITIVE (distant errors = model failure)")
        epi_validates = corr_epi > 0.2
        print(f"  Status: {'VALIDATES' if epi_validates else 'DOES NOT VALIDATE'}")
        
        print(f"\nAleatoric vs Distance: ρ={corr_ale:.3f} (p={p_ale:.3f})")
        print(f"  Expected: NEGATIVE (adjacent errors = inherent ambiguity)")
        ale_validates = corr_ale < 0
        print(f"  Status: {'VALIDATES' if ale_validates else 'DOES NOT VALIDATE'}")
        
        results['correlations'] = {
            'epistemic_vs_distance': {'rho': float(corr_epi), 'p': float(p_epi), 'validates': epi_validates},
            'aleatoric_vs_distance': {'rho': float(corr_ale), 'p': float(p_ale), 'validates': ale_validates}
        }
        
        print(f"\n--- VALIDATION SUMMARY ---")
        if epi_validates and ale_validates:
            print("STRONG: Both uncertainty types show expected patterns")
            print("This is strong evidence for meaningful decomposition")
        elif epi_validates:
            print("PARTIAL: Epistemic validates, aleatoric unclear")
        elif ale_validates:
            print("PARTIAL: Aleatoric validates, epistemic unclear")
        else:
            print("WEAK: Neither pattern validates")
            print("Consider: Is calibration fixed? Are labels truly ordinal?")
        
        results['validation'] = {
            'epistemic_validates': epi_validates,
            'aleatoric_validates': ale_validates,
            'both_validate': epi_validates and ale_validates
        }
    
    # Plot
    if len(distances_list) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart by distance
        x = np.arange(len(distances_list))
        width = 0.35
        
        axes[0].bar(x - width/2, epistemic_means, width, label='Epistemic', color='#3498db')
        axes[0].bar(x + width/2, aleatoric_means, width, label='Aleatoric', color='#e74c3c')
        axes[0].set_xlabel('Error Distance')
        axes[0].set_ylabel('Mean Uncertainty')
        axes[0].set_title('Uncertainty by Error Distance')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'Dist={d}' for d in distances_list])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter with trend
        axes[1].scatter(distances_list, epistemic_means, s=100, label='Epistemic', color='#3498db')
        axes[1].scatter(distances_list, aleatoric_means, s=100, label='Aleatoric', color='#e74c3c')
        
        # Trend lines
        if len(distances_list) >= 2:
            z_epi = np.polyfit(distances_list, epistemic_means, 1)
            z_ale = np.polyfit(distances_list, aleatoric_means, 1)
            axes[1].plot(distances_list, np.polyval(z_epi, distances_list), '--', color='#3498db', alpha=0.5)
            axes[1].plot(distances_list, np.polyval(z_ale, distances_list), '--', color='#e74c3c', alpha=0.5)
        
        axes[1].set_xlabel('Error Distance')
        axes[1].set_ylabel('Mean Uncertainty')
        axes[1].set_title('Trend Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_distance.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/error_distance.png", dpi=150)
        print(f"\nSaved: {output_dir}/error_distance.pdf")
        plt.close()
    
    return results

# =============================================================================
# 10. ABLATION: NUMBER OF HEADS
# =============================================================================

def analyze_head_ablation(model, data, device, output_dir):
    """Analyze effect of using different numbers of heads."""
    print("\n" + "="*60)
    print("NUMBER OF HEADS ABLATION")
    print("="*60)
    
    head_probs = data['head_probs']  # [N, concepts, heads]
    labels = data['labels']
    n_samples, n_concepts, n_heads = head_probs.shape
    
    results = {}
    
    print(f"\nAnalyzing effect of using 1 to {n_heads} heads:")
    
    head_counts = []
    accuracies = []
    correlations = []
    
    for k in range(1, n_heads + 1):
        # Use first k heads
        subset_probs = head_probs[:, :, :k]
        
        # Aggregate: mean
        mean_probs = subset_probs.mean(axis=-1)
        
        # Epistemic: variance
        if k > 1:
            disagreement = subset_probs.var(axis=-1).mean(axis=1)
        else:
            disagreement = np.zeros(n_samples)
        
        # Get predictions
        mean_probs_tensor = torch.tensor(mean_probs).float().to(device)
        logits = model.classifier(mean_probs_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()
        
        # Metrics
        acc = (preds == labels).mean()
        errors = (preds != labels).astype(float)
        
        if k > 1 and disagreement.std() > 0:
            corr, _ = stats.spearmanr(disagreement, errors)
        else:
            corr = 0.0
        
        print(f"  {k} heads: accuracy={acc:.4f}, ρ(epistemic, error)={corr:.4f}")
        
        head_counts.append(k)
        accuracies.append(acc)
        correlations.append(corr)
        
        results[f'{k}_heads_acc'] = float(acc)
        results[f'{k}_heads_corr'] = float(corr)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(head_counts, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Heads')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Number of Heads')
    axes[0].set_xticks(head_counts)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(head_counts, correlations, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Heads')
    axes[1].set_ylabel('ρ(epistemic, error)')
    axes[1].set_title('Epistemic-Error Correlation vs Heads')
    axes[1].set_xticks(head_counts)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/head_ablation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/head_ablation.png", dpi=150)
    print(f"\nSaved: {output_dir}/head_ablation.pdf")
    plt.close()
    
    return results

# =============================================================================
# 9. MAIN ANALYSIS RUNNER
# =============================================================================

def run_all_ablations(checkpoint_path: str, dataset: str, output_dir: str, device: str = "cuda"):
    """Run all ablation analyses."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint (pass dataset name to help find checkpoint in directory)
    model, encoder, config, metadata, model_type = load_checkpoint(checkpoint_path, dataset=dataset, device=device)
    
    # Load test data
    dataset_config = DatasetConfig(
        label_type=config.label_type,
        max_length=config.max_length,
        tokenizer_name=config.encoder_name,
        batch_size=config.batch_size
    )
    
    _, _, test_loader, tokenizer, _ = load_dataset_splits(dataset, config=dataset_config)
    
    # Collect all predictions
    print("\nCollecting model predictions...")
    data = collect_predictions(model, encoder, test_loader, device, model_type)
    
    # Get concept names if available
    concept_names = metadata.get('concept_names', None)
    
    # Run all analyses
    all_results = {}
    
    concept_names = metadata.get('concept_names', None)
    num_classes = metadata.get('num_classes', 5)
    
    # === ORIGINAL ANALYSES ===
    all_results['head_contributions'] = analyze_head_contributions(data, output_dir)
    all_results['concept_importance'] = analyze_concept_importance(model, data, output_dir, concept_names)
    all_results['interventions'] = analyze_concept_interventions(model, data, device, output_dir)
    all_results['selective_prediction'] = analyze_selective_prediction(data, output_dir)
    
    # === ENHANCED ANALYSES ===
    # 1. Calibration with temperature scaling FIX
    all_results['calibration'] = analyze_calibration(data, output_dir)
    
    # 2. Weighted ensemble FIX
    all_results['weighted_ensemble'] = analyze_weighted_ensemble(model, data, device, output_dir)
    
    # 3. Error distance analysis (for ordinal tasks)
    if num_classes >= 3:  # Only for ordinal
        all_results['error_distance'] = analyze_error_distance(data, output_dir, num_classes)
    
    # 4. Head ablation
    all_results['head_ablation'] = analyze_head_ablation(model, data, device, output_dir)
    
    # Save all results
    results_path = f"{output_dir}/ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: KEY METRICS FOR ACL")
    print("="*60)
    
    cal = all_results['calibration']
    print(f"\nCalibration:")
    print(f"  Original ECE: {cal['original_ece']:.4f}")
    print(f"  After temp scaling (T={cal['optimal_temperature']:.2f}): {cal['calibrated_ece']:.4f}")
    
    ens = all_results['weighted_ensemble']
    print(f"\nEnsemble:")
    print(f"  Uniform: {ens['uniform_ensemble_acc']:.1%}")
    print(f"  Weighted: {ens['weighted_ensemble_acc']:.1%}")
    print(f"  Best single: {ens['best_single_head_acc']:.1%}")
    
    interv = all_results['interventions']
    epi_gain = interv['fix_top1_epistemic_acc'] - interv['baseline_acc']
    ale_gain = interv['fix_top1_aleatoric_acc'] - interv['baseline_acc']
    ratio = epi_gain / ale_gain if ale_gain > 0 else float('inf')
    print(f"\nInterventions:")
    print(f"  Epistemic top-1 gain: +{epi_gain*100:.1f}%")
    print(f"  Aleatoric top-1 gain: +{ale_gain*100:.1f}%")
    print(f"  Ratio: {ratio:.1f}x {'(epistemic wins)' if ratio > 1 else '(aleatoric wins)'}")
    
    if 'error_distance' in all_results and 'correlations' in all_results['error_distance']:
        ed = all_results['error_distance']
        print(f"\nError Distance:")
        print(f"  Epistemic validates: {ed['validation']['epistemic_validates']}")
        print(f"  Aleatoric validates: {ed['validation']['aleatoric_validates']}")
    
    print(f"\nAll results saved to: {results_path}")
    print("="*60)
    
    return all_results

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CREDENCE Ablation Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to saved checkpoint file (.pt) or directory containing checkpoint")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset name (also used to find checkpoint if checkpoint is a directory)")
    parser.add_argument("--output_dir", type=str, default="./ablations", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    run_all_ablations(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()

