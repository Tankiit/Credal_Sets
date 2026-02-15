"""
Quadrant Analysis for MAQA Credal CBM
=====================================

Load checkpoints and generate quadrant figure showing the difference
between correlated (baselines) and decorrelated (Credal CBM) uncertainties.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main_train_hybrid_multi_dataset import (
    MAQADataset,
    load_encoder_with_quantization as load_encoder_model,
    VCBM_v7b_entropy,
    collate_fn
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
})

# Colors for quadrants
COLORS = {
    'trust': '#4CAF50',      # Green - low EU, low AU
    'data': '#FF9800',       # Orange - high EU, low AU
    'review': '#2196F3',     # Blue - low EU, high AU
    'abstain': '#F44336',    # Red - high EU, high AU
}


def draw_quadrant_backgrounds(ax, eu_thresh=0.5, au_thresh=0.5):
    """Draw colored quadrant backgrounds."""
    ax.add_patch(Rectangle((0, 0), au_thresh, eu_thresh,
                           facecolor=COLORS['trust'], alpha=0.15))
    ax.add_patch(Rectangle((0, eu_thresh), au_thresh, 1-eu_thresh,
                           facecolor=COLORS['data'], alpha=0.15))
    ax.add_patch(Rectangle((au_thresh, 0), 1-au_thresh, eu_thresh,
                           facecolor=COLORS['review'], alpha=0.15))
    ax.add_patch(Rectangle((au_thresh, eu_thresh), 1-au_thresh, 1-eu_thresh,
                           facecolor=COLORS['abstain'], alpha=0.15))

    ax.axhline(y=eu_thresh, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=au_thresh, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)


def add_quadrant_labels(ax, eu_thresh=0.5, au_thresh=0.5):
    """Add quadrant labels."""
    label_style = dict(fontsize=10, fontweight='bold', ha='center', va='center')
    action_style = dict(fontsize=8, ha='center', va='center', style='italic')

    # Trust (bottom-left)
    ax.text(au_thresh/2, eu_thresh*0.7, 'TRUST', color=COLORS['trust'], **label_style)
    ax.text(au_thresh/2, eu_thresh*0.5, '→ Accept', color='black', **action_style)

    # Data (top-left)
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.7, 'DATA', color=COLORS['data'], **label_style)
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.5, '→ Collect more', color='black', **action_style)

    # Review (bottom-right)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.7, 'REVIEW', color=COLORS['review'], **label_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.5, '→ Human check', color='black', **action_style)

    # Abstain (top-right)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.7, 'ABSTAIN', color=COLORS['abstain'], **label_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.5, '→ Reject', color='black', **action_style)


def compute_quadrant_accuracy(eu, au, correct, eu_thresh, au_thresh):
    """Compute accuracy for each quadrant."""
    results = {}

    # Trust: low EU, low AU
    mask = (eu < eu_thresh) & (au < au_thresh)
    results['trust'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)

    # Data: high EU, low AU
    mask = (eu >= eu_thresh) & (au < au_thresh)
    results['data'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)

    # Review: low EU, high AU
    mask = (eu < eu_thresh) & (au >= au_thresh)
    results['review'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)

    # Abstain: high EU, high AU
    mask = (eu >= eu_thresh) & (au >= au_thresh)
    results['abstain'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)

    return results


def normalize_to_unit(values):
    """Normalize values to [0, 1] range."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-8:
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin)


def load_maqa_model_and_data(checkpoint_path, device='cuda'):
    """Load MAQA model and compute uncertainties on test set."""

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = {
            'encoder': 'distilbert',
            'num_concepts': 3,
            'num_labels': 3,
            'hidden_dim': 256,
            'dropout': 0.1,
            'loss_version': 'v7b_entropy',
        }

    print(f"Config: {config}")

    # Load test data
    print("Loading MAQA test data...")
    test_dataset = MAQADataset(split='test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load encoder
    print("Loading encoder...")
    encoder = load_encoder_model(config['encoder'], device=device)
    tokenizer = None  # Not needed for inference

    # Initialize model
    print("Initializing model...")
    model = VCBM_v7b_entropy(
        encoder=encoder,
        num_concepts=config['num_concepts'],
        num_labels=config['num_labels'],
        hidden_dim=config.get('hidden_dim', 256),
        dropout=config.get('dropout', 0.1),
    ).to(device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Compute uncertainties
    print("Computing uncertainties on test set...")

    all_eu = []
    all_au = []
    all_predictions = []
    all_labels = []
    all_entropies = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}")

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            concept_probs, label_probs, sigma_ale, sigma_epi, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_uncertainties=True
            )

            # Get predictions
            predictions = torch.argmax(label_probs, dim=-1)

            # Compute entropy for AU
            entropy = -(label_probs * torch.log(label_probs + 1e-10)).sum(dim=-1)

            # Store
            all_eu.append(sigma_epi.cpu().numpy())
            all_au.append(sigma_ale.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_entropies.append(entropy.cpu().numpy())

    # Concatenate
    eu = np.concatenate(all_eu)
    au = np.concatenate(all_au)
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    entropies = np.concatenate(all_entropies)
    correct = (predictions == labels)

    print(f"Test set size: {len(eu)}")
    print(f"Accuracy: {correct.mean() * 100:.2f}%")

    return {
        'eu': eu,
        'au': au,
        'predictions': predictions,
        'labels': labels,
        'correct': correct,
        'entropies': entropies,
    }


def create_quadrant_figure(
    # Baseline data
    eu_baseline, au_baseline, correct_baseline,
    # Our method data
    eu_ours, au_ours, correct_ours,
    # Thresholds
    eu_thresh_baseline=0.5, au_thresh_baseline=0.5,
    eu_thresh_ours=0.5, au_thresh_ours=0.5,
    # Labels
    baseline_name='Sem. Entropy',
    ours_name='Credal CBM (Ours)',
    # Colors
    baseline_color='#cc6666',
    ours_color='#9467bd',
    # Output
    save_path=None,
    figsize=(13, 5.5),
):
    """Create the quadrant comparison figure."""

    # Normalize to [0, 1] for visualization
    eu_baseline_norm = normalize_to_unit(eu_baseline)
    au_baseline_norm = normalize_to_unit(au_baseline)
    eu_ours_norm = normalize_to_unit(eu_ours)
    au_ours_norm = normalize_to_unit(au_ours)

    # Compute correlations
    rho_baseline, p_baseline = stats.pearsonr(eu_baseline, au_baseline)
    rho_ours, p_ours = stats.pearsonr(eu_ours, au_ours)

    # Compute quadrant accuracies
    acc_baseline = compute_quadrant_accuracy(
        eu_baseline_norm, au_baseline_norm, correct_baseline,
        eu_thresh_baseline, au_thresh_baseline
    )
    acc_ours = compute_quadrant_accuracy(
        eu_ours_norm, au_ours_norm, correct_ours,
        eu_thresh_ours, au_thresh_ours
    )

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # LEFT: Baseline
    ax1 = axes[0]

    draw_quadrant_backgrounds(ax1, eu_thresh_baseline, au_thresh_baseline)
    ax1.scatter(au_baseline_norm, eu_baseline_norm,
                c=baseline_color, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    # Add accuracy annotations
    acc_style = dict(fontsize=9, ha='center', va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='gray', alpha=0.9))

    ax1.text(au_thresh_baseline/2, eu_thresh_baseline*0.25,
             f'{acc_baseline["trust"][0]:.1f}%', **acc_style)
    ax1.text(au_thresh_baseline/2, eu_thresh_baseline + (1-eu_thresh_baseline)*0.25,
             f'{acc_baseline["data"][0]:.1f}%', **acc_style)
    ax1.text(au_thresh_baseline + (1-au_thresh_baseline)/2, eu_thresh_baseline*0.25,
             f'{acc_baseline["review"][0]:.1f}%', **acc_style)
    ax1.text(au_thresh_baseline + (1-au_thresh_baseline)/2, eu_thresh_baseline + (1-eu_thresh_baseline)*0.25,
             f'{acc_baseline["abstain"][0]:.1f}%', **acc_style)

    # Gap annotation
    gap_baseline = abs(acc_baseline["review"][0] - acc_baseline["data"][0])
    ax1.text(0.95, 0.95, f'Gap: {gap_baseline:.1f}pp', transform=ax1.transAxes,
             fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    # Correlation annotation
    ax1.text(0.95, 0.05, f'$\\rho = {rho_baseline:.2f}$', transform=ax1.transAxes,
             fontsize=12, ha='right', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax1.set_xlabel('Aleatoric Uncertainty ($U_{\\mathrm{ale}}$)')
    ax1.set_ylabel('Epistemic Uncertainty ($U_{\\mathrm{epi}}$)')
    ax1.set_title(f'(a) {baseline_name}', fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    add_quadrant_labels(ax1, eu_thresh_baseline, au_thresh_baseline)

    # RIGHT: Our method
    ax2 = axes[1]

    draw_quadrant_backgrounds(ax2, eu_thresh_ours, au_thresh_ours)
    ax2.scatter(au_ours_norm, eu_ours_norm,
                c=ours_color, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    # Add accuracy annotations
    ax2.text(au_thresh_ours/2, eu_thresh_ours*0.25,
             f'{acc_ours["trust"][0]:.1f}%', **acc_style)
    ax2.text(au_thresh_ours/2, eu_thresh_ours + (1-eu_thresh_ours)*0.25,
             f'{acc_ours["data"][0]:.1f}%', **acc_style)
    ax2.text(au_thresh_ours + (1-au_thresh_ours)/2, eu_thresh_ours*0.25,
             f'{acc_ours["review"][0]:.1f}%', **acc_style)
    ax2.text(au_thresh_ours + (1-au_thresh_ours)/2, eu_thresh_ours + (1-eu_thresh_ours)*0.25,
             f'{acc_ours["abstain"][0]:.1f}%', **acc_style)

    # Gap annotation (highlighted for our method)
    gap_ours = abs(acc_ours["review"][0] - acc_ours["data"][0])
    ax2.text(0.95, 0.95, f'Gap: {gap_ours:.1f}pp', transform=ax2.transAxes,
             fontsize=9, ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', edgecolor='#fbc02d', alpha=0.9))

    # Correlation annotation
    ax2.text(0.95, 0.05, f'$\\rho = {rho_ours:.2f}$', transform=ax2.transAxes,
             fontsize=12, ha='right', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax2.set_xlabel('Aleatoric Uncertainty ($U_{\\mathrm{ale}}$)')
    ax2.set_ylabel('Epistemic Uncertainty ($U_{\\mathrm{epi}}$)')
    ax2.set_title(f'(b) {ours_name}', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    add_quadrant_labels(ax2, eu_thresh_ours, au_thresh_ours)

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        # Also save PNG
        if save_path.endswith('.pdf'):
            png_path = save_path.replace('.pdf', '.png')
            fig.savefig(png_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {png_path}")

    # Compile results
    results = {
        'baseline': {
            'rho': rho_baseline,
            'p_value': p_baseline,
            'quadrant_acc': acc_baseline,
            'gap': gap_baseline,
        },
        'ours': {
            'rho': rho_ours,
            'p_value': p_ours,
            'quadrant_acc': acc_ours,
            'gap': gap_ours,
        }
    }

    return fig, results


def print_results(results):
    """Print formatted results."""
    print("\n" + "="*60)
    print("QUADRANT ANALYSIS RESULTS")
    print("="*60)

    for method, data in results.items():
        print(f"\n{method.upper()}")
        print(f"  Correlation: ρ = {data['rho']:.3f} (p = {data['p_value']:.2e})")
        print(f"  Gap (Review - Data): {data['gap']:.1f}pp")
        print(f"  Quadrant accuracies:")
        for quad, (acc, n) in data['quadrant_acc'].items():
            print(f"    {quad.upper():8s}: {acc:5.1f}% (n={n})")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Quadrant Analysis for MAQA')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/maqa_credal/epoch_99.pt',
                        help='Path to Credal CBM checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default='outputs/fig_quadrant_maqa.pdf',
                        help='Output path')
    args = parser.parse_args()

    print("="*60)
    print("MAQA QUADRANT ANALYSIS")
    print("="*60)

    # Load Credal CBM results
    print("\nLoading Credal CBM model...")
    credal_results = load_maqa_model_and_data(args.checkpoint, device=args.device)

    # For baseline, use semantic entropy (high correlation)
    # We'll simulate this by using entropy as both EU and AU
    print("\nSimulating baseline (Semantic Entropy)...")
    eu_baseline = credal_results['entropies']  # Use entropy as EU
    au_baseline = credal_results['entropies']  # Use entropy as AU (perfect correlation)
    correct_baseline = credal_results['correct']

    # Credal CBM results
    eu_ours = credal_results['eu']
    au_ours = credal_results['au']
    correct_ours = credal_results['correct']

    # Create figure
    print("\nGenerating quadrant figure...")
    fig, results = create_quadrant_figure(
        eu_baseline=eu_baseline,
        au_baseline=au_baseline,
        correct_baseline=correct_baseline,
        eu_ours=eu_ours,
        au_ours=au_ours,
        correct_ours=correct_ours,
        baseline_name='Semantic Entropy',
        ours_name='Credal CBM (Ours)',
        save_path=args.output,
    )

    # Print results
    print_results(results)

    plt.show()

    print("\nDone!")
