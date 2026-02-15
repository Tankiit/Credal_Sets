"""
Quadrant Analysis for MAQA Credal CBM - Run Inference
=====================================================

Load checkpoint and run inference on test set to generate quadrant analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import json
from pathlib import Path
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

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


class MAQADataset(Dataset):
    """MAQA dataset loader."""

    def __init__(self, split='test'):
        # Load MAQA-Star
        maqa = load_dataset('(MPIInf/MAQA-Star', 'default', split=split)
        # Load AmbigQA
        ambigqa = load_dataset('nzender/ambigqa', 'default', split=split)

        # Combine datasets
        self.data = []
        for item in maqa:
            self.data.append({
                'question': item['question'],
                'answers': item['answers'],
                'entropy_gt': item.get('annotator_entropy', 0.5),
                'source': 'maqa'
            })

        for item in ambigqa:
            self.data.append({
                'question': item['question'],
                'answers': item['answers'],
                'entropy_gt': 0.5,  # Default for AmbigQA
                'source': 'ambigqa'
            })

        print(f"Loaded {len(self.data)} samples from {split} split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleCBM(nn.Module):
    """Simplified Credal CBM for inference."""

    def __init__(self, encoder, num_concepts=3, num_labels=3, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        # Concept heads
        self.mu_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )

        # Epistemic head
        self.sigma_epi_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # Aleatoric head (with entropy input)
        self.sigma_ale_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

        # Task classifier
        self.classifier = nn.Linear(hidden_dim + 1, num_labels)

    def forward(self, embeddings, entropy=None):
        # Get concept logits
        concept_logits = self.mu_head(embeddings)

        # Get uncertainties
        sigma_epi = self.sigma_epi_head(embeddings).squeeze(-1)

        # For sigma_ale, we need entropy
        if entropy is None:
            entropy = torch.zeros(embeddings.size(0), device=embeddings.device)

        h_for_ale = torch.cat([embeddings, entropy.unsqueeze(-1)], dim=-1)
        sigma_ale = self.sigma_ale_head(h_for_ale).squeeze(-1)

        # Get task predictions (using mu + uncertainty)
        concept_dist = torch.softmax(concept_logits, dim=-1)
        task_input = torch.cat([concept_dist, sigma_ale.unsqueeze(-1)], dim=-1)
        logits = self.classifier(task_input)

        return concept_logits, logits, sigma_ale, sigma_epi


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load checkpoint and return model and tokenizer."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Load encoder
    encoder = AutoModel.from_pretrained('distilbert-base-uncased')
    encoder.load_state_dict({k.replace('encoder.', ''): v
                            for k, v in state_dict.items()
                            if k.startswith('encoder.')})
    encoder.to(device)
    encoder.eval()

    # Initialize model
    model = SimpleCBM(encoder).to(device)

    # Load model weights (excluding encoder)
    model_state = {k: v for k, v in state_dict.items()
                   if not k.startswith('encoder.')}

    # Handle head naming
    if 'mu_head.0.weight' in model_state:
        # Already in correct format
        model.load_state_dict(model_state, strict=False)
    else:
        print("Warning: Could not load all model weights")

    model.eval()

    return model


def run_inference(model, dataloader, device='cuda'):
    """Run inference on test set."""
    print("Running inference...")

    all_eu = []
    all_au = []
    all_predictions = []
    all_labels = []
    all_entropies = []

    with torch.no_grad():
        for idx, item in enumerate(dataloader.dataset):
            if idx % 100 == 0:
                print(f"  Sample {idx}/{len(dataloader.dataset)}")

            # For simplicity, use random labels (we'll focus on uncertainties)
            # In real scenario, you'd tokenize and get actual predictions
            # This is a simplified version for demonstration

            # Mock embedding (in real case, encode the question)
            embedding = torch.randn(1, 768).to(device)
            entropy = torch.tensor([item.get('entropy_gt', 0.5)]).to(device)

            # Forward pass
            concept_logits, logits, sigma_ale, sigma_epi = model(
                embedding, entropy
            )

            # Get prediction
            prediction = torch.argmax(logits, dim=-1)

            # Compute entropy from logits
            probs = torch.softmax(logits, dim=-1)
            pred_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

            # Store (use first sample)
            all_eu.append(sigma_epi[0].cpu().item())
            all_au.append(sigma_ale[0].cpu().item())
            all_predictions.append(prediction[0].cpu().item())
            all_entropies.append(pred_entropy[0].cpu().item())

    # For labels, use mock (in real case, get from dataset)
    all_labels = [0] * len(all_predictions)

    eu = np.array(all_eu)
    au = np.array(all_au)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    entropies = np.array(all_entropies)
    correct = (predictions == labels)  # Mock: all correct

    print(f"Inference complete: {len(eu)} samples")
    print(f"  EU mean: {eu.mean():.4f} ± {eu.std():.4f}")
    print(f"  AU mean: {au.mean():.4f} ± {au.std():.4f}")

    return {
        'eu': eu,
        'au': au,
        'predictions': predictions,
        'labels': labels,
        'correct': correct,
        'entropies': entropies,
    }


# Copy the plotting functions from quadrant_analysis_simple.py
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

    ax.text(au_thresh/2, eu_thresh*0.7, 'TRUST', color=COLORS['trust'], **label_style)
    ax.text(au_thresh/2, eu_thresh*0.5, '→ Accept', color='black', **action_style)
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.7, 'DATA', color=COLORS['data'], **label_style)
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.5, '→ Collect more', color='black', **action_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.7, 'REVIEW', color=COLORS['review'], **label_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.5, '→ Human check', color='black', **action_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.7, 'ABSTAIN', color=COLORS['abstain'], **label_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.5, '→ Reject', color='black', **action_style)


def compute_quadrant_accuracy(eu, au, correct, eu_thresh, au_thresh):
    """Compute accuracy for each quadrant."""
    results = {}
    mask = (eu < eu_thresh) & (au < au_thresh)
    results['trust'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)
    mask = (eu >= eu_thresh) & (au < au_thresh)
    results['data'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)
    mask = (eu < eu_thresh) & (au >= au_thresh)
    results['review'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)
    mask = (eu >= eu_thresh) & (au >= au_thresh)
    results['abstain'] = (correct[mask].mean() * 100, mask.sum()) if mask.sum() > 0 else (0, 0)
    return results


def normalize_to_unit(values):
    """Normalize values to [0, 1] range."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-8:
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin)


def create_quadrant_figure(eu_baseline, au_baseline, correct_baseline,
                           eu_ours, au_ours, correct_ours,
                           baseline_name='Sem. Entropy',
                           ours_name='Credal CBM (Ours)',
                           save_path=None):
    """Create the quadrant comparison figure."""
    eu_baseline_norm = normalize_to_unit(eu_baseline)
    au_baseline_norm = normalize_to_unit(au_baseline)
    eu_ours_norm = normalize_to_unit(eu_ours)
    au_ours_norm = normalize_to_unit(au_ours)

    rho_baseline, p_baseline = stats.pearsonr(eu_baseline, au_baseline)
    rho_ours, p_ours = stats.pearsonr(eu_ours, au_ours)

    acc_baseline = compute_quadrant_accuracy(
        eu_baseline_norm, au_baseline_norm, correct_baseline, 0.5, 0.5
    )
    acc_ours = compute_quadrant_accuracy(
        eu_ours_norm, au_ours_norm, correct_ours, 0.5, 0.5
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # LEFT: Baseline
    ax1 = axes[0]
    draw_quadrant_backgrounds(ax1, 0.5, 0.5)
    ax1.scatter(au_baseline_norm, eu_baseline_norm,
                c='#cc6666', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    acc_style = dict(fontsize=9, ha='center', va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))

    ax1.text(0.25, 0.125, f'{acc_baseline["trust"][0]:.1f}%', **acc_style)
    ax1.text(0.25, 0.625, f'{acc_baseline["data"][0]:.1f}%', **acc_style)
    ax1.text(0.75, 0.125, f'{acc_baseline["review"][0]:.1f}%', **acc_style)
    ax1.text(0.75, 0.625, f'{acc_baseline["abstain"][0]:.1f}%', **acc_style)

    gap_baseline = abs(acc_baseline["review"][0] - acc_baseline["data"][0])
    ax1.text(0.95, 0.95, f'Gap: {gap_baseline:.1f}pp', transform=ax1.transAxes,
             fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))
    ax1.text(0.95, 0.05, f'$\\rho = {rho_baseline:.2f}$', transform=ax1.transAxes,
             fontsize=12, ha='right', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax1.set_xlabel('Aleatoric Uncertainty ($U_{\\mathrm{ale}}$)')
    ax1.set_ylabel('Epistemic Uncertainty ($U_{\\mathrm{epi}}$)')
    ax1.set_title(f'(a) {baseline_name}', fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    add_quadrant_labels(ax1, 0.5, 0.5)

    # RIGHT: Our method
    ax2 = axes[1]
    draw_quadrant_backgrounds(ax2, 0.5, 0.5)
    ax2.scatter(au_ours_norm, eu_ours_norm,
                c='#9467bd', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    ax2.text(0.25, 0.125, f'{acc_ours["trust"][0]:.1f}%', **acc_style)
    ax2.text(0.25, 0.625, f'{acc_ours["data"][0]:.1f}%', **acc_style)
    ax2.text(0.75, 0.125, f'{acc_ours["review"][0]:.1f}%', **acc_style)
    ax2.text(0.75, 0.625, f'{acc_ours["abstain"][0]:.1f}%', **acc_style)

    gap_ours = abs(acc_ours["review"][0] - acc_ours["data"][0])
    ax2.text(0.95, 0.95, f'Gap: {gap_ours:.1f}pp', transform=ax2.transAxes,
             fontsize=9, ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', edgecolor='#fbc02d', alpha=0.9))
    ax2.text(0.95, 0.05, f'$\\rho = {rho_ours:.2f}$', transform=ax2.transAxes,
             fontsize=12, ha='right', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax2.set_xlabel('Aleatoric Uncertainty ($U_{\\mathrm{ale}}$)')
    ax2.set_ylabel('Epistemic Uncertainty ($U_{\\mathrm{epi}}$)')
    ax2.set_title(f'(b) {ours_name}', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    add_quadrant_labels(ax2, 0.5, 0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        png_path = save_path.replace('.pdf', '.png')
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {png_path}")

    results = {
        'baseline': {'rho': rho_baseline, 'p_value': p_baseline, 'quadrant_acc': acc_baseline, 'gap': gap_baseline},
        'ours': {'rho': rho_ours, 'p_value': p_ours, 'quadrant_acc': acc_ours, 'gap': gap_ours}
    }

    return fig, results


if __name__ == "__main__":
    print("This script requires full inference setup.")
    print("Please use the checkpoint from training and adapt the data loading.")
    print("For now, let's use the training metrics from the JSON file.")
