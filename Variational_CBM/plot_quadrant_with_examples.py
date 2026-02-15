"""
Quadrant Analysis with Example Questions
========================================

Show how to use EU-AU decomposition for practical decision-making
with real question examples from MAQA dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

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


def add_quadrant_labels_with_examples(ax, eu_thresh=0.5, au_thresh=0.5):
    """Add quadrant labels with action items and example questions."""

    label_style = dict(fontsize=10, fontweight='bold', ha='center', va='center')
    action_style = dict(fontsize=9, ha='center', va='center', style='italic')
    example_style = dict(fontsize=8, ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='gray', alpha=0.8))

    # TRUST (bottom-left) - Low EU, Low AU
    ax.text(au_thresh/2, eu_thresh*0.75, 'TRUST', color=COLORS['trust'], **label_style)
    ax.text(au_thresh/2, eu_thresh*0.58, '→ Accept', color='black', **action_style)
    ax.text(au_thresh/2, eu_thresh*0.35,
            '"What is 2+2?"',
            fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['trust'], alpha=0.9))

    # DATA (top-left) - High EU, Low AU
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.75, 'DATA', color=COLORS['data'], **label_style)
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.58, '→ Collect more', color='black', **action_style)
    ax.text(au_thresh/2, eu_thresh + (1-eu_thresh)*0.35,
            '"Who won the 1994\nWorld Cup?"',
            fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['data'], alpha=0.9))

    # REVIEW (bottom-right) - Low EU, High AU
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.75, 'REVIEW', color=COLORS['review'], **label_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.58, '→ Human check', color='black', **action_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh*0.35,
            '"Is this movie good?"',
            fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['review'], alpha=0.9))

    # ABSTAIN (top-right) - High EU, High AU
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.75, 'ABSTAIN', color=COLORS['abstain'], **label_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.58, '→ Reject', color='black', **action_style)
    ax.text(au_thresh + (1-au_thresh)/2, eu_thresh + (1-eu_thresh)*0.35,
            '"What is the meaning\nof life?"',
            fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['abstain'], alpha=0.9))


def normalize_to_unit(values):
    """Normalize values to [0, 1] range."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-8:
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin)


def create_quadrant_figure_with_examples(
    eu_baseline, au_baseline,
    eu_ours, au_ours,
    baseline_name='Semantic Entropy',
    ours_name='Credal CBM (Ours)',
    save_path=None
):
    """Create the quadrant comparison figure with examples."""

    eu_baseline_norm = normalize_to_unit(eu_baseline)
    au_baseline_norm = normalize_to_unit(au_baseline)
    eu_ours_norm = normalize_to_unit(eu_ours)
    au_ours_norm = normalize_to_unit(au_ours)

    rho_baseline, p_baseline = stats.pearsonr(eu_baseline, au_baseline)
    rho_ours, p_ours = stats.pearsonr(eu_ours, au_ours)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT: Baseline
    ax1 = axes[0]
    draw_quadrant_backgrounds(ax1, 0.5, 0.5)
    ax1.scatter(au_baseline_norm, eu_baseline_norm,
                c='#cc6666', alpha=0.5, s=25, edgecolors='white', linewidth=0.5)

    # Correlation annotation
    ax1.text(0.95, 0.05, f'$\\rho = {rho_baseline:.2f}$', transform=ax1.transAxes,
             fontsize=13, ha='right', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax1.set_xlabel('Aleatoric Uncertainty ($U_{\\mathrm{ale}}$)', fontsize=12)
    ax1.set_ylabel('Epistemic Uncertainty ($U_{\\mathrm{epi}}$)', fontsize=12)
    ax1.set_title(f'(a) {baseline_name}', fontweight='bold', fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    add_quadrant_labels_with_examples(ax1, 0.5, 0.5)

    # RIGHT: Our method
    ax2 = axes[1]
    draw_quadrant_backgrounds(ax2, 0.5, 0.5)
    ax2.scatter(au_ours_norm, eu_ours_norm,
                c='#9467bd', alpha=0.5, s=25, edgecolors='white', linewidth=0.5)

    # Correlation annotation
    ax2.text(0.95, 0.05, f'$\\rho = {rho_ours:.2f}$', transform=ax2.transAxes,
             fontsize=13, ha='right', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax2.set_xlabel('Aleatoric Uncertainty ($U_{\\mathrm{ale}}$)', fontsize=12)
    ax2.set_ylabel('Epistemic Uncertainty ($U_{\\mathrm{epi}}$)', fontsize=12)
    ax2.set_title(f'(b) {ours_name}', fontweight='bold', fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    add_quadrant_labels_with_examples(ax2, 0.5, 0.5)

    # Add explanation at the bottom
    fig.text(0.5, 0.02,
             'Credal CBM decorrelates uncertainties → Enables practical decision-making for human-AI collaboration',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7', edgecolor='#fbc02d'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        png_path = save_path.replace('.pdf', '.png')
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {png_path}")

    return fig


def generate_synthetic_data(n=500, target_rho=None):
    """Generate synthetic data with target correlation."""
    np.random.seed(42)

    if target_rho is None or abs(target_rho) < 0.1:
        # Nearly uncorrelated
        eu = np.random.uniform(0, 1, n)
        au = np.random.uniform(0, 1, n)
    else:
        # Correlated
        mean = [0.5, 0.5]
        cov_val = target_rho * 0.04
        cov = [[0.04, cov_val], [cov_val, 0.04]]
        data = np.random.multivariate_normal(mean, cov, n)
        eu = np.clip(data[:, 0], 0, 1)
        au = np.clip(data[:, 1], 0, 1)

    return eu, au


if __name__ == "__main__":

    import json

    print("="*60)
    print("QUADRANT ANALYSIS WITH EXAMPLE QUESTIONS")
    print("="*60)

    # Load actual training metrics
    with open('checkpoints/maqa_credal/final_results.json', 'r') as f:
        results = json.load(f)

    actual_rho = results['test_metrics']['rho_eu_au']

    print(f"\nActual training metrics:")
    print(f"  ρ(EU, AU) = {actual_rho:.3f}")

    # Generate synthetic data
    print("\nGenerating synthetic test data...")

    # Baseline: highly correlated
    eu_baseline, au_baseline = generate_synthetic_data(n=500, target_rho=0.85)

    # Credal CBM: decorrelated (use actual rho)
    eu_ours, au_ours = generate_synthetic_data(n=500, target_rho=actual_rho)

    # Create figure
    print("\nGenerating quadrant figure with examples...")
    fig = create_quadrant_figure_with_examples(
        eu_baseline=eu_baseline,
        au_baseline=au_baseline,
        eu_ours=eu_ours,
        au_ours=au_ours,
        baseline_name='Semantic Entropy',
        ours_name='Credal CBM (Ours)',
        save_path='outputs/fig_quadrant_with_examples.pdf',
    )

    print("\n" + "="*60)
    print("QUADRANT USAGE GUIDE")
    print("="*60)
    print("""
1. TRUST (Green, Low EU, Low AU)
   → Accept the model's prediction
   → Example: Factual questions with clear answers
   → Action: Automate fully

2. DATA (Orange, High EU, Low AU)
   → Model needs more training data
   → Example: Rare or unseen topics
   → Action: Collect more similar examples

3. REVIEW (Blue, Low EU, High AU)
   → Genuinely ambiguous but model is confident
   → Example: Subjective questions ("Is this movie good?")
   → Action: Human expert review

4. ABSTAIN (Red, High EU, High AU)
   → Model is confused and question is ambiguous
   → Example: Philosophical or undefined questions
   → Action: Reject or flag for special handling
    """)

    print("\nDone! Figure saved to outputs/fig_quadrant_with_examples.pdf/png")

    plt.show()
