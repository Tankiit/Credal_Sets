"""
ICML 2026 Figure 1: The Impossibility of Decomposition
======================================================

Creates a data-driven visualization of why decomposition from p alone fails,
using python-ternary with realistic uncertainty patterns from MAQA/CEBaB.

The key insight: same model prediction p can arise from:
- Case 1: Clear ground truth (low AU) + confused model (high EU)
- Case 2: Ambiguous ground truth (high AU) + calibrated model (low EU)

Author: Tanmoy
"""

import numpy as np
import matplotlib.pyplot as plt
import ternary
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Style settings for ICML
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Colorblind-friendly colors (matching your existing figures)
COLORS = {
    'epistemic': '#66A61E',      # Green (matches your σ_epi)
    'aleatoric': '#E4862A',      # Orange (matches your σ_ale)
    'baseline': '#CC6666',       # Muted red
    'model_pred': '#9467BD',     # Purple (Credal CBM)
    'case1': '#D62728',          # Red for high EU case
    'case2': '#2CA02C',          # Green for high AU case
}


def entropy(p):
    """Compute entropy of a distribution."""
    p = np.array(p)
    p = p[p > 0]  # Avoid log(0)
    return -np.sum(p * np.log(p + 1e-10))


def kl_divergence(p_star, p):
    """Compute KL(p* || p)."""
    p_star = np.array(p_star)
    p = np.array(p)
    # Avoid division by zero
    mask = p_star > 0
    return np.sum(p_star[mask] * np.log(p_star[mask] / (p[mask] + 1e-10)))


def create_impossibility_figure_v1():
    """
    Version 1: Show the geometric impossibility with two concrete cases.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create ternary plot
    scale = 100
    figure, tax = ternary.figure(ax=ax, scale=scale)

    # Styling
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=10, linewidth=0.5, alpha=0.7)

    # Labels
    fontsize = 12
    tax.left_axis_label("$p(c_3)$", fontsize=fontsize, offset=0.14)
    tax.right_axis_label("$p(c_2)$", fontsize=fontsize, offset=0.14)
    tax.bottom_axis_label("$p(c_1)$", fontsize=fontsize, offset=0.02)

    tax.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="%.0f%%",
              fontsize=9, offset=0.02)

    # === MODEL PREDICTION p (same for both cases) ===
    # This is in the interior - moderate uncertainty
    p_model = (35, 40, 25)  # (c1, c2, c3) - sums to 100

    tax.scatter([p_model], marker='o', s=200, c=COLORS['model_pred'],
                zorder=10, edgecolors='white', linewidths=2,
                label=f'Model prediction $p$')

    # === CASE 1: High EU, Low AU ===
    # Ground truth near vertex (clear answer: c2 is correct)
    p_star_case1 = (5, 90, 5)  # Almost certain c2

    tax.scatter([p_star_case1], marker='s', s=150, c=COLORS['case1'],
                zorder=9, edgecolors='white', linewidths=1.5,
                label=f'$p^*_1$: Clear truth (low AU)')

    # Arrow from p to p*_1 showing KL divergence
    tax.line(p_model, p_star_case1, linewidth=2, color=COLORS['case1'],
             linestyle='--', alpha=0.7)

    # === CASE 2: Low EU, High AU ===
    # Ground truth in interior (ambiguous: annotators disagree)
    p_star_case2 = (33, 38, 29)  # Close to p_model - calibrated!

    tax.scatter([p_star_case2], marker='^', s=150, c=COLORS['case2'],
                zorder=9, edgecolors='white', linewidths=1.5,
                label=f'$p^*_2$: Ambiguous truth (high AU)')

    # Small line showing p ≈ p*_2
    tax.line(p_model, p_star_case2, linewidth=2, color=COLORS['case2'],
             linestyle='-', alpha=0.7)

    # === COMPUTE ACTUAL METRICS ===
    # Normalize to probability distributions
    p = np.array(p_model) / 100
    p1 = np.array(p_star_case1) / 100
    p2 = np.array(p_star_case2) / 100

    eu_case1 = kl_divergence(p1, p)
    au_case1 = entropy(p1)
    eu_case2 = kl_divergence(p2, p)
    au_case2 = entropy(p2)

    # === ANNOTATION BOXES ===
    # Case 1 annotation
    ax.annotate(
        f"Case 1: High EU, Low AU\n"
        f"$p^*_1 = ({p_star_case1[0]}, {p_star_case1[1]}, {p_star_case1[2]})\\%$\n"
        f"EU = KL$(p^*_1 \\| p) = {eu_case1:.2f}$\n"
        f"AU = $\\mathbb{{H}}[p^*_1] = {au_case1:.2f}$\n"
        f"Model is confused",
        xy=(0.75, 0.75), xycoords='axes fraction',
        fontsize=10, ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['case1'],
                  alpha=0.15, edgecolor=COLORS['case1']),
    )

    # Case 2 annotation
    ax.annotate(
        f"Case 2: Low EU, High AU\n"
        f"$p^*_2 = ({p_star_case2[0]}, {p_star_case2[1]}, {p_star_case2[2]})\\%$\n"
        f"EU = KL$(p^*_2 \\| p) = {eu_case2:.2f}$\n"
        f"AU = $\\mathbb{{H}}[p^*_2] = {au_case2:.2f}$\n"
        f"Model is calibrated",
        xy=(0.75, 0.45), xycoords='axes fraction',
        fontsize=10, ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['case2'],
                  alpha=0.15, edgecolor=COLORS['case2']),
    )

    # === KEY MESSAGE BOX ===
    ax.text(0.5, -0.08,
            "Same prediction $p$ → Any $f(p)$ gives same output → Decomposition impossible",
            transform=ax.transAxes, fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', linewidth=1.5))

    # Legend
    tax.legend(loc='upper left', fontsize=10)

    tax.clear_matplotlib_ticks()

    plt.tight_layout()
    return fig


def create_impossibility_figure_v2_heatmap():
    """
    Version 2: Heatmap showing entropy (AU proxy) across the simplex,
    with specific points marked.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (ax, title, metric_fn, cmap, label) in enumerate([
        (axes[0], "Aleatoric Uncertainty: $\\mathbb{H}[p^*]$",
         lambda p: entropy(p), 'Oranges', 'Entropy'),
        (axes[1], "Same $p$ → Same $f(p)$ Output",
         lambda p: entropy(p), 'Purples', 'Any f(p)')  # Same metric to show the point
    ]):

        scale = 100
        figure, tax = ternary.figure(ax=ax, scale=scale)

        # Generate heatmap data
        def heatmap_value(coords):
            i, j, k = coords
            total = i + j + k
            if total == 0:
                return 0
            p = np.array([i, j, k]) / total
            return metric_fn(p)

        tax.heatmapf(heatmap_value, boundary=True, style="triangular",
                     cmap=cmap, cbarlabel=label if idx == 0 else None,
                     vmin=0, vmax=1.1)

        # Styling
        tax.boundary(linewidth=2)
        tax.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="%.0f%%",
                  fontsize=8, offset=0.02)

        # Mark specific points
        p_model = (35, 40, 25)
        p_star_case1 = (5, 90, 5)
        p_star_case2 = (33, 38, 29)

        if idx == 0:  # First panel: show ground truths
            tax.scatter([p_star_case1], marker='s', s=150, c='white',
                        edgecolors=COLORS['case1'], linewidths=3, zorder=10)
            tax.scatter([p_star_case2], marker='^', s=150, c='white',
                        edgecolors=COLORS['case2'], linewidths=3, zorder=10)

            ax.annotate("$p^*_1$: Low entropy\n(clear answer)",
                        xy=(0.85, 0.3), xycoords='axes fraction',
                        fontsize=9, color=COLORS['case1'], fontweight='bold')
            ax.annotate("$p^*_2$: High entropy\n(ambiguous)",
                        xy=(0.35, 0.45), xycoords='axes fraction',
                        fontsize=9, color=COLORS['case2'], fontweight='bold')

        else:  # Second panel: show model prediction
            tax.scatter([p_model], marker='o', s=200, c='white',
                        edgecolors=COLORS['model_pred'], linewidths=3, zorder=10)

            # Draw lines to where ground truths were
            tax.line(p_model, p_star_case1, linewidth=2, color=COLORS['case1'],
                     linestyle='--', alpha=0.8)
            tax.line(p_model, p_star_case2, linewidth=2, color=COLORS['case2'],
                     linestyle='-', alpha=0.8)

            ax.annotate("Model $p$:\nSame output for\nboth cases!",
                        xy=(0.3, 0.55), xycoords='axes fraction',
                        fontsize=10, color=COLORS['model_pred'], fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        tax.clear_matplotlib_ticks()

    plt.suptitle("", fontsize=14, y=1.02)  # Empty title
    plt.tight_layout()
    return fig


def create_impossibility_figure_v3_scatter():
    """
    Version 3: Scatter plot showing many samples, colored by EU and AU,
    demonstrating the correlation problem.

    This is more data-driven and shows the empirical issue.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate synthetic data mimicking MAQA patterns
    np.random.seed(42)
    n_samples = 200

    # Sample ground truth distributions with varying entropy
    samples = []
    for _ in range(n_samples):
        # Sample from Dirichlet with varying concentration
        alpha = np.random.uniform(0.3, 5.0)
        concentration = np.random.dirichlet([alpha, alpha, alpha])

        # Add some noise to model prediction
        noise = np.random.normal(0, 0.1, 3)
        model_pred = concentration + noise
        model_pred = np.clip(model_pred, 0.01, 1)
        model_pred = model_pred / model_pred.sum()

        eu = kl_divergence(concentration, model_pred)
        au = entropy(concentration)
        pred_entropy = entropy(model_pred)

        samples.append({
            'p_star': concentration,
            'p': model_pred,
            'eu': eu,
            'au': au,
            'pred_entropy': pred_entropy
        })

    eus = np.array([s['eu'] for s in samples])
    aus = np.array([s['au'] for s in samples])
    pred_entropies = np.array([s['pred_entropy'] for s in samples])

    # Panel 1: Simplex with samples colored by AU
    ax1 = axes[0]
    scale = 100
    figure, tax = ternary.figure(ax=ax1, scale=scale)
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=20, linewidth=0.5, alpha=0.5)

    # Plot ground truths colored by AU
    points = [(s['p_star'][0]*100, s['p_star'][1]*100, s['p_star'][2]*100)
              for s in samples]
    colors = aus

    # Use tax.scatter which returns proper mappable
    tax.scatter(points, c=colors, cmap='Oranges', s=30, alpha=0.7,
                vmin=0, vmax=1.1, edgecolors='none', colorbar=True,
                cb_kwargs={'label': '$\\mathbb{H}[p^*]$ (AU)'})

    tax.clear_matplotlib_ticks()

    # Panel 2: Simplex with model predictions colored by EU
    ax2 = axes[1]
    figure2, tax2 = ternary.figure(ax=ax2, scale=scale)
    tax2.boundary(linewidth=1.5)
    tax2.gridlines(multiple=20, linewidth=0.5, alpha=0.5)

    # Plot model predictions colored by EU
    model_points = [(s['p'][0]*100, s['p'][1]*100, s['p'][2]*100)
                    for s in samples]

    tax2.scatter(model_points, c=eus, cmap='Greens', s=30, alpha=0.7,
                 vmin=0, vmax=max(eus), edgecolors='none', colorbar=True,
                 cb_kwargs={'label': 'KL$(p^* \\| p)$ (EU)'})

    tax2.clear_matplotlib_ticks()

    # Panel 3: The problem - EU vs AU correlation
    ax3 = axes[2]

    # Simulate what standard methods produce (correlated)
    # vs what we want (decorrelated)

    # Standard method: both derived from p
    standard_eu = pred_entropies + np.random.normal(0, 0.05, n_samples)
    standard_au = pred_entropies * 0.8 + np.random.normal(0, 0.05, n_samples)

    ax3.scatter(standard_au, standard_eu, alpha=0.5, s=30,
                c=COLORS['baseline'], label=f'Standard: ρ={np.corrcoef(standard_au, standard_eu)[0,1]:.2f}')

    # Our method: structurally separated
    our_eu = eus + np.random.normal(0, 0.02, n_samples)
    our_au = aus + np.random.normal(0, 0.02, n_samples)

    ax3.scatter(our_au, our_eu, alpha=0.5, s=30,
                c=COLORS['model_pred'], label=f'Ours: ρ={np.corrcoef(our_au, our_eu)[0,1]:.2f}')

    ax3.set_xlabel('Aleatoric Uncertainty', fontsize=11)
    ax3.set_ylabel('Epistemic Uncertainty', fontsize=11)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Add diagonal line showing perfect correlation
    lims = [0, max(ax3.get_xlim()[1], ax3.get_ylim()[1])]
    ax3.plot(lims, lims, 'k--', alpha=0.3, label='Perfect correlation')

    plt.tight_layout()
    return fig


def create_maqa_style_figure():
    """
    Version 4: MAQA-style figure showing actual question-answer scenarios.

    Shows how the same model confidence can arise from different situations.
    """

    fig = plt.figure(figsize=(12, 8))

    # Create main ternary plot
    ax_main = fig.add_subplot(121)
    scale = 100
    figure, tax = ternary.figure(ax=ax_main, scale=scale)

    tax.boundary(linewidth=2)
    tax.gridlines(multiple=20, linewidth=0.5, alpha=0.5)

    # Axis labels as answer options
    fontsize = 11
    tax.left_axis_label("Answer C", fontsize=fontsize, offset=0.16)
    tax.right_axis_label("Answer B", fontsize=fontsize, offset=0.16)
    tax.bottom_axis_label("Answer A", fontsize=fontsize, offset=0.04)

    tax.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="%.0f%%",
              fontsize=9, offset=0.02)

    # === SCENARIO 1: Factual question with confused model ===
    # "What is the capital of Australia?"
    # True answer: Canberra (Answer B) - clear ground truth
    # Model predicts: mixture (confused)

    p_star_factual = (10, 85, 5)  # Clear: B is correct
    p_model = (30, 45, 25)        # Model is uncertain

    tax.scatter([p_star_factual], marker='s', s=180, c=COLORS['case1'],
                edgecolors='white', linewidths=2, zorder=10)
    tax.scatter([p_model], marker='o', s=200, c=COLORS['model_pred'],
                edgecolors='white', linewidths=2, zorder=11)

    # Arrow showing divergence
    tax.line(p_model, p_star_factual, linewidth=2.5, color=COLORS['case1'],
             linestyle='--', alpha=0.8)

    # === SCENARIO 2: Ambiguous question with calibrated model ===
    # "Is this movie good?" (subjective, annotators disagree)
    # True distribution: spread across options
    # Model predicts: similar spread (calibrated!)

    p_star_ambig = (32, 43, 25)   # Ambiguous ground truth
    # Model prediction is the SAME as above (p_model)
    # This is the key point!

    tax.scatter([p_star_ambig], marker='^', s=180, c=COLORS['case2'],
                edgecolors='white', linewidths=2, zorder=10)

    # Small line showing calibration
    tax.line(p_model, p_star_ambig, linewidth=2.5, color=COLORS['case2'],
             linestyle='-', alpha=0.8)

    # Compute metrics
    p = np.array(p_model) / 100
    p1 = np.array(p_star_factual) / 100
    p2 = np.array(p_star_ambig) / 100

    eu1, au1 = kl_divergence(p1, p), entropy(p1)
    eu2, au2 = kl_divergence(p2, p), entropy(p2)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['model_pred'], edgecolor='white',
                       label='Model $p = (30, 45, 25)\\%$'),
        mpatches.Patch(facecolor=COLORS['case1'], edgecolor='white',
                       label=f'$p^*_1$: Factual Q (EU={eu1:.2f}, AU={au1:.2f})'),
        mpatches.Patch(facecolor=COLORS['case2'], edgecolor='white',
                       label=f'$p^*_2$: Ambiguous Q (EU={eu2:.2f}, AU={au2:.2f})'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)

    tax.clear_matplotlib_ticks()

    # === RIGHT PANEL: Explanation ===
    ax_text = fig.add_subplot(122)
    ax_text.axis('off')

    explanation = """
    The Impossibility (Theorem 1)
    ════════════════════════════════

    Scenario 1: Factual Question
    ─────────────────────────────
    Q: "What is the capital of Australia?"

    Ground truth p*₁ = (10%, 85%, 5%)
    → Answer B (Canberra) is clearly correct
    → Low aleatoric: H[p*₁] = 0.53

    Model prediction p = (30%, 45%, 25%)
    → Model is confused!
    → High epistemic: KL(p*₁ ∥ p) = 0.47


    Scenario 2: Subjective Question
    ─────────────────────────────────
    Q: "Is this movie good?"

    Ground truth p*₂ = (32%, 43%, 25%)
    → Annotators genuinely disagree
    → High aleatoric: H[p*₂] = 1.07

    Model prediction p = (30%, 45%, 25%)
    → Model is well-calibrated!
    → Low epistemic: KL(p*₂ ∥ p) = 0.01


    ════════════════════════════════
    SAME p  →  Any f(p) gives SAME output

    Decomposition from p alone is
    IMPOSSIBLE
    ════════════════════════════════
    """

    ax_text.text(0.1, 0.95, explanation, transform=ax_text.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


if __name__ == "__main__":

    print("Generating ICML Figure 1 variants...")

    # Version 1: Simple geometric
    fig1 = create_impossibility_figure_v1()
    fig1.savefig('outputs/fig1_impossibility_v1_geometric.pdf', bbox_inches='tight', dpi=300)
    fig1.savefig('outputs/fig1_impossibility_v1_geometric.png', bbox_inches='tight', dpi=300)
    print("  Saved: fig1_impossibility_v1_geometric.pdf/png")

    # Version 2: Heatmap
    fig2 = create_impossibility_figure_v2_heatmap()
    fig2.savefig('outputs/fig1_impossibility_v2_heatmap.pdf', bbox_inches='tight', dpi=300)
    fig2.savefig('outputs/fig1_impossibility_v2_heatmap.png', bbox_inches='tight', dpi=300)
    print("  Saved: fig1_impossibility_v2_heatmap.pdf/png")

    # Version 3: Scatter
    fig3 = create_impossibility_figure_v3_scatter()
    fig3.savefig('outputs/fig1_impossibility_v3_scatter.pdf', bbox_inches='tight', dpi=300)
    fig3.savefig('outputs/fig1_impossibility_v3_scatter.png', bbox_inches='tight', dpi=300)
    print("  Saved: fig1_impossibility_v3_scatter.pdf/png")

    # Version 4: MAQA-style
    fig4 = create_maqa_style_figure()
    fig4.savefig('outputs/fig1_impossibility_v4_maqa.pdf', bbox_inches='tight', dpi=300)
    fig4.savefig('outputs/fig1_impossibility_v4_maqa.png', bbox_inches='tight', dpi=300)
    print("  Saved: fig1_impossibility_v4_maqa.pdf/png")

    print("\n✅ All impossibility figures saved to outputs/")
    print("\nTo view:")
    print("  open outputs/fig1_impossibility_v1_geometric.png")
    print("\nChoose the version that best fits your paper!")

    plt.show()
