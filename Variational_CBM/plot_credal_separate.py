"""
Credal Set Visualization - Separate Images (No Axes)
====================================================

Two separate figures showing credal sets on probability simplex.
No axes, no borders, just the simplex geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import ternary
from matplotlib.patches import Ellipse

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'figure.dpi': 150,
})

COLORS = {
    'credal': '#9467bd',        # Purple
    'ground_truth': '#FFD700',   # Gold
}


def ternary_to_cartesian(p, scale=100):
    """Convert ternary coordinates to cartesian."""
    p1, p2, p3 = p[0]/scale, p[1]/scale, p[2]/scale
    x = 0.5 * (2 * p2 + p3)
    y = (np.sqrt(3) / 2) * p3
    return x * scale, y * scale


def draw_ellipse_on_simplex(ax, center, sigma_epi, color, scale=100):
    """Draw an ellipse (credal set) on the ternary plot."""
    cx, cy = ternary_to_cartesian(center, scale)
    width = sigma_epi * scale * 0.15
    height = sigma_epi * scale * 0.12

    ellipse = Ellipse(
        (cx, cy), width, height,
        angle=30,
        facecolor=color,
        edgecolor='black',
        alpha=0.6,
        linewidth=2.5
    )
    ax.add_patch(ellipse)


def create_factual_question():
    """Figure 1: Factual question with clear answer."""

    fig, ax = plt.subplots(figsize=(8, 8))
    scale = 100

    # Create ternary plot
    figure, tax = ternary.figure(ax=ax, scale=scale)

    # Draw simplex boundary
    tax.boundary(linewidth=3)

    # No grid lines

    # Vertex labels - "What is 2 + 2?"
    tax.left_corner_label("4", fontsize=18, offset=0.008, fontweight='bold')
    tax.right_corner_label("3", fontsize=16, offset=0.008)
    tax.top_corner_label("5", fontsize=16, offset=0.045)

    # Credal set near "4" vertex
    center = (85, 10, 5)
    draw_ellipse_on_simplex(ax, center, sigma_epi=1.8, color=COLORS['credal'], scale=scale)

    # Ground truth at "4" vertex
    gt = (98, 1, 1)
    cx, cy = ternary_to_cartesian(gt, scale)
    ax.plot(cx, cy, '*', color=COLORS['ground_truth'], markersize=20,
             markeredgecolor='black', markeredgewidth=2, zorder=11)

    # Remove all axes elements
    tax.clear_matplotlib_ticks()
    ax.set_axis_off()
    ax.set_aspect('equal')

    # Remove figure frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


def create_subjective_question():
    """Figure 2: Subjective question with ambiguous answer."""

    fig, ax = plt.subplots(figsize=(8, 8))
    scale = 100

    # Create ternary plot
    figure, tax = ternary.figure(ax=ax, scale=scale)

    # Draw simplex boundary
    tax.boundary(linewidth=3)

    # No grid lines

    # Vertex labels - "Is this movie good?"
    tax.left_corner_label("Yes", fontsize=16, offset=0.008)
    tax.right_corner_label("No", fontsize=16, offset=0.008)
    tax.top_corner_label("Maybe", fontsize=15, offset=0.045)

    # Credal set in interior
    center = (35, 35, 30)
    draw_ellipse_on_simplex(ax, center, sigma_epi=1.8, color=COLORS['credal'], scale=scale)

    # Ground truth in interior
    gt = (35, 35, 30)
    cx, cy = ternary_to_cartesian(gt, scale)
    ax.plot(cx, cy, '*', color=COLORS['ground_truth'], markersize=20,
             markeredgecolor='black', markeredgewidth=2, zorder=11)

    # Remove all axes elements
    tax.clear_matplotlib_ticks()
    ax.set_axis_off()
    ax.set_aspect('equal')

    # Remove figure frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


if __name__ == "__main__":

    print("Generating separate credal set figures (no axes)...")

    # Figure 1: Factual
    fig1 = create_factual_question()
    fig1.savefig('outputs/fig_credal_factual.pdf', bbox_inches='tight', dpi=300,
                pad_inches=0, transparent=False)
    fig1.savefig('outputs/fig_credal_factual.png', bbox_inches='tight', dpi=300,
                pad_inches=0, transparent=False)
    print("✓ Saved: fig_credal_factual.pdf/png")
    print("  Question: 'What is 2+2?'")
    print("  Vertices: Left='4', Right='3', Top='5'")

    # Figure 2: Subjective
    fig2 = create_subjective_question()
    fig2.savefig('outputs/fig_credal_subjective.pdf', bbox_inches='tight', dpi=300,
                pad_inches=0, transparent=False)
    fig2.savefig('outputs/fig_credal_subjective.png', bbox_inches='tight', dpi=300,
                pad_inches=0, transparent=False)
    print("✓ Saved: fig_credal_subjective.pdf/png")
    print("  Question: 'Is this movie good?'")
    print("  Vertices: Left='Yes', Right='No', Top='Maybe'")

    print("\n✅ Two separate figures generated:")
    print("  - fig_credal_factual: Credal set near vertex (clear answer)")
    print("  - fig_credal_subjective: Credal set in interior (ambiguous)")
    print("\nFeatures:")
    print("  - No axes")
    print("  - No borders")
    print("  - Only simplex geometry, labels, ellipses, and ground truth markers")

    plt.show()
