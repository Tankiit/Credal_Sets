"""
Credal Set Visualization - Clean Version (No Text Boxes)
=========================================================

Shows credal sets (ellipses) on probability simplex.
"""

import numpy as np
import matplotlib.pyplot as plt
import ternary
from matplotlib.patches import Ellipse

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'credal_confident': '#9467bd',    # Purple
    'credal_confused': '#d62728',      # Red
    'ground_truth': '#FFD700',         # Gold
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

    # Scale sigma to visual size
    width = sigma_epi * scale * 0.15
    height = sigma_epi * scale * 0.12

    ellipse = Ellipse(
        (cx, cy), width, height,
        angle=30,
        facecolor=color,
        edgecolor='black',
        alpha=0.6,
        linewidth=2
    )
    ax.add_patch(ellipse)


def create_credal_comparison_clean():
    """Create clean side-by-side comparison of credal sets."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scale = 100

    # LEFT: Factual Question
    ax1 = axes[0]
    figure1, tax1 = ternary.figure(ax=ax1, scale=scale)

    tax1.boundary(linewidth=2.5)
    tax1.gridlines(multiple=20, linewidth=0.5, alpha=0.4)

    # Answer labels for "What is 2 + 2?" - Correct ternary positions
    # In ternary plot: left corner = p1 (first option), right = p2, top = p3
    tax1.left_corner_label("4", fontsize=13, offset=0.02, fontweight='bold')
    tax1.right_corner_label("3", fontsize=12, offset=0.02)
    tax1.top_corner_label("5", fontsize=12, offset=0.06)

    tax1.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="",
               fontsize=9, offset=0.01)

    # Credal set near vertex
    center_factual = (85, 10, 5)
    draw_ellipse_on_simplex(ax1, center_factual, sigma_epi=1.8,
                            color=COLORS['credal_confident'], scale=scale)

    # Ground truth marker - at "4" vertex (p1 = 100%)
    gt1 = (98, 1, 1)
    cx1, cy1 = ternary_to_cartesian(gt1, scale)
    ax1.plot(cx1, cy1, '*', color=COLORS['ground_truth'], markersize=12,
             markeredgecolor='black', markeredgewidth=1, zorder=11)

    tax1.clear_matplotlib_ticks()
    ax1.set_aspect('equal')

    # RIGHT: Subjective Question
    ax2 = axes[1]
    figure2, tax2 = ternary.figure(ax=ax2, scale=scale)

    tax2.boundary(linewidth=2.5)
    tax2.gridlines(multiple=20, linewidth=0.5, alpha=0.4)

    # Answer labels for "Is this movie good?"
    # Different labels to clearly distinguish from factual question
    tax2.left_corner_label("Yes", fontsize=12, offset=0.02)
    tax2.right_corner_label("No", fontsize=12, offset=0.02)
    tax2.top_corner_label("Maybe", fontsize=11, offset=0.06)

    tax2.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="",
               fontsize=9, offset=0.01)

    # Credal set in interior (ambiguous)
    center_subjective = (35, 35, 30)
    draw_ellipse_on_simplex(ax2, center_subjective, sigma_epi=1.8,
                            color=COLORS['credal_confident'], scale=scale)

    # Ground truth marker - in interior (ambiguous)
    gt2 = (35, 35, 30)
    cx2, cy2 = ternary_to_cartesian(gt2, scale)
    ax2.plot(cx2, cy2, '*', color=COLORS['ground_truth'], markersize=12,
             markeredgecolor='black', markeredgewidth=1, zorder=11)

    tax2.clear_matplotlib_ticks()
    ax2.set_aspect('equal')

    plt.tight_layout()

    return fig


def create_four_case_clean():
    """Create 2x2 grid showing all four combinations of EU and AU."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    scale = 100

    cases = [
        (0, 0, 'Low EU, Low AU', (85, 10, 5), 1.2, ('4', '3', '5')),
        (0, 1, 'High EU, Low AU', (70, 18, 12), 4.0, ('4', '3', '5')),
        (1, 0, 'Low EU, High AU', (35, 35, 30), 1.2, ('Yes', 'No', 'Meh')),
        (1, 1, 'High EU, High AU', (38, 32, 30), 4.0, ('Yes', 'No', 'Meh')),
    ]

    for row, col, title, center, sigma, labels in cases:
        ax = axes[row, col]
        figure, tax = ternary.figure(ax=ax, scale=scale)

        tax.boundary(linewidth=2)
        tax.gridlines(multiple=25, linewidth=0.5, alpha=0.3)

        # Labels
        tax.left_corner_label(labels[0], fontsize=11, offset=0.02)
        tax.right_corner_label(labels[1], fontsize=11, offset=0.02)
        tax.top_corner_label(labels[2], fontsize=11, offset=0.05)

        # Color based on EU level
        color = COLORS['credal_confident'] if sigma < 2 else COLORS['credal_confused']

        draw_ellipse_on_simplex(ax, center, sigma_epi=sigma, color=color, scale=scale)

        tax.clear_matplotlib_ticks()
        ax.set_aspect('equal')

    plt.tight_layout()

    return fig


if __name__ == "__main__":

    print("Generating clean credal set figures...")

    # Main comparison
    fig1 = create_credal_comparison_clean()
    fig1.savefig('outputs/fig_credal_comparison.pdf', bbox_inches='tight', dpi=300)
    fig1.savefig('outputs/fig_credal_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved: fig_credal_comparison.pdf/png")
    print("  Left: Factual question (4, 3, 5)")
    print("  Right: Subjective question (Yes, No, Maybe)")

    # Four cases
    fig2 = create_four_case_clean()
    fig2.savefig('outputs/fig_credal_four_cases.pdf', bbox_inches='tight', dpi=300)
    fig2.savefig('outputs/fig_credal_four_cases.png', bbox_inches='tight', dpi=300)
    print("Saved: fig_credal_four_cases.pdf/png")

    plt.show()

    print("\nDone!")
    print("\nVerifying vertex labels:")
    print("  Factual question: Left=4, Right=3, Top=5")
    print("  Subjective question: Left=Yes, Right=No, Top=Maybe")
