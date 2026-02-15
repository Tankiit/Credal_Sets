"""
ICML 2026 Figure 1: The Impossibility of Decomposition (Clean Version)
"""
import numpy as np
import matplotlib.pyplot as plt
import ternary
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

COLORS = {
    'epistemic': '#66A61E',
    'aleatoric': '#E4862A',
    'baseline': '#CC6666',
    'model_pred': '#9467BD',
    'case1': '#D62728',
    'case2': '#2CA02C',
}

def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log(p + 1e-10))

def kl_divergence(p_star, p):
    p_star = np.array(p_star)
    p = np.array(p)
    mask = p_star > 0
    return np.sum(p_star[mask] * np.log(p_star[mask] / (p[mask] + 1e-10)))

def create_impossibility_figure_v1_clean():
    fig, ax = plt.subplots(figsize=(10, 8))
    scale = 100
    figure, tax = ternary.figure(ax=ax, scale=scale)
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=10, linewidth=0.5, alpha=0.7)
    fontsize = 12
    tax.left_axis_label("$p(c_3)$", fontsize=fontsize, offset=0.14)
    tax.right_axis_label("$p(c_2)$", fontsize=fontsize, offset=0.14)
    tax.bottom_axis_label("$p(c_1)$", fontsize=fontsize, offset=0.02)
    tax.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="%.0f%%",
              fontsize=9, offset=0.02)
    p_model = (35, 40, 25)
    p_star_case1 = (5, 90, 5)
    p_star_case2 = (33, 38, 29)
    tax.scatter([p_model], marker='o', s=200, c=COLORS['model_pred'],
                zorder=10, edgecolors='white', linewidths=2, label='Model $p$')
    tax.scatter([p_star_case1], marker='s', s=150, c=COLORS['case1'],
                zorder=9, edgecolors='white', linewidths=1.5, label='$p^*_1$: Clear truth')
    tax.line(p_model, p_star_case1, linewidth=2, color=COLORS['case1'],
             linestyle='--', alpha=0.7)
    tax.scatter([p_star_case2], marker='^', s=150, c=COLORS['case2'],
                zorder=9, edgecolors='white', linewidths=1.5, label='$p^*_2$: Ambiguous truth')
    tax.line(p_model, p_star_case2, linewidth=2, color=COLORS['case2'],
             linestyle='-', alpha=0.7)
    tax.legend(loc='upper left', fontsize=10)
    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    return fig

def create_maqa_style_figure_clean():
    fig = plt.figure(figsize=(12, 8))
    ax_main = fig.add_subplot(121)
    scale = 100
    figure, tax = ternary.figure(ax=ax_main, scale=scale)
    tax.boundary(linewidth=2)
    tax.gridlines(multiple=20, linewidth=0.5, alpha=0.5)
    fontsize = 11
    tax.left_axis_label("Answer C", fontsize=fontsize, offset=0.16)
    tax.right_axis_label("Answer B", fontsize=fontsize, offset=0.16)
    tax.bottom_axis_label("Answer A", fontsize=fontsize, offset=0.04)
    tax.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="%.0f%%",
              fontsize=9, offset=0.02)
    p_star_factual = (10, 85, 5)
    p_model = (30, 45, 25)
    p_star_ambig = (32, 43, 25)
    tax.scatter([p_star_factual], marker='s', s=180, c=COLORS['case1'],
                edgecolors='white', linewidths=2, zorder=10)
    tax.scatter([p_model], marker='o', s=200, c=COLORS['model_pred'],
                edgecolors='white', linewidths=2, zorder=11)
    tax.line(p_model, p_star_factual, linewidth=2.5, color=COLORS['case1'],
             linestyle='--', alpha=0.8)
    tax.scatter([p_star_ambig], marker='^', s=180, c=COLORS['case2'],
                edgecolors='white', linewidths=2, zorder=10)
    tax.line(p_model, p_star_ambig, linewidth=2.5, color=COLORS['case2'],
             linestyle='-', alpha=0.8)
    p = np.array(p_model) / 100
    p1 = np.array(p_star_factual) / 100
    p2 = np.array(p_star_ambig) / 100
    eu1, au1 = kl_divergence(p1, p), entropy(p1)
    eu2, au2 = kl_divergence(p2, p), entropy(p2)
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['model_pred'], edgecolor='white',
                       label=f'Model $p$ (EU={eu2:.2f}, AU={au2:.2f})'),
        mpatches.Patch(facecolor=COLORS['case1'], edgecolor='white',
                       label=f'$p^*_1$: Factual (EU={eu1:.2f}, AU={au1:.2f})'),
        mpatches.Patch(facecolor=COLORS['case2'], edgecolor='white',
                       label=f'$p^*_2$: Ambiguous (EU={eu2:.2f}, AU={au2:.2f})'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)
    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("Generating clean impossibility figures (no text boxes)...")
    
    fig1 = create_impossibility_figure_v1_clean()
    fig1.savefig('outputs/fig1_impossibility_v1_geometric.pdf', bbox_inches='tight', dpi=300)
    fig1.savefig('outputs/fig1_impossibility_v1_geometric.png', bbox_inches='tight', dpi=300)
    print("  ✓ v1_geometric")
    
    fig4 = create_maqa_style_figure_clean()
    fig4.savefig('outputs/fig1_impossibility_v4_maqa.pdf', bbox_inches='tight', dpi=300)
    fig4.savefig('outputs/fig1_impossibility_v4_maqa.png', bbox_inches='tight', dpi=300)
    print("  ✓ v4_maqa")
    
    print("\n✅ Clean impossibility figures saved to outputs/")
