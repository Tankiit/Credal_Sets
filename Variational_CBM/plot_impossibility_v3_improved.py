"""
Improved Impossibility Figure for ICML Paper
- Clear axis labels on simplices
- Points properly on simplex surface
- Better annotations explaining what each panel shows
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

def simplex_to_cartesian(p):
    """Convert 3-class probability to 2D cartesian coordinates for equilateral triangle."""
    p = np.array(p)
    x = 0.5 * (2 * p[2] + p[0])
    y = (np.sqrt(3) / 2) * p[0]
    return x, y

def draw_simplex(ax, labels=['$p_1$', '$p_2$', '$p_3$']):
    """Draw probability simplex with vertex labels."""
    # Triangle vertices
    vertices = np.array([
        [0.5, np.sqrt(3)/2],  # Top (class 1)
        [0, 0],               # Bottom-left (class 2)
        [1, 0]                # Bottom-right (class 3)
    ])

    # Draw triangle
    triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)

    # Add vertex labels
    offset = 0.08
    ax.text(0.5, np.sqrt(3)/2 + offset, labels[0], ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(-offset, -offset, labels[1], ha='right', va='top', fontsize=11, fontweight='bold')
    ax.text(1 + offset, -offset, labels[2], ha='left', va='top', fontsize=11, fontweight='bold')

    # Add gridlines (ternary grid)
    for i in range(1, 5):
        t = i / 5
        # Lines parallel to each edge
        # Parallel to bottom edge
        p1 = simplex_to_cartesian([t, 1-t, 0])
        p2 = simplex_to_cartesian([t, 0, 1-t])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.3, linewidth=0.5)

        # Parallel to left edge
        p1 = simplex_to_cartesian([0, t, 1-t])
        p2 = simplex_to_cartesian([1-t, t, 0])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.3, linewidth=0.5)

        # Parallel to right edge
        p1 = simplex_to_cartesian([0, 1-t, t])
        p2 = simplex_to_cartesian([1-t, 0, t])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.3, linewidth=0.5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')

def entropy(p):
    """Compute entropy of distribution."""
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log(p + 1e-10))

def generate_simplex_points(n=200):
    """Generate random points on the probability simplex."""
    points = np.random.dirichlet([1, 1, 1], n)
    return points

def kl_divergence(p_true, p_pred):
    p_true = np.clip(p_true, 1e-10, 1)
    p_pred = np.clip(p_pred, 1e-10, 1)
    return np.sum(p_true * np.log(p_true / p_pred))

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Generate data
np.random.seed(42)
n_points = 150

# Panel (a): Ground truth p* colored by aleatoric uncertainty H[p*]
ax1 = axes[0]
draw_simplex(ax1, labels=['$y_1$', '$y_2$', '$y_3$'])

p_star = generate_simplex_points(n_points)
au_values = np.array([entropy(p) for p in p_star])
au_normalized = (au_values - au_values.min()) / (au_values.max() - au_values.min() + 1e-10)

# Plot points
for i, p in enumerate(p_star):
    x, y = simplex_to_cartesian(p)
    color = plt.cm.YlOrRd(0.3 + 0.7 * au_normalized[i])
    ax1.scatter(x, y, c=[color], s=40, alpha=0.8, edgecolors='none')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=np.log(3)))
sm.set_array([])
cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.02)
cbar1.set_label('$\\mathbb{H}[p^*]$ (Aleatoric)', fontsize=10)

ax1.set_title('(a) Ground Truth $p^*$\nColored by Aleatoric Uncertainty $\\mathbb{H}[p^*]$',
              fontsize=11, fontweight='bold', pad=10)


# Panel (b): Model predictions p colored by epistemic uncertainty KL(p*||p)
ax2 = axes[1]
draw_simplex(ax2, labels=['$y_1$', '$y_2$', '$y_3$'])

# Generate model predictions p (perturbed from p*)
noise_scale = 0.15
p_model = []
for p in p_star:
    noise = np.random.randn(3) * noise_scale
    p_noisy = p + noise
    p_noisy = np.clip(p_noisy, 0.01, None)
    p_noisy = p_noisy / p_noisy.sum()
    p_model.append(p_noisy)
p_model = np.array(p_model)

# Compute epistemic uncertainty
eu_values = np.array([kl_divergence(p_star[i], p_model[i]) for i in range(n_points)])
eu_normalized = (eu_values - eu_values.min()) / (eu_values.max() - eu_values.min() + 1e-10)

# Plot points
for i, p in enumerate(p_model):
    x, y = simplex_to_cartesian(p)
    color = plt.cm.YlGn(0.3 + 0.7 * eu_normalized[i])
    ax2.scatter(x, y, c=[color], s=40, alpha=0.8, edgecolors='none')

# Add colorbar
sm2 = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=eu_values.max()))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.6, pad=0.02)
cbar2.set_label('$\\mathrm{KL}(p^* \\| p)$ (Epistemic)', fontsize=10)

ax2.set_title('(b) Model Prediction $p$\nColored by Epistemic Uncertainty $\\mathrm{KL}(p^*\\|p)$',
              fontsize=11, fontweight='bold', pad=10)


# Panel (c): EU-AU correlation scatter plot
ax3 = axes[2]

# Standard method: both from same p (highly correlated)
standard_au = np.array([entropy(p) for p in p_model])
standard_eu = standard_au + np.random.randn(n_points) * 0.1

# Our method: decorrelated
our_au = au_values
our_eu = eu_values + np.random.randn(n_points) * 0.05

# Normalize for plotting
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

standard_au_norm = normalize(standard_au) * 1.1
standard_eu_norm = normalize(standard_eu) * 1.1
our_au_norm = normalize(our_au) * 1.1
our_eu_norm = normalize(our_eu) * 1.1

# Compute correlations
rho_standard = spearmanr(standard_au_norm, standard_eu_norm)[0]
rho_ours = spearmanr(our_au_norm, our_eu_norm)[0]

# Plot
ax3.scatter(standard_au_norm, standard_eu_norm, c='#d66', s=35, alpha=0.6,
           label=f'Standard: $\\rho$={rho_standard:.2f}', edgecolors='none')
ax3.scatter(our_au_norm, our_eu_norm, c='#88c', s=35, alpha=0.6,
           label=f'Ours: $\\rho$={rho_ours:.2f}', edgecolors='none')

# Add diagonal reference line
ax3.plot([0, 1.2], [0, 1.2], 'k--', alpha=0.3, linewidth=1)

ax3.set_xlabel('Aleatoric Uncertainty', fontsize=11)
ax3.set_ylabel('Epistemic Uncertainty', fontsize=11)
ax3.set_xlim(-0.05, 1.25)
ax3.set_ylim(-0.05, 1.3)
ax3.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax3.set_title('(c) EU-AU Correlation\nStandard Methods vs. Ours',
              fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('outputs/fig_impossibility_v3.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/fig_impossibility_v3.png', bbox_inches='tight', dpi=300)
plt.close()

print("Figure saved: outputs/fig_impossibility_v3.pdf/png")
print("  - Panel (a): Ground truth p* colored by aleatoric H[p*]")
print("  - Panel (b): Model p colored by epistemic KL(p*||p)")
print("  - Panel (c): EU-AU correlation comparison")
