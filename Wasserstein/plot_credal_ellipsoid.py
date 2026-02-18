# ═══════════════════════════════════════════════════════════════
# 3D CREDAL ELLIPSOID — DRO EQUIVALENCE FIGURE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("GENERATING 3D CREDAL ELLIPSOID FIGURE")
print("="*70)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


def ellipsoid_surface(mu, sigmas, n_points=40):
    """Generate ellipsoid surface points."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = mu[0] + sigmas[0] * np.outer(np.cos(u), np.sin(v))
    y = mu[1] + sigmas[1] * np.outer(np.sin(u), np.sin(v))
    z = mu[2] + sigmas[2] * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def ellipse_2d(mu, sigmas, n_points=100):
    """Generate 2D ellipse points."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = mu[0] + sigmas[0] * np.cos(theta)
    y = mu[1] + sigmas[1] * np.sin(theta)
    return x, y


# ── Parameters ────────────────────────────────────────────────
# Center of credal set (concept probabilities for 3 concepts)
mu = np.array([0.55, 0.30, 0.15])  # neg, pos, unk — sum to 1

# Credal widths per concept (σ_k — semi-axis lengths)
sigmas = np.array([0.12, 0.18, 0.06])  # service most uncertain

# Inner contour (smaller σ)
sigmas_inner = sigmas * 0.5

# Height of ellipsoid above projection plane
z_floor = -0.15


# ═══════════════════════════════════════════════════════════════
# SINGLE PANEL: 3D Ellipsoid with DRO annotations
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ── Grid on the floor plane ───────────────────────────────────
grid_range = np.linspace(0, 1, 11)
for g in grid_range:
    ax.plot([g, g], [0, 1], [z_floor, z_floor],
            color='#bbbbbb', alpha=0.4, linewidth=0.5)
    ax.plot([0, 1], [g, g], [z_floor, z_floor],
            color='#bbbbbb', alpha=0.4, linewidth=0.5)

# ── Simplex triangle on floor ─────────────────────────────────
ax.plot([0, 1, 0, 0], [0, 0, 1, 0], [z_floor]*4,
        color='black', linewidth=1.5)

# ── Simplex labels ─────────────────────────────────────────────
ax.text(-0.05, -0.05, z_floor, 'neg', color='#222222', fontsize=11, fontweight='bold')
ax.text(-0.05, 1.05, z_floor, 'pos', color='#222222', fontsize=11, fontweight='bold')
ax.text(-0.05, -0.05, z_floor, 'unk', color='#222222', fontsize=11, fontweight='bold')

# ── 3D Wireframe Ellipsoid (RED) ──────────────────────────────
X, Y, Z = ellipsoid_surface(mu, sigmas, n_points=30)
# Shift Z so ellipsoid sits above floor
Z_shifted = Z * 1.5 + 0.15  # scale and shift upward

ax.plot_wireframe(X, Y, Z_shifted, color='#B30000', alpha=0.7,
                  linewidth=1.0, rstride=2, cstride=2)

# ── Projection: Outer ellipse (CYAN) on floor ─────────────────
ex, ey = ellipse_2d(mu[:2], sigmas[:2], n_points=100)
ax.plot(ex, ey, np.full_like(ex, z_floor),
        color='#006666', linewidth=3.0, alpha=1.0)

# ── Projection: Inner ellipse (BLUE) on floor ─────────────────
ex_in, ey_in = ellipse_2d(mu[:2], sigmas_inner[:2], n_points=100)
ax.plot(ex_in, ey_in, np.full_like(ex_in, z_floor),
        color='#111188', linewidth=2.0, alpha=0.6, linestyle='--')

# ── Center point (mean) on floor and at ellipsoid center ───────
ax.scatter(mu[0], mu[1], z_floor, color='#333333', s=60, zorder=10,
           edgecolors='#008B8B', linewidths=1.5)
ax.scatter(mu[0], mu[1], 0.15, color='#333333', s=40, zorder=10)

# ── Vertical connection lines (center to ellipsoid) ───────────
ax.plot([mu[0], mu[0]], [mu[1], mu[1]], [z_floor, 0.15],
        color='#888888', alpha=0.4, linewidth=1, linestyle='--')

# ── ε radius arrow on floor ───────────────────────────────────
eps_end_x = mu[0] + sigmas[0]
eps_end_y = mu[1]
ax.plot([mu[0], eps_end_x], [mu[1], eps_end_y], [z_floor, z_floor],
        color='#D45500', linewidth=3.0, alpha=1.0)
# Arrowhead (simple)
ax.scatter(eps_end_x, eps_end_y, z_floor, color='#D45500', s=40,
           marker='>', zorder=10)

# ── Horizontal slice lines (showing simplex planes) ───────────
for level in np.linspace(0, 0.3, 4):
    ax.plot([0, 1], [0, 0], [level]*2, color='#cccccc', alpha=0.15, linewidth=0.5)
    ax.plot([0, 0], [0, 1], [level]*2, color='#cccccc', alpha=0.15, linewidth=0.5)
    ax.plot([0, 1], [1-level, 1-level], [level]*2, color='#cccccc', alpha=0.15, linewidth=0.5)

# ── Annotations (kept) ────────────────────────────────────────
ax.text2D(0.50, 0.92,
          'Credal Ellipsoid C(x) ≡ DRO Ambiguity Set B_ε',
          transform=ax.transAxes, fontsize=14, fontweight='bold',
          color='black', ha='center')

# Sigma annotations
ax.text2D(0.15, 0.18,
          'σ1 (neg)',
          transform=ax.transAxes, fontsize=9, color='#006666')
ax.text2D(0.42, 0.13,
          'σ2 (pos) — widest: most uncertain',
          transform=ax.transAxes, fontsize=9, color='#006666')

# ε arrow label
ax.text2D(0.55, 0.22,
          'ε',
          transform=ax.transAxes, fontsize=14, fontweight='bold',
          color='#D45500')

# Legend
ax.text2D(0.02, 0.92, '■', transform=ax.transAxes, fontsize=12, color='#B30000')
ax.text2D(0.05, 0.92, 'Credal ellipsoid (3D)', transform=ax.transAxes,
          fontsize=9, color='#B30000')
ax.text2D(0.02, 0.88, '■', transform=ax.transAxes, fontsize=12, color='#006666')
ax.text2D(0.05, 0.88, 'Outer projection (ε boundary)', transform=ax.transAxes,
          fontsize=9, color='#006666')
ax.text2D(0.02, 0.84, '■', transform=ax.transAxes, fontsize=12, color='#111188')
ax.text2D(0.05, 0.84, 'Inner contour (ε/2)', transform=ax.transAxes,
          fontsize=9, color='#111188')
ax.text2D(0.02, 0.80, '—', transform=ax.transAxes, fontsize=12, color='#D45500')
ax.text2D(0.05, 0.80, 'Robustness radius ε(x)', transform=ax.transAxes,
          fontsize=9, color='#D45500')

# ── View angle ────────────────────────────────────────────────
ax.view_init(elev=28, azim=-55)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(z_floor, 0.45)

# Hide axes for clean look
ax.set_axis_off()

plt.tight_layout()

# Save
output_path = 'figure_credal_3d_dro'
for ext in ['pdf', 'png']:
    path = f'{output_path}.{ext}'
    plt.savefig(path, bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print(f"✅ Saved: {path}")

plt.show()
print("="*70)
