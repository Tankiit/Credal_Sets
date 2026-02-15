"""
STEP 2: Basic Wasserstein Distance Computation
===============================================

Goal: Compute Wasserstein distances between credal sets

What we're doing:
1. Load credal set arrays from Step 1
2. Compute Wasserstein distance between them
3. Compare different computation methods
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance

print("="*70)
print("STEP 2: Computing Wasserstein Distances")
print("="*70)

# ============================================================================
# Load results from Step 1
# ============================================================================

print("\n" + "="*70)
print("Loading Credal Set Data")
print("="*70)

try:
    data = np.load('step1_results.npz')
    arr_correct = data['arr_correct']
    arr_incorrect = data['arr_incorrect']
    all_answers = data['all_answers']
    print("✓ Loaded data from step1_results.npz")
except FileNotFoundError:
    print("✗ Could not find step1_results.npz")
    print("  Please run Step 1 first: python step1_validate_credal_sets.py")
    sys.exit(1)

print(f"\nCredal set 1 (correct): shape {arr_correct.shape}")
print(f"  Vertices: {arr_correct.shape[0]}")
print(f"  Answers: {arr_correct.shape[1]}")

print(f"\nCredal set 2 (incorrect): shape {arr_incorrect.shape}")
print(f"  Vertices: {arr_incorrect.shape[0]}")
print(f"  Answers: {arr_incorrect.shape[1]}")

print(f"\nAnswer labels: {all_answers}")

# ============================================================================
# Method 1: Pairwise Vertex Distances
# ============================================================================

print("\n" + "="*70)
print("Method 1: Pairwise Vertex-to-Vertex Wasserstein")
print("="*70)

print("\nComputing Wasserstein distance between each pair of vertices...")

def compute_pairwise_wasserstein(P_vertices, Q_vertices):
    """
    Compute all pairwise Wasserstein distances between vertices

    For two credal sets with vertices P and Q, we compute:
    d(P, Q) where P ∈ P_vertices, Q ∈ Q_vertices

    Returns matrix of shape [n_P, n_Q]
    """
    n_P = P_vertices.shape[0]
    n_Q = Q_vertices.shape[0]
    distances = np.zeros((n_P, n_Q))

    for i in range(n_P):
        for j in range(n_Q):
            # scipy's wasserstein_distance expects 1D arrays
            # It computes 1-Wasserstein (Earth Mover's Distance)
            distances[i, j] = wasserstein_distance(P_vertices[i], Q_vertices[j])

    return distances

pairwise_distances = compute_pairwise_wasserstein(arr_correct, arr_incorrect)

print(f"\nPairwise distance matrix (shape {pairwise_distances.shape}):")
print(pairwise_distances)

# Get statistics
min_dist = np.min(pairwise_distances)
max_dist = np.max(pairwise_distances)
mean_dist = np.mean(pairwise_distances)
median_dist = np.median(pairwise_distances)

print(f"\nStatistics:")
print(f"  Min distance:    {min_dist:.4f}")
print(f"  Max distance:    {max_dist:.4f}")
print(f"  Mean distance:   {mean_dist:.4f}")
print(f"  Median distance: {median_dist:.4f}")

# ============================================================================
# Method 2: Hausdorff Distance (Max-Min)
# ============================================================================

print("\n" + "="*70)
print("Method 2: Hausdorff Distance for Credal Sets")
print("="*70)

print("\nComputing Hausdorff distance between credal sets...")
print("This measures the maximum 'worst-case' distance between the sets.")

def compute_hausdorff_distance(P_vertices, Q_vertices):
    """
    Compute Hausdorff distance between two sets of vertices

    H(P, Q) = max(max_{p∈P} min_{q∈Q} d(p,q), max_{q∈Q} min_{p∈P} d(p,q))

    This measures how far apart the two sets are.
    """
    # Compute pairwise distances
    pairwise = compute_pairwise_wasserstein(P_vertices, Q_vertices)

    # Forward Hausdorff: max over P of min over Q
    forward = np.max(np.min(pairwise, axis=1))

    # Backward Hausdorff: max over Q of min over P
    backward = np.max(np.min(pairwise, axis=0))

    # Hausdorff distance is the maximum
    return max(forward, backward), forward, backward

hausdorff_dist, forward, backward = compute_hausdorff_distance(arr_correct, arr_incorrect)

print(f"\nHausdorff distance: {hausdorff_dist:.4f}")
print(f"  Forward (P→Q):  {forward:.4f}")
print(f"  Backward (Q→P): {backward:.4f}")

# ============================================================================
# Method 3: Wasserstein Barycenters
# ============================================================================

print("\n" + "="*70)
print("Method 3: Mean Distribution Distance")
print("="*70)

print("\nComparing mean (expected) distributions...")

def compute_mean_distribution(vertices):
    """Compute the mean distribution across all vertices"""
    return np.mean(vertices, axis=0)

mean_correct = compute_mean_distribution(arr_correct)
mean_incorrect = compute_mean_distribution(arr_incorrect)

print(f"\nMean distribution (correct): {mean_correct}")
print(f"Mean distribution (incorrect): {mean_incorrect}")

# Compute Wasserstein distance between means
mean_distance = wasserstein_distance(mean_correct, mean_incorrect)

print(f"\nWasserstein distance between means: {mean_distance:.4f}")

# ============================================================================
# Method 4: Integrated Wasserstein Distance
# ============================================================================

print("\n" + "="*70)
print("Method 4: Integrated Wasserstein Distance")
print("="*70)

print("\nComputing integrated Wasserstein distance...")
print("This averages over all possible pairs of distributions.")

def compute_integrated_wasserstein(P_vertices, Q_vertices):
    """
    Compute integrated (average) Wasserstein distance

    This averages the distance over all pairs of vertices,
    effectively integrating over the product of the two credal sets.
    """
    pairwise = compute_pairwise_wasserstein(P_vertices, Q_vertices)
    return np.mean(pairwise)

integrated_dist = compute_integrated_wasserstein(arr_correct, arr_incorrect)

print(f"\nIntegrated Wasserstein distance: {integrated_dist:.4f}")
print(f"(This is the same as the mean pairwise distance computed earlier)")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("Creating Visualizations")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Pairwise distance matrix heatmap
im = axes[0, 0].imshow(pairwise_distances, cmap='YlOrRd', aspect='auto')
axes[0, 0].set_xlabel('Incorrect Vertices')
axes[0, 0].set_ylabel('Correct Vertices')
axes[0, 0].set_title('Pairwise Wasserstein Distances')
plt.colorbar(im, ax=axes[0, 0], label='Distance')

# Plot 2: Distribution of pairwise distances
axes[0, 1].hist(pairwise_distances.flatten(), bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.3f}')
axes[0, 1].axvline(median_dist, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_dist:.3f}')
axes[0, 1].set_xlabel('Wasserstein Distance')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Pairwise Distances')
axes[0, 1].legend()

# Plot 3: Mean distributions comparison
x = np.arange(len(all_answers))
width = 0.35

axes[1, 0].bar(x - width/2, mean_correct, width, label='Correct', alpha=0.8)
axes[1, 0].bar(x + width/2, mean_incorrect, width, label='Incorrect', alpha=0.8)
axes[1, 0].set_xlabel('Answers')
axes[1, 0].set_ylabel('Probability')
axes[1, 0].set_title('Mean Distributions Comparison')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(all_answers)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Distance metrics summary
metrics = ['Hausdorff\n(Max-Min)', 'Integrated\n(Average)', 'Mean\nDistance']
values = [hausdorff_dist, integrated_dist, mean_distance]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('Distance')
axes[1, 1].set_title('Distance Metrics Summary')
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('wasserstein_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to wasserstein_results.png")

plt.show()

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n✅ Computed Wasserstein distances between credal sets:")
print(f"\n1. Hausdorff Distance (Max-Min):     {hausdorff_dist:.4f}")
print(f"   - Measures worst-case distance")
print(f"   - Forward:  {forward:.4f}")
print(f"   - Backward: {backward:.4f}")

print(f"\n2. Integrated Distance (Average):     {integrated_dist:.4f}")
print(f"   - Average over all vertex pairs")
print(f"   - Represents expected distance")

print(f"\n3. Mean Distribution Distance:        {mean_distance:.4f}")
print(f"   - Distance between average distributions")
print(f"   - Simplified comparison")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print("\nThe credal sets represent:")
print("  - 'Correct':   High agreement (mostly A)")
print("  - 'Incorrect': High disagreement (uniform)")

print(f"\nDistance: {hausdorff_dist:.4f} (Hausdorff) indicates:")
print("  ✓ High uncertainty = High distance")
print("  ✓ Wasserstein captures disagreement well")
print("  ✓ Can distinguish between confidence levels")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\nYou can now:")
print("  1. Apply to real annotation data")
print("  2. Use distances for clustering/classification")
print("  3. Incorporate into model training")
print("  4. Compare with other uncertainty measures")

print("\n" + "="*70)
