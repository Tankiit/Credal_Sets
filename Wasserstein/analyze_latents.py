"""
Analyze and Visualize Low-Rank Latent Representations

This script provides utilities to:
- Compare different extraction methods
- Visualize latent space structure
- Analyze explained variance
- Compare computational efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

from encoder import FrozenDistilBERTEncoder, extract_latents_from_loader
from dataloader import load_dataset_splits, DatasetConfig


def compare_extraction_methods(
    encoder: FrozenDistilBERTEncoder,
    dataloader,
    device: str,
    methods: List[Dict],
) -> Dict:
    """
    Compare different extraction methods.

    Args:
        encoder: Frozen encoder
        dataloader: Data to extract from
        device: Device to use
        methods: List of dicts with method configurations
            Example: [
                {"name": "CLS", "extraction_method": "cls"},
                {"name": "SVD-64", "extraction_method": "low_rank", "rank": 64, "method": "svd"},
                {"name": "PCA-32", "extraction_method": "low_rank", "rank": 32, "method": "pca"},
            ]

    Returns:
        Dict with results for each method
    """
    results = {}

    for method_config in methods:
        name = method_config.pop("name", "unknown")
        print(f"\nExtracting with: {name}")

        # Time the extraction
        start_time = time.time()

        latents, labels = extract_latents_from_loader(
            encoder,
            dataloader,
            device=device,
            max_batches=50,  # Limit for comparison
            **method_config,
        )

        elapsed = time.time() - start_time

        # Store results
        results[name] = {
            "latents": latents,
            "labels": labels,
            "dim": latents.shape[1],
            "time": elapsed,
            "memory_mb": latents.nbytes / (1024**2),
        }

        print(f"  Dimension: {latents.shape[1]}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {latents.nbytes / (1024**2):.2f} MB")

    return results


def plot_explained_variance(encoder: FrozenDistilBERTEncoder, dataloader, device: str):
    """Plot explained variance for different ranks."""
    print("\n" + "="*80)
    print("EXPLAINED VARIANCE ANALYSIS")
    print("="*80)

    ranks = [16, 32, 64, 128, 256, 512]
    explained_variances = []

    for rank in ranks:
        print(f"\nTesting rank: {rank}")
        latents, _ = extract_latents_from_loader(
            encoder,
            dataloader,
            device=device,
            max_batches=20,
            extraction_method="low_rank",
            rank=rank,
            method="pca",
        )

        if hasattr(encoder, '_pca_components'):
            var = encoder._pca_components['explained_variance']
            explained = var.sum() * 100
            explained_variances.append(explained)
            print(f"  Explained variance: {explained:.2f}%")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, explained_variances, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% variance')
    plt.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95% variance')
    plt.xlabel('Rank (Number of Components)', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.title('Explained Variance vs Rank (PCA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('explained_variance.png', dpi=150, bbox_inches='tight')
    print("\nSaved: explained_variance.png")


def visualize_latent_space(
    latents: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    title: str = "Latent Space Visualization",
    save_path: str = "latent_space.png",
):
    """
    Visualize latent space using dimensionality reduction.

    Args:
        latents: [N, D] latent representations
        labels: [N] class labels
        method: "tsne" or "pca"
        title: Plot title
        save_path: Where to save the plot
    """
    print(f"\nVisualizing with {method.upper()}...")

    # Reduce to 2D
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
    else:
        reducer = PCA(n_components=2, random_state=42)

    latents_2d = reducer.fit_transform(latents)

    # Plot
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=[colors[i]],
            label=f"Class {label}",
            alpha=0.6,
            s=50,
        )

    plt.xlabel(f'{method.upper()} 1', fontsize=12)
    plt.ylabel(f'{method.upper()} 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def compare_compression_ratio(
    results: Dict[str, Dict],
    save_path: str = "compression_comparison.png",
):
    """Plot compression ratio comparison."""
    print("\n" + "="*80)
    print("COMPRESSION ANALYSIS")
    print("="*80)

    names = list(results.keys())
    dims = [results[name]["dim"] for name in names]
    times = [results[name]["time"] for name in names]
    memories = [results[name]["memory_mb"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Dimensions
    axes[0].bar(names, dims, color='steelblue')
    axes[0].set_ylabel('Dimension', fontsize=12)
    axes[0].set_title('Latent Dimension', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    # Time
    axes[1].bar(names, times, color='coral')
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Extraction Time', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    # Memory
    axes[2].bar(names, memories, color='mediumseagreen')
    axes[2].set_ylabel('Memory (MB)', fontsize=12)
    axes[2].set_title('Memory Usage', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Print summary
    print("\nSummary:")
    print(f"{'Method':<15} {'Dim':<10} {'Time':<10} {'Memory':<10} {'Speedup':<10}")
    print("-" * 55)
    baseline_dim = results[names[0]]["dim"]
    baseline_time = results[names[0]]["time"]

    for name in names:
        r = results[name]
        speedup = baseline_time / r["time"]
        print(f"{name:<15} {r['dim']:<10} {r['time']:<10.2f} {r['memory_mb']:<10.2f} {speedup:<10.2f}x")


def analyze_class_separation(
    latents: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
):
    """
    Analyze how well different classes are separated in latent space.

    Uses simple metrics:
    - Within-class variance (lower is better)
    - Between-class variance (higher is better)
    - Separation index (ratio of between/within)
    """
    print("\n" + "="*80)
    print("CLASS SEPARATION ANALYSIS")
    print("="*80)

    unique_labels = np.unique(labels)

    # Compute means
    class_means = {}
    class_variances = {}

    for label in unique_labels:
        mask = labels == label
        class_latents = latents[mask]
        class_means[label] = class_latents.mean(axis=0)
        class_variances[label] = class_latents.var(axis=0).mean()

    # Between-class variance
    all_means = np.stack(list(class_means.values()))
    global_mean = latents.mean(axis=0)
    between_var = np.mean((all_means - global_mean)**2)

    # Average within-class variance
    within_var = np.mean(list(class_variances.values()))

    # Separation index
    separation_index = between_var / (within_var + 1e-10)

    print(f"\nBetween-class variance: {between_var:.4f}")
    print(f"Within-class variance: {within_var:.4f}")
    print(f"Separation index: {separation_index:.4f}")

    if separation_index > 1.0:
        print("✓ Good class separation!")
    else:
        print("⚠ Classes may be overlapping in latent space")

    # Per-class statistics
    print("\nPer-class statistics:")
    print(f"{'Class':<15} {'Mean Norm':<15} {'Variance':<15}")
    print("-" * 45)

    for label in unique_labels:
        name = f"Class {label}" if class_names is None else class_names[label]
        mean_norm = np.linalg.norm(class_means[label])
        var = class_variances[label]
        print(f"{name:<15} {mean_norm:<15.4f} {var:<15.4f}")


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_full_analysis(
    dataset_name: str = "sst2",
    encoder_name: str = "distilbert-base-uncased",
    device: str = None,
):
    """Run comprehensive analysis of latent representations."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print(f"LATENT SPACE ANALYSIS: {dataset_name}")
    print("="*80)
    print(f"Encoder: {encoder_name}")
    print(f"Device: {device}")

    # Load data
    config = DatasetConfig(
        dataset=dataset_name,
        batch_size=32,
        max_length=128,
    )

    train_loader, _, _, tokenizer, metadata = load_dataset_splits(dataset_name, config)

    # Load encoder
    encoder, tokenizer = create_encoder(
        model_name=encoder_name,
        freeze=True,
        device=device,
    )

    # Define methods to compare
    methods = [
        {"name": "CLS (768)", "extraction_method": "cls"},
        {"name": "SVD-128", "extraction_method": "low_rank", "rank": 128, "method": "svd", "layer": "last"},
        {"name": "SVD-64", "extraction_method": "low_rank", "rank": 64, "method": "svd", "layer": "last"},
        {"name": "PCA-32", "extraction_method": "low_rank", "rank": 32, "method": "pca", "layer": "last"},
    ]

    # Compare methods
    results = compare_extraction_methods(
        encoder, train_loader, device, methods
    )

    # Visualizations for each method
    for name, result in results.items():
        print(f"\nAnalyzing: {name}")

        # Class separation
        analyze_class_separation(
            result["latents"],
            result["labels"],
            metadata.get("class_names"),
        )

        # Latent space visualization (TSNE)
        # Sample if too large
        max_samples = 1000
        if len(result["latents"]) > max_samples:
            indices = np.random.choice(len(result["latents"]), max_samples, replace=False)
            latents_sample = result["latents"][indices]
            labels_sample = result["labels"][indices]
        else:
            latents_sample = result["latents"]
            labels_sample = result["labels"]

        visualize_latent_space(
            latents_sample,
            labels_sample,
            method="tsne",
            title=f"{name} - t-SNE Visualization",
            save_path=f"tsne_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
        )

    # Compression comparison
    compare_compression_ratio(results)

    # Explained variance (if PCA used)
    if any("PCA" in name for name in results.keys()):
        plot_explained_variance(encoder, train_loader, device)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze latent representations")
    parser.add_argument("--dataset", type=str, default="sst2",
                       help="Dataset to analyze")
    parser.add_argument("--encoder", type=str, default="distilbert-base-uncased",
                       help="Encoder model")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    run_full_analysis(
        dataset_name=args.dataset,
        encoder_name=args.encoder,
        device=args.device,
    )
