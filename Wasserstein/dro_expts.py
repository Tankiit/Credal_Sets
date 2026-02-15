"""
DRO (Distributionally Robust Optimization) Experiments - Latent Extraction

This script extracts and saves latent embeddings from datasets for DRO experiments.
Supports multiple extraction methods and datasets with comprehensive caching.

Usage:
    # Extract CEBaB latents with SVD-64
    python dro_expts.py --dataset cebab --method low_rank --rank 64 --svd

    # Extract all datasets with default settings
    python dro_expts.py --dataset all

    # Extract with multi-layer aggregation
    python dro_expts.py --dataset cebab --method multi_layer --layers 0 3 5 -1

    # Extract token-level for analysis
    python dro_expts.py --dataset sst2 --method token_level --layer -1
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from encoder import (
    FrozenDistilBERTEncoder,
    create_encoder,
    extract_latents_from_loader,
)
from dataloader import (
    load_dataset_splits,
    DatasetConfig,
    DATASET_INFO,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_EXTRACTION_CONFIGS = {
    "cebab": {
        "method": "low_rank",
        "rank": 128,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "hatexplain": {
        "method": "low_rank",
        "rank": 128,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "civil_comments": {
        "method": "low_rank",
        "rank": 128,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "goemotions": {
        "method": "low_rank",
        "rank": 128,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "sst2": {
        "method": "low_rank",
        "rank": 64,
        "svd": True,
        "layer": "last",
        "aggregate": "cls",
    },
    "sst5": {
        "method": "low_rank",
        "rank": 64,
        "svd": True,
        "layer": "last",
        "aggregate": "cls",
    },
    "imdb": {
        "method": "low_rank",
        "rank": 64,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "yelp": {
        "method": "low_rank",
        "rank": 64,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "chaosnli": {
        "method": "low_rank",
        "rank": 128,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
    "tid8": {
        "method": "low_rank",
        "rank": 128,
        "svd": True,
        "layer": "last",
        "aggregate": "mean",
    },
}


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_dataset_latents(
    dataset_name: str,
    config: DatasetConfig,
    encoder: FrozenDistilBERTEncoder,
    device: str,
    extraction_config: Dict,
    output_dir: str,
) -> Dict[str, np.ndarray]:
    """
    Extract latents for all splits of a dataset.

    Args:
        dataset_name: Name of dataset
        config: Dataset configuration
        encoder: Frozen encoder
        device: Device to use
        extraction_config: Extraction method configuration
        output_dir: Directory to save results

    Returns:
        Dict with 'train', 'val', 'test' latents and labels
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING LATENTS: {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        dataset_name,
        config=config,
    )

    print(f"\nDataset info:")
    print(f"  Task: {metadata['task']}")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Concepts: {metadata['num_concepts']}")
    print(f"  Train/Val/Test: {metadata['train_size']}/{metadata['val_size']}/{metadata['test_size']}")

    # Prepare extraction kwargs
    extraction_kwargs = {
        "extraction_method": extraction_config.get("method", "cls"),
    }

    # Add method-specific kwargs
    if extraction_config.get("method") == "low_rank":
        extraction_kwargs.update({
            "rank": extraction_config.get("rank", 128),
            "method": "svd" if extraction_config.get("svd") else "pca",
            "layer": extraction_config.get("layer", "last"),
            "aggregate": extraction_config.get("aggregate", "mean"),
        })
    elif extraction_config.get("method") == "multi_layer":
        extraction_kwargs.update({
            "layers": extraction_config.get("layers", [0, 3, 5, -1]),
            "aggregation": extraction_config.get("aggregation", "concat"),
        })
    elif extraction_config.get("method") == "token_level":
        extraction_kwargs.update({
            "layer": extraction_config.get("layer", -1),
        })

    print(f"\nExtraction config:")
    for key, value in extraction_kwargs.items():
        print(f"  {key}: {value}")

    # Extract for each split
    results = {}
    extraction_times = {}

    for split_name, split_loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        print(f"\nExtracting {split_name} split...")
        start_time = time.time()

        latents, labels = extract_latents_from_loader(
            encoder,
            split_loader,
            device=device,
            **extraction_kwargs,
        )

        elapsed = time.time() - start_time
        extraction_times[split_name] = elapsed

        print(f"  Shape: {latents.shape}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {latents.nbytes / (1024**2):.2f} MB")

        results[f"{split_name}_latents"] = latents
        results[f"{split_name}_labels"] = labels

    # Save results
    save_latents(dataset_name, results, metadata, extraction_config, output_dir)

    return results


def save_latents(
    dataset_name: str,
    results: Dict[str, np.ndarray],
    metadata: Dict,
    extraction_config: Dict,
    output_dir: str,
):
    """Save latents and metadata to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename based on config
    method = extraction_config.get("method", "cls")
    if method == "low_rank":
        rank = extraction_config.get("rank", 128)
        svd_pca = "svd" if extraction_config.get("svd") else "pca"
        filename = f"{dataset_name}_{method}_rank{rank}_{svd_pca}"
    elif method == "multi_layer":
        layers = "_".join(map(str, extraction_config.get("layers", [])))
        agg = extraction_config.get("aggregation", "concat")
        filename = f"{dataset_name}_{method}_layers{layers}_{agg}"
    else:
        filename = f"{dataset_name}_{method}"

    # Save latents
    for key, array in results.items():
        save_path = output_path / f"{filename}_{key}.npy"
        np.save(save_path, array)
        print(f"  Saved: {save_path}")

    # Save metadata
    metadata_path = output_path / f"{filename}_metadata.json"
    save_dict = {
        "dataset_name": dataset_name,
        "extraction_config": extraction_config,
        "metadata": metadata,
        "shapes": {k: v.shape for k, v in results.items()},
        "dtypes": {k: str(v.dtype) for k, v in results.items()},
    }

    with open(metadata_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    print(f"  Saved: {metadata_path}")


def load_latents(
    dataset_name: str,
    extraction_config: Dict,
    output_dir: str = "./latent_cache",
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load previously saved latents.

    Returns:
        (results_dict, metadata_dict)
    """
    output_path = Path(output_dir)

    # Generate filename
    method = extraction_config.get("method", "cls")
    if method == "low_rank":
        rank = extraction_config.get("rank", 128)
        svd_pca = "svd" if extraction_config.get("svd") else "pca"
        filename = f"{dataset_name}_{method}_rank{rank}_{svd_pca}"
    elif method == "multi_layer":
        layers = "_".join(map(str, extraction_config.get("layers", [])))
        agg = extraction_config.get("aggregation", "concat")
        filename = f"{dataset_name}_{method}_layers{layers}_{agg}"
    else:
        filename = f"{dataset_name}_{method}"

    # Load metadata
    metadata_path = output_path / f"{filename}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load latents
    results = {}
    for key in metadata["shapes"].keys():
        load_path = output_path / f"{filename}_{key}.npy"
        if not load_path.exists():
            raise FileNotFoundError(f"File not found: {load_path}")
        results[key] = np.load(load_path)

    print(f"Loaded latents from {metadata_path.parent}")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Method: {metadata['extraction_config']}")

    return results, metadata


# =============================================================================
# MULTI-DATASET EXTRACTION
# =============================================================================

def extract_all_datasets(
    datasets: List[str],
    encoder_name: str = "distilbert-base-uncased",
    device: str = None,
    output_dir: str = "./latent_cache",
    use_default_configs: bool = True,
    custom_config: Dict = None,
):
    """
    Extract latents for multiple datasets.

    Args:
        datasets: List of dataset names (or "all")
        encoder_name: Model to use for extraction
        device: Device to use
        output_dir: Where to save latents
        use_default_configs: Use predefined configs per dataset
        custom_config: Override default config
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("BATCH LATENT EXTRACTION FOR DRO EXPERIMENTS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Encoder: {encoder_name}")
    print(f"Output directory: {output_dir}")

    # Check datasets
    if "all" in datasets:
        datasets = list(DATASET_INFO.keys())

    print(f"\nDatasets to process: {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds}")

    # Load encoder
    print(f"\nLoading encoder...")
    encoder, tokenizer = create_encoder(
        model_name=encoder_name,
        freeze=True,
        device=device,
    )

    # Extract each dataset
    summary = []

    for dataset_name in datasets:
        try:
            # Get config
            if use_default_configs and dataset_name in DEFAULT_EXTRACTION_CONFIGS:
                extraction_config = DEFAULT_EXTRACTION_CONFIGS[dataset_name]
            else:
                extraction_config = custom_config or {}

            # Create dataset config
            dataset_info = DATASET_INFO[dataset_name]

            # Determine label type for CEBaB
            label_type = "ternary" if dataset_name == "cebab" else "binary" if dataset_name in ["sst2", "imdb"] else "default"

            # Note: DatasetConfig doesn't take 'dataset' as parameter
            # The dataset is passed separately to load_dataset_splits
            dataset_config = DatasetConfig(
                label_type=label_type,
                batch_size=32,
                max_length=128,
                tokenizer_name=encoder_name,
            )

            # Extract
            results = extract_dataset_latents(
                dataset_name=dataset_name,
                config=dataset_config,
                encoder=encoder,
                device=device,
                extraction_config=extraction_config,
                output_dir=output_dir,
            )

            # Summary
            summary.append({
                "dataset": dataset_name,
                "status": "success",
                "train_dim": results["train_latents"].shape[1],
                "train_size": results["train_latents"].shape[0],
                "method": extraction_config.get("method", "cls"),
            })

        except Exception as e:
            print(f"\n❌ ERROR processing {dataset_name}: {e}")
            summary.append({
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e),
            })

    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<20} {'Status':<10} {'Dim':<10} {'Samples':<10}")
    print("-" * 50)

    for s in summary:
        if s["status"] == "success":
            print(f"{s['dataset']:<20} {s['status']:<10} {s['train_dim']:<10} {s['train_size']:<10}")
        else:
            print(f"{s['dataset']:<20} {s['status']:<10} {'-':<10} {'-':<10}")

    # Save summary
    summary_path = Path(output_dir) / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# =============================================================================
# CACHED DATASET CLASS FOR TRAINING
# =============================================================================

class CachedLatentDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for cached latent features.

    Enables fast training without re-running encoder.
    """

    def __init__(
        self,
        latents: np.ndarray,
        labels: np.ndarray,
        concepts: Optional[np.ndarray] = None,
        is_unknown: Optional[np.ndarray] = None,
    ):
        """
        Args:
            latents: [N, D] latent features
            labels: [N] class labels
            concepts: [N, K] concept labels (optional)
            is_unknown: [N, K] unknown flags (optional)
        """
        self.latents = torch.from_numpy(latents).float()
        self.labels = torch.from_numpy(labels).long()

        if concepts is not None:
            self.concepts = torch.from_numpy(concepts).long()
        else:
            self.concepts = None

        if is_unknown is not None:
            self.is_unknown = torch.from_numpy(is_unknown).float()
        else:
            self.is_unknown = None

        print(f"Created CachedLatentDataset: {len(self)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'features': self.latents[idx],
            'labels': self.labels[idx],
        }

        if self.concepts is not None:
            item['concept_labels'] = self.concepts[idx]
            item['is_unknown'] = self.is_unknown[idx]
        else:
            # Dummy concepts for datasets without them
            item['concept_labels'] = torch.zeros(1, dtype=torch.long)
            item['is_unknown'] = torch.zeros(1, dtype=torch.float)

        return item


def create_cached_dataloaders(
    dataset_name: str,
    extraction_config: Dict,
    output_dir: str = "./latent_cache",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create PyTorch DataLoaders from cached latents.

    Args:
        dataset_name: Name of dataset
        extraction_config: Extraction config to load
        output_dir: Where cached files are stored
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    print(f"\nLoading cached latents for {dataset_name}...")

    # Load latents
    results, metadata = load_latents(dataset_name, extraction_config, output_dir)

    # Create datasets
    train_dataset = CachedLatentDataset(
        results["train_latents"],
        results["train_labels"],
    )

    val_dataset = CachedLatentDataset(
        results["val_latents"],
        results["val_labels"],
    )

    test_dataset = CachedLatentDataset(
        results["test_latents"],
        results["test_labels"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Feature dim: {results['train_latents'].shape[1]}")

    return train_loader, val_loader, test_loader, metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract latent embeddings for DRO experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract CEBaB with default settings (SVD-128)
  python dro_expts.py --dataset cebab

  # Extract SST-2 with lower rank
  python dro_expts.py --dataset sst2 --rank 32

  # Extract with PCA instead of SVD
  python dro_expts.py --dataset cebab --pca --rank 64

  # Extract all datasets
  python dro_expts.py --dataset all

  # Extract with multi-layer aggregation
  python dro_expts.py --dataset cebab --method multi_layer --layers 0 3 5 -1

  # Extract token-level for analysis
  python dro_expts.py --dataset sst2 --method token_level
        """
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (or 'all' for all datasets)",
        choices=list(DATASET_INFO.keys()) + ["all"],
    )

    # Encoder settings
    parser.add_argument(
        "--encoder",
        type=str,
        default="distilbert-base-uncased",
        help="Encoder model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto-detect)",
    )

    # Extraction method
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["cls", "low_rank", "multi_layer", "token_level"],
        help="Extraction method (default: use dataset-specific default)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank for low-rank extraction (default: dataset-specific)",
    )
    parser.add_argument(
        "--svd",
        action="store_true",
        default=None,
        help="Use SVD (default: True for most datasets)",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Use PCA instead of SVD",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer for low-rank (first/last/mean)",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        choices=["cls", "mean", "max"],
        help="Aggregation method (default: dataset-specific)",
    )

    # Multi-layer settings
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to extract (e.g., 0 3 5 -1)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default=None,
        choices=["concat", "mean", "sum"],
        help="Multi-layer aggregation method",
    )

    # Token-level settings
    parser.add_argument(
        "--token_layer",
        type=int,
        default=-1,
        help="Layer for token-level extraction",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./latent_cache",
        help="Directory to save cached latents (default: ./latent_cache)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name (creates subdirectory: output_dir/experiment_name/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for extraction",
    )

    args = parser.parse_args()

    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build output path with experiment name (if provided)
    output_dir = args.output_dir
    if args.experiment_name:
        output_dir = os.path.join(args.output_dir, args.experiment_name)
        print(f"Experiment directory: {output_dir}")

    # Build extraction config
    if args.dataset == "all":
        datasets = ["all"]
        use_default_configs = True
        extraction_config = None
    else:
        datasets = [args.dataset]
        use_default_configs = (args.method is None)

        # Build custom config if method specified
        if args.method is not None:
            extraction_config = {"method": args.method}

            if args.method == "low_rank":
                extraction_config["rank"] = args.rank or 128
                extraction_config["svd"] = not args.pca
                extraction_config["layer"] = args.layer or "last"
                extraction_config["aggregate"] = args.aggregate or "mean"

            elif args.method == "multi_layer":
                extraction_config["layers"] = args.layers or [0, 3, 5, -1]
                extraction_config["aggregation"] = args.aggregation or "concat"

            elif args.method == "token_level":
                extraction_config["layer"] = args.token_layer
        else:
            extraction_config = None

    # Extract
    extract_all_datasets(
        datasets=datasets,
        encoder_name=args.encoder,
        device=device,
        output_dir=output_dir,  # Use modified output_dir with experiment name
        use_default_configs=use_default_configs,
        custom_config=extraction_config,
    )


if __name__ == "__main__":
    main()
