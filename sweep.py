#!/usr/bin/env python3
"""
CREDENCE Hyperparameter Sweep Script
=====================================

Systematically test different models, datasets, and hyperparameters.
Outputs a summary table for easy comparison.

Usage:
    # Quick test (1 config per dataset)
    python sweep.py --mode quick
    
    # Full sweep (all combinations)
    python sweep.py --mode full
    
    # Single model across datasets
    python sweep.py --model roberta-base --datasets cebab hatexplain sst5
    
    # Single dataset across models
    python sweep.py --dataset cebab --models distilbert-base-uncased roberta-base
    
    # Custom sweep
    python sweep.py --models roberta-base --datasets cebab --lrs 1e-4 5e-5 --epochs 20 40
"""

import os
import json
import argparse
import itertools
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import from your modules
from dataloader import load_dataset_splits, DatasetConfig, DATASET_INFO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default sweep configurations
# 
# Available models (from credence.py MODEL_REGISTRY):
#   Encoder models: distilbert-base-uncased, roberta-base, roberta-large,
#                   microsoft/deberta-v3-base, microsoft/deberta-v3-large
#   LLM models (require LoRA): microsoft/phi-3-mini-4k-instruct,
#                               mistralai/Mistral-7B-v0.1, meta-llama/Llama-3.1-8B
#
# Available datasets (from dataloader.py DATASET_INFO):
#   Sentiment: cebab (concepts), sst2, sst5, imdb, yelp, amazon
#   Toxicity: hatexplain (concepts), civil_comments (concepts)
#   Emotion: goemotions (concepts)
#   NLI: mnli, snli
#   Topic: ag_news

MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "microsoft/deberta-v3-base",
    # "microsoft/deberta-v3-large",  # Uncomment if you have GPU memory
    # LLM models require special setup (LoRA), uncomment if needed:
    # "microsoft/phi-3-mini-4k-instruct",
    # "mistralai/Mistral-7B-v0.1",
    # "meta-llama/Llama-3.1-8B",
]

DATASETS = [
    # Sentiment with concepts
    "cebab",
    # Sentiment without concepts
    "sst2",
    "sst5",
    # Toxicity with concepts
    "hatexplain",
    "civil_comments",
    # Emotion with concepts
    "goemotions",
    # NLI
    # "mnli",
    # "snli",
    # Topic
    # "ag_news",
]

LEARNING_RATES = [1e-4, 5e-5, 2e-5]
N_HEADS = [3, 5]
BATCH_SIZES = [32, 64, 128, 256]
EPOCHS_OPTIONS = [50]

# Quick mode: just test defaults
QUICK_CONFIG = {
    "models": ["distilbert-base-uncased"],
    "datasets": ["cebab", "hatexplain", "sst2"],  # Mix of concept and non-concept datasets
    "lrs": [1e-4],
    "n_heads": [5],
    "batch_sizes": [32],
    "epochs": [50],
}

# Batch size focused sweep
BATCH_SIZE_SWEEP = {
    "models": ["distilbert-base-uncased"],
    "datasets": ["cebab"],
    "lrs": [1e-4, 5e-5],  # LR often needs adjusting with batch size
    "n_heads": [5],
    "batch_sizes": [32, 64, 128, 256],
    "epochs": [50],
}


@dataclass
class SweepConfig:
    """Configuration for a single experiment in the sweep."""
    model: str
    dataset: str
    lr: float
    n_heads: int
    batch_size: int
    epochs: int
    label_type: str = "ternary"
    max_length: int = 128
    concept_weight: float = 1.0
    aleatoric_weight: float = 0.5
    seed: int = 42
    
    def to_name(self) -> str:
        """Generate a unique name for this config."""
        model_short = self.model.split("/")[-1].split("-")[0]
        return f"{model_short}_{self.dataset}_lr{self.lr}_h{self.n_heads}_e{self.epochs}"


# =============================================================================
# MODEL COMPONENTS (Simplified from credence.py)
# =============================================================================

class ConceptHead(nn.Module):
    def __init__(self, input_dim: int, n_concepts: int, dropout: float = 0.1, pooling: str = "cls"):
        super().__init__()
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_concepts),
        )
    
    def pool(self, hidden_states, attention_mask):
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    
    def forward(self, hidden_states, attention_mask):
        x = self.pool(hidden_states, attention_mask)
        logits = self.net(x)
        return logits, torch.sigmoid(logits)


class AleatoricHead(nn.Module):
    def __init__(self, input_dim: int, n_concepts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_concepts),
        )
    
    def forward(self, hidden_states):
        return torch.sigmoid(self.net(hidden_states[:, 0, :]))


class CREDENCE(nn.Module):
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int, n_heads: int = 5):
        super().__init__()
        self.n_heads = n_heads
        self.n_concepts = n_concepts
        
        # Diverse heads
        dropouts = [0.05, 0.10, 0.15, 0.20, 0.25][:n_heads]
        poolings = ["cls", "cls", "cls", "mean", "mean"][:n_heads]
        
        self.heads = nn.ModuleList([
            ConceptHead(input_dim, n_concepts, dropouts[i], poolings[i])
            for i in range(n_heads)
        ])
        
        self.aleatoric_head = AleatoricHead(input_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)
    
    def forward(self, hidden_states, attention_mask):
        all_probs = []
        all_logits = []
        
        for head in self.heads:
            logits, probs = head(hidden_states, attention_mask)
            all_logits.append(logits)
            all_probs.append(probs)
        
        probs_stack = torch.stack(all_probs, dim=-1)
        concept_probs = probs_stack.mean(dim=-1)
        disagreement = probs_stack.var(dim=-1)
        ambiguity = self.aleatoric_head(hidden_states)
        
        logits = self.classifier(concept_probs)
        
        return {
            "logits": logits,
            "concept_probs": concept_probs,
            "disagreement": disagreement,
            "ambiguity": ambiguity,
            "head_logits": all_logits,
        }
    
    def compute_loss(self, outputs, labels, concepts, is_unknown, concept_weight=1.0, aleatoric_weight=0.5):
        device = labels.device
        
        task_loss = nn.functional.cross_entropy(outputs["logits"], labels)
        
        concept_targets = concepts.float() / 2.0
        concept_loss = sum(
            nn.functional.binary_cross_entropy_with_logits(logits, concept_targets)
            for logits in outputs["head_logits"]
        ) / self.n_heads
        
        aleatoric_loss = nn.functional.binary_cross_entropy(outputs["ambiguity"], is_unknown)
        
        total = task_loss + concept_weight * concept_loss + aleatoric_weight * aleatoric_loss
        
        return total, {"task": task_loss.item(), "concept": concept_loss.item(), "aleatoric": aleatoric_loss.item()}


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(config: SweepConfig, device: str, verbose: bool = False) -> Dict[str, Any]:
    """Run a single experiment and return metrics."""
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Determine label type based on dataset
    if config.dataset == "cebab":
        label_type = "ternary"
    else:
        label_type = "default"
    
    # Load data
    dataset_config = DatasetConfig(
        label_type=label_type,
        max_length=config.max_length,
        tokenizer_name=config.model,
        batch_size=config.batch_size,
    )
    
    try:
        train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
            config.dataset, config=dataset_config
        )
    except Exception as e:
        return {"error": str(e), "config": asdict(config)}
    
    # Load encoder
    encoder = AutoModel.from_pretrained(config.model)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder = encoder.to(device)
    encoder.eval()
    
    # Get hidden size
    with torch.no_grad():
        dummy = next(iter(train_loader))
        out = encoder(dummy['input_ids'][:1].to(device), attention_mask=dummy['attention_mask'][:1].to(device))
        hidden_size = out.last_hidden_state.shape[-1]
    
    # Create model
    model = CREDENCE(
        input_dim=hidden_size,
        n_concepts=metadata['num_concepts'],
        n_classes=metadata['num_classes'],
        n_heads=config.n_heads,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # Training loop
    best_val_acc = 0.0
    best_state = None
    history = []
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concepts = batch['concept_labels'].to(device)
            is_unknown = (concepts == 1).float()
            
            with torch.no_grad():
                hidden = encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            
            outputs = model(hidden, attention_mask)
            loss, _ = model.compute_loss(outputs, labels, concepts, is_unknown,
                                         config.concept_weight, config.aleatoric_weight)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        val_acc, val_rho = evaluate_quick(model, encoder, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        history.append({"epoch": epoch + 1, "loss": epoch_loss / n_batches, "val_acc": val_acc, "val_rho": val_rho})
        
        if verbose:
            print(f"  Epoch {epoch+1}/{config.epochs}: loss={epoch_loss/n_batches:.4f}, val_acc={val_acc:.4f}, val_rho={val_rho:.4f}")
    
    # Load best and test
    if best_state:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate_full(model, encoder, test_loader, device)
    
    return {
        "config": asdict(config),
        "metadata": {
            "num_classes": metadata['num_classes'],
            "num_concepts": metadata['num_concepts'],
            "train_size": metadata['train_size'],
            "test_size": metadata['test_size'],
        },
        "best_val_acc": best_val_acc,
        "test": test_metrics,
        "history": history,
    }


def evaluate_quick(model, encoder, loader, device) -> tuple:
    """Quick evaluation returning accuracy and rho."""
    model.eval()
    
    all_preds, all_labels, all_disagree = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            hidden = encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            outputs = model(hidden, attention_mask)
            
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_disagree.extend(outputs["disagreement"].mean(dim=-1).cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    disagree = np.array(all_disagree)
    
    acc = (preds == labels).mean()
    errors = (preds != labels).astype(float)
    
    if errors.std() > 0 and disagree.std() > 0:
        rho, _ = stats.spearmanr(disagree, errors)
    else:
        rho = 0.0
    
    return acc, rho


def evaluate_full(model, encoder, loader, device) -> Dict[str, float]:
    """Full evaluation with all metrics."""
    model.eval()
    
    all_preds, all_labels, all_disagree, all_ambig, all_is_unknown = [], [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concepts = batch['concept_labels']
            
            is_unknown = (concepts == 1).float().mean(dim=-1)
            
            hidden = encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            outputs = model(hidden, attention_mask)
            
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_disagree.extend(outputs["disagreement"].mean(dim=-1).cpu().numpy())
            all_ambig.extend(outputs["ambiguity"].mean(dim=-1).cpu().numpy())
            all_is_unknown.extend(is_unknown.numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    disagree = np.array(all_disagree)
    ambig = np.array(all_ambig)
    is_unknown = np.array(all_is_unknown)
    
    correct = (preds == labels)
    errors = ~correct
    
    acc = correct.mean()
    
    # Epistemic-error correlation
    if errors.astype(float).std() > 0 and disagree.std() > 0:
        rho_epi, p_epi = stats.spearmanr(disagree, errors.astype(float))
    else:
        rho_epi, p_epi = 0.0, 1.0
    
    # Aleatoric-unknown correlation
    if is_unknown.std() > 0 and ambig.std() > 0:
        rho_ale, p_ale = stats.spearmanr(ambig, is_unknown)
    else:
        rho_ale, p_ale = 0.0, 1.0
    
    # Disagree ratio
    if errors.sum() > 0 and correct.sum() > 0:
        ratio = disagree[errors].mean() / (disagree[correct].mean() + 1e-8)
    else:
        ratio = 1.0
    
    # Error detection efficiency
    sorted_idx = np.argsort(disagree)[::-1]
    sorted_errors = errors[sorted_idx].astype(float)
    if errors.sum() > 0:
        cumsum = np.cumsum(sorted_errors)
        pct_reviewed = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        pct_found = cumsum / errors.sum() * 100
        idx_50 = np.searchsorted(pct_found, 50)
        catch_50 = pct_reviewed[idx_50] if idx_50 < len(pct_reviewed) else 100
    else:
        catch_50 = 50.0
    
    return {
        "accuracy": float(acc),
        "rho_epi": float(rho_epi),
        "p_epi": float(p_epi),
        "rho_ale": float(rho_ale),
        "p_ale": float(p_ale),
        "disagree_ratio": float(ratio),
        "catch_50_pct": float(catch_50),
        "epi_mean": float(disagree.mean()),
        "epi_std": float(disagree.std()),
        "ale_mean": float(ambig.mean()),
        "ale_std": float(ambig.std()),
    }


# =============================================================================
# SWEEP RUNNER
# =============================================================================

def generate_configs(
    models: List[str],
    datasets: List[str],
    lrs: List[float],
    n_heads: List[int],
    batch_sizes: List[int],
    epochs: List[int],
) -> List[SweepConfig]:
    """Generate all combinations of configs."""
    configs = []
    
    for model, dataset, lr, heads, bs, ep in itertools.product(
        models, datasets, lrs, n_heads, batch_sizes, epochs
    ):
        configs.append(SweepConfig(
            model=model,
            dataset=dataset,
            lr=lr,
            n_heads=heads,
            batch_size=bs,
            epochs=ep,
        ))
    
    return configs


def run_sweep(
    configs: List[SweepConfig],
    output_dir: str = "./sweep_results",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run all experiments and collect results."""
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"CREDENCE Hyperparameter Sweep")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Total configs: {len(configs)}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.to_name()}")
        print(f"  Model: {config.model}")
        print(f"  Dataset: {config.dataset}")
        print(f"  LR: {config.lr}, Heads: {config.n_heads}, Epochs: {config.epochs}")
        
        try:
            result = train_and_evaluate(config, device, verbose=verbose)
            
            if "error" not in result:
                test = result["test"]
                print(f"  ✓ Acc: {test['accuracy']*100:.1f}%, ρ_epi: {test['rho_epi']:.3f}, ρ_ale: {test['rho_ale']:.3f}")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
            
            # Save intermediate results
            with open(f"{output_dir}/results_partial.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append({"config": asdict(config), "error": str(e)})
    
    return results


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a summary table of results."""
    
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    
    # Header
    print(f"{'Model':<25} {'Dataset':<12} {'LR':<8} {'H':<3} {'Ep':<4} "
          f"{'Acc%':<7} {'ρ_epi':<7} {'ρ_ale':<7} {'Ratio':<6} {'Catch50':<8}")
    print("-" * 100)
    
    # Sort by accuracy
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["test"]["accuracy"], reverse=True)
    
    for r in valid_results:
        c = r["config"]
        t = r["test"]
        
        model_short = c["model"].split("/")[-1][:24]
        
        print(f"{model_short:<25} {c['dataset']:<12} {c['lr']:<8.0e} {c['n_heads']:<3} {c['epochs']:<4} "
              f"{t['accuracy']*100:<7.1f} {t['rho_epi']:<7.3f} {t['rho_ale']:<7.3f} "
              f"{t['disagree_ratio']:<6.2f} {t['catch_50_pct']:<8.1f}")
    
    # Print errors
    error_results = [r for r in results if "error" in r]
    if error_results:
        print(f"\n{'='*50}")
        print(f"ERRORS ({len(error_results)})")
        print(f"{'='*50}")
        for r in error_results:
            c = r["config"]
            print(f"  {c['model']}/{c['dataset']}: {r['error'][:50]}")
    
    # Best config per dataset
    print(f"\n{'='*100}")
    print("BEST CONFIG PER DATASET")
    print(f"{'='*100}")
    
    datasets = set(r["config"]["dataset"] for r in valid_results)
    for ds in sorted(datasets):
        ds_results = [r for r in valid_results if r["config"]["dataset"] == ds]
        if ds_results:
            best = max(ds_results, key=lambda x: x["test"]["accuracy"])
            c = best["config"]
            t = best["test"]
            print(f"\n  {ds}:")
            print(f"    Model: {c['model']}")
            print(f"    LR: {c['lr']}, Heads: {c['n_heads']}, Epochs: {c['epochs']}")
            print(f"    Accuracy: {t['accuracy']*100:.2f}%")
            print(f"    ρ(epi, error): {t['rho_epi']:.4f}")
            print(f"    ρ(ale, unknown): {t['rho_ale']:.4f}")


def save_results_csv(results: List[Dict[str, Any]], filepath: str):
    """Save results to CSV for easy viewing."""
    
    valid_results = [r for r in results if "error" not in r]
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "model", "dataset", "lr", "n_heads", "batch_size", "epochs",
            "accuracy", "rho_epi", "p_epi", "rho_ale", "p_ale",
            "disagree_ratio", "catch_50_pct", "epi_mean", "ale_mean"
        ])
        
        for r in valid_results:
            c = r["config"]
            t = r["test"]
            
            writer.writerow([
                c["model"], c["dataset"], c["lr"], c["n_heads"], c["batch_size"], c["epochs"],
                f"{t['accuracy']:.4f}", f"{t['rho_epi']:.4f}", f"{t['p_epi']:.2e}",
                f"{t['rho_ale']:.4f}", f"{t['p_ale']:.2e}",
                f"{t['disagree_ratio']:.4f}", f"{t['catch_50_pct']:.1f}",
                f"{t['epi_mean']:.6f}", f"{t['ale_mean']:.4f}"
            ])
    
    print(f"\nSaved CSV: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CREDENCE Hyperparameter Sweep")
    
    # Mode
    parser.add_argument("--mode", type=str, default="quick",
                       choices=["quick", "full", "custom", "batch_size", "lr", "heads"],
                       help="Sweep mode: quick, full, batch_size (focused), lr (focused), heads (focused)")
    
    # Custom options
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to test (e.g., distilbert-base-uncased roberta-base)")
    parser.add_argument("--datasets", nargs="+", default=None,
                       help="Datasets to test (e.g., cebab hatexplain)")
    parser.add_argument("--lrs", nargs="+", type=float, default=None,
                       help="Learning rates (e.g., 1e-4 5e-5)")
    parser.add_argument("--n_heads", nargs="+", type=int, default=None,
                       help="Number of ensemble heads (e.g., 3 5)")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=None,
                       help="Batch sizes (e.g., 8 16 32 64)")
    parser.add_argument("--epochs", nargs="+", type=int, default=None,
                       help="Epochs (e.g., 10 20 40)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./sweep_results",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-epoch progress")
    
    args = parser.parse_args()
    
    # Determine configs based on mode
    if args.mode == "quick":
        cfg = QUICK_CONFIG
        models = args.models or cfg["models"]
        datasets = args.datasets or cfg["datasets"]
        lrs = args.lrs or cfg["lrs"]
        n_heads = args.n_heads or cfg["n_heads"]
        batch_sizes = args.batch_sizes or cfg["batch_sizes"]
        epochs = args.epochs or cfg["epochs"]
    
    elif args.mode == "batch_size":
        # Focused batch size sweep
        cfg = BATCH_SIZE_SWEEP
        models = args.models or cfg["models"]
        datasets = args.datasets or cfg["datasets"]
        lrs = args.lrs or cfg["lrs"]
        n_heads = args.n_heads or cfg["n_heads"]
        batch_sizes = args.batch_sizes or cfg["batch_sizes"]
        epochs = args.epochs or cfg["epochs"]
        print("\nBatch Size Focused Sweep")
        print("   Testing how batch size affects uncertainty calibration...")
    
    elif args.mode == "lr":
        # Focused LR sweep
        models = args.models or ["distilbert-base-uncased"]
        datasets = args.datasets or ["cebab"]
        lrs = args.lrs or [1e-3, 5e-4, 1e-4, 5e-5, 2e-5, 1e-5]
        n_heads = args.n_heads or [5]
        batch_sizes = args.batch_sizes or [32]
        epochs = args.epochs or [50]
        print("\nLearning Rate Focused Sweep")
    
    elif args.mode == "heads":
        # Focused heads sweep
        models = args.models or ["distilbert-base-uncased"]
        datasets = args.datasets or ["cebab"]
        lrs = args.lrs or [1e-4]
        n_heads = args.n_heads or [1, 3, 5, 7, 10]
        batch_sizes = args.batch_sizes or [32]
        epochs = args.epochs or [50]
        print("\nEnsemble Heads Focused Sweep")
        print("   Testing how number of heads affects epistemic uncertainty...")
        
    elif args.mode == "full":
        models = args.models or MODELS
        datasets = args.datasets or DATASETS
        lrs = args.lrs or LEARNING_RATES
        n_heads = args.n_heads or N_HEADS
        batch_sizes = args.batch_sizes or BATCH_SIZES
        epochs = args.epochs or EPOCHS_OPTIONS
        
    else:  # custom
        models = args.models or ["distilbert-base-uncased"]
        datasets = args.datasets or ["cebab"]
        lrs = args.lrs or [1e-4]
        n_heads = args.n_heads or [5]
        batch_sizes = args.batch_sizes or [32]
        epochs = args.epochs or [50]
    
    # Generate configs
    configs = generate_configs(models, datasets, lrs, n_heads, batch_sizes, epochs)
    
    print(f"\nGenerated {len(configs)} configurations")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  LRs: {lrs}")
    print(f"  Heads: {n_heads}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Epochs: {epochs}")
    
    # Estimate time
    est_time_per_config = 5  # minutes (rough estimate)
    est_total = len(configs) * est_time_per_config
    print(f"\n  Estimated time: ~{est_total} minutes ({est_total/60:.1f} hours)")
    
    # Run sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{timestamp}_{args.mode}"
    
    results = run_sweep(configs, output_dir, verbose=args.verbose)
    
    # Save results
    with open(f"{output_dir}/results_full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    save_results_csv(results, f"{output_dir}/results.csv")
    
    # Print summary
    print_summary_table(results)
    
    # Print focused analysis based on mode
    if args.mode == "batch_size":
        print_batch_size_analysis(results)
    elif args.mode == "lr":
        print_lr_analysis(results)
    elif args.mode == "heads":
        print_heads_analysis(results)
    
    print(f"\n{'='*70}")
    print(f"Sweep complete! Results saved to: {output_dir}")
    print(f"{'='*70}")


def print_batch_size_analysis(results: List[Dict[str, Any]]):
    """Analyze how batch size affects metrics."""
    
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    
    print(f"\n{'='*70}")
    print("BATCH SIZE ANALYSIS")
    print(f"{'='*70}")
    
    # Group by batch size
    by_bs = defaultdict(list)
    for r in valid:
        bs = r["config"]["batch_size"]
        by_bs[bs].append(r)
    
    print(f"\n{'Batch':<8} {'Acc%':<8} {'ρ_epi':<8} {'ρ_ale':<8} {'Epi_mean':<10} {'Epi_std':<10}")
    print("-" * 60)
    
    for bs in sorted(by_bs.keys()):
        runs = by_bs[bs]
        accs = [r["test"]["accuracy"] for r in runs]
        rho_epis = [r["test"]["rho_epi"] for r in runs]
        rho_ales = [r["test"]["rho_ale"] for r in runs]
        epi_means = [r["test"]["epi_mean"] for r in runs]
        epi_stds = [r["test"]["epi_std"] for r in runs]
        
        print(f"{bs:<8} {np.mean(accs)*100:<8.1f} {np.mean(rho_epis):<8.3f} "
              f"{np.mean(rho_ales):<8.3f} {np.mean(epi_means):<10.6f} {np.mean(epi_stds):<10.6f}")
    
    print(f"\nInsights:")
    print(f"   - Smaller batch sizes (32-64) often give higher epistemic variance")
    print(f"   - Larger batch sizes (128-256) may need higher LR for same convergence")
    print(f"   - Watch for ρ_epi: higher is better for error detection")


def print_lr_analysis(results: List[Dict[str, Any]]):
    """Analyze how learning rate affects metrics."""
    
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    
    print(f"\n{'='*70}")
    print("LEARNING RATE ANALYSIS")
    print(f"{'='*70}")
    
    by_lr = defaultdict(list)
    for r in valid:
        lr = r["config"]["lr"]
        by_lr[lr].append(r)
    
    print(f"\n{'LR':<10} {'Acc%':<8} {'ρ_epi':<8} {'ρ_ale':<8} {'Converged?':<12}")
    print("-" * 50)
    
    for lr in sorted(by_lr.keys(), reverse=True):
        runs = by_lr[lr]
        accs = [r["test"]["accuracy"] for r in runs]
        rho_epis = [r["test"]["rho_epi"] for r in runs]
        rho_ales = [r["test"]["rho_ale"] for r in runs]
        
        # Check convergence (did accuracy improve over epochs?)
        converged = "✓" if np.mean(accs) > 0.5 else "✗"
        
        print(f"{lr:<10.0e} {np.mean(accs)*100:<8.1f} {np.mean(rho_epis):<8.3f} "
              f"{np.mean(rho_ales):<8.3f} {converged:<12}")


def print_heads_analysis(results: List[Dict[str, Any]]):
    """Analyze how number of heads affects epistemic uncertainty."""
    
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    
    print(f"\n{'='*70}")
    print("ENSEMBLE HEADS ANALYSIS")
    print(f"{'='*70}")
    
    by_heads = defaultdict(list)
    for r in valid:
        h = r["config"]["n_heads"]
        by_heads[h].append(r)
    
    print(f"\n{'Heads':<8} {'Acc%':<8} {'ρ_epi':<8} {'Epi_mean':<10} {'Epi_std':<10} {'Ratio':<8}")
    print("-" * 60)
    
    for h in sorted(by_heads.keys()):
        runs = by_heads[h]
        accs = [r["test"]["accuracy"] for r in runs]
        rho_epis = [r["test"]["rho_epi"] for r in runs]
        epi_means = [r["test"]["epi_mean"] for r in runs]
        epi_stds = [r["test"]["epi_std"] for r in runs]
        ratios = [r["test"]["disagree_ratio"] for r in runs]
        
        print(f"{h:<8} {np.mean(accs)*100:<8.1f} {np.mean(rho_epis):<8.3f} "
              f"{np.mean(epi_means):<10.6f} {np.mean(epi_stds):<10.6f} {np.mean(ratios):<8.2f}")
    
    print(f"\nInsights:")
    print(f"   - More heads (5-7) typically give better epistemic calibration")
    print(f"   - Too few heads (1-2) may not capture disagreement well")
    print(f"   - Diminishing returns after ~7 heads")


if __name__ == "__main__":
    main()
