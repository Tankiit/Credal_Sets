"""
Variational Credal CBM - Main Entry Point
==========================================

This file contains:
- Configuration management
- Training loop
- Evaluation metrics
- Experiment orchestration
- Example usage

Author: Tanmoy
Target: ICML 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Import model components from VCBM.py
from VCBM import (
    VariationalCredalCBM,
    VariationalCredalConfig,
    CovarianceFamily,
    QuadrantRouter,
    InterventionExperiment,
    compute_uncertainty_correlations,
    compute_credal_coverage,
    run_covariance_ablation
)


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def get_config(
    encoder_name: str = "distilbert-base-uncased",
    num_concepts: int = 4,
    concept_classes: int = 3,
    num_classes: int = 2,
    covariance_family: str = "mean_field",
    **kwargs
) -> VariationalCredalConfig:
    """
    Create configuration object with parameters

    Args:
        encoder_name: HuggingFace model name
        num_concepts: Number of concepts
        concept_classes: K classes per concept (default: 3 for neg/unk/pos)
        num_classes: Number of task classes
        covariance_family: Type of covariance structure
        **kwargs: Additional config parameters

    Returns:
        VariationalCredalConfig object
    """
    # Map string to enum
    if isinstance(covariance_family, str):
        covariance_family = CovarianceFamily(covariance_family)

    config = VariationalCredalConfig(
        encoder_name=encoder_name,
        num_concepts=num_concepts,
        concept_classes=concept_classes,
        num_classes=num_classes,
        covariance_family=covariance_family,
        **kwargs
    )

    return config


def save_config(config: VariationalCredalConfig, save_path: str):
    """Save configuration to JSON file"""
    config_dict = {
        'encoder_name': config.encoder_name,
        'freeze_encoder': config.freeze_encoder,
        'num_concepts': config.num_concepts,
        'concept_names': config.concept_names,
        'concept_classes': config.concept_classes,
        'num_classes': config.num_classes,
        'covariance_family': config.covariance_family.value,
        'low_rank_dim': config.low_rank_dim,
        'prior_std': config.prior_std,
        'num_mc_samples': config.num_mc_samples,
        'credal_confidence': config.credal_confidence,
        'kl_weight': config.kl_weight,
        'concept_weight': config.concept_weight,
        'aleatoric_weight': config.aleatoric_weight,
        'epistemic_threshold': config.epistemic_threshold,
        'aleatoric_threshold': config.aleatoric_threshold
    }

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_path: str) -> VariationalCredalConfig:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Convert string back to enum
    config_dict['covariance_family'] = CovarianceFamily(config_dict['covariance_family'])

    config = VariationalCredalConfig(**config_dict)
    return config


# ============================================================================
# ENCODER MAPPING
# ============================================================================

ENCODER_MAP = {
    # Short names for encoders
    'roberta': 'roberta-base',
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
    'deberta': 'microsoft/deberta-v3-base',
    'deberta-v3': 'microsoft/deberta-v3-base',
    'deberta-v3-base': 'microsoft/deberta-v3-base',
    'deberta-v3-large': 'microsoft/deberta-v3-large',
    'distilbert': 'distilbert-base-uncased',
    'distilbert-base': 'distilbert-base-uncased',
    'modernbert': 'answerdotai/ModernBERT-base',
    'modernbert-base': 'answerdotai/ModernBERT-base',
    'modernbert-large': 'answerdotai/ModernBERT-large',

    # LLMs (use frozen feature extraction by default)
    'phi-3': 'microsoft/Phi-3-mini-4k-instruct',
    'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',
    'phi-3.5-mini': 'microsoft/Phi-3.5-mini-instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen-0.5b': 'Qwen/Qwen0.5-0.5B-Instruct',
    'qwen-1.5b': 'Qwen/Qwen1.5-1.5B-Instruct',
    'gemma-2b': 'google/gemma-2b-it',
}


def resolve_encoder_name(encoder_short: str) -> str:
    """
    Resolve short encoder name to full HuggingFace model name

    Args:
        encoder_short: Short name or full model name

    Returns:
        Full HuggingFace model name
    """
    encoder_short = encoder_short.lower().strip()

    # Check if it's already a full name (contains '/')
    if '/' in encoder_short:
        return encoder_short

    # Look up in map
    if encoder_short in ENCODER_MAP:
        return ENCODER_MAP[encoder_short]

    # Return as-is if not found (assume it's a full name)
    return encoder_short


def is_llm(encoder_name: str) -> bool:
    """
    Check if encoder is an LLM (needs special handling)

    Args:
        encoder_name: Full HuggingFace model name

    Returns:
        True if LLM, False otherwise
    """
    llm_keywords = ['phi', 'mistral', 'llama', 'qwen', 'gemma', 'instruct']
    encoder_lower = encoder_name.lower()
    return any(keyword in encoder_lower for keyword in llm_keywords)


# ============================================================================
# DATASET
# ============================================================================

class SentimentDataset(Dataset):
    """
    Simple sentiment dataset with concept annotations

    For demonstration purposes. Replace with actual dataset loader.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        concept_labels: Optional[List[List[int]]] = None,
        tokenizer=None,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.concept_labels = concept_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        # Add concept labels if available
        if self.concept_labels is not None:
            item['concept_labels'] = torch.tensor(
                self.concept_labels[idx],
                dtype=torch.long
            )

        return item


def load_amiqa(tokenizer=None, data_path: str = './data/amiqa', max_length: int = 128):
    """
    Load AMIQA dataset

    AMIQA is a multi-aspect sentiment analysis dataset with concept annotations

    Args:
        tokenizer: Tokenizer to use
        data_path: Path to AMIQA data
        max_length: Max sequence length

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # TODO: Implement actual AMIQA loading
    # This is a placeholder - replace with actual data loading logic

    print("Loading AMIQA dataset...")

    # Placeholder: Create dummy datasets
    # Replace this with actual AMIQA data loading
    train_dataset, _, _, _ = create_dummy_dataset(n_samples=1000, tokenizer=tokenizer, max_length=max_length)
    val_dataset, _, _, _ = create_dummy_dataset(n_samples=200, tokenizer=tokenizer, max_length=max_length)
    test_dataset, _, _, _ = create_dummy_dataset(n_samples=200, tokenizer=tokenizer, max_length=max_length)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Max sequence length: {max_length}")

    return train_dataset, val_dataset, test_dataset


def load_cebab(tokenizer=None, data_path: str = './data/cebab', max_length: int = 128):
    """
    Load CEBaB dataset with all intermediate aspects

    CEBaB (CEBaB 2.0) is a restaurant review dataset with multi-dimensional annotations

    Aspects available:
    - food_aspect_majority: Food quality (0, 1, 2)
    - service_aspect_majority: Service quality (0, 1, 2)
    - ambiance_aspect_majority: Ambiance (0, 1, 2)
    - noise_aspect_majority: Noise level (0, 1, 2)

    Args:
        tokenizer: Tokenizer to use
        data_path: Path to CEBaB data (not used, loading from HF)
        max_length: Max sequence length

    Returns:
        train_loader, val_loader, test_loader (PyTorch DataLoaders)
    """
    from load_ambigqa_dataset import create_cebab_dataloaders
    from torch.utils.data import DataLoader

    print("Loading CEBaB dataset from HuggingFace...")

    # Create dataloaders directly from CEBaB
    dataloaders = create_cebab_dataloaders(
        dataset_name="CEBaB/CEBaB",
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    # Add max_length attribute to each dataset
    for key, dataloader in dataloaders.items():
        if hasattr(dataloader, 'dataset'):
            dataloader.dataset.max_length = max_length

    print(f"  Train batches: {len(dataloaders.get('train_inclusive', dataloaders.get('train', [])))}")
    print(f"  Val batches: {len(dataloaders.get('validation', dataloaders.get('val', [])))}")
    if 'test' in dataloaders:
        print(f"  Test batches: {len(dataloaders['test'])}")
    print(f"  Concepts: food, service, ambiance, noise (all 0/1/2)")
    print(f"  Max sequence length: {max_length}")

    # Return the dataloaders directly
    # CEBaB uses 'train_inclusive' as the main training split
    train_loader = dataloaders.get('train_inclusive') or dataloaders.get('train')
    val_loader = dataloaders.get('validation') or dataloaders.get('val')
    test_loader = dataloaders.get('test')

    return train_loader, val_loader, test_loader


def get_dataset_max_length(dataset_name: str) -> int:
    """
    Get appropriate max sequence length for each dataset

    Args:
        dataset_name: Name of the dataset

    Returns:
        Max sequence length
    """
    dataset_lengths = {
        'cebab': 128,        # Short restaurant reviews
        'amiqa': 256,        # Longer QA pairs
        'hatexplain': 256,   # Medium length with rationales
        'goemotions': 128,   # Short Reddit comments
        'civil_comments': 256, # Medium length comments
        'tid8': 256,         # NLI premises+hypotheses
        'sst2': 128,         # Short sentences
        'dummy': 128,        # Dummy data
    }
    return dataset_lengths.get(dataset_name.lower(), 128)


def get_dataset_num_classes(dataset_name: str) -> int:
    """
    Get number of classes for each dataset

    Args:
        dataset_name: Name of the dataset

    Returns:
        Number of classes
    """
    dataset_classes = {
        'cebab': 6,          # 1-5 star ratings + 'no majority'
        'amiqa': 2,          # Binary sentiment
        'hatexplain': 3,     # 3 classes: hatespeech/normal/offensive
        'goemotions': 27,    # 27 emotion classes (multi-label)
        'civil_comments': 2, # Binary: non_toxic/toxic
        'tid8': 3,           # 3-class NLI: entailment/neutral/contradiction
        'sst2': 2,           # Binary sentiment
        'dummy': 2,          # Binary for testing
    }
    return dataset_classes.get(dataset_name.lower(), 2)


def load_dataset(
    dataset_name: str,
    tokenizer=None,
    data_path: str = './data'
):
    """
    Load dataset by name

    Args:
        dataset_name: 'amiqa' or 'cebab'
        tokenizer: Tokenizer to use
        data_path: Base path for data

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'amiqa':
        max_len = get_dataset_max_length('amiqa')
        # Override to use 128 for frozen encoder compatibility
        max_len = 128
        return load_amiqa(tokenizer, f"{data_path}/amiqa", max_length=max_len)
    elif dataset_name == 'cebab':
        max_len = get_dataset_max_length('cebab')
        return load_cebab(tokenizer, f"{data_path}/cebab", max_length=max_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: amiqa, cebab")


def create_dummy_dataset(n_samples: int = 100, tokenizer=None, max_length: int = 128):
    """
    Create a dummy dataset for testing

    Args:
        n_samples: Number of samples to generate
        tokenizer: Tokenizer to use
        max_length: Max sequence length

    Returns:
        dataset, texts, labels, concept_labels
    """
    np.random.seed(42)

    # Generate dummy texts
    texts = [
        "The food was great and service was excellent.",
        "Terrible experience, would not recommend.",
        "It was okay, nothing special.",
        "Amazing ambiance but the food was cold.",
    ] * (n_samples // 4)

    texts = texts[:n_samples]

    # Generate labels (0: negative, 1: positive)
    labels = np.random.randint(0, 2, size=n_samples).tolist()

    # Generate concept labels (0: negative, 1: unknown, 2: positive)
    # 4 concepts: food, service, ambiance, noise
    concept_labels = np.random.randint(0, 3, size=(n_samples, 4)).tolist()

    dataset = SentimentDataset(
        texts=texts,
        labels=labels,
        concept_labels=concept_labels,
        tokenizer=tokenizer,
        max_length=128  # Fixed for frozen encoder compatibility
    )

    return dataset, texts, labels, concept_labels


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: VariationalCredalCBM,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[object] = None,
    device: str = 'cuda',
    tokenizer: Optional[object] = None
) -> Dict[str, float]:
    """
    Train for one epoch

    Returns:
        Dictionary with average losses
    """
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    total_concept_loss = 0.0
    total_aleatoric_loss = 0.0

    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Handle both tokenized batches (from our dataset) and raw text (from CEBaB)
        if 'input_ids' in batch:
            # Already tokenized (our SentimentDataset)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch.get('concept_labels', None)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)
        else:
            # Raw text from CEBaB dataloader - need to tokenize
            texts = batch['description']
            # Get max_length from the dataset (stored in dataloader if available)
            max_len = getattr(dataloader.dataset, 'max_length', 128)
            # Tokenize batch
            encoded = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Get labels (handle both tensors and lists/strings)
            labels_batch = batch['review_majority']
            if isinstance(labels_batch, torch.Tensor):
                labels = labels_batch.to(device)
            elif isinstance(labels_batch, (list, tuple)):
                # Check if contains strings or numbers
                if all(isinstance(l, str) for l in labels_batch):
                    # Handle different string label formats
                    # Check for CEBaB format (includes 'no majority')
                    if any(l in ['1', '2', '3', '4', '5', 'no majority'] for l in labels_batch):
                        # CEBaB label mapping: '1'-'5' → 0-4, 'no majority' → 5
                        label_map = {
                            '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                            'no majority': 5  # 6th class for ambiguous cases
                        }
                        labels = torch.tensor([label_map.get(l, 5) for l in labels_batch], dtype=torch.long).to(device)
                    else:
                        # Standard Positive/Negative labels
                        label_map = {'Positive': 1, 'Negative': 0, 'positive': 1, 'negative': 0}
                        labels = torch.tensor([label_map.get(l, 1) for l in labels_batch], dtype=torch.long).to(device)
                else:
                    # Numeric labels
                    labels = torch.tensor(labels_batch, dtype=torch.long).to(device)
            else:
                # Single value
                labels = torch.tensor([labels_batch], dtype=torch.long).to(device)

            # Get concept labels (handle both tensors and lists/strings)
            concept_label_map = {
                'Positive': 2, 'Negative': 0, 'Unknown': 1,
                'positive': 2, 'negative': 0, 'neutral': 1,
                2: 2, 0: 0, 1: 1
            }

            aspects = [
                batch['food_aspect_majority'],
                batch['service_aspect_majority'],
                batch['ambiance_aspect_majority'],
                batch['noise_aspect_majority']
            ]

            # Convert each to tensor if needed
            aspect_tensors = []
            for aspect in aspects:
                if isinstance(aspect, torch.Tensor):
                    aspect_tensors.append(aspect)
                elif isinstance(aspect, (list, tuple)):
                    # List of strings or numbers
                    if all(isinstance(a, str) for a in aspect):
                        aspect_tensors.append(torch.tensor([concept_label_map.get(a, 1) for a in aspect], dtype=torch.long))
                    else:
                        aspect_tensors.append(torch.tensor([concept_label_map.get(a, 1) for a in aspect], dtype=torch.long))
                else:
                    # Single value
                    aspect_tensors.append(torch.tensor([concept_label_map.get(aspect, 1)], dtype=torch.long))

            concept_labels = torch.stack(aspect_tensors, dim=1).to(device)  # [batch, 4]

        # Forward pass
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            concept_labels=concept_labels
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate losses
        total_loss += loss.item()
        total_ce_loss += outputs.get('ce_loss', 0.0).item() if isinstance(outputs.get('ce_loss'), torch.Tensor) else outputs.get('ce_loss', 0.0)
        total_kl_loss += outputs.get('kl_loss', 0.0).item()
        total_concept_loss += outputs.get('concept_loss', 0.0).item() if isinstance(outputs.get('concept_loss'), torch.Tensor) else outputs.get('concept_loss', 0.0)
        total_aleatoric_loss += outputs.get('aleatoric_loss', 0.0).item() if isinstance(outputs.get('aleatoric_loss'), torch.Tensor) else outputs.get('aleatoric_loss', 0.0)

        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'concept_loss': total_concept_loss / num_batches,
        'aleatoric_loss': total_aleatoric_loss / num_batches
    }


def evaluate(
    model: VariationalCredalCBM,
    dataloader: DataLoader,
    device: str = 'cuda',
    tokenizer: Optional[object] = None
) -> Dict[str, float]:
    """
    Evaluate model

    Returns:
        Dictionary with metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []
    all_epistemic = []
    all_aleatoric = []
    all_credal_lower = []
    all_credal_upper = []
    all_concept_probs = []  # NEW: collect concept probabilities for credal coverage

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle both tokenized batches and raw text (from CEBaB)
            if 'input_ids' in batch:
                # Already tokenized
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            else:
                # Raw text from CEBaB dataloader - need to tokenize
                texts = batch['description']
                # Get max_length from the dataset
                max_len = getattr(dataloader.dataset, 'max_length', 128)
                encoded = tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                # Get labels (handle both tensors and lists/strings)
                labels_batch = batch['review_majority']
                if isinstance(labels_batch, torch.Tensor):
                    labels = labels_batch.to(device)
                elif isinstance(labels_batch, (list, tuple)):
                    if all(isinstance(l, str) for l in labels_batch):
                        # Handle different string label formats
                        # Check for CEBaB format (includes 'no majority')
                        if any(l in ['1', '2', '3', '4', '5', 'no majority'] for l in labels_batch):
                            # CEBaB label mapping: '1'-'5' → 0-4, 'no majority' → 5
                            label_map = {
                                '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                                'no majority': 5  # 6th class for ambiguous cases
                            }
                            labels = torch.tensor([label_map.get(l, 5) for l in labels_batch], dtype=torch.long).to(device)
                        else:
                            # Standard Positive/Negative labels
                            label_map = {'Positive': 1, 'Negative': 0, 'positive': 1, 'negative': 0}
                            labels = torch.tensor([label_map.get(l, 1) for l in labels_batch], dtype=torch.long).to(device)
                    else:
                        labels = torch.tensor(labels_batch, dtype=torch.long).to(device)
                else:
                    labels = torch.tensor([labels_batch], dtype=torch.long).to(device) if not isinstance(labels_batch, torch.Tensor) else labels_batch.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Collect predictions
            preds = outputs['predictions'].cpu().numpy()
            probs = outputs['probs'].cpu().numpy()

            all_predictions.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

            # Collect uncertainties
            all_epistemic.extend(outputs['epistemic'].cpu().numpy())
            all_aleatoric.extend(outputs['aleatoric'].cpu().numpy())
            all_credal_lower.extend(outputs['credal_lower'].cpu().numpy())
            all_credal_upper.extend(outputs['credal_upper'].cpu().numpy())
            all_concept_probs.extend(outputs['concept_probs'].cpu().numpy())  # NEW: collect concept probs

    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_epistemic = np.array(all_epistemic)
    all_aleatoric = np.array(all_aleatoric)
    all_credal_lower = np.array(all_credal_lower)
    all_credal_upper = np.array(all_credal_upper)
    all_concept_probs = np.array(all_concept_probs)  # NEW: convert concept probs

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    # Determine averaging method based on number of unique classes
    num_unique_classes = len(np.unique(all_labels + all_predictions))
    average_method = 'binary' if num_unique_classes == 2 else 'weighted'

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=average_method, zero_division=0
    )

    # Uncertainty correlations
    errors = (all_predictions != all_labels).astype(float)
    unknown_mask = (all_aleatoric > 0.5).astype(float)  # Proxy for unknown

    uncertainty_corr = compute_uncertainty_correlations(
        epistemic=all_epistemic,
        aleatoric=all_aleatoric,
        errors=errors,
        unknown_mask=unknown_mask
    )

    # Credal coverage - FIXED: use concept probs to match credal bounds shape
    # credal_lower/upper have shape [N, num_concepts], all_concept_probs has shape [N, num_concepts]
    credal_metrics = compute_credal_coverage(
        credal_lower=all_credal_lower,
        credal_upper=all_credal_upper,
        true_probs=all_concept_probs  # FIXED: Use concept probs, same shape as credal bounds!
    )

    # Quadrant analysis
    router = QuadrantRouter(
        epistemic_threshold=0.15,
        aleatoric_threshold=0.35
    )
    routing = router.route(
        torch.from_numpy(all_epistemic),
        torch.from_numpy(all_aleatoric)
    )

    quadrant_metrics = {}
    for quad in ['TRUST', 'REVIEW', 'DATA', 'ABSTAIN']:
        mask = routing[quad].cpu().numpy()
        if mask.sum() > 0:
            quad_acc = accuracy_score(all_labels[mask], all_predictions[mask])
            quadrant_metrics[f'{quad}_acc'] = quad_acc
            quadrant_metrics[f'{quad}_count'] = mask.sum()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **uncertainty_corr,
        **credal_metrics,
        **quadrant_metrics
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(
    config: VariationalCredalConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    use_compile: bool = True,
    compile_mode: str = 'default',
    tokenizer: Optional[object] = None,
    checkpoint_interval: int = 0
):
    """
    Full training loop

    Args:
        config: Model configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        use_compile: Whether to use torch.compile (PyTorch 2.0+)
        compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        tokenizer: Tokenizer for on-the-fly tokenization (needed for CEBaB)
        checkpoint_interval: Save checkpoint every N epochs (0 = only best/final)
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, save_path / 'config.json')

    # Create model
    model = VariationalCredalCBM(config).to(device)

    # Freeze encoder if specified
    if config.freeze_encoder:
        print(f"\n{'='*70}")
        print("Freezing encoder parameters...")
        print(f"{'='*70}")
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen_params = sum(p.numel() for p in model.encoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen: {frozen_params:,} parameters")
        print(f"  Trainable: {trainable_params:,} / {total_params:,} parameters")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*70}")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"{'='*70}")

    # Adaptive learning rate for frozen encoder training
    if config.freeze_encoder:
        # Much higher LR when only heads are training
        original_lr = learning_rate
        learning_rate = 1e-3  # 50x higher for frozen encoder
        print(f"\n{'='*70}")
        print(f"Adaptive learning rate for frozen encoder:")
        print(f"  Original LR: {original_lr:.2e}")
        print(f"  Adjusted LR: {learning_rate:.2e} (50x higher for head-only training)")
        print(f"{'='*70}")

    # Apply torch.compile if available and requested
    if use_compile:
        print(f"\n{'='*70}")
        print(f"Applying torch.compile (mode={compile_mode})...")
        print(f"{'='*70}")
        print("This may take a minute for the first compilation...")

        try:
            # Check PyTorch version (torch already imported at module level)
            torch_version = torch.__version__
            print(f"PyTorch version: {torch_version}")

            # Try to compile
            model = torch.compile(model, mode=compile_mode)
            print("✓ Model compiled successfully!")
        except Exception as e:
            print(f"✗ Compilation failed: {e}")
            print("Training without compilation...")

    # Optimizer and scheduler (only optimize trainable parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'val_metrics': []
    }

    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, tokenizer
        )

        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"  CE: {train_metrics['ce_loss']:.4f} | "
              f"KL: {train_metrics['kl_loss']:.6f} | "
              f"Concept: {train_metrics['concept_loss']:.4f} | "
              f"Aleatoric: {train_metrics['aleatoric_loss']:.4f}")

        history['train_loss'].append(train_metrics)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, tokenizer)
        history['val_metrics'].append(val_metrics)

        print(f"\nVal Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  Epistemic-Error Corr: {val_metrics['rho_epi_err']:.4f}")
        print(f"  Aleatoric-Unknown Corr: {val_metrics['rho_ale_unk']:.4f}")
        print(f"  Epistemic-Aleatoric Corr: {val_metrics['rho_epi_ale']:.4f}")
        print(f"  Separation: {val_metrics['separation']:.4f}")


        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config
            }, save_path / 'best_model.pt')
            print(f"\nSaved best model (acc: {best_val_acc:.4f})")

        # Save intermediate checkpoint
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config,
                'history': history
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

        print("\n")

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }, save_path / 'final_model.pt')

    print(f"{'='*70}")
    print(f"Training complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"{'='*70}\n")

    return model, history


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_intervention_experiments(
    model: VariationalCredalCBM,
    val_loader: DataLoader,
    device: str = 'cuda'
):
    """
    Run intervention experiments comparing epistemic vs aleatoric targeting
    """

    # Collect all batches from dataloader
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_concept_labels = []

    print("Collecting data from validation set...")

    for batch in val_loader:
        # CEBaB batches have different structure
        if 'input_ids' in batch:
            # Our dataset format
            all_input_ids.append(batch['input_ids'])
            all_attention_mask.append(batch['attention_mask'])
            all_labels.append(batch.get('labels', batch.get('review_majority')))

            # Get concept labels from the batch
            # Extract from CEBaB structure: food, service, ambiance, noise
            if 'concept_labels' in batch:
                all_concept_labels.append(batch['concept_labels'])
            elif 'food_aspect_majority' in batch:
                # Stack the 4 aspects
                concepts = torch.stack([
                    batch['food_aspect_majority'],
                    batch['service_aspect_majority'],
                    batch['ambiance_aspect_majority'],
                    batch['noise_aspect_majority']
                ], dim=1)  # [batch, 4]
                all_concept_labels.append(concepts)

    # Concatenate all batches
    if all_input_ids:
        input_ids = torch.cat(all_input_ids).to(device)
        attention_mask = torch.cat(all_attention_mask).to(device)
        labels = torch.cat(all_labels).to(device)
    else:
        print("Warning: No data found in dataloader")
        return None

    if all_concept_labels:
        concept_labels = torch.cat(all_concept_labels).to(device)
    else:
        print("Warning: No concept labels available, skipping intervention experiments")
        return None

    # Run experiments
    experiment = InterventionExperiment(model, device=device)
    results = experiment.compare_strategies(
        input_ids=input_ids,
        attention_mask=attention_mask,
        true_concepts=concept_labels,
        true_labels=labels,
        k_values=[1, 2, 3, 4]
    )

    return results


def run_covariance_ablation_study(
    train_dataset: Dataset,
    val_dataset: Dataset,
    base_config: VariationalCredalConfig,
    device: str = 'cuda'
):
    """
    Run ablation study over covariance structures
    """

    results = run_covariance_ablation(
        train_data={'dataset': train_dataset},
        val_data={'dataset': val_dataset},
        base_config=base_config,
        num_runs=3
    )


    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Variational Credal CBM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with RoBERTa on AMIQA
  python main.py --dataset amiqa --encoder roberta

  # Train with ModernBERT on CEBaB
  python main.py --dataset cebab --encoder modernbert

  # Use full model name
  python main.py --dataset amiqa --encoder answerdotai/ModernBERT-base

  # Train with LLM encoder (frozen by default)
  python main.py --dataset amiqa --encoder phi-3-mini

  # Run experiments
  python main.py --dataset cebab --encoder deberta --run-intervention --run-ablation

Available datasets:
  amiqa   - Multi-aspect sentiment analysis
  cebab   - Restaurant review dataset

Available encoders (short names):
  Encoder Models:
  - roberta, roberta-base, roberta-large
  - deberta, deberta-v3, deberta-v3-base, deberta-v3-large
  - distilbert, distilbert-base
  - modernbert, modernbert-base, modernbert-large

  LLM Models (frozen by default):
  - phi-3, phi-3-mini, phi-3.5-mini
  - mistral, mistral-7b
  - llama-3.2-1b, llama-3.2-3b
  - qwen-0.5b, qwen-1.5b
  - gemma-2b

  Or use full HuggingFace model names.
        """
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default='dummy',
                        choices=['dummy', 'amiqa', 'cebab'],
                        help='Dataset to use (default: dummy)')

    # Configuration
    parser.add_argument('--encoder', type=str, default='distilbert-base-uncased',
                        help='Encoder model name (short name or full HF name)')
    parser.add_argument('--num-concepts', type=int, default=4,
                        help='Number of concepts')
    parser.add_argument('--concept-classes', type=int, default=3,
                        help='K classes per concept')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of task classes (auto-detected from dataset if not specified)')
    parser.add_argument('--covariance', type=str, default='mean_field',
                        choices=['mean_field', 'low_rank', 'full'],
                        help='Covariance family')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # Optimization
    parser.add_argument('--freeze-encoder', type=bool, default=None,
                        help='Whether to freeze encoder (auto-detected for LLMs)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (PyTorch 2.0+ optimization)')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: default)')

    # Experiments
    parser.add_argument('--run-intervention', action='store_true',
                        help='Run intervention experiments')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run covariance ablation study')

    # Checkpointing
    parser.add_argument('--checkpoint-interval', type=int, default=0,
                        help='Save checkpoint every N epochs (0 = only best/final)')

    # Paths
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Base path for datasets')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (if loading)')

    args = parser.parse_args()

    # Resolve encoder name
    encoder_name = resolve_encoder_name(args.encoder)
    print(f"\n{'='*70}")
    print(f"Encoder: {encoder_name}")
    if is_llm(encoder_name):
        print("Type: LLM (using frozen feature extraction)")
        # Auto-set freeze for LLMs if not specified
        if args.freeze_encoder is None:
            args.freeze_encoder = True
    else:
        print("Type: Standard encoder")
        if args.freeze_encoder is None:
            args.freeze_encoder = False
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"{'='*70}\n")

    # Load or create configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        # Auto-detect num_classes from dataset if not explicitly provided
        num_classes = args.num_classes
        if args.num_classes is None:
            num_classes = get_dataset_num_classes(args.dataset)
            print(f"Auto-detected {num_classes} classes for {args.dataset}")

        config = get_config(
            encoder_name=encoder_name,
            num_concepts=args.num_concepts,
            concept_classes=args.concept_classes,
            num_classes=num_classes,
            covariance_family=args.covariance,
            freeze_encoder=args.freeze_encoder
        )

        # DEBUG: Verify num_classes in config
        print(f"DEBUG: config.num_classes = {config.num_classes}")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    print("-" * 70)

    if args.dataset == 'dummy':
        print("Using dummy dataset for testing...")
        train_dataset, _, _, _ = create_dummy_dataset(n_samples=500, tokenizer=tokenizer)
        val_dataset, _, _, _ = create_dummy_dataset(n_samples=100, tokenizer=tokenizer)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = None
    else:
        train_loader, val_loader, test_loader = load_dataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            data_path=args.data_path
        )

    print()

    # Train model
    model, history = train_model(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode,
        tokenizer=tokenizer,
        checkpoint_interval=args.checkpoint_interval
    )

    # Run experiments
    if args.run_intervention:
        print("\n" + "="*70)
        print("Running Intervention Experiments")
        print("="*70 + "\n")
        results = run_intervention_experiments(model, val_loader, args.device)

        # Print summary
        print("\nIntervention Results:")
        print("-" * 70)
        for k in range(4):
            print(f"\nk={k+1} concepts intervened:")
            for strategy in ['epistemic', 'aleatoric', 'random']:
                res = results[strategy][k]
                print(f"  {strategy:12s}: baseline={res['baseline_acc']:.4f}, "
                      f"intervened={res['intervened_acc']:.4f}, "
                      f"gain={res['accuracy_gain']:+.4f}")

    if args.run_ablation:
        print("\n" + "="*70)
        print("Running Covariance Ablation Study")
        print("="*70 + "\n")
        run_covariance_ablation_study(train_dataset, val_dataset, config, args.device)

    print("\n✓ All experiments complete!")


if __name__ == "__main__":
    main()
