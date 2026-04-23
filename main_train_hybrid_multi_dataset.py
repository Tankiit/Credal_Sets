"""
Hybrid Credal CBM Training on Multiple Datasets
==============================================

Training script for the Hybrid Credal CBM with support for:
- CEBaB (restaurant reviews)
- HateXplain (hate speech detection)
- GoEmotions (emotion classification)
- MAQA (question ambiguity)

Usage:
    python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v7b

Author: Tanmoy
Date: January 2026
"""

# =============================================================================
# CORE IMPORTS (always needed)
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
import numpy as np
from scipy import stats
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import argparse

# =============================================================================
# LOCAL MODULE IMPORTS - CBM Models
# =============================================================================
from VCBM import HybridCredalCBM, HybridCredalConfig
from load_cebab_direct import get_cebab_dataloaders
from load_hatexplain_direct import get_hatexplain_dataloaders

# =============================================================================
# MAQA IMPORTS (conditional)
# =============================================================================
HAS_MAQA = False
HAS_V7B_COMPLETE = False

try:
    # Core MAQA components
    # Use real data loader (works around HuggingFace datasets issue)
    from load_maqa_real import load_combined_maqa_ambigqa
    from maqa_credal_model import (
        CredalMAQA, MAQADataset, MAQACredalTrainer, maqa_collate_fn
    )

    # Loss versions V3-V6
    from maqa_credal_loss_v3 import (
        MAQACredalLossV3, CONFIG_V3, initialize_uncertainty_heads
    )
    from maqa_credal_loss_v4 import (
        MAQACredalLossV4, MAQACredalLossV4Adapter, CONFIG_V4, initialize_model_for_v4
    )
    from maqa_credal_loss_v5 import (
        MAQACredalLossV5, MAQACredalLossV5Adapter, CONFIG_V5
    )
    from maqa_credal_loss_v6 import (
        MAQACredalLossV6, MAQACredalLossV6Adapter, CONFIG_V6
    )

    # V7b (fixed version with proper key mapping)
    from maqa_credal_loss_v7b_fixed import (
        CONFIG_V7B, create_v7b_adapter, initialize_model_for_v7b
    )

    HAS_MAQA = True

    # V7b complete integration (entropy-aware model)
    try:
        from v7b_complete_integration import (
            CredalMAQA_V7b, MAQACredalTrainerV7b
        )
        HAS_V7B_COMPLETE = True
    except ImportError:
        print("Note: V7b complete integration not found, using standard model")

except ImportError as e:
    print(f"Warning: MAQA components not available: {e}")

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
# Multi-dataset loader
HAS_MULTI_LOADER = False
try:
    from credence_dataloader import load_dataset_splits, DatasetConfig
    HAS_MULTI_LOADER = True
except ImportError:
    pass

# Quantization (bitsandbytes)
HAS_BNB = False
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    pass

# LoRA (PEFT)
HAS_PEFT = False
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    HAS_PEFT = True
except ImportError:
    pass


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
DATASET_CONFIGS = {
    'cebab': {
        'name': 'CEBaB',
        'num_concepts': 4,
        'concept_names': ['food', 'service', 'ambiance', 'noise'],
        'num_classes': 5,
        'save_dir': './checkpoints/hybrid_credal_cebab',
        'data_loader': get_cebab_dataloaders,
        'use_multi_loader': False,
        'loader_kwargs': {'batch_size': 8, 'max_length': 256, 'num_workers': 0},
        'prior_sigma': 0.5,
        'error_scale': 2.0,
        'learning_rate': 1e-3,
        'num_epochs': 5,
    },
    'hatexplain': {
        'name': 'HateXplain',
        'num_concepts': 2,
        'concept_names': ['has_target', 'is_offensive'],
        'num_classes': 3,
        'save_dir': './checkpoints/hybrid_credal_hatexplain',
        'data_loader': get_hatexplain_dataloaders,
        'use_multi_loader': False,
        'loader_kwargs': {'batch_size': 16, 'max_length': 128, 'num_workers': 0},
        'prior_sigma': 0.5,
        'error_scale': 1.0,
        'learning_rate': 1e-3,
        'num_epochs': 10,
    },
    'goemotions': {
        'name': 'GoEmotions',
        'num_concepts': 28,
        'concept_names': [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise', 'neutral'
        ],
        'num_classes': 28,
        'save_dir': './checkpoints/hybrid_credal_goemotions',
        'use_multi_loader': True,
        'loader_kwargs': {'batch_size': 32, 'max_length': 64, 'num_workers': 0},
        'prior_sigma': 0.5,
        'error_scale': 1.0,
        'learning_rate': 5e-5,
        'num_epochs': 20,
    },
    'maqa': {
        'name': 'MAQA-Star',
        'num_concepts': 0,
        'concept_names': [],
        'num_classes': 10,
        'save_dir': './checkpoints/maqa_credal',
        'use_multi_loader': False,
        'use_maqa_model': True,
        'data_loader': None,
        'loader_kwargs': {'batch_size': 16, 'max_length': 128, 'num_workers': 0},
        'learning_rate': 1e-3,
        'num_epochs': 15,
        'projection_dim': 256,
        'dropout': 0.1,
    }
}


# =============================================================================
# MODEL REGISTRY
# =============================================================================
MODEL_REGISTRY = {
    "distilbert-base-uncased": {"type": "encoder", "hidden_size": 768, "max_length": 512},
    "roberta-base": {"type": "encoder", "hidden_size": 768, "max_length": 512},
    "roberta-large": {"type": "encoder", "hidden_size": 1024, "max_length": 512},
    "microsoft/deberta-v3-base": {"type": "encoder", "hidden_size": 768, "max_length": 512},
    "microsoft/deberta-v3-large": {"type": "encoder", "hidden_size": 1024, "max_length": 512},
}

ENCODER_SHORT_NAMES = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "roberta-large": "roberta-large",
    "deberta": "microsoft/deberta-v3-base",
    "deberta-v3": "microsoft/deberta-v3-base",
}


def expand_encoder_name(encoder_name: str) -> str:
    """Expand short encoder names to full HuggingFace model names."""
    return ENCODER_SHORT_NAMES.get(encoder_name.lower(), encoder_name)


def get_encoder_config(encoder_name: str) -> Dict:
    """Get encoder configuration from MODEL_REGISTRY."""
    full_name = expand_encoder_name(encoder_name)
    if full_name not in MODEL_REGISTRY:
        # Return default config for unknown encoders
        return {"type": "encoder", "hidden_size": 768, "max_length": 512}
    return MODEL_REGISTRY[full_name]


def load_encoder_with_quantization(encoder_name: str, quantization: str = "none", device_map: str = "auto"):
    """Load encoder with optional quantization."""
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)

    model_kwargs = {"device_map": device_map}

    if quantization == "4bit" and HAS_BNB:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print(f"  ✓ 4-bit quantization enabled")
    elif quantization == "8bit" and HAS_BNB:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(f"  ✓ 8-bit quantization enabled")

    encoder = AutoModel.from_pretrained(encoder_name, **model_kwargs)
    return encoder, tokenizer


# =============================================================================
# UNCERTAINTY METRICS
# =============================================================================
class UncertaintyMetrics:
    """Container for uncertainty metrics."""

    def __init__(self):
        self.accuracy = 0.0
        self.loss = 0.0
        self.concept_accs = {}
        self.mean_sigma_epi = 0.0
        self.std_sigma_epi = 0.0
        self.mean_eu = 0.0
        self.std_eu = 0.0
        self.mean_aleatoric = 0.0
        self.std_aleatoric = 0.0
        self.rho_eu_au = 0.0
        self.p_eu_au = 0.0
        self.rho_eu_error = 0.0
        self.p_eu_error = 0.0
        self.rho_ale_entropy = 0.0
        self.p_ale_entropy = 0.0

    def to_dict(self):
        return {k: float(v) if not isinstance(v, dict) else v
                for k, v in self.__dict__.items()}


# =============================================================================
# HYBRID CREDAL CBM TRAINER (for CEBaB, HateXplain, GoEmotions)
# =============================================================================
class HybridCredalCBMTrainer:
    """Trainer for Hybrid Credal CBM (non-MAQA datasets)."""

    def __init__(self, model: HybridCredalCBM, config: HybridCredalConfig,
                 device: str = "auto", save_dir: str = "./checkpoints"):
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else device
        )
        self.model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.current_epoch = 0

    def train_epoch(self, train_loader, optimizer, scheduler=None, **kwargs):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')
            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)
            annotator_entropy = batch.get('annotator_entropy')
            if annotator_entropy is not None:
                annotator_entropy = annotator_entropy.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, concept_labels=concept_labels,
                annotator_entropy=annotator_entropy
            )

            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': (np.array(all_preds) == np.array(all_labels)).mean()
        }

    @torch.no_grad()
    def evaluate(self, val_loader) -> UncertaintyMetrics:
        """Full evaluation with uncertainty metrics."""
        self.model.eval()

        all_preds, all_labels = [], []
        all_sigma_epi, all_eu, all_aleatoric, all_entropies = [], [], [], []
        total_loss = 0.0

        for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch.get('concept_labels')
            if concept_labels is not None:
                concept_labels = concept_labels.to(self.device)
            annotator_entropy = batch.get('annotator_entropy')
            if annotator_entropy is not None:
                annotator_entropy = annotator_entropy.to(self.device)

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, concept_labels=concept_labels,
                annotator_entropy=annotator_entropy
            )

            total_loss += outputs['loss'].item()
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sigma_epi.append(outputs['sigma_epi'].cpu())
            all_eu.append(outputs['epistemic'].cpu())
            all_aleatoric.append(outputs['aleatoric'].cpu())
            if annotator_entropy is not None:
                all_entropies.append(annotator_entropy.cpu())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_sigma_epi = torch.cat(all_sigma_epi).numpy()
        all_eu = torch.cat(all_eu).numpy()
        all_aleatoric = torch.cat(all_aleatoric).numpy()

        eu_sample = all_eu.mean(axis=-1) if all_eu.ndim > 1 else all_eu
        al_sample = all_aleatoric.mean(axis=-1) if all_aleatoric.ndim > 1 else all_aleatoric
        errors = (all_preds != all_labels).astype(float)

        # Correlations
        rho_eu_au, p_eu_au = stats.spearmanr(eu_sample, al_sample) if eu_sample.std() > 0 and al_sample.std() > 0 else (0, 1)
        rho_eu_err, p_eu_err = stats.spearmanr(eu_sample, errors) if errors.std() > 0 and eu_sample.std() > 0 else (0, 1)

        rho_ale_ent, p_ale_ent = 0.0, 1.0
        if all_entropies:
            ent = torch.cat(all_entropies).numpy()
            ent_sample = ent.mean(axis=-1) if ent.ndim > 1 else ent
            if ent_sample.std() > 0 and al_sample.std() > 0:
                rho_ale_ent, p_ale_ent = stats.spearmanr(al_sample, ent_sample)

        metrics = UncertaintyMetrics()
        metrics.accuracy = (all_preds == all_labels).mean()
        metrics.loss = total_loss / len(val_loader)
        metrics.mean_sigma_epi = all_sigma_epi.mean()
        metrics.std_sigma_epi = all_sigma_epi.std()
        metrics.mean_eu = eu_sample.mean()
        metrics.std_eu = eu_sample.std()
        metrics.mean_aleatoric = al_sample.mean()
        metrics.std_aleatoric = al_sample.std()
        metrics.rho_eu_au = rho_eu_au
        metrics.p_eu_au = p_eu_au
        metrics.rho_eu_error = rho_eu_err
        metrics.p_eu_error = p_eu_err
        metrics.rho_ale_entropy = rho_ale_ent
        metrics.p_ale_entropy = p_ale_ent

        return metrics

    def fit(self, train_loader, val_loader, num_epochs: int, lr: float = 2e-5,
            weight_decay: float = 0.01, warmup_steps: int = 100, save_every: int = 5,
            metadata: Dict = None, **kwargs) -> Dict:
        """Full training loop."""
        import time
        from datetime import datetime

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=len(train_loader) * num_epochs
        )

        history = []
        best_metrics = None
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training for {num_epochs} epochs | LR: {lr:.0e}")
        print(f"{'='*60}")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            val_metrics = self.evaluate(val_loader)

            print(f"\nEpoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics.accuracy:.4f}, "
                  f"ρ(EU,AU)={val_metrics.rho_eu_au:.3f}")

            epoch_data = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics.to_dict(),
                'timestamp': datetime.now().isoformat(),
            }
            history.append(epoch_data)

            if val_metrics.accuracy > self.best_val_acc:
                self.best_val_acc = val_metrics.accuracy
                best_metrics = val_metrics
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'metrics': val_metrics.to_dict(),
                }, self.save_dir / "best_model.pt")
                print(f"  ✓ New best model saved!")

            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, self.save_dir / f"checkpoint_epoch_{epoch}.pt")

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete! Best Val Acc: {self.best_val_acc:.4f}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"{'='*60}")

        # Save history
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump({'history': history, 'metadata': metadata or {}}, f, indent=2, default=float)

        return {'history': history, 'best_val_accuracy': self.best_val_acc}

    def load_best_model(self):
        """Load best model from checkpoint."""
        path = self.save_dir / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"✓ Loaded best model from epoch {ckpt['epoch']}")
            return ckpt.get('metrics')
        return None


# =============================================================================
# MAQA TRAINING FUNCTION
# =============================================================================
def train_maqa_model(trainer, num_epochs: int, save_dir: str,
                     run_metadata: Dict, test_loader) -> Dict:
    """Train MAQA model with per-epoch metrics collection."""
    import time

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")

        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate(trainer.val_loader)

        # Print summary
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"ρ(EU,AU): {val_metrics.get('rho_eu_au', 0):.3f} | "
              f"ρ(AU,H): {val_metrics.get('rho_au_entropy', 0):.3f}")

        history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics})

        # Save checkpoint
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            best_epoch = epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'is_best': is_best
        }, save_dir / f"epoch_{epoch}.pt")

        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
            }, save_dir / "best_model.pt")

    # Load best and evaluate on test
    best_ckpt = torch.load(save_dir / "best_model.pt", weights_only=False)
    trainer.model.load_state_dict(best_ckpt['model_state_dict'])

    print(f"\n{'='*60}\nFinal Test Evaluation\n{'='*60}")
    test_metrics = trainer.evaluate(test_loader)

    results = {
        'test_metrics': test_metrics,
        'training_summary': {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': time.time() - start_time,
        },
        'run_metadata': run_metadata,
        'history': history
    }

    with open(save_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    return results


# =============================================================================
# V3 ADAPTER
# =============================================================================
class MAQACredalLossV3Adapter(nn.Module):
    """Adapter for V3 loss to match trainer signature."""

    def __init__(self, loss_v3):
        super().__init__()
        self.loss_v3 = loss_v3

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        v3_losses = self.loss_v3(
            params_amb=params, params_clear=params_clear,
            p_star_amb=p_star, entropy_amb=entropy_gt,
            return_components=True,
        )

        loss_dict = {
            'loss_total': v3_losses['loss_total'],
            'loss_kl': v3_losses['loss_answer'],
            'loss_reg': v3_losses['loss_kl'],
            'loss_cal': v3_losses['loss_cal_mse'] + v3_losses['loss_cal_rank'],
            'loss_cont': v3_losses['loss_contrast'],
            'loss_answer': v3_losses['loss_answer'],
            'loss_kl_v3': v3_losses['loss_kl'],
            'loss_cal_mse': v3_losses['loss_cal_mse'],
            'loss_cal_rank': v3_losses['loss_cal_rank'],
            'loss_epi_error': v3_losses['loss_epi_error'],
            'loss_variance': v3_losses['loss_variance'],
        }
        return v3_losses['loss_total'], loss_dict


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Credal CBM')
    parser.add_argument('--dataset', type=str, default='maqa',
                       choices=['cebab', 'hatexplain', 'goemotions', 'maqa'])
    parser.add_argument('--encoder', type=str, default='distilbert')
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--quantization', type=str, default='none', choices=['none', '4bit', '8bit'])
    parser.add_argument('--loss_version', type=str, default='v7b',
                       choices=['v3', 'v4', 'v5', 'v6', 'v7b'])
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    encoder_name = expand_encoder_name(args.encoder)
    encoder_config = get_encoder_config(args.encoder)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Training on {config['name']} | Encoder: {encoder_name}")
    print(f"Device: {device} | Loss: {args.loss_version}")
    print(f"{'='*60}")

    # =========================================================================
    # MAQA PATH
    # =========================================================================
    if config.get('use_maqa_model'):
        if not HAS_MAQA:
            raise ImportError("MAQA components not available")

        # Get config for selected loss version
        loss_configs = {'v3': CONFIG_V3, 'v4': CONFIG_V4, 'v5': CONFIG_V5,
                        'v6': CONFIG_V6, 'v7b': CONFIG_V7B}
        config_to_use = loss_configs[args.loss_version]

        # Apply defaults
        batch_size = args.batch_size or config_to_use.get('batch_size', 16)
        lr = args.lr or config_to_use.get('learning_rate', 2e-5)
        num_epochs = args.num_epochs or config_to_use.get('num_epochs', 15)

        # Load encoder
        encoder, tokenizer = load_encoder_with_quantization(encoder_name, args.quantization, device)
        encoder_config = get_encoder_config(encoder_name)

        # Load data
        print("\nLoading MAQA + AmbigQA data...")
        maqa_data_raw = load_combined_maqa_ambigqa()

        # Preprocess
        maqa_data = {}
        for split in ['train', 'validation', 'test']:
            processed = []
            for item in maqa_data_raw[split]:
                probs = item.get('probabilities', [])
                if not probs:
                    continue
                p = np.array(probs, dtype=np.float32)
                if p.sum() <= 0:
                    continue
                p = p / p.sum()
                entropy = -np.sum(p * np.log(p + 1e-10))
                processed.append({
                    'text': item['question'],
                    'p_star': p.tolist(),
                    'entropy': float(entropy),
                    'ambiguity_level': 0 if entropy < 0.1 else 2,
                    'num_answers': len(p),
                    'answers': item.get('answers', []) or [0]*len(p),
                    'dominant_answer_idx': int(np.argmax(p))
                })
            maqa_data[split] = processed
            print(f"  {split}: {len(processed)} samples")

        max_answers = max(
            max([d['num_answers'] for d in maqa_data['train']], default=0),
            max([d['num_answers'] for d in maqa_data['validation']], default=0),
            max([d['num_answers'] for d in maqa_data['test']], default=0)
        )
        print(f"  Max answers: {max_answers}")

        # Create datasets
        train_dataset = MAQADataset(maqa_data['train'], tokenizer,
                                    max_length=config['loader_kwargs']['max_length'], use_paired=True)
        val_dataset = MAQADataset(maqa_data['validation'], tokenizer,
                                  max_length=config['loader_kwargs']['max_length'], use_paired=False)
        test_dataset = MAQADataset(maqa_data['test'], tokenizer,
                                   max_length=config['loader_kwargs']['max_length'], use_paired=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=maqa_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=maqa_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=maqa_collate_fn)

        # =====================================================================
        # CREATE MODEL & TRAINER BASED ON LOSS VERSION
        # =====================================================================
        if args.loss_version == 'v7b' and HAS_V7B_COMPLETE:
            # V7b with entropy-aware model
            print("\n✓ Using V7b COMPLETE (entropy-aware model)")
            model = CredalMAQA_V7b(
                encoder=encoder,
                hidden_size=encoder_config['hidden_size'],
                num_answers=max_answers,
                projection_dim=config.get('projection_dim', 256),
                dropout=config_to_use.get('dropout', 0.1),
            )
            model = initialize_model_for_v7b(model, config_to_use)
            trainer = MAQACredalTrainerV7b(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, learning_rate=lr, weight_decay=config_to_use.get('weight_decay', 0.01)
            )
            trainer.criterion = create_v7b_adapter(config=config_to_use)
        else:
            # Standard model for V3-V6 (and V7b fallback)
            model = CredalMAQA(
                encoder=encoder,
                hidden_size=encoder_config['hidden_size'],
                num_answers=max_answers,
                projection_dim=config.get('projection_dim', 256),
                dropout=config_to_use.get('dropout', 0.1)
            )

            trainer = MAQACredalTrainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, learning_rate=lr, weight_decay=config_to_use.get('weight_decay', 0.01)
            )

            # Set loss function
            if args.loss_version == 'v7b':
                print("⚠ V7b complete not available, using standard model")
                model = initialize_model_for_v7b(model, config_to_use)
                trainer.criterion = create_v7b_adapter(config=config_to_use)
            elif args.loss_version == 'v6':
                model = initialize_model_for_v4(model)
                loss_fn = MAQACredalLossV6(**{k: v for k, v in config_to_use.items()
                                              if k.startswith('lambda_') or k in ['rank_margin', 'prior_sigma',
                                                                                    'min_sigma_epi', 'max_sigma_epi',
                                                                                    'min_sigma_ale', 'max_sigma_ale']})
                trainer.criterion = MAQACredalLossV6Adapter(loss_fn)
            elif args.loss_version == 'v5':
                model = initialize_model_for_v4(model)
                loss_fn = MAQACredalLossV5(**{k: v for k, v in config_to_use.items()
                                              if k.startswith('lambda_') or k in ['rank_margin', 'prior_sigma',
                                                                                    'min_sigma_epi', 'max_sigma_epi',
                                                                                    'min_sigma_ale', 'max_sigma_ale']})
                trainer.criterion = MAQACredalLossV5Adapter(loss_fn)
            elif args.loss_version == 'v4':
                model = initialize_model_for_v4(model)
                loss_fn = MAQACredalLossV4(**{k: v for k, v in config_to_use.items()
                                              if k.startswith('lambda_') or k in ['rank_margin', 'prior_sigma',
                                                                                    'min_sigma_epi', 'max_sigma_epi',
                                                                                    'min_sigma_ale', 'max_sigma_ale']})
                trainer.criterion = MAQACredalLossV4Adapter(loss_fn)
            else:  # v3
                model = initialize_uncertainty_heads(model, config_to_use)
                loss_fn = MAQACredalLossV3(**{k: v for k, v in config_to_use.items()
                                              if k.startswith('lambda_') or k in ['rank_margin', 'prior_sigma',
                                                                                    'min_sigma_epi', 'max_sigma_epi',
                                                                                    'min_sigma_ale', 'max_sigma_ale']})
                trainer.criterion = MAQACredalLossV3Adapter(loss_fn)

        print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Train
        run_metadata = {
            'dataset': config['name'],
            'encoder': encoder_name,
            'loss_version': args.loss_version,
            'device': device,
        }

        results = train_maqa_model(trainer, num_epochs, config['save_dir'], run_metadata, test_loader)

        # Print final results
        print(f"\n{'='*60}")
        print("Final Test Results")
        print(f"{'='*60}")
        tm = results['test_metrics']
        print(f"  Val Loss: {tm['val_loss']:.4f}")
        print(f"  σ_epi: {tm['mean_sigma_epi']:.4f} | σ_ale: {tm['mean_sigma_ale']:.4f}")
        print(f"  ρ(EU,AU): {tm.get('rho_eu_au', 0):.3f} (target: <0.3)")
        print(f"  ρ(AU,H): {tm.get('rho_au_entropy', 0):.3f} (target: >0.5)")
        return

    # =========================================================================
    # NON-MAQA PATH (CEBaB, HateXplain, GoEmotions)
    # =========================================================================
        tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    loader_kwargs = config['loader_kwargs'].copy()
    if args.batch_size:
        loader_kwargs['batch_size'] = args.batch_size

    # Load data
    if config.get('use_multi_loader') and HAS_MULTI_LOADER:
        ds_config = DatasetConfig(
            max_length=loader_kwargs.get('max_length', 128),
            batch_size=loader_kwargs.get('batch_size', 16),
            tokenizer_name=encoder_name,
        )
        train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
            dataset_name=args.dataset, config=ds_config
        )
    else:
        train_loader, val_loader, test_loader, tokenizer, metadata = config['data_loader'](
            tokenizer=tokenizer, **loader_kwargs
        )

    # Create model
    model_config = HybridCredalConfig(
        encoder_name=encoder_name,
        freeze_encoder=True,
        num_concepts=config['num_concepts'],
        concept_names=config['concept_names'],
        num_classes=config['num_classes'],
        prior_sigma=config['prior_sigma'],
        error_scale=config['error_scale'],
    )
    model = HybridCredalCBM(model_config)

    # Train
    trainer = HybridCredalCBMTrainer(model, model_config, device=device, save_dir=config['save_dir'])
    num_epochs = args.num_epochs or config['num_epochs']
    lr = args.lr or config['learning_rate']

    results = trainer.fit(
        train_loader, val_loader, num_epochs=num_epochs, lr=lr,
        metadata={'dataset': config['name'], 'encoder': encoder_name}
    )

    # Test
    trainer.load_best_model()
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Acc: {test_metrics.accuracy:.4f} | ρ(EU,AU): {test_metrics.rho_eu_au:.3f}")


if __name__ == "__main__":
    main()
