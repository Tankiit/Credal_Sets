"""
Variational Credal Concept Bottleneck Models - Complete Implementation
======================================================================

A production-ready implementation using standard libraries:
- PyTorch Lightning for training
- Hugging Face Transformers for encoders
- torchmetrics for evaluation
- scipy for statistical tests

Paper: "Variational Credal CBMs: Bridging Imprecise Probabilities and Deep Learning"

Author: Tanmoy
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassCalibrationError
import numpy as np
from scipy import stats
from scipy.special import softmax
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any
from pathlib import Path
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
try:
    from sklearn.metrics import f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Complete configuration for Variational Credal CBM"""
    
    # Encoder
    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = False
    encoder_dropout: float = 0.1
    
    # Task
    num_classes: int = 2
    num_concepts: int = 4
    concept_classes: int = 3  # 0=negative, 1=unknown, 2=positive
    
    # Variational inference
    variational_family: Literal["mean_field", "low_rank"] = "mean_field"
    prior_std: float = 1.0
    posterior_init_std: float = 0.05
    num_mc_samples: int = 10
    low_rank_dim: int = 5  # For low_rank family
    
    # Loss weights
    kl_weight: float = 1e-4
    concept_weight: float = 0.5
    aleatoric_weight: float = 0.1
    
    # Aleatoric network
    aleatoric_hidden_dim: int = 64
    aleatoric_dropout: float = 0.3
    
    # Classifier
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.2
    use_concept_probs: bool = True  # Use P(positive) vs mean concept
    
    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 128
    
    # Learning rate scheduler
    use_lr_scheduler: bool = True  # Use ReduceLROnPlateau
    lr_scheduler_factor: float = 0.5  # Factor to reduce LR by
    lr_scheduler_patience: int = 3  # Epochs to wait before reducing LR
    lr_scheduler_min_lr: float = 1e-7  # Minimum learning rate
    lr_scheduler_mode: str = "max"  # "max" for accuracy, "min" for loss
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ModelConfig':
        """Create ModelConfig from argparse Namespace"""
        config_dict = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                # Handle boolean fields
                if isinstance(cls.__dataclass_fields__[field_name].type, type) and cls.__dataclass_fields__[field_name].type == bool:
                    config_dict[field_name] = bool(value)
                else:
                    config_dict[field_name] = value
        return cls(**config_dict)


def add_model_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all ModelConfig parameters as argparse arguments"""
    
    # Encoder arguments
    parser.add_argument('--encoder_name', type=str, default='distilbert-base-uncased',
                       help='HuggingFace model name for encoder')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder parameters during training')
    parser.add_argument('--encoder_dropout', type=float, default=0.1,
                       help='Dropout rate for encoder')
    
    # Task arguments
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of output classes')
    parser.add_argument('--num_concepts', type=int, default=4,
                       help='Number of concepts')
    parser.add_argument('--concept_classes', type=int, default=3,
                       help='Number of concept classes (0=neg, 1=unk, 2=pos)')
    
    # Variational inference arguments
    parser.add_argument('--variational_family', type=str, default='mean_field',
                       choices=['mean_field', 'low_rank'],
                       help='Variational family: mean_field or low_rank')
    parser.add_argument('--prior_std', type=float, default=1.0,
                       help='Standard deviation of prior distribution')
    parser.add_argument('--posterior_init_std', type=float, default=0.05,
                       help='Initial standard deviation of posterior')
    parser.add_argument('--num_mc_samples', type=int, default=10,
                       help='Number of Monte Carlo samples for uncertainty')
    parser.add_argument('--low_rank_dim', type=int, default=5,
                       help='Rank for low-rank variational family')
    
    # Loss weight arguments
    parser.add_argument('--kl_weight', type=float, default=1e-4,
                       help='Weight for KL divergence loss')
    parser.add_argument('--concept_weight', type=float, default=0.5,
                       help='Weight for concept supervision loss')
    parser.add_argument('--aleatoric_weight', type=float, default=0.1,
                       help='Weight for aleatoric uncertainty loss')
    
    # Aleatoric network arguments
    parser.add_argument('--aleatoric_hidden_dim', type=int, default=64,
                       help='Hidden dimension for aleatoric head')
    parser.add_argument('--aleatoric_dropout', type=float, default=0.3,
                       help='Dropout rate for aleatoric head')
    
    # Classifier arguments
    parser.add_argument('--classifier_hidden_dim', type=int, default=128,
                       help='Hidden dimension for task classifier')
    parser.add_argument('--classifier_dropout', type=float, default=0.2,
                       help='Dropout rate for task classifier')
    parser.add_argument('--use_concept_probs', action='store_true', default=True,
                       help='Use concept probabilities instead of mean')
    parser.add_argument('--no_use_concept_probs', dest='use_concept_probs', action='store_false',
                       help='Use mean concept values instead of probabilities')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for tokenization')
    
    # Learning rate scheduler arguments
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True,
                       help='Use ReduceLROnPlateau scheduler (default: True)')
    parser.add_argument('--no_lr_scheduler', dest='use_lr_scheduler', action='store_false',
                       help='Disable ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                       help='Factor to reduce LR by when plateau is reached')
    parser.add_argument('--lr_scheduler_patience', type=int, default=3,
                       help='Number of epochs to wait before reducing LR')
    parser.add_argument('--lr_scheduler_min_lr', type=float, default=1e-7,
                       help='Minimum learning rate')
    parser.add_argument('--lr_scheduler_mode', type=str, default='max', choices=['max', 'min'],
                       help='Mode: "max" for accuracy (reduce when not improving), "min" for loss')
    
    return parser


# ============================================================================
# VARIATIONAL LAYERS
# ============================================================================

class MeanFieldLayer(nn.Module):
    """
    Mean-Field Gaussian Variational Layer
    q(W) = ∏ᵢ N(μᵢ, σᵢ²)
    
    Uses reparameterization trick for differentiable sampling.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        posterior_init_std: float = 0.05
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Initialize
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out')
        nn.init.constant_(self.weight_rho, np.log(np.expm1(posterior_init_std)))
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, np.log(np.expm1(posterior_init_std)))
        
        # Prior (fixed)
        self.register_buffer('prior_std', torch.tensor(prior_std))
    
    def _get_std(self, rho: torch.Tensor) -> torch.Tensor:
        """Convert rho to std using softplus"""
        return F.softplus(rho)
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample weights using reparameterization trick"""
        weight_std = self._get_std(self.weight_rho)
        bias_std = self._get_std(self.bias_rho)
        
        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        
        return weight, bias
    
    def kl_divergence(self) -> torch.Tensor:
        """
        KL[q(W) || p(W)] for Gaussian
        Closed-form solution
        """
        weight_std = self._get_std(self.weight_rho)
        bias_std = self._get_std(self.bias_rho)
        
        kl = 0.5 * torch.sum(
            (weight_std / self.prior_std) ** 2
            + (self.weight_mu / self.prior_std) ** 2
            - 1.0
            - 2.0 * torch.log(weight_std / self.prior_std)
        )
        
        kl += 0.5 * torch.sum(
            (bias_std / self.prior_std) ** 2
            + (self.bias_mu / self.prior_std) ** 2
            - 1.0
            - 2.0 * torch.log(bias_std / self.prior_std)
        )
        
        return kl
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sampled weights"""
        weight, bias = self.sample()
        return F.linear(x, weight, bias)
    
    @property
    def posterior_std(self) -> torch.Tensor:
        """Get posterior standard deviation"""
        return self._get_std(self.weight_rho)


class LowRankLayer(nn.Module):
    """
    Low-Rank Gaussian Variational Layer
    q(W) = N(μ, VVᵀ + D) where V is rank-k
    
    Captures correlations between weights with O(nk) parameters.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 5,
        prior_std: float = 1.0,
        posterior_init_std: float = 0.05
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_params = out_features * in_features
        
        # Mean
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        
        # Low-rank covariance: Σ = VVᵀ + diag(d)
        self.cov_factor = nn.Parameter(torch.zeros(self.num_params, rank))
        self.cov_diag_rho = nn.Parameter(torch.zeros(self.num_params))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Initialize
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out')
        nn.init.normal_(self.cov_factor, std=0.01)
        nn.init.constant_(self.cov_diag_rho, np.log(np.expm1(posterior_init_std)))
        nn.init.constant_(self.bias_rho, np.log(np.expm1(posterior_init_std)))
        
        self.register_buffer('prior_std', torch.tensor(prior_std))
    
    def _get_std(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from low-rank Gaussian"""
        mu_flat = self.weight_mu.flatten()
        cov_diag = self._get_std(self.cov_diag_rho)
        
        # Sample using torch.distributions
        q = dist.LowRankMultivariateNormal(
            loc=mu_flat,
            cov_factor=self.cov_factor,
            cov_diag=cov_diag
        )
        weight_flat = q.rsample()
        weight = weight_flat.view(self.out_features, self.in_features)
        
        # Bias (diagonal)
        bias_std = self._get_std(self.bias_rho)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        
        return weight, bias
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence (Monte Carlo estimate)"""
        mu_flat = self.weight_mu.flatten()
        cov_diag = self._get_std(self.cov_diag_rho)
        
        q = dist.LowRankMultivariateNormal(
            loc=mu_flat,
            cov_factor=self.cov_factor,
            cov_diag=cov_diag
        )
        
        # Diagonal prior for efficiency
        p = dist.Independent(
            dist.Normal(torch.zeros_like(mu_flat), self.prior_std),
            reinterpreted_batch_ndims=1
        )
        
        # MC estimate of KL
        z = q.rsample()
        kl_weights = q.log_prob(z) - p.log_prob(z)
        
        # Bias KL (closed form)
        bias_std = self._get_std(self.bias_rho)
        kl_bias = 0.5 * torch.sum(
            (bias_std / self.prior_std) ** 2
            + (self.bias_mu / self.prior_std) ** 2
            - 1.0
            - 2.0 * torch.log(bias_std / self.prior_std)
        )
        
        return kl_weights + kl_bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self.sample()
        return F.linear(x, weight, bias)


# ============================================================================
# ALEATORIC NETWORK
# ============================================================================

class AleatoricHead(nn.Module):
    """
    Predicts aleatoric (data) uncertainty per concept.
    
    Output is always positive (using Softplus).
    Should correlate with "unknown" labels in concepts.
    """
    
    def __init__(
        self,
        in_features: int,
        num_concepts: int,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        # Initialize to predict low uncertainty initially
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# MAIN MODEL
# ============================================================================

class VariationalCredalCBM(pl.LightningModule):
    """
    Variational Credal Concept Bottleneck Model
    
    A complete PyTorch Lightning module for uncertainty-aware
    concept-based classification.
    
    Features:
    - Variational inference for epistemic uncertainty
    - Heteroscedastic modeling for aleatoric uncertainty
    - Credal set construction from both uncertainties
    - Concept-level interpretability
    
    Usage:
        config = ModelConfig(num_classes=5, num_concepts=4)
        model = VariationalCredalCBM(config)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model, train_loader, val_loader)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict())
        
        # === ENCODER ===
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)
        
        # === VARIATIONAL CONCEPT LAYER ===
        if config.variational_family == "mean_field":
            self.concept_layer = MeanFieldLayer(
                self.hidden_size,
                config.num_concepts,
                config.prior_std,
                config.posterior_init_std
            )
        elif config.variational_family == "low_rank":
            self.concept_layer = LowRankLayer(
                self.hidden_size,
                config.num_concepts,
                config.low_rank_dim,
                config.prior_std,
                config.posterior_init_std
            )
        else:
            raise ValueError(f"Unknown variational family: {config.variational_family}")
        
        # === CONCEPT CLASSIFIER (for supervision) ===
        self.concept_classifier = nn.Linear(
            self.hidden_size,
            config.num_concepts * config.concept_classes
        )
        
        # === ALEATORIC HEAD ===
        self.aleatoric_head = AleatoricHead(
            self.hidden_size,
            config.num_concepts,
            config.aleatoric_hidden_dim,
            config.aleatoric_dropout
        )
        
        # === TASK CLASSIFIER ===
        classifier_input = config.num_concepts
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )
        
        # === METRICS ===
        self._setup_metrics()
        
        # === TRACKING ===
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _setup_metrics(self):
        """Setup torchmetrics"""
        nc = self.config.num_classes
        
        # Task metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=nc)
        self.val_acc = Accuracy(task="multiclass", num_classes=nc)
        self.test_acc = Accuracy(task="multiclass", num_classes=nc)
        
        self.train_f1 = F1Score(task="multiclass", num_classes=nc, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=nc, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=nc, average="macro")
        
        # Calibration
        if nc > 2:
            self.val_ece = MulticlassCalibrationError(num_classes=nc, n_bins=10)
            self.test_ece = MulticlassCalibrationError(num_classes=nc, n_bins=10)
        
        # Concept metrics
        self.train_concept_acc = Accuracy(
            task="multiclass", num_classes=self.config.concept_classes
        )
        self.val_concept_acc = Accuracy(
            task="multiclass", num_classes=self.config.concept_classes
        )
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to hidden representation"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            hidden = outputs.pooler_output
        else:
            # Mean pooling
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        return self.encoder_dropout(hidden)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty quantification.
        
        Returns:
            Dict with:
            - logits: [B, num_classes]
            - predictions: [B]
            - concept_probs: [B, num_concepts] mean concept probabilities
            - concept_logits: [B, num_concepts, 3] ternary classifier
            - epistemic: [B, num_concepts] epistemic uncertainty
            - aleatoric: [B, num_concepts] aleatoric uncertainty
            - credal_lower: [B, num_concepts] credal set lower bound
            - credal_upper: [B, num_concepts] credal set upper bound
        """
        num_samples = num_samples or self.config.num_mc_samples
        batch_size = input_ids.size(0)
        
        # Encode
        hidden = self.encode(input_ids, attention_mask)
        
        # === MC sampling from variational posterior ===
        all_logits = []
        concept_samples = []
        
        for _ in range(num_samples):
            # Sample concept activations
            concept_logits = self.concept_layer(hidden)  # [B, K]
            concept_probs = torch.sigmoid(concept_logits)
            concept_samples.append(concept_probs)
            
            # Classify using THESE concept probs
            task_logits = self.classifier(concept_probs)  # [B, num_classes]
            all_logits.append(task_logits)
        
        concept_samples = torch.stack(concept_samples, dim=0)  # [S, B, K]
        all_logits = torch.stack(all_logits, dim=0)  # [S, B, num_classes]
        
        # Epistemic from BOTH concepts AND predictions
        concept_mean = concept_samples.mean(dim=0)
        epistemic = concept_samples.var(dim=0)
        
        # Final prediction: mean of sampled logits
        logits = all_logits.mean(dim=0)
        predictions = logits.argmax(dim=-1)
        
        # Prediction uncertainty (for epistemic-error correlation)
        prediction_epistemic = all_logits.var(dim=0).mean(dim=-1)  # [B]
        
        # === CONCEPT CLASSIFICATION (ternary) ===
        concept_logits = self.concept_classifier(hidden)
        concept_logits = concept_logits.view(
            batch_size, self.config.num_concepts, self.config.concept_classes
        )
        concept_class_probs = F.softmax(concept_logits, dim=-1)
        
        # === ALEATORIC ===
        aleatoric = self.aleatoric_head(hidden)
        
        # === CREDAL SETS ===
        total_std = torch.sqrt(epistemic + F.softplus(aleatoric))
        z = 1.96  # 95% confidence
        credal_lower = torch.clamp(concept_mean - z * total_std, 0.0, 1.0)
        credal_upper = torch.clamp(concept_mean + z * total_std, 0.0, 1.0)
        
        return {
            'logits': logits,
            'predictions': predictions,
            'concept_probs': concept_mean,
            'concept_logits': concept_logits,
            'concept_class_probs': concept_class_probs,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'prediction_epistemic': prediction_epistemic,
            'credal_lower': credal_lower,
            'credal_upper': credal_upper,
            'credal_width': credal_upper - credal_lower,
            'hidden': hidden
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Forward pass outputs
            labels: Task labels [B]
            concept_labels: Concept labels [B, K] with 0=neg, 1=unk, 2=pos
        
        Returns:
            Dict with loss components
        """
        losses = {}
        
        # === CLASSIFICATION LOSS ===
        losses['ce'] = F.cross_entropy(outputs['logits'], labels)
        
        # === KL DIVERGENCE ===
        losses['kl'] = self.concept_layer.kl_divergence()
        
        # === CONCEPT SUPERVISION ===
        if concept_labels is not None:
            # Masked supervision: exclude unknown (class 1)
            known_mask = concept_labels != 1
            
            if known_mask.any():
                losses['concept'] = F.cross_entropy(
                    outputs['concept_logits'][known_mask],
                    concept_labels[known_mask]
                )
            else:
                losses['concept'] = torch.tensor(0.0, device=labels.device)
            
            # === ALEATORIC SUPERVISION ===
            # Target: high for unknown, low for known
            unknown_target = (concept_labels == 1).float()
            losses['aleatoric'] = F.binary_cross_entropy_with_logits(
                outputs['aleatoric'], 
                unknown_target
            )
        else:
            losses['concept'] = torch.tensor(0.0, device=labels.device)
            losses['aleatoric'] = torch.tensor(0.0, device=labels.device)
        
        # === TOTAL LOSS ===
        losses['total'] = (
            losses['ce']
            + self.config.kl_weight * losses['kl']
            + self.config.concept_weight * losses['concept']
            + self.config.aleatoric_weight * losses['aleatoric']
        )
        
        return losses
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step"""
        outputs = self(batch['input_ids'], batch['attention_mask'])
        
        concept_labels = batch.get('concept_labels', None)
        losses = self.compute_loss(outputs, batch['labels'], concept_labels)
        
        # Log losses
        self.log('train/loss', losses['total'], prog_bar=True)
        self.log('train/ce_loss', losses['ce'])
        self.log('train/kl_loss', losses['kl'])
        self.log('train/concept_loss', losses['concept'])
        self.log('train/aleatoric_loss', losses['aleatoric'])
        
        # Log metrics
        self.train_acc(outputs['predictions'], batch['labels'])
        self.train_f1(outputs['predictions'], batch['labels'])
        self.log('train/acc', self.train_acc, prog_bar=True)
        self.log('train/f1', self.train_f1)
        
        # Log uncertainties
        self.log('train/mean_epistemic', outputs['epistemic'].mean())
        self.log('train/mean_aleatoric', outputs['aleatoric'].mean())
        self.log('train/mean_credal_width', outputs['credal_width'].mean())
        
        # Concept accuracy
        if concept_labels is not None:
            concept_preds = outputs['concept_logits'].argmax(dim=-1)
            self.train_concept_acc(concept_preds.flatten(), concept_labels.flatten())
            self.log('train/concept_acc', self.train_concept_acc)
        
        return losses['total']
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step"""
        outputs = self(batch['input_ids'], batch['attention_mask'])
        
        concept_labels = batch.get('concept_labels', None)
        losses = self.compute_loss(outputs, batch['labels'], concept_labels)
        
        # Log losses
        self.log('val/loss', losses['total'], prog_bar=True)
        self.log('val/ce_loss', losses['ce'])
        
        # Log metrics
        self.val_acc(outputs['predictions'], batch['labels'])
        self.val_f1(outputs['predictions'], batch['labels'])
        self.log('val/acc', self.val_acc, prog_bar=True)
        self.log('val/f1', self.val_f1)
        
        # Calibration
        if hasattr(self, 'val_ece'):
            probs = F.softmax(outputs['logits'], dim=-1)
            self.val_ece(probs, batch['labels'])
            self.log('val/ece', self.val_ece)
        
        # Log uncertainties
        self.log('val/mean_epistemic', outputs['epistemic'].mean())
        self.log('val/mean_aleatoric', outputs['aleatoric'].mean())
        
        # Concept accuracy
        if concept_labels is not None:
            concept_preds = outputs['concept_logits'].argmax(dim=-1)
            self.val_concept_acc(concept_preds.flatten(), concept_labels.flatten())
            self.log('val/concept_acc', self.val_concept_acc)
        
        # Store for epoch-end analysis
        self.validation_step_outputs.append({
            'predictions': outputs['predictions'].cpu(),
            'labels': batch['labels'].cpu(),
            'epistemic': outputs['epistemic'].cpu(),
            'aleatoric': outputs['aleatoric'].cpu(),
            'concept_labels': concept_labels.cpu() if concept_labels is not None else None
        })
        
        return losses
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics"""
        if not self.validation_step_outputs:
            return
        
        # Gather all outputs
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_epistemic = torch.cat([x['epistemic'] for x in self.validation_step_outputs])
        all_aleatoric = torch.cat([x['aleatoric'] for x in self.validation_step_outputs])
        
        # Compute errors
        errors = (all_preds != all_labels).float()
        
        # Epistemic-Error correlation
        if errors.std() > 0 and all_epistemic.mean(dim=1).std() > 0:
            corr, pval = stats.spearmanr(
                all_epistemic.mean(dim=1).numpy(),
                errors.numpy()
            )
            self.log('val/epistemic_error_corr', corr)
            self.log('val/epistemic_error_pval', pval)
        
        # Aleatoric-Unknown correlation
        if self.validation_step_outputs[0]['concept_labels'] is not None:
            all_concepts = torch.cat([x['concept_labels'] for x in self.validation_step_outputs])
            unknown_rate = (all_concepts == 1).float()
            
            if unknown_rate.std() > 0 and all_aleatoric.std() > 0:
                corr, pval = stats.spearmanr(
                    all_aleatoric.flatten().numpy(),
                    unknown_rate.flatten().numpy()
                )
                self.log('val/aleatoric_unknown_corr', corr)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step"""
        outputs = self(batch['input_ids'], batch['attention_mask'])
        
        concept_labels = batch.get('concept_labels', None)
        losses = self.compute_loss(outputs, batch['labels'], concept_labels)
        
        # Metrics
        self.test_acc(outputs['predictions'], batch['labels'])
        self.test_f1(outputs['predictions'], batch['labels'])
        self.log('test/acc', self.test_acc)
        self.log('test/f1', self.test_f1)
        
        if hasattr(self, 'test_ece'):
            probs = F.softmax(outputs['logits'], dim=-1)
            self.test_ece(probs, batch['labels'])
            self.log('test/ece', self.test_ece)
        
        # Store for analysis
        self.test_step_outputs.append({
            'predictions': outputs['predictions'].cpu(),
            'labels': batch['labels'].cpu(),
            'epistemic': outputs['epistemic'].cpu(),
            'aleatoric': outputs['aleatoric'].cpu(),
            'logits': outputs['logits'].cpu(),
            'concept_labels': concept_labels.cpu() if concept_labels is not None else None
        })
        
        return losses
    
    def on_test_epoch_end(self):
        """Compute final test metrics"""
        if not self.test_step_outputs:
            return
        
        # Gather outputs
        all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_epistemic = torch.cat([x['epistemic'] for x in self.test_step_outputs])
        all_aleatoric = torch.cat([x['aleatoric'] for x in self.test_step_outputs])
        
        errors = (all_preds != all_labels).float()
        
        # Correlations
        if errors.std() > 0:
            corr, pval = stats.spearmanr(
                all_epistemic.mean(dim=1).numpy(),
                errors.numpy()
            )
            self.log('test/epistemic_error_corr', corr)
        
        # Aleatoric-Unknown correlation
        if self.test_step_outputs[0]['concept_labels'] is not None:
            all_concepts = torch.cat([x['concept_labels'] for x in self.test_step_outputs])
            unknown_rate = (all_concepts == 1).float()
            
            if unknown_rate.std() > 0:
                corr, _ = stats.spearmanr(
                    all_aleatoric.flatten().numpy(),
                    unknown_rate.flatten().numpy()
                )
                self.log('test/aleatoric_unknown_corr', corr)
        
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Separate encoder and head parameters
        encoder_params = list(self.encoder.parameters())
        head_params = (
            list(self.concept_layer.parameters())
            + list(self.concept_classifier.parameters())
            + list(self.aleatoric_head.parameters())
            + list(self.classifier.parameters())
        )
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.config.learning_rate},
            {'params': head_params, 'lr': self.config.learning_rate * 10}
        ], weight_decay=self.config.weight_decay)
        
        # Configure learning rate scheduler
        if self.config.use_lr_scheduler:
            # Use ReduceLROnPlateau - reduces LR when accuracy plateaus
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.lr_scheduler_mode,
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
                min_lr=self.config.lr_scheduler_min_lr,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'val/acc'  # Monitor validation accuracy
                }
            }
        else:
            # Optional: Use warmup scheduler if LR scheduler is disabled
            if self.config.warmup_ratio > 0:
                if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
                    total_steps = self.trainer.estimated_stepping_batches
                else:
                    total_steps = 1000
                
                warmup_steps = int(total_steps * self.config.warmup_ratio)
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1
                    }
                }
            else:
                return {'optimizer': optimizer}
    
    def predict_with_uncertainty(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        batch_size: int = 32,
        num_samples: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer for encoding
            batch_size: Batch size for inference
            num_samples: MC samples for uncertainty
        
        Returns:
            Dict with predictions, uncertainties, and credal sets
        """
        self.eval()
        device = next(self.parameters()).device
        
        all_results = {
            'predictions': [],
            'probabilities': [],
            'epistemic': [],
            'aleatoric': [],
            'credal_lower': [],
            'credal_upper': [],
            'concept_probs': []
        }
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoding = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.config.max_length,
                    padding=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = self(input_ids, attention_mask, num_samples=num_samples)
                
                all_results['predictions'].append(outputs['predictions'].cpu().numpy())
                all_results['probabilities'].append(
                    F.softmax(outputs['logits'], dim=-1).cpu().numpy()
                )
                all_results['epistemic'].append(outputs['epistemic'].cpu().numpy())
                all_results['aleatoric'].append(outputs['aleatoric'].cpu().numpy())
                all_results['credal_lower'].append(outputs['credal_lower'].cpu().numpy())
                all_results['credal_upper'].append(outputs['credal_upper'].cpu().numpy())
                all_results['concept_probs'].append(outputs['concept_probs'].cpu().numpy())
        
        # Concatenate all batches
        return {k: np.concatenate(v, axis=0) for k, v in all_results.items()}


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_trainer(
    output_dir: str = "./outputs",
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "16-mixed",
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 3,
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 10
) -> pl.Trainer:
    """
    Create a configured PyTorch Lightning trainer.
    
    Args:
        output_dir: Directory for checkpoints and logs
        max_epochs: Maximum training epochs
        accelerator: "auto", "gpu", "cpu", "mps"
        devices: Number of devices
        precision: Training precision ("32", "16-mixed", "bf16-mixed")
        gradient_clip_val: Gradient clipping
        early_stopping_patience: Epochs for early stopping
        log_every_n_steps: Logging frequency
    
    Returns:
        Configured pl.Trainer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            filename="best-{epoch:02d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False
        ),
        ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            filename="epoch-{epoch:02d}",
            every_n_epochs=save_every_n_epochs,
            save_top_k=-1,
            save_last=False
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Add early stopping only if patience > 0
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/acc",
                patience=early_stopping_patience,
                mode="max",
                verbose=True
            )
        )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=log_every_n_steps
    )
    
    return trainer


# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

def quick_start_example():
    """Quick start example showing how to use the model."""
    # 1. Configuration
    config = ModelConfig(
        encoder_name="distilbert-base-uncased",
        num_classes=5,  # CEBaB multiclass
        num_concepts=4,  # food, service, ambiance, noise
        variational_family="mean_field",
        num_mc_samples=10
    )
    
    # 2. Create model
    model = VariationalCredalCBM(config)
    
    return model, config


# ============================================================================
# ERROR ANALYSIS MODULE
# ============================================================================

"""
Variational Credal CBM - Comprehensive Error Analysis & Diagnostics
====================================================================

This module provides detailed analysis tools to understand:
1. Why epistemic-error correlation might be low/negative
2. Whether posterior collapse is occurring
3. How uncertainty relates to error types (adjacent vs distant)
4. Calibration analysis
5. Concept-level uncertainty breakdown

Usage:
    from model import run_full_analysis, ErrorAnalyzer
    
    # After training, run analysis
    analyzer = ErrorAnalyzer(model, test_loader, tokenizer)
    results = analyzer.run_full_analysis()
    analyzer.print_report()
    analyzer.save_report("analysis_results.json")

Author: Tanmoy
Date: December 2024
"""

@dataclass
class UncertaintyStats:
    """Statistics for a single uncertainty type"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float


@dataclass 
class CorrelationResult:
    """Result of a correlation analysis"""
    rho: float
    p_value: float
    n_samples: int
    interpretation: str


@dataclass
class ErrorAnalysisResult:
    """Complete error analysis results"""
    # Basic metrics
    accuracy: float
    f1_macro: float
    n_samples: int
    n_errors: int
    error_rate: float
    
    # Error breakdown
    error_distance_distribution: Dict[int, int]
    adjacent_error_rate: float
    distant_error_rate: float
    
    # Uncertainty statistics
    epistemic_stats: UncertaintyStats
    aleatoric_stats: UncertaintyStats
    
    # Correlations
    epistemic_error_corr: CorrelationResult
    aleatoric_unknown_corr: CorrelationResult
    epistemic_distant_error_corr: Optional[CorrelationResult]
    epistemic_adjacent_error_corr: Optional[CorrelationResult]
    
    # Posterior diagnostics
    posterior_std_mean: float
    posterior_std_min: float
    posterior_std_max: float
    posterior_collapsed: bool
    
    # Calibration
    ece: float
    mce: float
    
    # Per-concept analysis
    concept_stats: Dict[str, Dict]
    
    # Recommendations
    issues_found: List[str]
    recommendations: List[str]


class ErrorAnalyzer:
    """
    Comprehensive analyzer for Variational Credal CBM predictions.
    
    Diagnoses issues with uncertainty decomposition and provides
    actionable recommendations.
    """
    
    def __init__(
        self,
        model,
        test_loader,
        concept_names: Optional[List[str]] = None,
        device: str = "auto"
    ):
        self.model = model
        self.test_loader = test_loader
        self.concept_names = concept_names or [f"concept_{i}" for i in range(model.config.num_concepts)]
        
        if device == "auto":
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        
        # Storage for collected data
        self.predictions = None
        self.labels = None
        self.epistemic = None
        self.aleatoric = None
        self.logits = None
        self.concept_labels = None
        self.concept_probs = None
        self.hidden_states = None
        
        # Results
        self.results = None
        
    def collect_predictions(self, num_mc_samples: int = 50):
        """
        Collect all predictions from test set with extended MC sampling.
        
        Args:
            num_mc_samples: Number of MC samples for uncertainty estimation
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_epistemic = []
        all_aleatoric = []
        all_logits = []
        all_concept_labels = []
        all_concept_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Processing batches"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                concept_labels = batch.get('concept_labels', None)
                
                # Forward with extended MC sampling
                outputs = self.model(
                    input_ids, 
                    attention_mask, 
                    num_samples=num_mc_samples
                )
                
                all_predictions.append(outputs['predictions'].cpu())
                all_labels.append(labels)
                all_epistemic.append(outputs['epistemic'].cpu())
                all_aleatoric.append(outputs['aleatoric'].cpu())
                all_logits.append(outputs['logits'].cpu())
                all_concept_probs.append(outputs['concept_probs'].cpu())
                
                if concept_labels is not None:
                    all_concept_labels.append(concept_labels)
        
        # Concatenate
        self.predictions = torch.cat(all_predictions)
        self.labels = torch.cat(all_labels)
        self.epistemic = torch.cat(all_epistemic)
        self.aleatoric = torch.cat(all_aleatoric)
        self.logits = torch.cat(all_logits)
        self.concept_probs = torch.cat(all_concept_probs)
        
        if all_concept_labels:
            self.concept_labels = torch.cat(all_concept_labels)
        
    def analyze_basic_metrics(self) -> Dict:
        """Compute basic accuracy and error metrics"""
        errors = (self.predictions != self.labels)
        correct = ~errors
        
        # Accuracy
        accuracy = correct.float().mean().item()
        
        # F1 (macro)
        if HAS_SKLEARN:
            f1 = f1_score(self.labels.numpy(), self.predictions.numpy(), average='macro')
        else:
            f1 = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'n_samples': len(self.predictions),
            'n_errors': errors.sum().item(),
            'error_rate': errors.float().mean().item()
        }
    
    def analyze_error_distances(self) -> Dict:
        """Analyze errors by distance between predicted and true class"""
        errors = (self.predictions != self.labels)
        
        if not errors.any():
            return {
                'error_distance_distribution': {},
                'adjacent_error_rate': 0.0,
                'distant_error_rate': 0.0
            }
        
        error_distances = torch.abs(self.predictions[errors] - self.labels[errors])
        
        # Distribution
        distribution = {}
        for d in range(1, self.model.config.num_classes):
            count = (error_distances == d).sum().item()
            distribution[d] = count
        
        # Adjacent (distance=1) vs Distant (distance>1)
        n_errors = errors.sum().item()
        adjacent = (error_distances == 1).sum().item()
        distant = (error_distances > 1).sum().item()
        
        return {
            'error_distance_distribution': distribution,
            'adjacent_error_rate': adjacent / n_errors if n_errors > 0 else 0,
            'distant_error_rate': distant / n_errors if n_errors > 0 else 0,
            'n_adjacent_errors': adjacent,
            'n_distant_errors': distant
        }
    
    def analyze_uncertainty_stats(self) -> Tuple[UncertaintyStats, UncertaintyStats]:
        """Compute statistics for both uncertainty types"""
        def compute_stats(tensor: torch.Tensor) -> UncertaintyStats:
            # Average across concepts if multi-dimensional
            if tensor.dim() > 1:
                tensor = tensor.mean(dim=1)
            
            return UncertaintyStats(
                mean=tensor.mean().item(),
                std=tensor.std().item(),
                min=tensor.min().item(),
                max=tensor.max().item(),
                median=tensor.median().item(),
                q25=tensor.quantile(0.25).item(),
                q75=tensor.quantile(0.75).item()
            )
        
        return compute_stats(self.epistemic), compute_stats(self.aleatoric)
    
    def analyze_correlations(self) -> Dict[str, CorrelationResult]:
        """Compute all relevant correlations"""
        results = {}
        
        errors = (self.predictions != self.labels).float()
        epistemic_mean = self.epistemic.mean(dim=1) if self.epistemic.dim() > 1 else self.epistemic
        aleatoric_mean = self.aleatoric.mean(dim=1) if self.aleatoric.dim() > 1 else self.aleatoric
        
        # 1. Epistemic-Error correlation
        if errors.std() > 0 and epistemic_mean.std() > 0:
            rho, p = stats.spearmanr(epistemic_mean.numpy(), errors.numpy())
            results['epistemic_error'] = CorrelationResult(
                rho=rho, p_value=p, n_samples=len(errors),
                interpretation=self._interpret_correlation(rho, "epistemic-error")
            )
        else:
            results['epistemic_error'] = CorrelationResult(
                rho=0.0, p_value=1.0, n_samples=len(errors),
                interpretation="Cannot compute: zero variance"
            )
        
        # 2. Aleatoric-Unknown correlation
        if self.concept_labels is not None:
            unknown_rate = (self.concept_labels == 1).float()
            aleatoric_flat = self.aleatoric.flatten()
            unknown_flat = unknown_rate.flatten()
            
            if unknown_flat.std() > 0 and aleatoric_flat.std() > 0:
                rho, p = stats.spearmanr(aleatoric_flat.numpy(), unknown_flat.numpy())
                results['aleatoric_unknown'] = CorrelationResult(
                    rho=rho, p_value=p, n_samples=len(aleatoric_flat),
                    interpretation=self._interpret_correlation(rho, "aleatoric-unknown")
                )
        
        # 3. Epistemic correlation for DISTANT errors only
        error_mask = (self.predictions != self.labels)
        if error_mask.any():
            distances = torch.abs(self.predictions - self.labels)
            distant_mask = error_mask & (distances > 1)
            adjacent_mask = error_mask & (distances == 1)
            
            # Distant errors
            if distant_mask.sum() > 10:
                distant_epistemic = epistemic_mean[distant_mask]
                correct_epistemic = epistemic_mean[~error_mask]
                
                # Compare distant errors vs correct
                combined = torch.cat([distant_epistemic, correct_epistemic])
                labels_binary = torch.cat([
                    torch.ones(len(distant_epistemic)),
                    torch.zeros(len(correct_epistemic))
                ])
                
                if combined.std() > 0:
                    rho, p = stats.spearmanr(combined.numpy(), labels_binary.numpy())
                    results['epistemic_distant_error'] = CorrelationResult(
                        rho=rho, p_value=p, n_samples=len(combined),
                        interpretation=self._interpret_correlation(rho, "epistemic-distant_error")
                    )
            
            # Adjacent errors
            if adjacent_mask.sum() > 10:
                adjacent_epistemic = epistemic_mean[adjacent_mask]
                correct_epistemic = epistemic_mean[~error_mask]
                
                combined = torch.cat([adjacent_epistemic, correct_epistemic])
                labels_binary = torch.cat([
                    torch.ones(len(adjacent_epistemic)),
                    torch.zeros(len(correct_epistemic))
                ])
                
                if combined.std() > 0:
                    rho, p = stats.spearmanr(combined.numpy(), labels_binary.numpy())
                    results['epistemic_adjacent_error'] = CorrelationResult(
                        rho=rho, p_value=p, n_samples=len(combined),
                        interpretation=self._interpret_correlation(rho, "epistemic-adjacent_error")
                    )
        
        return results
    
    def _interpret_correlation(self, rho: float, corr_type: str) -> str:
        """Interpret correlation value"""
        if corr_type == "epistemic-error":
            if rho > 0.3:
                return "Strong positive - epistemic predicts errors well"
            elif rho > 0.1:
                return "Weak positive - epistemic has some predictive value"
            elif rho > -0.1:
                return "Near zero - epistemic doesn't predict errors"
            else:
                return "Negative - PROBLEM: high epistemic on correct predictions"
        
        elif corr_type == "aleatoric-unknown":
            if rho > 0.5:
                return "Strong positive - aleatoric captures ambiguity well"
            elif rho > 0.3:
                return "Moderate positive - aleatoric partially captures ambiguity"
            elif rho > 0.1:
                return "Weak positive - some aleatoric signal"
            else:
                return "Low/negative - aleatoric not learning ambiguity"
        
        elif corr_type == "epistemic-distant_error":
            if rho > 0.2:
                return "Positive - epistemic correctly identifies serious errors"
            else:
                return "Low - epistemic doesn't distinguish serious errors"
        
        return f"rho={rho:.3f}"
    
    def analyze_posterior(self) -> Dict:
        """Analyze variational posterior for collapse"""
        layer = self.model.concept_layer
        
        if hasattr(layer, 'posterior_std'):
            posterior_std = layer.posterior_std.detach().cpu()
        elif hasattr(layer, '_get_std') and hasattr(layer, 'weight_rho'):
            posterior_std = layer._get_std(layer.weight_rho).detach().cpu()
        else:
            return {
                'posterior_std_mean': None,
                'posterior_std_min': None,
                'posterior_std_max': None,
                'posterior_collapsed': None
            }
        
        std_mean = posterior_std.mean().item()
        std_min = posterior_std.min().item()
        std_max = posterior_std.max().item()
        
        # Check for collapse (very small std)
        collapsed = std_min < 1e-4 or (std_max / (std_min + 1e-8)) > 1000
        
        return {
            'posterior_std_mean': std_mean,
            'posterior_std_min': std_min,
            'posterior_std_max': std_max,
            'posterior_collapsed': collapsed
        }
    
    def analyze_calibration(self) -> Dict:
        """Compute calibration metrics (ECE, MCE)"""
        probs = F.softmax(self.logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == self.labels).float()
        
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                gap = torch.abs(avg_accuracy - avg_confidence)
                
                ece += prop_in_bin * gap
                mce = max(mce, gap.item())
        
        return {
            'ece': ece.item(),
            'mce': mce
        }
    
    def analyze_concepts(self) -> Dict[str, Dict]:
        """Per-concept uncertainty analysis"""
        if self.concept_labels is None:
            return {}
        
        results = {}
        
        for i, name in enumerate(self.concept_names):
            concept_epistemic = self.epistemic[:, i] if self.epistemic.dim() > 1 else self.epistemic
            concept_aleatoric = self.aleatoric[:, i] if self.aleatoric.dim() > 1 else self.aleatoric
            concept_label = self.concept_labels[:, i]
            
            # Unknown rate
            unknown_rate = (concept_label == 1).float().mean().item()
            
            # Uncertainty by label type
            unknown_mask = concept_label == 1
            known_mask = concept_label != 1
            
            results[name] = {
                'unknown_rate': unknown_rate,
                'epistemic_mean': concept_epistemic.mean().item(),
                'aleatoric_mean': concept_aleatoric.mean().item(),
                'epistemic_on_unknown': concept_epistemic[unknown_mask].mean().item() if unknown_mask.any() else 0,
                'epistemic_on_known': concept_epistemic[known_mask].mean().item() if known_mask.any() else 0,
                'aleatoric_on_unknown': concept_aleatoric[unknown_mask].mean().item() if unknown_mask.any() else 0,
                'aleatoric_on_known': concept_aleatoric[known_mask].mean().item() if known_mask.any() else 0,
            }
            
            # Aleatoric should be higher on unknown
            if unknown_mask.any() and known_mask.any():
                ratio = results[name]['aleatoric_on_unknown'] / (results[name]['aleatoric_on_known'] + 1e-8)
                results[name]['aleatoric_unknown_ratio'] = ratio
                results[name]['aleatoric_correctly_higher'] = ratio > 1.0
        
        return results
    
    def diagnose_issues(self) -> Tuple[List[str], List[str]]:
        """Identify issues and provide recommendations"""
        issues = []
        recommendations = []
        
        # Check epistemic-error correlation
        if hasattr(self, '_correlations'):
            epi_err = self._correlations.get('epistemic_error')
            if epi_err and epi_err.rho < 0:
                issues.append(f"Negative epistemic-error correlation (ρ={epi_err.rho:.3f})")
                recommendations.append("Check if most errors are adjacent-class (aleatoric, not epistemic)")
                recommendations.append("Try increasing KL weight to prevent posterior collapse")
                recommendations.append("Consider running binary classification to isolate epistemic signal")
        
        # Check posterior collapse
        if hasattr(self, '_posterior'):
            if self._posterior.get('posterior_collapsed'):
                issues.append("Posterior collapse detected (very small variance)")
                recommendations.append("Increase KL weight (try 1e-3 or 1e-2)")
                recommendations.append("Use KL annealing (gradually increase KL weight)")
                recommendations.append("Check prior_std setting (try smaller value like 0.1)")
        
        # Check calibration
        if hasattr(self, '_calibration'):
            ece = self._calibration.get('ece', 0)
            if ece > 0.1:
                issues.append(f"Poor calibration (ECE={ece:.3f})")
                recommendations.append("Apply temperature scaling post-training")
                recommendations.append("Consider using focal loss")
        
        # Check aleatoric-unknown correlation
        if hasattr(self, '_correlations'):
            ale_unk = self._correlations.get('aleatoric_unknown')
            if ale_unk and ale_unk.rho < 0.3:
                issues.append(f"Weak aleatoric-unknown correlation (ρ={ale_unk.rho:.3f})")
                recommendations.append("Check concept encoding (should be 0=neg, 1=unknown, 2=pos)")
                recommendations.append("Increase aleatoric_weight in loss")
        
        # Check adjacent vs distant error epistemic
        if hasattr(self, '_correlations'):
            epi_dist = self._correlations.get('epistemic_distant_error')
            epi_adj = self._correlations.get('epistemic_adjacent_error')
            
            if epi_dist and epi_adj:
                if epi_dist.rho > epi_adj.rho + 0.1:
                    # This is actually GOOD - epistemic better predicts distant errors
                    pass  
                elif epi_dist.rho <= epi_adj.rho:
                    issues.append("Epistemic doesn't distinguish distant from adjacent errors")
                    recommendations.append("Consider architectural changes to concept layer")
        
        if not issues:
            issues.append("No major issues detected")
        
        return issues, recommendations
    
    def run_full_analysis(self, num_mc_samples: int = 50) -> ErrorAnalysisResult:
        """Run complete analysis pipeline"""
        # Collect predictions
        self.collect_predictions(num_mc_samples)
        
        # Run all analyses
        basic = self.analyze_basic_metrics()
        errors = self.analyze_error_distances()
        epistemic_stats, aleatoric_stats = self.analyze_uncertainty_stats()
        self._correlations = self.analyze_correlations()
        self._posterior = self.analyze_posterior()
        self._calibration = self.analyze_calibration()
        concept_stats = self.analyze_concepts()
        issues, recommendations = self.diagnose_issues()
        
        # Compile results
        self.results = ErrorAnalysisResult(
            accuracy=basic['accuracy'],
            f1_macro=basic['f1_macro'],
            n_samples=basic['n_samples'],
            n_errors=basic['n_errors'],
            error_rate=basic['error_rate'],
            error_distance_distribution=errors['error_distance_distribution'],
            adjacent_error_rate=errors['adjacent_error_rate'],
            distant_error_rate=errors['distant_error_rate'],
            epistemic_stats=epistemic_stats,
            aleatoric_stats=aleatoric_stats,
            epistemic_error_corr=self._correlations.get('epistemic_error'),
            aleatoric_unknown_corr=self._correlations.get('aleatoric_unknown'),
            epistemic_distant_error_corr=self._correlations.get('epistemic_distant_error'),
            epistemic_adjacent_error_corr=self._correlations.get('epistemic_adjacent_error'),
            posterior_std_mean=self._posterior.get('posterior_std_mean'),
            posterior_std_min=self._posterior.get('posterior_std_min'),
            posterior_std_max=self._posterior.get('posterior_std_max'),
            posterior_collapsed=self._posterior.get('posterior_collapsed', False),
            ece=self._calibration['ece'],
            mce=self._calibration['mce'],
            concept_stats=concept_stats,
            issues_found=issues,
            recommendations=recommendations
        )
        
        return self.results
    
    def print_report(self):
        """Print formatted analysis report"""
        if self.results is None:
            return
        
        r = self.results
    
    def save_report(self, filepath: str):
        """Save analysis results to JSON"""
        if self.results is None:
            return
        
        # Convert to dict
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_dict(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(to_dict(self.results), f, indent=2)
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Generate visualization plots"""
        if not HAS_PLOTTING:
            return
            
        if self.results is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Error distance distribution
        ax = axes[0, 0]
        distances = list(self.results.error_distance_distribution.keys())
        counts = list(self.results.error_distance_distribution.values())
        ax.bar(distances, counts, color=['#ff6b6b' if d == 1 else '#4ecdc4' for d in distances])
        ax.set_xlabel('Error Distance')
        ax.set_ylabel('Count')
        ax.set_title('Error Distance Distribution')
        ax.set_xticks(distances)
        
        # 2. Uncertainty distributions
        ax = axes[0, 1]
        epistemic_mean = self.epistemic.mean(dim=1).numpy() if self.epistemic.dim() > 1 else self.epistemic.numpy()
        aleatoric_mean = self.aleatoric.mean(dim=1).numpy() if self.aleatoric.dim() > 1 else self.aleatoric.numpy()
        ax.hist(epistemic_mean, bins=50, alpha=0.5, label='Epistemic', color='blue')
        ax.hist(aleatoric_mean, bins=50, alpha=0.5, label='Aleatoric', color='orange')
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Count')
        ax.set_title('Uncertainty Distributions')
        ax.legend()
        
        # 3. Uncertainty vs Correctness
        ax = axes[0, 2]
        errors = (self.predictions != self.labels).numpy()
        ax.boxplot([epistemic_mean[~errors], epistemic_mean[errors]], 
                   labels=['Correct', 'Error'])
        ax.set_ylabel('Epistemic Uncertainty')
        ax.set_title('Epistemic by Correctness')
        
        # 4. Aleatoric vs Unknown
        ax = axes[1, 0]
        if self.concept_labels is not None:
            unknown_rate = (self.concept_labels == 1).float().mean(dim=1).numpy()
            ax.scatter(unknown_rate, aleatoric_mean, alpha=0.3, s=10)
            ax.set_xlabel('Unknown Rate')
            ax.set_ylabel('Aleatoric Uncertainty')
            if self.results.aleatoric_unknown_corr:
                ax.set_title(f'Aleatoric vs Unknown (ρ={self.results.aleatoric_unknown_corr.rho:.3f})')
        
        # 5. Calibration curve
        ax = axes[1, 1]
        probs = F.softmax(self.logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == self.labels).float()
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.any():
                bin_accs.append(accuracies[in_bin].mean().item())
                bin_confs.append(confidences[in_bin].mean().item())
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, label='Model')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Calibration (ECE={self.results.ece:.3f})')
        ax.legend()
        
        # 6. Concept-wise uncertainty
        ax = axes[1, 2]
        if self.results.concept_stats:
            concepts = list(self.results.concept_stats.keys())
            ale_unknown = [self.results.concept_stats[c]['aleatoric_on_unknown'] for c in concepts]
            ale_known = [self.results.concept_stats[c]['aleatoric_on_known'] for c in concepts]
            
            x = np.arange(len(concepts))
            width = 0.35
            ax.bar(x - width/2, ale_unknown, width, label='On Unknown', color='#ff6b6b')
            ax.bar(x + width/2, ale_known, width, label='On Known', color='#4ecdc4')
            ax.set_xticks(x)
            ax.set_xticklabels(concepts, rotation=45, ha='right')
            ax.set_ylabel('Aleatoric Uncertainty')
            ax.set_title('Aleatoric by Concept & Label Type')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def run_full_analysis(
    model,
    test_loader,
    concept_names: Optional[List[str]] = None,
    num_mc_samples: int = 50,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None
) -> ErrorAnalysisResult:
    """
    Convenience function to run full analysis.
    
    Args:
        model: Trained VariationalCredalCBM model
        test_loader: Test data loader
        concept_names: Names of concepts
        num_mc_samples: MC samples for uncertainty
        save_path: Path to save JSON report
        plot_path: Path to save plots
    
    Returns:
        ErrorAnalysisResult
    """
    analyzer = ErrorAnalyzer(model, test_loader, concept_names)
    results = analyzer.run_full_analysis(num_mc_samples)
    analyzer.print_report()
    
    if save_path:
        analyzer.save_report(save_path)
    
    if plot_path:
        analyzer.plot_analysis(plot_path)
    
    return results


def quick_diagnosis(
    model,
    test_loader,
    concept_names: Optional[List[str]] = None
) -> None:
    """
    Quick diagnosis that just prints key metrics.
    
    Args:
        model: Trained VariationalCredalCBM model
        test_loader: Test data loader
        concept_names: Names of concepts
    """
    analyzer = ErrorAnalyzer(model, test_loader, concept_names)
    analyzer.collect_predictions(num_mc_samples=20)
    
    basic = analyzer.analyze_basic_metrics()
    errors = analyzer.analyze_error_distances()
    correlations = analyzer.analyze_correlations()
    posterior = analyzer.analyze_posterior()


# ============================================================================
# ERROR DISTANCE ANALYSIS MODULE
# ============================================================================

"""
Error Distance Analysis for Variational Credal CBM
===================================================

Analyzes how epistemic and aleatoric uncertainty relate to 
different types of errors (adjacent vs distant class predictions).

Key Hypothesis:
- Adjacent errors (e.g., 3->4 stars) are ALEATORIC (inherent ambiguity)
- Distant errors (e.g., 1->5 stars) are EPISTEMIC (model failure)

Usage:
    from model import analyze_error_distances
    results = analyze_error_distances(model, test_loader)

Author: Tanmoy
Date: December 2024
"""

@dataclass
class ErrorDistanceResults:
    """Complete results from error distance analysis"""
    
    # Basic stats
    n_samples: int
    n_errors: int
    n_correct: int
    accuracy: float
    
    # Error distribution
    error_counts_by_distance: Dict[int, int]
    error_percentages_by_distance: Dict[int, float]
    n_adjacent_errors: int
    n_distant_errors: int
    adjacent_error_rate: float
    distant_error_rate: float
    
    # Uncertainty by correctness
    epistemic_correct_mean: float
    epistemic_correct_std: float
    epistemic_error_mean: float
    epistemic_error_std: float
    aleatoric_correct_mean: float
    aleatoric_correct_std: float
    aleatoric_error_mean: float
    aleatoric_error_std: float
    
    # Uncertainty by error distance
    epistemic_by_distance: Dict[int, Tuple[float, float]]  # distance -> (mean, std)
    aleatoric_by_distance: Dict[int, Tuple[float, float]]
    
    # Correlations
    overall_epistemic_error_corr: float
    overall_epistemic_error_pval: float
    adjacent_epistemic_error_corr: float
    adjacent_epistemic_error_pval: float
    distant_epistemic_error_corr: float
    distant_epistemic_error_pval: float
    
    # Key ratios
    epistemic_error_vs_correct_ratio: float
    epistemic_distant_vs_adjacent_ratio: float
    aleatoric_adjacent_vs_distant_ratio: float
    
    # Statistical tests
    epistemic_ttest_error_vs_correct: Tuple[float, float]  # (t-stat, p-value)
    epistemic_ttest_distant_vs_adjacent: Tuple[float, float]
    aleatoric_ttest_adjacent_vs_distant: Tuple[float, float]


def analyze_error_distances(
    model,
    test_loader,
    num_mc_samples: int = 50,
    device: str = "auto"
) -> ErrorDistanceResults:
    """
    Comprehensive error distance analysis.
    
    Args:
        model: Trained VariationalCredalCBM
        test_loader: Test data loader
        num_mc_samples: MC samples for uncertainty
        device: Device to run on
    
    Returns:
        ErrorDistanceResults with all analysis
    """
    if device == "auto":
        device = next(model.parameters()).device
    
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_epistemic = []
    all_aleatoric = []
    all_pred_epistemic = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask, num_samples=num_mc_samples)
            
            all_preds.append(outputs['predictions'].cpu())
            all_labels.append(batch['labels'])
            all_epistemic.append(outputs['epistemic'].cpu())
            all_aleatoric.append(outputs['aleatoric'].cpu())
            
            if 'prediction_epistemic' in outputs:
                all_pred_epistemic.append(outputs['prediction_epistemic'].cpu())
    
    # Concatenate
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    epistemic = torch.cat(all_epistemic)
    aleatoric = torch.cat(all_aleatoric)
    
    # Use prediction epistemic if available, else mean of concept epistemic
    if all_pred_epistemic:
        pred_epistemic = torch.cat(all_pred_epistemic)
    else:
        pred_epistemic = epistemic.mean(dim=1)
    
    # Mean across concepts for per-sample uncertainty
    epistemic_mean = epistemic.mean(dim=1) if epistemic.dim() > 1 else epistemic
    aleatoric_mean = aleatoric.mean(dim=1) if aleatoric.dim() > 1 else aleatoric
    
    # Apply softplus to aleatoric if raw logits
    aleatoric_mean = F.softplus(aleatoric_mean)
    
    # Basic stats
    n_samples = len(preds)
    errors_mask = (preds != labels)
    correct_mask = ~errors_mask
    n_errors = errors_mask.sum().item()
    n_correct = correct_mask.sum().item()
    accuracy = n_correct / n_samples
    
    # Error distance analysis
    error_distances = torch.abs(preds[errors_mask] - labels[errors_mask])
    
    # Count by distance
    num_classes = model.config.num_classes
    error_counts = {}
    error_percentages = {}
    
    for d in range(1, num_classes):
        count = (error_distances == d).sum().item()
        error_counts[d] = count
        error_percentages[d] = count / n_errors * 100 if n_errors > 0 else 0
    
    # Adjacent (d=1) vs Distant (d>1)
    n_adjacent = error_counts.get(1, 0)
    n_distant = sum(error_counts.get(d, 0) for d in range(2, num_classes))
    
    # Uncertainty by correctness
    epi_correct = pred_epistemic[correct_mask]
    epi_error = pred_epistemic[errors_mask]
    ale_correct = aleatoric_mean[correct_mask]
    ale_error = aleatoric_mean[errors_mask]
    
    # T-test for significance
    from scipy.stats import ttest_ind
    t_epi, p_epi = ttest_ind(epi_error.numpy(), epi_correct.numpy())
    t_ale, p_ale = ttest_ind(ale_error.numpy(), ale_correct.numpy())
    
    # Uncertainty by error distance
    epistemic_by_dist = {}
    aleatoric_by_dist = {}
    
    # Get indices of errors
    error_indices = torch.where(errors_mask)[0]
    
    for d in range(1, num_classes):
        dist_mask = error_distances == d
        if dist_mask.sum() > 0:
            dist_indices = error_indices[dist_mask]
            
            epi_d = pred_epistemic[dist_indices]
            ale_d = aleatoric_mean[dist_indices]
            
            epistemic_by_dist[d] = (epi_d.mean().item(), epi_d.std().item())
            aleatoric_by_dist[d] = (ale_d.mean().item(), ale_d.std().item())
    
    # Adjacent vs Distant comparison
    adjacent_mask = error_distances == 1
    distant_mask = error_distances > 1
    
    adjacent_indices = error_indices[adjacent_mask]
    distant_indices = error_indices[distant_mask]
    
    epi_adjacent = pred_epistemic[adjacent_indices]
    epi_distant = pred_epistemic[distant_indices]
    ale_adjacent = aleatoric_mean[adjacent_indices]
    ale_distant = aleatoric_mean[distant_indices]
    
    epi_dist_adj_ratio = epi_distant.mean() / epi_adjacent.mean() if epi_adjacent.mean() > 0 else 0
    ale_adj_dist_ratio = ale_adjacent.mean() / ale_distant.mean() if ale_distant.mean() > 0 else 0
    
    # T-tests
    if len(epi_distant) > 5 and len(epi_adjacent) > 5:
        t_epi_dist, p_epi_dist = ttest_ind(epi_distant.numpy(), epi_adjacent.numpy())
        t_ale_dist, p_ale_dist = ttest_ind(ale_adjacent.numpy(), ale_distant.numpy())
    else:
        t_epi_dist, p_epi_dist = 0, 1
        t_ale_dist, p_ale_dist = 0, 1
    
    # Correlation analysis
    errors_float = errors_mask.float()
    
    # Overall epistemic-error correlation
    if errors_float.std() > 0 and pred_epistemic.std() > 0:
        rho_overall, p_overall = stats.spearmanr(pred_epistemic.numpy(), errors_float.numpy())
    else:
        rho_overall, p_overall = 0, 1
    
    # Correlation for adjacent errors vs correct
    if n_adjacent > 10:
        adjacent_or_correct = correct_mask | torch.isin(torch.arange(n_samples), adjacent_indices)
        subset_epistemic = pred_epistemic[adjacent_or_correct]
        subset_errors = torch.zeros(n_samples)
        subset_errors[adjacent_indices] = 1
        subset_errors = subset_errors[adjacent_or_correct]
        
        if subset_errors.std() > 0:
            rho_adj, p_adj = stats.spearmanr(subset_epistemic.numpy(), subset_errors.numpy())
        else:
            rho_adj, p_adj = 0, 1
    else:
        rho_adj, p_adj = 0, 1
    
    # Correlation for distant errors vs correct
    if n_distant > 10:
        distant_or_correct = correct_mask | torch.isin(torch.arange(n_samples), distant_indices)
        subset_epistemic = pred_epistemic[distant_or_correct]
        subset_errors = torch.zeros(n_samples)
        subset_errors[distant_indices] = 1
        subset_errors = subset_errors[distant_or_correct]
        
        if subset_errors.std() > 0:
            rho_dist, p_dist = stats.spearmanr(subset_epistemic.numpy(), subset_errors.numpy())
        else:
            rho_dist, p_dist = 0, 1
    else:
        rho_dist, p_dist = 0, 1
    
    # Compile results
    results = ErrorDistanceResults(
        n_samples=n_samples,
        n_errors=n_errors,
        n_correct=n_correct,
        accuracy=accuracy,
        error_counts_by_distance=error_counts,
        error_percentages_by_distance=error_percentages,
        n_adjacent_errors=n_adjacent,
        n_distant_errors=n_distant,
        adjacent_error_rate=n_adjacent / n_errors if n_errors > 0 else 0,
        distant_error_rate=n_distant / n_errors if n_errors > 0 else 0,
        epistemic_correct_mean=epi_correct.mean().item(),
        epistemic_correct_std=epi_correct.std().item(),
        epistemic_error_mean=epi_error.mean().item(),
        epistemic_error_std=epi_error.std().item(),
        aleatoric_correct_mean=ale_correct.mean().item(),
        aleatoric_correct_std=ale_correct.std().item(),
        aleatoric_error_mean=ale_error.mean().item(),
        aleatoric_error_std=ale_error.std().item(),
        epistemic_by_distance=epistemic_by_dist,
        aleatoric_by_distance=aleatoric_by_dist,
        overall_epistemic_error_corr=rho_overall,
        overall_epistemic_error_pval=p_overall,
        adjacent_epistemic_error_corr=rho_adj,
        adjacent_epistemic_error_pval=p_adj,
        distant_epistemic_error_corr=rho_dist,
        distant_epistemic_error_pval=p_dist,
        epistemic_error_vs_correct_ratio=epi_error.mean().item() / epi_correct.mean().item() if epi_correct.mean() > 0 else 0,
        epistemic_distant_vs_adjacent_ratio=epi_dist_adj_ratio,
        aleatoric_adjacent_vs_distant_ratio=ale_adj_dist_ratio,
        epistemic_ttest_error_vs_correct=(t_epi, p_epi),
        epistemic_ttest_distant_vs_adjacent=(t_epi_dist, p_epi_dist),
        aleatoric_ttest_adjacent_vs_distant=(t_ale_dist, p_ale_dist)
    )
    
    return results


def plot_error_distance_analysis(
    results: ErrorDistanceResults,
    save_path: Optional[str] = None
):
    """Create visualization of error distance analysis"""
    
    if not HAS_PLOTTING:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Error distribution by distance
    ax = axes[0, 0]
    distances = list(results.error_counts_by_distance.keys())
    counts = list(results.error_counts_by_distance.values())
    colors = ['#ff6b6b' if d == 1 else '#4ecdc4' for d in distances]
    bars = ax.bar(distances, counts, color=colors)
    ax.set_xlabel('Error Distance', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Distribution by Class Distance', fontsize=14)
    ax.set_xticks(distances)
    
    # Add percentage labels
    for bar, pct in zip(bars, results.error_percentages_by_distance.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff6b6b', label='Adjacent (d=1)'),
                       Patch(facecolor='#4ecdc4', label='Distant (d>1)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 2. Epistemic by distance
    ax = axes[0, 1]
    distances = list(results.epistemic_by_distance.keys())
    means = [results.epistemic_by_distance[d][0] for d in distances]
    stds = [results.epistemic_by_distance[d][1] for d in distances]
    
    ax.bar(distances, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xlabel('Error Distance', fontsize=12)
    ax.set_ylabel('Epistemic Uncertainty', fontsize=12)
    ax.set_title('Epistemic Uncertainty by Error Distance', fontsize=14)
    ax.set_xticks(distances)
    
    # Add trend line
    if len(distances) > 1:
        z = np.polyfit(distances, means, 1)
        p = np.poly1d(z)
        ax.plot(distances, p(distances), "r--", alpha=0.8, label='Trend')
        ax.legend()
    
    # 3. Aleatoric by distance
    ax = axes[1, 0]
    distances = list(results.aleatoric_by_distance.keys())
    means = [results.aleatoric_by_distance[d][0] for d in distances]
    stds = [results.aleatoric_by_distance[d][1] for d in distances]
    
    ax.bar(distances, means, yerr=stds, capsize=5, color='coral', alpha=0.7)
    ax.set_xlabel('Error Distance', fontsize=12)
    ax.set_ylabel('Aleatoric Uncertainty', fontsize=12)
    ax.set_title('Aleatoric Uncertainty by Error Distance', fontsize=14)
    ax.set_xticks(distances)
    
    # Add trend line
    if len(distances) > 1:
        z = np.polyfit(distances, means, 1)
        p = np.poly1d(z)
        ax.plot(distances, p(distances), "r--", alpha=0.8, label='Trend')
        ax.legend()
    
    # 4. Comparison bar chart
    ax = axes[1, 1]
    
    categories = ['Correct', 'Adjacent\nError', 'Distant\nError']
    epistemic_vals = [
        results.epistemic_correct_mean,
        results.epistemic_by_distance.get(1, (0, 0))[0],
        np.mean([results.epistemic_by_distance.get(d, (0, 0))[0] 
                 for d in results.epistemic_by_distance.keys() if d > 1]) if any(d > 1 for d in results.epistemic_by_distance.keys()) else 0
    ]
    aleatoric_vals = [
        results.aleatoric_correct_mean,
        results.aleatoric_by_distance.get(1, (0, 0))[0],
        np.mean([results.aleatoric_by_distance.get(d, (0, 0))[0] 
                 for d in results.aleatoric_by_distance.keys() if d > 1]) if any(d > 1 for d in results.aleatoric_by_distance.keys()) else 0
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, epistemic_vals, width, label='Epistemic', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, aleatoric_vals, width, label='Aleatoric', color='coral', alpha=0.7)
    
    ax.set_ylabel('Uncertainty', fontsize=12)
    ax.set_title('Uncertainty Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_paper_table(results: ErrorDistanceResults) -> str:
    """Generate LaTeX table for paper"""
    
    distant_epi = np.mean([results.epistemic_by_distance.get(d, (0,0))[0] 
                          for d in results.epistemic_by_distance.keys() if d > 1])
    distant_ale = np.mean([results.aleatoric_by_distance.get(d, (0,0))[0] 
                          for d in results.aleatoric_by_distance.keys() if d > 1])
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Error Distance Analysis on CEBaB (5-class)}
\label{tab:error_distance}
\begin{tabular}{lcccc}
\toprule
\textbf{Category} & \textbf{N} & \textbf{Epistemic} & \textbf{Aleatoric} & \textbf{$\rho$(epi, err)} \\
\midrule
Correct & """ + f"{results.n_correct}" + r""" & """ + f"{results.epistemic_correct_mean:.4f}" + r""" & """ + f"{results.aleatoric_correct_mean:.4f}" + r""" & -- \\
\midrule
Adjacent (d=1) & """ + f"{results.n_adjacent_errors}" + r""" & """ + f"{results.epistemic_by_distance.get(1, (0,0))[0]:.4f}" + r""" & """ + f"{results.aleatoric_by_distance.get(1, (0,0))[0]:.4f}" + r""" & """ + f"{results.adjacent_epistemic_error_corr:.3f}" + r""" \\
Distant (d>1) & """ + f"{results.n_distant_errors}" + r""" & """ + f"{distant_epi:.4f}" + r""" & """ + f"{distant_ale:.4f}" + r""" & """ + f"{results.distant_epistemic_error_corr:.3f}" + r""" \\
\midrule
\textbf{Ratio} & & \textbf{""" + f"{results.epistemic_distant_vs_adjacent_ratio:.2f}" + r"""x} & \textbf{""" + f"{results.aleatoric_adjacent_vs_distant_ratio:.2f}" + r"""x} & \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(results: ErrorDistanceResults) -> str:
    """Generate Markdown table for documentation"""
    
    distant_epi = np.mean([results.epistemic_by_distance.get(d, (0,0))[0] 
                          for d in results.epistemic_by_distance.keys() if d > 1])
    distant_ale = np.mean([results.aleatoric_by_distance.get(d, (0,0))[0] 
                          for d in results.aleatoric_by_distance.keys() if d > 1])
    
    md = f"""
## Error Distance Analysis Results

| Category | N | Epistemic | Aleatoric | ρ(epi, err) |
|----------|---|-----------|-----------|-------------|
| Correct | {results.n_correct} | {results.epistemic_correct_mean:.4f} | {results.aleatoric_correct_mean:.4f} | -- |
| Adjacent (d=1) | {results.n_adjacent_errors} | {results.epistemic_by_distance.get(1, (0,0))[0]:.4f} | {results.aleatoric_by_distance.get(1, (0,0))[0]:.4f} | {results.adjacent_epistemic_error_corr:.3f} |
| Distant (d>1) | {results.n_distant_errors} | {distant_epi:.4f} | {distant_ale:.4f} | {results.distant_epistemic_error_corr:.3f} |
| **Ratio** | | **{results.epistemic_distant_vs_adjacent_ratio:.2f}x** | **{results.aleatoric_adjacent_vs_distant_ratio:.2f}x** | |

### Key Findings

- Adjacent errors: {results.n_adjacent_errors} ({results.adjacent_error_rate*100:.1f}% of errors)
- Distant errors: {results.n_distant_errors} ({results.distant_error_rate*100:.1f}% of errors)
- Epistemic is **{results.epistemic_distant_vs_adjacent_ratio:.1f}x higher** for distant errors
- Aleatoric is **{results.aleatoric_adjacent_vs_distant_ratio:.1f}x higher** for adjacent errors
"""
    return md


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    quick_start_example()

