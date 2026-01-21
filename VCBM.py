"""
Variational Credal CBM - Gradient Separated Version
====================================================

Key changes from original:
1. σ_var detached from task loss (only KL + alignment trains it)
2. Error prediction head for epistemic supervision
3. Scheduled combination: epistemic = α·σ_var + (1-α)·σ_err
4. Warmup period before error head activates

Author: Tanmoy
Target: ICML 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

class CovarianceFamily(Enum):
    MEAN_FIELD = "mean_field"
    LOW_RANK = "low_rank"
    FULL = "full"


@dataclass
class GradSeparatedConfig:
    """Configuration for Gradient-Separated Variational Credal CBM"""

    # Encoder
    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = True
    pooling_strategy: str = "cls"

    # Concepts
    num_concepts: int = 4
    concept_names: list = None  # Optional list of concept names
    concept_classes: int = 3  # neg/unk/pos

    # Task
    num_classes: int = 2

    # Variational
    covariance_family: CovarianceFamily = CovarianceFamily.MEAN_FIELD
    prior_std: float = 1.0
    num_mc_samples: int = 20
    min_std: float = 0.05  # Floor to prevent collapse

    # Loss weights
    kl_weight: float = 1e-3
    concept_weight: float = 2.0
    aleatoric_weight: float = 0.2
    error_pred_weight: float = 1.0
    alignment_weight: float = 0.5  # σ_var aligns with σ_err

    # Error head
    error_warmup_epochs: int = 3  # Train concepts first

    # Epistemic combination schedule
    alpha_start: float = 1.0   # Start with pure variational
    alpha_end: float = 0.3     # End with more error-based
    alpha_warmup_epochs: int = 10  # Epochs to anneal α

    # Orthogonal projection
    use_orthogonal_projection: bool = True


# ============================================================================
# ORTHOGONAL PROJECTION (2-way: epistemic vs aleatoric)
# ============================================================================

class OrthogonalProjection(nn.Module):
    """Projects hidden into orthogonal subspaces for epistemic and aleatoric."""

    def __init__(self, hidden_size: int):
        super().__init__()
        proj_dim = hidden_size // 2

        self.W_epi = nn.Linear(hidden_size, proj_dim, bias=False)
        self.W_ale = nn.Linear(hidden_size, proj_dim, bias=False)

        # Initialize orthogonally
        nn.init.orthogonal_(self.W_epi.weight)
        nn.init.orthogonal_(self.W_ale.weight)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.W_epi(hidden), self.W_ale(hidden)

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize non-orthogonality."""
        cross = self.W_epi.weight @ self.W_ale.weight.T
        return torch.norm(cross, p='fro') ** 2


# ============================================================================
# GRADIENT-SEPARATED VARIATIONAL LAYER
# ============================================================================

class VariationalLinearGradSeparated(nn.Module):
    """
    Variational linear layer with proper gradient separation.

    Key insight: σ_var is DETACHED from task loss.
    It only receives gradients from:
      1. KL divergence (regularization toward prior)
      2. Alignment with error prediction (meaningful signal)

    This prevents the collapse where task loss drives σ → 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        min_std: float = 0.05
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.min_std = min_std

        # Mean parameters (receive task + concept gradients)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))

        # Std parameters (receive KL + alignment gradients ONLY)
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -2.0)
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -2.0)

        # Prior
        self.register_buffer('prior_std', torch.tensor(prior_std))

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-6

    def get_weight_std(self) -> torch.Tensor:
        """Get weight std with minimum floor."""
        std = self._softplus(self.weight_rho)
        return torch.clamp(std, min=self.min_std)

    def forward(
        self,
        x: torch.Tensor,
        n_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with DETACHED std for task gradient path.

        Args:
            x: [batch, in_features]
            n_samples: MC samples

        Returns:
            mean, epistemic_var, logits_mean, kl
        """
        device = x.device
        batch_size = x.size(0)

        weight_std = self.get_weight_std()
        bias_std = torch.clamp(self._softplus(self.bias_rho), min=self.min_std)

        # === CRITICAL: Detach std from task gradient path ===
        weight_std_detached = weight_std.detach()
        bias_std_detached = bias_std.detach()

        # Sample weights using detached std
        eps_w = torch.randn(n_samples, self.out_features, self.in_features, device=device)
        eps_b = torch.randn(n_samples, self.out_features, device=device)

        weights = self.weight_mu + eps_w * weight_std_detached
        biases = self.bias_mu + eps_b * bias_std_detached

        # Forward pass: [S, O, I] x [B, I] -> [S, B, O]
        mc_logits = torch.einsum('soi,bi->sbo', weights, x) + biases.unsqueeze(1)
        mc_probs = torch.sigmoid(mc_logits)

        # Statistics
        mean = mc_probs.mean(dim=0)  # [B, O]
        epistemic_var = mc_probs.var(dim=0)  # [B, O]
        logits_mean = mc_logits.mean(dim=0)  # [B, O]

        return {
            'mean': mean,
            'epistemic_var': epistemic_var,
            'logits_mean': logits_mean,
            'mc_samples': mc_probs,  # [S, B, O]
            'weight_std': weight_std,  # For monitoring
        }

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(W) || p(W)) - this DOES backprop to weight_rho."""
        weight_std = self.get_weight_std()
        prior_std = self.prior_std.to(weight_std.dtype)

        # Closed-form KL for diagonal Gaussian
        kl = 0.5 * (
            (weight_std / prior_std) ** 2
            + (self.weight_mu / prior_std) ** 2
            - 1
            - 2 * torch.log(weight_std / prior_std)
        )

        # Bias KL
        bias_std = torch.clamp(self._softplus(self.bias_rho), min=self.min_std)
        kl_bias = 0.5 * (
            (bias_std / prior_std) ** 2
            + (self.bias_mu / prior_std) ** 2
            - 1
            - 2 * torch.log(bias_std / prior_std)
        )

        return kl.sum() + kl_bias.sum()


# ============================================================================
# ERROR PREDICTION HEAD
# ============================================================================

class ErrorPredictionHead(nn.Module):
    """
    Predicts soft concept-level errors.

    Target: |sigmoid(logit) - ground_truth| for each concept
    This is the BCE "gap" that Tomov-style approaches use.

    Provides supervision signal for epistemic uncertainty:
    - High predicted error → high epistemic (model unsure)
    - Low predicted error → low epistemic (model confident)
    """

    def __init__(self, in_features: int, num_concepts: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
            nn.Sigmoid()  # Output in [0, 1] (predicted error magnitude)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, in_features] - h_epi from orthogonal projection

        Returns:
            error_pred: [batch, num_concepts] - predicted error per concept
        """
        return self.net(h)


# ============================================================================
# ALEATORIC HEAD (unchanged from before)
# ============================================================================

class AleatoricHead(nn.Module):
    """Heteroscedastic aleatoric uncertainty head."""

    def __init__(self, in_features: int, num_concepts: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )

        # Learnable prior (encourages some baseline uncertainty)
        self.log_prior = nn.Parameter(torch.ones(num_concepts) * -1.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(h) + self.log_prior
        logits = torch.clamp(logits, -10, 10)
        return torch.sigmoid(logits)


# ============================================================================
# TASK CLASSIFIER
# ============================================================================

class TaskClassifier(nn.Module):
    """Simple linear classifier from concepts to task."""

    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, concept_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.linear(concept_probs)
        return {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'predictions': logits.argmax(dim=-1)
        }


# ============================================================================
# ALPHA SCHEDULER
# ============================================================================

class AlphaScheduler:
    """
    Schedules α for epistemic combination:
        epistemic = α * σ_var + (1-α) * σ_err

    Starts at α=1.0 (pure variational), anneals to α_end (more error-based).
    """

    def __init__(
        self,
        alpha_start: float = 1.0,
        alpha_end: float = 0.3,
        warmup_epochs: int = 10
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_epochs = warmup_epochs

    def get_alpha(self, epoch: int) -> float:
        """Get α value for given epoch."""
        if epoch >= self.warmup_epochs:
            return self.alpha_end

        # Linear annealing
        progress = epoch / self.warmup_epochs
        return self.alpha_start + progress * (self.alpha_end - self.alpha_start)


# ============================================================================
# MAIN MODEL
# ============================================================================

class GradSeparatedCredalCBM(nn.Module):
    """
    Gradient-Separated Variational Credal CBM

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Input → Encoder → hidden                                       │
    │              │                                                  │
    │      ┌───────┴───────┐                                         │
    │      ↓               ↓                                         │
    │   [W_epi]         [W_ale]     ← Orthogonal projections         │
    │      ↓               ↓                                         │
    │   h_epi           h_ale                                        │
    │      │               │                                         │
    │      ├───────┐       │                                         │
    │      ↓       ↓       ↓                                         │
    │ ┌─────────┐ ┌─────┐ ┌─────────┐                               │
    │ │Variation│ │Error│ │Aleatoric│                               │
    │ │ Linear  │ │Head │ │  Head   │                               │
    │ │         │ │     │ │         │                               │
    │ │μ, σ_var │ │σ_err│ │  σ_ale  │                               │
    │ └────┬────┘ └──┬──┘ └────┬────┘                               │
    │      │         │         │                                     │
    │      └────┬────┘         │                                     │
    │           ↓              │                                     │
    │   epistemic = α·σ_var + (1-α)·σ_err                           │
    │                          │                                     │
    │                     aleatoric                                  │
    │                                                                │
    │  Gradient flow:                                                │
    │  • μ        ← task_loss + concept_bce                         │
    │  • σ_var    ← kl_loss + alignment_loss (DETACHED from task)   │
    │  • σ_err    ← error_prediction_loss (after warmup)            │
    │  • σ_ale    ← aleatoric_nll (detached concept preds)          │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: GradSeparatedConfig):
        super().__init__()
        self.config = config

        # Encoder
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Orthogonal projection
        if config.use_orthogonal_projection:
            self.projection = OrthogonalProjection(self.hidden_size)
            proj_dim = self.hidden_size // 2
        else:
            self.projection = None
            proj_dim = self.hidden_size

        # Epistemic pathway: Variational + Error head
        self.variational_layer = VariationalLinearGradSeparated(
            in_features=proj_dim,
            out_features=config.num_concepts,
            prior_std=config.prior_std,
            min_std=config.min_std
        )

        self.error_head = ErrorPredictionHead(
            in_features=proj_dim,
            num_concepts=config.num_concepts
        )

        # Aleatoric pathway
        self.aleatoric_head = AleatoricHead(
            in_features=proj_dim,
            num_concepts=config.num_concepts
        )

        # Task classifier
        self.task_classifier = TaskClassifier(
            num_concepts=config.num_concepts,
            num_classes=config.num_classes
        )

        # Alpha scheduler
        self.alpha_scheduler = AlphaScheduler(
            alpha_start=config.alpha_start,
            alpha_end=config.alpha_end,
            warmup_epochs=config.alpha_warmup_epochs
        )

        # Track training state
        self.current_epoch = 0

        self._print_config()

    def _print_config(self):
        print("\n" + "=" * 70)
        print("GRADIENT-SEPARATED CREDAL CBM")
        print("=" * 70)
        print(f"Encoder: {self.config.encoder_name}")
        print(f"Concepts: {self.config.num_concepts}")
        if self.config.concept_names:
            print(f"  {self.config.concept_names}")
        print(f"Orthogonal projection: {self.config.use_orthogonal_projection}")
        print(f"\nGradient separation:")
        print(f"  • μ ← task_loss + concept_bce")
        print(f"  • σ_var ← kl_loss + alignment (DETACHED from task)")
        print(f"  • σ_err ← error_pred_loss (warmup: {self.config.error_warmup_epochs} epochs)")
        print(f"  • σ_ale ← aleatoric_nll")
        print(f"\nEpistemic combination: α·σ_var + (1-α)·σ_err")
        print(f"  • α: {self.config.alpha_start} → {self.config.alpha_end} over {self.config.alpha_warmup_epochs} epochs")
        print("=" * 70 + "\n")

    def set_epoch(self, epoch: int):
        """Set current epoch for scheduling."""
        self.current_epoch = epoch

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden representation."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.config.pooling_strategy == "cls":
            return outputs.last_hidden_state[:, 0, :]
        else:
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        concept_labels: Optional[torch.Tensor] = None,
        n_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gradient separation.
        """
        if n_samples is None:
            n_samples = self.config.num_mc_samples

        # Encode
        hidden = self.encode(input_ids, attention_mask)

        # Project to orthogonal subspaces
        if self.projection is not None:
            h_epi, h_ale = self.projection(hidden)
        else:
            h_epi = hidden
            h_ale = hidden

        # === EPISTEMIC PATHWAY ===
        var_out = self.variational_layer(h_epi, n_samples=n_samples)
        concept_probs = var_out['mean']          # [B, K]
        epistemic_var = var_out['epistemic_var'] # [B, K] from MC variance

        # Error prediction (uses same h_epi)
        epistemic_err = self.error_head(h_epi)   # [B, K]

        # Combine with scheduled α
        alpha = self.alpha_scheduler.get_alpha(self.current_epoch)
        epistemic = alpha * epistemic_var + (1 - alpha) * epistemic_err

        # === ALEATORIC PATHWAY ===
        aleatoric = self.aleatoric_head(h_ale)   # [B, K]

        # === TASK CLASSIFICATION ===
        task_out = self.task_classifier(concept_probs)

        # === BUILD RESULT ===
        result = {
            # Task
            'predictions': task_out['predictions'],
            'logits': task_out['logits'],
            'probs': task_out['probs'],

            # Concepts
            'concept_probs': concept_probs,
            'concept_logits': var_out['logits_mean'],

            # Epistemic (combined)
            'epistemic': epistemic,
            'epistemic_var': epistemic_var,  # From variational
            'epistemic_err': epistemic_err,  # From error head
            'alpha': alpha,

            # Aleatoric
            'aleatoric': aleatoric,

            # Monitoring
            'weight_std': var_out['weight_std'],
            'mc_samples': var_out['mc_samples'],
        }

        # === COMPUTE LOSSES ===
        if labels is not None or concept_labels is not None:
            losses = self._compute_losses(result, labels, concept_labels, var_out)
            result.update(losses)

        return result

    def _compute_losses(
        self,
        result: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor],
        concept_labels: Optional[torch.Tensor],
        var_out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with proper gradient separation.
        """
        losses = {}
        device = result['predictions'].device

        # =====================================================================
        # TERM 1: Task loss (gradients → μ via concept_probs)
        # =====================================================================
        if labels is not None:
            losses['task_loss'] = F.cross_entropy(result['logits'], labels)

        # =====================================================================
        # TERM 2: KL divergence (gradients → σ_var via weight_rho)
        # =====================================================================
        losses['kl_loss'] = self.variational_layer.kl_divergence()

        # =====================================================================
        # TERM 3: Concept BCE (gradients → μ)
        # =====================================================================
        if concept_labels is not None:
            known_mask = (concept_labels != 1)  # Exclude unknown

            if known_mask.any():
                # Convert to binary: 0 (neg) → 0.0, 2 (pos) → 1.0
                targets = (concept_labels[known_mask].float() / 2.0)
                preds = result['concept_probs'][known_mask]
                preds_clamped = torch.clamp(preds, 1e-7, 1 - 1e-7)

                losses['concept_bce'] = F.binary_cross_entropy(preds_clamped, targets)

                # =============================================================
                # TERM 4: Error prediction loss (gradients → error_head)
                # Only active after warmup
                # =============================================================
                if self.current_epoch >= self.config.error_warmup_epochs:
                    # Soft error: |pred - target|
                    soft_errors = torch.abs(preds.detach() - targets)
                    error_pred = result['epistemic_err'][known_mask]

                    losses['error_pred_loss'] = F.mse_loss(error_pred, soft_errors)

                    # =============================================================
                    # TERM 5: Alignment loss (gradients → σ_var)
                    # σ_var should track σ_err
                    # =============================================================
                    epistemic_var = result['epistemic_var'][known_mask]
                    losses['alignment_loss'] = F.mse_loss(
                        epistemic_var,
                        error_pred.detach()  # Detach to not affect error_head
                    )

                # =============================================================
                # TERM 6: Aleatoric NLL (gradients → aleatoric_head)
                # Uses DETACHED concept predictions
                # =============================================================
                preds_det = preds.detach()
                ale = torch.clamp(result['aleatoric'][known_mask], 1e-4, 10.0)

                # Gaussian NLL
                nll = 0.5 * torch.log(2 * np.pi * ale) + 0.5 * (preds_det - targets)**2 / ale
                losses['aleatoric_nll'] = nll.mean()

            # Unknown concepts should have high aleatoric
            unknown_mask = (concept_labels == 1)
            if unknown_mask.any():
                losses['aleatoric_unknown'] = F.mse_loss(
                    result['aleatoric'][unknown_mask],
                    torch.ones_like(result['aleatoric'][unknown_mask])
                )

        # =====================================================================
        # TERM 7: Orthogonality penalty
        # =====================================================================
        if self.projection is not None:
            losses['orth_penalty'] = self.projection.orthogonality_loss()

        # =====================================================================
        # COMBINE
        # =====================================================================
        total = torch.tensor(0.0, device=device)

        if 'task_loss' in losses:
            total = total + losses['task_loss']

        if 'concept_bce' in losses:
            total = total + self.config.concept_weight * losses['concept_bce']

        total = total + self.config.kl_weight * losses['kl_loss']

        if 'error_pred_loss' in losses:
            total = total + self.config.error_pred_weight * losses['error_pred_loss']

        if 'alignment_loss' in losses:
            total = total + self.config.alignment_weight * losses['alignment_loss']

        if 'aleatoric_nll' in losses:
            total = total + self.config.aleatoric_weight * losses['aleatoric_nll']

        if 'aleatoric_unknown' in losses:
            total = total + 0.5 * self.config.aleatoric_weight * losses['aleatoric_unknown']

        if 'orth_penalty' in losses:
            total = total + 0.001 * losses['orth_penalty']

        losses['loss'] = total
        return losses


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def diagnose_gradient_separation(
    model: GradSeparatedCredalCBM,
    dataloader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Diagnose whether gradient separation is working.
    """
    model.eval()

    # Diagnostic print statements removed for clean output

    results = {}

    # Check orthogonality
    if model.projection is not None:
        orth_loss = model.projection.orthogonality_loss().item()
        results['orthogonality_loss'] = orth_loss

    # Check weight std (should NOT collapse)
    weight_std = model.variational_layer.get_weight_std()
    results['weight_std_mean'] = weight_std.mean().item()
    results['weight_std_min'] = weight_std.min().item()

    # Check alpha
    alpha = model.alpha_scheduler.get_alpha(model.current_epoch)
    results['alpha'] = alpha

    # Collect predictions
    all_epi_var = []
    all_epi_err = []
    all_ale = []
    all_errors = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            concept_labels = batch.get('concept_labels')

            outputs = model(input_ids, attention_mask)

            all_epi_var.append(outputs['epistemic_var'].cpu())
            all_epi_err.append(outputs['epistemic_err'].cpu())
            all_ale.append(outputs['aleatoric'].cpu())

            if concept_labels is not None:
                concept_labels = concept_labels.to(device)
                known_mask = (concept_labels != 1)
                if known_mask.any():
                    targets = (concept_labels[known_mask].float() / 2.0)
                    preds = (outputs['concept_probs'][known_mask] > 0.5).float()
                    errors = (preds != targets).float()
                    all_errors.append(errors.cpu())

    epi_var = torch.cat(all_epi_var).numpy().mean(axis=-1)
    epi_err = torch.cat(all_epi_err).numpy().mean(axis=-1)
    ale = torch.cat(all_ale).numpy().mean(axis=-1)

    # Correlations
    from scipy import stats

    # σ_var vs σ_err (should be correlated after alignment training)
    rho, p = stats.spearmanr(epi_var, epi_err)
    results['rho_var_err'] = rho

    # Epistemic vs Aleatoric (should be LOW)
    epi_combined = alpha * epi_var + (1 - alpha) * epi_err
    rho, p = stats.spearmanr(epi_combined, ale)
    results['rho_epi_ale'] = rho

    # Epistemic vs Errors (should be POSITIVE)
    if len(all_errors) > 0:
        errors = torch.cat(all_errors).numpy().mean(axis=-1) if all_errors[0].dim() > 0 else torch.cat(all_errors).numpy()
        # Truncate to match
        min_len = min(len(epi_combined), len(errors))
        rho, p = stats.spearmanr(epi_combined[:min_len], errors[:min_len])
        results['rho_epi_error'] = rho

    return results


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_epoch(
    model: GradSeparatedCredalCBM,
    dataloader,
    optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Train one epoch with proper gradient separation.
    """
    model.train()
    model.set_epoch(epoch)

    total_loss = 0
    loss_components = {}

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels')
        concept_labels = batch.get('concept_labels')

        if labels is not None:
            labels = labels.to(device)
        if concept_labels is not None:
            concept_labels = concept_labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            concept_labels=concept_labels
        )

        loss = outputs['loss']
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Track components
        for key in ['task_loss', 'concept_bce', 'kl_loss', 'error_pred_loss',
                    'alignment_loss', 'aleatoric_nll']:
            if key in outputs:
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += outputs[key].item()

    n_batches = len(dataloader)
    metrics = {'loss': total_loss / n_batches}
    for key, val in loss_components.items():
        metrics[key] = val / n_batches

    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Config
    config = GradSeparatedConfig(
        encoder_name="distilbert-base-uncased",
        num_concepts=4,
        num_classes=2,
        error_warmup_epochs=3,
        alpha_start=1.0,
        alpha_end=0.3,
        alpha_warmup_epochs=10
    )

    # Create model
    model = GradSeparatedCredalCBM(config)

    # Example forward
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    texts = ["The food was great but service was slow."]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )

    # Example output print statements removed for clean code
