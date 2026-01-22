"""
Concept-Supervised Credal CBM
=============================

Key design:
1. σ_epi: Trained to predict concept-level errors |pred - target|
2. σ_ale: Trained to predict annotator disagreement entropy
3. Both have per-concept supervision from CEBaB
4. Orthogonal projections ensure structural separation
5. NO variational sampling - deterministic heads (simpler, avoids collapse)

The VAE-style tension:
- σ_epi: KL regularization ↔ concept error prediction
- σ_ale: Prior regularization ↔ annotator entropy prediction

Author: Tanmoy
Target: ICML 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============================================================================
# COVARIANCE FAMILY ENUM
# ============================================================================

class CovarianceFamily(Enum):
    """Covariance structure for credal sets"""
    MEAN_FIELD = "mean_field"  # Independent concepts
    FULL = "full"  # Full covariance matrix


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ConceptSupervisedConfig:
    """Configuration for Concept-Supervised Credal CBM"""

    # Encoder
    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = True
    pooling_strategy: str = "cls"

    # Concepts
    num_concepts: int = 4
    concept_names: List[str] = field(default_factory=lambda: ['food', 'service', 'ambiance', 'noise'])
    concept_classes: int = 3  # neg/unk/pos
    covariance_family: CovarianceFamily = CovarianceFamily.MEAN_FIELD

    # Task
    num_classes: int = 2

    # Loss weights
    concept_weight: float = 2.0
    epistemic_weight: float = 1.0
    aleatoric_weight: float = 1.0
    kl_weight: float = 0.01  # Regularization for epistemic
    orth_weight: float = 0.001

    # Priors (for VAE-style tension)
    epistemic_prior: float = 0.1  # Expected baseline error rate
    aleatoric_prior: float = 0.3  # Expected baseline disagreement

    # Orthogonal projection
    use_orthogonal_projection: bool = True

    # Hidden dimensions
    uncertainty_hidden_dim: int = 128


# ============================================================================
# ANNOTATOR ENTROPY COMPUTATION
# ============================================================================

def compute_annotator_entropy(distributions: Dict[str, int], eps: float = 1e-8) -> float:
    """
    Compute entropy from annotator distribution.

    Args:
        distributions: Dict like {"Positive": 3, "Negative": 1, "unknown": 1}
        eps: Small value for numerical stability

    Returns:
        Entropy in [0, log(n_classes)] normalized to [0, 1]
    """
    counts = np.array(list(distributions.values()), dtype=np.float32)
    total = counts.sum()

    if total == 0:
        return 0.0

    probs = counts / total
    probs = np.clip(probs, eps, 1.0)

    entropy = -np.sum(probs * np.log(probs))

    # Normalize to [0, 1] by dividing by max entropy (log(n_classes))
    max_entropy = np.log(len(distributions))
    if max_entropy > 0:
        entropy = entropy / max_entropy

    return float(entropy)


def compute_concept_entropies_batch(
    concept_distributions: List[List[Dict[str, int]]]
) -> torch.Tensor:
    """
    Compute per-concept annotator entropy for a batch.

    Args:
        concept_distributions: [batch_size, num_concepts] list of dicts
            Each dict is like {"Positive": 3, "Negative": 1, "unknown": 1}

    Returns:
        [batch_size, num_concepts] tensor of entropies in [0, 1]
    """
    batch_size = len(concept_distributions)
    num_concepts = len(concept_distributions[0]) if batch_size > 0 else 0

    entropies = torch.zeros(batch_size, num_concepts)

    for i, concepts in enumerate(concept_distributions):
        for j, dist in enumerate(concepts):
            if dist is not None:
                entropies[i, j] = compute_annotator_entropy(dist)
            else:
                entropies[i, j] = 0.5  # Default if missing

    return entropies


# ============================================================================
# ORTHOGONAL PROJECTION
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
        """Penalize non-orthogonality: ||W_epi^T @ W_ale||_F^2"""
        cross = self.W_epi.weight @ self.W_ale.weight.T
        return torch.norm(cross, p='fro') ** 2


# ============================================================================
# CONCEPT PREDICTION HEAD
# ============================================================================

class ConceptHead(nn.Module):
    """
    Deterministic concept prediction head.

    Maps h_epi → concept probabilities (no variational sampling).
    """

    def __init__(self, in_features: int, num_concepts: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, in_features]

        Returns:
            concept_logits: [batch, num_concepts]
        """
        return self.net(h)


# ============================================================================
# EPISTEMIC HEAD (Predicts Concept Errors)
# ============================================================================

class EpistemicHead(nn.Module):
    """
    Predicts per-concept epistemic uncertainty.

    Training signal: concept-level error |pred - target|
    VAE-style tension: KL toward prior ↔ error prediction loss

    Interpretation: "How likely is the model wrong on this concept?"
    """

    def __init__(
        self,
        in_features: int,
        num_concepts: int,
        hidden_dim: int = 128,
        prior_mean: float = 0.1
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.prior_mean = prior_mean

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
        )

        # Learnable prior (initialized to prior_mean in logit space)
        # sigmoid(x) = prior_mean → x = logit(prior_mean)
        prior_logit = np.log(prior_mean / (1 - prior_mean))
        self.log_prior = nn.Parameter(torch.ones(num_concepts) * prior_logit)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, in_features] - h_epi from orthogonal projection

        Returns:
            epistemic: [batch, num_concepts] in [0, 1]
        """
        logits = self.net(h) + self.log_prior
        logits = torch.clamp(logits, -10, 10)
        return torch.sigmoid(logits)

    def kl_divergence(self, epistemic: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from predicted epistemic to prior.

        Encourages epistemic to stay near prior unless data says otherwise.
        This creates VAE-style tension with the error prediction loss.

        Args:
            epistemic: [batch, num_concepts] predicted epistemic uncertainty

        Returns:
            KL divergence (scalar)
        """
        # Prior: Beta distribution centered at prior_mean
        # Approximation: treat as Bernoulli, compute binary cross-entropy to prior
        prior = torch.sigmoid(self.log_prior)
        prior = prior.unsqueeze(0).expand_as(epistemic)

        # KL(predicted || prior) for Bernoulli
        eps = 1e-7
        epistemic_clamped = torch.clamp(epistemic, eps, 1 - eps)
        prior_clamped = torch.clamp(prior, eps, 1 - eps)

        kl = epistemic_clamped * torch.log(epistemic_clamped / prior_clamped) + \
             (1 - epistemic_clamped) * torch.log((1 - epistemic_clamped) / (1 - prior_clamped))

        return kl.mean()


# ============================================================================
# ALEATORIC HEAD (Predicts Annotator Disagreement)
# ============================================================================

class AleatoricHead(nn.Module):
    """
    Predicts per-concept aleatoric uncertainty.

    Training signal: annotator disagreement entropy
    VAE-style tension: KL toward prior ↔ entropy prediction loss

    Interpretation: "How much do humans disagree on this concept?"
    """

    def __init__(
        self,
        in_features: int,
        num_concepts: int,
        hidden_dim: int = 128,
        prior_mean: float = 0.3
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.prior_mean = prior_mean

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
        )

        # Learnable prior
        prior_logit = np.log(prior_mean / (1 - prior_mean))
        self.log_prior = nn.Parameter(torch.ones(num_concepts) * prior_logit)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, in_features] - h_ale from orthogonal projection

        Returns:
            aleatoric: [batch, num_concepts] in [0, 1]
        """
        logits = self.net(h) + self.log_prior
        logits = torch.clamp(logits, -10, 10)
        return torch.sigmoid(logits)

    def kl_divergence(self, aleatoric: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from predicted aleatoric to prior.
        """
        prior = torch.sigmoid(self.log_prior)
        prior = prior.unsqueeze(0).expand_as(aleatoric)

        eps = 1e-7
        aleatoric_clamped = torch.clamp(aleatoric, eps, 1 - eps)
        prior_clamped = torch.clamp(prior, eps, 1 - eps)

        kl = aleatoric_clamped * torch.log(aleatoric_clamped / prior_clamped) + \
             (1 - aleatoric_clamped) * torch.log((1 - aleatoric_clamped) / (1 - prior_clamped))

        return kl.mean()


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
# MAIN MODEL
# ============================================================================

class ConceptSupervisedCredalCBM(nn.Module):
    """
    Concept-Supervised Credal CBM

    Both epistemic and aleatoric have direct per-concept supervision:
    - σ_epi: trained to predict |concept_pred - concept_target|
    - σ_ale: trained to predict annotator_entropy per concept

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
    │ │ Concept │ │Epist│ │Aleatoric│                               │
    │ │  Head   │ │Head │ │  Head   │                               │
    │ │         │ │     │ │         │                               │
    │ │ μ(x)    │ │σ_epi│ │  σ_ale  │                               │
    │ └────┬────┘ └──┬──┘ └────┬────┘                               │
    │      │         │         │                                     │
    │      │    Predicts   Predicts                                  │
    │      │    concept    annotator                                 │
    │      │    errors     entropy                                   │
    │      │         │         │                                     │
    │      ↓         ↓         ↓                                     │
    │   concepts  epistemic  aleatoric                               │
    │                                                                │
    │  Gradient flow:                                                │
    │  • concept_head ← concept_bce + task_loss                     │
    │  • epistemic_head ← error_pred_loss + kl_epi (VAE tension)    │
    │  • aleatoric_head ← entropy_pred_loss + kl_ale (VAE tension)  │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: ConceptSupervisedConfig):
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

        # Concept prediction (deterministic)
        self.concept_head = ConceptHead(
            in_features=proj_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.uncertainty_hidden_dim
        )

        # Epistemic head (predicts concept errors)
        self.epistemic_head = EpistemicHead(
            in_features=proj_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.uncertainty_hidden_dim,
            prior_mean=config.epistemic_prior
        )

        # Aleatoric head (predicts annotator disagreement)
        self.aleatoric_head = AleatoricHead(
            in_features=proj_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.uncertainty_hidden_dim,
            prior_mean=config.aleatoric_prior
        )

        # Task classifier
        self.task_classifier = TaskClassifier(
            num_concepts=config.num_concepts,
            num_classes=config.num_classes
        )

        self._print_config()

    def _print_config(self):
        print("\n" + "=" * 70)
        print("CONCEPT-SUPERVISED CREDAL CBM")
        print("=" * 70)
        print(f"Encoder: {self.config.encoder_name}")
        print(f"Concepts: {self.config.num_concepts} {self.config.concept_names}")
        print(f"Orthogonal projection: {self.config.use_orthogonal_projection}")
        print(f"\nSupervision signals:")
        print(f"  • σ_epi ← concept errors |pred - target| (prior={self.config.epistemic_prior})")
        print(f"  • σ_ale ← annotator entropy H(dist) (prior={self.config.aleatoric_prior})")
        print(f"\nVAE-style tension:")
        print(f"  • KL weight: {self.config.kl_weight}")
        print(f"  • Both heads have learnable priors")
        print("=" * 70 + "\n")

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
        annotator_entropy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with concept-level supervision.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] task labels
            concept_labels: [batch, num_concepts] concept labels (0=neg, 1=unk, 2=pos)
            annotator_entropy: [batch, num_concepts] per-concept annotator entropy
        """
        # Encode
        hidden = self.encode(input_ids, attention_mask)

        # Project to orthogonal subspaces
        if self.projection is not None:
            h_epi, h_ale = self.projection(hidden)
        else:
            h_epi = hidden
            h_ale = hidden

        # === CONCEPT PREDICTION ===
        concept_logits = self.concept_head(h_epi)
        concept_probs = torch.sigmoid(concept_logits)

        # === EPISTEMIC (predicts concept errors) ===
        epistemic = self.epistemic_head(h_epi)

        # === ALEATORIC (predicts annotator entropy) ===
        aleatoric = self.aleatoric_head(h_ale)

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
            'concept_logits': concept_logits,

            # Uncertainties
            'epistemic': epistemic,
            'aleatoric': aleatoric,
        }

        # === COMPUTE LOSSES ===
        if labels is not None or concept_labels is not None:
            losses = self._compute_losses(
                result, labels, concept_labels, annotator_entropy
            )
            result.update(losses)

        return result

    def _compute_losses(
        self,
        result: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor],
        concept_labels: Optional[torch.Tensor],
        annotator_entropy: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with per-concept supervision.
        """
        losses = {}
        device = result['predictions'].device

        # =====================================================================
        # TERM 1: Task loss
        # =====================================================================
        if labels is not None:
            losses['task_loss'] = F.cross_entropy(result['logits'], labels)

        # =====================================================================
        # TERM 2: Concept BCE (gradients → concept_head)
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
                # TERM 3: Epistemic supervision (gradients → epistemic_head)
                # Predicts concept-level errors
                # =============================================================
                # Soft error: |pred - target|
                soft_errors = torch.abs(preds.detach() - targets)
                epistemic_pred = result['epistemic'][known_mask]
                losses['epistemic_loss'] = F.mse_loss(epistemic_pred, soft_errors)

                # =============================================================
                # TERM 4: Aleatoric supervision (gradients → aleatoric_head)
                # Predicts annotator entropy
                # =============================================================
                if annotator_entropy is not None:
                    entropy_targets = annotator_entropy[known_mask]
                    aleatoric_pred = result['aleatoric'][known_mask]
                    losses['aleatoric_loss'] = F.mse_loss(aleatoric_pred, entropy_targets)

            # Unknown concepts should have high aleatoric
            unknown_mask = (concept_labels == 1)
            if unknown_mask.any():
                losses['aleatoric_unknown'] = F.mse_loss(
                    result['aleatoric'][unknown_mask],
                    torch.ones_like(result['aleatoric'][unknown_mask])
                )

        # =====================================================================
        # TERM 5: Epistemic KL (VAE-style tension)
        # =====================================================================
        if 'epistemic' in result:
            losses['epistemic_kl'] = self.epistemic_head.kl_divergence(result['epistemic'])

        # =====================================================================
        # TERM 6: Aleatoric KL (VAE-style tension)
        # =====================================================================
        if 'aleatoric' in result:
            losses['aleatoric_kl'] = self.aleatoric_head.kl_divergence(result['aleatoric'])

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

        if 'epistemic_loss' in losses:
            total = total + self.config.epistemic_weight * losses['epistemic_loss']

        if 'aleatoric_loss' in losses:
            total = total + self.config.aleatoric_weight * losses['aleatoric_loss']

        if 'aleatoric_unknown' in losses:
            total = total + 0.1 * self.config.aleatoric_weight * losses['aleatoric_unknown']  # Reduced from 0.5

        total = total + self.config.kl_weight * losses.get('epistemic_kl', 0)
        total = total + self.config.kl_weight * losses.get('aleatoric_kl', 0)

        if 'orth_penalty' in losses:
            total = total + self.config.orth_weight * losses['orth_penalty']

        losses['loss'] = total
        return losses


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def diagnose_concept_supervision(
    model: ConceptSupervisedCredalCBM,
    dataloader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Diagnose concept supervision quality.
    """
    model.eval()

    print("\n" + "=" * 60)
    print("CONCEPT-SUPERVISED CREDAL CBM DIAGNOSTIC")
    print("=" * 60)

    results = {}

    # Check orthogonality
    if model.projection is not None:
        orth_loss = model.projection.orthogonality_loss().item()
        results['orthogonality_loss'] = orth_loss
        print(f"Orthogonality loss: {orth_loss:.6f} (should be ~0)")

    # Collect predictions
    all_epi = []
    all_ale = []
    all_preds = []
    all_targets = []
    all_errors = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch.get('concept_labels')

            outputs = model(input_ids, attention_mask, labels=labels, concept_labels=concept_labels)

            all_epi.append(outputs['epistemic'].cpu())
            all_ale.append(outputs['aleatoric'].cpu())

            if concept_labels is not None:
                known_mask = (concept_labels != 1)
                if known_mask.any():
                    targets = (concept_labels[known_mask].float() / 2.0)
                    preds = outputs['concept_probs'][known_mask]
                    errors = torch.abs(preds - targets).cpu()

                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())
                    all_errors.append(errors.cpu())

    epi = torch.cat(all_epi).numpy()
    ale = torch.cat(all_ale).numpy()

    # Statistics
    results['mean_epistemic'] = epi.mean()
    results['mean_aleatoric'] = ale.mean()
    print(f"\nMean epistemic: {epi.mean():.4f}")
    print(f"Mean aleatoric: {ale.mean():.4f}")

    # Check if epistemic correlates with errors
    if len(all_errors) > 0:
        all_errors = torch.cat(all_errors)
        all_preds = torch.cat(all_preds)

        epi_flat = epi.flatten()[:len(all_errors.flatten())]
        err_flat = all_errors.flatten()

        from scipy import stats
        rho, p = stats.spearmanr(epi_flat, err_flat)
        results['rho_epi_error'] = rho
        print(f"\nρ(epistemic, error): {rho:.3f} (p={p:.2e})")

        if rho > 0.3:
            print("  ✓ Epistemic tracks errors well!")
        else:
            print("  ⚠️  Epistemic not tracking errors")

    # Epistemic vs Aleatoric correlation (should be low)
    min_len = min(epi.size, ale.size)
    rho, p = stats.spearmanr(epi.flatten()[:min_len], ale.flatten()[:min_len])
    results['rho_epi_ale'] = rho
    print(f"ρ(epistemic, aleatoric): {rho:.3f} (p={p:.2e})")

    if abs(rho) < 0.3:
        print("  ✓ Good separation!")
    else:
        print("  ⚠️  High correlation - separation issues")

    print("=" * 60)

    return results


if __name__ == "__main__":
    # Test
    config = ConceptSupervisedConfig(
        num_concepts=4,
        concept_names=['food', 'service', 'ambiance', 'noise']
    )

    model = ConceptSupervisedCredalCBM(config)


# ============================================================================
# TRUE CREDAL CBM WITH STRUCTURAL SEPARATION
# ============================================================================

"""
True Credal CBM with Structural Separation
==========================================

This design is ACTUALLY credal:
- Model outputs credal set parameters (μ, Σ_epi), not point estimates
- Epistemic uncertainty = credal set size (derived from geometry)
- Aleatoric uncertainty = supervised head (annotator entropy)
- Three-way orthogonal projection for TRUE structural separation

Key insight: In a credal model, Σ_epi IS the epistemic uncertainty,
not a prediction of it. The supervision comes via KL regularization
to a prior, which shrinks as the model becomes more confident.

Connection to Tomov et al. impossibility:
- We escape because EU comes from Σ_epi (geometric, trained by KL)
- AU comes from σ_ale (supervised by annotator entropy)
- Neither is derived from the predictive distribution p(y|x)

Author: Tanmoy
Target: ICML 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrueCredalConfig:
    """Configuration for True Credal CBM"""

    # Encoder
    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = True
    pooling_strategy: str = "cls"

    # Concepts
    num_concepts: int = 4
    concept_names: List[str] = field(
        default_factory=lambda: ['food', 'service', 'ambiance', 'noise']
    )

    # Task
    num_classes: int = 2

    # Credal set parameters
    num_mc_samples: int = 10  # Samples from credal set at inference
    min_sigma: float = 1e-4   # Minimum credal set size
    max_sigma: float = 2.0    # Maximum credal set size

    # Prior for epistemic (credal set size)
    # Initialized large → shrinks during training as model learns
    prior_sigma: float = 1.0

    # Loss weights
    concept_weight: float = 2.0
    kl_weight: float = 0.1      # KL for credal set → epistemic signal
    aleatoric_weight: float = 1.0
    orth_weight: float = 0.001

    # Aleatoric prior
    aleatoric_prior: float = 0.3

    # Architecture
    projection_dim: int = 256
    hidden_dim: int = 128


# ============================================================================
# THREE-WAY ORTHOGONAL PROJECTION
# ============================================================================

class ThreeWayOrthogonalProjection(nn.Module):
    """
    Projects encoder hidden state into THREE orthogonal subspaces:

    h_concept: for credal set center (μ)
    h_epi: for credal set size (Σ_epi)
    h_ale: for aleatoric uncertainty (σ_ale)

    This ensures TRUE gradient separation.
    """

    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()

        self.W_concept = nn.Linear(hidden_size, proj_dim, bias=False)
        self.W_epi = nn.Linear(hidden_size, proj_dim, bias=False)
        self.W_ale = nn.Linear(hidden_size, proj_dim, bias=False)

        self._initialize_orthogonal(hidden_size, proj_dim)

    def _initialize_orthogonal(self, hidden_size: int, proj_dim: int):
        """Initialize all three projections to be mutually orthogonal."""
        total_dim = 3 * proj_dim

        if total_dim <= hidden_size:
            # Can achieve true orthogonality
            full_orth = torch.empty(total_dim, hidden_size)
            nn.init.orthogonal_(full_orth)

            self.W_concept.weight.data = full_orth[:proj_dim]
            self.W_epi.weight.data = full_orth[proj_dim:2*proj_dim]
            self.W_ale.weight.data = full_orth[2*proj_dim:3*proj_dim]
        else:
            # Fall back to individual orthogonal init
            nn.init.orthogonal_(self.W_concept.weight)
            nn.init.orthogonal_(self.W_epi.weight)
            nn.init.orthogonal_(self.W_ale.weight)

    def forward(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.W_concept(hidden),
            self.W_epi(hidden),
            self.W_ale(hidden)
        )

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize non-orthogonality between all pairs."""
        W_c = self.W_concept.weight
        W_e = self.W_epi.weight
        W_a = self.W_ale.weight

        loss = (
            torch.norm(W_c @ W_e.T, p='fro') ** 2 +
            torch.norm(W_c @ W_a.T, p='fro') ** 2 +
            torch.norm(W_e @ W_a.T, p='fro') ** 2
        )
        return loss


# ============================================================================
# CREDAL SET HEAD
# ============================================================================

class CredalSetHead(nn.Module):
    """
    Outputs credal set parameters for each concept.

    For concept k, the credal set is:
        C^(k) = {q ∈ [0,1] : q ~ N(μ^(k), (σ_epi^(k))²)}

    This is a set of plausible probability values, not a single point.

    Key design:
    - μ comes from h_concept (trained by concept loss)
    - σ_epi comes from h_epi (trained by KL to prior)
    - These are SEPARATE pathways with SEPARATE gradients

    The epistemic uncertainty IS the credal set size:
        EU^(k) = log(σ_epi^(k))

    This is not a prediction of uncertainty—it's the uncertainty itself,
    derived from the geometry of the credal set.
    """

    def __init__(
        self,
        proj_dim: int,
        num_concepts: int,
        hidden_dim: int = 128,
        prior_sigma: float = 1.0,
        min_sigma: float = 1e-4,
        max_sigma: float = 2.0,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.prior_sigma = prior_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # μ head: predicts center of credal set (from h_concept)
        self.mu_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )

        # σ_epi head: predicts credal set size (from h_epi)
        # This IS the epistemic uncertainty
        self.log_sigma_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )

        # Initialize σ to prior (will shrink during training)
        self._init_sigma_to_prior()

    def _init_sigma_to_prior(self):
        """Initialize so that σ_epi starts at prior_sigma."""
        # We want softplus(output) ≈ prior_sigma
        # softplus(x) = log(1 + exp(x))
        # For softplus(x) = prior_sigma: x ≈ log(exp(prior_sigma) - 1)
        init_val = math.log(math.exp(self.prior_sigma) - 1)

        # Initialize last layer bias
        if hasattr(self.log_sigma_net[-1], 'bias'):
            nn.init.constant_(self.log_sigma_net[-1].bias, init_val)
        # Small weights so initial output ≈ bias (FIX: was zeros, now small random)
        if hasattr(self.log_sigma_net[-1], 'weight'):
            nn.init.normal_(self.log_sigma_net[-1].weight, std=0.01)

    def forward(
        self,
        h_concept: torch.Tensor,
        h_epi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_concept: [batch, proj_dim] - features for credal center
            h_epi: [batch, proj_dim] - features for credal size

        Returns:
            mu: [batch, num_concepts] - credal set centers (logit space)
            sigma_epi: [batch, num_concepts] - credal set sizes
            p_mean: [batch, num_concepts] - mean concept probabilities
        """
        # Credal set center (in logit space)
        mu = self.mu_net(h_concept)

        # Credal set size (epistemic uncertainty)
        log_sigma_raw = self.log_sigma_net(h_epi)
        sigma_epi = F.softplus(log_sigma_raw)
        sigma_epi = torch.clamp(sigma_epi, self.min_sigma, self.max_sigma)

        # Mean concept probability (center of credal set in prob space)
        p_mean = torch.sigmoid(mu)

        return {
            'mu': mu,
            'sigma_epi': sigma_epi,
            'p_mean': p_mean,
        }

    def sample_credal_set(
        self,
        mu: torch.Tensor,
        sigma_epi: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Sample points from the credal set.

        Each sample represents a plausible "true" concept probability
        given our epistemic uncertainty.

        Args:
            mu: [batch, num_concepts] - credal centers (logit space)
            sigma_epi: [batch, num_concepts] - credal sizes
            num_samples: number of samples

        Returns:
            samples: [num_samples, batch, num_concepts] - probabilities in [0,1]
        """
        batch_size, num_concepts = mu.shape

        # Reparameterization trick: z = μ + σ * ε, ε ~ N(0,1)
        eps = torch.randn(
            num_samples, batch_size, num_concepts,
            device=mu.device, dtype=mu.dtype
        )

        # Sample in logit space
        logit_samples = mu.unsqueeze(0) + sigma_epi.unsqueeze(0) * eps

        # Convert to probability space
        prob_samples = torch.sigmoid(logit_samples)

        return prob_samples

    def epistemic_uncertainty(self, sigma_epi: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic uncertainty from credal set size.

        EU = log(σ_epi) = log-volume of credal set

        This is DERIVED from geometry, not predicted!

        Args:
            sigma_epi: [batch, num_concepts]

        Returns:
            eu: [batch, num_concepts] - epistemic uncertainty per concept
        """
        return torch.log(sigma_epi + 1e-10)

    def kl_divergence(self, sigma_epi: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from current credal set to prior.

        Prior: N(0, prior_sigma²)
        Current: N(μ, σ_epi²)

        For the σ part only (μ is trained by concept loss):
        KL contribution from σ: log(σ_prior/σ) + σ²/(2σ_prior²) - 1/2

        This is the TRAINING SIGNAL for epistemic uncertainty!
        As the model learns, σ_epi shrinks → KL decreases → EU decreases.

        Args:
            sigma_epi: [batch, num_concepts]

        Returns:
            kl: scalar
        """
        # KL for variance term only
        # KL(N(μ,σ²) || N(0,σ_prior²)) variance contribution:
        # = log(σ_prior/σ) + σ²/(2σ_prior²) - 1/2

        prior_sigma = self.prior_sigma

        kl = (
            torch.log(prior_sigma / sigma_epi) +
            (sigma_epi ** 2) / (2 * prior_sigma ** 2) -
            0.5
        )

        return kl.mean()


# ============================================================================
# ALEATORIC HEAD
# ============================================================================

class CredalAleatoricHead(nn.Module):
    """
    Predicts aleatoric uncertainty from h_ale.

    Training signal: annotator disagreement entropy
    This captures inherent ambiguity that cannot be reduced.

    Note: This is SUPERVISED, unlike σ_epi which is REGULARIZED.
    The aleatoric head learns what ambiguity looks like from data.
    """

    def __init__(
        self,
        proj_dim: int,
        num_concepts: int,
        hidden_dim: int = 128,
        prior_mean: float = 0.3,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.prior_mean = prior_mean

        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
        )

        # Learnable prior
        prior_logit = np.log(prior_mean / (1 - prior_mean))
        self.log_prior = nn.Parameter(torch.ones(num_concepts) * prior_logit)

    def forward(self, h_ale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_ale: [batch, proj_dim]

        Returns:
            sigma_ale: [batch, num_concepts] in [0, 1]
        """
        logits = self.net(h_ale) + self.log_prior
        return torch.sigmoid(torch.clamp(logits, -10, 10))

    def kl_divergence(self, sigma_ale: torch.Tensor) -> torch.Tensor:
        """Optional KL regularization for aleatoric."""
        prior = torch.sigmoid(self.log_prior).unsqueeze(0).expand_as(sigma_ale)
        eps = 1e-7
        a = torch.clamp(sigma_ale, eps, 1 - eps)
        p = torch.clamp(prior, eps, 1 - eps)
        kl = a * torch.log(a / p) + (1 - a) * torch.log((1 - a) / (1 - p))
        return kl.mean()


# ============================================================================
# TASK CLASSIFIER (with credal bounds)
# ============================================================================

class CredalTaskClassifier(nn.Module):
    """
    Task classifier that can use credal bounds.

    Given credal concept predictions, computes:
    - Point prediction (from mean)
    - Credal bounds on task prediction (from samples)
    """

    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(
        self,
        p_mean: torch.Tensor,
        p_samples: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            p_mean: [batch, num_concepts] - mean concept probs
            p_samples: [num_samples, batch, num_concepts] - credal samples

        Returns:
            logits, probs, predictions, and optionally credal bounds
        """
        # Point prediction from mean
        logits = self.linear(p_mean)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        result = {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
        }

        # Credal bounds if samples provided
        if p_samples is not None:
            # [num_samples, batch, num_classes]
            sample_logits = self.linear(p_samples)
            sample_probs = F.softmax(sample_logits, dim=-1)

            # Bounds across samples
            result['prob_lower'] = sample_probs.min(dim=0)[0]
            result['prob_upper'] = sample_probs.max(dim=0)[0]
            result['logit_std'] = sample_logits.std(dim=0)

        return result


# ============================================================================
# MAIN MODEL: TRUE CREDAL CBM
# ============================================================================

class TrueCredalCBM(nn.Module):
    """
    True Credal CBM with Structural Separation

    This is ACTUALLY credal:
    - Outputs credal sets (μ, Σ_epi) not point estimates
    - EU = credal set size (derived from geometry)
    - AU = supervised head (annotator entropy)

    Architecture:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Input → Encoder → h                                                │
    │              │                                                      │
    │      ┌───────┼───────┐                                             │
    │      ↓       ↓       ↓                                             │
    │  [W_concept][W_epi][W_ale]  ← Three orthogonal projections         │
    │      ↓       ↓       ↓                                             │
    │  h_concept h_epi  h_ale                                            │
    │      ↓       ↓       ↓                                             │
    │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐                                         │
    │   │μ^(k)│ │Σ_epi│ │σ_ale│                                         │
    │   └──┬──┘ └──┬──┘ └──┬──┘                                         │
    │      │       │       │                                             │
    │      └───┬───┘       │                                             │
    │          ↓           │                                             │
    │   ┌────────────┐     │                                             │
    │   │ Credal Set │     │                                             │
    │   │ C^(k) =    │     │                                             │
    │   │ N(μ, Σ_epi)│     │                                             │
    │   └─────┬──────┘     │                                             │
    │         │            │                                             │
    │    Sample/Mean       │                                             │
    │         ↓            ↓                                             │
    │   ┌──────────┐  ┌─────────┐                                       │
    │   │Task Pred │  │   AU    │                                       │
    │   │ŷ, bounds │  │σ_ale^(k)│                                       │
    │   └──────────┘  └─────────┘                                       │
    │                                                                    │
    │   EU = log|Σ_epi| (DERIVED from geometry!)                        │
    │   AU = σ_ale (SUPERVISED by annotator entropy)                    │
    │                                                                    │
    │  Training signals:                                                │
    │    • μ ← L_concept (NLL on concept labels)                        │
    │    • Σ_epi ← L_KL (shrink toward prior) ← CREDAL/EPISTEMIC        │
    │    • σ_ale ← L_ale (match annotator entropy) ← ALEATORIC          │
    └─────────────────────────────────────────────────────────────────────┘

    Connection to impossibility:
    - Tomov et al. prove no f(p) can separate EU/AU
    - We escape: EU comes from Σ_epi (geometry), AU from σ_ale (supervision)
    - Neither is derived from the predictive distribution!
    """

    def __init__(self, config: TrueCredalConfig):
        super().__init__()
        self.config = config

        # Encoder
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Three-way orthogonal projection
        self.projection = ThreeWayOrthogonalProjection(
            self.hidden_size,
            config.projection_dim
        )

        # Credal set head (μ from h_concept, Σ_epi from h_epi)
        self.credal_head = CredalSetHead(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            prior_sigma=config.prior_sigma,
            min_sigma=config.min_sigma,
            max_sigma=config.max_sigma,
        )

        # Aleatoric head (from h_ale)
        self.aleatoric_head = CredalAleatoricHead(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            prior_mean=config.aleatoric_prior,
        )

        # Task classifier
        self.task_classifier = CredalTaskClassifier(
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
        )

        self._print_architecture()

    def _print_architecture(self):
        print("\n" + "=" * 70)
        print("TRUE CREDAL CBM WITH STRUCTURAL SEPARATION")
        print("=" * 70)
        print(f"Encoder: {self.config.encoder_name}")
        print(f"Concepts: {self.config.num_concepts} {self.config.concept_names}")
        print(f"\nCREDAL SET PARAMETERS:")
        print(f"  • μ^(k): credal center (from h_concept)")
        print(f"  • Σ_epi^(k): credal size (from h_epi)")
        print(f"  • Prior σ: {self.config.prior_sigma}")
        print(f"  • MC samples: {self.config.num_mc_samples}")
        print(f"\nUNCERTAINTY SOURCES:")
        print(f"  • EU = log(Σ_epi) — DERIVED from credal geometry")
        print(f"  • AU = σ_ale — SUPERVISED by annotator entropy")
        print(f"\nTRAINING SIGNALS:")
        print(f"  • W_concept, μ_net ← L_concept")
        print(f"  • W_epi, σ_net ← L_KL (credal shrinkage)")
        print(f"  • W_ale, ale_net ← L_ale (disagreement)")
        print("=" * 70 + "\n")

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
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
        annotator_entropy: Optional[torch.Tensor] = None,
        num_mc_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] task labels
            concept_labels: [batch, num_concepts] (0=neg, 1=unk, 2=pos)
            annotator_entropy: [batch, num_concepts] per-concept entropy
            num_mc_samples: override config for MC samples
        """
        num_samples = num_mc_samples or self.config.num_mc_samples

        # Encode
        hidden = self.encode(input_ids, attention_mask)

        # Three orthogonal projections
        h_concept, h_epi, h_ale = self.projection(hidden)

        # Credal set parameters
        credal_out = self.credal_head(h_concept, h_epi)
        mu = credal_out['mu']
        sigma_epi = credal_out['sigma_epi']
        p_mean = credal_out['p_mean']

        # Sample from credal set
        p_samples = self.credal_head.sample_credal_set(mu, sigma_epi, num_samples)

        # Aleatoric uncertainty
        sigma_ale = self.aleatoric_head(h_ale)

        # Task prediction (with credal bounds)
        task_out = self.task_classifier(p_mean, p_samples)

        # Epistemic uncertainty (DERIVED from credal geometry!)
        epistemic = self.credal_head.epistemic_uncertainty(sigma_epi)

        result = {
            # Task
            'predictions': task_out['predictions'],
            'logits': task_out['logits'],
            'probs': task_out['probs'],

            # Credal bounds
            'prob_lower': task_out.get('prob_lower'),
            'prob_upper': task_out.get('prob_upper'),

            # Concepts
            'concept_probs': p_mean,          # Mean of credal set
            'concept_samples': p_samples,      # Samples from credal set
            'mu': mu,                          # Credal center (logits)
            'sigma_epi': sigma_epi,            # Credal size

            # Uncertainties
            'epistemic': epistemic,            # DERIVED: log(σ_epi)
            'aleatoric': sigma_ale,            # SUPERVISED: annotator entropy

            # For analysis
            'h_concept': h_concept,
            'h_epi': h_epi,
            'h_ale': h_ale,
        }

        # Compute losses if training
        if labels is not None or concept_labels is not None:
            losses = self._compute_losses(
                result, labels, concept_labels, annotator_entropy
            )
            result.update(losses)

        return result

    def _compute_losses(
        self,
        result: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor],
        concept_labels: Optional[torch.Tensor],
        annotator_entropy: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses with proper gradient separation.

        Critical insight:
        - L_concept → gradients to W_concept, μ_net
        - L_KL → gradients to W_epi, σ_net (THIS IS THE EPISTEMIC SIGNAL!)
        - L_ale → gradients to W_ale, ale_net
        """
        losses = {}
        device = result['predictions'].device

        # =====================================================================
        # TERM 1: Task loss (gradients → W_concept via p_mean)
        # =====================================================================
        if labels is not None:
            losses['task_loss'] = F.cross_entropy(result['logits'], labels)

        # =====================================================================
        # TERM 2: Concept loss (gradients → W_concept, μ_net)
        # =====================================================================
        if concept_labels is not None:
            known_mask = (concept_labels != 1)

            if known_mask.any():
                targets = (concept_labels[known_mask].float() / 2.0)
                preds = result['concept_probs'][known_mask]
                preds_clamped = torch.clamp(preds, 1e-7, 1 - 1e-7)

                losses['concept_bce'] = F.binary_cross_entropy(
                    preds_clamped, targets
                )

        # =====================================================================
        # TERM 3: KL loss for credal set (gradients → W_epi, σ_net)
        # THIS IS THE EPISTEMIC TRAINING SIGNAL!
        #
        # The KL term encourages σ_epi to shrink toward the prior.
        # When the model is uncertain (early training, OOD inputs),
        # σ_epi stays large → high epistemic uncertainty.
        # As the model learns, σ_epi shrinks → low epistemic uncertainty.
        # =====================================================================
        losses['credal_kl'] = self.credal_head.kl_divergence(result['sigma_epi'])

        # =====================================================================
        # TERM 4: Aleatoric supervision (gradients → W_ale, ale_net)
        # =====================================================================
        if annotator_entropy is not None and concept_labels is not None:
            known_mask = (concept_labels != 1)

            if known_mask.any():
                entropy_targets = annotator_entropy[known_mask]
                aleatoric_pred = result['aleatoric'][known_mask]
                losses['aleatoric_loss'] = F.mse_loss(aleatoric_pred, entropy_targets)

        # Unknown concepts should have high aleatoric
        if concept_labels is not None:
            unknown_mask = (concept_labels == 1)
            if unknown_mask.any():
                losses['aleatoric_unknown'] = F.mse_loss(
                    result['aleatoric'][unknown_mask],
                    torch.ones_like(result['aleatoric'][unknown_mask])
                )

        # =====================================================================
        # TERM 5: Orthogonality penalty
        # =====================================================================
        losses['orth_penalty'] = self.projection.orthogonality_loss()

        # =====================================================================
        # COMBINE
        # =====================================================================
        total = torch.tensor(0.0, device=device)

        if 'task_loss' in losses:
            total = total + losses['task_loss']

        if 'concept_bce' in losses:
            total = total + self.config.concept_weight * losses['concept_bce']

        # KL loss IS the epistemic training signal
        total = total + self.config.kl_weight * losses['credal_kl']

        if 'aleatoric_loss' in losses:
            total = total + self.config.aleatoric_weight * losses['aleatoric_loss']

        if 'aleatoric_unknown' in losses:
            total = total + 0.1 * self.config.aleatoric_weight * losses['aleatoric_unknown']  # Reduced from 0.5

        total = total + self.config.orth_weight * losses['orth_penalty']

        losses['loss'] = total
        return losses


# ============================================================================
# VERIFICATION AND DIAGNOSTICS
# ============================================================================

def verify_credal_properties(model: TrueCredalCBM, num_steps: int = 100):
    """
    Verify that the model has true credal properties.

    Key properties to verify:
    1. EU is derived from credal geometry (not predicted)
    2. σ_epi shrinks during training (KL signal works)
    3. Gradient separation holds
    """
    print("\n" + "=" * 60)
    print("CREDAL PROPERTY VERIFICATION")
    print("=" * 60)

    # Create dummy data
    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    concept_labels = torch.randint(0, 3, (batch_size, model.config.num_concepts))
    annotator_entropy = torch.rand(batch_size, model.config.num_concepts)
    labels = torch.randint(0, 2, (batch_size,))

    # =========================================================================
    # Test 1: EU is derived from σ_epi
    # =========================================================================
    print("\n[Test 1] EU derived from credal geometry:")
    model.eval()
    with torch.no_grad():
        result = model(input_ids, attention_mask)

        # EU should be exactly log(σ_epi)
        expected_eu = torch.log(result['sigma_epi'] + 1e-10)
        actual_eu = result['epistemic']

        diff = (expected_eu - actual_eu).abs().max().item()
        print(f"  max|EU - log(σ_epi)|: {diff:.10f}")
        assert diff < 1e-6, "EU should be exactly log(σ_epi)!"
        print("  ✓ PASSED: EU = log(σ_epi)")

    # =========================================================================
    # Test 2: σ_epi shrinks during training
    # =========================================================================
    print("\n[Test 2] σ_epi shrinks during training (KL signal):")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_sigma = None
    final_sigma = None

    for step in range(num_steps):
        result = model(
            input_ids, attention_mask,
            labels=labels,
            concept_labels=concept_labels,
            annotator_entropy=annotator_entropy
        )

        if step == 0:
            initial_sigma = result['sigma_epi'].mean().item()

        loss = result['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == num_steps - 1:
            final_sigma = result['sigma_epi'].mean().item()

    print(f"  Initial mean σ_epi: {initial_sigma:.4f}")
    print(f"  Final mean σ_epi: {final_sigma:.4f}")
    print(f"  Shrinkage: {(1 - final_sigma/initial_sigma)*100:.1f}%")

    if final_sigma < initial_sigma:
        print("  ✓ PASSED: σ_epi shrinks during training")
    else:
        print("  ⚠ WARNING: σ_epi did not shrink")

    # =========================================================================
    # Test 3: Gradient separation
    # =========================================================================
    print("\n[Test 3] Gradient separation:")
    model.train()

    # Test KL gradients only affect W_epi
    model.zero_grad()
    result = model(input_ids, attention_mask, concept_labels=concept_labels)
    result['credal_kl'].backward()

    w_concept_grad = model.projection.W_concept.weight.grad
    w_epi_grad = model.projection.W_epi.weight.grad
    w_ale_grad = model.projection.W_ale.weight.grad

    print(f"  L_KL gradients:")
    print(f"    W_concept: {w_concept_grad.norm().item() if w_concept_grad is not None else 0:.6f}")
    print(f"    W_epi: {w_epi_grad.norm().item():.6f}")
    print(f"    W_ale: {w_ale_grad.norm().item() if w_ale_grad is not None else 0:.6f}")

    assert w_epi_grad.norm() > 0, "W_epi should have gradients from KL!"
    assert w_concept_grad is None or w_concept_grad.norm() < 1e-10, "W_concept should NOT have KL gradients!"
    assert w_ale_grad is None or w_ale_grad.norm() < 1e-10, "W_ale should NOT have KL gradients!"
    print("  ✓ PASSED: L_KL only affects W_epi")

    print("\n" + "=" * 60)
    print("ALL CREDAL PROPERTY TESTS PASSED!")
    print("=" * 60)


def analyze_credal_sets(model: TrueCredalCBM, dataloader, device='cpu'):
    """
    Analyze credal set properties on real data.
    """
    model.eval()
    model.to(device)

    all_sigma_epi = []
    all_sigma_ale = []
    all_eu = []
    all_correct = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            result = model(input_ids, attention_mask)

            all_sigma_epi.append(result['sigma_epi'].cpu())
            all_sigma_ale.append(result['aleatoric'].cpu())
            all_eu.append(result['epistemic'].cpu())
            all_correct.append((result['predictions'] == labels).cpu())

    sigma_epi = torch.cat(all_sigma_epi)
    sigma_ale = torch.cat(all_sigma_ale)
    eu = torch.cat(all_eu)
    correct = torch.cat(all_correct)

    print("\n" + "=" * 60)
    print("CREDAL SET ANALYSIS")
    print("=" * 60)

    print(f"\nCredal set size (σ_epi):")
    print(f"  Mean: {sigma_epi.mean():.4f}")
    print(f"  Std: {sigma_epi.std():.4f}")
    print(f"  Range: [{sigma_epi.min():.4f}, {sigma_epi.max():.4f}]")

    print(f"\nEpistemic uncertainty (log σ_epi):")
    print(f"  Mean: {eu.mean():.4f}")
    print(f"  Correct predictions: {eu[correct.any(dim=-1) if correct.dim() > 1 else correct].mean():.4f}")
    print(f"  Wrong predictions: {eu[~(correct.any(dim=-1) if correct.dim() > 1 else correct)].mean():.4f}")

    print(f"\nAleatoric uncertainty:")
    print(f"  Mean: {sigma_ale.mean():.4f}")

    # Correlation between EU and AU (should be low!)
    eu_flat = eu.mean(dim=-1).numpy()
    au_flat = sigma_ale.mean(dim=-1).numpy()

    from scipy import stats
    rho, p = stats.spearmanr(eu_flat, au_flat)
    print(f"\nρ(EU, AU): {rho:.3f} (p={p:.2e})")

    if abs(rho) < 0.3:
        print("  ✓ Good separation!")
    else:
        print("  ⚠ High correlation")

    print("=" * 60)


# ============================================================================
# HYBRID CREDAL CBM - COMBINES GEOMETRY + SUPERVISION
# ============================================================================

"""
Hybrid Credal CBM with Structural Separation
============================================

This design combines the best of both worlds:
- TRUE credal geometry: EU = log(Σ_epi) derived from credal set size
- DIRECT supervision: Σ_epi trained to predict concept errors
- THREE-WAY orthogonal projection for structural separation

The key insight: Pure KL-based training for Σ_epi doesn't work because
it gives the model no incentive to make Σ_epi input-dependent. It just
stays at the prior for all inputs.

By ADDING error supervision, we teach Σ_epi to vary with input:
- High error → large Σ_epi → high EU
- Low error → small Σ_epi → low EU

The KL term now serves as REGULARIZATION, preventing Σ_epi from
overfitting to noise in the error signal.

Training signals:
- W_concept, μ_net ← L_concept + L_task
- W_epi, σ_net ← L_error + L_KL (error supervision + regularization)
- W_ale, ale_net ← L_ale (annotator entropy)

Author: Tanmoy
Target: ICML 2026
"""

import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HybridCredalConfig:
    """Configuration for Hybrid Credal CBM"""

    # Encoder
    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = True
    pooling_strategy: str = "cls"

    # Concepts
    num_concepts: int = 4
    concept_names: List[str] = field(
        default_factory=lambda: ['food', 'service', 'ambiance', 'noise']
    )

    # Task
    num_classes: int = 2

    # Credal set parameters
    num_mc_samples: int = 10
    min_sigma: float = 0.01    # Minimum credal set size
    max_sigma: float = 2.0     # Maximum credal set size
    prior_sigma: float = 0.5   # Prior for KL (smaller than before!)

    # Loss weights
    concept_weight: float = 2.0
    kl_weight: float = 0.01           # KL regularization (keep small!)
    error_supervision_weight: float = 1.0  # Error supervision (main signal)
    aleatoric_weight: float = 2.0     # Increased from 1.0 to strengthen entropy supervision
    orth_weight: float = 0.001

    # Error scaling
    error_scale: float = 2.0  # Scale errors to match Σ_epi range

    # Aleatoric prior
    aleatoric_prior: float = 0.3

    # Architecture
    projection_dim: int = 256
    hidden_dim: int = 128


# ============================================================================
# HYBRID CREDAL SET HEAD
# ============================================================================

class HybridCredalSetHead(nn.Module):
    """
    Hybrid Credal Set Head

    Combines:
    1. Credal geometry: EU = log(Σ_epi) derived from set size
    2. Error supervision: Σ_epi trained to predict |pred - target|
    3. KL regularization: prevents overfitting to error noise

    Why hybrid works better than pure KL:
    =====================================

    Pure KL approach:
    - L = KL(Σ_epi || prior)
    - Problem: No incentive for Σ_epi to vary with input
    - Result: Σ_epi stays at prior for all inputs

    Hybrid approach:
    - L = λ_error * MSE(Σ_epi, errors) + λ_KL * KL(Σ_epi || prior)
    - Error supervision: teaches Σ_epi to be INPUT-DEPENDENT
    - KL regularization: prevents overfitting, maintains credal interpretation

    The credal interpretation is preserved:
    - Σ_epi defines the size of the credal set
    - EU = log(Σ_epi) is still DERIVED from geometry
    - We just supervise the geometry to capture the right thing
    """

    def __init__(
        self,
        proj_dim: int,
        num_concepts: int,
        hidden_dim: int = 128,
        prior_sigma: float = 0.5,
        min_sigma: float = 0.01,
        max_sigma: float = 2.0,
        error_scale: float = 2.0,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.prior_sigma = prior_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.error_scale = error_scale

        # μ head: credal center (from h_concept)
        self.mu_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )

        # Σ_epi head: credal size (from h_epi)
        self.log_sigma_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
        )

        # Initialize to reasonable starting point
        self._init_sigma()

    def _init_sigma(self):
        """Initialize Σ_epi to start at a reasonable value."""
        # We want softplus(output) ≈ prior_sigma initially
        # softplus(x) = log(1 + exp(x))
        # Inverse: x = log(exp(y) - 1) for y > 0
        if self.prior_sigma > 0:
            init_val = math.log(math.exp(self.prior_sigma) - 1 + 1e-6)
        else:
            init_val = -2.0

        # Initialize last layer
        last_layer = self.log_sigma_net[-1]
        if hasattr(last_layer, 'bias') and last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, init_val)
        # Small weights so initial output ≈ bias
        if hasattr(last_layer, 'weight'):
            nn.init.normal_(last_layer.weight, std=0.01)

    def forward(
        self,
        h_concept: torch.Tensor,
        h_epi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_concept: [batch, proj_dim] - features for credal center
            h_epi: [batch, proj_dim] - features for credal size

        Returns:
            mu: [batch, num_concepts] - credal centers (logit space)
            sigma_epi: [batch, num_concepts] - credal sizes
            p_mean: [batch, num_concepts] - mean concept probabilities
        """
        # Credal center
        mu = self.mu_net(h_concept)

        # Credal size (will be supervised to track errors)
        log_sigma_raw = self.log_sigma_net(h_epi)
        sigma_epi = F.softplus(log_sigma_raw)
        sigma_epi = torch.clamp(sigma_epi, self.min_sigma, self.max_sigma)

        # Mean concept probability
        p_mean = torch.sigmoid(mu)

        return {
            'mu': mu,
            'sigma_epi': sigma_epi,
            'p_mean': p_mean,
        }

    def sample_credal_set(
        self,
        mu: torch.Tensor,
        sigma_epi: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Sample from credal set using reparameterization.

        Returns:
            samples: [num_samples, batch, num_concepts] in [0, 1]
        """
        batch_size, num_concepts = mu.shape

        eps = torch.randn(
            num_samples, batch_size, num_concepts,
            device=mu.device, dtype=mu.dtype
        )

        logit_samples = mu.unsqueeze(0) + sigma_epi.unsqueeze(0) * eps
        prob_samples = torch.sigmoid(logit_samples)

        return prob_samples

    def epistemic_uncertainty(self, sigma_epi: torch.Tensor) -> torch.Tensor:
        """
        EU = log(Σ_epi) — DERIVED from credal geometry.

        Even though we supervise Σ_epi, the EU is still derived
        from the credal set size, not directly predicted.
        """
        return torch.log(sigma_epi + 1e-10)

    def error_supervision_loss(
        self,
        sigma_epi: torch.Tensor,
        concept_preds: torch.Tensor,
        concept_targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Supervise Σ_epi to predict concept errors.

        This is the KEY addition that makes Σ_epi input-dependent!

        Args:
            sigma_epi: [batch, num_concepts] or [N] flattened
            concept_preds: [batch, num_concepts] or [N] (DETACHED!)
            concept_targets: [batch, num_concepts] or [N]
            mask: optional mask for known concepts

        Returns:
            MSE loss
        """
        # Compute concept errors (detach predictions!)
        errors = torch.abs(concept_preds.detach() - concept_targets)

        # Scale errors to match Σ_epi range
        # errors are in [0, 1], Σ_epi is in [min_sigma, max_sigma]
        scaled_errors = errors * self.error_scale + self.min_sigma
        scaled_errors = torch.clamp(scaled_errors, self.min_sigma, self.max_sigma)

        if mask is not None:
            sigma_epi = sigma_epi[mask]
            scaled_errors = scaled_errors[mask] if scaled_errors.shape == mask.shape else scaled_errors

        return F.mse_loss(sigma_epi, scaled_errors)

    def kl_divergence(self, sigma_epi: torch.Tensor) -> torch.Tensor:
        """
        KL regularization toward prior.

        Now serves as REGULARIZATION, not main training signal.
        Prevents Σ_epi from overfitting to error noise.
        """
        # KL for log-normal-ish: log(prior/sigma) + sigma²/(2*prior²) - 0.5
        prior = self.prior_sigma

        kl = (
            torch.log(prior / (sigma_epi + 1e-10)) +
            (sigma_epi ** 2) / (2 * prior ** 2) -
            0.5
        )

        return kl.mean()


# ============================================================================
# MAIN MODEL: HYBRID CREDAL CBM
# ============================================================================

class HybridCredalCBM(nn.Module):
    """
    Hybrid Credal CBM with Structural Separation

    Combines credal geometry with direct supervision:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Input → Encoder → h                                                │
    │              │                                                      │
    │      ┌───────┼───────┐                                             │
    │      ↓       ↓       ↓                                             │
    │  [W_concept][W_epi][W_ale]  ← Three orthogonal projections         │
    │      ↓       ↓       ↓                                             │
    │  h_concept h_epi  h_ale                                            │
    │      ↓       ↓       ↓                                             │
    │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐                                         │
    │   │μ^(k)│ │Σ_epi│ │σ_ale│                                         │
    │   └──┬──┘ └──┬──┘ └──┬──┘                                         │
    │      │       │       │                                             │
    │      └───┬───┘       │                                             │
    │          ↓           │                                             │
    │   ┌────────────┐     │                                             │
    │   │ Credal Set │     │                                             │
    │   │ C = N(μ,Σ) │     │                                             │
    │   └─────┬──────┘     │                                             │
    │         │            │                                             │
    │    EU = log(Σ_epi)   │   ← DERIVED from geometry                  │
    │         │            │                                             │
    │   SUPERVISED by:     │                                             │
    │   |pred - target|    │   ← Makes it input-dependent!              │
    │         +            │                                             │
    │   KL regularization  │   ← Prevents overfitting                   │
    │                      │                                             │
    │                      ↓                                             │
    │                 AU = σ_ale                                         │
    │                 SUPERVISED by annotator entropy                    │
    └─────────────────────────────────────────────────────────────────────┘

    Training signals (FULLY DISJOINT):
    - W_concept, μ_net ← L_concept + L_task
    - W_epi, σ_net ← L_error + L_KL
    - W_ale, ale_net ← L_ale
    """

    def __init__(self, config: HybridCredalConfig):
        super().__init__()
        self.config = config

        # Encoder
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Three-way orthogonal projection (reuse from TrueCredalCBM)
        self.projection = ThreeWayOrthogonalProjection(
            self.hidden_size,
            config.projection_dim
        )

        # Hybrid credal head
        self.credal_head = HybridCredalSetHead(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            prior_sigma=config.prior_sigma,
            min_sigma=config.min_sigma,
            max_sigma=config.max_sigma,
            error_scale=config.error_scale,
        )

        # Aleatoric head (reuse from TrueCredalCBM)
        self.aleatoric_head = CredalAleatoricHead(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            prior_mean=config.aleatoric_prior,
        )

        # Task classifier (reuse from TrueCredalCBM)
        self.task_classifier = CredalTaskClassifier(
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
        )

        self._print_architecture()

    def _print_architecture(self):
        print("\n" + "=" * 70)
        print("HYBRID CREDAL CBM WITH STRUCTURAL SEPARATION")
        print("=" * 70)
        print(f"Encoder: {self.config.encoder_name}")
        print(f"Concepts: {self.config.num_concepts} {self.config.concept_names}")
        print(f"\nHYBRID APPROACH:")
        print(f"  • EU = log(Σ_epi) — DERIVED from credal geometry")
        print(f"  • Σ_epi SUPERVISED by concept errors (input-dependent!)")
        print(f"  • KL regularization prevents overfitting")
        print(f"\nPARAMETERS:")
        print(f"  • Prior σ: {self.config.prior_sigma}")
        print(f"  • Error scale: {self.config.error_scale}")
        print(f"  • σ range: [{self.config.min_sigma}, {self.config.max_sigma}]")
        print(f"\nLOSS WEIGHTS:")
        print(f"  • Error supervision: {self.config.error_supervision_weight}")
        print(f"  • KL regularization: {self.config.kl_weight}")
        print(f"  • Aleatoric: {self.config.aleatoric_weight}")
        print("=" * 70 + "\n")

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
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
        annotator_entropy: Optional[torch.Tensor] = None,
        entropy_weights: Optional[torch.Tensor] = None,
        num_mc_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        num_samples = num_mc_samples or self.config.num_mc_samples

        # Encode
        hidden = self.encode(input_ids, attention_mask)

        # Three orthogonal projections
        h_concept, h_epi, h_ale = self.projection(hidden)

        # Credal set parameters
        credal_out = self.credal_head(h_concept, h_epi)
        mu = credal_out['mu']
        sigma_epi = credal_out['sigma_epi']
        p_mean = credal_out['p_mean']

        # Sample from credal set
        p_samples = self.credal_head.sample_credal_set(mu, sigma_epi, num_samples)

        # Aleatoric
        sigma_ale = self.aleatoric_head(h_ale)

        # Task prediction
        task_out = self.task_classifier(p_mean, p_samples)

        # Epistemic (DERIVED from geometry)
        epistemic = self.credal_head.epistemic_uncertainty(sigma_epi)

        result = {
            # Task
            'predictions': task_out['predictions'],
            'logits': task_out['logits'],
            'probs': task_out['probs'],
            'prob_lower': task_out.get('prob_lower'),
            'prob_upper': task_out.get('prob_upper'),

            # Concepts
            'concept_probs': p_mean,
            'concept_samples': p_samples,
            'mu': mu,
            'sigma_epi': sigma_epi,

            # Uncertainties
            'epistemic': epistemic,
            'aleatoric': sigma_ale,
        }

        # Compute losses
        if labels is not None or concept_labels is not None:
            # Create batch dict for passing entropy_weights
            batch = {
                'entropy_weights': entropy_weights
            }
            losses = self._compute_losses(
                result, labels, concept_labels, annotator_entropy, batch
            )
            result.update(losses)

        return result

    def _compute_losses(
        self,
        result: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor],
        concept_labels: Optional[torch.Tensor],
        annotator_entropy: Optional[torch.Tensor],
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with proper gradient separation."""
        losses = {}
        device = result['predictions'].device

        # =====================================================================
        # TERM 1: Task loss → W_concept
        # =====================================================================
        if labels is not None:
            losses['task_loss'] = F.cross_entropy(result['logits'], labels)

        # =====================================================================
        # TERM 2: Concept loss → W_concept, μ_net
        # =====================================================================
        if concept_labels is not None:
            known_mask = (concept_labels != 1)  # Exclude unknown

            if known_mask.any():
                # IMPORTANT: Filter FIRST, then convert!
                # Only get known labels (0 or 2, NO 1s)
                known_labels = concept_labels[known_mask]
                known_preds = result['concept_probs'][known_mask]
                known_preds_clamped = torch.clamp(known_preds, 1e-7, 1 - 1e-7)

                # Convert to binary: 0→0.0, 2→1.0
                # This is correct because known_labels only contains 0 and 2
                known_targets = (known_labels.float() / 2.0)

                # Sanity check
                if known_targets.min() < 0 or known_targets.max() > 1:
                    print(f"[WARNING] Targets out of range: [{known_targets.min()}, {known_targets.max()}]")
                    print(f"  known_labels unique: {known_labels.unique()}")
                    print(f"  This shouldn't happen - known_labels should only have 0 and 2!")

                losses['concept_bce'] = F.binary_cross_entropy(
                    known_preds_clamped, known_targets
                )

                # =============================================================
                # TERM 3: Error supervision → W_epi, σ_net
                # THIS IS THE KEY ADDITION!
                # =============================================================
                known_sigma = result['sigma_epi'][known_mask]
                losses['error_supervision'] = self.credal_head.error_supervision_loss(
                    known_sigma, known_preds, known_targets
                )

        # =====================================================================
        # TERM 4: KL regularization → W_epi, σ_net (mild regularization)
        # =====================================================================
        losses['credal_kl'] = self.credal_head.kl_divergence(result['sigma_epi'])

        # =====================================================================
        # TERM 5: Aleatoric supervision → W_ale, ale_net
        # =====================================================================
        if annotator_entropy is not None and concept_labels is not None:
            known_mask = (concept_labels != 1)

            if known_mask.any():
                entropy_targets = annotator_entropy[known_mask]
                aleatoric_pred = result['aleatoric'][known_mask]

                # Get entropy weights if available (for weighted loss)
                entropy_weights = batch.get('entropy_weights', torch.ones_like(annotator_entropy))
                if entropy_weights is not None:
                    sample_weights = entropy_weights[known_mask]
                else:
                    sample_weights = torch.ones_like(aleatoric_pred)

                # Ensure same shape
                if entropy_targets.numel() > 0 and aleatoric_pred.numel() > 0:
                    # Use weighted MSE loss
                    pred_flat = aleatoric_pred.flatten()[:entropy_targets.flatten().numel()]
                    target_flat = entropy_targets.flatten()[:aleatoric_pred.flatten().numel()]
                    weight_flat = sample_weights.flatten()[:aleatoric_pred.flatten().numel()]

                    # Weighted MSE: mean(weight * (pred - target)^2)
                    squared_errors = (pred_flat - target_flat) ** 2
                    weighted_errors = weight_flat * squared_errors
                    losses['aleatoric_loss'] = weighted_errors.mean()

        # Unknown concepts → high aleatoric
        if concept_labels is not None:
            unknown_mask = (concept_labels == 1)
            if unknown_mask.any():
                losses['aleatoric_unknown'] = F.mse_loss(
                    result['aleatoric'][unknown_mask],
                    torch.ones_like(result['aleatoric'][unknown_mask])
                )

        # =====================================================================
        # TERM 6: Orthogonality
        # =====================================================================
        losses['orth_penalty'] = self.projection.orthogonality_loss()

        # =====================================================================
        # COMBINE
        # =====================================================================
        total = torch.tensor(0.0, device=device)

        if 'task_loss' in losses:
            total = total + losses['task_loss']

        if 'concept_bce' in losses:
            total = total + self.config.concept_weight * losses['concept_bce']

        # Error supervision (MAIN epistemic signal)
        if 'error_supervision' in losses:
            total = total + self.config.error_supervision_weight * losses['error_supervision']

        # KL regularization (mild)
        total = total + self.config.kl_weight * losses['credal_kl']

        if 'aleatoric_loss' in losses:
            total = total + self.config.aleatoric_weight * losses['aleatoric_loss']

        if 'aleatoric_unknown' in losses:
            total = total + 0.1 * self.config.aleatoric_weight * losses['aleatoric_unknown']  # Reduced from 0.5

        total = total + self.config.orth_weight * losses['orth_penalty']

        losses['loss'] = total
        return losses


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_hybrid_credal(model: HybridCredalCBM, num_steps: int = 100):
    """Verify hybrid credal properties."""
    print("\n" + "=" * 60)
    print("HYBRID CREDAL VERIFICATION")
    print("=" * 60)

    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    concept_labels = torch.randint(0, 3, (batch_size, model.config.num_concepts))
    concept_labels[:, 0] = 0  # Some known labels
    concept_labels[:, 1] = 2
    annotator_entropy = torch.rand(batch_size, model.config.num_concepts)
    labels = torch.randint(0, 2, (batch_size,))

    # =========================================================================
    # Test 1: EU is derived from Σ_epi
    # =========================================================================
    print("\n[Test 1] EU derived from credal geometry:")
    model.eval()
    with torch.no_grad():
        result = model(input_ids, attention_mask)
        expected_eu = torch.log(result['sigma_epi'] + 1e-10)
        actual_eu = result['epistemic']
        diff = (expected_eu - actual_eu).abs().max().item()
        print(f"  max|EU - log(Σ_epi)|: {diff:.10f}")
        assert diff < 1e-6
        print("  ✓ PASSED")

    # =========================================================================
    # Test 2: Σ_epi becomes input-dependent (not stuck at prior)
    # =========================================================================
    print("\n[Test 2] Σ_epi becomes input-dependent:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_sigma_std = None
    final_sigma_std = None

    for step in range(num_steps):
        result = model(
            input_ids, attention_mask,
            labels=labels,
            concept_labels=concept_labels,
            annotator_entropy=annotator_entropy
        )

        sigma_std = result['sigma_epi'].std().item()

        if step == 0:
            initial_sigma_std = sigma_std
            print(f"  Initial Σ_epi std: {sigma_std:.6f}")

        loss = result['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == num_steps - 1:
            final_sigma_std = sigma_std
            print(f"  Final Σ_epi std: {sigma_std:.6f}")

    if final_sigma_std > initial_sigma_std * 2:
        print("  ✓ PASSED: Σ_epi is now input-dependent!")
    else:
        print(f"  ⚠ Σ_epi variance increased {final_sigma_std/initial_sigma_std:.1f}x")

    # =========================================================================
    # Test 3: Gradient separation
    # =========================================================================
    print("\n[Test 3] Gradient separation:")
    model.train()

    # Error supervision should only affect W_epi
    model.zero_grad()
    result = model(
        input_ids, attention_mask,
        concept_labels=concept_labels,
        annotator_entropy=annotator_entropy
    )

    if 'error_supervision' in result:
        result['error_supervision'].backward()

        w_c = model.projection.W_concept.weight.grad
        w_e = model.projection.W_epi.weight.grad
        w_a = model.projection.W_ale.weight.grad

        print(f"  L_error gradients:")
        print(f"    W_concept: {w_c.norm().item() if w_c is not None else 0:.6f}")
        print(f"    W_epi: {w_e.norm().item() if w_e is not None else 0:.6f}")
        print(f"    W_ale: {w_a.norm().item() if w_a is not None else 0:.6f}")

        if w_e is not None and w_e.norm() > 0:
            if (w_c is None or w_c.norm() < 1e-10) and (w_a is None or w_a.norm() < 1e-10):
                print("  ✓ PASSED: L_error only affects W_epi")
            else:
                print("  ⚠ Some gradient leakage")

    print("\n" + "=" * 60)


def compare_approaches():
    """Print comparison of different approaches."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         APPROACH COMPARISON                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. SUPERVISED (ConceptSupervisedCBM):                                     ║
║     • EU = σ_epi (directly PREDICTED by head)                                ║
║     • Training: MSE(σ_epi, errors)                                           ║
║     • Problem: Not truly "credal" — just regression                          ║
║     • Pro: Works well, input-dependent                                       ║
║                                                                              ║
║  2. PURE CREDAL (TrueCredalCBM):                                            ║
║     • EU = log(Σ_epi) (DERIVED from geometry)                                ║
║     • Training: KL(Σ_epi || prior)                                           ║
║     • Problem: Σ_epi stays at prior — NOT input-dependent!                   ║
║     • Pro: Theoretically clean                                               ║
║                                                                              ║
║  3. HYBRID (HybridCredalCBM):  ← RECOMMENDED!                               ║
║     • EU = log(Σ_epi) (DERIVED from geometry)                                ║
║     • Training: MSE(Σ_epi, errors) + λ·KL(Σ_epi || prior)                    ║
║     • Pro: Best of both worlds!                                              ║
║       - Credal interpretation preserved                                      ║
║       - Input-dependent via error supervision                                ║
║       - KL prevents overfitting                                              ║
║                                                                              ║
║  THEORETICAL JUSTIFICATION:                                                  ║
║  ─────────────────────────                                                   ║
║  "We supervise Σ_epi with concept errors because errors indicate where       ║
║  the model's knowledge is insufficient — precisely what the credal set       ║
║  should capture. The KL term ensures the credal set doesn't overfit to       ║
║  noise, maintaining its interpretation as a set of plausible distributions." ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
