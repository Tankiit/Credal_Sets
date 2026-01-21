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
            total = total + 0.5 * self.config.aleatoric_weight * losses['aleatoric_unknown']

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
