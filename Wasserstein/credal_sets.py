"""
Credal DRO Module — Concept Heads + Ellipsoidal DRO
=====================================================

Implements:
  1. ConceptHead: Single MLP head mapping latents → concept probabilities
  2. ConceptEnsemble: N heads producing μ(x), σ²(x) per concept
  3. CredalEllipsoid: Geometry utilities (projection, ε, width penalty)
  4. PGDInnerMax: Projected gradient ascent to find worst-case p* in C(x)
  5. CredalDROModule: Combines everything into a single forward pass

Math reference (UAI doc §2):
  C(x) = { p ∈ [0,1]^K : Σ_k (p_k - μ_k)² / σ_k² ≤ 1 }
  L_robust = max_{p ∈ C(x)} CE(Wp + b, y)
  ε(x) = √(tr(Σ_epi)) = √(Σ_k σ_k²)
  L_total = L_task + λ_c·L_concept + λ_dro·L_robust + β·Ω(Σ)

Author: Tanmoy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Optional, List
import math


# =============================================================================
# CONFIGURATION
# =============================================================================

class DROMode(Enum):
    """Three-way comparison modes from UAI doc §5.2."""
    POST_HOC = "post_hoc"       # (A) λ_dro=0, train ELBO only, extract ε after
    FIXED_EPS = "fixed_eps"     # (B) stop_grad on σ², fixed ε for DRO
    JOINT = "joint"             # (C) ε(x) learned jointly, gradients through σ²


@dataclass
class CredalDROConfig:
    """
    Full configuration for the credal DRO pipeline with phased training.

    Features:
      - Phased warmup (warmup_epochs + dro_ramp_epochs)
      - Diversity loss (diversity_weight + diversity_mode)
      - Scheduling helpers (get_epoch_weights)
    """

    # ══════════════════════════════════════════════════════════════════
    # Architecture
    # ══════════════════════════════════════════════════════════════════
    num_concepts: int = 4
    num_classes: int = 3
    input_dim: int = 128            # latent dimension from encoder
    head_hidden_dim: int = 256      # MLP width inside each concept head
    n_heads: int = 5                # ensemble size
    concept_classes: int = 3        # 2=binary, 3=ternary {neg, unk, pos}
    unknown_weight: float = 0.5     # CE weight for unknown class

    # ══════════════════════════════════════════════════════════════════
    # DRO mode (three-way comparison)
    # ══════════════════════════════════════════════════════════════════
    mode: DROMode = DROMode.JOINT

    # ══════════════════════════════════════════════════════════════════
    # PGD inner loop
    # ══════════════════════════════════════════════════════════════════
    pgd_steps: int = 10
    pgd_lr: float = 0.01

    # ══════════════════════════════════════════════════════════════════
    # Loss weights: L_task + λ_c·L_concept + λ_dro·L_robust + β·Ω(Σ) + λ_ale·L_ale
    # ══════════════════════════════════════════════════════════════════
    lambda_dro: float = 0.1         # weight on worst-case loss
    beta_width: float = 0.01        # weight on width penalty
    lambda_concept: float = 1.0     # weight on concept supervision

    # ══════════════════════════════════════════════════════════════════
    # Width penalty
    # ══════════════════════════════════════════════════════════════════
    width_penalty: str = "log_det"  # "log_det" | "trace" | "none"

    # ══════════════════════════════════════════════════════════════════
    # Sigma bounds
    # ══════════════════════════════════════════════════════════════════
    sigma_min: float = 1e-4
    sigma_max: float = 2.0

    # ══════════════════════════════════════════════════════════════════
    # Dropout diversity
    # ══════════════════════════════════════════════════════════════════
    dropout_min: float = 0.05
    dropout_max: float = 0.30

    # ══════════════════════════════════════════════════════════════════
    # Fixed-ε baseline
    # ══════════════════════════════════════════════════════════════════
    fixed_eps: float = 0.1

    # ══════════════════════════════════════════════════════════════════
    # Huber OOD
    # ══════════════════════════════════════════════════════════════════
    use_huber: bool = False
    huber_tau: float = 2.0

    # ══════════════════════════════════════════════════════════════════
    # Aleatoric head
    # ══════════════════════════════════════════════════════════════════
    use_aleatoric: bool = False     # whether to predict aleatoric uncertainty
    aleatoric_hidden_dim: int = 128  # hidden dim for aleatoric head
    lambda_ale: float = 1.0          # weight on aleatoric MSE loss
    use_aleatoric_weighting: bool = False  # whether to weight concepts by (1-a)

    # ══════════════════════════════════════════════════════════════════
    # NEW: Phased Training (warmup → ramp → full)
    # ══════════════════════════════════════════════════════════════════
    warmup_epochs: int = 0          # Phase 1: concept-only warmup (λ_dro=0)
    dro_ramp_epochs: int = 5        # Phase 2: linearly ramp λ_dro from 0 → base
    # Phase 3: full training (all losses at base values)
    # Total: warmup + ramp + remaining = num_epochs

    # ══════════════════════════════════════════════════════════════════
    # NEW: Diversity Loss
    # ══════════════════════════════════════════════════════════════════
    diversity_weight: float = 0.0   # 0 = off. Try 0.05-0.1 for hard tasks.
    diversity_mode: str = "variance"  # "variance" | "cosine" | "det_kernel"
    diversity_decay: float = 0.5    # decay factor applied at phase transitions
    # Phase 1: diversity_weight (full, encourage initial disagreement)
    # Phase 2: diversity_weight * decay (relax as DRO takes over)
    # Phase 3: div disabled

    # ══════════════════════════════════════════════════════════════════
    # Training utilities
    # ══════════════════════════════════════════════════════════════════
    early_stopping_patience: int = 0  # 0 = disabled. >0 = stop after N epochs without val improvement
    weight_decay: float = 0.0         # Adam weight decay

    def get_effective_lambda_dro(self) -> float:
        """POST_HOC mode zeroes out the DRO loss."""
        if self.mode == DROMode.POST_HOC:
            return 0.0
        return self.lambda_dro

    def get_stop_grad_sigma(self) -> bool:
        """FIXED_EPS and POST_HOC don't backprop through σ²."""
        return self.mode != DROMode.JOINT

    def get_epoch_weights(self, epoch: int) -> Dict[str, float]:
        """
        Compute effective loss weights for a given epoch.

        Three phases:
          Phase 1 (warmup):  λ_dro=0, diversity at full strength
          Phase 2 (ramp):    λ_dro linearly increases, diversity decays
          Phase 3 (full):    all weights at base values, diversity minimal

        Args:
            epoch: 0-indexed current epoch

        Returns:
            dict with effective weights and current phase label
        """
        warmup_end = self.warmup_epochs
        ramp_end = self.warmup_epochs + self.dro_ramp_epochs

        if epoch < warmup_end:
            # Phase 1: Concept warmup — no DRO, full diversity
            phase = "warmup"
            eff_lambda_dro = 0.0
            eff_diversity = self.diversity_weight

        elif epoch < ramp_end:
            # Phase 2: DRO ramp — linear increase, diversity decays
            phase = "ramp"
            ramp_progress = (epoch - warmup_end) / max(1, self.dro_ramp_epochs)
            base_dro = self.get_effective_lambda_dro()
            eff_lambda_dro = base_dro * ramp_progress
            eff_diversity = self.diversity_weight * self.diversity_decay

        else:
            # Phase 3: Full training — all at base values
            phase = "full"
            eff_lambda_dro = self.get_effective_lambda_dro()
            eff_diversity = self.diversity_weight * (self.diversity_decay ** 2)

        return {
            'lambda_dro': eff_lambda_dro,
            'lambda_concept': self.lambda_concept,
            'beta_width': self.beta_width,
            'diversity_weight': eff_diversity,
            'lambda_ale': self.lambda_ale if self.use_aleatoric else 0.0,
            'phase': phase,
            'epoch': epoch,
        }

    def describe(self) -> str:
        """Human-readable config summary."""
        lines = [
            "CredalDROConfig",
            "=" * 50,
            f"  Mode:           {self.mode.value}",
            f"  Concepts:       {self.num_concepts} ({self.concept_classes}-class)",
            f"  Classes:        {self.num_classes}",
            f"  Heads:          {self.n_heads} (dropout {self.dropout_min}-{self.dropout_max})",
            f"  ─── Losses ───",
            f"  λ_concept:      {self.lambda_concept}",
            f"  λ_dro:          {self.lambda_dro}",
            f"  β_width:        {self.beta_width} ({self.width_penalty})",
            f"  ─── Phased Training ───",
            f"  Warmup:         {self.warmup_epochs} epochs",
            f"  DRO ramp:       {self.dro_ramp_epochs} epochs",
            f"  ─── Diversity ───",
            f"  Weight:         {self.diversity_weight} ({self.diversity_mode})",
            f"  Decay:          {self.diversity_decay}",
            f"  ─── Sigma ───",
            f"  Range:          [{self.sigma_min}, {self.sigma_max}]",
        ]
        if self.use_aleatoric:
            lines.append(f"  ─── Aleatoric ───")
            lines.append(f"  λ_ale:          {self.lambda_ale}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# PRESET CONFIGS for quick experiments
# ═══════════════════════════════════════════════════════════════════════

def cebab_joint_config(**overrides) -> CredalDROConfig:
    """CEBaB joint training preset — 4 concepts, ternary."""
    cfg = CredalDROConfig(
        num_concepts=4,
        num_classes=3,
        concept_classes=3,
        unknown_weight=0.5,
        mode=DROMode.JOINT,
        n_heads=5,
        lambda_dro=0.1,
        beta_width=0.01,
        lambda_concept=1.0,
        warmup_epochs=0,        # CEBaB doesn't need warmup
        dro_ramp_epochs=5,
        diversity_weight=0.0,   # log-barrier sufficient for 4 concepts
        use_aleatoric=True,
        lambda_ale=1.0,
        early_stopping_patience=10,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def goemotions_joint_config(**overrides) -> CredalDROConfig:
    """GoEmotions joint training — needs warmup + diversity."""
    cfg = CredalDROConfig(
        num_concepts=28,        # 28 emotions
        num_classes=28,         # multi-class
        concept_classes=3,
        unknown_weight=0.3,
        mode=DROMode.JOINT,
        n_heads=5,
        lambda_dro=0.1,
        beta_width=0.01,
        lambda_concept=1.0,
        warmup_epochs=10,       # concepts need time to learn before DRO
        dro_ramp_epochs=5,      # gentle ramp
        diversity_weight=0.1,   # explicit push for 28-class task
        diversity_mode="variance",
        diversity_decay=0.5,
        early_stopping_patience=10,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def snli_joint_config(**overrides) -> CredalDROConfig:
    """SNLI joint training — NLI with annotator disagreement concepts."""
    cfg = CredalDROConfig(
        num_concepts=3,        # entailment, neutral, contradiction
        num_classes=3,         # 3-way NLI
        concept_classes=3,     # ternary: low/unknown/high
        unknown_weight=0.3,
        mode=DROMode.JOINT,
        n_heads=5,
        lambda_dro=0.1,
        beta_width=0.01,
        lambda_concept=1.0,
        warmup_epochs=5,       # warmup for concept learning
        dro_ramp_epochs=5,     # gentle ramp
        diversity_weight=0.05, # moderate diversity
        diversity_mode="variance",
        diversity_decay=0.5,
        early_stopping_patience=10,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_three_way_configs(
    num_concepts=4, num_classes=3, input_dim=128, **shared
) -> Dict[str, CredalDROConfig]:
    """Three-way comparison configs with identical base settings."""
    base = dict(
        num_concepts=num_concepts,
        num_classes=num_classes,
        input_dim=input_dim,
        **shared,
    )
    return {
        "post_hoc": CredalDROConfig(**base, mode=DROMode.POST_HOC,
                                     lambda_dro=0.0, beta_width=0.0),
        "fixed_eps": CredalDROConfig(**base, mode=DROMode.FIXED_EPS,
                                      lambda_dro=0.1, fixed_eps=0.1),
        "joint": CredalDROConfig(**base, mode=DROMode.JOINT,
                                  lambda_dro=0.1, beta_width=0.01),
    }


# =============================================================================
# CONCEPT HEAD (single head) - Unified binary/ternary support
# =============================================================================

class ConceptHead(nn.Module):
    """
    Single ensemble member: latent → K concept predictions.

    For ternary concepts (neg/unk/pos): outputs [B, K, 3] logits.
    For binary concepts: outputs [B, K] probabilities via sigmoid.
    """
    def __init__(self, input_dim: int, num_concepts: int, hidden_dim: int = 256,
                 dropout: float = 0.1, concept_classes: int = 2):
        super().__init__()
        self.concept_classes = concept_classes
        self.num_concepts = num_concepts
        out_dim = num_concepts * concept_classes if concept_classes > 2 else num_concepts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            probs: [B, K] concept probabilities (positive-class prob)
            logits: [B, K, C] raw logits (for ternary CE supervision), or None for binary
        """
        raw = self.net(x)  # [B, K*C] or [B, K]
        if self.concept_classes > 2:
            B = x.shape[0]
            logits = raw.view(B, self.num_concepts, self.concept_classes)  # [B, K, 3]
            # Concept probability = P(positive) from softmax
            # Convention: class 0=neg, 1=unk, 2=pos
            probs = F.softmax(logits, dim=-1)[:, :, 2]  # [B, K] positive prob
            return probs, logits
        else:
            probs = torch.sigmoid(raw)  # [B, K]
            return probs, None


# =============================================================================
# CONCEPT ENSEMBLE → μ(x), σ²(x)
# =============================================================================

class ConceptEnsemble(nn.Module):
    """
    N concept heads whose disagreement defines the credal set.

    μ_k(x)  = mean_h[ p_h^(k)(x) ]       — ensemble mean per concept
    σ²_k(x) = var_h[ p_h^(k)(x) ]         — ensemble variance per concept
                                             (epistemic uncertainty)

    The credal ellipsoid C(x) has centre μ and axes σ.
    """

    def __init__(self, config: CredalDROConfig):
        super().__init__()
        self.config = config
        self.concept_classes = getattr(config, 'concept_classes', 2)

        # Create N heads with linearly spaced dropout for diversity
        dropouts = torch.linspace(
            config.dropout_min, config.dropout_max, config.n_heads
        ).tolist()

        self.heads = nn.ModuleList([
            ConceptHead(
                input_dim=config.input_dim,
                num_concepts=config.num_concepts,
                hidden_dim=config.head_hidden_dim,
                dropout=d,
                concept_classes=self.concept_classes,
            )
            for d in dropouts
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, D] latent features
        Returns:
            mu:         [B, K]     ensemble mean concept probs
            sigma_sq:   [B, K]     ensemble variance (epistemic)
            all_probs:  [N, B, K]  per-head positive probs
            all_logits: [N, B, K, C] or None  per-head logits (for ternary CE)
        """
        all_probs = []
        all_logits = []
        for head in self.heads:
            probs, logits = head(x)
            all_probs.append(probs)
            if logits is not None:
                all_logits.append(logits)

        # Stack all head predictions: [N, B, K]
        all_probs = torch.stack(all_probs, dim=0)

        # Ensemble mean and variance across heads (dim=0)
        mu = all_probs.mean(dim=0)              # [B, K]
        sigma_sq = all_probs.var(dim=0)         # [B, K]  (unbiased by default)

        # Clamp sigma to configured bounds
        sigma_sq = sigma_sq.clamp(
            min=self.config.sigma_min ** 2,
            max=self.config.sigma_max ** 2,
        )

        # Stack logits if available
        all_logits = torch.stack(all_logits, dim=0) if all_logits else None

        return mu, sigma_sq, all_probs, all_logits


# =============================================================================
# LABEL HEAD: concepts → class logits
# =============================================================================

class LabelHead(nn.Module):
    """
    Linear map from concept probabilities to class logits.
    L_robust = max_{p ∈ C(x)} CE(Wp + b, y)

    Keeping this linear makes the DRO inner loop a convex problem
    (CE is convex in logits, logits are linear in p).
    """

    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: [B, K] concept vector (could be μ or adversarial p*)
        Returns:
            [B, J] class logits
        """
        return self.linear(p)

    @property
    def W(self) -> torch.Tensor:
        """Weight matrix [J, K]."""
        return self.linear.weight

    @property
    def b(self) -> torch.Tensor:
        """Bias vector [J]."""
        return self.linear.bias


# =============================================================================
# ALEATORIC HEAD: features → per-concept ambiguity
# =============================================================================

class AleatoricHead(nn.Module):
    """
    Predicts per-concept aleatoric ambiguity a_k(x) in [0,1].
    Supervise with concept_entropy (normalised to [0,1]) from annotator distributions.

    The aleatoric uncertainty captures inherent ambiguity in the data due to
    annotator disagreement, as opposed to epistemic uncertainty (captured
    by the ensemble variance σ²).

    Loss: MSE(a_pred, a_target) where a_target = normalized entropy
    """

    def __init__(self, input_dim: int, num_concepts: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] latent features
        Returns:
            [B, K] aleatoric uncertainty in (0, 1) for each concept
        """
        return torch.sigmoid(self.net(features))


# =============================================================================
# TERNARY CONCEPT LOSS
# =============================================================================

def compute_ternary_concept_loss(
    all_logits: torch.Tensor,   # [N, B, K, C]
    concept_labels: torch.Tensor,  # [B, K] with values in {0, 1, 2}
    is_unknown: Optional[torch.Tensor] = None,  # [B, K] mask
    unknown_weight: float = 0.3,  # downweight unknown labels
) -> torch.Tensor:
    """
    Cross-entropy concept loss across all heads.

    Optionally downweights "unknown" (class 1) labels since they're
    overrepresented and less informative.

    Args:
        all_logits:     [N, B, K, C] per-head concept logits
        concept_labels: [B, K] integer targets {0, 1, 2}
        is_unknown:     [B, K] binary mask (1 = unknown concept)
        unknown_weight: weight for unknown concept supervision (< 1 to downweight)
    """
    N, B, K, C = all_logits.shape

    # Per-sample, per-concept weights
    if is_unknown is not None:
        # Known concepts get weight 1.0, unknown get unknown_weight
        weights = torch.where(is_unknown.bool(), unknown_weight, 1.0)  # [B, K]
    else:
        weights = torch.ones(B, K, device=all_logits.device)

    total_loss = torch.tensor(0.0, device=all_logits.device)

    for h in range(N):
        logits_h = all_logits[h]  # [B, K, C]
        # Reshape for CE: [B*K, C] vs [B*K]
        logits_flat = logits_h.reshape(B * K, C)
        targets_flat = concept_labels.reshape(B * K).long()
        weights_flat = weights.reshape(B * K)

        # Weighted CE
        loss_per_element = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # [B*K]
        total_loss = total_loss + (loss_per_element * weights_flat).mean()

    return total_loss / N


# =============================================================================
# CREDAL ELLIPSOID GEOMETRY
# =============================================================================

class CredalEllipsoid:
    """
    Static utilities for the axis-aligned ellipsoid in concept space:
      C(x) = { p ∈ [0,1]^K : Σ_k (p_k - μ_k)² / σ_k² ≤ 1 }
    """

    @staticmethod
    def project_onto_ellipsoid(
        p: torch.Tensor,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project p back onto the credal ellipsoid (and box [0,1]^K).

        Uses Mahalanobis rescaling:
          if ||p - μ||_Σ > 1, rescale direction to unit Mahalanobis norm
          then clamp to [0, 1]

        Args:
            p:        [B, K] current point
            mu:       [B, K] ellipsoid centre
            sigma_sq: [B, K] axis variances
        Returns:
            [B, K] projected point
        """
        delta = p - mu                                      # [B, K]
        # Mahalanobis squared norm: Σ_k δ_k² / σ_k²
        maha_sq = (delta ** 2 / sigma_sq.clamp(min=1e-8)).sum(dim=-1, keepdim=True)  # [B, 1]

        # Rescale if outside ellipsoid
        scale = torch.where(
            maha_sq > 1.0,
            1.0 / (maha_sq.sqrt() + 1e-8),
            torch.ones_like(maha_sq),
        )
        p_proj = mu + delta * scale                         # [B, K]

        # Box constraint
        p_proj = p_proj.clamp(0.0, 1.0)

        return p_proj

    @staticmethod
    def epsilon(sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Instance-specific robustness radius.
          ε(x) = √(tr(Σ_epi)) = √(Σ_k σ_k²)

        Args:
            sigma_sq: [B, K]
        Returns:
            [B] robustness radii
        """
        return sigma_sq.sum(dim=-1).sqrt()                  # [B]

    @staticmethod
    def width_penalty_log_det(sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Ω(Σ) = -log det(Σ) = -Σ_k log(σ_k²)  [FIXED: negative sign added]

        FIXED: The penalty should be POSITIVE and go to +∞ as σ² → 0.
        This creates a barrier against collapse.

        OLD (BUGGY): Ω = +log(σ²) → negative when σ² < 1 → rewards collapse
        NEW (FIXED): Ω = -log(σ²) → positive barrier against collapse

        Gradient ∝ -1/σ_k² — pushes σ² AWAY from zero.

        Args:
            sigma_sq: [B, K]
        Returns:
            [B] negative log-det penalty per example (barrier function)
        """
        # FIXED: Added negative sign to create barrier against collapse
        return -torch.log(sigma_sq.clamp(min=1e-8)).sum(dim=-1)   # [B]

    @staticmethod
    def width_penalty_trace(sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Ω(Σ) = tr(Σ) = Σ_k σ_k²

        Args:
            sigma_sq: [B, K]
        Returns:
            [B] trace penalty per example
        """
        return sigma_sq.sum(dim=-1)                         # [B]


# =============================================================================
# PGD INNER MAXIMISATION
# =============================================================================

class PGDInnerMax(nn.Module):
    """
    Projected Gradient Descent to solve the DRO inner problem:
      p* = argmax_{p ∈ C(x)} CE(Wp + b, y)

    For each input x, we start at μ(x) and take T gradient-ascent
    steps on CE loss, projecting back onto the credal ellipsoid
    after each step.

    The label head (W, b) is linear, so CE(Wp+b, y) is convex in p.
    PGD finds the global max (up to projection approximation).
    """

    def __init__(self, config: CredalDROConfig):
        super().__init__()
        self.steps = config.pgd_steps
        self.lr = config.pgd_lr

    def forward(
        self,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu:       [B, K]  ellipsoid centre
            sigma_sq: [B, K]  ellipsoid axis variances
            W:        [J, K]  label head weight
            b:        [J]     label head bias
            y:        [B]     true labels (long)
        Returns:
            p_star:   [B, K]  worst-case concept vector
            loss_wc:  [B]     worst-case CE loss per example
        """
        B, K = mu.shape

        # Detach — PGD operates in a no-grad-for-model context.
        # We only differentiate w.r.t. p (the adversarial variable).
        mu_d = mu.detach()
        sigma_sq_d = sigma_sq.detach()
        W_d = W.detach()
        b_d = b.detach()

        # Initialise at ellipsoid centre
        p = mu_d.clone()

        # Manual gradient ascent loop (enable gradients for p only)
        with torch.enable_grad():
            for _ in range(self.steps):
                # Forward: logits = p @ W^T + b
                p = p.detach().requires_grad_(True)
                logits = F.linear(p, W_d, b_d)         # [B, J]
                loss = F.cross_entropy(logits, y, reduction='none')  # [B]

                # Gradient ascent on p
                grad_p = torch.autograd.grad(loss.sum(), p, create_graph=False)[0]

                # Update p
                p = p.detach() + self.lr * grad_p

                # Project back onto ellipsoid ∩ [0,1]^K
                p = CredalEllipsoid.project_onto_ellipsoid(p, mu_d, sigma_sq_d)

        # Final worst-case loss (with model gradients this time)
        # Recompute using the *live* W, b so that label head gets gradients
        logits_wc = F.linear(p, W, b)      # [B, J]  — p detached, W/b live
        loss_wc = F.cross_entropy(logits_wc, y, reduction='none')  # [B]

        return p, loss_wc


# =============================================================================
# FIXED-ε PGD VARIANT
# =============================================================================

class FixedEpsPGDInnerMax(nn.Module):
    """
    PGD inner max with a uniform ε for all examples.
    The ambiguity set becomes a hypersphere: ||p - μ||₂ ≤ ε.

    Used as baseline (B) in the three-way comparison.
    """

    def __init__(self, config: CredalDROConfig):
        super().__init__()
        self.steps = config.pgd_steps
        self.lr = config.pgd_lr
        self.eps = config.fixed_eps

    def forward(
        self,
        mu: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu:  [B, K]  concept predictions (no per-concept variance needed)
            W:   [J, K]  label head weight
            b:   [J]     label head bias
            y:   [B]     true labels
        Returns:
            p_star:  [B, K]
            loss_wc: [B]
        """
        mu_d = mu.detach()
        W_d = W.detach()
        b_d = b.detach()

        p = mu_d.clone()

        # Manual gradient ascent loop (enable gradients for p only)
        with torch.enable_grad():
            for _ in range(self.steps):
                # Forward: logits = p @ W^T + b
                p = p.detach().requires_grad_(True)
                logits = F.linear(p, W_d, b_d)
                loss = F.cross_entropy(logits, y, reduction='none')

                # Gradient ascent on p
                grad_p = torch.autograd.grad(loss.sum(), p, create_graph=False)[0]

                if grad_p is None:
                    grad_p = torch.zeros_like(p)

                # Update p
                p = p.detach() + self.lr * grad_p

                # Project onto L2 ball of radius ε, then clamp [0,1]
                delta = p - mu_d
                norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                delta = torch.where(norm > self.eps, delta * self.eps / norm, delta)
                p = (mu_d + delta).clamp(0.0, 1.0)

        # Final worst-case loss (with model gradients this time)
        logits_wc = F.linear(p, W, b)
        loss_wc = F.cross_entropy(logits_wc, y, reduction='none')

        return p, loss_wc


# =============================================================================
# HUBER Z-SCORE (RQ4 extension)
# =============================================================================

class HuberZScore(nn.Module):
    """
    Huber contamination weighting for OOD robustness (UAI doc §2.5, Exp 5).

    z(x) = ||μ(x) - μ_train|| / σ_train
    δ(x) = max(0, 1 - τ / z(x))

    When z > τ, the DRO loss contribution is downweighted by (1 - δ),
    preventing OOD outliers from distorting the robust objective.
    """

    def __init__(self, tau: float = 2.0):
        super().__init__()
        self.tau = tau
        # Running stats — call update_stats() after first epoch
        self.register_buffer('mu_train', torch.zeros(1))
        self.register_buffer('sigma_train', torch.ones(1))
        self.stats_initialized = False

    def update_stats(self, all_mu: torch.Tensor):
        """
        Compute training set statistics for z-score normalisation.
        Args:
            all_mu: [N, K] concept means over entire training set
        """
        self.mu_train = all_mu.mean(dim=0)          # [K]
        self.sigma_train = all_mu.std(dim=0).clamp(min=1e-6)  # [K]
        self.stats_initialized = True

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: [B, K] concept means for current batch
        Returns:
            weights: [B] in [0, 1], where 1 = fully trusted, 0 = fully OOD
        """
        if not self.stats_initialized:
            return torch.ones(mu.shape[0], device=mu.device)

        z = ((mu - self.mu_train) / self.sigma_train).norm(dim=-1)  # [B]
        delta = torch.clamp(1.0 - self.tau / z.clamp(min=1e-8), min=0.0)  # [B]
        return 1.0 - delta  # [B] weights


# =============================================================================
# UNIFIED CREDAL DRO MODULE
# =============================================================================

class CredalDROModule(nn.Module):
    """
    Full pipeline: latents → concepts → DRO robust loss.

    Supports three modes (UAI doc §5.2):
      POST_HOC:   Train with L_task + L_concept only, extract ε after
      FIXED_EPS:  Train with fixed-ε DRO (no per-instance σ)
      JOINT:      Train with instance-adaptive ε(x) from ensemble σ²

    Forward returns a dict with all loss components and diagnostics.
    """

    def __init__(self, config: CredalDROConfig):
        super().__init__()
        self.config = config

        # Concept ensemble: latents → μ, σ² (unified binary/ternary support)
        self.concept_ensemble = ConceptEnsemble(config)

        # Label head: concepts → class logits (kept linear for convex DRO)
        self.label_head = LabelHead(config.num_concepts, config.num_classes)

        # Aleatoric head: latents → per-concept ambiguity (optional)
        if config.use_aleatoric:
            self.aleatoric_head = AleatoricHead(
                input_dim=config.input_dim,
                num_concepts=config.num_concepts,
                hidden_dim=128,  # Fixed as per specification
            )
            self.lambda_ale = getattr(config, "lambda_ale", 1.0)
            self.use_aleatoric_weighting = getattr(config, "use_aleatoric_weighting", False)
        else:
            self.aleatoric_head = None
            self.lambda_ale = 0.0
            self.use_aleatoric_weighting = False

        # PGD inner max (instance-adaptive for JOINT and POST_HOC)
        self.pgd = PGDInnerMax(config)

        # Fixed-ε PGD for FIXED_EPS mode
        self.pgd_fixed = FixedEpsPGDInnerMax(config)

        # Huber z-score (optional, RQ4)
        if config.use_huber:
            self.huber = HuberZScore(tau=config.huber_tau)
        else:
            self.huber = None

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
        is_unknown: Optional[torch.Tensor] = None,
        concept_entropy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features:         [B, D] latent features from frozen encoder
            labels:           [B]    task labels (long)
            concept_labels:   [B, K] concept supervision (optional, long/float)
            is_unknown:       [B, K] unknown concept mask (optional, for ternary)
            concept_entropy:  [B, K] aleatoric uncertainty targets (optional, float)
        Returns:
            dict with keys:
              loss_total:      scalar, the combined training loss
              loss_task:       scalar, CE on nominal (μ-based) prediction
              loss_concept:    scalar, concept supervision loss (0 if no labels)
              loss_aleatoric:  scalar, aleatoric uncertainty loss (0 if no targets)
              loss_robust:     scalar, DRO worst-case loss
              loss_width:      scalar, width penalty Ω(Σ)
              epsilon:         [B]    instance robustness radii
              mu:              [B, K] concept means
              sigma_sq:        [B, K] concept variances
              aleatoric:       [B, K] predicted aleatoric uncertainty (if enabled)
              p_star:          [B, K] worst-case concept vectors
              logits:          [B, J] nominal logits (for accuracy computation)
        """
        B = features.shape[0]
        cfg = self.config

        # ----- 1. Concept ensemble → μ, σ² -----
        # Unified ensemble returns (mu, sigma_sq, all_probs, all_logits)
        mu, sigma_sq, all_probs, all_logits = self.concept_ensemble(features)

        # ----- 2. Nominal prediction (using ensemble mean) -----
        logits = self.label_head(mu)                         # [B, J]
        loss_task = F.cross_entropy(logits, labels)          # scalar

        # ----- 3. Concept supervision loss -----
        loss_concept = torch.tensor(0.0, device=features.device)
        if concept_labels is not None and cfg.lambda_concept > 0:
            # Check for dummy concepts (single dimension)
            if concept_labels.shape[1] <= 1:
                # Skip concept loss for dummy concepts
                pass
            elif all_logits is not None:
                # Ternary concepts: use CrossEntropyLoss with class weights
                # all_logits is [N, B, K, C], concept_labels is [B, K] in {0,1,2}
                unknown_weight = getattr(cfg, 'unknown_weight', 0.5)
                class_weights = torch.tensor([1.0, unknown_weight, 1.0], device=features.device)
                for h in range(cfg.n_heads):
                    logits_h = all_logits[h]  # [B, K, C]
                    # Reshape for cross_entropy: [B*K, C] vs [B*K]
                    loss_h = F.cross_entropy(
                        logits_h.reshape(-1, logits_h.shape[-1]),
                        concept_labels.reshape(-1).long(),
                        weight=class_weights,
                        reduction='mean',
                    )
                    loss_concept = loss_concept + loss_h
                loss_concept = loss_concept / cfg.n_heads
            else:
                # Binary concepts: use BCE (original behavior)
                for h in range(cfg.n_heads):
                    pred_h = all_probs[h]                        # [B, K]
                    # BCE per concept (treat concept_labels as float targets)
                    target = concept_labels.float()
                    loss_concept = loss_concept + F.binary_cross_entropy(
                        pred_h, target, reduction='mean'
                    )
                loss_concept = loss_concept / cfg.n_heads

        # ----- 4. Aleatoric uncertainty prediction -----
        loss_ale = torch.tensor(0.0, device=features.device)
        a_hat = None

        if self.aleatoric_head is not None and concept_entropy is not None and self.lambda_ale > 0:
            # Predict aleatoric uncertainty from features
            a_hat = self.aleatoric_head(features)  # [B, K] in (0, 1)

            # MSE loss against normalized entropy from annotators
            # concept_entropy assumed in [0,1], shape [B,K]
            loss_ale = F.mse_loss(a_hat, concept_entropy, reduction="mean")

        # ----- 5. Instance robustness radius ε(x) -----
        epsilon = CredalEllipsoid.epsilon(sigma_sq)          # [B]

        # ----- 6. DRO robust loss -----
        loss_robust = torch.tensor(0.0, device=features.device)
        p_star = mu.detach().clone()  # default: no adversarial perturbation

        effective_lambda = cfg.get_effective_lambda_dro()

        if effective_lambda > 0:
            W = self.label_head.W
            b_vec = self.label_head.b

            if cfg.mode == DROMode.FIXED_EPS:
                # (B) Fixed-ε: uniform ball, no per-instance σ
                p_star, loss_wc = self.pgd_fixed(mu, W, b_vec, labels)

            else:
                # (C) Joint: instance-adaptive ellipsoid
                # Optionally stop gradient through σ²
                sigma_for_pgd = sigma_sq
                if cfg.get_stop_grad_sigma():
                    sigma_for_pgd = sigma_sq.detach()

                p_star, loss_wc = self.pgd(
                    mu, sigma_for_pgd, W, b_vec, labels
                )

            # Huber weighting (optional)
            if self.huber is not None:
                huber_weights = self.huber(mu)               # [B]
                loss_robust = (loss_wc * huber_weights).mean()
            else:
                loss_robust = loss_wc.mean()

        # ----- 7. Width penalty -----
        loss_width = torch.tensor(0.0, device=features.device)
        if cfg.beta_width > 0 and cfg.width_penalty != "none":
            if cfg.width_penalty == "log_det":
                loss_width = CredalEllipsoid.width_penalty_log_det(sigma_sq).mean()
            elif cfg.width_penalty == "trace":
                loss_width = CredalEllipsoid.width_penalty_trace(sigma_sq).mean()

        # ----- 8. Total loss -----
        loss_total = (
            loss_task
            + cfg.lambda_concept * loss_concept
            + effective_lambda * loss_robust
            + cfg.beta_width * loss_width
            + self.lambda_ale * loss_ale
        )

        # Prepare return dict
        output_dict = {
            'loss_total': loss_total,
            'loss_task': loss_task.detach(),
            'loss_concept': loss_concept.detach(),
            'loss_robust': loss_robust.detach() if isinstance(loss_robust, torch.Tensor) else loss_robust,
            'loss_width': loss_width.detach() if isinstance(loss_width, torch.Tensor) else loss_width,
            'epsilon': epsilon.detach(),
            'mu': mu.detach(),
            'sigma_sq': sigma_sq.detach(),
            'p_star': p_star.detach(),
            'logits': logits.detach(),
        }

        # Add aleatoric outputs if available
        if a_hat is not None:
            output_dict['a_hat'] = a_hat.detach()
            output_dict['loss_ale'] = loss_ale.detach()
        else:
            output_dict['a_hat'] = torch.zeros(1)  # Placeholder
            output_dict['loss_ale'] = torch.tensor(0.0)

        return output_dict


# =============================================================================
# THREE-WAY COMPARISON FACTORY
# =============================================================================

def get_three_way_configs(
    num_concepts: int = 4,
    num_classes: int = 3,
    input_dim: int = 128,
) -> Dict[str, CredalDROConfig]:
    """
    Return configs for the three-way comparison (UAI doc §5.2).

    (A) post_hoc:  λ_dro=0        — ELBO only, extract ε after convergence
    (B) fixed_eps: stop_grad σ²    — same ε for all inputs during training
    (C) joint:     full gradients  — ε(x) learned jointly
    """
    base = dict(
        num_concepts=num_concepts,
        num_classes=num_classes,
        input_dim=input_dim,
    )

    return {
        "post_hoc": CredalDROConfig(
            **base,
            mode=DROMode.POST_HOC,
            lambda_dro=0.0,
            beta_width=0.0,
        ),
        "fixed_eps": CredalDROConfig(
            **base,
            mode=DROMode.FIXED_EPS,
            lambda_dro=0.1,
            fixed_eps=0.1,
        ),
        "joint": CredalDROConfig(
            **base,
            mode=DROMode.JOINT,
            lambda_dro=0.1,
            beta_width=0.01,
        ),
    }


# =============================================================================
# QUICK SMOKE TEST
# =============================================================================

def run_smoke_test():
    """Run smoke test for the three DRO modes."""
    print("=" * 70)
    print("CREDAL DRO MODULE — SMOKE TEST")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, D, K, J = 16, 128, 4, 3

    configs = get_three_way_configs(
        num_concepts=K, num_classes=J, input_dim=D
    )

    features = torch.randn(B, D, device=device)
    labels = torch.randint(0, J, (B,), device=device)
    concepts = torch.randint(0, 3, (B, K), device=device)  # ternary

    for name, cfg in configs.items():
        print(f"\n--- Mode: {name} ({cfg.mode.value}) ---")
        model = CredalDROModule(cfg).to(device)
        out = model(features, labels, concepts)

        print(f"  loss_total:   {out['loss_total'].item():.4f}")
        print(f"  loss_task:    {out['loss_task'].item():.4f}")
        print(f"  loss_concept: {out['loss_concept'].item():.4f}")
        print(f"  loss_robust:  {out['loss_robust'].item():.4f}")
        print(f"  loss_width:   {out['loss_width'].item():.4f}")
        print(f"  ε mean:       {out['epsilon'].mean().item():.4f}")
        print(f"  ε std:        {out['epsilon'].std().item():.4f}")
        print(f"  σ² mean:      {out['sigma_sq'].mean().item():.6f}")

    print("\n✅ All three modes pass.")


# =============================================================================
# UAI 2026 PRIORITY UPGRADES
# =============================================================================
"""
Credal DRO — Four Priority Upgrades for UAI 2026
==================================================

UPGRADE 1: Ternary Concept Heads (already implemented above)
UPGRADE 2: Diversity Loss in Forward Pass
UPGRADE 3: ε-Calibration Metrics (headline evaluation)
UPGRADE 4: Wasserstein Identity Verification (Theorem 1 check)
"""

import numpy as np
from scipy.stats import spearmanr


# =============================================================================
# UPGRADE 2: DIVERSITY LOSS
# =============================================================================

def compute_diversity_loss(all_probs: torch.Tensor, mode: str = "variance") -> torch.Tensor:
    """
    Diversity loss to encourage head disagreement in the ensemble.

    Args:
        all_probs: [H, B, K] per-head concept probabilities
        mode: One of "variance", "cosine", "det_kernel"

    Returns:
        Scalar loss (minimize to maximize diversity)
    """
    H, B, K = all_probs.shape
    if mode == "variance":
        # Maximize variance across heads
        return -all_probs.var(dim=0).mean()
    elif mode == "cosine":
        # Minimize cosine similarity between head predictions
        flat = F.normalize(all_probs.reshape(H, B * K), dim=1)
        sim = flat @ flat.T
        mask = ~torch.eye(H, dtype=torch.bool, device=sim.device)
        return sim[mask].mean()
    elif mode == "det_kernel":
        # Maximize determinant of kernel matrix (diversity in RKHS)
        flat = all_probs.reshape(H, B * K)
        kernel = flat @ flat.T / (B * K) + 1e-4 * torch.eye(H, device=flat.device)
        return -torch.logdet(kernel)
    return torch.tensor(0.0, device=all_probs.device)


# =============================================================================
# UPGRADE 3: ε-CALIBRATION METRICS
# =============================================================================

def epsilon_error_calibration(epsilon: np.ndarray, is_error: np.ndarray,
                              n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    Compute ε-Error Expected Calibration Error.

    Higher ε should predict higher error rate. ECE measures the gap.

    Args:
        epsilon: [N] instance robustness radii
        is_error: [N] binary error indicators (0 or 1)
        n_bins: number of calibration bins

    Returns:
        ece: scalar ECE value (lower is better calibrated)
        bins_data: per-bin statistics for plotting
    """
    N = len(epsilon)
    bin_edges = np.quantile(epsilon, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8  # Include max value
    eps_min, eps_max = epsilon.min(), epsilon.max()
    eps_norm = (epsilon - eps_min) / (eps_max - eps_min + 1e-10)

    ece = 0.0
    bins_data = []
    for i in range(n_bins):
        mask = (epsilon >= bin_edges[i]) & (epsilon < bin_edges[i + 1])
        n_b = mask.sum()
        if n_b == 0:
            continue
        avg_eps = eps_norm[mask].mean()
        avg_err = is_error[mask].mean()
        ece += (n_b / N) * abs(avg_eps - avg_err)
        bins_data.append({
            'bin': i,
            'n_samples': int(n_b),
            'avg_eps_raw': float(epsilon[mask].mean()),
            'avg_eps_norm': float(avg_eps),
            'avg_error_rate': float(avg_err),
        })
    return float(ece), bins_data


def per_concept_calibration(sigma_sq: np.ndarray, concept_labels: np.ndarray,
                            mu: np.ndarray, concept_names: Optional[List[str]] = None,
                            n_bins: int = 10) -> Dict:
    """
    Check if σ²_k predicts per-concept error (aleatoric uncertainty calibration).

    Args:
        sigma_sq: [N, K] per-concept epistemic variances
        concept_labels: [N, K] ground truth concept labels (0, 1, 2 for ternary)
        mu: [N, K] predicted concept means
        concept_names: optional list of concept names
        n_bins: number of calibration bins

    Returns:
        Dict with per-concept Spearman correlations and ECE
    """
    N, K = sigma_sq.shape
    # Normalize ternary labels to [0, 1] range for error computation
    if concept_labels.max() > 1:
        target = concept_labels / 2.0
    else:
        target = concept_labels.astype(float)
    concept_error = np.abs(mu - target)

    results = {'per_concept': [], 'mean_rho': 0.0, 'mean_ece': 0.0}
    for k in range(K):
        name = concept_names[k] if concept_names else f"concept_{k}"
        rho, p_val = spearmanr(sigma_sq[:, k], concept_error[:, k])
        ece, _ = epsilon_error_calibration(
            sigma_sq[:, k], (concept_error[:, k] > 0.25).astype(float), n_bins
        )
        results['per_concept'].append({
            'name': name,
            'rho': float(rho),
            'p': float(p_val),
            'ece': float(ece),
            'sigma_mean': float(sigma_sq[:, k].mean()),
            'error_mean': float(concept_error[:, k].mean()),
        })
    results['mean_rho'] = float(np.mean([c['rho'] for c in results['per_concept']]))
    results['mean_ece'] = float(np.mean([c['ece'] for c in results['per_concept']]))
    return results


# =============================================================================
# UPGRADE 4: WASSERSTEIN IDENTITY VERIFICATION
# =============================================================================

def verify_wasserstein_identity(all_probs: torch.Tensor, mu: torch.Tensor,
                                 sigma_sq: torch.Tensor) -> Dict[str, float]:
    """
    Verify Theorem 1: W₂²(P̂^H, δ_μ) = Tr(Σ_epi).

    The empirical Wasserstein distance from ensemble to mean should equal
    the trace of the epistemic covariance matrix.

    Note: torch.var() uses Bessel correction (H-1), while the true W₂² uses H.
    So we expect: W₂² / ((H-1)/H · Tr(Σ)) ≈ 1.0

    Args:
        all_probs: [H, N, K] per-head concept probabilities
        mu: [N, K] ensemble mean
        sigma_sq: [N, K] ensemble variance (from torch.var)

    Returns:
        Dict with verification metrics
    """
    H, N, K = all_probs.shape
    deviations = all_probs - mu.unsqueeze(0)  # [H, N, K]

    # Empirical W₂²: mean squared deviation from mean
    w2_sq = (deviations ** 2).sum(dim=-1).mean(dim=0)  # [N]

    # Tr(Σ_epi) from variance
    tr_sigma = sigma_sq.sum(dim=-1)  # [N]

    # Bessel correction factor
    bessel = (H - 1) / H

    # Ratio should be ≈ 1.0
    ratio = w2_sq / (bessel * tr_sigma + 1e-10)

    return {
        'ratio_mean': float(ratio.mean().item()),
        'ratio_std': float(ratio.std().item()),
        'w2_sq_mean': float(w2_sq.mean().item()),
        'tr_sigma_mean': float(tr_sigma.mean().item()),
        'bessel_factor': float(bessel),
        'paper_sentence': (
            f"Theorem 1 verified: W₂²/((H-1)/H · Tr(Σ)) = "
            f"{ratio.mean().item():.6f} ± {ratio.std().item():.6f} "
            f"(N={N}, H={H})"
        ),
    }


# =============================================================================
# UAI 2026 UPGRADES SMOKE TEST
# =============================================================================

def run_uai_upgrades_smoke_test():
    """Run smoke test for all UAI 2026 upgrades."""
    print("=" * 70)
    print("CREDAL UAI 2026 UPGRADES — SMOKE TEST")
    print("=" * 70)

    B, D, K, J, H = 128, 64, 4, 3, 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Upgrade 2: Diversity Loss
    print("\n--- Upgrade 2: Diversity Loss ---")
    all_probs = torch.rand(H, B, K, device=device)
    for mode in ["variance", "cosine", "det_kernel"]:
        div = compute_diversity_loss(all_probs, mode=mode)
        print(f"  {mode}: {div.item():.4f}")
    print("  ✅ OK")

    # Upgrade 3: ε-Calibration
    print("\n--- Upgrade 3: ε-Error Calibration ---")
    eps_np = np.random.exponential(0.3, size=200)
    err_prob = 1.0 / (1.0 + np.exp(-3 * (eps_np - eps_np.mean())))
    is_error = (np.random.rand(200) < err_prob).astype(float)
    ece, bins = epsilon_error_calibration(eps_np, is_error)
    print(f"  ε-Error ECE: {ece:.4f}, bins: {len(bins)}")

    sigma_sq_np = np.random.exponential(0.1, size=(200, K))
    concept_labels_np = np.random.randint(0, 3, size=(200, K))
    mu_np = np.random.rand(200, K)
    cal = per_concept_calibration(sigma_sq_np, concept_labels_np, mu_np,
                                   concept_names=["food", "ambiance", "service", "noise"])
    print(f"  Per-concept mean ρ: {cal['mean_rho']:.3f}")
    print("  ✅ OK")

    # Upgrade 4: Wasserstein Identity
    print("\n--- Upgrade 4: Wasserstein Identity ---")
    all_probs_t = torch.rand(H, 200, K, device=device)
    mu_t = all_probs_t.mean(dim=0)
    sigma_sq_t = all_probs_t.var(dim=0)
    w2 = verify_wasserstein_identity(all_probs_t, mu_t, sigma_sq_t)
    print(f"  {w2['paper_sentence']}")
    assert abs(w2['ratio_mean'] - 1.0) < 1e-4, f"Got {w2['ratio_mean']}"
    print("  ✅ OK")

    print("\n" + "=" * 70)
    print("ALL UAI 2026 UPGRADES PASS ✅")
    print("=" * 70)


if __name__ == "__main__":
    # Run original smoke test
    run_smoke_test()

    # Run UAI 2026 upgrades smoke test
    run_uai_upgrades_smoke_test()
