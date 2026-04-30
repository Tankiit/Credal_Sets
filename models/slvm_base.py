"""
Shared base class for supervised latent variable models (SLVMs).

An SLVM is any architecture with an intermediate layer of individually
supervised units. This base class factors out the common uncertainty
machinery (three-head credal decomposition, aleatoric head, task
classifier, loss computation) and leaves the architecture-specific
"body" as a swappable component.

Concrete bodies:
- CBMBody: MLP producing concept logits directly (standard CBM)
- SENNBody: relevance-weighted classifier with stability regulariser
- ProtoPBody: prototype-similarity layer with attribute supervision
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================================
# Config
# ============================================================================
@dataclass
class SLVMConfig:
    """Base config for any SLVM variant (CBM / SENN / ProtoP)."""
    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = True
    pooling_strategy: str = "cls"

    num_concepts: int = 4
    concept_names: List[str] = field(default_factory=list)
    num_classes: int = 2

    # Credal set parameters
    num_mc_samples: int = 10
    min_sigma: float = 0.01
    max_sigma: float = 2.0
    prior_sigma: float = 0.5

    # Architecture dimensions
    projection_dim: int = 256
    hidden_dim: int = 128

    # Loss weights
    concept_weight: float = 2.0
    kl_weight: float = 0.01
    error_supervision_weight: float = 1.0
    aleatoric_weight: float = 2.0
    aleatoric_unknown_weight: float = 0.0    # Path A default: off
    orth_weight: float = 0.001

    # AU head prior (Path A fix)
    aleatoric_prior: float = 0.05

    # Error scaling for σ_epi supervision
    error_scale: float = 2.0

    # Architecture type — set by subclass config
    model_type: str = "base"  # "cbm", "senn", "protop"


# ============================================================================
# Shared components (reused from existing HybridCredalCBM code)
# ============================================================================

class ThreeWayOrthogonalProjection(nn.Module):
    """Projects encoder hidden to three orthogonal subspaces."""

    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.W_concept = nn.Linear(hidden_size, proj_dim, bias=False)
        self.W_epi = nn.Linear(hidden_size, proj_dim, bias=False)
        self.W_ale = nn.Linear(hidden_size, proj_dim, bias=False)

        total_dim = 3 * proj_dim
        if total_dim <= hidden_size:
            full = torch.empty(total_dim, hidden_size)
            nn.init.orthogonal_(full)
            self.W_concept.weight.data = full[:proj_dim]
            self.W_epi.weight.data = full[proj_dim:2*proj_dim]
            self.W_ale.weight.data = full[2*proj_dim:]
        else:
            for W in [self.W_concept, self.W_epi, self.W_ale]:
                nn.init.orthogonal_(W.weight)

    def forward(self, hidden):
        return self.W_concept(hidden), self.W_epi(hidden), self.W_ale(hidden)

    def orthogonality_loss(self):
        W_c, W_e, W_a = self.W_concept.weight, self.W_epi.weight, self.W_ale.weight
        return (
            torch.norm(W_c @ W_e.T, p='fro') ** 2
            + torch.norm(W_c @ W_a.T, p='fro') ** 2
            + torch.norm(W_e @ W_a.T, p='fro') ** 2
        )


class EpistemicSigmaHead(nn.Module):
    """σ_epi head: error-supervised + KL-regularised credal set size."""

    def __init__(self, proj_dim, num_concepts, hidden_dim=128,
                 prior_sigma=0.5, min_sigma=0.01, max_sigma=2.0, error_scale=2.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.error_scale = error_scale

        self.log_sigma_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
        )

        # Initialise σ near prior
        init_val = math.log(math.exp(prior_sigma) - 1 + 1e-6)
        last = self.log_sigma_net[-1]
        nn.init.constant_(last.bias, init_val)
        nn.init.normal_(last.weight, std=0.01)

    def forward(self, h_epi):
        raw = self.log_sigma_net(h_epi)
        sigma = F.softplus(raw)
        return torch.clamp(sigma, self.min_sigma, self.max_sigma)

    def error_supervision_loss(self, sigma_epi, concept_preds, concept_targets):
        errors = torch.abs(concept_preds.detach() - concept_targets)
        scaled = errors * self.error_scale + self.min_sigma
        scaled = torch.clamp(scaled, self.min_sigma, self.max_sigma)
        return F.mse_loss(sigma_epi, scaled)

    def kl_divergence(self, sigma_epi):
        prior = self.prior_sigma
        kl = (torch.log(prior / (sigma_epi + 1e-10))
              + (sigma_epi ** 2) / (2 * prior ** 2) - 0.5)
        return kl.mean()


class AleatoricHead(nn.Module):
    """σ_ale head: supervised by annotator entropy (Path A)."""

    def __init__(self, proj_dim, num_concepts, hidden_dim=128, prior_mean=0.05):
        super().__init__()
        self.num_concepts = num_concepts

        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_concepts),
        )

        import numpy as np
        prior_logit = np.log(max(prior_mean, 1e-6) / (1 - prior_mean + 1e-6))
        self.log_prior = nn.Parameter(torch.ones(num_concepts) * prior_logit)

    def forward(self, h_ale):
        logits = self.net(h_ale) + self.log_prior
        return torch.sigmoid(torch.clamp(logits, -10, 10))


# ============================================================================
# Body interface — what concrete architectures implement
# ============================================================================

class SLVMBody(nn.Module):
    """
    Abstract interface for an SLVM body.

    A body consumes h_concept (from the orthogonal projection) and
    produces:
      - concept_scores: [batch, num_concepts] in ~[0, 1] after sigmoid
      - mu: [batch, num_concepts] — the "logit-space" version used by
        the credal set sampling (concept_scores = sigmoid(mu))
      - task_logits: [batch, num_classes] — the downstream task prediction
      - body_aux_loss: optional extra loss term (e.g. SENN stability regulariser)

    CBMBody:   mu = MLP(h_concept);  concept_scores = sigmoid(mu);
               task_logits = linear(concept_scores)
    SENNBody:  mu = MLP(h_concept);  concept_scores = sigmoid(mu);
               relevance = relevance_net(h_concept);
               task_logits = relevance @ concept_scores (batched)
               body_aux_loss = stability regulariser on relevance
    ProtoPBody: concept_scores = similarity(h_concept, prototypes);
                mu = logit(concept_scores);
                task_logits = linear(concept_scores);
                supervision: concept_scores should match attribute labels
    """

    def forward(self, h_concept: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute_aux_loss(self, **kwargs) -> Optional[torch.Tensor]:
        """Optional body-specific regulariser (e.g. SENN stability)."""
        return None


# ============================================================================
# Generic SLVM shell that wraps any body
# ============================================================================

class HybridCredalSLVM(nn.Module):
    """
    Architecture-agnostic SLVM shell.

    Takes a body component (CBM / SENN / ProtoP) and attaches the three-head
    uncertainty machinery. The body determines how concept scores are computed
    from h_concept; the shell determines how uncertainty flows through the
    three orthogonal projections.
    """

    def __init__(self, config: SLVMConfig, body: SLVMBody):
        super().__init__()
        self.config = config
        self.body = body

        # Encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        if config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Three orthogonal projections
        self.projection = ThreeWayOrthogonalProjection(self.hidden_size, config.projection_dim)

        # μ head: the center of the credal set
        # For CBM and SENN, this is a plain MLP. For ProtoP, the "μ" is
        # computed inside the body from prototype similarities. See note
        # in _forward_mu below.
        self.mu_net = nn.Sequential(
            nn.Linear(config.projection_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_concepts),
        )

        # σ_epi head: credal set size
        self.sigma_epi_head = EpistemicSigmaHead(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            prior_sigma=config.prior_sigma,
            min_sigma=config.min_sigma,
            max_sigma=config.max_sigma,
            error_scale=config.error_scale,
        )

        # σ_ale head: annotator entropy supervision
        self.aleatoric_head = AleatoricHead(
            proj_dim=config.projection_dim,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            prior_mean=config.aleatoric_prior,
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.config.pooling_strategy == "cls":
            return outputs.last_hidden_state[:, 0, :]
        h = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids, attention_mask,
        labels=None, concept_labels=None, annotator_entropy=None,
    ) -> Dict[str, torch.Tensor]:
        # Encode
        hidden = self.encode(input_ids, attention_mask)

        # Three orthogonal projections
        h_concept, h_epi, h_ale = self.projection(hidden)

        # --- body produces concept_scores + task_logits from h_concept ---
        body_out = self.body(h_concept, input_ids=input_ids, attention_mask=attention_mask)
        concept_scores = body_out["concept_scores"]   # [batch, K] in [0,1]
        task_logits = body_out["task_logits"]         # [batch, num_classes]

        # μ is the logit-space version. For CBMBody / SENNBody this is
        # computed in the body directly; for ProtoPBody it's derived from
        # concept_scores (since scores are directly computed from similarity).
        mu = body_out.get("mu", None)
        if mu is None:
            # Body didn't provide μ; derive from concept_scores
            mu = torch.logit(concept_scores.clamp(1e-6, 1 - 1e-6))

        # σ_epi from the epistemic projection
        sigma_epi = self.sigma_epi_head(h_epi)

        # σ_ale from the aleatoric projection
        sigma_ale = self.aleatoric_head(h_ale)

        # Epistemic uncertainty — derived from credal set size
        epistemic = torch.log(sigma_epi + 1e-10)

        result = {
            "predictions": task_logits.argmax(dim=-1),
            "logits": task_logits,
            "probs": F.softmax(task_logits, dim=-1),
            "concept_probs": concept_scores,
            "mu": mu,
            "sigma_epi": sigma_epi,
            "aleatoric": sigma_ale,
            "epistemic": epistemic,
        }

        # Losses
        if labels is not None or concept_labels is not None:
            losses = self._compute_losses(
                result, labels, concept_labels, annotator_entropy, body_out
            )
            result.update(losses)

        return result

    def _compute_losses(self, result, labels, concept_labels, annotator_entropy, body_out):
        losses = {}
        device = result["predictions"].device

        # Task loss
        if labels is not None:
            losses["task_loss"] = F.cross_entropy(result["logits"], labels)
            losses["ce_loss"] = losses["task_loss"]

        # Concept supervision + error supervision for σ_epi
        if concept_labels is not None:
            known_mask = (concept_labels != 1)
            if known_mask.any():
                known_labels = concept_labels[known_mask]
                known_preds = result["concept_probs"][known_mask]
                known_preds_clamped = torch.clamp(known_preds, 1e-7, 1 - 1e-7)
                known_targets = (known_labels.float() / 2.0)

                losses["concept_bce"] = F.binary_cross_entropy(known_preds_clamped, known_targets)
                losses["concept_loss"] = losses["concept_bce"]

                known_sigma = result["sigma_epi"][known_mask]
                losses["error_supervision"] = self.sigma_epi_head.error_supervision_loss(
                    known_sigma, known_preds, known_targets
                )

        # KL regularisation for σ_epi
        losses["credal_kl"] = self.sigma_epi_head.kl_divergence(result["sigma_epi"])
        losses["kl_loss"] = losses["credal_kl"]

        # σ_ale supervision (H-supervision, Path A)
        if annotator_entropy is not None and concept_labels is not None:
            known_mask = (concept_labels != 1)
            if known_mask.any():
                if annotator_entropy.dim() == 1 and concept_labels.dim() == 2:
                    annotator_entropy = annotator_entropy.unsqueeze(-1).expand_as(concept_labels)
                elif annotator_entropy.shape != concept_labels.shape:
                    annotator_entropy = annotator_entropy.reshape_as(concept_labels)
                targets = annotator_entropy[known_mask]
                preds = result["aleatoric"][known_mask]
                losses["aleatoric_loss"] = F.mse_loss(preds, targets)

        # U-supervision (ablation only)
        if self.config.aleatoric_unknown_weight > 0 and concept_labels is not None:
            unknown_mask = (concept_labels == 1)
            if unknown_mask.any():
                losses["aleatoric_unknown"] = F.mse_loss(
                    result["aleatoric"][unknown_mask],
                    torch.ones_like(result["aleatoric"][unknown_mask]),
                )

        # Body-specific aux loss (SENN stability regulariser, etc.)
        aux = body_out.get("aux_loss")
        if aux is not None:
            losses["body_aux"] = aux

        # Orthogonality
        losses["orth_penalty"] = self.projection.orthogonality_loss()

        # Combine
        total = torch.tensor(0.0, device=device)
        if "task_loss" in losses:
            total = total + losses["task_loss"]
        if "concept_bce" in losses:
            total = total + self.config.concept_weight * losses["concept_bce"]
        if "error_supervision" in losses:
            total = total + self.config.error_supervision_weight * losses["error_supervision"]
        total = total + self.config.kl_weight * losses["credal_kl"]
        if "aleatoric_loss" in losses:
            total = total + self.config.aleatoric_weight * losses["aleatoric_loss"]
        if "aleatoric_unknown" in losses:
            total = total + self.config.aleatoric_unknown_weight * losses["aleatoric_unknown"]
        if "body_aux" in losses:
            total = total + losses["body_aux"]  # body decides its own weight
        total = total + self.config.orth_weight * losses["orth_penalty"]

        losses["loss"] = total
        return losses
