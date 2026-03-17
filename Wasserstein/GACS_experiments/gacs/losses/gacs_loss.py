"""
GACS Loss Functions — Implementing the Three Desiderata
========================================================

D1: Scale Invariance  → L2-normalized reconstruction
D2: Smooth Convergence → Sigmoid KL annealing
D3: Geometric Separability → Concept sparsity + diversity via eigenvalue regularization
"""

import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np


class GACSLoss:
    """
    Combined loss for GACS training.

    Total = w_recon * L_recon(D1)
          + β(t) * L_KL(D2)
          + w_sparse * L_sparse(D3)
          + w_diverse * L_diverse(D3)
          + w_concept * L_concept_supervised  (when labels available)
          + w_class * L_classification
    """

    def __init__(self, config):
        self.config = config
        self.lc = config.loss
        self.eps = 1e-8

    # -------------------------------------------------------------------
    # D2: Sigmoid KL Annealing Schedule
    # -------------------------------------------------------------------
    def get_kl_weight(self, epoch: int) -> float:
        """
        β(t) = σ(s · (t/T_w − 0.5)) · β_max

        Sigmoid annealing ensures vanishing derivative at convergence,
        yielding stable loss landscape geometry for Hessian characterization.
        """
        if self.lc.kl_annealing == "constant":
            return self.lc.kl_weight_max

        progress = epoch / max(self.lc.kl_warmup_epochs, 1)

        if self.lc.kl_annealing == "sigmoid":
            beta = 1.0 / (1.0 + np.exp(-self.lc.kl_sigmoid_steepness * (progress - 0.5)))
        elif self.lc.kl_annealing == "linear":
            beta = min(progress, 1.0)
        else:
            beta = 1.0

        return beta * self.lc.kl_weight_max

    # -------------------------------------------------------------------
    # D1: Scale-Invariant Reconstruction Loss
    # -------------------------------------------------------------------
    def compute_recon_loss(self, h: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        L_recon on L2-normalized inputs/outputs.

        Normalizing prevents Hessian eigenspectrum from reflecting
        input magnitude artifacts — critical for transfer across domains.
        """
        if self.lc.recon_normalize:
            h_norm = F.normalize(h.detach(), p=2, dim=-1)  # detach: don't backprop through encoder for recon
            recon_norm = F.normalize(recon, p=2, dim=-1)
            loss = F.mse_loss(recon_norm, h_norm)
        else:
            loss = F.mse_loss(recon, h.detach())

        return torch.clamp(loss, max=100.0)

    # -------------------------------------------------------------------
    # KL Divergence
    # -------------------------------------------------------------------
    def compute_kl_loss(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Standard KL(q(z|x) || N(0,I)) with clamping for stability."""
        z_logvar = torch.clamp(z_logvar, min=-20.0, max=2.0)
        z_mu = torch.clamp(z_mu, min=-20.0, max=20.0)

        kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)
        return kl.mean()

    # -------------------------------------------------------------------
    # D3: Concept Sparsity Loss
    # -------------------------------------------------------------------
    def compute_sparsity_loss(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        L_sparse = E[|c|] + Σ √λ_i  (nuclear norm on concept covariance)

        Encourages sparse concept activations and penalizes redundant
        concept directions, ensuring flat parameter directions correspond
        to genuine epistemic degeneracy.
        """
        # L1 sparsity on concept activations
        l1_loss = torch.mean(torch.abs(concepts))

        # Nuclear norm on concept covariance (via eigenvalues)
        if concepts.size(0) > 1:
            concepts_centered = concepts - concepts.mean(dim=0, keepdim=True)
            # Covariance matrix (concept_dim × concept_dim)
            cov = (concepts_centered.T @ concepts_centered) / (concepts.size(0) - 1)
            cov = cov + torch.eye(cov.size(0), device=cov.device) * self.eps

            # Eigenvalues of covariance
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=self.eps)

            nuclear_norm = torch.sum(torch.sqrt(eigenvalues))
        else:
            nuclear_norm = torch.tensor(0.0, device=concepts.device)

        return l1_loss + 0.1 * nuclear_norm

    # -------------------------------------------------------------------
    # D3: Concept Diversity Loss
    # -------------------------------------------------------------------
    def compute_diversity_loss(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Eigenvalue-weighted pairwise similarity penalty.

        Prevents correlated concepts from creating spurious flat directions
        in the loss landscape.
        """
        if concepts.size(1) <= 1 or concepts.size(0) <= 1:
            return torch.tensor(0.0, device=concepts.device)

        # Normalize concepts for cosine similarity
        concepts_norm = F.normalize(concepts, p=2, dim=0)  # normalize across batch

        # Concept correlation matrix (concept_dim × concept_dim)
        corr = concepts_norm.T @ concepts_norm / concepts.size(0)

        # Penalize off-diagonal elements (want orthogonal concepts)
        eye = torch.eye(corr.size(0), device=corr.device)
        diversity_loss = ((corr - eye) ** 2).mean()

        return diversity_loss

    # -------------------------------------------------------------------
    # Supervised Concept Loss (CEBaB)
    # -------------------------------------------------------------------
    def compute_concept_supervision_loss(
        self, concepts: torch.Tensor, concept_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE between predicted concepts and ground-truth concept labels.
        Only used when concept labels are available (e.g., CEBaB).
        """
        # Apply sigmoid to concept logits for comparison with [0,1] labels
        concept_probs = torch.sigmoid(concepts)
        return F.mse_loss(concept_probs, concept_labels)

    # -------------------------------------------------------------------
    # Classification Loss
    # -------------------------------------------------------------------
    def compute_classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    # -------------------------------------------------------------------
    # Combined Loss
    # -------------------------------------------------------------------
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with all components.

        Returns dict with 'total_loss' (for backprop) and individual components (for logging).
        """
        # Individual losses
        recon_loss = self.compute_recon_loss(outputs["h"], outputs["recon"])
        kl_loss = self.compute_kl_loss(outputs["z_mu"], outputs["z_logvar"])
        sparsity_loss = self.compute_sparsity_loss(outputs["concepts"])
        diversity_loss = self.compute_diversity_loss(outputs["concepts"])
        classification_loss = self.compute_classification_loss(
            outputs["logits"], batch["labels"]
        )

        # Concept supervision (only when labels are meaningful)
        has_concept_labels = batch["concept_labels"].abs().sum() > 0
        if has_concept_labels and self.lc.concept_supervision_weight > 0:
            concept_sup_loss = self.compute_concept_supervision_loss(
                outputs["concepts"], batch["concept_labels"]
            )
        else:
            concept_sup_loss = torch.tensor(0.0, device=outputs["logits"].device)

        # KL weight schedule
        kl_weight = self.get_kl_weight(epoch)

        # Total loss
        total = (
            self.lc.recon_weight * recon_loss
            + kl_weight * kl_loss
            + self.lc.concept_sparsity_weight * sparsity_loss
            + self.lc.concept_diversity_weight * diversity_loss
            + self.lc.concept_supervision_weight * concept_sup_loss
            + self.lc.classification_weight * classification_loss
        )

        if self.lc.max_loss_value > 0:
            total = torch.clamp(total, max=self.lc.max_loss_value)

        return {
            "total_loss": total,
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "concept_sup_loss": concept_sup_loss.item(),
            "classification_loss": classification_loss.item(),
            "kl_weight": kl_weight,
        }
