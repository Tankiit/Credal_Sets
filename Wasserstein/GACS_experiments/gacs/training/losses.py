"""
GACS Loss Functions

Implements the training objective satisfying three desiderata:
  D1 (Scale Invariance): Normalized reconstruction loss
  D2 (Smooth Convergence): Sigmoid KL annealing
  D3 (Geometric Separability): Eigenvalue-aware concept sparsity + diversity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np


class GACSLoss:
    """
    Combined loss for training the stochastic concept bottleneck model.
    
    L_total = L_recon + β(t)·L_KL + L_concept + L_classify + L_sparse + L_diverse
    """
    
    def __init__(self, config):
        self.config = config.loss
        self.eps = 1e-8
    
    # ----- Desideratum 1: Scale-invariant reconstruction -----
    
    def reconstruction_loss(
        self,
        h_original: torch.Tensor,
        h_recon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive reconstruction loss on CLS embeddings.
        Normalized to satisfy D1 (scale invariance).
        
        We use cosine similarity loss rather than raw MSE to ensure
        the Hessian eigenspectrum reflects structural properties
        of the encoding, not input scale variations.
        """
        # L2-normalize both
        h_orig_norm = F.normalize(h_original, p=2, dim=-1)
        h_recon_norm = F.normalize(h_recon, p=2, dim=-1)
        
        # Cosine similarity loss: 1 - cos_sim
        cos_sim = (h_orig_norm * h_recon_norm).sum(dim=-1)  # [B]
        loss = (1.0 - cos_sim).mean()
        
        return loss
    
    # ----- Desideratum 2: Smooth convergence via sigmoid annealing -----
    
    def kl_weight(self, epoch: int) -> float:
        """
        Sigmoid KL annealing schedule.
        
        β(t) = σ(10·(t/T_w - 0.5)) · β_max
        
        Has vanishing derivative at both ends, ensuring the loss landscape
        at convergence is approximately stationary w.r.t. the annealing
        parameter (Desideratum 2).
        """
        warmup = self.config.kl_warmup_epochs
        if warmup <= 0:
            return self.config.kl_weight_max
        
        progress = min(epoch / warmup, 1.0)
        # Sigmoid centered at 0.5 with steepness 10
        beta = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        return beta * self.config.kl_weight_max
    
    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence: KL(q(z|x) || p(z)), where p(z) = N(0, I).
        
        Clamped for numerical stability.
        """
        # Clamp for stability
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        mu = torch.clamp(mu, min=-20.0, max=20.0)
        
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    # ----- Concept supervision -----
    
    def concept_loss(
        self,
        concept_logits: torch.Tensor,
        concept_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for concept prediction.
        
        concept_logits: [B, K] raw logits
        concept_targets: [B, K] integer labels (0=neg, 1=unknown, 2=pos)
        
        We treat each concept as an independent 3-class classification
        (negative, neutral/unknown, positive for CEBaB).
        However, concept_logits from our model are single scalars per concept,
        so we convert to binary: concept present (positive) vs not.
        
        Simplified: we use BCE where target=1 if concept is positive (2),
        target=0 if negative (0), and mask unknown (1).
        """
        # Create binary targets and mask
        # positive (2) → 1, negative (0) → 0, unknown (1) → masked
        mask = (concept_targets != 1).float()  # [B, K]
        targets = (concept_targets == 2).float()  # [B, K]
        
        # BCE loss, masked
        bce = F.binary_cross_entropy_with_logits(
            concept_logits, targets, reduction="none"
        )  # [B, K]
        
        # Apply mask and average
        masked_bce = (bce * mask).sum() / (mask.sum() + self.eps)
        return masked_bce
    
    # ----- Classification loss -----
    
    def classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Standard cross-entropy for sentiment classification."""
        return F.cross_entropy(logits, labels)
    
    # ----- Desideratum 3: Geometric separability losses -----
    
    def concept_sparsity_loss(
        self,
        concept_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_sparse = E[|c|] + Σ_i √(λ_i(Σ_c))
        
        Combines L1 activation sparsity with nuclear norm penalty on
        concept covariance. The nuclear norm term prevents linearly
        dependent concepts from creating spurious flat directions
        in the loss landscape.
        """
        # L1 activation sparsity
        l1_loss = concept_probs.abs().mean()
        
        # Nuclear norm on concept covariance
        if concept_probs.size(0) > 1 and concept_probs.size(1) > 1:
            # Center concepts
            c_centered = concept_probs - concept_probs.mean(dim=0, keepdim=True)
            # Empirical covariance
            cov = (c_centered.T @ c_centered) / (concept_probs.size(0) - 1)
            # Add small regularization for numerical stability
            cov = cov + torch.eye(cov.size(0), device=cov.device) * self.eps
            # Eigenvalues
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=self.eps)
            # Nuclear norm ≈ sum of sqrt(eigenvalues)
            nuclear_loss = torch.sqrt(eigenvalues).mean()
        else:
            nuclear_loss = torch.tensor(0.0, device=concept_probs.device)
        
        return l1_loss + nuclear_loss
    
    def concept_diversity_loss(
        self,
        concept_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eigenvalue-weighted diversity loss.
        
        Penalizes pairwise similarity between concept activation vectors,
        weighted by concept covariance eigenvalues to focus decorrelation
        on geometrically important directions.
        """
        B, K = concept_probs.shape
        if B < 2 or K < 2:
            return torch.tensor(0.0, device=concept_probs.device)
        
        # Compute concept covariance eigenvalues for weighting
        c_centered = concept_probs - concept_probs.mean(dim=0, keepdim=True)
        cov = (c_centered.T @ c_centered) / (B - 1)
        cov = cov + torch.eye(K, device=cov.device) * self.eps
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        
        # Pairwise cosine similarity of concept vectors across samples
        c_norm = F.normalize(concept_probs, p=2, dim=-1)  # [B, K]
        sim = c_norm @ c_norm.T  # [B, B]
        
        # Target: identity (each sample's concept vector should be unique)
        eye = torch.eye(B, device=sim.device)
        
        # Eigenvalue-weighted deviation from identity
        # Weight by outer product of eigenvalues (focus on high-variance dirs)
        weight = torch.outer(eigenvalues, eigenvalues)  # [K, K]
        # Average weight across concept dimensions for the sample-level sim matrix
        avg_weight = weight.mean()
        
        diversity_loss = ((sim - eye).pow(2)).mean() * avg_weight
        
        return diversity_loss
    
    # ----- Total loss -----
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        concepts: torch.Tensor,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with all components.
        
        Args:
            outputs: Dict from model forward pass
            labels: [B] sentiment labels
            concepts: [B, K] concept labels
            epoch: current epoch (for KL annealing)
        
        Returns:
            Dict with 'total_loss' and individual component values
        """
        # D1: Scale-invariant reconstruction
        l_recon = self.reconstruction_loss(
            outputs["h_original"], outputs["h_recon"]
        )
        
        # D2: KL with sigmoid annealing
        l_kl = self.kl_divergence(outputs["mu"], outputs["logvar"])
        beta = self.kl_weight(epoch)
        
        # Concept supervision
        l_concept = self.concept_loss(outputs["concept_logits"], concepts)
        
        # Classification
        l_classify = self.classification_loss(outputs["logits"], labels)
        
        # D3: Concept regularization for geometric separability
        l_sparse = self.concept_sparsity_loss(outputs["concept_probs"])
        l_diverse = self.concept_diversity_loss(outputs["concept_probs"])
        
        # Total
        total = (
            self.config.recon_weight * l_recon
            + beta * l_kl
            + self.config.concept_weight * l_concept
            + self.config.classify_weight * l_classify
            + self.config.concept_sparsity_weight * l_sparse
            + self.config.concept_diversity_weight * l_diverse
        )
        
        return {
            "total_loss": total,
            "recon_loss": l_recon.item(),
            "kl_loss": l_kl.item(),
            "kl_weight": beta,
            "concept_loss": l_concept.item(),
            "classify_loss": l_classify.item(),
            "sparsity_loss": l_sparse.item(),
            "diversity_loss": l_diverse.item(),
        }
