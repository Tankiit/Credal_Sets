"""
Vision Loss Functions for GACS

Same desiderata as the NLP loss, but with pixel reconstruction
instead of CLS embedding contrastive loss.

D1 (Scale Invariance): Normalized pixel reconstruction (BCE on [0,1] images)
D2 (Smooth Convergence): Same sigmoid KL annealing
D3 (Geometric Separability): Same concept sparsity + diversity

Since MedMNIST has no concept labels, concept supervision is dropped.
Concepts are learned unsupervised and regularized for diversity.
"""
import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np


class VisionGACSLoss:
    """Loss for vision VAE. Reuses D2/D3 logic from NLP loss."""
    
    def __init__(self, config):
        self.config = config.loss
        self.eps = 1e-8
    
    def reconstruction_loss(
        self,
        x_recon: torch.Tensor,
        x_original: torch.Tensor,
    ) -> torch.Tensor:
        """
        D1: Scale-invariant pixel reconstruction.
        
        BCE loss on [0,1] normalized images.
        BCE is naturally scale-invariant (operates on probabilities),
        satisfying Desideratum 1.
        """
        # BCE treats decoder output as Bernoulli parameters
        return F.binary_cross_entropy(
            x_recon, x_original, reduction="mean"
        )
    
    def kl_weight(self, epoch: int) -> float:
        """D2: Sigmoid KL annealing (identical to NLP version)."""
        warmup = self.config.kl_warmup_epochs
        if warmup <= 0:
            return self.config.kl_weight_max
        progress = min(epoch / warmup, 1.0)
        beta = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        return beta * self.config.kl_weight_max
    
    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q(z|x) || N(0,I))."""
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        mu = torch.clamp(mu, min=-20.0, max=20.0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    def concept_sparsity_loss(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """D3: L1 + nuclear norm on concept covariance."""
        l1 = concept_probs.abs().mean()
        
        if concept_probs.size(0) > 1 and concept_probs.size(1) > 1:
            c = concept_probs - concept_probs.mean(dim=0, keepdim=True)
            cov = (c.T @ c) / (concept_probs.size(0) - 1)
            cov = cov + torch.eye(cov.size(0), device=cov.device) * self.eps
            eigs = torch.linalg.eigvalsh(cov)
            eigs = torch.clamp(eigs, min=self.eps)
            nuclear = torch.sqrt(eigs).mean()
        else:
            nuclear = torch.tensor(0.0, device=concept_probs.device)
        
        return l1 + nuclear
    
    def concept_diversity_loss(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """D3: Eigenvalue-weighted pairwise diversity."""
        B, K = concept_probs.shape
        if B < 2 or K < 2:
            return torch.tensor(0.0, device=concept_probs.device)
        
        c_norm = F.normalize(concept_probs, p=2, dim=-1)
        sim = c_norm @ c_norm.T
        eye = torch.eye(B, device=sim.device)
        return ((sim - eye).pow(2)).mean()
    
    def classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(logits, labels)
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        concepts: torch.Tensor = None,  # unused for vision, kept for API compat
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss."""
        # D1: Reconstruction
        l_recon = self.reconstruction_loss(
            outputs["x_recon"], outputs["x_original"]
        )
        
        # D2: KL with annealing
        l_kl = self.kl_divergence(outputs["mu"], outputs["logvar"])
        beta = self.kl_weight(epoch)
        
        # Classification
        l_cls = self.classification_loss(outputs["logits"], labels)
        
        # D3: Concept regularization (unsupervised)
        l_sparse = self.concept_sparsity_loss(outputs["concept_probs"])
        l_diverse = self.concept_diversity_loss(outputs["concept_probs"])
        
        total = (
            self.config.recon_weight * l_recon
            + beta * l_kl
            + self.config.classify_weight * l_cls
            + self.config.concept_sparsity_weight * l_sparse
            + self.config.concept_diversity_weight * l_diverse
        )
        
        return {
            "total_loss": total,
            "recon_loss": l_recon.item(),
            "kl_loss": l_kl.item(),
            "kl_weight": beta,
            "classify_loss": l_cls.item(),
            "sparsity_loss": l_sparse.item(),
            "diversity_loss": l_diverse.item(),
            "concept_loss": 0.0,  # no concept supervision in vision
        }
