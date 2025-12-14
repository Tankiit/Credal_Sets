"""
Variational Credal Concept Bottleneck Models for ACL 2026
Implementation with Mean-Field and Low-Rank Gaussian posteriors

Author: Tanmoy
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class VariationalConfig:
    """Configuration for variational credal CBM"""
    # Model architecture
    encoder_name: str = "distilbert-base-uncased"
    num_concepts: int = 20
    num_classes: int = 2

    # Variational family
    variational_family: str = "mean_field"  # "mean_field" or "low_rank"
    low_rank_dim: int = 5  # Only for low_rank

    # Prior
    prior_std: float = 1.0

    # Training
    kl_weight: float = 1e-5
    num_mc_samples: int = 10

    # Aleatoric network
    aleatoric_hidden_dim: int = 64
    aleatoric_dropout: float = 0.3


# ============================================================================
# CORE VARIATIONAL LAYERS
# ============================================================================

class MeanFieldVariationalLayer(nn.Module):
    """
    Mean-Field variational layer: q(W) = ∏ᵢ N(μᵢ, σᵢ²)
    O(n) parameters, no correlations between weights
    """
    def __init__(self, input_dim: int, num_concepts: int, prior_std: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts

        # Variational parameters (μ, ρ where σ = log(1 + exp(ρ)))
        self.weight_mu = nn.Parameter(torch.randn(num_concepts, input_dim) * 0.01)
        self.weight_rho = nn.Parameter(torch.ones(num_concepts, input_dim) * -3.0)
        self.bias_mu = nn.Parameter(torch.zeros(num_concepts))
        self.bias_rho = nn.Parameter(torch.ones(num_concepts) * -3.0)

        # Prior
        self.register_buffer('prior_std', torch.tensor(prior_std))

    def get_weight_std(self):
        """Compute weight standard deviation from rho"""
        return torch.log1p(torch.exp(self.weight_rho))

    def get_bias_std(self):
        """Compute bias standard deviation from rho"""
        return torch.log1p(torch.exp(self.bias_rho))

    def sample_weights(self):
        """Sample weights using reparameterization trick"""
        # W = μ + σ * ε, where ε ~ N(0, 1)
        weight_std = self.get_weight_std()
        eps_w = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * eps_w

        bias_std = self.get_bias_std()
        eps_b = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_std * eps_b

        return weight, bias

    def kl_divergence(self):
        """
        KL[q(W) || p(W)] for Gaussian posterior and prior
        Closed-form solution for mean-field
        """
        # For weights
        weight_std = self.get_weight_std()
        kl_weight = -0.5 * torch.sum(
            1 + torch.log(weight_std**2)
            - self.weight_mu**2 / self.prior_std**2
            - weight_std**2 / self.prior_std**2
        )

        # For biases
        bias_std = self.get_bias_std()
        kl_bias = -0.5 * torch.sum(
            1 + torch.log(bias_std**2)
            - self.bias_mu**2 / self.prior_std**2
            - bias_std**2 / self.prior_std**2
        )

        return kl_weight + kl_bias

    def forward(self, x: torch.Tensor):
        """Single forward pass with sampled weights"""
        weight, bias = self.sample_weights()
        logits = F.linear(x, weight, bias)
        probs = torch.sigmoid(logits)
        return probs


class LowRankVariationalLayer(nn.Module):
    """
    Low-Rank variational layer: q(W) ~ N(μ, VV^T + D)
    O(nk) parameters, captures global correlations via rank-k factor
    """
    def __init__(self, input_dim: int, num_concepts: int,
                 rank: int = 5, prior_std: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.rank = min(rank, num_concepts)  # Ensure rank ≤ num_concepts

        # Mean parameters (same as mean-field)
        self.weight_mu = nn.Parameter(torch.randn(num_concepts, input_dim) * 0.01)
        self.bias_mu = nn.Parameter(torch.zeros(num_concepts))

        # Low-rank covariance factor: Σ = VV^T + D
        # V is [num_concepts*input_dim, rank]
        total_params = num_concepts * input_dim
        self.cov_factor = nn.Parameter(torch.randn(total_params, self.rank) * 0.01)

        # Diagonal component D (log parameterization for positivity)
        self.cov_diag_rho = nn.Parameter(torch.ones(total_params) * -3.0)

        # Bias covariance (keep as diagonal for simplicity)
        self.bias_rho = nn.Parameter(torch.ones(num_concepts) * -3.0)

        # Prior
        self.register_buffer('prior_std', torch.tensor(prior_std))

    def get_cov_diag(self):
        """Compute diagonal covariance component"""
        return torch.log1p(torch.exp(self.cov_diag_rho))

    def get_bias_std(self):
        """Compute bias standard deviation"""
        return torch.log1p(torch.exp(self.bias_rho))

    def sample_weights(self):
        """
        Sample from low-rank Gaussian using torch.distributions
        Uses built-in LowRankMultivariateNormal
        """
        # Flatten weight parameters
        mu_flat = self.weight_mu.flatten()

        # Create low-rank multivariate normal distribution
        # Σ = cov_factor @ cov_factor^T + diag(cov_diag)
        posterior = dist.LowRankMultivariateNormal(
            loc=mu_flat,
            cov_factor=self.cov_factor,
            cov_diag=self.get_cov_diag()
        )

        # Sample and reshape
        weight_sample = posterior.rsample()
        weight = weight_sample.reshape(self.num_concepts, self.input_dim)

        # Sample bias (still diagonal)
        bias_std = self.get_bias_std()
        eps_b = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_std * eps_b

        return weight, bias

    def kl_divergence(self):
        """
        KL divergence approximation for low-rank Gaussian
        Uses Monte Carlo estimation since no closed form
        """
        # Sample from posterior
        mu_flat = self.weight_mu.flatten()
        posterior = dist.LowRankMultivariateNormal(
            loc=mu_flat,
            cov_factor=self.cov_factor,
            cov_diag=self.get_cov_diag()
        )

        # Prior (isotropic Gaussian)
        prior = dist.MultivariateNormal(
            loc=torch.zeros_like(mu_flat),
            covariance_matrix=torch.eye(len(mu_flat), device=mu_flat.device) * self.prior_std**2
        )

        # Monte Carlo KL estimation (1 sample for efficiency)
        # KL[q||p] ≈ log q(z) - log p(z) where z ~ q
        z = posterior.rsample()
        kl_weights = posterior.log_prob(z) - prior.log_prob(z)

        # Bias KL (closed form, still diagonal)
        bias_std = self.get_bias_std()
        kl_bias = -0.5 * torch.sum(
            1 + torch.log(bias_std**2)
            - self.bias_mu**2 / self.prior_std**2
            - bias_std**2 / self.prior_std**2
        )

        return kl_weights + kl_bias

    def forward(self, x: torch.Tensor):
        """Single forward pass with sampled weights"""
        weight, bias = self.sample_weights()
        logits = F.linear(x, weight, bias)
        probs = torch.sigmoid(logits)
        return probs


class AleatoricUncertaintyNetwork(nn.Module):
    """
    Network to predict aleatoric (data) uncertainty
    Maps encoder outputs to per-concept uncertainty estimates
    """
    def __init__(self, input_dim: int, num_concepts: int,
                 hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
            nn.Softplus()  # Ensure positive output
        )

    def forward(self, x: torch.Tensor):
        """Predict aleatoric uncertainty (variance)"""
        return self.network(x)


# ============================================================================
# MAIN MODEL
# ============================================================================

class VariationalCredalCBM(nn.Module):
    """
    Variational Credal Concept Bottleneck Model

    Supports two variational families:
    1. Mean-Field: q(W) = ∏ᵢ N(μᵢ, σᵢ²)
    2. Low-Rank: q(W) ~ N(μ, VV^T + D)
    """
    def __init__(self, config: VariationalConfig):
        super().__init__()
        self.config = config

        # Text encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Variational concept layer
        if config.variational_family == "mean_field":
            self.concept_layer = MeanFieldVariationalLayer(
                input_dim=self.hidden_size,
                num_concepts=config.num_concepts,
                prior_std=config.prior_std
            )
        elif config.variational_family == "low_rank":
            self.concept_layer = LowRankVariationalLayer(
                input_dim=self.hidden_size,
                num_concepts=config.num_concepts,
                rank=config.low_rank_dim,
                prior_std=config.prior_std
            )
        else:
            raise ValueError(f"Unknown variational family: {config.variational_family}")

        # Aleatoric uncertainty network
        self.aleatoric_net = AleatoricUncertaintyNetwork(
            input_dim=self.hidden_size,
            num_concepts=config.num_concepts,
            hidden_dim=config.aleatoric_hidden_dim,
            dropout=config.aleatoric_dropout
        )

        # Concept-to-class classifier
        self.classifier = nn.Linear(config.num_concepts, config.num_classes)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Encode text to hidden representation"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # Mean pooling
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass with uncertainty quantification

        Returns:
            Dict containing:
            - concept_probs: Mean concept predictions [batch, num_concepts]
            - epistemic_uncertainty: Variance across MC samples [batch, num_concepts]
            - aleatoric_uncertainty: Predicted data uncertainty [batch, num_concepts]
            - logits: Classification logits [batch, num_classes]
            - loss: Total loss (if labels provided)
        """
        batch_size = input_ids.size(0)

        # Encode text
        hidden = self.encode(input_ids, attention_mask)  # [batch, hidden_size]

        # Monte Carlo sampling for epistemic uncertainty
        concept_samples = []
        for _ in range(self.config.num_mc_samples):
            concepts = self.concept_layer(hidden)  # [batch, num_concepts]
            concept_samples.append(concepts)

        # Stack samples: [num_samples, batch, num_concepts]
        concept_samples = torch.stack(concept_samples, dim=0)

        # Compute epistemic uncertainty (variance across samples)
        mean_concepts = concept_samples.mean(dim=0)  # [batch, num_concepts]
        epistemic_uncertainty = concept_samples.var(dim=0)  # [batch, num_concepts]

        # Compute aleatoric uncertainty
        aleatoric_uncertainty = self.aleatoric_net(hidden)  # [batch, num_concepts]

        # Classification from mean concepts
        logits = self.classifier(mean_concepts)  # [batch, num_classes]

        # Compute losses
        loss_dict = {}
        if labels is not None:
            # Classification loss
            ce_loss = F.cross_entropy(logits, labels)

            # KL divergence (regularization)
            kl_loss = self.concept_layer.kl_divergence()

            # Aleatoric loss (encourage meaningful uncertainty)
            # Negative log-likelihood under Gaussian with predicted variance
            # For binary concepts, we could use:
            # NLL = 0.5 * (log(2π) + log(σ²) + (c - μ)² / σ²)
            # Here we use a simple regularization: penalize very high/low uncertainty
            aleatoric_reg = torch.mean(
                torch.abs(aleatoric_uncertainty - 0.1)  # Target: moderate uncertainty
            )

            # Total loss
            total_loss = (
                ce_loss
                + self.config.kl_weight * kl_loss
                + 0.01 * aleatoric_reg
            )

            loss_dict = {
                'loss': total_loss,
                'ce_loss': ce_loss,
                'kl_loss': kl_loss,
                'aleatoric_reg': aleatoric_reg
            }

        return {
            'concept_probs': mean_concepts,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=-1),
            **loss_dict
        }


# ============================================================================
# CREDAL SET OPERATIONS
# ============================================================================

class CredalSet:
    """
    Credal set: convex set of probability distributions
    Represented by lower/upper bounds on probabilities
    """
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        """
        Args:
            lower: Lower probability bounds [batch, num_concepts]
            upper: Upper probability bounds [batch, num_concepts]
        """
        self.lower = lower
        self.upper = upper
        self.width = upper - lower

    @classmethod
    def from_uncertainty(cls, mean: torch.Tensor,
                        epistemic: torch.Tensor,
                        aleatoric: torch.Tensor,
                        confidence_level: float = 0.95):
        """
        Construct credal set from epistemic + aleatoric uncertainty

        Uses confidence intervals:
        [μ - z*√(σ²_epistemic + σ²_aleatoric), μ + z*√(σ²_epistemic + σ²_aleatoric)]
        """
        from scipy.stats import norm
        z = norm.ppf((1 + confidence_level) / 2)  # ~1.96 for 95%

        total_std = torch.sqrt(epistemic + aleatoric)
        margin = z * total_std

        lower = torch.clamp(mean - margin, 0.0, 1.0)
        upper = torch.clamp(mean + margin, 0.0, 1.0)

        return cls(lower, upper)

    def contains(self, point: torch.Tensor) -> torch.Tensor:
        """Check if point is in credal set"""
        return (self.lower <= point) & (point <= self.upper)

    def intersection(self, other: 'CredalSet') -> 'CredalSet':
        """Compute intersection of two credal sets"""
        new_lower = torch.max(self.lower, other.lower)
        new_upper = torch.min(self.upper, other.upper)
        return CredalSet(new_lower, new_upper)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_calibration_metrics(probs: torch.Tensor, targets: torch.Tensor,
                                num_bins: int = 10):
    """
    Compute Expected Calibration Error (ECE)

    Args:
        probs: Predicted probabilities [N]
        targets: Binary targets [N]
        num_bins: Number of bins for calibration

    Returns:
        ECE, MCE, Brier score
    """
    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].float().mean()
            avg_confidence_in_bin = probs[in_bin].mean()

            # ECE: weighted average of |accuracy - confidence|
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            # MCE: maximum calibration error
            mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin))

    # Brier score
    brier = torch.mean((probs - targets.float())**2)

    return ece.item(), mce, brier.item()


def epistemic_error_correlation(epistemic_uncertainty: torch.Tensor,
                                errors: torch.Tensor):
    """
    Compute correlation between epistemic uncertainty and prediction errors

    Args:
        epistemic_uncertainty: Per-sample uncertainty [N] or [N, concepts]
        errors: Binary error indicator [N]

    Returns:
        Pearson correlation coefficient
    """
    # If epistemic_uncertainty is multi-dimensional, take mean across concepts
    if epistemic_uncertainty.dim() > 1:
        epistemic_uncertainty = epistemic_uncertainty.mean(dim=1)

    # Ensure we have the right number of samples
    if len(epistemic_uncertainty) != len(errors):
        print(f"Warning: Shape mismatch - epistemic: {epistemic_uncertainty.shape}, errors: {errors.shape}")
        min_len = min(len(epistemic_uncertainty), len(errors))
        epistemic_uncertainty = epistemic_uncertainty[:min_len]
        errors = errors[:min_len]

    # Compute Pearson correlation
    eps_mean = epistemic_uncertainty.mean()
    err_mean = errors.float().mean()

    numerator = torch.mean((epistemic_uncertainty - eps_mean) * (errors.float() - err_mean))
    denominator = torch.std(epistemic_uncertainty) * torch.std(errors.float())

    if denominator > 0:
        return (numerator / denominator).item()
    else:
        return 0.0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = VariationalConfig(
        encoder_name="distilbert-base-uncased",
        num_concepts=20,
        num_classes=2,
        variational_family="mean_field",  # or "low_rank"
        low_rank_dim=5,
        num_mc_samples=10
    )

    # Create model
    model = VariationalCredalCBM(config)

    # Dummy data
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size,))

    # Forward pass
    outputs = model(input_ids, attention_mask, labels)

    print("\n=== Model Outputs ===")
    print(f"Concept predictions: {outputs['concept_probs'].shape}")
    print(f"Epistemic uncertainty: {outputs['epistemic_uncertainty'].shape}")
    print(f"Aleatoric uncertainty: {outputs['aleatoric_uncertainty'].shape}")
    print(f"Logits: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Create credal sets
    credal_sets = CredalSet.from_uncertainty(
        mean=outputs['concept_probs'],
        epistemic=outputs['epistemic_uncertainty'],
        aleatoric=outputs['aleatoric_uncertainty'],
        confidence_level=0.95
    )

    print(f"\n=== Credal Sets ===")
    print(f"Lower bounds: {credal_sets.lower[0, :5]}")  # First 5 concepts
    print(f"Upper bounds: {credal_sets.upper[0, :5]}")
    print(f"Interval widths: {credal_sets.width[0, :5]}")

    # Compare mean-field vs low-rank
    print("\n=== Comparing Variational Families ===")

    config_mf = VariationalConfig(variational_family="mean_field")
    model_mf = VariationalCredalCBM(config_mf)

    config_lr = VariationalConfig(variational_family="low_rank", low_rank_dim=5)
    model_lr = VariationalCredalCBM(config_lr)

    # Count parameters
    mf_params = sum(p.numel() for p in model_mf.concept_layer.parameters())
    lr_params = sum(p.numel() for p in model_lr.concept_layer.parameters())

    print(f"Mean-Field concept layer params: {mf_params:,}")
    print(f"Low-Rank concept layer params: {lr_params:,}")
    print(f"Ratio: {lr_params / mf_params:.2f}x")