"""
MAQA Credal Loss V7b - FINAL FIXED VERSION
==========================================

This is the complete, tested version that integrates with main_train_hybrid_multi_dataset.py

Key fixes:
1. CredalSetParams as class (not dataclass) with proper property methods
2. Constructor accepts both config dict and kwargs
3. Adapter provides ALL keys expected by trainer
4. initialize_model_for_v7b with proper initialization

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ==============================================================================
# CREDAL SET STRUCTURE - Using class for property support
# ==============================================================================

class CredalSetParams:
    """
    Parameters defining a credal set of Gaussian distributions.

    Using a class instead of NamedTuple/dataclass to support @property methods.
    """

    def __init__(self, mu: torch.Tensor, sigma_epi: torch.Tensor, sigma_ale: torch.Tensor):
        self.mu = mu
        self.sigma_epi = sigma_epi
        self.sigma_ale = sigma_ale

    @property
    def sigma_lower(self) -> torch.Tensor:
        """Lower bound of credal set (epistemic only)."""
        return self.sigma_epi

    @property
    def sigma_upper(self) -> torch.Tensor:
        """Upper bound of credal set (total uncertainty)."""
        return torch.sqrt(self.sigma_epi**2 + self.sigma_ale**2)

    @property
    def credal_width(self) -> torch.Tensor:
        """Width of credal set (measures imprecision)."""
        return self.sigma_upper - self.sigma_lower


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG_V7B = {
    # Training
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'dropout': 0.2,

    # === LOSS WEIGHTS ===
    'lambda_answer': 1.0,

    # Epistemic losses
    'lambda_kl_epi': 0.001,
    'lambda_epi_residual': 1.5,

    # Aleatoric losses
    'lambda_ale_entropy': 2.0,
    'lambda_ale_rank': 1.0,

    # DISENTANGLEMENT
    'lambda_decorr': 5.0,
    'lambda_gradient_isolation': 1.0,

    # Credal-specific
    'lambda_credal_width': 0.5,
    'min_credal_width': 0.1,

    # Capacity constraint
    'lambda_capacity': 0.5,
    'target_total_sigma': 0.8,

    # Variance regularization
    'lambda_variance': 0.5,
    'target_std_epi': 0.1,
    'target_std_ale': 0.15,

    # Bounds
    'min_sigma': 0.05,
    'max_sigma': 1.5,
    'prior_sigma_epi': 0.3,

    # Ranking
    'rank_margin': 0.1,
    'rank_num_pairs': 64,
}


# ==============================================================================
# LOSS FUNCTION V7B
# ==============================================================================

class MAQACredalLossV7b(nn.Module):
    """
    V7b: Credal-preserving disentanglement with gradient isolation.

    Key mechanisms:
    1. σ_ale → entropy (aleatoric = inherent ambiguity)
    2. σ_epi → residual error with .detach() (gradient isolation)
    3. Decorrelation penalty
    4. Capacity constraint (β-VAE style)
    """

    def __init__(self, config: Dict = None, **kwargs):
        super().__init__()

        # Merge config sources: defaults < config dict < kwargs
        cfg = {**CONFIG_V7B}
        if config is not None:
            cfg.update(config)
        cfg.update(kwargs)

        # Store all hyperparameters
        for key, value in cfg.items():
            setattr(self, key, value)

    def forward(
        self,
        params,  # Can be CredalSetParams or any object with mu, sigma_epi, sigma_ale
        p_star: torch.Tensor,
        entropy: torch.Tensor,
        params_clear = None,
    ) -> Dict[str, torch.Tensor]:

        device = params.mu.device

        # Extract and clamp uncertainties
        sigma_epi = torch.clamp(params.sigma_epi, self.min_sigma, self.max_sigma)
        sigma_ale = torch.clamp(params.sigma_ale, self.min_sigma, self.max_sigma)

        # Compute credal properties
        sigma_upper = torch.sqrt(sigma_epi**2 + sigma_ale**2)
        credal_width = sigma_upper - sigma_epi

        # -----------------------------------------------------------------
        # 1. ANSWER LOSS + PREDICTION ERROR
        # -----------------------------------------------------------------
        answer_loss, pred_error = self._compute_answer_loss_and_error(params.mu, p_star)

        # -----------------------------------------------------------------
        # 2. EPISTEMIC LOSSES
        # -----------------------------------------------------------------
        kl_epi_loss = self._compute_kl_loss(params.mu, sigma_epi)

        # Residual error with GRADIENT ISOLATION
        # KEY: .detach() on sigma_ale prevents gradient flow!
        residual_error = pred_error - sigma_ale.detach()
        residual_normalized = torch.sigmoid(residual_error * 3)
        target_epi = self.min_sigma + (self.max_sigma - self.min_sigma) * residual_normalized
        epi_residual_loss = F.mse_loss(sigma_epi, target_epi.detach())

        # -----------------------------------------------------------------
        # 3. ALEATORIC LOSSES
        # -----------------------------------------------------------------
        ale_entropy_loss = F.mse_loss(sigma_ale, entropy)
        ale_rank_loss = self._compute_ranking_loss(sigma_ale, entropy)

        # -----------------------------------------------------------------
        # 4. DISENTANGLEMENT LOSSES
        # -----------------------------------------------------------------
        decorr_loss = self._compute_decorrelation_loss(sigma_epi, sigma_ale)
        gradient_isolation_loss = self._compute_gradient_isolation_loss(sigma_epi, entropy)

        # -----------------------------------------------------------------
        # 5. CREDAL-SPECIFIC LOSSES
        # -----------------------------------------------------------------
        width_loss = F.relu(self.min_credal_width - credal_width).mean()
        capacity_loss = F.mse_loss(
            sigma_upper,
            torch.full_like(sigma_upper, self.target_total_sigma)
        )

        # -----------------------------------------------------------------
        # 6. VARIANCE REGULARIZATION
        # -----------------------------------------------------------------
        variance_loss = self._compute_variance_loss(sigma_epi, sigma_ale)

        # -----------------------------------------------------------------
        # TOTAL LOSS
        # -----------------------------------------------------------------
        total_loss = (
            self.lambda_answer * answer_loss +
            self.lambda_kl_epi * kl_epi_loss +
            self.lambda_epi_residual * epi_residual_loss +
            self.lambda_ale_entropy * ale_entropy_loss +
            self.lambda_ale_rank * ale_rank_loss +
            self.lambda_decorr * decorr_loss +
            self.lambda_gradient_isolation * gradient_isolation_loss +
            self.lambda_credal_width * width_loss +
            self.lambda_capacity * capacity_loss +
            self.lambda_variance * variance_loss
        )

        # Compute correlations for logging
        with torch.no_grad():
            rho_eu_au = self._compute_correlation(sigma_epi, sigma_ale)
            rho_au_entropy = self._compute_correlation(sigma_ale, entropy)
            rho_eu_entropy = self._compute_correlation(sigma_epi, entropy)

        return {
            # Total
            'loss_total': total_loss,

            # Individual losses
            'loss_answer': answer_loss,
            'loss_kl_epi': kl_epi_loss,
            'loss_epi_residual': epi_residual_loss,
            'loss_ale_entropy': ale_entropy_loss,
            'loss_ale_rank': ale_rank_loss,
            'loss_decorr': decorr_loss,
            'loss_gradient_isolation': gradient_isolation_loss,
            'loss_credal_width': width_loss,
            'loss_capacity': capacity_loss,
            'loss_variance': variance_loss,

            # Correlations
            'rho_eu_au': rho_eu_au,
            'rho_au_entropy': rho_au_entropy,
            'rho_eu_entropy': rho_eu_entropy,

            # Credal metrics
            'mean_credal_width': credal_width.mean(),
            'mean_sigma_epi': sigma_epi.mean(),
            'mean_sigma_ale': sigma_ale.mean(),
            'mean_sigma_upper': sigma_upper.mean(),
            'mean_sigma_lower': sigma_epi.mean(),  # sigma_lower = sigma_epi

            # Diagnostics
            'mean_pred_error': pred_error.mean(),
            'mean_residual': residual_error.mean(),
        }

    def _compute_answer_loss_and_error(self, mu, p_star):
        batch_size = mu.size(0)
        losses = []
        errors = []

        for i in range(batch_size):
            num_answers = (p_star[i] > 0).sum().item()

            if num_answers > 0:
                mu_i = mu[i, :num_answers]
                p_star_i = p_star[i, :num_answers]
                pred_dist = F.softmax(mu_i, dim=-1)

                kl = F.kl_div(pred_dist.log().clamp(min=-10), p_star_i, reduction='sum')
                losses.append(kl)

                prob_correct = (pred_dist * p_star_i).sum()
                errors.append(1.0 - prob_correct)
            else:
                losses.append(torch.tensor(0.0, device=mu.device))
                errors.append(torch.tensor(0.5, device=mu.device))

        return torch.stack(losses).mean(), torch.stack(errors)

    def _compute_kl_loss(self, mu, sigma_epi):
        prior_var = self.prior_sigma_epi ** 2
        if sigma_epi.dim() == 1:
            sigma_epi = sigma_epi.unsqueeze(-1)
        var_ratio = (sigma_epi ** 2) / prior_var
        mu_term = (mu ** 2) / prior_var
        kl_per_dim = 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio + 1e-10))
        return kl_per_dim.sum(dim=-1).mean()

    def _compute_ranking_loss(self, sigma, target, num_pairs=None):
        if num_pairs is None:
            num_pairs = self.rank_num_pairs
        batch_size = sigma.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=sigma.device)

        num_pairs = min(num_pairs, batch_size * (batch_size - 1) // 2)
        losses = []

        for _ in range(num_pairs):
            i, j = torch.randint(0, batch_size, (2,)).tolist()
            if i == j:
                continue
            if target[i] > target[j]:
                loss = F.relu(self.rank_margin - (sigma[i] - sigma[j]))
            else:
                loss = F.relu(self.rank_margin - (sigma[j] - sigma[i]))
            losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma.device)

    def _compute_decorrelation_loss(self, sigma_epi, sigma_ale):
        corr = self._compute_correlation(sigma_epi, sigma_ale)
        return corr ** 2

    def _compute_gradient_isolation_loss(self, sigma_epi, entropy):
        corr = self._compute_correlation(sigma_epi, entropy)
        return corr ** 2

    def _compute_variance_loss(self, sigma_epi, sigma_ale):
        penalty_epi = F.relu(self.target_std_epi ** 2 - sigma_epi.var())
        penalty_ale = F.relu(self.target_std_ale ** 2 - sigma_ale.var())
        return penalty_epi + penalty_ale

    def _compute_correlation(self, x, y):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        cov = (x_centered * y_centered).mean()
        std_x = x.std() + 1e-8
        std_y = y.std() + 1e-8
        return cov / (std_x * std_y)


# ==============================================================================
# ADAPTER FOR TRAINER COMPATIBILITY
# ==============================================================================

class MAQACredalLossV7bAdapter(nn.Module):
    """
    Adapter that maps V7b loss outputs to the keys expected by MAQACredalTrainer.

    Provides ALL keys the trainer expects, with proper aliases for compatibility.
    """

    def __init__(self, loss_v7b: MAQACredalLossV7b):
        super().__init__()
        self.loss_v7b = loss_v7b

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        """
        Forward pass with comprehensive key mapping.
        """
        # Convert to CredalSetParams if needed (for property access)
        if not isinstance(params, CredalSetParams):
            params = CredalSetParams(
                mu=params.mu,
                sigma_epi=params.sigma_epi,
                sigma_ale=params.sigma_ale,
            )

        losses = self.loss_v7b(params, p_star, entropy_gt, params_clear)

        device = losses['loss_total'].device

        # Comprehensive key mapping for trainer compatibility
        loss_dict = {
            # === REQUIRED BY TRAINER ===
            'loss_total': losses['loss_total'],

            # Answer loss (multiple aliases)
            'loss_kl': losses['loss_answer'],
            'loss_answer': losses['loss_answer'],
            'answer_loss': losses['loss_answer'],

            # KL regularization (multiple aliases)
            'loss_reg': losses['loss_kl_epi'],
            'loss_kl_v3': losses['loss_kl_epi'],
            'kl_reg_loss': losses['loss_kl_epi'],

            # Calibration (sum of aleatoric losses, multiple aliases)
            'loss_cal': losses['loss_ale_entropy'] + losses['loss_ale_rank'],
            'calibration_loss': losses['loss_ale_entropy'] + losses['loss_ale_rank'],
            'loss_cal_mse': losses['loss_ale_entropy'],
            'loss_cal_rank': losses['loss_ale_rank'],

            # Contrastive (V7b doesn't use, set to 0)
            'loss_cont': torch.tensor(0.0, device=device),
            'contrastive_loss': torch.tensor(0.0, device=device),

            # === V7B SPECIFIC ===
            'loss_epi_residual': losses['loss_epi_residual'],
            'loss_epi_error': losses['loss_epi_residual'],  # Alias
            'loss_ale_entropy': losses['loss_ale_entropy'],
            'loss_ale_rank': losses['loss_ale_rank'],
            'loss_decorr': losses['loss_decorr'],
            'loss_gradient_isolation': losses['loss_gradient_isolation'],
            'loss_credal_width': losses['loss_credal_width'],
            'loss_capacity': losses['loss_capacity'],
            'loss_variance': losses['loss_variance'],

            # === CORRELATIONS ===
            'rho_eu_au': losses['rho_eu_au'],
            'rho_au_entropy': losses['rho_au_entropy'],
            'rho_eu_entropy': losses['rho_eu_entropy'],

            # === CREDAL METRICS ===
            'mean_credal_width': losses['mean_credal_width'],
            'mean_sigma_epi': losses['mean_sigma_epi'],
            'mean_sigma_ale': losses['mean_sigma_ale'],
            'mean_sigma_upper': losses['mean_sigma_upper'],
            'mean_sigma_lower': losses['mean_sigma_lower'],

            # === DIAGNOSTICS ===
            'mean_pred_error': losses['mean_pred_error'],
            'mean_residual': losses['mean_residual'],
        }

        return losses['loss_total'], loss_dict


# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================

def initialize_model_for_v7b(model, config=None):
    """
    Initialize model's uncertainty heads for V7b training.

    σ_ale → ~0.5 (mean entropy)
    σ_epi → ~0.3 (residual)
    """
    if config is None:
        config = CONFIG_V7B

    # Initialize σ_ale head to output ~0.5
    if hasattr(model, 'sigma_ale_head'):
        if hasattr(model.sigma_ale_head, 'weight'):
            nn.init.xavier_uniform_(model.sigma_ale_head.weight, gain=0.5)
        if hasattr(model.sigma_ale_head, 'bias'):
            nn.init.constant_(model.sigma_ale_head.bias, -0.2)

    # Initialize σ_epi head to output ~0.3
    if hasattr(model, 'sigma_epi_head'):
        if hasattr(model.sigma_epi_head, 'weight'):
            nn.init.xavier_uniform_(model.sigma_epi_head.weight, gain=0.3)
        if hasattr(model.sigma_epi_head, 'bias'):
            nn.init.constant_(model.sigma_epi_head.bias, -1.0)

    print(f"  ✓ Model initialized for V7b:")
    print(f"    σ_ale target: ~0.5 (mean entropy)")
    print(f"    σ_epi target: ~0.3 (residual)")

    return model


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_v7b_loss(config: Dict = None, **kwargs) -> MAQACredalLossV7b:
    """Create V7b loss with optional config overrides."""
    return MAQACredalLossV7b(config=config, **kwargs)


def create_v7b_adapter(config: Dict = None, **kwargs) -> MAQACredalLossV7bAdapter:
    """Create V7b loss wrapped in adapter for trainer compatibility."""
    loss = create_v7b_loss(config=config, **kwargs)
    return MAQACredalLossV7bAdapter(loss)


# ==============================================================================
# SUMMARY
# ==============================================================================

SUMMARY = """
V7b Loss - Credal-Preserving Disentanglement (FINAL VERSION)
============================================================

Credal Structure:
  C(x) = {N(μ, σ²) : σ_epi² ≤ σ² ≤ σ_epi² + σ_ale²}

Disentanglement Mechanisms:
  1. σ_ale → entropy H[p*] (aleatoric = inherent ambiguity)
  2. σ_epi → residual error with .detach() (gradient isolation!)
  3. Decorrelation penalty: ρ(σ_epi, σ_ale) → 0
  4. Gradient isolation penalty: ρ(σ_epi, entropy) → 0
  5. Capacity constraint: σ_upper ≈ target (β-VAE style)

Expected Results:
  ρ(EU, AU) < 0.3        (decorrelation)
  ρ(AU, H[p*]) > 0.5    (aleatoric validity)
  ρ(EU, Error|H) > 0.3  (epistemic validity)

Usage:
  from maqa_credal_loss_v7b_fixed import (
      create_v7b_adapter,
      initialize_model_for_v7b,
      CONFIG_V7B,
  )

  model = initialize_model_for_v7b(model, CONFIG_V7B)
  trainer.criterion = create_v7b_adapter(config=CONFIG_V7B)
"""

if __name__ == "__main__":
    print(SUMMARY)
    print("\nConfig:")
    for k, v in CONFIG_V7B.items():
        print(f"  {k}: {v}")
