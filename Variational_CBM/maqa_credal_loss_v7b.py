"""
MAQA Credal Loss V7b - Hybrid Disentanglement with Credal Structure
===================================================================

Addressing the key questions:

Q1: What should each uncertainty capture?
  - σ_ale: Annotator disagreement, H[p*], linguistic ambiguity
  - σ_epi: Model capacity limits, OOD detection, reducible uncertainty

Q2: Can we maintain credal set interpretation?
  - YES! Credal set C(x) = {p : σ_epi² ≤ var ≤ σ_epi² + σ_ale²}
  - The architecture doesn't change the interpretation

V7b Approach - Hybrid Disentanglement:
  - σ_ale sees BOTH h (encoder) AND entropy, but with constraints
  - σ_epi sees h only
  - Disentanglement via: (1) different supervision, (2) capacity constraint, (3) gradient isolation

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, NamedTuple
from dataclasses import dataclass


# ==============================================================================
# CREDAL SET STRUCTURE
# ==============================================================================

@dataclass
class CredalSetParams:
    """
    Parameters defining a credal set of Gaussian distributions.

    The credal set is: C(x) = {N(μ, σ²) : σ_lower² ≤ σ² ≤ σ_upper²}

    Where:
      σ_lower = σ_epi (epistemic only - reducible uncertainty)
      σ_upper = sqrt(σ_epi² + σ_ale²) (total uncertainty)

    Interpretation:
      - Inner distribution N(μ, σ_epi²): Best case (if we had infinite data)
      - Outer distribution N(μ, σ_epi² + σ_ale²): Worst case (inherent + model)
    """
    mu: torch.Tensor           # [B, num_answers] - prediction mean
    sigma_epi: torch.Tensor    # [B] - epistemic (model uncertainty)
    sigma_ale: torch.Tensor    # [B] - aleatoric (data uncertainty)

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

    # Epistemic: KL regularization + residual error
    'lambda_kl_epi': 0.001,
    'lambda_epi_residual': 1.5,

    # Aleatoric: entropy matching + annotator signal
    'lambda_ale_entropy': 2.0,
    'lambda_ale_rank': 1.0,

    # DISENTANGLEMENT (key!)
    'lambda_decorr': 5.0,           # Soft decorrelation
    'lambda_gradient_isolation': 1.0,  # Stop gradient between heads

    # Credal-specific
    'lambda_credal_width': 0.5,     # Encourage meaningful credal width
    'min_credal_width': 0.1,        # Minimum width (prevents collapse)

    # Capacity constraint (β-VAE style)
    'lambda_capacity': 0.5,
    'target_total_sigma': 0.8,

    # Bounds
    'min_sigma': 0.05,
    'max_sigma': 1.5,
    'prior_sigma_epi': 0.3,
}


# ==============================================================================
# LOSS FUNCTION V7B
# ==============================================================================

class MAQACredalLossV7b(nn.Module):
    """
    V7b: Hybrid disentanglement maintaining credal structure.

    Key innovations:
    1. Both heads see h, but with DIFFERENT supervision signals
    2. Gradient isolation: stop_gradient between uncertainty computations
    3. Credal width regularization: ensure meaningful set width
    4. Capacity constraint: complementary usage of uncertainties

    Credal interpretation preserved:
      C(x) = {N(μ, σ²) : σ_epi² ≤ σ² ≤ σ_epi² + σ_ale²}
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, default in CONFIG_V7B.items():
            if key.startswith('lambda_') or key.startswith('min_') or key.startswith('max_') or key.startswith('target_') or key.startswith('prior_'):
                setattr(self, key, kwargs.get(key, default))

    def forward(
        self,
        params: CredalSetParams,
        p_star: torch.Tensor,
        entropy: torch.Tensor,
        params_clear: Optional[CredalSetParams] = None,
    ) -> Dict[str, torch.Tensor]:

        device = params.mu.device

        sigma_epi = params.sigma_epi
        sigma_ale = params.sigma_ale

        # -----------------------------------------------------------------
        # 1. Answer Loss + Prediction Error
        # -----------------------------------------------------------------
        answer_loss, pred_error = self._answer_loss_and_error(params.mu, p_star)

        # -----------------------------------------------------------------
        # 2. Epistemic Losses
        # -----------------------------------------------------------------
        # KL regularization
        kl_epi = self._kl_loss(params.mu, sigma_epi)

        # Residual error: error NOT explained by aleatoric
        # Use stop_gradient on sigma_ale to prevent entanglement!
        residual_error = pred_error - sigma_ale.detach()  # STOP GRADIENT
        residual_normalized = torch.sigmoid(residual_error * 3)
        target_epi = self.min_sigma + (self.max_sigma - self.min_sigma) * residual_normalized
        epi_residual_loss = F.mse_loss(sigma_epi, target_epi.detach())

        # -----------------------------------------------------------------
        # 3. Aleatoric Losses
        # -----------------------------------------------------------------
        # Direct entropy matching (strong supervision)
        ale_entropy_loss = F.mse_loss(sigma_ale, entropy)

        # Ranking: preserve entropy ordering
        ale_rank_loss = self._ranking_loss(sigma_ale, entropy)

        # -----------------------------------------------------------------
        # 4. Disentanglement Losses
        # -----------------------------------------------------------------
        # Soft decorrelation
        decorr_loss = self._decorrelation_loss(sigma_epi, sigma_ale)

        # Gradient isolation loss: penalize if σ_epi changes with entropy
        # (σ_epi should be invariant to aleatoric signal)
        gradient_isolation_loss = self._gradient_isolation_loss(sigma_epi, entropy)

        # -----------------------------------------------------------------
        # 5. Credal-Specific Losses
        # -----------------------------------------------------------------
        # Encourage meaningful credal width
        credal_width = params.credal_width
        width_loss = F.relu(self.min_credal_width - credal_width).mean()

        # Capacity constraint (total uncertainty bounded)
        total_sigma = params.sigma_upper
        capacity_loss = F.mse_loss(total_sigma,
                                    torch.full_like(total_sigma, self.target_total_sigma))

        # -----------------------------------------------------------------
        # Total Loss
        # -----------------------------------------------------------------
        total = (
            self.lambda_answer * answer_loss +
            self.lambda_kl_epi * kl_epi +
            self.lambda_epi_residual * epi_residual_loss +
            self.lambda_ale_entropy * ale_entropy_loss +
            self.lambda_ale_rank * ale_rank_loss +
            self.lambda_decorr * decorr_loss +
            self.lambda_gradient_isolation * gradient_isolation_loss +
            self.lambda_credal_width * width_loss +
            self.lambda_capacity * capacity_loss
        )

        # Logging
        with torch.no_grad():
            corr = self._correlation(sigma_epi, sigma_ale)

        return {
            'loss_total': total,
            'loss_answer': answer_loss,
            'loss_kl_epi': kl_epi,
            'loss_epi_residual': epi_residual_loss,
            'loss_ale_entropy': ale_entropy_loss,
            'loss_ale_rank': ale_rank_loss,
            'loss_decorr': decorr_loss,
            'loss_gradient_isolation': gradient_isolation_loss,
            'loss_credal_width': width_loss,
            'loss_capacity': capacity_loss,
            # Metrics
            'rho_eu_au': corr,
            'mean_credal_width': credal_width.mean(),
            'mean_sigma_lower': params.sigma_lower.mean(),
            'mean_sigma_upper': params.sigma_upper.mean(),
        }

    def _decorrelation_loss(self, sigma_epi, sigma_ale):
        """Penalize correlation between uncertainties."""
        epi_c = sigma_epi - sigma_epi.mean()
        ale_c = sigma_ale - sigma_ale.mean()
        corr = (epi_c * ale_c).mean() / (sigma_epi.std() * sigma_ale.std() + 1e-8)
        return corr ** 2  # Squared for stronger penalty at high |corr|

    def _gradient_isolation_loss(self, sigma_epi, entropy):
        """
        Penalize if σ_epi is correlated with entropy.

        σ_epi should capture model uncertainty, NOT data ambiguity.
        If σ_epi correlates with entropy, it's "leaking" aleatoric signal.
        """
        epi_c = sigma_epi - sigma_epi.mean()
        ent_c = entropy - entropy.mean()
        corr = (epi_c * ent_c).mean() / (sigma_epi.std() * entropy.std() + 1e-8)
        return corr ** 2

    def _correlation(self, x, y):
        x_c = x - x.mean()
        y_c = y - y.mean()
        return (x_c * y_c).mean() / (x.std() * y.std() + 1e-8)

    def _answer_loss_and_error(self, mu, p_star):
        B = mu.size(0)
        losses, errors = [], []
        for i in range(B):
            n = (p_star[i] > 0).sum().item()
            if n > 0:
                mu_i = mu[i, :n]
                p_i = p_star[i, :n]
                pred = F.softmax(mu_i, dim=-1)
                kl = F.kl_div(pred.log().clamp(min=-10), p_i, reduction='sum')
                losses.append(kl)
                errors.append(1.0 - (pred * p_i).sum())
            else:
                losses.append(torch.tensor(0.0, device=mu.device))
                errors.append(torch.tensor(0.5, device=mu.device))
        return torch.stack(losses).mean(), torch.stack(errors)

    def _kl_loss(self, mu, sigma_epi):
        prior_var = self.prior_sigma_epi ** 2
        if sigma_epi.dim() == 1:
            sigma_epi = sigma_epi.unsqueeze(-1)
        var_ratio = (sigma_epi ** 2) / prior_var
        mu_term = (mu ** 2) / prior_var
        kl = 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio + 1e-10))
        return kl.sum(dim=-1).mean()

    def _ranking_loss(self, sigma, target, num_pairs=64):
        B = sigma.size(0)
        if B < 2:
            return torch.tensor(0.0, device=sigma.device)
        losses = []
        for _ in range(min(num_pairs, B * (B - 1) // 2)):
            i, j = torch.randint(0, B, (2,)).tolist()
            if i == j:
                continue
            margin = 0.1
            if target[i] > target[j]:
                losses.append(F.relu(margin - (sigma[i] - sigma[j])))
            else:
                losses.append(F.relu(margin - (sigma[j] - sigma[i])))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma.device)


# ==============================================================================
# ADAPTER
# ==============================================================================

class MAQACredalLossV7bAdapter(nn.Module):
    """Adapter for existing training loop."""

    def __init__(self, loss_v7b: MAQACredalLossV7b):
        super().__init__()
        self.loss_v7b = loss_v7b

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        # Convert to CredalSetParams if needed
        if not isinstance(params, CredalSetParams):
            params = CredalSetParams(
                mu=params.mu,
                sigma_epi=params.sigma_epi,
                sigma_ale=params.sigma_ale,
            )

        losses = self.loss_v7b(params, p_star, entropy_gt, params_clear)

        loss_dict = {
            'loss_total': losses['loss_total'],
            'loss_kl': losses['loss_answer'],
            'loss_reg': losses['loss_kl_epi'],
            'loss_cal': losses['loss_ale_entropy'] + losses['loss_ale_rank'],
            'loss_cont': torch.tensor(0.0, device=losses['loss_total'].device),
            # V7b specific
            'loss_epi_residual': losses['loss_epi_residual'],
            'loss_ale_entropy': losses['loss_ale_entropy'],
            'loss_ale_rank': losses['loss_ale_rank'],
            'loss_decorr': losses['loss_decorr'],
            'loss_gradient_isolation': losses['loss_gradient_isolation'],
            'loss_credal_width': losses['loss_credal_width'],
            'loss_capacity': losses['loss_capacity'],
            'rho_eu_au': losses['rho_eu_au'],
            'mean_credal_width': losses['mean_credal_width'],
            'mean_sigma_lower': losses['mean_sigma_lower'],
            'mean_sigma_upper': losses['mean_sigma_upper'],
        }

        return losses['loss_total'], loss_dict


# ==============================================================================
# FACTORY
# ==============================================================================

def create_v7b_loss(**kwargs):
    """Create V7b loss."""
    config = {**CONFIG_V7B, **kwargs}
    return MAQACredalLossV7b(**config)


# ==============================================================================
# SUMMARY: HOW CREDAL FACILITY IS PRESERVED
# ==============================================================================

CREDAL_EXPLANATION = """
================================================================================
HOW V7B MAINTAINS CREDAL SET INTERPRETATION
================================================================================

A credal set C(x) represents IMPRECISE probability - a SET of distributions
rather than a single point estimate.

For Gaussian credal sets with our parameterization:

  C(x) = { N(μ, σ²) : σ_epi² ≤ σ² ≤ σ_epi² + σ_ale² }

This means:
  - LOWER bound: N(μ, σ_epi²)
    → "Best case" - only model uncertainty, no data ambiguity

  - UPPER bound: N(μ, σ_epi² + σ_ale²)
    → "Worst case" - model + data uncertainty combined

The WIDTH of the credal set = σ_upper - σ_lower measures IMPRECISION:
  - Wide set → high data ambiguity (aleatoric dominates)
  - Narrow set → low data ambiguity (epistemic dominates)

================================================================================
WHY DISENTANGLEMENT DOESN'T BREAK CREDAL INTERPRETATION
================================================================================

The credal set interpretation depends on:
  1. Having separate σ_epi and σ_ale
  2. σ_epi representing reducible (epistemic) uncertainty
  3. σ_ale representing irreducible (aleatoric) uncertainty

V7b satisfies all three:
  1. ✅ Separate heads for σ_epi and σ_ale
  2. ✅ σ_epi trained on residual error (what model got wrong)
  3. ✅ σ_ale trained on entropy H[p*] (inherent ambiguity)

The ARCHITECTURE (where signals come from) doesn't change the INTERPRETATION.
What matters is that the two uncertainties capture different aspects.

================================================================================
CONSTRAINTS AND CREDAL FACILITY
================================================================================

(i) Decorrelation: ρ(σ_epi, σ_ale) → 0
    Credal interpretation: The two uncertainty types are DISTINCT
    ✅ Still valid - ensures credal width is meaningful

(ii) σ_ale tracks entropy: ρ(σ_ale, H[p*]) → 1
    Credal interpretation: σ_ale captures inherent data ambiguity
    ✅ Still valid - this IS what aleatoric should capture

(iii) σ_epi tracks residual error: ρ(σ_epi, error|entropy) > 0
    Credal interpretation: σ_epi captures model-specific uncertainty
    ✅ Still valid - residual = what model got wrong BEYOND ambiguity

All three constraints STRENGTHEN the credal interpretation!
================================================================================
"""

SUMMARY = """
V7b Loss - Credal-Preserving Disentanglement
=============================================

Key Innovation:
  Maintains credal set interpretation while ensuring disentanglement.

Credal Structure:
  C(x) = {N(μ, σ²) : σ_epi² ≤ σ² ≤ σ_epi² + σ_ale²}

Disentanglement Mechanisms:
  1. σ_ale → entropy H[p*] (aleatoric = inherent ambiguity)
  2. σ_epi → residual error (epistemic = model error beyond ambiguity)
  3. Gradient isolation: stop_grad prevents entanglement
  4. Decorrelation penalty: ρ(σ_epi, σ_ale) → 0
  5. Capacity constraint: total uncertainty bounded

Expected Results:
  ρ(EU, AU) < 0.3        (decorrelation)
  ρ(AU, H[p*]) > 0.5    (aleatoric validity)
  ρ(EU, Error|H) > 0.3  (epistemic validity)
  Meaningful credal width (imprecision measure)

Key Config:
  lambda_ale_entropy: 2.0  (strong entropy supervision)
  lambda_epi_residual: 1.5 (residual error supervision)
  lambda_decorr: 5.0       (decorrelation strength)
"""

if __name__ == "__main__":
    print(CREDAL_EXPLANATION)
    print("\n" + SUMMARY)
    print("\nV7b Config:")
    for k, v in CONFIG_V7B.items():
        print(f"  {k}: {v}")
