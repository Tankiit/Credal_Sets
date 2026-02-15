"""
MAQA Credal Loss V5 - Orthogonalized Epistemic Supervision
==========================================================

V4 Problem: pred_error and H[p*] are correlated!
  - High entropy → hard to predict → high error
  - So σ_epi and σ_ale end up tracking the same signal
  - Result: ρ(EU, AU) can be high (violates Theorem 1)

V5 Solution: Add explicit decorrelation loss

  Option A (Complex): Orthogonalize signals
    epistemic_signal = pred_error - E[pred_error | entropy]

  Option B (Simple): Direct decorrelation penalty [RECOMMENDED]
    loss_decorr = |correlation(σ_epi, σ_ale)|

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG_V5 = {
    # Training
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'dropout': 0.2,

    # === LOSS WEIGHTS ===
    'lambda_answer': 1.0,
    'lambda_kl': 0.0001,

    # Aleatoric: σ_ale → entropy
    'lambda_ale_mse': 1.0,
    'lambda_ale_rank': 3.0,

    # Epistemic: σ_epi → pred_error
    'lambda_epi_mse': 2.0,
    'lambda_epi_rank': 2.0,

    # DECORRELATION (the key addition!)
    'lambda_decorr': 2.0,  # High to enforce ρ(EU, AU) → 0

    # Regularization
    'lambda_variance': 0.5,

    # Ranking margin
    'rank_margin': 0.1,

    # Bounds
    'min_sigma_epi': 0.05,
    'max_sigma_epi': 1.0,
    'prior_sigma': 0.5,
    'min_sigma_ale': 0.1,
    'max_sigma_ale': 2.0,
}


# ==============================================================================
# LOSS FUNCTION V5 (SIMPLE VERSION - RECOMMENDED)
# ==============================================================================

class MAQACredalLossV5(nn.Module):
    """
    V5: V4 + Explicit Decorrelation Loss

    Key addition: Penalize correlation between σ_epi and σ_ale directly.

    This is simpler and more robust than trying to orthogonalize the signals.
    Let the optimization figure out how to balance:
      - σ_epi tracking pred_error
      - σ_ale tracking entropy
      - Maintaining low correlation
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, default in CONFIG_V5.items():
            if key.startswith('lambda_') or key.startswith('min_') or key.startswith('max_') or key in ['rank_margin', 'prior_sigma']:
                setattr(self, key, kwargs.get(key, default))

    def forward(
        self,
        params_amb,
        params_clear: Optional[object],
        p_star_amb: torch.Tensor,
        entropy_amb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        device = params_amb.mu.device

        sigma_epi = torch.clamp(params_amb.sigma_epi, self.min_sigma_epi, self.max_sigma_epi)
        sigma_ale = torch.clamp(params_amb.sigma_ale, self.min_sigma_ale, self.max_sigma_ale)

        # -----------------------------------------------------------------
        # 1. Answer Loss + Prediction Error
        # -----------------------------------------------------------------
        answer_loss, pred_error = self._answer_loss_and_error(params_amb.mu, p_star_amb)

        # -----------------------------------------------------------------
        # 2. KL Regularization
        # -----------------------------------------------------------------
        kl_loss = self._kl_loss(params_amb.mu, sigma_epi)

        # -----------------------------------------------------------------
        # 3. Aleatoric → Entropy
        # -----------------------------------------------------------------
        ale_mse = F.mse_loss(sigma_ale, entropy_amb)
        ale_rank = self._ranking_loss(sigma_ale, entropy_amb)

        # -----------------------------------------------------------------
        # 4. Epistemic → Pred Error (same as V4)
        # -----------------------------------------------------------------
        target_epi = self.min_sigma_epi + (self.max_sigma_epi - self.min_sigma_epi) * pred_error
        epi_mse = F.mse_loss(sigma_epi, target_epi.detach())
        epi_rank = self._ranking_loss(sigma_epi, pred_error.detach())

        # -----------------------------------------------------------------
        # 5. DECORRELATION LOSS (THE KEY ADDITION!)
        # -----------------------------------------------------------------
        decorr_loss = self._decorrelation_loss(sigma_epi, sigma_ale)

        # -----------------------------------------------------------------
        # 6. Variance Regularization
        # -----------------------------------------------------------------
        var_loss = self._variance_loss(sigma_epi, sigma_ale)

        # -----------------------------------------------------------------
        # Total
        # -----------------------------------------------------------------
        total = (
            self.lambda_answer * answer_loss +
            self.lambda_kl * kl_loss +
            self.lambda_ale_mse * ale_mse +
            self.lambda_ale_rank * ale_rank +
            self.lambda_epi_mse * epi_mse +
            self.lambda_epi_rank * epi_rank +
            self.lambda_decorr * decorr_loss +
            self.lambda_variance * var_loss
        )

        # Compute actual correlation for logging
        with torch.no_grad():
            actual_corr = self._correlation(sigma_epi, sigma_ale)
            error_entropy_corr = self._correlation(pred_error, entropy_amb)

        return {
            'loss_total': total,
            'loss_answer': answer_loss,
            'loss_kl': kl_loss,
            'loss_ale_mse': ale_mse,
            'loss_ale_rank': ale_rank,
            'loss_epi_mse': epi_mse,
            'loss_epi_rank': epi_rank,
            'loss_decorr': decorr_loss,
            'loss_variance': var_loss,
            # Diagnostics
            'mean_pred_error': pred_error.mean(),
            'actual_eu_au_corr': actual_corr,
            'error_entropy_corr': error_entropy_corr,
        }

    def _decorrelation_loss(self, sigma_epi, sigma_ale):
        """
        Penalize correlation between EU and AU.

        Uses absolute correlation: |corr|
        """
        # Center
        epi_centered = sigma_epi - sigma_epi.mean()
        ale_centered = sigma_ale - sigma_ale.mean()

        # Correlation
        cov = (epi_centered * ale_centered).mean()
        std_epi = sigma_epi.std() + 1e-6
        std_ale = sigma_ale.std() + 1e-6
        corr = cov / (std_epi * std_ale)

        # Penalize absolute correlation
        return corr.abs()

    def _correlation(self, x, y):
        """Compute Pearson correlation."""
        x_c = x - x.mean()
        y_c = y - y.mean()
        cov = (x_c * y_c).mean()
        return cov / (x.std() * y.std() + 1e-6)

    def _answer_loss_and_error(self, mu, p_star):
        """Compute KL loss and prediction error."""
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
        """KL to prior."""
        prior_var = self.prior_sigma ** 2
        if sigma_epi.dim() == 1:
            sigma_epi = sigma_epi.unsqueeze(-1)
        var_ratio = (sigma_epi ** 2) / prior_var
        mu_term = (mu ** 2) / prior_var
        kl = 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio + 1e-10))
        return kl.sum(dim=-1).mean()

    def _ranking_loss(self, sigma, target, num_pairs=64):
        """Enforce σ preserves ordering of target."""
        B = sigma.size(0)
        if B < 2:
            return torch.tensor(0.0, device=sigma.device)

        losses = []
        for _ in range(min(num_pairs, B * (B - 1) // 2)):
            i, j = torch.randint(0, B, (2,)).tolist()
            if i == j:
                continue
            if target[i] > target[j]:
                losses.append(F.relu(self.rank_margin - (sigma[i] - sigma[j])))
            else:
                losses.append(F.relu(self.rank_margin - (sigma[j] - sigma[i])))

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma.device)

    def _variance_loss(self, sigma_epi, sigma_ale):
        """Penalize low variance."""
        penalty_epi = F.relu(0.01 - sigma_epi.var())
        penalty_ale = F.relu(0.0225 - sigma_ale.var())
        return penalty_epi + penalty_ale


# ==============================================================================
# ADAPTER
# ==============================================================================

class MAQACredalLossV5Adapter(nn.Module):
    def __init__(self, loss_v5):
        super().__init__()
        self.loss_v5 = loss_v5

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        losses = self.loss_v5(params, params_clear, p_star, entropy_gt)

        loss_dict = {
            'loss_total': losses['loss_total'],
            'loss_kl': losses['loss_answer'],  # Trainer expects this
            'loss_reg': losses['loss_kl'],
            'loss_cal': losses['loss_ale_mse'] + losses['loss_ale_rank'],
            'loss_cont': losses.get('loss_contrast', torch.tensor(0.0, device=losses['loss_total'].device)),
            # V5 specific
            'loss_ale_mse': losses['loss_ale_mse'],
            'loss_ale_rank': losses['loss_ale_rank'],
            'loss_epi_mse': losses['loss_epi_mse'],
            'loss_epi_rank': losses['loss_epi_rank'],
            'loss_decorr': losses['loss_decorr'],
            'mean_pred_error': losses['mean_pred_error'],
            'actual_eu_au_corr': losses['actual_eu_au_corr'],
        }

        return losses['loss_total'], loss_dict


# ==============================================================================
# HELPERS
# ==============================================================================

def create_v5_loss():
    """Create V5 loss with default config."""
    return MAQACredalLossV5(**CONFIG_V5)


# ==============================================================================
# SUMMARY
# ==============================================================================

SUMMARY = """
V5 Loss - V4 + Decorrelation
=============================

V4 Problem:
  pred_error and entropy are correlated (ρ ≈ 0.5-0.7)
  Both σ_epi and σ_ale track correlated signals
  Result: ρ(EU, AU) can be high

V5 Solution:
  Keep V4's supervision (it works!)
  Add explicit decorrelation penalty

  loss_decorr = |correlation(σ_epi, σ_ale)|

This enforces:
  1. σ_epi → pred_error (epistemic validity)
  2. σ_ale → entropy (aleatoric validity)
  3. ρ(EU, AU) → 0 (decorrelation)

Expected Results:
  ρ(EU, AU) < 0.3        ✓ (enforced by decorr loss)
  ρ(AU, Entropy) > 0.3   ✓ (from ale supervision)
  ρ(EU, Error) > 0.2     ✓ (from epi supervision)

Key Parameter:
  lambda_decorr = 2.0  # High enough to enforce decorrelation
"""

if __name__ == "__main__":
    print(SUMMARY)
    print("\nV5 Configuration:")
    for k, v in CONFIG_V5.items():
        print(f"  {k}: {v}")
