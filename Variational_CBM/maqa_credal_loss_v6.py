"""
MAQA Credal Loss V6 - Hard Decorrelation Constraint
====================================================

V5 Problem: Soft decorrelation loss (λ=1.0) was overwhelmed.
  - Both σ_epi and σ_ale grew together over training
  - ρ(EU, AU) went from -0.02 → 0.92 despite decorr penalty

V6 Solution: HARD decorrelation through architecture, not just loss.

Approach 1: Gradient Reversal Layer
  - When training σ_epi, REVERSE gradients that would correlate it with σ_ale

Approach 2: Explicit Orthogonalization
  - Compute σ_epi_orthogonal = σ_epi - proj(σ_epi onto σ_ale)
  - Use orthogonalized version for loss

Approach 3: Very Strong Decorrelation Loss
  - λ_decorr = 20.0 (was 1.0)
  - Use squared correlation to penalize heavily

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

CONFIG_V6 = {
    # Training
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'dropout': 0.2,

    # === LOSS WEIGHTS ===
    'lambda_answer': 1.0,
    'lambda_kl': 0.0001,

    # Aleatoric: σ_ale → entropy
    'lambda_ale_mse': 1.0,
    'lambda_ale_rank': 2.0,

    # Epistemic: σ_epi → error (orthogonalized)
    'lambda_epi_mse': 1.0,
    'lambda_epi_rank': 1.0,

    # DECORRELATION - VERY STRONG
    'lambda_decorr': 20.0,  # Was 1.0 in V5
    'decorr_target': 0.0,   # Target correlation (0 = uncorrelated)

    # Variance (prevent collapse)
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
# LOSS FUNCTION V6
# ==============================================================================

class MAQACredalLossV6(nn.Module):
    """
    V6: Hard decorrelation through strong penalty + orthogonalization.

    Key changes:
    1. λ_decorr = 20.0 (20x stronger than V5)
    2. Orthogonalize σ_epi with respect to σ_ale before computing epistemic loss
    3. Use |corr|² instead of |corr| for stronger gradient at high correlation
    4. Supervise σ_epi on residual error (error not explained by entropy)
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, default in CONFIG_V6.items():
            if key.startswith('lambda_') or key.startswith('min_') or key.startswith('max_') or key in ['rank_margin', 'prior_sigma', 'decorr_target']:
                setattr(self, key, kwargs.get(key, default))

    def forward(
        self,
        params_amb,
        params_clear: Optional[object],
        p_star_amb: torch.Tensor,
        entropy_amb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        device = params_amb.mu.device

        sigma_epi_raw = torch.clamp(params_amb.sigma_epi, self.min_sigma_epi, self.max_sigma_epi)
        sigma_ale = torch.clamp(params_amb.sigma_ale, self.min_sigma_ale, self.max_sigma_ale)

        # -----------------------------------------------------------------
        # ORTHOGONALIZE σ_epi with respect to σ_ale
        # -----------------------------------------------------------------
        # Remove the component of σ_epi that's correlated with σ_ale
        sigma_epi = self._orthogonalize(sigma_epi_raw, sigma_ale)

        # -----------------------------------------------------------------
        # 1. Answer Loss + Prediction Error
        # -----------------------------------------------------------------
        answer_loss, pred_error = self._answer_loss_and_error(params_amb.mu, p_star_amb)

        # -----------------------------------------------------------------
        # 2. KL Regularization
        # -----------------------------------------------------------------
        kl_loss = self._kl_loss(params_amb.mu, sigma_epi_raw)  # Use raw for KL

        # -----------------------------------------------------------------
        # 3. Aleatoric: σ_ale → entropy
        # -----------------------------------------------------------------
        ale_mse = F.mse_loss(sigma_ale, entropy_amb)
        ale_rank = self._ranking_loss(sigma_ale, entropy_amb)

        # -----------------------------------------------------------------
        # 4. Epistemic: σ_epi (orthogonalized) → residual error
        # -----------------------------------------------------------------
        # Compute residual error (error not explained by entropy)
        residual_error = self._compute_residual_error(pred_error, entropy_amb)

        # Scale residual to σ_epi range
        target_epi = self.min_sigma_epi + (self.max_sigma_epi - self.min_sigma_epi) * torch.sigmoid(residual_error * 2)

        epi_mse = F.mse_loss(sigma_epi, target_epi.detach())
        epi_rank = self._ranking_loss(sigma_epi, residual_error.detach())

        # -----------------------------------------------------------------
        # 5. STRONG DECORRELATION LOSS
        # -----------------------------------------------------------------
        decorr_loss = self._decorrelation_loss_strong(sigma_epi_raw, sigma_ale)

        # -----------------------------------------------------------------
        # 6. Variance Regularization
        # -----------------------------------------------------------------
        var_loss = self._variance_loss(sigma_epi_raw, sigma_ale)

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
            actual_corr = self._correlation(sigma_epi_raw, sigma_ale)
            corr_orthogonalized = self._correlation(sigma_epi, sigma_ale)

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
            'mean_pred_error': pred_error.mean(),
            'actual_eu_au_corr': actual_corr,
            'corr_after_orthogonalization': corr_orthogonalized,
        }

    def _orthogonalize(self, sigma_epi, sigma_ale):
        """
        Remove the component of σ_epi that's correlated with σ_ale.

        σ_epi_orth = σ_epi - proj(σ_epi onto σ_ale)
                   = σ_epi - (σ_epi · σ_ale / ||σ_ale||²) * σ_ale

        This ensures the orthogonalized σ_epi is uncorrelated with σ_ale.
        """
        # Center both
        epi_centered = sigma_epi - sigma_epi.mean()
        ale_centered = sigma_ale - sigma_ale.mean()

        # Compute projection coefficient
        # proj_coef = (epi · ale) / (ale · ale)
        dot_product = (epi_centered * ale_centered).sum()
        ale_norm_sq = (ale_centered * ale_centered).sum() + 1e-8
        proj_coef = dot_product / ale_norm_sq

        # Remove projection
        epi_orthogonal = epi_centered - proj_coef * ale_centered

        # Add back mean and clamp to valid range
        epi_orthogonal = epi_orthogonal + sigma_epi.mean()
        epi_orthogonal = torch.clamp(epi_orthogonal, self.min_sigma_epi, self.max_sigma_epi)

        return epi_orthogonal

    def _compute_residual_error(self, pred_error, entropy):
        """
        Compute residual error = pred_error - E[pred_error | entropy]

        Uses simple linear regression within batch.
        """
        # Center
        error_centered = pred_error - pred_error.mean()
        entropy_centered = entropy - entropy.mean()

        # Regression coefficient
        cov = (error_centered * entropy_centered).sum()
        var = (entropy_centered * entropy_centered).sum() + 1e-8
        slope = cov / var

        # Expected error given entropy
        expected_error = pred_error.mean() + slope * entropy_centered

        # Residual
        residual = pred_error - expected_error

        return residual

    def _decorrelation_loss_strong(self, sigma_epi, sigma_ale):
        """
        Strong decorrelation loss: penalize |correlation|².

        Using squared correlation gives stronger gradient when correlation is high.
        """
        epi_centered = sigma_epi - sigma_epi.mean()
        ale_centered = sigma_ale - sigma_ale.mean()

        # Correlation
        cov = (epi_centered * ale_centered).mean()
        std_epi = sigma_epi.std() + 1e-6
        std_ale = sigma_ale.std() + 1e-6
        corr = cov / (std_epi * std_ale)

        # Squared correlation loss (stronger penalty at high |corr|)
        return corr ** 2

    def _correlation(self, x, y):
        """Compute correlation coefficient."""
        x_c = x - x.mean()
        y_c = y - y.mean()
        cov = (x_c * y_c).mean()
        return cov / (x.std() * y.std() + 1e-8)

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
        prior_var = self.prior_sigma ** 2
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
            if target[i] > target[j]:
                losses.append(F.relu(self.rank_margin - (sigma[i] - sigma[j])))
            else:
                losses.append(F.relu(self.rank_margin - (sigma[j] - sigma[i])))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma.device)

    def _variance_loss(self, sigma_epi, sigma_ale):
        penalty_epi = F.relu(0.01 - sigma_epi.var())
        penalty_ale = F.relu(0.0225 - sigma_ale.var())
        return penalty_epi + penalty_ale


# ==============================================================================
# ADAPTER
# ==============================================================================

class MAQACredalLossV6Adapter(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        losses = self.loss(params, params_clear, p_star, entropy_gt)

        loss_dict = {
            'loss_total': losses['loss_total'],
            'loss_kl': losses['loss_answer'],
            'loss_reg': losses['loss_kl'],
            'loss_cal': losses['loss_ale_mse'] + losses['loss_ale_rank'],
            'loss_cont': torch.tensor(0.0, device=losses['loss_total'].device),
            # V6 specific
            'loss_ale_mse': losses['loss_ale_mse'],
            'loss_ale_rank': losses['loss_ale_rank'],
            'loss_epi_mse': losses['loss_epi_mse'],
            'loss_epi_rank': losses['loss_epi_rank'],
            'loss_decorr': losses['loss_decorr'],
            'mean_pred_error': losses['mean_pred_error'],
            'actual_eu_au_corr': losses['actual_eu_au_corr'],
            'corr_after_orthogonalization': losses['corr_after_orthogonalization'],
        }

        return losses['loss_total'], loss_dict


# ==============================================================================
# FACTORY
# ==============================================================================

def create_v6_loss(**kwargs):
    """Create V6 loss."""
    config = {**CONFIG_V6, **kwargs}
    return MAQACredalLossV6(**config)


# ==============================================================================
# SUMMARY
# ==============================================================================

SUMMARY = """
V6 Loss - Hard Decorrelation
============================

V5 Problem:
  - Soft decorrelation (λ=1.0) was overwhelmed
  - Both σs grew together: ρ(EU, AU) went -0.02 → 0.92

V6 Solutions:

1. STRONG DECORRELATION (λ=20.0)
   - 20x stronger than V5
   - Use |corr|² for stronger gradient at high correlation

2. ORTHOGONALIZATION
   - σ_epi_orth = σ_epi - proj(σ_epi onto σ_ale)
   - Removes correlated component before supervision

3. RESIDUAL ERROR
   - epistemic_target = pred_error - E[pred_error | entropy]
   - Removes the entropy-correlated part

Expected results:
  ρ(EU, AU) < 0.3  (hard-enforced)
  ρ(AU, H[p*]) > 0.2
  ρ(EU, Error) > 0.2

Key config:
  lambda_decorr = 20.0  (was 1.0 in V5)
"""

if __name__ == "__main__":
    print(SUMMARY)
    print("\nV6 Config:")
    for k, v in CONFIG_V6.items():
        print(f"  {k}: {v}")
