"""
MAQA Credal Loss V4 - Clean Prediction Error Supervision
=========================================================

The cleanest approach:

  Aleatoric:  σ_ale → H[p*]          (ground-truth entropy)
  Epistemic:  σ_epi → pred_error     (1 - P(correct))

Both use MSE + Ranking losses for robust supervision.

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

CONFIG_V4 = {
    # Model
    'encoder_name': 'microsoft/deberta-v3-base',
    'freeze_encoder': True,

    # Training
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,

    # === LOSS WEIGHTS ===
    'lambda_answer': 1.0,       # KL(pred || p*)
    'lambda_kl': 0.0001,        # KL regularization (very weak)

    # Aleatoric: σ_ale → entropy
    'lambda_ale_mse': 1.0,
    'lambda_ale_rank': 3.0,

    # Epistemic: σ_epi → pred_error
    'lambda_epi_mse': 2.0,
    'lambda_epi_rank': 2.0,

    # Regularization
    'lambda_variance': 0.5,
    'lambda_contrast': 0.5,

    # Ranking margin
    'rank_margin': 0.1,

    # === BOUNDS ===
    'min_sigma_epi': 0.05,
    'max_sigma_epi': 1.0,
    'prior_sigma': 0.5,
    'min_sigma_ale': 0.1,
    'max_sigma_ale': 2.0,

    'dropout': 0.2,
}


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

class MAQACredalLossV4(nn.Module):
    """
    Clean loss with explicit supervision for both uncertainties.

    Aleatoric (σ_ale):
        target = H[p*]
        L = MSE(σ_ale, target) + Ranking(σ_ale, target)

    Epistemic (σ_epi):
        pred_error = 1 - Σ softmax(μ) × p*
        target = scale(pred_error) to [min_σ, max_σ]
        L = MSE(σ_epi, target.detach()) + Ranking(σ_epi, pred_error.detach())
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Set defaults from CONFIG_V4, override with kwargs
        for key, default in CONFIG_V4.items():
            if key.startswith('lambda_') or key in ['rank_margin', 'min_sigma_epi',
                'max_sigma_epi', 'prior_sigma', 'min_sigma_ale', 'max_sigma_ale']:
                setattr(self, key, kwargs.get(key, default))

    def forward(
        self,
        params_amb,
        params_clear: Optional[object],
        p_star_amb: torch.Tensor,
        entropy_amb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        device = params_amb.mu.device

        # Clamp uncertainties
        sigma_epi = torch.clamp(params_amb.sigma_epi, self.min_sigma_epi, self.max_sigma_epi)
        sigma_ale = torch.clamp(params_amb.sigma_ale, self.min_sigma_ale, self.max_sigma_ale)

        # -----------------------------------------------------------------
        # 1. Answer Loss + Prediction Error
        # -----------------------------------------------------------------
        answer_loss, pred_error = self._answer_loss_and_error(params_amb.mu, p_star_amb)

        # -----------------------------------------------------------------
        # 2. KL Regularization (weak)
        # -----------------------------------------------------------------
        kl_loss = self._kl_loss(params_amb.mu, sigma_epi)

        # -----------------------------------------------------------------
        # 3. Aleatoric Calibration: σ_ale → H[p*]
        # -----------------------------------------------------------------
        ale_mse = F.mse_loss(sigma_ale, entropy_amb)
        ale_rank = self._ranking_loss(sigma_ale, entropy_amb)

        # -----------------------------------------------------------------
        # 4. Epistemic Calibration: σ_epi → pred_error (THE KEY PART)
        # -----------------------------------------------------------------
        # Scale pred_error [0,1] → σ_epi range [min, max]
        target_epi = self.min_sigma_epi + (self.max_sigma_epi - self.min_sigma_epi) * pred_error

        # .detach() is CRITICAL: σ_epi predicts error, doesn't minimize it
        epi_mse = F.mse_loss(sigma_epi, target_epi.detach())
        epi_rank = self._ranking_loss(sigma_epi, pred_error.detach())

        # -----------------------------------------------------------------
        # 5. Variance Regularization
        # -----------------------------------------------------------------
        var_loss = self._variance_loss(sigma_epi, sigma_ale)

        # -----------------------------------------------------------------
        # 6. Contrastive (optional)
        # -----------------------------------------------------------------
        if params_clear is not None:
            sigma_ale_clear = torch.clamp(params_clear.sigma_ale, self.min_sigma_ale, self.max_sigma_ale)
            contrast = F.relu(sigma_ale_clear - sigma_ale + self.rank_margin).mean()
        else:
            contrast = torch.tensor(0.0, device=device)

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
            self.lambda_variance * var_loss +
            self.lambda_contrast * contrast
        )

        return {
            'loss_total': total,
            'loss_answer': answer_loss,
            'loss_kl': kl_loss,
            'loss_ale_mse': ale_mse,
            'loss_ale_rank': ale_rank,
            'loss_epi_mse': epi_mse,
            'loss_epi_rank': epi_rank,
            'loss_variance': var_loss,
            'loss_contrast': contrast,
            'mean_pred_error': pred_error.mean(),
            'mean_target_epi': target_epi.mean(),
        }

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

                # KL loss
                kl = F.kl_div(pred.log().clamp(min=-10), p_i, reduction='sum')
                losses.append(kl)

                # Prediction error = 1 - P(correct)
                prob_correct = (pred * p_i).sum()
                errors.append(1.0 - prob_correct)
            else:
                losses.append(torch.tensor(0.0, device=mu.device))
                errors.append(torch.tensor(0.5, device=mu.device))

        return torch.stack(losses).mean(), torch.stack(errors)

    def _kl_loss(self, mu, sigma_epi):
        """KL to prior (very weak)."""
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
        penalty_epi = F.relu(0.01 - sigma_epi.var())  # target std > 0.1
        penalty_ale = F.relu(0.0225 - sigma_ale.var())  # target std > 0.15
        return penalty_epi + penalty_ale


# ==============================================================================
# ADAPTER
# ==============================================================================

class MAQACredalLossV4Adapter(nn.Module):
    def __init__(self, loss_v4):
        super().__init__()
        self.loss_v4 = loss_v4

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        losses = self.loss_v4(params, params_clear, p_star, entropy_gt)

        return losses['loss_total'], {
            'loss_total': losses['loss_total'],
            'loss_kl': losses['loss_answer'],
            'loss_reg': losses['loss_kl'],
            'loss_cal': losses['loss_ale_mse'] + losses['loss_ale_rank'],
            'loss_cont': losses['loss_contrast'],
            'loss_epi_mse': losses['loss_epi_mse'],
            'loss_epi_rank': losses['loss_epi_rank'],
            'mean_pred_error': losses['mean_pred_error'],
        }


# ==============================================================================
# HELPERS
# ==============================================================================

def create_v4_loss():
    """Create V4 loss with default config."""
    return MAQACredalLossV4(**CONFIG_V4)


def initialize_model_for_v4(model):
    """Initialize σ heads for better starting point."""
    def init_bias(target):
        return float(np.log(np.exp(target) - 1 + 1e-6))

    if hasattr(model, 'sigma_epi_head'):
        for layer in reversed(list(model.sigma_epi_head.modules())):
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, init_bias(0.3))
                print(f"✓ σ_epi head initialized → ~0.3")
                break

    if hasattr(model, 'sigma_ale_head'):
        for layer in reversed(list(model.sigma_ale_head.modules())):
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, init_bias(0.5))
                print(f"✓ σ_ale head initialized → ~0.5")
                break

    return model


# ==============================================================================
# SUMMARY
# ==============================================================================

SUMMARY = """
V4 Loss - Clean Supervision
===========================

Aleatoric (σ_ale):
  Target: H[p*] (ground-truth entropy)
  Loss: MSE + Ranking

Epistemic (σ_epi):
  Target: pred_error = 1 - Σ softmax(μ) × p*
  Loss: MSE + Ranking (with .detach()!)

Expected results:
  ρ(EU, AU) < 0.1      (decorrelation)
  ρ(AU, H[p*]) > 0.4   (aleatoric validity)
  ρ(EU, Error) > 0.3   (epistemic validity) ← NEW!
"""

if __name__ == "__main__":
    print(SUMMARY)
    print("\nConfig:")
    for k, v in CONFIG_V4.items():
        print(f"  {k}: {v}")
