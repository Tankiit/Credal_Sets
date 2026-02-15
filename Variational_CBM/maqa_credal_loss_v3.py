"""
MAQA Credal Loss V3 - Fixing Collapsed Uncertainties
=====================================================

Problems identified:
1. σ_ale collapsed to constant (~0.757) - not tracking entropy
2. σ_epi collapsed to prior (~0.089) - not tracking errors
3. MSE calibration loss allows "mean-seeking" behavior

Solutions:
1. RANKING LOSS for aleatoric: high-entropy examples should have higher σ_ale
2. ERROR-AWARE LOSS for epistemic: hard examples should have higher σ_epi  
3. VARIANCE REGULARIZATION: prevent collapse to constants
4. CONTRASTIVE PAIRS: use paired data more effectively

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION V3
# ==============================================================================

CONFIG_V3 = {
    # Model
    'encoder_name': 'microsoft/deberta-v3-base',
    'hidden_dim': 256,
    'freeze_encoder': True,

    # Training
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,

    # === KEY CHANGES ===
    
    # Loss weights - REBALANCED
    'lambda_answer': 1.0,       # Answer prediction
    'lambda_kl': 0.001,         # DECREASED - was pulling σ_epi to prior too hard
    'lambda_cal_mse': 0.1,      # DECREASED MSE component
    'lambda_cal_rank': 5.0,     # NEW: Ranking loss (high entropy → high σ_ale)
    'lambda_epi_error': 2.0,    # NEW: Error-aware epistemic loss
    'lambda_variance': 1.0,     # NEW: Variance regularization
    'lambda_contrast': 0.5,     # Contrastive (ambig vs clear)
    
    # Ranking loss
    'rank_margin': 0.1,         # Margin for ranking loss
    'rank_pairs_per_batch': 32, # Number of ranking pairs to sample
    
    # Epistemic prior - RELAXED
    'prior_sigma': 0.3,         # INCREASED from 0.1 (allow more variation)
    'min_sigma_epi': 0.05,
    'max_sigma_epi': 1.0,       # DECREASED max to prevent explosion
    
    # Aleatoric bounds
    'min_sigma_ale': 0.1,       # NEW: Floor
    'max_sigma_ale': 2.0,       # NEW: Ceiling (should cover entropy range)
    
    # Initialization
    'init_sigma_epi': 0.2,      # Initialize σ_epi head to output ~0.2
    'init_sigma_ale': 0.5,      # Initialize σ_ale head to output ~0.5
    
    # Dropout
    'dropout': 0.2,
}


# ==============================================================================
# LOSS FUNCTION V3
# ==============================================================================

class MAQACredalLossV3(nn.Module):
    """
    Improved loss function that prevents uncertainty collapse.
    
    Key innovations:
    1. Ranking loss for aleatoric calibration
    2. Error-aware epistemic supervision
    3. Variance regularization to prevent collapse
    """
    
    def __init__(
        self,
        lambda_answer: float = 1.0,
        lambda_kl: float = 0.001,
        lambda_cal_mse: float = 0.1,
        lambda_cal_rank: float = 5.0,
        lambda_epi_error: float = 2.0,
        lambda_variance: float = 1.0,
        lambda_contrast: float = 0.5,
        rank_margin: float = 0.1,
        prior_sigma: float = 0.3,
        min_sigma_epi: float = 0.05,
        max_sigma_epi: float = 1.0,
        min_sigma_ale: float = 0.1,
        max_sigma_ale: float = 2.0,
    ):
        super().__init__()
        self.lambda_answer = lambda_answer
        self.lambda_kl = lambda_kl
        self.lambda_cal_mse = lambda_cal_mse
        self.lambda_cal_rank = lambda_cal_rank
        self.lambda_epi_error = lambda_epi_error
        self.lambda_variance = lambda_variance
        self.lambda_contrast = lambda_contrast
        self.rank_margin = rank_margin
        self.prior_sigma = prior_sigma
        self.min_sigma_epi = min_sigma_epi
        self.max_sigma_epi = max_sigma_epi
        self.min_sigma_ale = min_sigma_ale
        self.max_sigma_ale = max_sigma_ale
        
    def forward(
        self,
        params_amb: 'CredalQAParameters',
        params_clear: Optional['CredalQAParameters'],
        p_star_amb: torch.Tensor,
        entropy_amb: torch.Tensor,
        return_components: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            params_amb: Credal parameters for ambiguous questions
            params_clear: Credal parameters for clear questions (can be None)
            p_star_amb: Ground-truth distribution [B, num_answers]
            entropy_amb: Ground-truth entropy H[p*] [B]
            
        Returns:
            Dictionary with loss components
        """
        device = params_amb.mu.device
        batch_size = params_amb.mu.size(0)
        
        # Clamp uncertainties to valid ranges
        sigma_epi = torch.clamp(params_amb.sigma_epi, self.min_sigma_epi, self.max_sigma_epi)
        sigma_ale = torch.clamp(params_amb.sigma_ale, self.min_sigma_ale, self.max_sigma_ale)
        
        # =====================================================================
        # LOSS 1: Answer Prediction (KL to p*)
        # =====================================================================
        answer_loss = self._compute_answer_loss(params_amb.mu, p_star_amb)
        
        # =====================================================================
        # LOSS 2: KL Regularization (WEAKENED)
        # =====================================================================
        kl_loss = self._compute_kl_loss(params_amb.mu, sigma_epi)
        
        # =====================================================================
        # LOSS 3: Aleatoric Calibration - MSE (WEAKENED)
        # =====================================================================
        # Scale entropy to match σ_ale range
        # If entropy is [0, 2], and we want σ_ale in [0.1, 2.0], use linear scaling
        entropy_scaled = entropy_amb  # Already in reasonable range
        cal_mse_loss = F.mse_loss(sigma_ale, entropy_scaled)
        
        # =====================================================================
        # LOSS 4: Aleatoric Calibration - RANKING (NEW!)
        # =====================================================================
        # This is the KEY fix: enforce that high-entropy examples should have higher σ_ale
        cal_rank_loss = self._compute_ranking_loss(sigma_ale, entropy_amb)
        
        # =====================================================================
        # LOSS 5: Epistemic Error-Awareness (NEW!)
        # =====================================================================
        # Compute prediction error (soft version using KL)
        pred_error = self._compute_prediction_error(params_amb.mu, p_star_amb)
        # σ_epi should be higher when prediction error is higher
        epi_error_loss = self._compute_error_awareness_loss(sigma_epi, pred_error)
        
        # =====================================================================
        # LOSS 6: Variance Regularization (NEW!)
        # =====================================================================
        # Prevent collapse to constant by encouraging variance in outputs
        var_loss = self._compute_variance_regularization(sigma_epi, sigma_ale)
        
        # =====================================================================
        # LOSS 7: Contrastive (ambig σ_ale > clear σ_ale)
        # =====================================================================
        if params_clear is not None:
            sigma_ale_clear = torch.clamp(params_clear.sigma_ale, self.min_sigma_ale, self.max_sigma_ale)
            contrast_loss = F.relu(sigma_ale_clear - sigma_ale + self.rank_margin).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total_loss = (
            self.lambda_answer * answer_loss +
            self.lambda_kl * kl_loss +
            self.lambda_cal_mse * cal_mse_loss +
            self.lambda_cal_rank * cal_rank_loss +
            self.lambda_epi_error * epi_error_loss +
            self.lambda_variance * var_loss +
            self.lambda_contrast * contrast_loss
        )
        
        return {
            'loss_total': total_loss,
            'loss_answer': answer_loss,
            'loss_kl': kl_loss,
            'loss_cal_mse': cal_mse_loss,
            'loss_cal_rank': cal_rank_loss,
            'loss_epi_error': epi_error_loss,
            'loss_variance': var_loss,
            'loss_contrast': contrast_loss,
        }
    
    def _compute_answer_loss(self, mu: torch.Tensor, p_star: torch.Tensor) -> torch.Tensor:
        """KL divergence between predicted and ground-truth distributions."""
        batch_size = mu.size(0)
        losses = []
        
        for i in range(batch_size):
            num_answers = (p_star[i] > 0).sum().item()
            if num_answers > 0:
                mu_trunc = mu[i, :num_answers]
                p_star_trunc = p_star[i, :num_answers]
                
                # Softmax over truncated logits for proper distribution
                pred_dist = F.softmax(mu_trunc, dim=-1)
                
                # KL(p* || pred)
                kl = F.kl_div(
                    pred_dist.log().clamp(min=-10),
                    p_star_trunc,
                    reduction='sum'
                )
                losses.append(kl)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=mu.device)
    
    def _compute_kl_loss(self, mu: torch.Tensor, sigma_epi: torch.Tensor) -> torch.Tensor:
        """KL(N(μ, σ²_epi) || N(0, σ²_prior)) - regularization."""
        prior_var = self.prior_sigma ** 2
        
        if sigma_epi.dim() == 1:
            sigma_epi = sigma_epi.unsqueeze(-1)
        
        var_ratio = (sigma_epi ** 2) / prior_var
        mu_term = (mu ** 2) / prior_var
        
        kl_per_dim = 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio + 1e-10))
        
        return kl_per_dim.sum(dim=-1).mean()
    
    def _compute_ranking_loss(self, sigma_ale: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """
        Ranking loss: for any pair (i, j), if entropy[i] > entropy[j], 
        then σ_ale[i] should be > σ_ale[j].
        
        This is the KEY innovation to fix collapsed aleatoric uncertainty!
        """
        batch_size = sigma_ale.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=sigma_ale.device)
        
        # Create all pairs
        # For efficiency, we sample random pairs instead of all O(n²) pairs
        num_pairs = min(batch_size * (batch_size - 1) // 2, 64)
        
        losses = []
        for _ in range(num_pairs):
            i, j = torch.randint(0, batch_size, (2,))
            if i == j:
                continue
                
            # If entropy[i] > entropy[j], we want σ_ale[i] > σ_ale[j]
            if entropy[i] > entropy[j]:
                # Margin ranking: σ_ale[i] - σ_ale[j] > margin
                loss = F.relu(self.rank_margin - (sigma_ale[i] - sigma_ale[j]))
            else:
                # entropy[j] >= entropy[i], so σ_ale[j] >= σ_ale[i]
                loss = F.relu(self.rank_margin - (sigma_ale[j] - sigma_ale[i]))
            
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma_ale.device)
    
    def _compute_prediction_error(self, mu: torch.Tensor, p_star: torch.Tensor) -> torch.Tensor:
        """Compute soft prediction error for each example."""
        batch_size = mu.size(0)
        errors = []
        
        for i in range(batch_size):
            num_answers = (p_star[i] > 0).sum().item()
            if num_answers > 0:
                mu_trunc = mu[i, :num_answers]
                p_star_trunc = p_star[i, :num_answers]
                
                pred_dist = F.softmax(mu_trunc, dim=-1)
                
                # Error = 1 - probability assigned to correct answer
                # For multi-answer, use expected probability under p*
                correct_prob = (pred_dist * p_star_trunc).sum()
                error = 1 - correct_prob
                errors.append(error)
            else:
                errors.append(torch.tensor(0.5, device=mu.device))
        
        return torch.stack(errors)
    
    def _compute_error_awareness_loss(self, sigma_epi: torch.Tensor, pred_error: torch.Tensor) -> torch.Tensor:
        """
        Epistemic uncertainty should correlate with prediction error.
        High error → high σ_epi, Low error → low σ_epi.
        
        Uses ranking loss similar to aleatoric calibration.
        """
        batch_size = sigma_epi.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=sigma_epi.device)
        
        num_pairs = min(batch_size * (batch_size - 1) // 2, 64)
        
        losses = []
        for _ in range(num_pairs):
            i, j = torch.randint(0, batch_size, (2,))
            if i == j:
                continue
            
            # If error[i] > error[j], we want σ_epi[i] > σ_epi[j]
            if pred_error[i] > pred_error[j]:
                loss = F.relu(self.rank_margin - (sigma_epi[i] - sigma_epi[j]))
            else:
                loss = F.relu(self.rank_margin - (sigma_epi[j] - sigma_epi[i]))
            
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma_epi.device)
    
    def _compute_variance_regularization(self, sigma_epi: torch.Tensor, sigma_ale: torch.Tensor) -> torch.Tensor:
        """
        Encourage variance in uncertainty outputs to prevent collapse.
        
        If std(σ) is too low, add penalty.
        Target: std(σ_epi) > 0.1, std(σ_ale) > 0.2
        """
        # Compute batch variance
        var_epi = sigma_epi.var()
        var_ale = sigma_ale.var()
        
        # Penalize low variance
        target_std_epi = 0.1
        target_std_ale = 0.2
        
        penalty_epi = F.relu(target_std_epi ** 2 - var_epi)
        penalty_ale = F.relu(target_std_ale ** 2 - var_ale)
        
        return penalty_epi + penalty_ale


# ==============================================================================
# ADAPTER FOR EXISTING TRAINER
# ==============================================================================

class MAQACredalLossV3Adapter(nn.Module):
    """Adapter to match existing MAQACredalTrainer signature."""
    
    def __init__(self, loss_v3: MAQACredalLossV3):
        super().__init__()
        self.loss_v3 = loss_v3
    
    def forward(self, params, p_star, entropy_gt, params_clear=None):
        """
        Forward pass with adaptation to trainer signature.
        """
        v3_losses = self.loss_v3(
            params_amb=params,
            params_clear=params_clear,
            p_star_amb=p_star,
            entropy_amb=entropy_gt,
        )
        
        total_loss = v3_losses['loss_total']
        
        # Remap keys to match trainer expectations
        loss_dict = {
            'loss_total': total_loss,
            'loss_kl': v3_losses['loss_answer'],  # Trainer's "answer" loss
            'loss_reg': v3_losses['loss_kl'],      # Trainer's "reg" loss
            'loss_cal': v3_losses['loss_cal_mse'] + v3_losses['loss_cal_rank'],
            'loss_cont': v3_losses['loss_contrast'],
            
            # Additional tracking
            'loss_cal_mse': v3_losses['loss_cal_mse'],
            'loss_cal_rank': v3_losses['loss_cal_rank'],
            'loss_epi_error': v3_losses['loss_epi_error'],
            'loss_variance': v3_losses['loss_variance'],
        }
        
        return total_loss, loss_dict


# ==============================================================================
# INITIALIZATION HELPER
# ==============================================================================

def initialize_uncertainty_heads(model, config: Dict):
    """
    Initialize uncertainty heads to output reasonable initial values.
    
    This prevents the "start from random" problem where heads might
    collapse to local minima early in training.
    """
    # Find the sigma_epi head
    if hasattr(model, 'sigma_epi_head'):
        # Initialize bias to output ~config['init_sigma_epi']
        # softplus(x) ≈ x for x > 0, so bias = log(exp(target) - 1)
        target_epi = config.get('init_sigma_epi', 0.2)
        init_bias_epi = np.log(np.exp(target_epi) - 1 + 1e-6)
        
        # Find the last layer with bias
        for layer in reversed(list(model.sigma_epi_head.modules())):
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, init_bias_epi)
                print(f"  ✓ Initialized σ_epi head bias to {init_bias_epi:.3f} (target output: {target_epi})")
                break
    
    # Find the sigma_ale head
    if hasattr(model, 'sigma_ale_head'):
        target_ale = config.get('init_sigma_ale', 0.5)
        init_bias_ale = np.log(np.exp(target_ale) - 1 + 1e-6)
        
        for layer in reversed(list(model.sigma_ale_head.modules())):
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, init_bias_ale)
                print(f"  ✓ Initialized σ_ale head bias to {init_bias_ale:.3f} (target output: {target_ale})")
                break
    
    return model


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

def create_loss_v3():
    """Create the V3 loss function with recommended settings."""
    return MAQACredalLossV3(
        lambda_answer=CONFIG_V3['lambda_answer'],
        lambda_kl=CONFIG_V3['lambda_kl'],
        lambda_cal_mse=CONFIG_V3['lambda_cal_mse'],
        lambda_cal_rank=CONFIG_V3['lambda_cal_rank'],
        lambda_epi_error=CONFIG_V3['lambda_epi_error'],
        lambda_variance=CONFIG_V3['lambda_variance'],
        lambda_contrast=CONFIG_V3['lambda_contrast'],
        rank_margin=CONFIG_V3['rank_margin'],
        prior_sigma=CONFIG_V3['prior_sigma'],
        min_sigma_epi=CONFIG_V3['min_sigma_epi'],
        max_sigma_epi=CONFIG_V3['max_sigma_epi'],
        min_sigma_ale=CONFIG_V3['min_sigma_ale'],
        max_sigma_ale=CONFIG_V3['max_sigma_ale'],
    )


# ==============================================================================
# QUICK INTEGRATION GUIDE
# ==============================================================================

INTEGRATION_GUIDE = """
================================================================================
HOW TO USE V3 LOSS IN YOUR TRAINING SCRIPT
================================================================================

1. Import the new loss:
   
   from maqa_credal_loss_v3 import \
       MAQACredalLossV3, 
       MAQACredalLossV3Adapter, 
       CONFIG_V3,
       initialize_uncertainty_heads

2. After creating model, initialize heads:
   
   model = CredalMAQA(...)
   model = initialize_uncertainty_heads(model, CONFIG_V3)

3. Replace loss function:
   
   # OLD:
   # loss_v2 = MAQACredalLossV2(...)
   # trainer.criterion = MAQACredalLossV2Adapter(loss_v2)
   
   # NEW:
   loss_v3 = MAQACredalLossV3(
       lambda_answer=1.0,
       lambda_kl=0.001,         # REDUCED from 0.01
       lambda_cal_mse=0.1,      # REDUCED from 1.0
       lambda_cal_rank=5.0,     # NEW
       lambda_epi_error=2.0,    # NEW
       lambda_variance=1.0,     # NEW
       prior_sigma=0.3,         # INCREASED from 0.1
   )
   trainer.criterion = MAQACredalLossV3Adapter(loss_v3)

4. Monitor new metrics during training:
   
   - loss_cal_rank: Should decrease as σ_ale learns to rank by entropy
   - loss_epi_error: Should decrease as σ_epi learns to rank by error
   - loss_variance: Should be near 0 if uncertainties have enough spread

================================================================================
EXPECTED IMPROVEMENTS
================================================================================

BEFORE (V2):
  σ_epi: Mean=0.089, Range=[0.083, 0.100] (collapsed!)
  σ_ale: Mean=0.757, Range=[0.755, 0.759] (collapsed!)
  ρ(AU, Entropy) = 0.10

AFTER (V3 expected):
  σ_epi: Mean=0.2-0.4, Range=[0.05, 1.0] (spread across examples)
  σ_ale: Mean=0.5-1.0, Range=[0.1, 2.0] (tracking entropy)
  ρ(AU, Entropy) > 0.4 (meaningful calibration)

================================================================================
"""

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
    print("\nV3 Configuration:")
    for k, v in CONFIG_V3.items():
        print(f"  {k}: {v}")
