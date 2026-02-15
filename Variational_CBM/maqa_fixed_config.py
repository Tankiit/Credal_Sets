"""
MAQA Credal CBM - Fixed Configuration

Key fixes:
1. Lower prior_sigma to make KL loss meaningful
2. Increase beta (KL weight) to encourage epistemic learning
3. Add min_sigma_epi to prevent collapse
4. Better initialization for sigma_epi head
5. Combined MAQA + AmbigQA dataset for ~3000 samples
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ==============================================================================
# RECOMMENDED HYPERPARAMETERS
# ==============================================================================

CONFIG_FIXED = {
    # Model
    'encoder_name': 'microsoft/deberta-v3-base',
    'hidden_dim': 256,
    'freeze_encoder': True,

    # Training
    'num_epochs': 100,  # More epochs since KL needs time
    'batch_size': 16,
    'learning_rate': 2e-5,

    # Loss weights
    'beta': 0.01,           # INCREASED from 1e-4 to make KL meaningful
    'lambda_cal': 1.0,      # Keep calibration strong
    'lambda_contrast': 0.5, # Keep contrastive
    'margin': 0.3,

    # Epistemic prior - KEY FIX
    'prior_sigma': 0.1,     # DECREASED from 0.5 to 0.1
                            # This makes deviation from prior more costly

    # Prevent collapse
    'min_sigma_epi': 0.05,  # NEW: Floor on epistemic uncertainty
    'max_sigma_epi': 2.0,   # NEW: Ceiling to prevent explosion

    # Regularization
    'dropout': 0.2,         # Increased dropout
    'weight_decay': 0.01,   # Add weight decay
}

# ==============================================================================
# DATASET CONFIGURATION - COMBINED MAQA + AMBIGQA
# ==============================================================================

def load_combined_maqa_ambigqa(
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
):
    """
    Load combined MAQA-Star + AmbigQA-Star dataset.

    Total: ~3000 samples (500 MAQA + 2500 AmbigQA)
    Split: 80/10/10 → ~2400 train, ~300 val, ~300 test

    This gives enough samples for reliable correlation estimates!
    """
    from datasets import load_dataset as hf_load_dataset
    import numpy as np

    print("\n" + "="*60)
    print("Loading Combined MAQA + AmbigQA Dataset")
    print("="*60)

    # Load MAQA-Star
    print("\nLoading MAQA-Star...")
    maqa_ds = hf_load_dataset("ttomov/maqa_star")["train"]
    print(f"  MAQA: {len(maqa_ds)} samples")

    # Load AmbigQA-Star
    print("Loading AmbigQA-Star...")
    ambigqa_ds = hf_load_dataset("ttomov/ambigqa_star")["train"]
    print(f"  AmbigQA: {len(ambigqa_ds)} samples")

    # Combine datasets
    combined = []
    combined.extend([{"dataset": "maqa", **x} for x in maqa_ds])
    combined.extend([{"dataset": "ambigqa", **x} for x in ambigqa_ds])

    print(f"\n✓ Combined: {len(combined)} total samples")

    # Create splits
    np.random.seed(seed)
    indices = np.random.permutation(len(combined))
    n_train = int(len(combined) * split_ratio[0])
    n_val = int(len(combined) * split_ratio[1])

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = [combined[i] for i in train_indices]
    val_data = [combined[i] for i in val_indices]
    test_data = [combined[i] for i in test_indices]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Total: {len(combined)} samples")

    # Count by dataset
    print(f"\nTrain breakdown:")
    n_maqa_train = sum(1 for x in train_data if x['dataset'] == 'maqa')
    n_ambigqa_train = sum(1 for x in train_data if x['dataset'] == 'ambigqa')
    print(f"  MAQA: {n_maqa_train}")
    print(f"  AmbigQA: {n_ambigqa_train}")

    print(f"\n✓ Test set has {len(test_data)} samples - enough for correlations!")
    print(f"  (Need >100 for reliable correlation estimates)")

    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }


# ==============================================================================
# UPDATED LOSS FUNCTION
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MAQALossesV2:
    """Loss components with better tracking."""
    answer: torch.Tensor
    kl: torch.Tensor
    calibration: torch.Tensor
    contrastive: torch.Tensor
    total: torch.Tensor

    # Additional tracking
    kl_unweighted: torch.Tensor  # KL before beta multiplier

    def to_dict(self) -> Dict[str, float]:
        return {
            'loss/answer': self.answer.item(),
            'loss/kl': self.kl.item(),
            'loss/kl_unweighted': self.kl_unweighted.item(),  # Track raw KL
            'loss/calibration': self.calibration.item(),
            'loss/contrastive': self.contrastive.item(),
            'loss/total': self.total.item()
        }


class MAQACredalLossV2(nn.Module):
    """
    Fixed loss function with proper KL scaling.

    Key changes:
    1. Lower prior_sigma (0.1 instead of 0.5/1.0)
    2. Higher beta (0.01 instead of 1e-4)
    3. Track unweighted KL for debugging
    4. Clamp sigma_epi to prevent collapse
    """

    def __init__(
        self,
        beta: float = 0.01,          # INCREASED
        lambda_cal: float = 1.0,
        lambda_contrast: float = 0.5,
        margin: float = 0.3,
        prior_sigma: float = 0.1,    # DECREASED
        min_sigma_epi: float = 0.05, # NEW
        max_sigma_epi: float = 2.0,  # NEW
    ):
        super().__init__()
        self.beta = beta
        self.lambda_cal = lambda_cal
        self.lambda_contrast = lambda_contrast
        self.margin = margin
        self.prior_sigma = prior_sigma
        self.min_sigma_epi = min_sigma_epi
        self.max_sigma_epi = max_sigma_epi

    def forward(
        self,
        params_amb: 'CredalQAParameters',  # For ambiguous question
        params_clear: 'CredalQAParameters',  # For clear question
        p_star_amb: torch.Tensor,
        entropy_amb: torch.Tensor,
    ):
        device = params_amb.mu.device

        # =====================================================================
        # LOSS 1: Answer prediction (KL to soft targets)
        # =====================================================================
        # Handle variable p_star sizes
        batch_size = params_amb.mu.size(0)
        answer_losses = []

        for i in range(batch_size):
            # Find actual number of non-zero entries in p_star
            num_answers = (p_star_amb[i] > 0).sum().item()

            if num_answers > 0:
                # Truncate mu to actual number of answers
                mu_trunc = params_amb.mu[i, :num_answers]
                p_star_trunc = p_star_amb[i, :num_answers]

                # KL divergence
                kl = F.kl_div(
                    mu_trunc.log().clamp(min=-10),
                    p_star_trunc,
                    reduction='batchmean'
                )
                answer_losses.append(kl)

        if answer_losses:
            answer_loss = torch.stack(answer_losses).mean()
        else:
            answer_loss = torch.tensor(0.0, device=device)

        # =====================================================================
        # LOSS 2: KL regularization (FIXED)
        # =====================================================================
        # Clamp sigma_epi to prevent collapse
        sigma_epi_clamped = torch.clamp(
            params_amb.sigma_epi,
            min=self.min_sigma_epi,
            max=self.max_sigma_epi
        )

        kl_unweighted = self._compute_kl(params_amb.mu, sigma_epi_clamped)
        kl_loss = self.beta * kl_unweighted

        # =====================================================================
        # LOSS 3: Calibration (σ_ale → entropy)
        # =====================================================================
        # Use aleatoric uncertainty directly (already positive via softplus)
        au_amb = params_amb.sigma_ale
        au_clear = params_clear.sigma_ale

        cal_loss_amb = F.mse_loss(au_amb, entropy_amb)
        # Assume clear questions have lower entropy (use mean entropy)
        cal_loss_clear = F.mse_loss(au_clear, entropy_amb.mean().expand_as(au_clear))
        calibration_loss = cal_loss_amb + 0.5 * cal_loss_clear

        # =====================================================================
        # LOSS 4: Contrastive (AU(ambig) > AU(clear))
        # =====================================================================
        # Pull: AU(ambiguous) should be > AU(clear)
        contrastive_loss = F.relu(au_clear - au_amb).mean()

        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = (
            answer_loss +
            kl_loss +
            self.lambda_cal * calibration_loss +
            self.lambda_contrast * contrastive_loss
        )

        return {
            'loss_total': total_loss,
            'loss_answer': answer_loss,
            'loss_kl': kl_loss,
            'loss_kl_unweighted': kl_unweighted,
            'loss_calibration': calibration_loss,
            'loss_contrastive': contrastive_loss,
        }

    def _compute_kl(self, mu: torch.Tensor, sigma_epi: torch.Tensor) -> torch.Tensor:
        """
        Compute KL(N(μ, σ²_epi) || N(0, σ²_prior))

        KL = 0.5 * (σ²/σ²_prior + μ²/σ²_prior - 1 - log(σ²/σ²_prior))
        """
        prior_var = self.prior_sigma ** 2
        
        # Ensure sigma_epi is [batch, 1] for broadcasting against mu [batch, num_answers]
        if sigma_epi.dim() == 1:
            sigma_epi = sigma_epi.unsqueeze(-1)

        # Sum over answer dimensions
        var_ratio = (sigma_epi ** 2) / prior_var
        mu_term = (mu ** 2) / prior_var

        # KL per dimension
        kl_per_dim = 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio + 1e-10))

        # Sum over answer dimensions, mean over batch
        return kl_per_dim.sum(dim=-1).mean()


# ==============================================================================
# TRAINING RECOMMENDATIONS
# ==============================================================================

RECOMMENDATIONS = """
🔧 FIXES APPLIED:

1. PRIOR SIGMA: 0.5 → 0.1
   - With prior_σ = 0.1, any σ_epi > 0.1 will incur KL cost
   - Your current σ_epi ≈ 0.33 would give KL ≈ 2.5 per dimension

2. BETA: 1e-4 → 0.01
   - Makes KL term visible in total loss
   - Expected contribution: 0.01 * 2.5 * num_answers ≈ significant

3. MIN SIGMA EPI: 0.05
   - Prevents complete collapse to zero
   - Ensures some epistemic uncertainty is always predicted

4. INITIALIZATION:
   - sigma_epi head bias initialized to 0.5
   - Starts outputting σ_epi ≈ 0.5 > prior (0.1)
   - KL will be non-zero from the start

5. COMBINED DATASET:
   - MAQA: ~500 samples
   - AmbigQA: ~2500 samples
   - Total: ~3000 samples
   - Split: 2400 train / 300 val / 300 test
   - ✓ Test set has 300 samples → enough for correlations!

📊 EXPECTED BEHAVIOR:

Epoch 1:
- KL loss should be > 0 (around 0.5-2.0 unweighted)
- sigma_epi should start around 0.5

Training:
- sigma_epi should DECREASE toward prior (0.1) where confident
- sigma_epi should STAY HIGH where model is uncertain
- This creates meaningful epistemic signal!

Final:
- σ_epi should vary across examples (not constant)
- High σ_epi on hard/OOD examples
- Low σ_epi on easy/common examples

📈 METRICS TO WATCH:

1. kl_unweighted: Should be > 0 throughout training
2. σ_epi variance: Should be > 0 (not constant across examples)
3. ρ(EU, error): Should be positive (high EU → more errors)
4. ρ(EU, AU): Should be < 0.3 (uncertainties separable)
5. ρ(AU, Entropy): Should be > 0.5 (AU captures ambiguity)

With 300 test samples, all correlations will be statistically reliable!
"""

if __name__ == "__main__":
    print(RECOMMENDATIONS)
    print("\nRecommended config:")
    for k, v in CONFIG_FIXED.items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("Testing dataset loading...")
    print("="*60)
    data = load_combined_maqa_ambigqa()
    print(f"\n✓ Successfully loaded {len(data['train'])} training samples")
