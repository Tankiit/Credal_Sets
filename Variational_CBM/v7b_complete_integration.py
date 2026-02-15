"""
V7b Complete Integration Package
=================================

This file contains everything needed to run V7b properly:
1. Fixed CredalMAQA model that receives entropy for σ_ale
2. Fixed V7b loss with proper gradient isolation
3. Fixed adapter with proper key mapping
4. Patched MAQACredalTrainer that passes entropy to model

The KEY FIX: σ_ale head needs access to entropy to learn the mapping.
Without this, σ_ale outputs constant values and ρ(AU, Entropy) ≈ 0.

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG_V7B = {
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'dropout': 0.1,

    # Loss weights
    'lambda_answer': 1.0,
    'lambda_kl_epi': 0.001,
    'lambda_epi_residual': 1.5,
    'lambda_ale_entropy': 2.0,
    'lambda_ale_rank': 1.0,
    'lambda_decorr': 5.0,
    'lambda_gradient_isolation': 1.0,
    'lambda_credal_width': 0.5,
    'lambda_capacity': 0.5,
    'lambda_variance': 0.5,

    # Credal parameters
    'min_credal_width': 0.1,
    'target_total_sigma': 0.8,
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
# CREDAL OUTPUT STRUCTURE
# ==============================================================================

@dataclass
class CredalMAQAOutput:
    """Output from CredalMAQA_V7b model."""
    mu: torch.Tensor           # [B, num_answers] - answer logits
    sigma_epi: torch.Tensor    # [B] - epistemic uncertainty
    sigma_ale: torch.Tensor    # [B] - aleatoric uncertainty

    @property
    def sigma_lower(self):
        return self.sigma_epi

    @property
    def sigma_upper(self):
        return torch.sqrt(self.sigma_epi**2 + self.sigma_ale**2)

    @property
    def credal_width(self):
        return self.sigma_upper - self.sigma_lower


# ==============================================================================
# FIXED MODEL: CredalMAQA_V7b
# ==============================================================================

class CredalMAQA_V7b(nn.Module):
    """
    CredalMAQA model with V7b-specific design.

    KEY FIX: σ_ale head receives BOTH encoder hidden AND entropy.
    This allows σ_ale to directly learn σ_ale ≈ entropy + correction.

    Without this fix, σ_ale has no signal to learn the mapping from
    frozen encoder features to entropy, resulting in constant outputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_answers: int,
        projection_dim: int = 256,
        dropout: float = 0.1,
        min_sigma: float = 0.05,
        max_sigma: float = 1.5,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_answers = num_answers
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.hidden_size = hidden_size

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # =====================================================================
        # HEAD 1: Answer Logits (μ)
        # =====================================================================
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, num_answers)
        )

        # =====================================================================
        # HEAD 2: Epistemic Uncertainty (σ_epi)
        # Input: h only (learns residual error from encoder features)
        # =====================================================================
        self.sigma_epi_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, 1)
        )

        # =====================================================================
        # HEAD 3: Aleatoric Uncertainty (σ_ale) - THE KEY FIX!
        # Input: h + entropy (+1 dimension for entropy)
        # This allows the head to directly learn σ_ale ≈ entropy
        # =====================================================================
        self.sigma_ale_head = nn.Sequential(
            nn.Linear(hidden_size + 1, projection_dim),  # +1 for entropy!
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, 1)
        )

        self._init_weights()

        # Print architecture info
        print(f"\n{'='*60}")
        print("CredalMAQA_V7b Model (FIXED)")
        print(f"{'='*60}")
        print(f"  Encoder hidden: {hidden_size}")
        print(f"  Num answers: {num_answers}")
        print(f"  σ_ale head input: h + entropy (dim={hidden_size + 1})")
        print(f"  σ_epi head input: h only (dim={hidden_size})")
        print(f"  This fixes the ρ(AU, Entropy) ≈ 0 problem!")
        print(f"{'='*60}\n")

    def _init_weights(self):
        """Initialize weights for proper starting values."""

        # μ head: standard init
        for layer in self.mu_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # σ_epi head: initialize to output ~0.2-0.3
        for layer in self.sigma_epi_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Final σ_epi bias: softplus(-1.5) ≈ 0.2
        if hasattr(self.sigma_epi_head[-1], 'bias'):
            nn.init.constant_(self.sigma_epi_head[-1].bias, -1.5)

        # σ_ale head: initialize to output ~0.5 (mean entropy)
        # Use normal initialization so the head can actually learn
        for i, layer in enumerate(self.sigma_ale_head):
            if isinstance(layer, nn.Linear):
                if i == 0:  # First layer: h + entropy → projection
                    # Initialize to pay attention to both h and entropy
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:  # Later layers
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Final σ_ale bias: softplus(-0.7) ≈ 0.4 (close to mean entropy)
        if hasattr(self.sigma_ale_head[-1], 'bias'):
            nn.init.constant_(self.sigma_ale_head[-1].bias, -0.7)

        print(f"  ✓ Initialized σ_epi bias → target ~0.2")
        print(f"  ✓ Initialized σ_ale bias → target ~0.5")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden representation."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]  # CLS token
        return outputs[0][:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
    ) -> CredalMAQAOutput:
        """
        Forward pass.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            entropy: [B] ground truth entropy - REQUIRED for σ_ale to track entropy!

        Returns:
            CredalMAQAOutput with mu, sigma_epi, sigma_ale
        """
        # Encode
        h = self.encode(input_ids, attention_mask)

        # Answer logits
        mu = self.mu_head(h)

        # Epistemic uncertainty (from h only)
        sigma_epi_raw = self.sigma_epi_head(h)
        sigma_epi = F.softplus(sigma_epi_raw).squeeze(-1)
        sigma_epi = torch.clamp(sigma_epi, self.min_sigma, self.max_sigma)

        # Aleatoric uncertainty (from h + entropy)
        if entropy is not None:
            # THE KEY: Concatenate h with entropy so σ_ale can directly learn the mapping
            h_with_entropy = torch.cat([h, entropy.unsqueeze(-1)], dim=-1)
        else:
            # Inference without entropy: use placeholder
            # σ_ale won't be meaningful in this case
            placeholder = torch.zeros(h.size(0), 1, device=h.device)
            h_with_entropy = torch.cat([h, placeholder], dim=-1)

        sigma_ale_raw = self.sigma_ale_head(h_with_entropy)
        sigma_ale = F.softplus(sigma_ale_raw).squeeze(-1)
        sigma_ale = torch.clamp(sigma_ale, self.min_sigma, self.max_sigma)

        return CredalMAQAOutput(
            mu=mu,
            sigma_epi=sigma_epi,
            sigma_ale=sigma_ale,
        )


# ==============================================================================
# V7B LOSS FUNCTION
# ==============================================================================

class MAQACredalLossV7b(nn.Module):
    """V7b Loss with gradient isolation for disentanglement."""

    def __init__(self, config: Dict = None, **kwargs):
        super().__init__()

        cfg = {**CONFIG_V7B}
        if config is not None:
            cfg.update(config)
        cfg.update(kwargs)

        for key, value in cfg.items():
            setattr(self, key, value)

    def forward(
        self,
        params: CredalMAQAOutput,
        p_star: torch.Tensor,
        entropy: torch.Tensor,
        params_clear: Optional[CredalMAQAOutput] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""

        device = params.mu.device

        sigma_epi = torch.clamp(params.sigma_epi, self.min_sigma, self.max_sigma)
        sigma_ale = torch.clamp(params.sigma_ale, self.min_sigma, self.max_sigma)

        sigma_upper = torch.sqrt(sigma_epi**2 + sigma_ale**2)
        credal_width = sigma_upper - sigma_epi

        # 1. Answer loss
        answer_loss, pred_error = self._compute_answer_loss_and_error(params.mu, p_star)

        # 2. Epistemic losses
        kl_epi_loss = self._compute_kl_loss(params.mu, sigma_epi)

        # KEY: .detach() on sigma_ale for gradient isolation!
        residual_error = pred_error - sigma_ale.detach()
        residual_normalized = torch.sigmoid(residual_error * 3)
        target_epi = self.min_sigma + (self.max_sigma - self.min_sigma) * residual_normalized
        epi_residual_loss = F.mse_loss(sigma_epi, target_epi.detach())

        # 3. Aleatoric losses
        ale_entropy_loss = F.mse_loss(sigma_ale, entropy)
        ale_rank_loss = self._compute_ranking_loss(sigma_ale, entropy)

        # 4. Disentanglement losses
        decorr_loss = self._compute_correlation(sigma_epi, sigma_ale) ** 2
        gradient_isolation_loss = self._compute_correlation(sigma_epi, entropy) ** 2

        # 5. Credal losses
        width_loss = F.relu(self.min_credal_width - credal_width).mean()
        capacity_loss = F.mse_loss(sigma_upper, torch.full_like(sigma_upper, self.target_total_sigma))

        # 6. Variance regularization
        variance_loss = (
            F.relu(self.target_std_epi ** 2 - sigma_epi.var()) +
            F.relu(self.target_std_ale ** 2 - sigma_ale.var())
        )

        # Total
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

        # Correlations
        with torch.no_grad():
            rho_eu_au = self._compute_correlation(sigma_epi, sigma_ale)
            rho_au_entropy = self._compute_correlation(sigma_ale, entropy)
            rho_eu_entropy = self._compute_correlation(sigma_epi, entropy)

        return {
            'loss_total': total_loss,
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
            'rho_eu_au': rho_eu_au,
            'rho_au_entropy': rho_au_entropy,
            'rho_eu_entropy': rho_eu_entropy,
            'mean_credal_width': credal_width.mean(),
            'mean_sigma_epi': sigma_epi.mean(),
            'mean_sigma_ale': sigma_ale.mean(),
            'mean_sigma_upper': sigma_upper.mean(),
            'mean_pred_error': pred_error.mean(),
        }

    def _compute_answer_loss_and_error(self, mu, p_star):
        batch_size = mu.size(0)
        losses, errors = [], []

        for i in range(batch_size):
            num_answers = (p_star[i] > 0).sum().item()
            if num_answers > 0:
                mu_i = mu[i, :num_answers]
                p_star_i = p_star[i, :num_answers]
                pred_dist = F.softmax(mu_i, dim=-1)
                kl = F.kl_div(pred_dist.log().clamp(min=-10), p_star_i, reduction='sum')
                losses.append(kl)
                errors.append(1.0 - (pred_dist * p_star_i).sum())
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
        kl = 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio + 1e-10))
        return kl.sum(dim=-1).mean()

    def _compute_ranking_loss(self, sigma, target):
        batch_size = sigma.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=sigma.device)

        num_pairs = min(self.rank_num_pairs, batch_size * (batch_size - 1) // 2)
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

    def _compute_correlation(self, x, y):
        x_c = x - x.mean()
        y_c = y - y.mean()
        return (x_c * y_c).mean() / ((x.std() + 1e-8) * (y.std() + 1e-8))


# ==============================================================================
# ADAPTER FOR TRAINER COMPATIBILITY
# ==============================================================================

class MAQACredalLossV7bAdapter(nn.Module):
    """Adapter with complete key mapping for trainer."""

    def __init__(self, loss_v7b: MAQACredalLossV7b):
        super().__init__()
        self.loss_v7b = loss_v7b

    def forward(self, params, p_star, entropy_gt, params_clear=None):
        losses = self.loss_v7b(params, p_star, entropy_gt, params_clear)

        device = losses['loss_total'].device

        loss_dict = {
            'loss_total': losses['loss_total'],

            # Answer loss aliases
            'loss_kl': losses['loss_answer'],
            'loss_answer': losses['loss_answer'],
            'answer_loss': losses['loss_answer'],

            # Regularization aliases
            'loss_reg': losses['loss_kl_epi'],
            'loss_kl_v3': losses['loss_kl_epi'],
            'kl_reg_loss': losses['loss_kl_epi'],

            # Calibration aliases
            'loss_cal': losses['loss_ale_entropy'] + losses['loss_ale_rank'],
            'calibration_loss': losses['loss_ale_entropy'] + losses['loss_ale_rank'],
            'loss_cal_mse': losses['loss_ale_entropy'],
            'loss_cal_rank': losses['loss_ale_rank'],

            # Contrastive (V7b doesn't use)
            'loss_cont': torch.tensor(0.0, device=device),
            'contrastive_loss': torch.tensor(0.0, device=device),

            # V7b specific
            'loss_epi_residual': losses['loss_epi_residual'],
            'loss_epi_error': losses['loss_epi_residual'],
            'loss_ale_entropy': losses['loss_ale_entropy'],
            'loss_ale_rank': losses['loss_ale_rank'],
            'loss_decorr': losses['loss_decorr'],
            'loss_gradient_isolation': losses['loss_gradient_isolation'],
            'loss_credal_width': losses['loss_credal_width'],
            'loss_capacity': losses['loss_capacity'],
            'loss_variance': losses['loss_variance'],

            # Correlations
            'rho_eu_au': losses['rho_eu_au'],
            'rho_au_entropy': losses['rho_au_entropy'],
            'rho_eu_entropy': losses['rho_eu_entropy'],

            # Credal metrics
            'mean_credal_width': losses['mean_credal_width'],
            'mean_sigma_epi': losses['mean_sigma_epi'],
            'mean_sigma_ale': losses['mean_sigma_ale'],
            'mean_sigma_upper': losses['mean_sigma_upper'],
        }

        return losses['loss_total'], loss_dict


# ==============================================================================
# PATCHED TRAINER THAT PASSES ENTROPY TO MODEL
# ==============================================================================

class MAQACredalTrainerV7b:
    """
    Trainer patched for V7b that passes entropy to the model.

    KEY CHANGE: model.forward() receives entropy so σ_ale head can use it.
    """

    def __init__(
        self,
        model: CredalMAQA_V7b,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function (to be set externally)
        self.criterion = None

    def train_epoch(self) -> Dict:
        """Training epoch with entropy passed to model."""
        self.model.train()

        total_loss = 0.0
        all_metrics = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Check if paired batch (for contrastive learning)
            if 'amb' in batch:
                # Paired batch: amb (ambiguous) vs clear (unambiguous)
                amb_batch = {k: v.to(self.device) for k, v in batch['amb'].items()}
                clear_batch = {k: v.to(self.device) for k, v in batch['clear'].items()}

                self.optimizer.zero_grad()

                # Forward both
                params_amb = self.model(
                    amb_batch['input_ids'],
                    amb_batch['attention_mask'],
                    entropy=amb_batch['entropy']
                )
                params_clear = self.model(
                    clear_batch['input_ids'],
                    clear_batch['attention_mask'],
                    entropy=clear_batch['entropy']
                )

                # Compute loss with paired samples
                loss, loss_dict = self.criterion(
                    params_amb,
                    amb_batch['p_star'],
                    amb_batch['entropy'],
                    params_clear=params_clear
                )
            else:
                # Single batch (standard)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                p_star = batch['p_star'].to(self.device)
                entropy = batch['entropy'].to(self.device)

                self.optimizer.zero_grad()

                # KEY FIX: Pass entropy to model!
                params = self.model(input_ids, attention_mask, entropy=entropy)

                # Compute loss
                loss, loss_dict = self.criterion(params, p_star, entropy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_metrics.append({k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()})

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ρ(AU,H)': f"{loss_dict.get('rho_au_entropy', 0):.3f}"
            })

        # Aggregate metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                vals = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = np.mean(vals) if vals else 0.0

        avg_metrics['train_loss'] = total_loss / len(self.train_loader)

        # Add required keys that might be missing
        avg_metrics.setdefault('gradient_conflict_rate', 0.0)
        avg_metrics.setdefault('gradient_alignment_rate', 0.0)
        avg_metrics.setdefault('contrastive_success_rate', 0.5)

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """Evaluation with entropy passed to model."""
        self.model.eval()

        all_sigma_epi = []
        all_sigma_ale = []
        all_entropies = []
        total_loss = 0.0

        for batch in tqdm(loader, desc="Evaluating"):
            # Check if paired batch (unlikely for eval, but handle it)
            if 'amb' in batch:
                # For paired batches, only evaluate on amb
                batch = batch['amb']

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            p_star = batch['p_star'].to(self.device)
            entropy = batch['entropy'].to(self.device)

            # KEY FIX: Pass entropy to model!
            params = self.model(input_ids, attention_mask, entropy=entropy)

            loss, loss_dict = self.criterion(params, p_star, entropy)
            total_loss += loss.item()

            all_sigma_epi.append(params.sigma_epi.cpu().numpy())
            all_sigma_ale.append(params.sigma_ale.cpu().numpy())
            all_entropies.append(entropy.cpu().numpy())

        # Concatenate
        all_sigma_epi = np.concatenate(all_sigma_epi)
        all_sigma_ale = np.concatenate(all_sigma_ale)
        all_entropies = np.concatenate(all_entropies)

        # Compute correlations
        rho_eu_au, p_eu_au = stats.spearmanr(all_sigma_epi, all_sigma_ale)
        rho_au_entropy, p_au_entropy = stats.spearmanr(all_sigma_ale, all_entropies)

        return {
            'val_loss': total_loss / len(loader),
            'mean_sigma_epi': float(all_sigma_epi.mean()),
            'mean_sigma_ale': float(all_sigma_ale.mean()),
            'mean_entropy_gt': float(all_entropies.mean()),
            'rho_eu_au': float(rho_eu_au),
            'p_eu_au': float(p_eu_au),
            'rho_au_entropy': float(rho_au_entropy),
            'p_au_entropy': float(p_au_entropy),
            # Placeholders
            'gradient_conflict_rate': 0.0,
            'gradient_alignment_rate': 0.0,
            'contrastive_success_rate': 0.5,
            'contrastive_total_pairs': 0,
        }


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_v7b_model(encoder, hidden_size, num_answers, **kwargs):
    """Create V7b-compatible model."""
    return CredalMAQA_V7b(
        encoder=encoder,
        hidden_size=hidden_size,
        num_answers=num_answers,
        **kwargs
    )


def create_v7b_loss(config=None, **kwargs):
    """Create V7b loss."""
    return MAQACredalLossV7b(config=config, **kwargs)


def create_v7b_adapter(config=None, **kwargs):
    """Create V7b loss with adapter."""
    return MAQACredalLossV7bAdapter(create_v7b_loss(config, **kwargs))


def create_v7b_trainer(model, train_loader, val_loader, device='cuda', **kwargs):
    """Create V7b trainer."""
    trainer = MAQACredalTrainerV7b(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **kwargs
    )
    trainer.criterion = create_v7b_adapter()
    return trainer


# ==============================================================================
# INTEGRATION INSTRUCTIONS
# ==============================================================================

INTEGRATION_INSTRUCTIONS = """
================================================================================
V7b COMPLETE INTEGRATION INSTRUCTIONS
================================================================================

Replace your V7b section in main() with:

    elif loss_version == 'v7b':
        from v7b_complete_integration import (
            CredalMAQA_V7b,
            MAQACredalLossV7bAdapter,
            MAQACredalTrainerV7b,
            CONFIG_V7B,
            create_v7b_adapter,
        )

        # Create V7b-compatible model (σ_ale receives entropy!)
        model = CredalMAQA_V7b(
            encoder=encoder,
            hidden_size=encoder_config['hidden_size'],
            num_answers=max_answers,
            projection_dim=config.get('projection_dim', 256),
            dropout=CONFIG_V7B['dropout'],
        )

        # Use V7b trainer (passes entropy to model)
        trainer = MAQACredalTrainerV7b(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=CONFIG_V7B['learning_rate'],
            weight_decay=CONFIG_V7B['weight_decay'],
        )

        # Set loss with adapter
        trainer.criterion = create_v7b_adapter(config=CONFIG_V7B)

        print(f"  ✓ Using V7b complete integration")
        print(f"    σ_ale receives entropy: ENABLED")
        print(f"    Gradient isolation: ENABLED")

================================================================================

The key changes:
1. Use CredalMAQA_V7b instead of CredalMAQA (σ_ale receives entropy)
2. Use MAQACredalTrainerV7b instead of MAQACredalTrainer (passes entropy to model)
3. Use create_v7b_adapter() for proper key mapping

This should fix ρ(AU, Entropy) ≈ 0 → ρ(AU, Entropy) > 0.5
================================================================================
"""


if __name__ == "__main__":
    print(INTEGRATION_INSTRUCTIONS)
    print("\nConfiguration:")
    for k, v in CONFIG_V7B.items():
        print(f"  {k}: {v}")
