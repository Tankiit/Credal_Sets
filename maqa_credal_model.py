"""
MAQA Credal Model Implementation
================================

Specialized implementation for MAQA dataset with:
- Paired ambiguous/clear questions
- Ground-truth answer distributions (p*)
- Contrastive learning between paired questions
- Gradient separation tracking

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CredalQAParameters:
    """Parameters for credal QA model."""
    mu: torch.Tensor  # Mean prediction [batch, num_answers]
    sigma_epi: torch.Tensor  # Epistemic uncertainty [batch]
    sigma_ale: torch.Tensor  # Aleatoric uncertainty [batch]


# ============================================================================
# DATASET
# ============================================================================

class MAQADataset(Dataset):
    """
    MAQA Dataset with paired questions.

    Each item contains:
    - text: Question text
    - p_star: Ground-truth answer distribution
    - entropy: Ground-truth aleatoric uncertainty
    - ambiguity_level: 0=clear, 1=medium, 2=ambiguous
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        use_paired: bool = True
    ):
        """
        Args:
            data: Processed MAQA data from load_maqa_direct
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            use_paired: If True, create ambiguous/clear pairs
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_paired = use_paired

        # Create pairs if needed
        if use_paired:
            self.pairs = self._create_pairs()
        else:
            self.pairs = None

    def _create_pairs(self) -> List[Tuple[int, int]]:
        """Create ambiguous-clear pairs for contrastive learning."""
        # Split by ambiguity level
        ambiguous = [i for i, d in enumerate(self.data)
                     if d['ambiguity_level'] >= 2]  # High ambiguity
        clear = [i for i, d in enumerate(self.data)
                if d['ambiguity_level'] == 0]  # Low ambiguity

        # Create pairs (simple: match by index)
        pairs = []
        n_pairs = min(len(ambiguous), len(clear))
        for i in range(n_pairs):
            pairs.append((ambiguous[i], clear[i]))

        print(f"Created {len(pairs)} ambiguous-clear pairs")
        return pairs

    def __len__(self):
        if self.use_paired and self.pairs:
            return len(self.pairs)
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_paired and self.pairs:
            # Return paired sample
            amb_idx, clear_idx = self.pairs[idx]

            amb_item = self._get_item(amb_idx)
            clear_item = self._get_item(clear_idx)

            return {
                'amb': amb_item,
                'clear': clear_item
            }
        else:
            return self._get_item(idx)

    def _get_item(self, idx):
        """Get single item."""
        item = self.data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'p_star': torch.tensor(item['p_star'], dtype=torch.float32),
            'entropy': torch.tensor(item['entropy'], dtype=torch.float32),
            'ambiguity_level': torch.tensor(item['ambiguity_level'], dtype=torch.long),
            'num_answers': len(item['answers']),
            'dominant_answer_idx': torch.tensor(item['dominant_answer_idx'], dtype=torch.long),
        }


# ============================================================================
# CUSTOM COLLATE FUNCTION
# ============================================================================

def maqa_collate_fn(batch):
    """
    Custom collate function for MAQA.

    Handles variable-length p_star vectors.
    """
    if 'amb' in batch[0]:
        # Paired batch
        amb_batch = [item['amb'] for item in batch]
        clear_batch = [item['clear'] for item in batch]

        return {
            'amb': collate_single(amb_batch),
            'clear': collate_single(clear_batch)
        }
    else:
        # Single batch
        return collate_single(batch)


def collate_single(batch):
    """Collate single (non-paired) batch."""
    # Find max number of answers
    max_answers = max(item['p_star'].size(0) for item in batch)

    # Pad p_star to max_answers
    p_stars = []
    for item in batch:
        p_star = item['p_star']
        if p_star.size(0) < max_answers:
            # Pad with zeros
            p_star = F.pad(p_star, (0, max_answers - p_star.size(0)))
        p_stars.append(p_star)

    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'p_star': torch.stack(p_stars),
        'entropy': torch.stack([item['entropy'] for item in batch]),
        'ambiguity_level': torch.stack([item['ambiguity_level'] for item in batch]),
        'num_answers': torch.tensor([item['num_answers'] for item in batch]),
        'dominant_answer_idx': torch.stack([item['dominant_answer_idx'] for item in batch]),
    }


# ============================================================================
# MODEL
# ============================================================================

class CredalMAQA(nn.Module):
    """
    Credal MAQA Model with three separate prediction heads.

    Architecture:
    1. Encoder (frozen) → hidden representations
    2. μ head: Predict answer distribution
    3. σ_epi head: Predict epistemic uncertainty
    4. σ_ale head: Predict aleatoric uncertainty
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_answers: int = 10,  # Max number of answers
        projection_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = encoder
        self.hidden_size = hidden_size
        self.num_answers = num_answers

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

        # μ head: Answer distribution prediction
        self.mu_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_answers),
            nn.Softmax(dim=-1)  # Ensure probability distribution
        )

        # σ_epi head: Epistemic uncertainty (positive)
        self.sigma_epi_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )

        # σ_ale head: Aleatoric uncertainty (positive)
        self.sigma_ale_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> CredalQAParameters:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            return_embeddings: If True, return encoder embeddings

        Returns:
            CredalQAParameters with mu, sigma_epi, sigma_ale
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        hidden = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # Project
        projected = self.projection(hidden)  # [batch, projection_dim]

        # Predict
        mu = self.mu_head(projected)  # [batch, num_answers]
        sigma_epi = self.sigma_epi_head(projected).squeeze(-1)  # [batch]
        sigma_ale = self.sigma_ale_head(projected).squeeze(-1)  # [batch]

        if return_embeddings:
            return CredalQAParameters(mu, sigma_epi, sigma_ale), hidden

        return CredalQAParameters(mu, sigma_epi, sigma_ale)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class MAQACredalLoss(nn.Module):
    """
    Loss function for MAQA Credal Model.

    Components:
    1. Answer prediction loss (KL divergence)
    2. KL regularization (keep distributions separate)
    3. Calibration loss (σ_ale should match ground-truth entropy)
    4. Contrastive loss (paired questions)
    """

    def __init__(
        self,
        alpha_kl: float = 1.0,  # Answer prediction weight
        alpha_reg: float = 0.1,  # KL regularization weight
        alpha_cal: float = 0.5,  # Calibration weight
        alpha_cont: float = 0.3,  # Contrastive weight
    ):
        super().__init__()
        self.alpha_kl = alpha_kl
        self.alpha_reg = alpha_reg
        self.alpha_cal = alpha_cal
        self.alpha_cont = alpha_cont

    def forward(
        self,
        params: CredalQAParameters,
        p_star: torch.Tensor,
        entropy_gt: torch.Tensor,
        params_clear: Optional[CredalQAParameters] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss.

        Args:
            params: Model parameters for (ambiguous) question
            p_star: Ground-truth answer distribution
            entropy_gt: Ground-truth entropy
            params_clear: Model parameters for clear question (optional, for contrastive)

        Returns:
            Total loss and loss components dict
        """
        losses = {}

        # ====================================================================
        # 1. Answer prediction loss (KL divergence)
        # ====================================================================
        # KL(p* || μ) - encourage μ to match ground-truth distribution
        # Truncate to actual number of answers
        batch_size = p_star.size(0)
        kl_losses = []

        for i in range(batch_size):
            # Find actual number of non-zero entries in p_star
            num_answers = (p_star[i] > 0).sum().item()

            if num_answers > 0:
                # Truncate μ to actual number of answers
                mu_trunc = params.mu[i, :num_answers]
                p_star_trunc = p_star[i, :num_answers]

                # KL divergence
                kl = F.kl_div(
                    mu_trunc.log().clamp(min=-10),
                    p_star_trunc,
                    reduction='batchmean'
                )
                kl_losses.append(kl)

        loss_kl = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0)
        losses['loss_kl'] = loss_kl

        # ====================================================================
        # 2. KL regularization (keep distributions separate)
        # ====================================================================
        # Encourage σ_epi to stay away from σ_ale
        loss_reg = F.relu(
            torch.tensor(0.1) - torch.abs(params.sigma_epi - params.sigma_ale)
        ).mean()
        losses['loss_reg'] = loss_reg

        # ====================================================================
        # 3. Calibration loss (σ_ale should match entropy)
        # ====================================================================
        # MSE between predicted σ_ale and ground-truth entropy
        loss_cal = F.mse_loss(params.sigma_ale, entropy_gt)
        losses['loss_cal'] = loss_cal

        # ====================================================================
        # 4. Contrastive loss (paired questions)
        # ====================================================================
        if params_clear is not None:
            # AU(ambiguous) should be > AU(clear)
            # Pull: σ_ale(amb) > σ_ale(clear)
            loss_cont = F.relu(
                params_clear.sigma_ale - params.sigma_ale
            ).mean()
            losses['loss_cont'] = loss_cont
        else:
            loss_cont = torch.tensor(0.0)
            losses['loss_cont'] = loss_cont

        # ====================================================================
        # Total loss
        # ====================================================================
        total_loss = (
            self.alpha_kl * loss_kl +
            self.alpha_reg * loss_reg +
            self.alpha_cal * loss_cal +
            self.alpha_cont * loss_cont
        )
        losses['loss_total'] = total_loss

        return total_loss, losses


# ============================================================================
# TRAINER WITH GRADIENT SEPARATION
# ============================================================================

class MAQACredalTrainer:
    """
    Trainer for MAQA Credal Model with gradient separation tracking.

    Features:
    - Separate optimizers for μ, σ_epi, σ_ale heads
    - Track gradient conflicts
    - Per-epoch metrics collection
    """

    def __init__(
        self,
        model: CredalMAQA,
        train_loader,
        val_loader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Separate optimizers for each head
        self.mu_params = list(model.mu_head.parameters())
        self.sigma_epi_params = list(model.sigma_epi_head.parameters())
        self.sigma_ale_params = list(model.sigma_ale_head.parameters())
        self.proj_params = list(model.projection.parameters())

        # Combined optimizer
        all_params = (
            self.mu_params +
            self.sigma_epi_params +
            self.sigma_ale_params +
            self.proj_params
        )

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        self.criterion = MAQACredalLoss()

        # Gradient separation tracking
        self.gradient_conflicts = []
        self.gradient_alignments = []

    def train_epoch(self) -> Dict:
        """Train for one epoch with detailed loss tracking."""
        self.model.train()

        total_loss = 0.0
        total_answer_loss = 0.0
        total_kl_reg_loss = 0.0
        total_calibration_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0

        epoch_conflicts = []
        epoch_alignments = []

        # For contrastive success rate
        contrastive_success_count = 0
        contrastive_total = 0

        for batch in self.train_loader:
            # Check if paired
            if 'amb' in batch:
                # Paired batch
                amb_batch = {k: v.to(self.device) for k, v in batch['amb'].items()}
                clear_batch = {k: v.to(self.device) for k, v in batch['clear'].items()}

                # Forward
                params_amb = self.model(
                    amb_batch['input_ids'],
                    amb_batch['attention_mask']
                )
                params_clear = self.model(
                    clear_batch['input_ids'],
                    clear_batch['attention_mask']
                )

                # Loss
                loss, losses = self.criterion(
                    params_amb,
                    amb_batch['p_star'],
                    amb_batch['entropy'],
                    params_clear=params_clear
                )

                # Track contrastive success
                if params_amb.sigma_ale.mean() > params_clear.sigma_ale.mean():
                    contrastive_success_count += 1
                contrastive_total += 1
            else:
                # Single batch
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward
                params = self.model(
                    batch['input_ids'],
                    batch['attention_mask']
                )

                # Loss
                loss, losses = self.criterion(
                    params,
                    batch['p_star'],
                    batch['entropy']
                )

            # Backward
            self.optimizer.zero_grad()

            # Track gradients before step
            conflict = self._check_gradient_conflict()
            alignment = self._check_gradient_alignment()

            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_answer_loss += losses['loss_kl'].item()
            total_kl_reg_loss += losses.get('loss_reg', torch.tensor(0.0)).item()
            total_calibration_loss += losses['loss_cal'].item()
            total_contrastive_loss += losses['loss_cont'].item()
            num_batches += 1

            epoch_conflicts.append(conflict)
            epoch_alignments.append(alignment)

        # Compute metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'answer_loss': total_answer_loss / num_batches,
            'kl_reg_loss': total_kl_reg_loss / num_batches,
            'calibration_loss': total_calibration_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'gradient_conflict_rate': np.mean(epoch_conflicts),
            'gradient_alignment_rate': np.mean(epoch_alignments),
            'contrastive_success_rate': contrastive_success_count / contrastive_total if contrastive_total > 0 else 0.0,
        }

        return metrics

    def evaluate(self, loader) -> Dict:
        """Evaluate model with comprehensive metrics."""
        self.model.eval()

        total_loss = 0.0
        total_answer_loss = 0.0
        total_kl_loss = 0.0
        total_calibration_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0

        all_sigma_epi = []
        all_sigma_ale = []
        all_entropy_gt = []

        # For contrastive success rate
        contrastive_success_count = 0
        contrastive_total = 0

        with torch.no_grad():
            for batch in loader:
                # Check if paired
                if 'amb' in batch:
                    # Paired batch - evaluate contrastive learning
                    amb_batch = {k: v.to(self.device) for k, v in batch['amb'].items()}
                    clear_batch = {k: v.to(self.device) for k, v in batch['clear'].items()}

                    # Forward both
                    params_amb = self.model(
                        amb_batch['input_ids'],
                        amb_batch['attention_mask']
                    )
                    params_clear = self.model(
                        clear_batch['input_ids'],
                        clear_batch['attention_mask']
                    )

                    # Loss with contrastive
                    loss, loss_dict = self.criterion(
                        params_amb,
                        amb_batch['p_star'],
                        amb_batch['entropy'],
                        params_clear=params_clear
                    )

                    # Track contrastive success
                    # AU(ambig) should be > AU(clear)
                    if params_amb.sigma_ale.mean() > params_clear.sigma_ale.mean():
                        contrastive_success_count += 1
                    contrastive_total += 1

                    # Use ambiguous for metrics
                    params = params_amb
                    batch_metrics = amb_batch

                else:
                    # Single batch
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward
                    params = self.model(
                        batch['input_ids'],
                        batch['attention_mask']
                    )

                    # Loss
                    loss, loss_dict = self.criterion(
                        params,
                        batch['p_star'],
                        batch['entropy']
                    )
                    batch_metrics = batch

                total_loss += loss.item()
                total_answer_loss += loss_dict['loss_kl'].item()
                total_kl_loss += loss_dict.get('loss_reg', torch.tensor(0.0)).item()
                total_calibration_loss += loss_dict['loss_cal'].item()
                total_contrastive_loss += loss_dict['loss_cont'].item()
                num_batches += 1

                # Collect predictions (flatten batches)
                all_sigma_epi.append(params.sigma_epi.cpu())
                all_sigma_ale.append(params.sigma_ale.cpu())
                all_entropy_gt.append(batch_metrics['entropy'].cpu())

        # Compute metrics (concatenate batches)
        all_sigma_epi = torch.cat(all_sigma_epi, dim=0)
        all_sigma_ale = torch.cat(all_sigma_ale, dim=0)
        all_entropy_gt = torch.cat(all_entropy_gt, dim=0)

        # Basic metrics
        metrics = {
            'val_loss': total_loss / num_batches,
            'answer_loss': total_answer_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'calibration_loss': total_calibration_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'mean_sigma_epi': all_sigma_epi.mean().item(),
            'mean_sigma_ale': all_sigma_ale.mean().item(),
            'mean_entropy_gt': all_entropy_gt.mean().item(),
        }

        # Add correlations if enough samples
        if len(all_sigma_epi) > 100:
            from scipy import stats
            # ρ(EU, AU): Bifurcation success (target: < 0.3)
            rho_eu_au, p_eu_au = stats.pearsonr(
                all_sigma_epi.numpy(),
                all_sigma_ale.numpy()
            )
            metrics['rho_eu_au'] = rho_eu_au
            metrics['p_eu_au'] = p_eu_au

            # ρ(AU, Entropy): AU validity (target: > 0.5)
            rho_au_entropy, p_au_entropy = stats.pearsonr(
                all_sigma_ale.numpy(),
                all_entropy_gt.numpy()
            )
            metrics['rho_au_entropy'] = rho_au_entropy
            metrics['p_au_entropy'] = p_au_entropy

        # Contrastive success rate (target: > 80%)
        if contrastive_total > 0:
            metrics['contrastive_success_rate'] = contrastive_success_count / contrastive_total
            metrics['contrastive_total_pairs'] = contrastive_total

        return metrics

    def _check_gradient_conflict(self) -> float:
        """Check if gradients for σ_epi and σ_ale are in conflict."""
        if not self.sigma_epi_params or not self.sigma_ale_params:
            return 0.0

        # Get gradients
        grad_epi = self.sigma_epi_params[0].grad
        grad_ale = self.sigma_ale_params[0].grad

        if grad_epi is None or grad_ale is None:
            return 0.0

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            grad_epi.flatten().unsqueeze(0),
            grad_ale.flatten().unsqueeze(0)
        )

        # Conflict if negative similarity
        return float(cos_sim.item() < 0)

    def _check_gradient_alignment(self) -> float:
        """Check if gradients for σ_epi and σ_ale are aligned."""
        if not self.sigma_epi_params or not self.sigma_ale_params:
            return 0.0

        # Get gradients
        grad_epi = self.sigma_epi_params[0].grad
        grad_ale = self.sigma_ale_params[0].grad

        if grad_epi is None or grad_ale is None:
            return 0.0

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            grad_epi.flatten().unsqueeze(0),
            grad_ale.flatten().unsqueeze(0)
        )

        # Alignment if positive similarity
        return float(cos_sim.item() > 0)
