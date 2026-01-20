"""
Utils Module for Variational Credal CBM
==========================================

This file contains:
- CEBaB-specific configuration
- Label processing utilities
- Loss functions
- Evaluation metrics

Author: Tanmoy
Target: ICML 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats


# =============================================================
# CEBaB CONFIGURATION
# =============================================================

@dataclass
class CEBaBConfig:
    """Configuration for CEBaB label processing"""

    # Task labels: '1'-'5' → 0-4, 'no majority' → 5
    num_task_classes: int = 6
    abstain_class: int = 5

    # Concept labels: Negative=0, unknown=1, Positive=2
    num_concept_classes: int = 3

    # Soft label settings
    use_soft_task_labels: bool = False   # Use review_label_distribution for task
    use_soft_concept_labels: bool = True  # Use aspect_label_distribution for concepts
    soft_label_temperature: float = 1.5   # T > 1 preserves disagreement signal

    # Loss weights
    abstain_penalty: float = 0.1   # Weight for abstain calibration loss
    concept_weight: float = 1.0    # Weight for concept loss

    # Aspect names (CEBaB specific)
    aspect_names: Tuple[str, ...] = ('food', 'service', 'ambiance', 'noise')

    def __post_init__(self):
        """Validate configuration"""
        assert self.num_task_classes == 6, "CEBaB requires 6 task classes (5 ratings + abstain)"
        assert self.num_concept_classes == 3, "CEBaB concepts are ternary (Neg/Unk/Pos)"
        assert self.soft_label_temperature > 0, "Temperature must be positive"


# =============================================================
# LABEL PROCESSOR
# =============================================================

class CEBaBLabelProcessor:
    """
    Process CEBaB labels with:
    - 6 task classes (1-5 stars + abstain for 'no majority')
    - Temperature-scaled soft labels from annotator distributions
    - Soft concept labels from aspect distributions

    Key insight for CREDENCE:
    - 'no majority' samples have high annotator disagreement → aleatoric uncertainty
    - Soft concept labels preserve this disagreement signal
    """

    # Task label mapping
    TASK_LABEL_MAP = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, 'no majority': 5}
    TASK_LABEL_MAP_INV = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: 'no majority'}

    # Concept label mapping
    CONCEPT_LABEL_MAP = {
        'Negative': 0, 'negative': 0,
        'unknown': 1, 'Unknown': 1, 'no majority': 1,
        'Positive': 2, 'positive': 2,
    }
    CONCEPT_LABEL_MAP_INV = {0: 'Negative', 1: 'unknown', 2: 'Positive'}

    def __init__(self, config: Optional[CEBaBConfig] = None):
        self.config = config or CEBaBConfig()

    def process_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Process a full batch from CEBaB dataloader

        Args:
            batch: Raw batch from CEBaBDataset

        Returns:
            Dictionary with processed labels ready for training:
            - task_hard_labels: [batch_size] integer labels (0-5)
            - task_soft_labels: [batch_size, 6] probability distributions
            - is_ambiguous: [batch_size] boolean mask for 'no majority'
            - concept_hard_labels: [batch_size, 4] integer labels (0-2)
            - concept_soft_labels: [batch_size, 4, 3] probability distributions
        """
        task_labels = self._process_task_labels(batch)
        concept_labels = self._process_concept_labels(batch)

        return {
            'task_hard_labels': task_labels['hard_labels'],
            'task_soft_labels': task_labels['soft_labels'],
            'is_ambiguous': task_labels['is_ambiguous'],
            'concept_hard_labels': concept_labels['hard_labels'],
            'concept_soft_labels': concept_labels['soft_labels'],
        }

    def _process_task_labels(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Process task (review) labels"""
        labels_batch = batch['review_majority']
        distributions = batch.get('review_label_distribution', None)

        batch_size = len(labels_batch)
        hard_labels = torch.zeros(batch_size, dtype=torch.long)
        soft_labels = torch.zeros(batch_size, self.config.num_task_classes)
        is_ambiguous = torch.zeros(batch_size, dtype=torch.bool)

        for i, label in enumerate(labels_batch):
            # Handle tensor vs string
            label = self._to_string(label)

            # Hard label
            hard_labels[i] = self.TASK_LABEL_MAP.get(label, self.config.abstain_class)
            is_ambiguous[i] = (label == 'no majority')

            # Soft label from distribution
            if self.config.use_soft_task_labels and distributions is not None:
                dist = self._get_distribution(distributions, i)
                soft = self._task_distribution_to_soft_label(dist)
                soft_labels[i] = torch.tensor(soft)
            else:
                # One-hot from hard label
                soft_labels[i, hard_labels[i]] = 1.0

        return {
            'hard_labels': hard_labels,
            'soft_labels': soft_labels,
            'is_ambiguous': is_ambiguous,
        }

    def _process_concept_labels(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Process concept (aspect) labels with soft distributions"""
        first_aspect = f'{self.config.aspect_names[0]}_aspect_majority'
        batch_size = len(batch[first_aspect])
        num_concepts = len(self.config.aspect_names)

        hard_labels = torch.zeros(batch_size, num_concepts, dtype=torch.long)
        soft_labels = torch.zeros(batch_size, num_concepts, self.config.num_concept_classes)

        for i, aspect in enumerate(self.config.aspect_names):
            majority_key = f'{aspect}_aspect_majority'
            dist_key = f'{aspect}_aspect_label_distribution'

            for j in range(batch_size):
                # Get majority value
                majority = self._to_string(batch[majority_key][j])

                # Hard label
                hard_labels[j, i] = self.CONCEPT_LABEL_MAP.get(majority, 1)  # default unknown

                # Soft label from distribution
                if self.config.use_soft_concept_labels and dist_key in batch:
                    dist = self._get_distribution(batch[dist_key], j)
                    soft = self._concept_distribution_to_soft_label(dist)
                    soft_labels[j, i] = torch.tensor(soft)
                else:
                    # One-hot
                    soft_labels[j, i, hard_labels[j, i]] = 1.0

        return {
            'hard_labels': hard_labels,
            'soft_labels': soft_labels,
        }

    def _to_string(self, value) -> str:
        """Convert various types to string"""
        if isinstance(value, torch.Tensor):
            return str(value.item()) if value.dim() == 0 else str(value)
        return str(value) if not isinstance(value, str) else value

    def _get_distribution(self, distributions, idx: int) -> Dict:
        """Safely extract distribution at index"""
        if distributions is None:
            return {}
        if isinstance(distributions, (list, tuple)):
            return distributions[idx] if idx < len(distributions) else {}
        return distributions

    def _task_distribution_to_soft_label(self, dist: Dict) -> List[float]:
        """Convert review_label_distribution to 6-class soft label"""
        if not dist or not isinstance(dist, dict):
            return [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

        # Extract counts for classes 1-5
        counts = [float(dist.get(str(i), 0)) for i in range(1, 6)]
        total = sum(counts)

        if total == 0:
            return [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

        # Normalize
        probs = [c / total for c in counts]

        # Temperature scaling
        probs = self._apply_temperature(probs)

        # Add 0 for abstain class (class 5)
        probs.append(0.0)

        return probs

    def _concept_distribution_to_soft_label(self, dist: Dict) -> List[float]:
        """Convert aspect_label_distribution to 3-class soft label"""
        if not dist or not isinstance(dist, dict):
            return [1.0 / 3, 1.0 / 3, 1.0 / 3]

        # Map keys to indices
        key_map = {'Negative': 0, 'unknown': 1, 'Positive': 2}
        counts = [0.0, 0.0, 0.0]

        for key, count in dist.items():
            if key in key_map:
                counts[key_map[key]] = float(count)

        total = sum(counts)
        if total == 0:
            return [1.0 / 3, 1.0 / 3, 1.0 / 3]

        # Normalize
        probs = [c / total for c in counts]

        # Temperature scaling
        probs = self._apply_temperature(probs)

        return probs

    def _apply_temperature(self, probs: List[float]) -> List[float]:
        """
        Apply temperature scaling to probability distribution

        T < 1: Sharpen (more confident)
        T = 1: No change
        T > 1: Soften (more uncertain, preserves disagreement)
        """
        T = self.config.soft_label_temperature
        if T == 1.0:
            return probs

        # Avoid log(0)
        probs = [max(p, 1e-8) for p in probs]

        # Temperature scaling: p_i^(1/T) / sum(p_j^(1/T))
        scaled = [p ** (1.0 / T) for p in probs]
        total = sum(scaled)

        return [s / total for s in scaled]


# =============================================================
# LOSS FUNCTION
# =============================================================

class CEBaBLoss(nn.Module):
    """
    Loss function for CEBaB with:
    - Cross-entropy for task prediction (supports soft labels)
    - Soft cross-entropy for concept prediction
    - Abstain calibration regularization

    Key design: Abstain loss is SEPARATE from task accuracy
    - Trains model to abstain when annotators disagree
    - Does NOT penalize abstaining on clear samples in accuracy
    """

    def __init__(self, config: Optional[CEBaBConfig] = None):
        super().__init__()
        self.config = config or CEBaBConfig()

    def forward(
        self,
        task_logits: torch.Tensor,
        task_hard_labels: torch.Tensor,
        task_soft_labels: Optional[torch.Tensor] = None,
        concept_logits: Optional[torch.Tensor] = None,
        concept_soft_labels: Optional[torch.Tensor] = None,
        is_ambiguous: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss

        Args:
            task_logits: [batch, 6] task prediction logits
            task_hard_labels: [batch] hard labels (0-5)
            task_soft_labels: [batch, 6] soft label distributions
            concept_logits: [batch, 4, 3] concept prediction logits
            concept_soft_labels: [batch, 4, 3] soft concept distributions
            is_ambiguous: [batch] mask for 'no majority' samples

        Returns:
            Dictionary of losses including 'total_loss'
        """
        losses = {}
        device = task_logits.device

        # === Task Loss ===
        if self.config.use_soft_task_labels and task_soft_labels is not None:
            log_probs = F.log_softmax(task_logits, dim=-1)
            task_loss = -torch.sum(task_soft_labels.to(device) * log_probs, dim=-1).mean()
        else:
            task_loss = F.cross_entropy(task_logits, task_hard_labels.to(device))

        losses['task_loss'] = task_loss

        # === Abstain Calibration Loss ===
        if is_ambiguous is not None:
            is_ambiguous = is_ambiguous.to(device)
            abstain_probs = F.softmax(task_logits, dim=-1)[:, self.config.abstain_class]

            abstain_loss = torch.tensor(0.0, device=device)

            # Encourage abstaining on ambiguous samples (target = 1)
            if is_ambiguous.any():
                abstain_loss_pos = F.binary_cross_entropy(
                    abstain_probs[is_ambiguous],
                    torch.ones_like(abstain_probs[is_ambiguous])
                )
                abstain_loss = abstain_loss + abstain_loss_pos

            # Discourage abstaining on clear samples (target = 0)
            if (~is_ambiguous).any():
                abstain_loss_neg = F.binary_cross_entropy(
                    abstain_probs[~is_ambiguous],
                    torch.zeros_like(abstain_probs[~is_ambiguous])
                )
                abstain_loss = abstain_loss + abstain_loss_neg

            losses['abstain_loss'] = self.config.abstain_penalty * abstain_loss

        # === Concept Loss with Soft Labels ===
        if concept_logits is not None and concept_soft_labels is not None:
            concept_soft_labels = concept_soft_labels.to(device)
            log_probs = F.log_softmax(concept_logits, dim=-1)
            concept_loss = -torch.sum(concept_soft_labels * log_probs, dim=-1).mean()
            losses['concept_loss'] = self.config.concept_weight * concept_loss

        # === Total Loss ===
        losses['total_loss'] = sum(losses.values())

        return losses


# =============================================================
# EVALUATION METRICS
# =============================================================

def compute_cebab_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    is_ambiguous: np.ndarray,
    probs: Optional[np.ndarray] = None,
    abstain_class: int = 5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics with abstain calibration (SEPARATE from accuracy)

    Key principle:
    - Accuracy metrics: measure prediction quality
    - Abstain metrics: measure calibration of "I don't know"
    - These are SEPARATE concerns

    Args:
        predictions: [N] predicted class (0-5)
        labels: [N] true class (0-5)
        is_ambiguous: [N] boolean mask for 'no majority' samples
        probs: [N, 6] prediction probabilities (optional)
        abstain_class: Index of abstain class (5)

    Returns:
        Dictionary of metrics
    """
    N = len(predictions)

    # Masks
    pred_abstain = predictions == abstain_class
    true_abstain = is_ambiguous.astype(bool)
    clear_mask = ~true_abstain

    metrics = {}

    # ==========================================================
    # ACCURACY METRICS (your 69% → 76% target)
    # ==========================================================

    # Overall accuracy (all 6 classes, including abstain-as-class)
    metrics['accuracy_all'] = (predictions == labels).mean()

    # Accuracy on CLEAR samples only (true label is 0-4, not 'no majority')
    # This is likely what you want to optimize
    if clear_mask.sum() > 0:
        metrics['accuracy_clear'] = (predictions[clear_mask] == labels[clear_mask]).mean()
    else:
        metrics['accuracy_clear'] = 0.0

    # 5-class accuracy: on samples where model predicts 0-4 AND true is 0-4
    valid_5class = (~pred_abstain) & clear_mask
    if valid_5class.sum() > 0:
        metrics['accuracy_5class'] = (predictions[valid_5class] == labels[valid_5class]).mean()
    else:
        metrics['accuracy_5class'] = 0.0

    # Selective accuracy: accuracy when model chooses to predict (doesn't abstain)
    if (~pred_abstain).sum() > 0:
        metrics['accuracy_selective'] = (predictions[~pred_abstain] == labels[~pred_abstain]).mean()
    else:
        metrics['accuracy_selective'] = 0.0

    # ==========================================================
    # ABSTAIN CALIBRATION METRICS (separate from accuracy)
    # ==========================================================

    # Abstain precision: P(true ambiguous | predicted abstain)
    # "When I say I don't know, am I right to be uncertain?"
    if pred_abstain.sum() > 0:
        metrics['abstain_precision'] = true_abstain[pred_abstain].mean()
    else:
        metrics['abstain_precision'] = 0.0

    # Abstain recall: P(predicted abstain | true ambiguous)
    # "Do I recognize all the ambiguous cases?"
    if true_abstain.sum() > 0:
        metrics['abstain_recall'] = pred_abstain[true_abstain].mean()
    else:
        metrics['abstain_recall'] = 0.0

    # Abstain F1
    p, r = metrics['abstain_precision'], metrics['abstain_recall']
    metrics['abstain_f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # Coverage: fraction of samples model makes a prediction on (doesn't abstain)
    metrics['coverage'] = (~pred_abstain).mean()

    # ==========================================================
    # PER-CLASS ACCURACY (for debugging)
    # ==========================================================
    for c in range(5):
        class_mask = labels == c
        if class_mask.sum() > 0:
            metrics[f'accuracy_class_{c+1}star'] = (predictions[class_mask] == c).mean()

    # ==========================================================
    # CALIBRATION METRICS (if probs provided)
    # ==========================================================
    if probs is not None and clear_mask.sum() > 0:
        # ECE on clear samples
        confidences = probs[clear_mask].max(axis=1)
        accuracies = (predictions[clear_mask] == labels[clear_mask]).astype(float)

        # 10-bin ECE
        bin_boundaries = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(10):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if in_bin.sum() > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].mean()
                ece += in_bin.sum() * abs(avg_conf - avg_acc)
        metrics['ece_clear'] = ece / clear_mask.sum()

    return metrics


def compute_abstain_uncertainty_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    is_ambiguous: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    abstain_class: int = 5
) -> Dict[str, float]:
    """
    Analyze relationship between abstain behavior and uncertainty

    Key insight for CREDENCE:
    - Abstaining should correlate with ALEATORIC uncertainty (annotator disagreement)
    - NOT with EPISTEMIC uncertainty (model ignorance)

    Args:
        predictions: [N] predicted class
        labels: [N] true class
        is_ambiguous: [N] 'no majority' mask
        epistemic: [N] or [N, K] epistemic uncertainty
        aleatoric: [N] or [N, K] aleatoric uncertainty

    Returns:
        Dictionary of correlation metrics
    """
    pred_abstain = predictions == abstain_class
    true_abstain = is_ambiguous.astype(bool)

    # Aggregate if per-concept
    if epistemic.ndim > 1:
        epistemic = epistemic.mean(axis=-1)
    if aleatoric.ndim > 1:
        aleatoric = aleatoric.mean(axis=-1)

    metrics = {}

    # === Mean Uncertainties by Abstain Status ===
    if pred_abstain.sum() > 0:
        metrics['epistemic_when_abstain'] = float(epistemic[pred_abstain].mean())
        metrics['aleatoric_when_abstain'] = float(aleatoric[pred_abstain].mean())
    else:
        metrics['epistemic_when_abstain'] = 0.0
        metrics['aleatoric_when_abstain'] = 0.0

    if (~pred_abstain).sum() > 0:
        metrics['epistemic_when_predict'] = float(epistemic[~pred_abstain].mean())
        metrics['aleatoric_when_predict'] = float(aleatoric[~pred_abstain].mean())
    else:
        metrics['epistemic_when_predict'] = 0.0
        metrics['aleatoric_when_predict'] = 0.0

    # === Correlations ===
    # Key: abstain should correlate MORE with aleatoric than epistemic
    if len(np.unique(pred_abstain)) > 1:
        rho_epi, p_epi = stats.spearmanr(pred_abstain.astype(float), epistemic)
        rho_ale, p_ale = stats.spearmanr(pred_abstain.astype(float), aleatoric)

        metrics['rho_abstain_epistemic'] = float(rho_epi)
        metrics['rho_abstain_aleatoric'] = float(rho_ale)
        metrics['p_abstain_epistemic'] = float(p_epi)
        metrics['p_abstain_aleatoric'] = float(p_ale)

        # KEY METRIC: Aleatoric dominance
        # Positive = abstain correlates more with aleatoric (GOOD)
        # Negative = abstain correlates more with epistemic (BAD)
        metrics['abstain_aleatoric_dominance'] = float(rho_ale - rho_epi)
    else:
        metrics['rho_abstain_epistemic'] = 0.0
        metrics['rho_abstain_aleatoric'] = 0.0
        metrics['abstain_aleatoric_dominance'] = 0.0

    # === Uncertainty on True Ambiguous vs Clear ===
    if true_abstain.sum() > 0:
        metrics['epistemic_on_true_ambiguous'] = float(epistemic[true_abstain].mean())
        metrics['aleatoric_on_true_ambiguous'] = float(aleatoric[true_abstain].mean())

    if (~true_abstain).sum() > 0:
        metrics['epistemic_on_true_clear'] = float(epistemic[~true_abstain].mean())
        metrics['aleatoric_on_true_clear'] = float(aleatoric[~true_abstain].mean())

    return metrics
