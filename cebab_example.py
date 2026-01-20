import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Optional, Tuple
import numpy as np
import math

import torch_concepts as pyc


import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


# =============================================================
# CONFIGURATION
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
    use_soft_task_labels: bool = False  # Use review_label_distribution
    use_soft_concept_labels: bool = True  # Use aspect_label_distribution
    soft_label_temperature: float = 1.5  # T > 1 preserves disagreement
    
    # Loss weights
    abstain_penalty: float = 0.1  # Weight for abstain regularization
    concept_weight: float = 1.0   # Weight for concept loss
    
    # Aspect names
    aspect_names: Tuple[str, ...] = ('food', 'service', 'ambiance', 'noise')


# =============================================================
# LABEL PROCESSOR
# =============================================================

class CEBaBLabelProcessor:
    """
    Process CEBaB labels with:
    - 6 task classes (1-5 stars + abstain for 'no majority')
    - Soft labels from annotator distributions
    - Soft concept labels from aspect distributions
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
            Dictionary with processed labels ready for training
        """
        task_labels = self.process_task_labels(batch)
        concept_labels = self.process_concept_labels(batch)
        
        return {
            **task_labels,
            **{f'concept_{k}': v for k, v in concept_labels.items()}
        }
    
    def process_task_labels(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Process task (review) labels
        
        Returns:
            hard_labels: [batch_size] integer labels (0-5)
            soft_labels: [batch_size, 6] probability distributions
            is_ambiguous: [batch_size] boolean mask for 'no majority' samples
        """
        labels_batch = batch['review_majority']
        distributions = batch.get('review_label_distribution', None)
        
        batch_size = len(labels_batch)
        hard_labels = torch.zeros(batch_size, dtype=torch.long)
        soft_labels = torch.zeros(batch_size, self.config.num_task_classes)
        is_ambiguous = torch.zeros(batch_size, dtype=torch.bool)
        
        for i, label in enumerate(labels_batch):
            # Handle tensor vs string
            if isinstance(label, torch.Tensor):
                label = label.item() if label.dim() == 0 else str(label)
            label = str(label)
            
            # Hard label
            hard_labels[i] = self.TASK_LABEL_MAP.get(label, self.config.abstain_class)
            is_ambiguous[i] = (label == 'no majority')
            
            # Soft label from distribution
            if self.config.use_soft_task_labels and distributions is not None:
                dist = distributions[i] if isinstance(distributions, (list, tuple)) else distributions
                soft = self._task_distribution_to_soft_label(dist)
                soft_labels[i] = torch.tensor(soft)
            else:
                # One-hot from hard label
                soft_labels[i, hard_labels[i]] = 1.0
        
        return {
            'task_hard_labels': hard_labels,
            'task_soft_labels': soft_labels,
            'is_ambiguous': is_ambiguous,
        }
    
    def process_concept_labels(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Process concept (aspect) labels with soft distributions
        
        Returns:
            hard_labels: [batch_size, 4] integer labels (0-2)
            soft_labels: [batch_size, 4, 3] probability distributions
        """
        # Determine batch size from first aspect
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
                majority = batch[majority_key][j]
                if isinstance(majority, torch.Tensor):
                    majority = majority.item() if majority.dim() == 0 else str(majority)
                majority = str(majority) if not isinstance(majority, str) else majority
                
                # Hard label
                hard_labels[j, i] = self.CONCEPT_LABEL_MAP.get(majority, 1)  # default unknown
                
                # Soft label from distribution
                if self.config.use_soft_concept_labels and dist_key in batch:
                    dist = batch[dist_key][j]
                    soft = self._concept_distribution_to_soft_label(dist)
                    soft_labels[j, i] = torch.tensor(soft)
                else:
                    # One-hot
                    soft_labels[j, i, hard_labels[j, i]] = 1.0
        
        return {
            'hard_labels': hard_labels,
            'soft_labels': soft_labels,
        }
    
    def _task_distribution_to_soft_label(self, dist: Dict) -> List[float]:
        """Convert review_label_distribution to 6-class soft label"""
        if dist is None or not isinstance(dist, dict):
            # Uniform over 5 classes, 0 for abstain
            return [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        
        # Extract counts for classes 1-5
        counts = [float(dist.get(str(i), 0)) for i in range(1, 6)]
        total = sum(counts)
        
        if total == 0:
            return [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        
        # Normalize
        probs = [c / total for c in counts]
        
        # Temperature scaling
        if self.config.soft_label_temperature != 1.0:
            probs = self._apply_temperature(probs)
        
        # Add 0 for abstain class
        probs.append(0.0)
        
        return probs
    
    def _concept_distribution_to_soft_label(self, dist: Dict) -> List[float]:
        """Convert aspect_label_distribution to 3-class soft label"""
        if dist is None or not isinstance(dist, dict):
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
        if self.config.soft_label_temperature != 1.0:
            probs = self._apply_temperature(probs)
        
        return probs
    
    def _apply_temperature(self, probs: List[float]) -> List[float]:
        """Apply temperature scaling to probability distribution"""
        T = self.config.soft_label_temperature
        
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
    - Cross-entropy for task prediction (with optional soft labels)
    - Soft cross-entropy for concept prediction
    - Abstain calibration regularization
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
            # Soft label cross-entropy (KL divergence style)
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
            
            # Encourage abstaining on ambiguous samples
            if is_ambiguous.any():
                # Use BCE: target=1 for ambiguous (should abstain)
                abstain_loss_pos = F.binary_cross_entropy(
                    abstain_probs[is_ambiguous],
                    torch.ones_like(abstain_probs[is_ambiguous])
                )
                abstain_loss = abstain_loss + abstain_loss_pos
            
            # Discourage abstaining on clear samples
            if (~is_ambiguous).any():
                # Use BCE: target=0 for clear (should not abstain)
                abstain_loss_neg = F.binary_cross_entropy(
                    abstain_probs[~is_ambiguous],
                    torch.zeros_like(abstain_probs[~is_ambiguous])
                )
                abstain_loss = abstain_loss + abstain_loss_neg
            
            losses['abstain_loss'] = self.config.abstain_penalty * abstain_loss
        
        # === Concept Loss with Soft Labels ===
        if concept_logits is not None and concept_soft_labels is not None:
            concept_soft_labels = concept_soft_labels.to(device)
            # concept_logits: [batch, num_concepts, num_classes]
            # concept_soft_labels: [batch, num_concepts, num_classes]
            
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
    Compute comprehensive metrics with abstain calibration
    
    Args:
        predictions: [N] predicted class (0-5)
        labels: [N] true class (0-5)  
        is_ambiguous: [N] boolean mask for 'no majority' samples
        probs: [N, 6] prediction probabilities (optional)
        abstain_class: Index of abstain class
        
    Returns:
        Dictionary of metrics
    """
    N = len(predictions)
    
    # Masks
    pred_abstain = predictions == abstain_class
    true_abstain = is_ambiguous.astype(bool)
    clear_mask = ~true_abstain
    
    metrics = {}
    
    # === Accuracy Metrics ===
    
    # Overall accuracy (all 6 classes)
    metrics['accuracy_all'] = (predictions == labels).mean()
    
    # Accuracy on clear samples only (excluding 'no majority' ground truth)
    if clear_mask.sum() > 0:
        metrics['accuracy_clear'] = (predictions[clear_mask] == labels[clear_mask]).mean()
    else:
        metrics['accuracy_clear'] = 0.0
    
    # 5-class accuracy: only on samples where model didn't abstain AND true label isn't abstain
    valid_5class = (~pred_abstain) & clear_mask
    if valid_5class.sum() > 0:
        metrics['accuracy_5class'] = (predictions[valid_5class] == labels[valid_5class]).mean()
    else:
        metrics['accuracy_5class'] = 0.0
    
    # === Abstain Calibration Metrics ===
    
    # Abstain precision: P(true ambiguous | predicted abstain)
    if pred_abstain.sum() > 0:
        metrics['abstain_precision'] = true_abstain[pred_abstain].mean()
    else:
        metrics['abstain_precision'] = 0.0
    
    # Abstain recall: P(predicted abstain | true ambiguous)
    if true_abstain.sum() > 0:
        metrics['abstain_recall'] = pred_abstain[true_abstain].mean()
    else:
        metrics['abstain_recall'] = 0.0
    
    # Abstain F1
    if metrics['abstain_precision'] + metrics['abstain_recall'] > 0:
        metrics['abstain_f1'] = 2 * (metrics['abstain_precision'] * metrics['abstain_recall']) / \
                                (metrics['abstain_precision'] + metrics['abstain_recall'])
    else:
        metrics['abstain_f1'] = 0.0
    
    # Coverage: fraction of samples not abstained on
    metrics['coverage'] = (~pred_abstain).mean()
    
    # === Per-Class Metrics (5 sentiment classes) ===
    for c in range(5):
        class_mask = labels == c
        if class_mask.sum() > 0:
            metrics[f'accuracy_class_{c+1}'] = (predictions[class_mask] == c).mean()
    
    # === Confidence Calibration (if probs provided) ===
    if probs is not None:
        # Expected Calibration Error on clear samples
        if clear_mask.sum() > 0:
            confidences = probs[clear_mask].max(axis=1)
            accuracies = (predictions[clear_mask] == labels[clear_mask]).astype(float)
            
            # Bin into 10 bins
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


def compute_abstain_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    is_ambiguous: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    abstain_class: int = 5
) -> Dict[str, float]:
    """
    Analyze relationship between abstain behavior and uncertainty
    
    This is key for CREDENCE: does the model abstain due to epistemic or aleatoric uncertainty?
    
    Args:
        predictions: [N] predicted class
        labels: [N] true class
        is_ambiguous: [N] 'no majority' mask
        epistemic: [N] epistemic uncertainty scores
        aleatoric: [N] aleatoric uncertainty scores
        
    Returns:
        Dictionary of analysis metrics
    """
    pred_abstain = predictions == abstain_class
    true_abstain = is_ambiguous.astype(bool)
    
    metrics = {}
    
    # Mean uncertainties by abstain status
    if pred_abstain.sum() > 0:
        metrics['epistemic_when_abstain'] = epistemic[pred_abstain].mean()
        metrics['aleatoric_when_abstain'] = aleatoric[pred_abstain].mean()
    
    if (~pred_abstain).sum() > 0:
        metrics['epistemic_when_predict'] = epistemic[~pred_abstain].mean()
        metrics['aleatoric_when_predict'] = aleatoric[~pred_abstain].mean()
    
    # Correlation: does abstaining correlate with aleatoric (as it should)?
    from scipy.stats import pearsonr, spearmanr
    
    if len(np.unique(pred_abstain)) > 1:  # Need variance
        rho_abstain_epistemic, _ = spearmanr(pred_abstain.astype(float), epistemic)
        rho_abstain_aleatoric, _ = spearmanr(pred_abstain.astype(float), aleatoric)
        
        metrics['rho_abstain_epistemic'] = rho_abstain_epistemic
        metrics['rho_abstain_aleatoric'] = rho_abstain_aleatoric
        
        # Key insight: abstain should correlate MORE with aleatoric than epistemic
        metrics['abstain_aleatoric_dominance'] = rho_abstain_aleatoric - rho_abstain_epistemic
    
    # On true ambiguous samples: what's the uncertainty profile?
    if true_abstain.sum() > 0:
        metrics['epistemic_on_ambiguous'] = epistemic[true_abstain].mean()
        metrics['aleatoric_on_ambiguous'] = aleatoric[true_abstain].mean()
    
    if (~true_abstain).sum() > 0:
        metrics['epistemic_on_clear'] = epistemic[~true_abstain].mean()
        metrics['aleatoric_on_clear'] = aleatoric[~true_abstain].mean()
    
    return metrics


# =============================================================
# INTEGRATION WITH VCBM TRAINING
# =============================================================

def train_epoch_cebab(
    model,  # VariationalCredalCBM
    dataloader,
    optimizer,
    scheduler=None,
    device: str = 'cuda',
    tokenizer=None,
    label_processor: Optional[CEBaBLabelProcessor] = None,
    loss_fn: Optional[CEBaBLoss] = None
) -> Dict[str, float]:
    """
    Train for one epoch on CEBaB with proper label handling
    
    Args:
        model: VariationalCredalCBM model
        dataloader: CEBaB dataloader
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        device: Device to train on
        tokenizer: Tokenizer for encoding text
        label_processor: CEBaBLabelProcessor instance
        loss_fn: CEBaBLoss instance
        
    Returns:
        Dictionary of average losses
    """
    from tqdm import tqdm
    
    model.train()
    
    # Defaults
    if label_processor is None:
        label_processor = CEBaBLabelProcessor()
    if loss_fn is None:
        loss_fn = CEBaBLoss(label_processor.config)
    
    # Tracking
    total_losses = {
        'total_loss': 0.0,
        'task_loss': 0.0,
        'abstain_loss': 0.0,
        'concept_loss': 0.0,
        'kl_loss': 0.0,
    }
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # === Tokenize text ===
        texts = batch['description']
        max_len = getattr(dataloader.dataset, 'max_length', 128)
        
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # === Process labels ===
        processed = label_processor.process_batch(batch)
        
        task_hard_labels = processed['task_hard_labels'].to(device)
        task_soft_labels = processed['task_soft_labels'].to(device)
        is_ambiguous = processed['is_ambiguous'].to(device)
        concept_hard_labels = processed['concept_hard_labels'].to(device)
        concept_soft_labels = processed['concept_soft_labels'].to(device)
        
        # === Forward pass ===
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=task_hard_labels,
            concept_labels=concept_hard_labels
        )
        
        # === Compute loss ===
        # Get logits from model output
        task_logits = outputs.get('logits', outputs.get('task_logits'))
        concept_logits = outputs.get('concept_logits')
        
        losses = loss_fn(
            task_logits=task_logits,
            task_hard_labels=task_hard_labels,
            task_soft_labels=task_soft_labels,
            concept_logits=concept_logits,
            concept_soft_labels=concept_soft_labels,
            is_ambiguous=is_ambiguous
        )
        
        # Add model's internal losses (KL, etc.)
        model_loss = outputs.get('loss', torch.tensor(0.0))
        kl_loss = outputs.get('kl_loss', torch.tensor(0.0))
        
        # Combine: use our task/concept/abstain losses + model's KL
        if isinstance(kl_loss, torch.Tensor):
            total_loss = losses['total_loss'] + kl_loss
        else:
            total_loss = losses['total_loss']
        
        # === Backward pass ===
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # === Track losses ===
        total_losses['total_loss'] += total_loss.item()
        total_losses['task_loss'] += losses['task_loss'].item()
        total_losses['abstain_loss'] += losses.get('abstain_loss', torch.tensor(0.0)).item()
        total_losses['concept_loss'] += losses.get('concept_loss', torch.tensor(0.0)).item()
        if isinstance(kl_loss, torch.Tensor):
            total_losses['kl_loss'] += kl_loss.item()
        
        num_batches += 1
    
    # Average
    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def evaluate_cebab(
    model,  # VariationalCredalCBM
    dataloader,
    device: str = 'cuda',
    tokenizer=None,
    label_processor: Optional[CEBaBLabelProcessor] = None
) -> Dict[str, float]:
    """
    Evaluate on CEBaB with comprehensive metrics
    
    Args:
        model: VariationalCredalCBM model
        dataloader: CEBaB dataloader
        device: Device
        tokenizer: Tokenizer
        label_processor: Label processor
        
    Returns:
        Dictionary of evaluation metrics
    """
    from tqdm import tqdm
    
    model.eval()
    
    if label_processor is None:
        label_processor = CEBaBLabelProcessor()
    
    # Collectors
    all_predictions = []
    all_labels = []
    all_probs = []
    all_is_ambiguous = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # === Tokenize ===
            texts = batch['description']
            max_len = getattr(dataloader.dataset, 'max_length', 128)
            
            encoded = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # === Process labels ===
            processed = label_processor.process_batch(batch)
            task_hard_labels = processed['task_hard_labels']
            is_ambiguous = processed['is_ambiguous']
            
            # === Forward pass ===
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # === Collect results ===
            predictions = outputs['predictions'].cpu().numpy()
            probs = outputs['probs'].cpu().numpy()
            epistemic = outputs['epistemic'].cpu().numpy()
            aleatoric = outputs['aleatoric'].cpu().numpy()
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(task_hard_labels.numpy().tolist())
            all_probs.extend(probs.tolist())
            all_is_ambiguous.extend(is_ambiguous.numpy().tolist())
            all_epistemic.extend(epistemic.tolist())
            all_aleatoric.extend(aleatoric.tolist())
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_is_ambiguous = np.array(all_is_ambiguous)
    all_epistemic = np.array(all_epistemic)
    all_aleatoric = np.array(all_aleatoric)
    
    # === Compute metrics ===
    basic_metrics = compute_cebab_metrics(
        predictions=all_predictions,
        labels=all_labels,
        is_ambiguous=all_is_ambiguous,
        probs=all_probs,
        abstain_class=label_processor.config.abstain_class
    )
    
    abstain_analysis = compute_abstain_analysis(
        predictions=all_predictions,
        labels=all_labels,
        is_ambiguous=all_is_ambiguous,
        epistemic=all_epistemic,
        aleatoric=all_aleatoric,
        abstain_class=label_processor.config.abstain_class
    )
    
    return {**basic_metrics, **abstain_analysis}


# =============================================================
# EXAMPLE USAGE
# =============================================================

if __name__ == "__main__":
    # Example configuration
    config = CEBaBConfig(
        use_soft_task_labels=False,  # Hard labels for task
        use_soft_concept_labels=True,  # Soft labels for concepts
        soft_label_temperature=1.5,  # Preserve disagreement
        abstain_penalty=0.1,
        concept_weight=1.0
    )
    
    # Create processor and loss
    processor = CEBaBLabelProcessor(config)
    loss_fn = CEBaBLoss(config)
    
    print("CEBaB Label Processor initialized")
    print(f"  Task classes: {config.num_task_classes} (including abstain)")
    print(f"  Concept classes: {config.num_concept_classes}")
    print(f"  Soft concept labels: {config.use_soft_concept_labels}")
    print(f"  Temperature: {config.soft_label_temperature}")
