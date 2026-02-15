"""
Base components for Credal Concept Bottleneck Models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class ConceptAnnotation:
    """Structure for concept annotations"""
    concept_name: str
    concept_value: float  # 0-1 for binary, continuous for others
    uncertainty: float = 0.0
    annotator_confidence: float = 1.0

class CredalSet:
    """Enhanced credal set with more operations for real datasets"""
    def __init__(self, extreme_points: np.ndarray, concept_names: List[str] = None):
        self.extreme_points = extreme_points
        self.n_points, self.n_classes = extreme_points.shape
        self.concept_names = concept_names or [f"concept_{i}" for i in range(self.n_classes)]
        
        # Validate probability distributions
        assert np.allclose(extreme_points.sum(axis=1), 1.0), "Invalid probability distributions"
        assert np.all(extreme_points >= 0), "Negative probabilities detected"
    
    def entropy(self) -> float:
        """Calculate entropy-based uncertainty measure"""
        mean_dist = np.mean(self.extreme_points, axis=0)
        return -np.sum(mean_dist * np.log(mean_dist + 1e-8))
    
    def variance(self) -> np.ndarray:
        """Calculate variance across extreme points for each concept"""
        return np.var(self.extreme_points, axis=0)
    
    def size(self) -> float:
        """Calculate the size/spread of the credal set"""
        # Use the range between upper and lower probabilities as a measure of size
        ranges = np.max(self.extreme_points, axis=0) - np.min(self.extreme_points, axis=0)
        return np.mean(ranges)  # Average range across all classes
    
    def interval_probability(self, class_idx: int) -> Tuple[float, float]:
        """Get probability interval [lower, upper] for a specific class"""
        probs = self.extreme_points[:, class_idx]
        return float(np.min(probs)), float(np.max(probs))
    
    def lower_probability(self, class_idx: int) -> float:
        """Get lower probability for a specific class"""
        return np.min(self.extreme_points[:, class_idx])
    
    def upper_probability(self, class_idx: int) -> float:
        """Get upper probability for a specific class"""
        return np.max(self.extreme_points[:, class_idx])
    
    def credal_dominance_score(self, target_concept: int) -> float:
        """Score indicating how strongly a concept is supported"""
        lower = self.lower_probability(target_concept)
        upper = self.upper_probability(target_concept)
        return lower  # Conservative estimate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            'extreme_points': self.extreme_points.tolist(),
            'concept_names': self.concept_names,
            'entropy': self.entropy(),
            'variance': self.variance().tolist()
        }

class RealWorldCredalCBM(nn.Module):
    """
    Enhanced Credal CBM for real-world datasets with better uncertainty modeling
    """
    def __init__(self, input_dim: int, concept_names: List[str], class_names: List[str],
                 n_credal_points: int = 5, hidden_dim: int = 256, 
                 uncertainty_method: str = 'ensemble'):
        super().__init__()
        
        self.input_dim = input_dim
        self.concept_names = concept_names
        self.class_names = class_names
        self.n_concepts = len(concept_names)
        self.n_classes = len(class_names)
        self.n_credal_points = n_credal_points
        self.uncertainty_method = uncertainty_method
        
        # Enhanced concept encoder with multiple heads for credal estimation
        self.concept_encoder_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multiple heads for credal set estimation
        if uncertainty_method == 'ensemble':
            self.concept_heads = nn.ModuleList([
                nn.Linear(hidden_dim // 2, self.n_concepts)
                for _ in range(n_credal_points)
            ])
        else:
            # Single head with uncertainty estimation
            self.concept_head = nn.Linear(hidden_dim // 2, self.n_concepts)
            self.uncertainty_head = nn.Linear(hidden_dim // 2, self.n_concepts)
        
        # Label predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(self.n_concepts, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, self.n_classes)
        )
        
        # Concept importance weights (learnable)
        self.concept_importance = nn.Parameter(torch.ones(self.n_concepts))
    
    def forward(self, x: torch.Tensor) -> Tuple[List[List[CredalSet]], torch.Tensor, Dict]:
        """Enhanced forward pass with detailed uncertainty information"""
        batch_size = x.shape[0]
        
        # Shared encoding
        shared_features = self.concept_encoder_shared(x)
        
        # Generate credal sets
        if self.uncertainty_method == 'ensemble':
            # Multiple predictions for ensemble-based credal sets
            concept_logits = []
            for head in self.concept_heads:
                logits = torch.sigmoid(head(shared_features))
                concept_logits.append(logits)
            concept_logits = torch.stack(concept_logits, dim=1)  # [batch, n_heads, n_concepts]
        else:
            # Single prediction with learned uncertainty
            concept_mean = torch.sigmoid(self.concept_head(shared_features))
            concept_std = torch.softplus(self.uncertainty_head(shared_features)) + 1e-6
            
            # Sample multiple points for credal set
            concept_logits = []
            for _ in range(self.n_credal_points):
                noise = torch.randn_like(concept_mean)
                sample = torch.sigmoid(concept_mean + concept_std * noise)
                concept_logits.append(sample)
            concept_logits = torch.stack(concept_logits, dim=1)
        
        # Convert to credal sets
        batch_credal_sets = []
        concept_expectations = []
        
        for b in range(batch_size):
            sample_credal_sets = []
            sample_expectations = []
            
            for c in range(self.n_concepts):
                # Get predictions for this concept across all heads/samples
                concept_predictions = concept_logits[b, :, c].detach().cpu().numpy()
                
                # Create binary credal set (present/absent)
                extreme_points = np.column_stack([
                    1 - concept_predictions,  # P(concept absent)
                    concept_predictions       # P(concept present)
                ])
                
                credal_set = CredalSet(extreme_points, [f"¬{self.concept_names[c]}", self.concept_names[c]])
                sample_credal_sets.append(credal_set)
                
                # Use expectation for downstream prediction
                expectation = np.mean(concept_predictions)
                sample_expectations.append(expectation)
            
            batch_credal_sets.append(sample_credal_sets)
            concept_expectations.append(sample_expectations)
        
        # Final prediction using concept expectations
        concept_tensor = torch.tensor(concept_expectations, device=x.device)
        
        # Apply concept importance weighting
        weighted_concepts = concept_tensor * torch.sigmoid(self.concept_importance)
        final_predictions = self.label_predictor(weighted_concepts)
        
        # Additional uncertainty metrics
        uncertainty_metrics = {
            'concept_importance': torch.sigmoid(self.concept_importance).detach().cpu().numpy(),
            'prediction_entropy': self._compute_prediction_entropy(final_predictions),
            'concept_uncertainties': self._compute_concept_uncertainties(batch_credal_sets)
        }
        
        return batch_credal_sets, final_predictions, uncertainty_metrics
    
    def _compute_prediction_entropy(self, predictions: torch.Tensor) -> np.ndarray:
        """Compute prediction entropy for each sample"""
        probs = F.softmax(predictions, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.detach().cpu().numpy()
    
    def _compute_concept_uncertainties(self, batch_credal_sets: List[List[CredalSet]]) -> np.ndarray:
        """Compute uncertainty for each concept across batch"""
        batch_size = len(batch_credal_sets)
        concept_uncertainties = np.zeros((batch_size, self.n_concepts))
        
        for b, credal_sets in enumerate(batch_credal_sets):
            for c, credal_set in enumerate(credal_sets):
                concept_uncertainties[b, c] = credal_set.size()
        
        return concept_uncertainties 