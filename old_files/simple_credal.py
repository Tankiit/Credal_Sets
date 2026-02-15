"""
Credal Concept Bottleneck Models with Real Datasets
Demonstrates uncertainty-aware CBMs on Animals with Attributes, CUB-200, and other datasets
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import os
from typing import List, Tuple, Dict, Optional
import warnings
from dataclasses import dataclass
from scipy.optimize import linprog
import tqdm
warnings.filterwarnings('ignore')

# Import base components
from credal_base import CredalSet, ConceptAnnotation, RealWorldCredalCBM

# Import MMD components
from mmd_credal import MMDEnhancedCredalCBM, train_mmd_enhanced_credal_cbm, analyze_concept_stability

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

class AnimalsWithAttributesDataset(Dataset):
    """
    Animals with Attributes dataset loader
    Downloads and processes AwA2 dataset with concept annotations
    """
    def __init__(self, root_dir: str = "./awa_data", train: bool = True, 
                 transform: Optional[transforms.Compose] = None,
                 download: bool = True):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform or self.default_transform()
        
        if download:
            self._download_dataset()
        
        self._load_data()
    
    def default_transform(self):
        """Default image preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _download_dataset(self):
        """Download and prepare AwA dataset"""
        os.makedirs(self.root_dir, exist_ok=True)
        
        # For demo purposes, create synthetic AwA-like data
        # In practice, you'd download from the actual AwA dataset
        self._create_synthetic_awa_data()
    
    def _create_synthetic_awa_data(self):
        """Create synthetic AwA-like data for demonstration"""
        np.random.seed(42)
        
        # Animal classes (subset of AwA)
        self.animal_classes = [
            'antelope', 'grizzly_bear', 'killer_whale', 'beaver', 'dalmatian',
            'persian_cat', 'horse', 'german_shepherd', 'blue_whale', 'siamese_cat',
            'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose',
            'spider_monkey', 'humpback_whale', 'elephant', 'gorilla'
        ]
        
        # Concept attributes (from AwA)
        self.concept_names = [
            'black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow',
            'patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin',
            'big', 'small', 'bulbous', 'lean', 'flippers', 'hands', 'hooves',
            'pads', 'paws', 'longleg', 'longneck', 'tail', 'chewteeth',
            'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws', 'tusks',
            'smelly', 'flys', 'hops', 'swims', 'tunnels', 'walks', 'fast',
            'slow', 'strong', 'weak', 'muscle', 'bipedal', 'quadrapedal',
            'active', 'inactive', 'nocturnal', 'hibernate', 'agility', 'fish',
            'meat', 'plankton', 'vegetation', 'insects', 'forager', 'grazer',
            'hunter', 'scavenger', 'skimmer', 'stalker', 'newworld', 'oldworld',
            'arctic', 'coastal', 'desert', 'bush', 'plains', 'forest',
            'fields', 'jungle', 'mountains', 'ocean', 'ground', 'water',
            'tree', 'cave', 'fierce', 'timid', 'smart', 'group', 'solitary',
            'nestspot', 'domestic'
        ]
        
        n_animals = len(self.animal_classes)
        n_concepts = len(self.concept_names)
        n_samples_per_class = 50
        
        # Create realistic concept-animal associations
        self.concept_matrix = self._create_realistic_concept_matrix(n_animals, n_concepts)
        
        # Generate synthetic features and labels
        self.features = []
        self.concepts = []
        self.labels = []
        self.image_paths = []
        
        for class_idx, animal in enumerate(self.animal_classes):
            for sample_idx in range(n_samples_per_class):
                # Generate features (simulating CNN features)
                feature = np.random.randn(2048) * 0.1 + self.concept_matrix[class_idx] @ np.random.randn(n_concepts, 2048) * 0.05
                
                # Add noise to concept annotations (simulating annotation uncertainty)
                noise_level = 0.1
                concept_vector = self.concept_matrix[class_idx] + np.random.randn(n_concepts) * noise_level
                concept_vector = np.clip(concept_vector, 0, 1)
                
                self.features.append(feature)
                self.concepts.append(concept_vector)
                self.labels.append(class_idx)
                self.image_paths.append(f"{animal}_{sample_idx}.jpg")
        
        # Convert to numpy arrays
        self.features = np.array(self.features, dtype=np.float32)
        self.concepts = np.array(self.concepts, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Train/test split
        n_total = len(self.features)
        n_train = int(0.8 * n_total)
        
        if self.train:
            self.features = self.features[:n_train]
            self.concepts = self.concepts[:n_train]
            self.labels = self.labels[:n_train]
            self.image_paths = self.image_paths[:n_train]
        else:
            self.features = self.features[n_train:]
            self.concepts = self.concepts[n_train:]
            self.labels = self.labels[n_train:]
            self.image_paths = self.image_paths[n_train:]
    
    def _create_realistic_concept_matrix(self, n_animals: int, n_concepts: int) -> np.ndarray:
        """Create realistic animal-concept associations"""
        np.random.seed(42)
        
        # Start with random baseline
        concept_matrix = np.random.beta(2, 5, (n_animals, n_concepts))
        
        # Add realistic patterns
        for i, animal in enumerate(self.animal_classes):
            # Color concepts
            if 'dalmatian' in animal:
                concept_matrix[i, self.concept_names.index('black')] = 0.9
                concept_matrix[i, self.concept_names.index('white')] = 0.9
                concept_matrix[i, self.concept_names.index('spots')] = 0.95
            
            if 'tiger' in animal:
                concept_matrix[i, self.concept_names.index('orange')] = 0.9
                concept_matrix[i, self.concept_names.index('stripes')] = 0.95
                concept_matrix[i, self.concept_names.index('fierce')] = 0.9
            
            # Size concepts
            if any(large_animal in animal for large_animal in ['elephant', 'whale', 'bear', 'moose']):
                concept_matrix[i, self.concept_names.index('big')] = 0.9
                concept_matrix[i, self.concept_names.index('small')] = 0.1
            
            # Habitat concepts
            if 'whale' in animal:
                concept_matrix[i, self.concept_names.index('ocean')] = 0.95
                concept_matrix[i, self.concept_names.index('swims')] = 0.95
                concept_matrix[i, self.concept_names.index('big')] = 0.9
        
        return concept_matrix
    
    def _load_data(self):
        """Load preprocessed data"""
        pass  # Data is created in _create_synthetic_awa_data
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        concept = torch.tensor(self.concepts[idx])
        label = torch.tensor(self.labels[idx])
        
        return {
            'features': feature,
            'concepts': concept,
            'label': label,
            'animal_name': self.animal_classes[label.item()],
            'image_path': self.image_paths[idx]
        }
    
    def get_concept_names(self) -> List[str]:
        """Get list of concept names"""
        return self.concept_names
    
    def get_class_names(self) -> List[str]:
        """Get list of animal class names"""
        return self.animal_classes

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

def train_real_world_credal_cbm(model, train_loader, val_loader, epochs=50,
                               device='cpu'):
    """Enhanced training with validation and uncertainty regularization"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    concept_loss_fn = nn.BCELoss()
    prediction_loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for features, concepts, labels in train_loader:
            features = features.to(device)
            concepts = concepts.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            credal_sets, predictions, metrics = model(features)
            
            # Compute concept loss using expectations
            concept_expectations = []
            for sample_credal_sets in credal_sets:
                sample_exp = [cs.credal_dominance_score(1) for cs in sample_credal_sets]
                concept_expectations.append(sample_exp)
            
            concept_exp_tensor = torch.tensor(concept_expectations, device=device)
            concept_loss = concept_loss_fn(concept_exp_tensor, concepts)
            
            # Prediction loss
            prediction_loss = prediction_loss_fn(predictions, labels)
            
            # Uncertainty regularization
            uncertainty_reg = 0
            for sample_credal_sets in credal_sets:
                for credal_set in sample_credal_sets:
                    uncertainty_reg += max(0, 0.1 - credal_set.size())
            uncertainty_reg = uncertainty_reg / (len(credal_sets) * len(credal_sets[0]))
            
            # Total loss
            total_loss = concept_loss + prediction_loss + 0.01 * uncertainty_reg
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, concepts, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                _, predictions, _ = model(features)
                
                _, predicted = torch.max(predictions, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        avg_loss = epoch_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    
    return train_losses, val_accuracies

def analyze_concept_uncertainty(model, dataset, device='cpu', n_samples=100):
    """Analyze concept uncertainty patterns using pre-extracted features"""
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    all_credal_sets = []
    all_labels = []
    all_predictions = []
    all_metrics = []
    samples_seen = 0
    
    with torch.no_grad():
        for features, concepts, labels in dataloader:
            if samples_seen >= n_samples:
                break
            
            features = features.to(device)
            credal_sets, predictions, metrics = model(features)
            
            all_credal_sets.extend(credal_sets)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_metrics.append(metrics)
            
            samples_seen += features.size(0)
    
    # Analysis 1: Uncertainty by concept
    concept_uncertainties = np.zeros((len(all_credal_sets), len(dataset.get_concept_names())))
    concept_values = np.zeros((len(all_credal_sets), len(dataset.get_concept_names())))
    
    for i, credal_sets in enumerate(all_credal_sets):
        for j, credal_set in enumerate(credal_sets):
            concept_uncertainties[i, j] = credal_set.size()
            concept_values[i, j] = credal_set.credal_dominance_score(1)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average uncertainty per concept
    avg_uncertainty = np.mean(concept_uncertainties, axis=0)
    concept_indices = np.argsort(avg_uncertainty)[::-1][:20]  # Top 20 most uncertain
    
    axes[0, 0].bar(range(len(concept_indices)), avg_uncertainty[concept_indices])
    axes[0, 0].set_xlabel('Concept Index')
    axes[0, 0].set_ylabel('Average Uncertainty')
    axes[0, 0].set_title('Most Uncertain Concepts')
    axes[0, 0].set_xticks(range(len(concept_indices)))
    axes[0, 0].set_xticklabels([dataset.get_concept_names()[i] for i in concept_indices], 
                               rotation=45, ha='right')
    
    # Plot 2: Uncertainty vs Concept Value correlation
    axes[0, 1].scatter(concept_values.flatten(), concept_uncertainties.flatten(), 
                       alpha=0.5, s=1)
    axes[0, 1].set_xlabel('Concept Activation')
    axes[0, 1].set_ylabel('Uncertainty')
    axes[0, 1].set_title('Uncertainty vs Activation')
    
    # Plot 3: Uncertainty by animal class
    class_uncertainties = []
    for class_idx in range(len(dataset.get_class_names())):
        class_mask = np.array(all_labels) == class_idx
        if np.any(class_mask):
            class_unc = np.mean(concept_uncertainties[class_mask])
            class_uncertainties.append((dataset.get_class_names()[class_idx], class_unc))
    
    class_uncertainties.sort(key=lambda x: x[1], reverse=True)
    class_names, class_uncs = zip(*class_uncertainties[:15])
    
    axes[1, 0].barh(range(len(class_names)), class_uncs)
    axes[1, 0].set_xlabel('Average Uncertainty')
    axes[1, 0].set_ylabel('Animal Class')
    axes[1, 0].set_title('Most Uncertain Animal Classes')
    axes[1, 0].set_yticks(range(len(class_names)))
    axes[1, 0].set_yticklabels(class_names)
    
    # Plot 4: Concept importance from model
    if all_metrics:
        concept_importance = all_metrics[0]['concept_importance']
        important_concepts = np.argsort(concept_importance)[::-1][:20]
        
        axes[1, 1].bar(range(len(important_concepts)), concept_importance[important_concepts])
        axes[1, 1].set_xlabel('Concept Index')
        axes[1, 1].set_ylabel('Learned Importance')
        axes[1, 1].set_title('Most Important Concepts (Learned)')
        axes[1, 1].set_xticks(range(len(important_concepts)))
        axes[1, 1].set_xticklabels([dataset.get_concept_names()[i] for i in important_concepts], 
                                   rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'concept_uncertainties': concept_uncertainties,
        'concept_values': concept_values,
        'most_uncertain_concepts': [dataset.get_concept_names()[i] for i in concept_indices[:10]],
        'most_uncertain_classes': class_names[:10]
    }

def demonstrate_uncertainty_guided_annotation(model, dataset, uncertainty_threshold=0.3, device='cpu'):
    """Show uncertainty-guided annotation with pre-extracted features"""
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("🎯 Uncertainty-Guided Annotation Demonstration")
    print("=" * 60)
    
    samples_analyzed = 0
    high_uncertainty_samples = []
    
    with torch.no_grad():
        for features, concepts, labels in dataloader:
            if samples_analyzed >= 20:  # Analyze first 20 samples
                break
            
            features = features.to(device)
            credal_sets, predictions, metrics = model(features)
            sample_credal_sets = credal_sets[0]
            
            # Find uncertain concepts
            uncertain_concepts = []
            for i, credal_set in enumerate(sample_credal_sets):
                if credal_set.size() > uncertainty_threshold:
                    uncertain_concepts.append({
                        'concept': dataset.get_concept_names()[i],
                        'uncertainty': credal_set.size(),
                        'interval': credal_set.interval_probability(1),
                        'entropy': credal_set.entropy()
                    })
            
            if uncertain_concepts:
                # Sort by uncertainty
                uncertain_concepts.sort(key=lambda x: x['uncertainty'], reverse=True)
                
                sample_info = {
                    'animal': dataset.get_class_names()[labels[0]],
                    'prediction_confidence': F.softmax(predictions[0], dim=0).max().item(),
                    'uncertain_concepts': uncertain_concepts[:5]  # Top 5 most uncertain
                }
                
                high_uncertainty_samples.append(sample_info)
                
                print(f"\n🦁 Animal: {sample_info['animal']}")
                print(f"   Model confidence: {sample_info['prediction_confidence']:.3f}")
                print(f"   Uncertain concepts:")
                
                for concept_info in uncertain_concepts[:3]:  # Show top 3
                    interval = concept_info['interval']
                    print(f"     • {concept_info['concept']}: "
                          f"P ∈ [{interval[0]:.3f}, {interval[1]:.3f}] "
                          f"(uncertainty: {concept_info['uncertainty']:.3f})")
            
            samples_analyzed += 1
    
    # Summarize annotation priorities
    print(f"\n📋 Annotation Priority Summary")
    print("-" * 40)
    
    if high_uncertainty_samples:
        # Sort samples by average uncertainty
        for sample in high_uncertainty_samples:
            avg_uncertainty = np.mean([c['uncertainty'] for c in sample['uncertain_concepts']])
            sample['avg_uncertainty'] = avg_uncertainty
        
        high_uncertainty_samples.sort(key=lambda x: x['avg_uncertainty'], reverse=True)
        
        print(f"🔴 High priority samples for annotation:")
        for i, sample in enumerate(high_uncertainty_samples[:5]):
            print(f"   {i+1}. {sample['animal']} (avg uncertainty: {sample['avg_uncertainty']:.3f})")
        
        # Most commonly uncertain concepts
        all_uncertain_concepts = []
        for sample in high_uncertainty_samples:
            all_uncertain_concepts.extend([c['concept'] for c in sample['uncertain_concepts']])
        
        from collections import Counter
        concept_counts = Counter(all_uncertain_concepts)
        
        print(f"\n🎯 Most commonly uncertain concepts:")
        for concept, count in concept_counts.most_common(10):
            print(f"   • {concept}: appears in {count} uncertain samples")
    
    return high_uncertainty_samples

def main():
    """Main demonstration of credal CBMs with real datasets"""
    
    print("🚀 Credal CBMs with Real-World Datasets")
    print("=" * 50)
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Animals with Attributes dataset
    print("\n📁 Loading Animals with Attributes dataset...")
    from Animal import AnimalDataset
    
    train_dataset = AnimalDataset('trainclasses.txt', root_dir='/Users/tanmoy/research/data/Animals_with_Attributes2')
    val_dataset = AnimalDataset('testclasses.txt', root_dir='/Users/tanmoy/research/data/Animals_with_Attributes2')
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Concepts: {len(train_dataset.get_concept_names())}")
    print(f"   Animal classes: {len(train_dataset.get_class_names())}")
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize feature extractor
    print("\n🧠 Initializing feature extractor...")
    feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor.fc = nn.Identity()  # Remove classification head
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Cache features to disk if not already cached
    train_cache = 'train_features.npz'
    val_cache = 'val_features.npz'
    
    def extract_and_cache_features(loader, cache_file):
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            data = np.load(cache_file)
            return (torch.tensor(data['features']), 
                   torch.tensor(data['concepts']), 
                   torch.tensor(data['labels']))
        
        print(f"Extracting and caching features to {cache_file}")
        all_features = []
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                images = batch['features'].to(device)
                concepts = batch['concepts']
                labels = batch['label']
                
                # Extract features in smaller sub-batches if needed
                features = feature_extractor(images).cpu()
                
                all_features.append(features.numpy())
                all_concepts.append(concepts.numpy())
                all_labels.append(labels.numpy())
        
        # Concatenate and save
        features = np.concatenate(all_features)
        concepts = np.concatenate(all_concepts)
        labels = np.concatenate(all_labels)
        
        np.savez(cache_file, 
                 features=features,
                 concepts=concepts,
                 labels=labels)
        
        return (torch.tensor(features), 
                torch.tensor(concepts), 
                torch.tensor(labels))
    
    print("\n📦 Processing dataset features...")
    train_features, train_concepts, train_labels = extract_and_cache_features(
        train_loader, train_cache)
    val_features, val_concepts, val_labels = extract_and_cache_features(
        val_loader, val_cache)
    
    # Create memory-efficient feature datasets
    from torch.utils.data import TensorDataset
    train_feature_dataset = TensorDataset(train_features, train_concepts, train_labels)
    val_feature_dataset = TensorDataset(val_features, val_concepts, val_labels)
    
    # Create feature loaders with smaller batch size
    train_feature_loader = DataLoader(train_feature_dataset, batch_size=32, 
                                    shuffle=True, num_workers=2)
    val_feature_loader = DataLoader(val_feature_dataset, batch_size=32, 
                                   shuffle=False, num_workers=2)
    
    # Initialize MMD-Enhanced Credal CBM
    print("\n🧠 Initializing MMD-Enhanced Credal CBM...")
    model = MMDEnhancedCredalCBM(
        input_dim=2048,  # ResNet50 feature dimension
        concept_names=train_dataset.get_concept_names(),
        class_names=train_dataset.get_class_names(),
        n_credal_points=5,
        uncertainty_method='ensemble',
        mmd_weight=0.01,
        enable_drift_detection=True
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   MMD weight: {model.mmd_weight}")
    print(f"   Drift detection: {'enabled' if model.enable_drift_detection else 'disabled'}")
    
    # Train model using pre-extracted features with MMD regularization
    print("\n🎯 Training MMD-Enhanced Credal CBM...")
    train_losses, val_accuracies, mmd_losses = train_mmd_enhanced_credal_cbm(
        model, train_feature_loader, val_feature_loader, 
        epochs=30, device=device, feature_extractor=None
    )
    
    print(f"   Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"   Final MMD loss: {mmd_losses[-1]:.6f}")
    
    # Analyze concept stability and uncertainty patterns
    print("\n🔍 Analyzing Concept Stability and Uncertainty Patterns...")
    stability_analysis = analyze_concept_stability(
        model, val_feature_dataset, device=device, n_samples=200
    )
    
    print(f"\n📊 Stability Analysis Results:")
    print(f"   Mean stability score: {stability_analysis['mean_stability']:.6f}")
    print(f"   Drift rate: {stability_analysis['drift_rate']:.2%}")
    print(f"   Number of drift events: {len(stability_analysis['drift_events'])}")
    
    # Demonstrate uncertainty-guided annotation
    print("\n🎯 Demonstrating Uncertainty-Guided Annotation...")
    annotation_priorities = demonstrate_uncertainty_guided_annotation(
        model, val_feature_dataset, uncertainty_threshold=0.2,
        device=device
    )
    
    return model, stability_analysis, annotation_priorities

if __name__ == "__main__":
    # Run the complete demonstration
    model, analysis, priorities = main()