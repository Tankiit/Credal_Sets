"""
MMD-Enhanced Credal Concept Bottleneck Models
Combines Maximum Mean Discrepancy with credal sets for improved concept learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import base components
from credal_base import CredalSet, ConceptAnnotation, RealWorldCredalCBM

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss for distributional regularization
    """
    def __init__(self, kernel_type='gaussian', bandwidths=None, fix_sigma=None):
        super().__init__()
        self.kernel_type = kernel_type
        self.bandwidths = bandwidths or [0.1, 0.5, 1.0, 2.0, 5.0]  # Multi-scale approach
        self.fix_sigma = fix_sigma
        
    def gaussian_kernel(self, x, y, sigma):
        """Gaussian RBF kernel"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input / (2 * sigma**2))
    
    def compute_mmd(self, x, y, sigma):
        """Compute MMD between two sample sets"""
        x_kernel = self.gaussian_kernel(x, x, sigma)
        y_kernel = self.gaussian_kernel(y, y, sigma)
        xy_kernel = self.gaussian_kernel(x, y, sigma)
        
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd
    
    def forward(self, source_features, target_features):
        """
        Compute multi-scale MMD loss
        Args:
            source_features: tensor of shape (batch_size, feature_dim)
            target_features: tensor of shape (batch_size, feature_dim)
        """
        if self.fix_sigma:
            return self.compute_mmd(source_features, target_features, self.fix_sigma)
        
        # Multi-scale MMD
        total_loss = 0
        for sigma in self.bandwidths:
            total_loss += self.compute_mmd(source_features, target_features, sigma)
        
        return total_loss / len(self.bandwidths)

class ConceptDriftDetector:
    """
    MMD-based concept drift detection for monitoring concept stability
    """
    def __init__(self, reference_size=1000, threshold_percentile=95):
        self.reference_concepts = None
        self.reference_size = reference_size
        self.threshold_percentile = threshold_percentile
        self.mmd_history = []
        self.drift_threshold = None
        
    def set_reference(self, concept_activations):
        """Set reference concept distribution"""
        if len(concept_activations) > self.reference_size:
            indices = torch.randperm(len(concept_activations))[:self.reference_size]
            self.reference_concepts = concept_activations[indices]
        else:
            self.reference_concepts = concept_activations.clone()
            
        # Compute initial threshold from reference distribution
        self._compute_threshold()
    
    def _compute_threshold(self):
        """Compute drift detection threshold from reference distribution"""
        if self.reference_concepts is None:
            return
            
        mmd_loss = MMDLoss()
        bootstrap_mmds = []
        
        # Bootstrap sampling to estimate MMD distribution
        for _ in range(100):
            n = len(self.reference_concepts)
            indices1 = torch.randint(0, n, (n//2,))
            indices2 = torch.randint(0, n, (n//2,))
            
            sample1 = self.reference_concepts[indices1]
            sample2 = self.reference_concepts[indices2]
            
            mmd_val = mmd_loss(sample1, sample2).item()
            bootstrap_mmds.append(mmd_val)
        
        self.drift_threshold = np.percentile(bootstrap_mmds, self.threshold_percentile)
    
    def detect_drift(self, current_concepts):
        """
        Detect if current concepts have drifted from reference
        Returns: (is_drift, mmd_score, confidence)
        """
        if self.reference_concepts is None:
            raise ValueError("Reference concepts not set. Call set_reference() first.")
        
        mmd_loss = MMDLoss()
        mmd_score = mmd_loss(current_concepts, self.reference_concepts).item()
        self.mmd_history.append(mmd_score)
        
        is_drift = mmd_score > self.drift_threshold
        confidence = min(mmd_score / self.drift_threshold, 2.0)  # Cap at 2.0
        
        return is_drift, mmd_score, confidence

class MMDEnhancedCredalCBM(RealWorldCredalCBM):
    """
    Enhanced Credal CBM with MMD-based regularization and drift detection
    """
    def __init__(self, input_dim: int, concept_names: List[str], class_names: List[str],
                 n_credal_points: int = 5, hidden_dim: int = 256, 
                 uncertainty_method: str = 'ensemble',
                 mmd_weight: float = 0.01, enable_drift_detection: bool = True):
        
        super().__init__(input_dim, concept_names, class_names, n_credal_points, 
                        hidden_dim, uncertainty_method)
        
        self.mmd_weight = mmd_weight
        self.enable_drift_detection = enable_drift_detection
        
        # MMD components
        self.mmd_loss = MMDLoss()
        self.concept_drift_detector = ConceptDriftDetector() if enable_drift_detection else None
        
        # Reference concepts for regularization (learnable)
        self.register_buffer('reference_concepts', torch.zeros(len(concept_names), hidden_dim // 2))
        self.reference_initialized = False
        
        # Concept quality metrics
        self.concept_quality_history = []
        
    def _initialize_reference_concepts(self, concept_features):
        """Initialize reference concept distribution"""
        if not self.reference_initialized:
            # Use mean of initial batch as reference
            self.reference_concepts = concept_features.mean(dim=0, keepdim=True).repeat(len(self.concept_names), 1)
            self.reference_initialized = True
    
    def compute_concept_quality_loss(self, concept_features, target_concepts):
        """
        Compute MMD-based concept quality regularization
        Encourages concept features to match target distributional properties
        """
        batch_size = concept_features.size(0)
        
        # Initialize reference if needed
        self._initialize_reference_concepts(concept_features)
        
        # Create target concept distribution based on annotations
        target_features = []
        for i, concept_name in enumerate(self.concept_names):
            # Get features for samples where this concept is highly activated
            concept_mask = target_concepts[:, i] > 0.7
            if concept_mask.sum() > 0:
                concept_specific_features = concept_features[concept_mask]
                target_features.append(concept_specific_features.mean(dim=0))
            else:
                # Use reference if no positive examples
                target_features.append(self.reference_concepts[i])
        
        if len(target_features) == 0:
            return torch.tensor(0.0, device=concept_features.device)
        
        target_features = torch.stack(target_features)
        
        # Compute MMD between concept features and target distribution
        mmd_loss = self.mmd_loss(concept_features, target_features)
        
        return mmd_loss
    
    def forward(self, x: torch.Tensor, target_concepts: torch.Tensor = None) -> Tuple[List[List[CredalSet]], torch.Tensor, Dict]:
        """Enhanced forward pass with MMD regularization"""
        batch_size = x.shape[0]
        
        # Get shared features (we'll use these for MMD)
        shared_features = self.concept_encoder_shared(x)
        
        # Continue with original credal set generation
        if self.uncertainty_method == 'ensemble':
            concept_logits = []
            for head in self.concept_heads:
                logits = torch.sigmoid(head(shared_features))
                concept_logits.append(logits)
            concept_logits = torch.stack(concept_logits, dim=1)
        else:
            concept_mean = torch.sigmoid(self.concept_head(shared_features))
            concept_std = torch.softplus(self.uncertainty_head(shared_features)) + 1e-6
            
            concept_logits = []
            for _ in range(self.n_credal_points):
                noise = torch.randn_like(concept_mean)
                sample = torch.sigmoid(concept_mean + concept_std * noise)
                concept_logits.append(sample)
            concept_logits = torch.stack(concept_logits, dim=1)
        
        # Convert to credal sets (same as original)
        batch_credal_sets = []
        concept_expectations = []
        
        for b in range(batch_size):
            sample_credal_sets = []
            sample_expectations = []
            
            for c in range(self.n_concepts):
                concept_predictions = concept_logits[b, :, c].detach().cpu().numpy()
                
                extreme_points = np.column_stack([
                    1 - concept_predictions,
                    concept_predictions
                ])
                
                credal_set = CredalSet(extreme_points, [f"¬{self.concept_names[c]}", self.concept_names[c]])
                sample_credal_sets.append(credal_set)
                
                expectation = np.mean(concept_predictions)
                sample_expectations.append(expectation)
            
            batch_credal_sets.append(sample_credal_sets)
            concept_expectations.append(sample_expectations)
        
        # Final prediction
        concept_tensor = torch.tensor(concept_expectations, device=x.device)
        weighted_concepts = concept_tensor * torch.sigmoid(self.concept_importance)
        final_predictions = self.label_predictor(weighted_concepts)
        
        # Enhanced uncertainty metrics with MMD
        uncertainty_metrics = {
            'concept_importance': torch.sigmoid(self.concept_importance).detach().cpu().numpy(),
            'prediction_entropy': self._compute_prediction_entropy(final_predictions),
            'concept_uncertainties': self._compute_concept_uncertainties(batch_credal_sets),
            'shared_features': shared_features  # For MMD computation
        }
        
        # Compute concept quality if target concepts provided
        if target_concepts is not None:
            concept_quality_loss = self.compute_concept_quality_loss(shared_features, target_concepts)
            uncertainty_metrics['concept_quality_loss'] = concept_quality_loss.item()
            
            # Update concept quality history
            self.concept_quality_history.append(concept_quality_loss.item())
        
        return batch_credal_sets, final_predictions, uncertainty_metrics

def train_mmd_enhanced_credal_cbm(model, train_loader, val_loader, epochs=50,
                                device='cpu', feature_extractor=None):
    """Enhanced training with MMD regularization"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    concept_loss_fn = nn.BCELoss()
    prediction_loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    mmd_losses = []
    
    # Initialize drift detector with first batch
    if model.enable_drift_detection:
        first_batch = next(iter(train_loader))
        with torch.no_grad():
            # Handle both dictionary and tuple batch formats
            if isinstance(first_batch, dict):
                features = first_batch['features'].to(device)
            else:
                features = first_batch[0].to(device)  # First element is features
            
            if feature_extractor is not None:
                features = feature_extractor(features)
            
            _, _, metrics = model(features)
            model.concept_drift_detector.set_reference(metrics['shared_features'])
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_mmd_loss = 0
        
        for batch in train_loader:
            # Handle both dictionary and tuple batch formats
            if isinstance(batch, dict):
                features = batch['features'].to(device)
                concepts = batch['concepts'].to(device)
                labels = batch['label'].to(device)
            else:
                features, concepts, labels = [b.to(device) for b in batch]
            
            # Extract features if needed
            if feature_extractor is not None:
                with torch.no_grad():
                    features = feature_extractor(features)
            
            optimizer.zero_grad()
            credal_sets, predictions, metrics = model(features, concepts)
            
            # Compute concept loss
            concept_expectations = []
            for sample_credal_sets in credal_sets:
                sample_exp = [cs.credal_dominance_score(1) for cs in sample_credal_sets]
                concept_expectations.append(sample_exp)
            
            concept_exp_tensor = torch.tensor(concept_expectations, device=device)
            concept_loss = concept_loss_fn(concept_exp_tensor, concepts)
            
            # Prediction loss
            prediction_loss = prediction_loss_fn(predictions, labels)
            
            # MMD regularization loss
            mmd_reg_loss = 0
            if 'concept_quality_loss' in metrics:
                mmd_reg_loss = metrics['concept_quality_loss']
            
            # Uncertainty regularization
            uncertainty_reg = 0
            for sample_credal_sets in credal_sets:
                for credal_set in sample_credal_sets:
                    uncertainty_reg += max(0, 0.1 - credal_set.size())
            uncertainty_reg = uncertainty_reg / (len(credal_sets) * len(credal_sets[0]))
            
            # Total loss with MMD regularization
            total_loss = (concept_loss + prediction_loss + 
                         model.mmd_weight * mmd_reg_loss + 
                         0.01 * uncertainty_reg)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_mmd_loss += mmd_reg_loss if isinstance(mmd_reg_loss, float) else mmd_reg_loss.item()
        
        # Validation phase with drift detection
        model.eval()
        val_correct = 0
        val_total = 0
        drift_detected = False
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle both dictionary and tuple batch formats
                if isinstance(batch, dict):
                    features = batch['features'].to(device)
                    labels = batch['label'].to(device)
                else:
                    features, _, labels = [b.to(device) for b in batch]
                
                if feature_extractor is not None:
                    features = feature_extractor(features)
                
                _, predictions, metrics = model(features)
                
                # Drift detection
                if model.enable_drift_detection and epoch > 5:  # Start after 5 epochs
                    is_drift, mmd_score, confidence = model.concept_drift_detector.detect_drift(
                        metrics['shared_features']
                    )
                    if is_drift:
                        drift_detected = True
                
                _, predicted = torch.max(predictions, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        avg_loss = epoch_loss / len(train_loader)
        avg_mmd_loss = epoch_mmd_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        mmd_losses.append(avg_mmd_loss)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            drift_status = " [DRIFT DETECTED]" if drift_detected else ""
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}, "
                  f"MMD Loss = {avg_mmd_loss:.6f}{drift_status}")
    
    return train_losses, val_accuracies, mmd_losses

def analyze_concept_stability(model, dataset, device='cpu', n_samples=500, feature_extractor=None):
    """Analyze concept stability using MMD-based drift detection"""
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    concept_stability_scores = []
    drift_events = []
    concept_quality_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * 32 >= n_samples:
                break
                
            images = batch['features'].to(device)
            
            if feature_extractor is not None:
                features = feature_extractor(images)
            else:
                features = images
            
            _, _, metrics = model(features)
            
            # Check for drift
            if model.enable_drift_detection:
                is_drift, mmd_score, confidence = model.concept_drift_detector.detect_drift(
                    metrics['shared_features']
                )
                
                concept_stability_scores.append(mmd_score)
                
                if is_drift:
                    drift_events.append({
                        'batch': i,
                        'mmd_score': mmd_score,
                        'confidence': confidence
                    })
            
            # Track concept quality
            if 'concept_quality_loss' in metrics:
                concept_quality_scores.append(metrics['concept_quality_loss'])
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Concept stability over time
    if concept_stability_scores:
        axes[0, 0].plot(concept_stability_scores, 'b-', alpha=0.7)
        if model.concept_drift_detector.drift_threshold:
            axes[0, 0].axhline(y=model.concept_drift_detector.drift_threshold, 
                              color='r', linestyle='--', label='Drift Threshold')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('MMD Score')
        axes[0, 0].set_title('Concept Stability Over Time')
        axes[0, 0].legend()
    
    # Plot 2: Drift events
    if drift_events:
        drift_batches = [d['batch'] for d in drift_events]
        drift_scores = [d['mmd_score'] for d in drift_events]
        axes[0, 1].scatter(drift_batches, drift_scores, c='red', s=50, alpha=0.7)
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('MMD Score at Drift')
        axes[0, 1].set_title(f'Drift Events (Total: {len(drift_events)})')
    
    # Plot 3: Concept quality evolution
    if concept_quality_scores:
        axes[1, 0].plot(concept_quality_scores, 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Concept Quality Loss')
        axes[1, 0].set_title('Concept Quality Over Time')
    
    # Plot 4: MMD distribution
    if concept_stability_scores:
        axes[1, 1].hist(concept_stability_scores, bins=30, alpha=0.7, color='blue')
        if model.concept_drift_detector.drift_threshold:
            axes[1, 1].axvline(x=model.concept_drift_detector.drift_threshold, 
                              color='r', linestyle='--', label='Drift Threshold')
        axes[1, 1].set_xlabel('MMD Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('MMD Score Distribution')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'stability_scores': concept_stability_scores,
        'drift_events': drift_events,
        'quality_scores': concept_quality_scores,
        'mean_stability': np.mean(concept_stability_scores) if concept_stability_scores else 0,
        'drift_rate': len(drift_events) / len(concept_stability_scores) if concept_stability_scores else 0
    }

def demonstrate_mmd_concept_regularization():
    """Demonstrate MMD-based concept regularization"""
    
    print("🎯 MMD-Enhanced Credal CBM Demonstration")
    print("=" * 50)
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic data for demonstration
    torch.manual_seed(42)
    n_samples = 1000
    n_concepts = 10
    n_classes = 5
    feature_dim = 512
    
    # Generate synthetic features and concepts
    features = torch.randn(n_samples, feature_dim)
    concepts = torch.sigmoid(torch.randn(n_samples, n_concepts))
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Create datasets
    from torch.utils.data import TensorDataset
    train_size = int(0.8 * n_samples)
    train_dataset = TensorDataset(features[:train_size], concepts[:train_size], labels[:train_size])
    val_dataset = TensorDataset(features[train_size:], concepts[train_size:], labels[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize models for comparison
    concept_names = [f"concept_{i}" for i in range(n_concepts)]
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Standard credal CBM
    standard_model = RealWorldCredalCBM(
        input_dim=feature_dim,
        concept_names=concept_names,
        class_names=class_names,
        n_credal_points=5
    ).to(device)
    
    # MMD-enhanced credal CBM
    mmd_model = MMDEnhancedCredalCBM(
        input_dim=feature_dim,
        concept_names=concept_names,
        class_names=class_names,
        n_credal_points=5,
        mmd_weight=0.05,
        enable_drift_detection=True
    ).to(device)
    
    print(f"\n🧠 Training Standard Credal CBM...")
    print(f"   Model parameters: {sum(p.numel() for p in standard_model.parameters()):,}")
    
    print(f"\n🚀 Training MMD-Enhanced Credal CBM...")
    print(f"   Model parameters: {sum(p.numel() for p in mmd_model.parameters()):,}")
    print(f"   MMD weight: {mmd_model.mmd_weight}")
    
    # Train MMD-enhanced model
    class SyntheticBatch:
        def __init__(self, features, concepts, labels):
            self.data = {'features': features, 'concepts': concepts, 'label': labels}
        
        def __getitem__(self, key):
            return self.data[key]
    
    # Modify train function for synthetic data
    def train_synthetic_model(model, train_loader, val_loader, epochs=30):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        concept_loss_fn = nn.BCELoss()
        prediction_loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for features_batch, concepts_batch, labels_batch in train_loader:
                features_batch = features_batch.to(device)
                concepts_batch = concepts_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                optimizer.zero_grad()
                
                if hasattr(model, 'mmd_weight'):
                    # MMD-enhanced model
                    credal_sets, predictions, metrics = model(features_batch, concepts_batch)
                else:
                    # Standard model
                    credal_sets, predictions, metrics = model(features_batch)
                
                # Compute losses
                concept_expectations = []
                for sample_credal_sets in credal_sets:
                    sample_exp = [cs.credal_dominance_score(1) for cs in sample_credal_sets]
                    concept_expectations.append(sample_exp)
                
                concept_exp_tensor = torch.tensor(concept_expectations, device=device)
                concept_loss = concept_loss_fn(concept_exp_tensor, concepts_batch)
                prediction_loss = prediction_loss_fn(predictions, labels_batch)
                
                total_loss = concept_loss + prediction_loss
                
                # Add MMD regularization if available
                if hasattr(model, 'mmd_weight') and 'concept_quality_loss' in metrics:
                    total_loss += model.mmd_weight * metrics['concept_quality_loss']
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")
    
    # Train both models
    train_synthetic_model(mmd_model, train_loader, val_loader)
    
    # Analyze concept stability
    print(f"\n🔍 Analyzing Concept Stability...")
    
    # Create a simple dataset class for analysis
    class SimpleDataset:
        def __init__(self, features, concepts, labels):
            self.features = features
            self.concepts = concepts
            self.labels = labels
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return {
                'features': self.features[idx],
                'concepts': self.concepts[idx],
                'label': self.labels[idx]
            }
    
    analysis_dataset = SimpleDataset(features[train_size:], concepts[train_size:], labels[train_size:])
    stability_analysis = analyze_concept_stability(mmd_model, analysis_dataset, device=device)
    
    print(f"\n📊 Stability Analysis Results:")
    print(f"   Mean stability score: {stability_analysis['mean_stability']:.6f}")
    print(f"   Drift rate: {stability_analysis['drift_rate']:.2%}")
    print(f"   Number of drift events: {len(stability_analysis['drift_events'])}")
    
    # Demonstrate concept quality improvement
    if mmd_model.concept_quality_history:
        print(f"\n📈 Concept Quality Improvement:")
        initial_quality = mmd_model.concept_quality_history[0]
        final_quality = mmd_model.concept_quality_history[-1]
        improvement = (initial_quality - final_quality) / initial_quality * 100
        print(f"   Initial quality loss: {initial_quality:.6f}")
        print(f"   Final quality loss: {final_quality:.6f}")
        print(f"   Improvement: {improvement:.1f}%")
    
    return mmd_model, stability_analysis

if __name__ == "__main__":
    # Run the demonstration
    model, analysis = demonstrate_mmd_concept_regularization()