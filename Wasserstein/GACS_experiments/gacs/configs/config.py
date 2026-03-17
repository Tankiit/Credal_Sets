"""
GACS Configuration
Geometry-Aware Credal Predictive Sets for Latent Variable Models
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """VAE with interpretable bottleneck architecture config."""
    # Encoder
    encoder_name: str = "bert-base-uncased"
    freeze_encoder_layers: int = 8  # freeze first N layers for speed
    
    # Latent space
    z_dim: int = 64  # latent dimension
    
    # Concept bottleneck
    num_concepts: int = 4  # CEBaB: food, service, ambiance, noise
    concept_names: List[str] = field(default_factory=lambda: [
        "food", "service", "ambiance", "noise"
    ])
    
    # Classifier
    num_classes: int = 3  # CEBaB: Negative/unknown/Positive (3-class sentiment)
    
    # Architecture details
    hidden_dim: int = 256  # intermediate MLP dimension
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basics
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42
    
    # Scheduler
    warmup_ratio: float = 0.1
    
    # Early stopping
    patience: int = 5
    
    # Logging
    log_every: int = 50
    eval_every_epoch: int = 1


@dataclass
class LossConfig:
    """Loss weights and scheduling."""
    # Reconstruction (contrastive on CLS embeddings)
    recon_weight: float = 1.0
    
    # KL divergence with sigmoid annealing
    kl_weight_max: float = 0.1
    kl_warmup_epochs: int = 5
    
    # Concept supervision (cross-entropy with ground-truth labels)
    concept_weight: float = 1.0
    
    # Classification
    classify_weight: float = 1.0
    
    # Concept regularization
    concept_sparsity_weight: float = 0.05
    concept_diversity_weight: float = 0.05


@dataclass
class ProbeConfig:
    """Geometric probe configuration."""
    # Perturbation probe
    num_directions: int = 100  # m random directions
    perturbation_scale: float = 0.01  # ε
    sensitivity_threshold: float = 0.01  # τ
    
    # Which parameters to probe
    probe_scope: str = "concept"  # "all", "concept", "classifier"
    
    # Hessian probe (optional, more expensive)
    use_hessian: bool = False
    lanczos_steps: int = 50  # k top eigenvalues


@dataclass
class CredalConfig:
    """Credal set calibration configuration."""
    # Imprecision bounds
    epsilon_min: float = 0.01
    epsilon_max: float = 0.3
    
    # Calibration function g(ρ)
    calibration_type: str = "sigmoid"  # "linear", "sigmoid"
    sigmoid_steepness: float = 10.0
    sigmoid_midpoint: float = 0.5
    
    # Coverage target for calibration
    target_coverage: float = 0.9


@dataclass
class GACSConfig:
    """Master config combining all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    credal: CredalConfig = field(default_factory=CredalConfig)
    
    # Experiment
    experiment_name: str = "gacs_cebab"
    output_dir: str = "outputs"
    
    @classmethod
    def for_cebab(cls):
        """CEBaB-specific defaults."""
        return cls(
            model=ModelConfig(num_concepts=4, num_classes=3),  # 3-class sentiment
            experiment_name="gacs_cebab",
        )
    
    @classmethod
    def for_amazon(cls):
        """Amazon cross-domain defaults."""
        return cls(
            model=ModelConfig(
                num_concepts=8,  # learned, not supervised
                num_classes=2,   # binary sentiment
                concept_names=[f"concept_{i}" for i in range(8)],
            ),
            loss=LossConfig(concept_weight=0.0),  # no concept supervision
            experiment_name="gacs_amazon",
        )
    
    @classmethod
    def for_debug(cls):
        """Fast debug config."""
        config = cls.for_cebab()
        config.training.batch_size = 8
        config.training.epochs = 3
        config.training.max_seq_length = 64
        config.model.z_dim = 16
        config.model.hidden_dim = 64
        config.probe.num_directions = 10
        return config
