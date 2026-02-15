"""
Configuration for CREDENCE with Distributionally Robust Optimization (DRO)

This module provides comprehensive configuration for training CREDENCE models
with uncertainty decomposition and distributional robustness.

Author: Tanmoy
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json


@dataclass
class ArchitectureConfig:
    """Model architecture parameters."""

    # Core dimensions
    num_concepts: int = 4          # C: binary concepts (same as CREDENCE)
    num_classes: int = 3           # J: output label classes
    hidden_dim: int = 768          # d: encoder hidden size (set from model)
    head_hidden_dim: int = 256     # internal MLP width (matches ConceptHead)
    pooling: str = "cls"           # "cls" | "mean" | "last"

    # Ensemble settings
    n_heads: int = 5               # Number of ensemble heads
    dropout_min: float = 0.05      # Minimum dropout rate
    dropout_max: float = 0.30      # Maximum dropout rate
    use_pooling_diversity: bool = True  # Use diverse pooling strategies


@dataclass
class PGDConfig:
    """Projected Gradient Descent (inner loop) parameters."""

    pgd_steps: int = 10            # T: inner-loop iterations
    pgd_lr: float = 0.01           # α: step size for gradient ascent


@dataclass
class LossWeightConfig:
    """Loss component weights."""

    # L_total = L_nom + λ_rob * L_rob + β_width * L_width + λ_concept * L_concept
    lambda_rob: float = 0.1        # weight on worst-case loss
    beta_width: float = 0.01       # weight on width penalty
    concept_weight: float = 1.0    # weight on concept supervision (if available)
    aleatoric_weight: float = 0.5  # weight on aleatoric uncertainty loss


@dataclass
class WidthPenaltyConfig:
    """Credal set width penalty parameters."""

    width_penalty: str = "log_det"  # "log_det" | "trace" | "none"
    # log_det: log determinant of covariance matrix (encourages diversity)
    # trace: sum of variances (simpler)
    # none: no width penalty


@dataclass
class SigmaConfig:
    """Sigma (uncertainty) parameters."""

    # Bounds
    sigma_min: float = 1e-4        # floor for softplus output
    sigma_max: float = 2.0         # ceiling (logit space is wider than [0,1])

    # Gradient settings
    stop_grad_sigma: bool = True   # True = safe (default), False = joint training


@dataclass
class RobustLossConfig:
    """Robust loss function parameters (for inner-loop only)."""

    robust_loss: str = "none"      # "none" | "clip" | "huber"
    tau: float = 5.0               # clipping threshold / Huber centre
    kappa: float = 1.0             # Huber transition width


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 16
    seed: int = 42

    # Optimizer settings
    optimizer: str = "adamw"       # "adamw" | "adam" | "sgd"
    betas: tuple = (0.9, 0.999)     # Adam betas
    eps: float = 1e-8               # Adam epsilon

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_ratio: float = 0.1       # Warmup for 10% of training

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 5               # Stop if val doesn't improve for N epochs
    min_delta: float = 1e-4         # Minimum improvement to reset patience


@dataclass
class EncoderConfig:
    """Encoder model parameters."""

    encoder_name: str = "distilbert-base-uncased"
    freeze_encoder: bool = True
    max_length: int = 128

    # Feature extraction
    latent_method: str = "cls"      # "cls" | "low_rank" | "multi_layer"
    latent_rank: Optional[int] = None  # For low-rank extraction
    latent_layer: str = "last"      # "first" | "last" | "mean"


@dataclass
class DatasetConfig:
    """Dataset parameters."""

    dataset: str = "cebab"
    label_type: str = "ternary"     # For CEBaB: "binary" | "ternary" | "5way"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configurations
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    pgd: PGDConfig = field(default_factory=PGDConfig)
    loss_weights: LossWeightConfig = field(default_factory=LossWeightConfig)
    width_penalty: WidthPenaltyConfig = field(default_factory=WidthPenaltyConfig)
    sigma: SigmaConfig = field(default_factory=SigmaConfig)
    robust_loss: RobustLossConfig = field(default_factory=RobustLossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Experiment metadata
    experiment_name: str = "credence_dro"
    output_dir: str = "./results"
    log_interval: int = 10          # Log every N batches
    eval_interval: int = 1          # Evaluate every N epochs

    # Reproducibility
    seed: int = 42
    deterministic: bool = False     # Use deterministic algorithms (slower)

    # Device
    device: str = "cuda"            # "cuda" | "cpu" | "mps"

    # Mixed precision
    use_amp: bool = False           # Automatic mixed precision
    amp_dtype: str = "float16"      # "float16" | "bfloat16"

    # Profiling
    enable_profiler: bool = False
    profiler_warmup: int = 1
    profiler_active: int = 3
    profiler_repeat: int = 1

    def __post_init__(self):
        """Validate and sync configuration."""
        # Sync seed
        self.training.seed = self.seed
        self.dataset.seed = self.seed

        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

    def save(self, path: str):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_cebab_config() -> ExperimentConfig:
    """Default configuration for CEBaB dataset."""
    config = ExperimentConfig(
        experiment_name="credence_dro_cebab",
        architecture=ArchitectureConfig(
            num_concepts=4,
            num_classes=3,
            hidden_dim=768,
            head_hidden_dim=256,
            pooling="cls",
            n_heads=5,
        ),
        dataset=DatasetConfig(
            dataset="cebab",
            label_type="ternary",
        ),
        encoder=EncoderConfig(
            encoder_name="distilbert-base-uncased",
            freeze_encoder=True,
            max_length=128,
        ),
        training=TrainingConfig(
            epochs=40,
            lr=1e-4,
            batch_size=16,
        ),
    )
    return config


def get_hatexplain_config() -> ExperimentConfig:
    """Default configuration for HateXplain dataset."""
    config = ExperimentConfig(
        experiment_name="credence_dro_hatexplain",
        architecture=ArchitectureConfig(
            num_concepts=2,
            num_classes=3,
            hidden_dim=768,
            head_hidden_dim=256,
            pooling="cls",
            n_heads=5,
        ),
        dataset=DatasetConfig(
            dataset="hatexplain",
        ),
        encoder=EncoderConfig(
            encoder_name="distilbert-base-uncased",
            freeze_encoder=True,
            max_length=128,
        ),
        training=TrainingConfig(
            epochs=30,
            lr=1e-4,
            batch_size=16,
        ),
    )
    return config


def get_sst2_config() -> ExperimentConfig:
    """Default configuration for SST-2 dataset."""
    config = ExperimentConfig(
        experiment_name="credence_dro_sst2",
        architecture=ArchitectureConfig(
            num_concepts=0,  # No concepts
            num_classes=2,
            hidden_dim=768,
            head_hidden_dim=256,
            pooling="cls",
            n_heads=5,
        ),
        dataset=DatasetConfig(
            dataset="sst2",
        ),
        encoder=EncoderConfig(
            encoder_name="distilbert-base-uncased",
            freeze_encoder=True,
            max_length=128,
        ),
        training=TrainingConfig(
            epochs=20,
            lr=1e-4,
            batch_size=32,
        ),
    )
    return config


def get_low_rank_config(dataset: str = "cebab", rank: int = 128) -> ExperimentConfig:
    """Configuration with low-rank latent extraction."""
    if dataset == "cebab":
        config = get_cebab_config()
    elif dataset == "hatexplain":
        config = get_hatexplain_config()
    elif dataset == "sst2":
        config = get_sst2_config()
    else:
        config = get_cebab_config()

    # Override encoder settings
    config.encoder.latent_method = "low_rank"
    config.encoder.latent_rank = rank
    config.architecture.hidden_dim = rank  # Update to match latent dim

    config.experiment_name = f"credence_dro_{dataset}_lowrank{rank}"

    return config


def get_multilayer_config(dataset: str = "cebab") -> ExperimentConfig:
    """Configuration with multi-layer latent extraction."""
    if dataset == "cebab":
        config = get_cebab_config()
    elif dataset == "hatexplain":
        config = get_hatexplain_config()
    else:
        config = get_cebab_config()

    # Override encoder settings
    config.encoder.latent_method = "multi_layer"
    config.architecture.hidden_dim = 3072  # 4 layers × 768

    config.experiment_name = f"credence_dro_{dataset}_multilayer"

    return config


# =============================================================================
# CONFIG PRESETS FOR DRO VARIANTS
# =============================================================================

def get_dro_strong_config() -> ExperimentConfig:
    """Strong DRO (high robustness weight)."""
    config = get_cebab_config()
    config.loss_weights.lambda_rob = 0.5  # Stronger robustness
    config.pgd.pgd_steps = 20  # More inner loop iterations
    config.experiment_name = "credence_dro_strong"
    return config


def get_dro_weak_config() -> ExperimentConfig:
    """Weak DRO (low robustness weight)."""
    config = get_cebab_config()
    config.loss_weights.lambda_rob = 0.01  # Weaker robustness
    config.pgd.pgd_steps = 5  # Fewer iterations
    config.experiment_name = "credence_dro_weak"
    return config


def get_width_penalty_config(penalty_type: str = "log_det") -> ExperimentConfig:
    """Configuration with width penalty."""
    config = get_cebab_config()
    config.width_penalty.width_penalty = penalty_type
    config.loss_weights.beta_width = 0.1  # Enable width penalty
    config.experiment_name = f"credence_dro_width_{penalty_type}"
    return config


def get_joint_sigma_config() -> ExperimentConfig:
    """Configuration with joint sigma training (not stop-gradient)."""
    config = get_cebab_config()
    config.sigma.stop_grad_sigma = False  # Joint training
    config.experiment_name = "credence_dro_joint_sigma"
    return config


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # Check architecture
    if config.architecture.num_concepts < 0:
        warnings.append(f"Invalid num_concepts: {config.architecture.num_concepts}")

    if config.architecture.num_classes < 2:
        warnings.append(f"Invalid num_classes: {config.architecture.num_classes}")

    if config.architecture.hidden_dim <= 0:
        warnings.append(f"Invalid hidden_dim: {config.architecture.hidden_dim}")

    # Check PGD
    if config.pgd.pgd_steps < 1:
        warnings.append(f"Invalid pgd_steps: {config.pgd.pgd_steps}")

    if config.pgd.pgd_lr <= 0:
        warnings.append(f"Invalid pgd_lr: {config.pgd.pgd_lr}")

    # Check loss weights
    if config.loss_weights.lambda_rob < 0:
        warnings.append(f"Invalid lambda_rob: {config.loss_weights.lambda_rob}")

    if config.loss_weights.beta_width < 0:
        warnings.append(f"Invalid beta_width: {config.loss_weights.beta_width}")

    # Check sigma bounds
    if config.sigma.sigma_min >= config.sigma.sigma_max:
        warnings.append(f"sigma_min ({config.sigma.sigma_min}) >= sigma_max ({config.sigma.sigma_max})")

    # Check training
    if config.training.lr <= 0:
        warnings.append(f"Invalid learning rate: {config.training.lr}")

    if config.training.batch_size < 1:
        warnings.append(f"Invalid batch_size: {config.training.batch_size}")

    if config.training.epochs < 1:
        warnings.append(f"Invalid epochs: {config.training.epochs}")

    # Check encoder
    valid_encoders = [
        "distilbert-base-uncased",
        "roberta-base",
        "microsoft/deberta-v3-base",
    ]
    if config.encoder.encoder_name not in valid_encoders:
        warnings.append(f"Unknown encoder: {config.encoder.encoder_name}")

    return warnings


def print_config(config: ExperimentConfig):
    """Pretty print configuration."""
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)

    print(f"\nExperiment: {config.experiment_name}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")

    print("\n--- Architecture ---")
    print(f"  Concepts: {config.architecture.num_concepts}")
    print(f"  Classes: {config.architecture.num_classes}")
    print(f"  Hidden dim: {config.architecture.hidden_dim}")
    print(f"  Heads: {config.architecture.n_heads}")
    print(f"  Pooling: {config.architecture.pooling}")

    print("\n--- PGD (Inner Loop) ---")
    print(f"  Steps: {config.pgd.pgd_steps}")
    print(f"  LR: {config.pgd.pgd_lr}")

    print("\n--- Loss Weights ---")
    print(f"  λ_rob: {config.loss_weights.lambda_rob}")
    print(f"  β_width: {config.loss_weights.beta_width}")
    print(f"  Concept: {config.loss_weights.concept_weight}")
    print(f"  Aleatoric: {config.loss_weights.aleatoric_weight}")

    print("\n--- Width Penalty ---")
    print(f"  Type: {config.width_penalty.width_penalty}")

    print("\n--- Sigma ---")
    print(f"  Min: {config.sigma.sigma_min}")
    print(f"  Max: {config.sigma.sigma_max}")
    print(f"  Stop grad: {config.sigma.stop_grad_sigma}")

    print("\n--- Robust Loss ---")
    print(f"  Type: {config.robust_loss.robust_loss}")
    if config.robust_loss.robust_loss != "none":
        print(f"  Tau: {config.robust_loss.tau}")
        print(f"  Kappa: {config.robust_loss.kappa}")

    print("\n--- Training ---")
    print(f"  LR: {config.training.lr}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")

    print("\n--- Encoder ---")
    print(f"  Model: {config.encoder.encoder_name}")
    print(f"  Freeze: {config.encoder.freeze_encoder}")
    print(f"  Max length: {config.encoder.max_length}")
    if config.encoder.latent_method != "cls":
        print(f"  Latent method: {config.encoder.latent_method}")
        if config.encoder.latent_method == "low_rank":
            print(f"  Rank: {config.encoder.latent_rank}")

    print("\n--- Dataset ---")
    print(f"  Name: {config.dataset.dataset}")
    if config.dataset.dataset == "cebab":
        print(f"  Label type: {config.dataset.label_type}")

    print("\n" + "="*80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import torch

    # Test configuration
    config = get_cebab_config()

    # Print configuration
    print_config(config)

    # Validate
    warnings = validate_config(config)
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ Configuration is valid!")

    # Save example
    config.save("example_config.json")
    print("\nExample configuration saved to: example_config.json")
