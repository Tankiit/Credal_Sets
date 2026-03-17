"""
GACS Configuration
==================
Geometry-Aware Credal Predictive Sets for Latent Variable Models
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import torch


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: Literal["cebab", "amazon", "sst2", "imdb", "agnews"] = "cebab"
    max_seq_length: int = 128
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    # CEBaB-specific
    cebab_split: str = "train_exclusive"  # train_exclusive / train_inclusive
    cebab_num_concepts: int = 4  # food, service, ambiance, noise
    cebab_num_classes: int = 5  # 1-5 star sentiment

    # Amazon cross-domain
    amazon_source_domain: str = "books"
    amazon_target_domain: str = "electronics"
    amazon_num_classes: int = 2  # binary sentiment

    # SST-2 / IMDB
    sst2_num_classes: int = 2
    imdb_num_classes: int = 2
    agnews_num_classes: int = 4

    @property
    def num_classes(self) -> int:
        return {
            "cebab": self.cebab_num_classes,
            "amazon": self.amazon_num_classes,
            "sst2": self.sst2_num_classes,
            "imdb": self.imdb_num_classes,
            "agnews": self.agnews_num_classes,
        }[self.name]

    @property
    def num_concepts(self) -> int:
        """Number of supervised concept dimensions. 0 = unsupervised."""
        if self.name == "cebab":
            return self.cebab_num_concepts
        return 0  # other datasets: unsupervised bottleneck


@dataclass
class ModelConfig:
    """Model architecture."""
    encoder_name: str = "bert-base-uncased"
    freeze_encoder: bool = False
    freeze_encoder_layers: int = 8  # freeze first N layers (0=none)

    # Latent space
    z_dim: int = 64  # latent dimension
    concept_dim: int = 4  # interpretable bottleneck (set from data config)

    # Architecture
    hidden_dim: int = 256  # intermediate MLP dimension
    dropout: float = 0.1

    # Reconstruction target
    recon_target: Literal["cls", "mean_pool", "contrastive"] = "contrastive"


@dataclass
class LossConfig:
    """Loss function configuration — implements the three desiderata."""

    # D1: Scale-invariant reconstruction
    recon_weight: float = 1.0
    recon_normalize: bool = True  # L2-normalize before MSE (Desideratum 1)

    # D2: Smooth convergence via sigmoid KL annealing
    kl_weight_max: float = 0.1
    kl_annealing: Literal["sigmoid", "linear", "constant"] = "sigmoid"
    kl_warmup_epochs: int = 10
    kl_sigmoid_steepness: float = 10.0  # steepness of sigmoid schedule

    # D3: Geometric separability
    concept_sparsity_weight: float = 0.05
    concept_diversity_weight: float = 0.05

    # Supervised concept loss (when labels available, e.g., CEBaB)
    concept_supervision_weight: float = 1.0

    # Classification loss
    classification_weight: float = 1.0

    # Stability
    max_loss_value: float = 1e4
    grad_clip_norm: float = 1.0


@dataclass
class ProbeConfig:
    """Geometric probe configuration."""
    # Perturbation probe
    num_directions: int = 100  # m: number of random directions
    perturbation_epsilon: float = 0.01  # ε: perturbation magnitude
    insensitivity_threshold: float = 0.01  # τ: threshold for "insensitive"

    # Which parameters to probe
    probe_scope: Literal["all", "concept_layer", "classifier", "decoder"] = "all"

    # Hessian probe (optional, expensive)
    use_hessian: bool = False
    hessian_top_k: int = 50  # number of top eigenvalues via Lanczos
    hessian_num_batches: int = 5  # batches for Hessian-vector products


@dataclass
class CredalConfig:
    """Credal set configuration."""
    # Mapping from degeneracy ratio ρ to credal set width ε
    epsilon_min: float = 0.01  # minimum imprecision
    epsilon_max: float = 0.3  # maximum imprecision
    mapping_fn: Literal["linear", "sigmoid", "sqrt"] = "linear"

    # Calibration
    calibrate_on_val: bool = True  # tune ε_min, ε_max on validation


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 30
    learning_rate: float = 2e-5
    encoder_lr: float = 2e-5  # separate LR for BERT
    head_lr: float = 1e-3  # separate LR for heads
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: Literal["linear", "cosine", "constant"] = "cosine"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_every: int = 50  # log every N steps
    eval_every_epoch: int = 1
    save_best: bool = True

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Experiment
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class GACSConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    credal: CredalConfig = field(default_factory=CredalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Sync model config with data config."""
        self.model.concept_dim = max(self.data.num_concepts, 4)
        # If no concept supervision, use unsupervised bottleneck
        if self.data.num_concepts == 0:
            self.loss.concept_supervision_weight = 0.0


def get_config(
    dataset: str = "cebab",
    quick: bool = False,
) -> GACSConfig:
    """Factory for common configurations."""
    config = GACSConfig()
    config.data.name = dataset

    if quick:
        config.training.epochs = 5
        config.data.batch_size = 16
        config.model.z_dim = 32
        config.probe.num_directions = 30
        config.loss.kl_warmup_epochs = 2

    config.__post_init__()
    return config
