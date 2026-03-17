"""
Vision-specific GACS configurations for MedMNIST experiments.
"""
from gacs.configs.config import GACSConfig, ModelConfig, TrainingConfig, LossConfig, ProbeConfig


def medmnist_config(dataset_name: str = "dermamnist") -> GACSConfig:
    """
    Config optimized for MedMNIST on RTX 3090/4090.
    
    28×28 images + small CNN = very fast training.
    batch_size=128 uses ~2-3GB VRAM.
    """
    from gacs.data.medmnist import MEDMNIST_INFO
    info = MEDMNIST_INFO[dataset_name]
    
    config = GACSConfig(
        model=ModelConfig(
            encoder_name="cnn",  # flag to use vision model
            z_dim=64,
            num_concepts=8,  # learned unsupervised
            num_classes=info["n_classes"],
            hidden_dim=256,
            dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=128,
            epochs=50,
            learning_rate=1e-3,  # higher LR for CNN (no pretrained weights)
            weight_decay=1e-4,
            max_grad_norm=1.0,
            patience=10,
            num_workers=4,
            log_every=20,
        ),
        loss=LossConfig(
            recon_weight=1.0,
            kl_weight_max=0.05,  # lower for vision (reconstruction matters more)
            kl_warmup_epochs=10,
            concept_weight=0.0,  # no concept supervision for MedMNIST
            classify_weight=1.0,
            concept_sparsity_weight=0.05,
            concept_diversity_weight=0.05,
        ),
        probe=ProbeConfig(
            num_directions=100,
            perturbation_scale=0.01,
            sensitivity_threshold=0.01,
            probe_scope="concept",
        ),
    )
    config.experiment_name = f"gacs_vision_{dataset_name}"
    return config


def organ_shift_config() -> GACSConfig:
    """Config for organ CT cross-view shift experiment."""
    config = medmnist_config("organamnist")
    # OrganMNIST uses 1-channel grayscale
    config.experiment_name = "gacs_vision_organ_shift"
    return config


def medmnist_debug_config(dataset_name: str = "dermamnist") -> GACSConfig:
    """Quick debug config."""
    config = medmnist_config(dataset_name)
    config.training.batch_size = 32
    config.training.epochs = 3
    config.training.num_workers = 0
    config.model.z_dim = 16
    config.model.hidden_dim = 64
    config.model.num_concepts = 4
    config.probe.num_directions = 10
    config.experiment_name = f"gacs_vision_debug_{dataset_name}"
    return config
